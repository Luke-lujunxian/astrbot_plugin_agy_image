"""Gemini Web (gemini-webapi) image generation backend with Firefox cookie auto-discovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import glob
import math
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from astrbot.api import logger

_COOKIE_HOSTS = (".google.com", "gemini.google.com", ".gemini.google.com")
_FIREFOX_SEARCH_ROOTS = (
    Path.home() / ".mozilla" / "firefox",
    Path.home() / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
    Path.home() / ".var" / "app" / "org.mozilla.firefox" / ".mozilla" / "firefox",
)
_IMAGE_INTENT_MARKERS = (
    "generate an image",
    "create an image",
    "draw ",
    "生成图片",
    "画一张",
    "画一幅",
    "生图",
)
_EDIT_INTENT_MARKERS = (
    "edit and generate",
    "generate a new image",
    "generate an image",
    "create an image",
    "生成图片",
    "画一张",
    "画一幅",
    "改图",
    "编辑图片",
)
_PLACEHOLDER_MODEL_NAMES = frozenset({"", "gemini-web", "gemini_web", "default", "auto"})
_CONTENT_POLICY_MARKERS = (
    "content policy",
    "content_policy",
    "safety",
    "policy",
    "cannot generate",
    "can't generate",
    "unable to generate",
    "i cannot create",
    "i can't create",
    "inappropriate",
    "harmful",
    "violat",
    "不能生成",
    "无法生成",
    "安全政策",
    "违规",
)
_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests", "quota", "usage limit")
_ASPECT_HINT_RE = re.compile(
    r"(?:\baspect\s*ratio\b|\bratio\b|\bportrait\b|\blandscape\b|\bvertical\b|\bhorizontal\b|"
    r"\b\d+:\d+\b|横屏|竖屏|比例)",
    re.IGNORECASE,
)
_SIZE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)
_SHIM_INSTALLED = False


@dataclass(frozen=True, slots=True)
class GeminiWebResult:
    """Structured outcome from a Gemini Web image generation or edit call."""

    path: Path | None
    error: str = ""
    error_code: str = ""
    detail: str = ""


def _resolve_http_proxy(configured_proxy: str = "") -> str | None:
    """Resolve explicit or environment HTTP(S) proxy without touching ALL_PROXY socks://."""
    explicit = str(configured_proxy or "").strip().rstrip("/")
    if explicit:
        return explicit
    for key in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        val = str(os.environ.get(key, "") or "").strip().rstrip("/")
        if val:
            return val
    return None


def _install_curl_cffi_localhost_shim() -> None:
    """Avoid curl_cffi CurlFollow.SAFE SSRF false positives on localhost proxies."""
    global _SHIM_INSTALLED
    if _SHIM_INSTALLED:
        return
    from curl_cffi import CurlFollow
    from curl_cffi.requests import AsyncSession

    orig_init = AsyncSession.__init__

    def _patched_init(self, *args, **kwargs):
        proxy_val = kwargs.get("proxy") or kwargs.get("proxies") or ""
        if (
            kwargs.get("allow_redirects") == CurlFollow.SAFE
            and any(host in str(proxy_val) for host in ("127.0.0.1", "localhost", "::1"))
        ):
            kwargs["allow_redirects"] = True
        kwargs.setdefault("timeout", 30)
        return orig_init(self, *args, **kwargs)

    AsyncSession.__init__ = _patched_init
    _SHIM_INSTALLED = True


def _firefox_db_score(db_path: Path) -> tuple[int, float]:
    """Rank Firefox cookie DBs by latest Google/Gemini auth cookie access time."""
    last_accessed = 0
    try:
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "cookies.sqlite"
            shutil.copy2(db_path, dst)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(db_path) + suffix)
                if sidecar.exists():
                    shutil.copy2(sidecar, Path(str(dst) + suffix))
            conn = sqlite3.connect(dst)
            try:
                row = conn.execute(
                    "SELECT MAX(lastAccessed) FROM moz_cookies "
                    "WHERE host IN ('.google.com', 'gemini.google.com') "
                    "AND name IN ('__Secure-1PSID', 'OSID')"
                ).fetchone()
                if row and row[0]:
                    last_accessed = int(row[0])
            finally:
                conn.close()
    except Exception:
        last_accessed = 0
    try:
        mtime = db_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (last_accessed, mtime)


def _discover_firefox_cookie_db(configured_path: str = "") -> Path | None:
    """Locate Firefox's cookies.sqlite from explicit config or standard profile dirs."""
    raw = str(configured_path or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            cookie_file = candidate / "cookies.sqlite"
            if cookie_file.is_file():
                return cookie_file
        return None

    found: list[Path] = []
    for root in _FIREFOX_SEARCH_ROOTS:
        if root.is_dir():
            for match in glob.glob(str(root / "*" / "cookies.sqlite")):
                p = Path(match)
                if p.is_file():
                    found.append(p)
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    found.sort(key=_firefox_db_score, reverse=True)
    return found[0]


def _load_firefox_cookies(db_path: Path) -> list[tuple[str, str, str, bool]]:
    """Safely copy Firefox cookies.sqlite (including WAL/SHM) and extract Google/Gemini cookies."""
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "cookies.sqlite"
        shutil.copy2(db_path, dst)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, Path(str(dst) + suffix))
        conn = sqlite3.connect(dst)
        try:
            rows = conn.execute(
                "SELECT host, name, value, isSecure FROM moz_cookies "
                "WHERE host IN ('.google.com', 'gemini.google.com', '.gemini.google.com')"
            ).fetchall()
        finally:
            conn.close()
    return [(str(h), str(n), str(v), bool(sec)) for h, n, v, sec in rows]


def _detect_image_ext(raw: bytes) -> str:
    """Detect image file extension from magic bytes, defaulting to .png."""
    if raw.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _normalize_image_extension(img_path: Path) -> Path:
    """Rename saved image file suffix to match its real magic bytes."""
    try:
        with img_path.open("rb") as fp:
            header = fp.read(12)
    except OSError:
        return img_path

    if not (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or (len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP")
    ):
        return img_path

    ext = _detect_image_ext(header)
    if img_path.suffix.lower() != ext:
        target = img_path.with_suffix(ext)
        if target.exists():
            target.unlink()
        img_path.rename(target)
        return target
    return img_path


def _normalize_model_name(model: str | None) -> str | None:
    """Map placeholder OpenAI model aliases to None so gemini_webapi uses account default."""
    cleaned = str(model or "").strip()
    if cleaned.lower() in _PLACEHOLDER_MODEL_NAMES:
        return None
    return cleaned


def _aspect_ratio_hint(size: str, prompt: str) -> str:
    """Build a concise aspect ratio hint from WxH when prompt omits ratio/orientation."""
    if not size or _ASPECT_HINT_RE.search(prompt):
        return ""
    match = _SIZE_RE.match(size)
    if not match:
        return ""
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0 or width == height:
        return ""
    divisor = math.gcd(width, height)
    return f"aspect ratio {width // divisor}:{height // divisor}"


def _ensure_image_prompt(prompt: str, *, is_edit: bool = False, size: str = "") -> str:
    """Prepend an explicit image generation/edit directive when the prompt lacks one."""
    text = str(prompt or "").strip()
    lower = text.lower()
    if is_edit:
        if any(marker in lower for marker in _EDIT_INTENT_MARKERS):
            result = text
        else:
            result = (
                "Edit and generate a new image following these reference images and instructions:\n"
                + text
            )
    else:
        if any(marker in lower for marker in _IMAGE_INTENT_MARKERS):
            result = text
        else:
            result = f"Generate an image: {text}"
    ratio_hint = _aspect_ratio_hint(size, result)
    if ratio_hint:
        result = f"{result} ({ratio_hint})"
    return result


class GeminiWebImageService:
    """Manages a persistent GeminiClient session for web-based image generation."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._client: Any = None
        self._client_key: tuple[Any, ...] | None = None

    @staticmethod
    def _build_cookies(config: dict) -> tuple[str, str, list[tuple[str, str, str, bool]], str]:
        manual_psid = str(config.get("gemini_web_1psid", "") or "").strip()
        if manual_psid:
            manual_psidts = str(config.get("gemini_web_1psidts", "") or "").strip()
            manual_osid = str(config.get("gemini_web_osid", "") or "").strip()
            manual_sec_osid = str(config.get("gemini_web_secure_osid", "") or "").strip()
            cookies: list[tuple[str, str, str, bool]] = [
                (".google.com", "__Secure-1PSID", manual_psid, True)
            ]
            if manual_psidts:
                cookies.append((".google.com", "__Secure-1PSIDTS", manual_psidts, True))
            if manual_osid:
                cookies.append(("gemini.google.com", "OSID", manual_osid, True))
            if manual_sec_osid:
                cookies.append(("gemini.google.com", "__Secure-OSID", manual_sec_osid, True))
            return manual_psid, manual_psidts, cookies, "manual"

        db_path = _discover_firefox_cookie_db(str(config.get("firefox_profile_path", "") or ""))
        if db_path is None:
            raise RuntimeError(
                "未找到 Firefox 的 cookies.sqlite，请先在 Firefox 登录 gemini.google.com 或在插件配置中填写 Cookie。"
            )
        cookies = _load_firefox_cookies(db_path)
        by_name = {name: val for _, name, val, _ in cookies}
        psid = by_name.get("__Secure-1PSID", "")
        psidts = by_name.get("__Secure-1PSIDTS", "")
        if not psid:
            raise RuntimeError(
                "Firefox Cookie 中缺少 __Secure-1PSID，请先在 Firefox 打开并登录 gemini.google.com。"
            )
        return psid, psidts, cookies, f"firefox:{db_path.parent.name}"

    @staticmethod
    def _clear_stale_cookie_cache() -> None:
        try:
            from gemini_webapi.utils.get_access_token import _get_cookie_cache_dir

            cache_dir = _get_cookie_cache_dir()
            for stale in cache_dir.glob(".cached_cookies_*.json"):
                stale.unlink(missing_ok=True)
        except Exception:
            pass

    async def _ensure_client(self, config: dict, timeout: int, *, force_reinit: bool = False):
        proxy = _resolve_http_proxy(str(config.get("proxy_url", "") or ""))
        manual_psid = str(config.get("gemini_web_1psid", "") or "").strip()
        firefox_path = str(config.get("firefox_profile_path", "") or "").strip()
        auto_refresh = bool(config.get("gemini_web_auto_refresh", True))
        key = (proxy, len(manual_psid), firefox_path, auto_refresh)

        if (
            self._client is not None
            and not force_reinit
            and self._client_key == key
            and getattr(self._client, "_running", False)
        ):
            return self._client

        await self.close()
        _install_curl_cffi_localhost_shim()
        if force_reinit:
            self._clear_stale_cookie_cache()

        psid, psidts, all_cookies, source_label = self._build_cookies(config)

        from gemini_webapi import GeminiClient

        client = GeminiClient(psid, psidts or None, proxy=proxy)
        for host, name, val, is_secure in all_cookies:
            if name not in ("__Secure-1PSID", "__Secure-1PSIDTS"):
                client.cookies.set(name, val, domain=host, secure=bool(is_secure))

        logger.info(
            f"[agy_image] 初始化 gemini_web 客户端 (source={source_label}, cookies={len(all_cookies)})"
        )
        init_timeout = min(max(int(timeout), 45), 90)
        try:
            await asyncio.wait_for(
                client.init(
                    timeout=timeout,
                    auto_close=False,
                    auto_refresh=auto_refresh,
                    watchdog_timeout=min(timeout, 90),
                    verbose=False,
                ),
                timeout=init_timeout,
            )
        except Exception:
            try:
                await asyncio.wait_for(client.close(), timeout=5)
            except Exception:
                pass
            raise

        self._client = client
        self._client_key = key
        return client

    @staticmethod
    def _is_retryable_session_error(exc: Exception) -> bool:
        try:
            from gemini_webapi.exceptions import AuthError

            if isinstance(exc, AuthError):
                return True
        except Exception:
            pass
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        return (
            "auth" in name
            or "closed" in msg
            or "connection" in name
            or "unauthenticated" in msg
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> str:
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        try:
            from gemini_webapi.exceptions import AuthError

            if isinstance(exc, AuthError):
                return "auth_error"
        except Exception:
            pass
        if isinstance(exc, asyncio.TimeoutError) or "timeout" in name:
            return "timeout"
        if (
            "auth" in name
            or "unauthenticated" in msg
            or "cookies.sqlite" in msg
            or "__secure-1psid" in msg
            or "登录 gemini.google.com" in msg
        ):
            return "auth_error"
        if any(marker in msg for marker in _RATE_LIMIT_MARKERS):
            return "rate_limited"
        if any(marker in msg for marker in _CONTENT_POLICY_MARKERS):
            return "content_policy"
        return "backend_error"

    @staticmethod
    def _classify_text_refusal(text: str) -> str:
        lower = str(text or "").lower()
        if any(marker in lower for marker in _RATE_LIMIT_MARKERS):
            return "rate_limited"
        if any(marker in lower for marker in _CONTENT_POLICY_MARKERS):
            return "content_policy"
        return "no_image"

    async def generate_image_result(
        self,
        prompt: str,
        job_dir: Path,
        safe_name: str,
        config: dict,
        *,
        timeout: int = 120,
        reference_files: list[Path] | None = None,
        model_override: str | None = None,
        size: str = "",
    ) -> GeminiWebResult:
        """Generate or edit an image via Gemini Web and return a structured result."""
        files_arg = [str(Path(p)) for p in reference_files] if reference_files else None
        req_prompt = _ensure_image_prompt(prompt, is_edit=bool(files_arg), size=size)
        raw_model = model_override if model_override is not None else config.get("model", "")
        model = _normalize_model_name(raw_model)
        temporary = bool(config.get("gemini_web_temporary", True))

        gen_kwargs: dict[str, Any] = {"model": model, "temporary": temporary}
        if files_arg:
            gen_kwargs["files"] = files_arg

        async with self._lock:
            resp = None
            client = None
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    client = await self._ensure_client(
                        config, timeout=timeout, force_reinit=(attempt == 1)
                    )
                    resp = await asyncio.wait_for(
                        client.generate_content(req_prompt, **gen_kwargs),
                        timeout=timeout,
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt == 0 and self._is_retryable_session_error(exc):
                        logger.warning(
                            f"[agy_image] gemini_web 会话异常，尝试重新读取 Cookie 重试: {type(exc).__name__}: {exc}"
                        )
                        continue
                    break

            if last_exc is not None:
                code = self._classify_exception(last_exc)
                msg = f"gemini_web 生图失败：{type(last_exc).__name__}: {last_exc}"
                return GeminiWebResult(path=None, error=msg, error_code=code, detail=str(last_exc)[:300])

            imgs = list(getattr(resp, "images", None) or [])
            gen_imgs = [img for img in imgs if type(img).__name__ == "GeneratedImage"]
            target_imgs = gen_imgs or imgs
            if not target_imgs:
                text = str(getattr(resp, "text", "") or "").strip()
                detail = text[:300] if text else "响应中未包含图片"
                code = self._classify_text_refusal(text)
                return GeminiWebResult(
                    path=None,
                    error=f"gemini_web 未返回图片：{detail}",
                    error_code=code,
                    detail=detail,
                )

            img = target_imgs[0]
            try:
                saved = await asyncio.wait_for(
                    img.save(
                        path=str(job_dir),
                        filename=safe_name,
                        verbose=False,
                        client=getattr(client, "client", None),
                    ),
                    timeout=45,
                )
            except Exception as exc:
                return GeminiWebResult(
                    path=None,
                    error=f"gemini_web 图片下载失败：{type(exc).__name__}: {exc}",
                    error_code="download_error",
                    detail=str(exc)[:300],
                )

            saved_path = Path(saved) if saved else (job_dir / safe_name)
            if not saved_path.is_file() or saved_path.stat().st_size <= 0:
                return GeminiWebResult(
                    path=None,
                    error="gemini_web 保存图片文件为空。",
                    error_code="download_error",
                )
            final_path = _normalize_image_extension(saved_path)
            return GeminiWebResult(path=final_path)

    async def generate_image(
        self,
        prompt: str,
        job_dir: Path,
        safe_name: str,
        config: dict,
        timeout: int = 120,
        reference_files: list[Path] | None = None,
        model_override: str | None = None,
        size: str = "",
    ) -> tuple[Path | None, str]:
        """Generate an image via Gemini Web and save it into job_dir."""
        result = await self.generate_image_result(
            prompt,
            job_dir,
            safe_name,
            config,
            timeout=timeout,
            reference_files=reference_files,
            model_override=model_override,
            size=size,
        )
        return result.path, result.error

    async def check_connectivity(self, config: dict, timeout: int = 30) -> dict[str, Any]:
        """Verify cookie discovery and Gemini Web session initialization without generating an image."""
        import time

        started = time.monotonic()
        proxy = _resolve_http_proxy(str(config.get("proxy_url", "") or ""))
        try:
            _, _, all_cookies, source_label = self._build_cookies(config)
        except Exception as exc:
            return {
                "ok": False,
                "source": "",
                "cookie_count": 0,
                "proxy": proxy or "direct",
                "elapsed_ms": round((time.monotonic() - started) * 1000),
                "error_code": "auth_error",
                "error": str(exc),
            }

        async with self._lock:
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    await self._ensure_client(
                        config, timeout=timeout, force_reinit=(attempt == 1)
                    )
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt == 0 and self._is_retryable_session_error(exc):
                        continue
                    break

        elapsed_ms = round((time.monotonic() - started) * 1000)
        if last_exc is not None:
            return {
                "ok": False,
                "source": source_label,
                "cookie_count": len(all_cookies),
                "proxy": proxy or "direct",
                "elapsed_ms": elapsed_ms,
                "error_code": self._classify_exception(last_exc),
                "error": f"{type(last_exc).__name__}: {last_exc}",
            }
        return {
            "ok": True,
            "source": source_label,
            "cookie_count": len(all_cookies),
            "proxy": proxy or "direct",
            "elapsed_ms": elapsed_ms,
            "error_code": "",
            "error": "",
        }

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._client_key = None
        if client is not None:
            try:
                await asyncio.wait_for(client.close(), timeout=10)
            except Exception as exc:
                logger.debug(f"[agy_image] 关闭 gemini_web 客户端忽略异常: {exc}")
