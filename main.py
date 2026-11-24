"""通过网页版 Gemini (gemini_web)、gemini-cli (nanobanana) 或 agy CLI 生图并发送到当前会话的 AstrBot 插件。"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from astrbot.api import AstrBotConfig, logger, web
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

try:
    from .gemini_web_backend import GeminiWebImageService
except ImportError:
    from gemini_web_backend import GeminiWebImageService

try:
    from .openai_bridge import OpenAIImageBridgeServer
except ImportError:
    from openai_bridge import OpenAIImageBridgeServer

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
TEMP_ROOT = Path("data/temp/agy_image").resolve()


def _safe_filename(name: str) -> str:
    base = Path(str(name or "output.png")).name.strip()
    if not base or base in (".", ".."):
        base = "output.png"
    if not base.lower().endswith(IMAGE_EXTS):
        base += ".png"
    return base


def _parse_cli_output(stdout_text: str, stderr_text: str) -> tuple[str, str]:
    """解析 gemini -o json 或 agy stream-json，返回 (conversation_id, detail)。"""
    raw = (stdout_text or "").strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if "session_id" in obj or "response" in obj or "error" in obj:
                err_obj = obj.get("error")
                err = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj or "")
                resp = str(obj.get("response") or "").strip()
                detail = resp or err
                if err and resp and err not in resp:
                    detail = f"{err}: {resp}"
                return str(obj.get("session_id") or ""), (detail[:500] if detail else "CLI 未返回可用图片。")
        except Exception:
            pass

    for line in reversed((stdout_text or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("event") == "result":
            res = obj.get("result") or {}
            conv_id = str(res.get("conversation_id") or "").strip()
            err = str(res.get("error") or "").strip()
            resp = str(res.get("response") or "").strip()
            detail = resp or err
            if err and resp and err not in resp:
                detail = f"{err}: {resp}"
            return conv_id, (detail[:500] if detail else "CLI 未返回可用图片。")
    for text in (stderr_text, stdout_text):
        for line in (text or "").splitlines():
            line = line.strip()
            if line and not line.startswith("{"):
                return "", line[:300]
    return "", "CLI 未返回可用图片。"


def _find_generated_image(work_dir: Path, preferred_name: str, conversation_id: str = "") -> Path | None:
    preferred = work_dir / preferred_name
    if preferred.is_file() and preferred.stat().st_size > 0:
        return preferred
    search_dirs = [work_dir]
    if conversation_id:
        app_data = Path(os.environ.get("AGY_APP_DATA_DIR") or (Path.home() / ".gemini" / "antigravity-cli"))
        brain_dir = app_data / "brain" / conversation_id
        if brain_dir.is_dir():
            search_dirs.append(brain_dir)
    candidates = []
    for root in search_dirs:
        candidates.extend(
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 0
        )
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@register(
    "astrbot_plugin_agy_image",
    "luke_lu",
    "通过网页版 Gemini (Firefox Cookie)、gemini-cli 或 agy CLI 生成图片并发送到当前会话",
    "0.4.0",
)
class AgyImagePlugin(Star):
    """封装网页版 Gemini、gemini-cli 与 agy 生图能力的轻量插件。"""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self._gemini_web = GeminiWebImageService()
        self._bridge: OpenAIImageBridgeServer | None = None
        if context is not None and callable(getattr(context, "register_web_api", None)):
            context.register_web_api(
                "/astrbot_plugin_agy_image/status",
                self._web_status_api,
                ["GET"],
                "读取 agy_image 当前后端与本地桥接状态",
            )
            context.register_web_api(
                "/astrbot_plugin_agy_image/connectivity-test",
                self._web_connectivity_test_api,
                ["POST"],
                "执行 agy_image 连通性测试或完整生图测试",
            )

    async def initialize(self):
        if self.config.get("openai_bridge_enabled", True):
            host = str(self.config.get("openai_bridge_host", "127.0.0.1") or "127.0.0.1").strip()
            raw_port = self.config.get("openai_bridge_port")
            try:
                port = 18765 if raw_port is None or raw_port == "" else int(raw_port)
            except (TypeError, ValueError):
                port = 18765
            api_key = str(self.config.get("openai_bridge_api_key", "") or "")
            backend_name = str(self.config.get("backend", "gemini_web") or "gemini_web").strip().lower()
            bridge = OpenAIImageBridgeServer(
                host=host,
                port=port,
                api_key=api_key,
                generator=self._generate_for_bridge,
                backend_name=backend_name,
            )
            try:
                bound_port = await bridge.start()
                self._bridge = bridge
                logger.info(
                    f"[agy_image] 本地 OpenAI 兼容生图桥接已启动: http://{bridge.host}:{bound_port}/v1"
                )
            except OSError as exc:
                logger.warning(f"[agy_image] 启动本地生图桥接失败 ({host}:{port}): {exc}")
            except Exception as exc:
                logger.warning(f"[agy_image] 启动本地生图桥接异常: {exc}")

        if not self.config.get("expose_llm_tool", True) and self.context is not None:
            try:
                deact_async = getattr(self.context, "deactivate_llm_tool_async", None)
                deact_sync = getattr(self.context, "deactivate_llm_tool", None)
                if callable(deact_async):
                    await deact_async("agy_generate_image")
                    logger.info("[agy_image] 已按配置隐藏独立工具 agy_generate_image（保留本地桥接与 /agy生图 命令）")
                elif callable(deact_sync):
                    deact_sync("agy_generate_image")
                    logger.info("[agy_image] 已按配置隐藏独立工具 agy_generate_image（保留本地桥接与 /agy生图 命令）")
            except Exception as exc:
                logger.warning(f"[agy_image] 隐藏 agy_generate_image 工具时忽略异常: {exc}")

    async def terminate(self):
        if self._bridge is not None:
            await self._bridge.stop()
            self._bridge = None
        await self._gemini_web.close()

    async def _generate_for_bridge(
        self,
        prompt: str,
        *,
        reference_files: list[Path] | None = None,
        model: str = "",
        size: str = "",
    ) -> tuple[bytes | None, str, str]:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return None, "生图提示词不能为空。", "invalid_request"

        backend = str(self.config.get("backend", "gemini_web") or "gemini_web").strip().lower()
        timeout = max(15, min(600, int(self.config.get("timeout_seconds", 120) or 120)))
        if backend == "gemini_web":
            job_dir = TEMP_ROOT / ("bridge_" + uuid.uuid4().hex[:12])
            job_dir.mkdir(parents=True, exist_ok=True)
            try:
                res = await self._gemini_web.generate_image_result(
                    prompt_text,
                    job_dir,
                    "output.png",
                    self.config,
                    timeout=timeout,
                    reference_files=reference_files,
                    model_override=model or None,
                    size=size,
                )
                if res.path is not None and res.path.is_file():
                    return res.path.read_bytes(), "", ""
                return None, res.error or "gemini_web 生图失败。", res.error_code or "backend_error"
            finally:
                shutil.rmtree(job_dir, ignore_errors=True)

        if reference_files:
            return (
                None,
                f"当前后端 ({backend}) 不支持参考图编辑，请将 agy_image 的 backend 设为 gemini_web。",
                "invalid_request",
            )
        img_path, err = await self._run_cli_image(prompt_text, filename="output.png")
        if img_path is not None and img_path.is_file():
            try:
                return img_path.read_bytes(), "", ""
            finally:
                shutil.rmtree(img_path.parent, ignore_errors=True)
        code = "timeout" if "超时" in str(err) else "backend_error"
        return None, err or f"{backend} 生图失败。", code

    def _status_snapshot(self) -> dict:
        backend = str(self.config.get("backend", "gemini_web") or "gemini_web").strip().lower()
        bridge_running = bool(self._bridge is not None and self._bridge.is_running)
        bridge_url = (
            f"http://{self._bridge.host}:{self._bridge.port}/v1"
            if bridge_running and self._bridge is not None
            else ""
        )
        return {
            "backend": backend,
            "bridge_enabled": bool(self.config.get("openai_bridge_enabled", True)),
            "bridge_running": bridge_running,
            "bridge_url": bridge_url,
            "expose_llm_tool": bool(self.config.get("expose_llm_tool", True)),
            "proxy_configured": bool(str(self.config.get("proxy_url", "") or "").strip()),
        }

    async def test_connectivity(self, *, generate_image: bool = False) -> dict:
        import aiohttp

        started = time.monotonic()
        snapshot = self._status_snapshot()
        backend = snapshot["backend"]
        result: dict = {
            "ok": False,
            "mode": "generate" if generate_image else "ping",
            "status": snapshot,
            "bridge_health": None,
            "backend_check": None,
            "image_base64": "",
            "mime_type": "",
            "size_bytes": 0,
            "elapsed_ms": 0,
            "error": "",
        }

        if snapshot["bridge_running"] and self._bridge is not None:
            health_url = f"http://{self._bridge.host}:{self._bridge.port}/health"
            try:
                async with aiohttp.ClientSession(
                    trust_env=False, timeout=aiohttp.ClientTimeout(total=10)
                ) as client:
                    async with client.get(health_url) as resp:
                        body = await resp.json()
                        result["bridge_health"] = {"ok": resp.status == 200, "status": resp.status, "body": body}
            except Exception as exc:
                result["bridge_health"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        if backend == "gemini_web":
            backend_res = await self._gemini_web.check_connectivity(self.config, timeout=30)
            result["backend_check"] = backend_res
            if not backend_res.get("ok"):
                result["error"] = backend_res.get("error") or "gemini_web 连通性校验失败"
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                return result
        else:
            key = "agy_command" if backend == "agy" else "gemini_command"
            default_bin = "agy" if backend == "agy" else "gemini"
            resolved = self._resolve_bin(key, default_bin)
            exists = Path(resolved).is_file() or bool(shutil.which(resolved))
            result["backend_check"] = {"ok": exists, "command": resolved}
            if not exists:
                result["error"] = f"未找到可执行文件：{resolved}"
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                return result

        if not generate_image:
            bridge_ok = (not snapshot["bridge_enabled"]) or bool(
                result["bridge_health"] and result["bridge_health"].get("ok")
            )
            result["ok"] = bool(result["backend_check"].get("ok")) and bridge_ok
            if not bridge_ok and not result["error"]:
                result["error"] = "本地 OpenAI 桥接服务未就绪"
            result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            return result

        prompt = "Gemini猫娘说OK"
        if snapshot["bridge_running"] and self._bridge is not None:
            gen_url = f"http://{self._bridge.host}:{self._bridge.port}/v1/images/generations"
            headers = {}
            if self._bridge.api_key:
                headers["Authorization"] = f"Bearer {self._bridge.api_key}"
            timeout = max(30, min(600, int(self.config.get("timeout_seconds", 120) or 120)))
            try:
                async with aiohttp.ClientSession(
                    trust_env=False, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as client:
                    async with client.post(
                        gen_url,
                        headers=headers,
                        json={"model": "gemini-web", "prompt": prompt, "n": 1, "response_format": "b64_json"},
                    ) as resp:
                        data = await resp.json()
                        if resp.status != 200:
                            msg = (data.get("error") or {}).get("message") if isinstance(data, dict) else str(data)
                            result["error"] = f"HTTP {resp.status}: {msg}"
                            result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                            return result
                        b64 = ((data.get("data") or [{}])[0]).get("b64_json", "")
                        raw = base64.b64decode(b64, validate=True)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                return result
        else:
            raw, err, _ = await self._generate_for_bridge(prompt)
            if not raw:
                result["error"] = err or "生图测试失败"
                result["elapsed_ms"] = round((time.monotonic() - started) * 1000)
                return result

        mime = "image/jpeg" if raw.startswith(b"\xff\xd8\xff") else (
            "image/webp" if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" else "image/png"
        )
        result.update(
            ok=True,
            image_base64=base64.b64encode(raw).decode("ascii"),
            mime_type=mime,
            size_bytes=len(raw),
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        return result

    async def _web_status_api(self):
        return web.json_response({"status": "ok", "data": self._status_snapshot()})

    async def _web_connectivity_test_api(self):
        try:
            payload = await web.request.json()
        except Exception:
            payload = {}
        generate_image = bool(isinstance(payload, dict) and payload.get("generate_image"))
        res = await self.test_connectivity(generate_image=generate_image)
        return web.json_response({"status": "ok" if res["ok"] else "error", "data": res})

    def _resolve_bin(self, key: str, default_name: str) -> str:
        configured = str(self.config.get(key, "") or "").strip()
        if configured:
            return configured
        found = shutil.which(default_name)
        if found:
            return found
        local_candidate = Path.home() / ".local" / "bin" / default_name
        if local_candidate.exists():
            return str(local_candidate)
        return default_name

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = local_bin + os.pathsep + env.get("PATH", "")
        proxy = str(self.config.get("proxy_url", "") or "").strip()
        if proxy:
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
                env[key] = proxy
        nano_key = str(self.config.get("nanobanana_api_key", "") or "").strip()
        if nano_key:
            env["NANOBANANA_API_KEY"] = nano_key
            env.setdefault("GEMINI_API_KEY", nano_key)
        nano_model = str(self.config.get("nanobanana_model", "") or "").strip()
        if nano_model:
            env["NANOBANANA_MODEL"] = nano_model
        return env

    async def _run_cli_image(self, prompt: str, filename: str = "output.png") -> tuple[Path | None, str]:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return None, "生图提示词不能为空。"

        safe_name = _safe_filename(filename)
        job_dir = TEMP_ROOT / uuid.uuid4().hex[:12]
        job_dir.mkdir(parents=True, exist_ok=True)
        target_path = job_dir / safe_name

        backend = str(self.config.get("backend", "gemini_web") or "gemini_web").strip().lower()
        timeout = max(15, min(600, int(self.config.get("timeout_seconds", 120) or 120)))

        if backend == "gemini_web":
            return await self._gemini_web.generate_image(
                prompt_text, job_dir, safe_name, self.config, timeout=timeout
            )

        model = str(self.config.get("model", "") or "").strip()
        if backend == "agy":
            cli_bin = self._resolve_bin("agy_command", "agy")
            cli_prompt = (
                f"Use the image-generator subagent to generate an image based on this description, "
                f"then copy the generated image file to `{target_path}`:\n{prompt_text}"
            )
            cmd = [
                cli_bin,
                "-p",
                cli_prompt,
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
            ]
            if model:
                cmd.extend(["--model", model])
        else:
            cli_bin = self._resolve_bin("gemini_command", "gemini")
            cli_prompt = (
                f"Use the nanobanana generate_image tool to generate an image for this prompt: "
                f"{prompt_text}"
            )
            cmd = [
                cli_bin,
                "-p",
                cli_prompt,
                "--yolo",
                "-e",
                "nanobanana",
                "-o",
                "json",
            ]
            if model:
                cmd.extend(["-m", model])

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(job_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            return None, f"{backend} 生图超时（>{timeout}s）。"
        except FileNotFoundError:
            return None, f"未找到可执行文件：{cli_bin}"
        except Exception as exc:
            return None, f"调用 {backend} 失败：{exc}"

        stdout_text = (stdout_b or b"").decode("utf-8", errors="replace")
        stderr_text = (stderr_b or b"").decode("utf-8", errors="replace")
        conv_id, detail = _parse_cli_output(stdout_text, stderr_text)
        img_path = _find_generated_image(job_dir, safe_name, conversation_id=conv_id)
        if img_path is None:
            logger.warning(f"[agy_image] 生图未产出文件({backend}): {detail}")
            return None, f"{backend} 生图失败：{detail}"
        if img_path.parent != job_dir:
            copied = job_dir / f"{img_path.stem}{img_path.suffix}"
            shutil.copy2(img_path, copied)
            img_path = copied
        return img_path, ""

    @filter.llm_tool(name="agy_generate_image")
    async def agy_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        filename: str = "output.png",
    ) -> str:
        """Generate an image using Gemini Web (default), gemini-cli, or agy CLI and send it to the user.

        Args:
            prompt(string): Detailed visual description of the image to generate.
            filename(string): Optional output image filename (default output.png).
        """
        img_path, err = await self._run_cli_image(prompt, filename=filename)
        if img_path is None:
            return err

        if self.config.get("auto_send_to_user", True):
            try:
                await event.send(MessageChain([Comp.Image.fromFileSystem(str(img_path))]))
                return f"已生成并发送图片到当前会话：{img_path}"
            except Exception as exc:
                logger.warning(f"[agy_image] 发送图片失败，返回本地路径: {exc}")
        return f"已生成图片，本地路径：{img_path}"

    @filter.command("agy生图")
    async def cmd_agy_image(self, event: AstrMessageEvent):
        """手动测试生图：/agy生图 <描述>"""
        raw = (event.message_str or "").strip()
        parts = raw.split(maxsplit=1)
        prompt = parts[1].strip() if len(parts) >= 2 else ""
        if not prompt:
            yield event.plain_result("用法：/agy生图 <图片描述>")
            return
        yield event.plain_result("正在生图，请稍候...")
        img_path, err = await self._run_cli_image(prompt)
        if img_path is None:
            yield event.plain_result(f"❌ {err}")
            return
        yield event.image_result(str(img_path))
