"""Localhost OpenAI-compatible Images API bridge backed by Gemini Web."""

from __future__ import annotations

import base64
import hmac
from pathlib import Path
import tempfile
import time
from typing import Any, Awaitable, Callable

from aiohttp import web
from astrbot.api import logger

try:
    from .gemini_web_backend import _detect_image_ext
except ImportError:
    from gemini_web_backend import _detect_image_ext

MAX_EDIT_FILES = 16
MAX_EDIT_IMAGE_BYTES = 50 * 1024 * 1024
MAX_FIELD_BYTES = 256 * 1024
CLIENT_MAX_SIZE = 64 * 1024 * 1024
IMAGE_FIELD_NAMES = frozenset({"image", "image[]", "images", "images[]"})

GeneratorCallback = Callable[..., Awaitable[tuple[bytes | None, str, str]]]

_ERROR_MAP: dict[str, tuple[int, str, str]] = {
    "invalid_request": (400, "invalid_request_error", "invalid_request"),
    "auth_error": (401, "authentication_error", "invalid_api_key"),
    "content_policy": (400, "invalid_request_error", "content_policy_violation"),
    "no_image": (422, "invalid_request_error", "no_image_generated"),
    "rate_limited": (429, "rate_limit_error", "rate_limit_exceeded"),
    "payload_too_large": (413, "invalid_request_error", "payload_too_large"),
    "timeout": (504, "server_error", "timeout"),
    "download_error": (502, "server_error", "backend_error"),
    "backend_error": (502, "server_error", "backend_error"),
}


def _json_error(error_code: str, message: str) -> web.Response:
    status, err_type, api_code = _ERROR_MAP.get(
        error_code, (502, "server_error", "backend_error")
    )
    msg = str(message or api_code).strip()
    if error_code == "content_policy" and "content_policy" not in msg.lower():
        msg = f"content_policy: {msg}"
    return web.json_response(
        {
            "error": {
                "message": msg,
                "type": err_type,
                "param": None,
                "code": api_code,
            }
        },
        status=status,
    )


class OpenAIImageBridgeServer:
    """Expose `/v1/images/generations` and `/v1/images/edits` on loopback."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 18765,
        api_key: str = "",
        generator: GeneratorCallback,
        backend_name: str = "gemini_web",
    ) -> None:
        self.host = str(host or "127.0.0.1").strip() or "127.0.0.1"
        self.port = int(port)
        self.api_key = str(api_key or "").strip()
        self.generator = generator
        self.backend_name = str(backend_name or "gemini_web")
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    @property
    def is_running(self) -> bool:
        return self._runner is not None and self._site is not None

    def _check_auth(self, request: web.Request) -> web.Response | None:
        if not self.api_key:
            return None
        header = str(request.headers.get("Authorization", "") or "").strip()
        prefix = "Bearer "
        if not header.startswith(prefix):
            return _json_error("auth_error", "Unauthorized")
        token = header[len(prefix) :].strip()
        if not hmac.compare_digest(token.encode("utf-8"), self.api_key.encode("utf-8")):
            return _json_error("auth_error", "Unauthorized")
        return None

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "backend": self.backend_name})

    async def _handle_models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": "gemini-web",
                        "object": "model",
                        "owned_by": "astrbot_plugin_agy_image",
                    }
                ],
            }
        )

    async def _handle_generations(self, request: web.Request) -> web.Response:
        auth_err = self._check_auth(request)
        if auth_err is not None:
            return auth_err
        try:
            payload: Any = await request.json()
        except Exception:
            return _json_error("invalid_request", "Request body must be valid JSON")
        if not isinstance(payload, dict):
            return _json_error("invalid_request", "JSON body must be an object")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _json_error("invalid_request", "Field 'prompt' must be a non-empty string")
        model = str(payload.get("model") or "").strip()
        size = str(payload.get("size") or "").strip()

        img_bytes, err_msg, err_code = await self.generator(
            prompt.strip(),
            reference_files=None,
            model=model,
            size=size,
        )
        if not img_bytes:
            return _json_error(err_code or "backend_error", err_msg or "Generation failed")
        return web.json_response(
            {
                "created": int(time.time()),
                "data": [{"b64_json": base64.b64encode(img_bytes).decode("ascii")}],
            }
        )

    async def _handle_edits(self, request: web.Request) -> web.Response:
        auth_err = self._check_auth(request)
        if auth_err is not None:
            return auth_err
        content_type = str(request.content_type or "").lower()
        if not content_type.startswith("multipart/"):
            return _json_error("invalid_request", "Expected multipart/form-data body")

        fields: dict[str, str] = {}
        saved_files: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="agy_bridge_") as tmpdir:
            tmp_path = Path(tmpdir)
            try:
                reader = await request.multipart()
            except Exception:
                return _json_error("invalid_request", "Malformed multipart/form-data request")

            while True:
                part = await reader.next()
                if part is None:
                    break
                field_name = str(part.name or "")
                if field_name in IMAGE_FIELD_NAMES:
                    if len(saved_files) >= MAX_EDIT_FILES:
                        return _json_error(
                            "invalid_request",
                            f"At most {MAX_EDIT_FILES} reference images are allowed",
                        )
                    raw = bytearray()
                    while True:
                        chunk = await part.read_chunk(65536)
                        if not chunk:
                            break
                        raw.extend(chunk)
                        if len(raw) > MAX_EDIT_IMAGE_BYTES:
                            return _json_error(
                                "payload_too_large",
                                "Reference image exceeds 50MB size limit",
                            )
                    if not raw:
                        return _json_error("invalid_request", "Uploaded image part is empty")
                    img_data = bytes(raw)
                    ext = _detect_image_ext(img_data)
                    ref_path = tmp_path / f"reference_{len(saved_files) + 1}{ext}"
                    ref_path.write_bytes(img_data)
                    saved_files.append(ref_path)
                else:
                    raw_field = bytearray()
                    while True:
                        chunk = await part.read_chunk(16384)
                        if not chunk:
                            break
                        raw_field.extend(chunk)
                        if len(raw_field) > MAX_FIELD_BYTES:
                            return _json_error(
                                "invalid_request", f"Field '{field_name}' is too large"
                            )
                    try:
                        fields[field_name] = raw_field.decode("utf-8", errors="strict")
                    except UnicodeDecodeError:
                        return _json_error(
                            "invalid_request", f"Field '{field_name}' must be valid UTF-8"
                        )

            prompt = fields.get("prompt", "")
            if not isinstance(prompt, str) or not prompt.strip():
                return _json_error("invalid_request", "Field 'prompt' must be a non-empty string")
            if not saved_files:
                return _json_error("invalid_request", "At least one reference image is required")

            model = str(fields.get("model") or "").strip()
            size = str(fields.get("size") or "").strip()
            img_bytes, err_msg, err_code = await self.generator(
                prompt.strip(),
                reference_files=saved_files,
                model=model,
                size=size,
            )

        if not img_bytes:
            return _json_error(err_code or "backend_error", err_msg or "Image edit failed")
        return web.json_response(
            {
                "created": int(time.time()),
                "data": [{"b64_json": base64.b64encode(img_bytes).decode("ascii")}],
            }
        )

    def _build_app(self) -> web.Application:
        app = web.Application(client_max_size=CLIENT_MAX_SIZE)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_get("/models", self._handle_models)
        app.router.add_post("/v1/images/generations", self._handle_generations)
        app.router.add_post("/images/generations", self._handle_generations)
        app.router.add_post("/v1/images/edits", self._handle_edits)
        app.router.add_post("/images/edits", self._handle_edits)
        return app

    async def start(self) -> int:
        if self._runner is not None:
            return self.port
        app = self._build_app()
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            site = web.TCPSite(runner, self.host, self.port)
            await site.start()
        except Exception:
            await runner.cleanup()
            raise
        self._runner = runner
        self._site = site
        sockets = getattr(getattr(site, "_server", None), "sockets", None) or ()
        if sockets:
            try:
                self.port = int(sockets[0].getsockname()[1])
            except Exception:
                pass
        return self.port

    async def stop(self) -> None:
        runner = self._runner
        self._runner = None
        self._site = None
        if runner is not None:
            try:
                await runner.cleanup()
            except Exception as exc:
                logger.debug(f"[agy_image] 停止本地桥接服务忽略异常: {exc}")
