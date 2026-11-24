"""Unit tests for astrbot_plugin_agy_image."""

import asyncio
import json
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))


def _install_astrbot_stub():
    if "astrbot.api" in sys.modules:
        return
    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    comp = types.ModuleType("astrbot.api.message_components")

    api.AstrBotConfig = dict
    api.web = types.SimpleNamespace(
        request=types.SimpleNamespace(json=AsyncMock(return_value={})),
        json_response=lambda data, **kw: data,
    )
    api.logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )

    class FilterStub:
        @staticmethod
        def llm_tool(**kwargs):
            return lambda fn: fn

        @staticmethod
        def command(*args, **kwargs):
            return lambda fn: fn

    class StarStub:
        def __init__(self, context=None):
            self.context = context

    event.AstrMessageEvent = object
    event.MessageChain = lambda items=None: list(items or [])
    event.filter = FilterStub
    star.Context = object
    star.Star = StarStub
    star.register = lambda *a, **k: (lambda cls: cls)
    comp.Image = types.SimpleNamespace(fromFileSystem=lambda p: ("image", p))

    sys.modules["astrbot"] = astrbot
    sys.modules["astrbot.api"] = api
    sys.modules["astrbot.api.event"] = event
    sys.modules["astrbot.api.star"] = star
    sys.modules["astrbot.api.message_components"] = comp


_install_astrbot_stub()
import gemini_web_backend as web_backend  # noqa: E402
import main as plugin_main  # noqa: E402
import openai_bridge  # noqa: E402

import aiohttp  # noqa: E402
import base64  # noqa: E402


class AgyImageTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_stream_json_result_prefers_result_response_and_conv_id(self):
        out = "\n".join([
            "error: outer",
            json.dumps({
                "event": "result",
                "result": {
                    "conversation_id": "conv-123",
                    "status": "ERROR",
                    "error": "interrupted",
                    "response": "No capacity available for model gemini-3.1-flash-image",
                },
            }),
        ])
        conv_id, detail = plugin_main._parse_cli_output(out, "")
        self.assertEqual(conv_id, "conv-123")
        self.assertIn("No capacity available", detail)

    def test_parse_gemini_json_error(self):
        out = json.dumps({"session_id": "s1", "error": {"message": "NANOBANANA_API_KEY not set"}})
        conv_id, detail = plugin_main._parse_cli_output(out, "")
        self.assertEqual(conv_id, "s1")
        self.assertEqual(detail, "NANOBANANA_API_KEY not set")

    def test_safe_filename_sanitizes_traversal_and_adds_ext(self):
        self.assertEqual(plugin_main._safe_filename("../../evil"), "evil.png")
        self.assertEqual(plugin_main._safe_filename("cat.webp"), "cat.webp")

    def test_load_firefox_cookies_filters_google_and_gemini_hosts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "cookies.sqlite"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE moz_cookies (host TEXT, name TEXT, value TEXT, isSecure INTEGER, lastAccessed INTEGER)"
            )
            conn.executemany(
                "INSERT INTO moz_cookies VALUES (?, ?, ?, ?, ?)",
                [
                    (".google.com", "__Secure-1PSID", "psid-val", 1, 100),
                    ("gemini.google.com", "OSID", "osid-val", 1, 100),
                    (".gemini.google.com", "_ga", "ga-val", 0, 100),
                    ("example.com", "unrelated", "nope", 1, 100),
                ],
            )
            conn.commit()
            conn.close()

            cookies = web_backend._load_firefox_cookies(db)

        names = {(h, n, v, s) for h, n, v, s in cookies}
        self.assertEqual(
            names,
            {
                (".google.com", "__Secure-1PSID", "psid-val", True),
                ("gemini.google.com", "OSID", "osid-val", True),
                (".gemini.google.com", "_ga", "ga-val", False),
            },
        )

    def test_normalize_image_extension_renames_jpeg_and_keeps_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            jpg_misnamed = Path(tmp) / "output.png"
            jpg_misnamed.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
            fixed = web_backend._normalize_image_extension(jpg_misnamed)
            self.assertEqual(fixed.name, "output.jpg")
            self.assertTrue(fixed.exists())
            self.assertFalse(jpg_misnamed.exists())

            real_png = Path(tmp) / "real.png"
            real_png.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
            kept = web_backend._normalize_image_extension(real_png)
            self.assertEqual(kept.name, "real.png")
            self.assertTrue(kept.exists())

    def test_ensure_image_prompt_adds_prefix_only_when_missing(self):
        self.assertEqual(
            web_backend._ensure_image_prompt("a cute blue cat"),
            "Generate an image: a cute blue cat",
        )
        self.assertEqual(
            web_backend._ensure_image_prompt("Generate an image of a cat"),
            "Generate an image of a cat",
        )
        self.assertEqual(
            web_backend._ensure_image_prompt("画一张蓝紫猫娘"),
            "画一张蓝紫猫娘",
        )

    async def test_gemini_cli_backend_finds_nanobanana_subdir_and_sends(self):
        plugin = plugin_main.AgyImagePlugin(
            context=None,
            config={"backend": "gemini_cli", "nanobanana_api_key": "test-key", "auto_send_to_user": True},
        )
        event = types.SimpleNamespace(send=AsyncMock())

        with tempfile.TemporaryDirectory() as tmp:
            async def fake_exec(*cmd, **kwargs):
                self.assertIn("--yolo", cmd)
                self.assertIn("nanobanana", cmd)
                self.assertEqual(kwargs["env"]["NANOBANANA_API_KEY"], "test-key")
                out_dir = Path(kwargs["cwd"]) / "nanobanana-output"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "cat.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
                return types.SimpleNamespace(
                    returncode=0,
                    communicate=AsyncMock(return_value=(b'{"session_id":"s1","response":"done"}', b"")),
                )

            with patch.object(plugin_main, "TEMP_ROOT", Path(tmp)), patch.object(
                asyncio, "create_subprocess_exec", side_effect=fake_exec
            ):
                res = await plugin.agy_generate_image(event, "a cute cat")

        self.assertIn("已生成并发送图片", res)
        event.send.assert_awaited_once()

    async def test_agy_backend_returns_cli_error_on_failure(self):
        plugin = plugin_main.AgyImagePlugin(context=None, config={"backend": "agy"})
        event = types.SimpleNamespace(send=AsyncMock())

        with tempfile.TemporaryDirectory() as tmp:
            async def fake_exec(*cmd, **kwargs):
                payload = json.dumps({"event": "result", "result": {"error": "location unsupported"}}).encode()
                return types.SimpleNamespace(
                    returncode=1,
                    communicate=AsyncMock(return_value=(payload, b"")),
                )

            with patch.object(plugin_main, "TEMP_ROOT", Path(tmp)), patch.object(
                asyncio, "create_subprocess_exec", side_effect=fake_exec
            ):
                res = await plugin.agy_generate_image(event, "a cat")

        self.assertEqual(res, "agy 生图失败：location unsupported")
        event.send.assert_not_awaited()

    async def test_gemini_web_backend_sends_image_and_terminate_closes_service(self):
        plugin = plugin_main.AgyImagePlugin(
            context=None,
            config={"backend": "gemini_web", "auto_send_to_user": True},
        )
        event = types.SimpleNamespace(send=AsyncMock())

        with tempfile.TemporaryDirectory() as tmp:
            fake_img = Path(tmp) / "out.jpg"
            fake_img.write_bytes(b"\xff\xd8\xff\xe0jpg")
            plugin._gemini_web.generate_image = AsyncMock(return_value=(fake_img, ""))
            plugin._gemini_web.close = AsyncMock()

            with patch.object(plugin_main, "TEMP_ROOT", Path(tmp)):
                res = await plugin.agy_generate_image(event, "a cute cat")

            await plugin.terminate()

        self.assertIn("已生成并发送图片", res)
        event.send.assert_awaited_once()
        sent_chain = event.send.await_args.args[0]
        self.assertEqual(sent_chain[0], ("image", str(fake_img)))
        plugin._gemini_web.close.assert_awaited_once()

    async def test_gemini_web_service_retries_once_on_auth_error(self):
        from gemini_webapi.exceptions import AuthError

        class GeneratedImage:
            async def save(self, path=".", filename=None, verbose=False, client=None):
                p = Path(path) / (filename or "out.png")
                p.write_bytes(b"\xff\xd8\xff\xe0jpegdata")
                return str(p)

        svc = web_backend.GeminiWebImageService()
        good_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(
                return_value=types.SimpleNamespace(text="ok", images=[GeneratedImage()])
            ),
        )
        bad_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(side_effect=AuthError("expired")),
        )
        svc._ensure_client = AsyncMock(side_effect=[bad_client, good_client])

        with tempfile.TemporaryDirectory() as tmp:
            img_path, err = await svc.generate_image("cat", Path(tmp), "test.png", {}, timeout=30)

        self.assertEqual(err, "")
        self.assertIsNotNone(img_path)
        self.assertEqual(img_path.name, "test.jpg")
        self.assertEqual(svc._ensure_client.await_count, 2)
        self.assertFalse(svc._ensure_client.await_args_list[0].kwargs["force_reinit"])
        self.assertTrue(svc._ensure_client.await_args_list[1].kwargs["force_reinit"])

    def test_normalize_model_name_and_prompt_helpers(self):
        self.assertIsNone(web_backend._normalize_model_name(""))
        self.assertIsNone(web_backend._normalize_model_name("gemini-web"))
        self.assertIsNone(web_backend._normalize_model_name("GEMINI_WEB"))
        self.assertIsNone(web_backend._normalize_model_name("default"))
        self.assertEqual(web_backend._normalize_model_name("gemini-2.5-flash"), "gemini-2.5-flash")

        self.assertEqual(web_backend._detect_image_ext(b"\xff\xd8\xff\xe0abc"), ".jpg")
        self.assertEqual(web_backend._detect_image_ext(b"\x89PNG\r\n\x1a\nabc"), ".png")
        self.assertEqual(web_backend._detect_image_ext(b"RIFF1234WEBPabc"), ".webp")
        self.assertEqual(web_backend._detect_image_ext(b"unknown"), ".png")

        self.assertEqual(
            web_backend._ensure_image_prompt("a cup", size="768x1024"),
            "Generate an image: a cup (aspect ratio 3:4)",
        )
        self.assertEqual(
            web_backend._ensure_image_prompt("a cup portrait 3:4", size="768x1024"),
            "Generate an image: a cup portrait 3:4",
        )
        edit_prompt = web_backend._ensure_image_prompt(
            "Reference images are uploaded in this exact order:\nimage 1 = identity\nUser request:\nselfie at cafe"
            ,
            is_edit=True,
        )
        self.assertTrue(edit_prompt.startswith("Edit and generate a new image"))
        self.assertIn("selfie at cafe", edit_prompt)

    async def test_gemini_web_generate_image_result_passes_files_and_classifies_errors(self):
        from gemini_webapi.exceptions import AuthError

        class GeneratedImage:
            async def save(self, path=".", filename=None, verbose=False, client=None):
                p = Path(path) / (filename or "out.png")
                p.write_bytes(b"\x89PNG\r\n\x1a\nfakepng")
                return str(p)

        svc = web_backend.GeminiWebImageService()
        ok_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(
                return_value=types.SimpleNamespace(text="ok", images=[GeneratedImage()])
            ),
        )
        policy_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(
                return_value=types.SimpleNamespace(
                    text="I cannot generate that image due to safety policy.", images=[]
                )
            ),
        )
        empty_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(return_value=types.SimpleNamespace(text="Just text reply", images=[])),
        )
        timeout_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(side_effect=asyncio.TimeoutError()),
        )
        auth_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(side_effect=AuthError("bad cookies")),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref = tmp_path / "ref1.jpg"
            ref.write_bytes(b"\xff\xd8\xff\xe0ref")

            svc._ensure_client = AsyncMock(return_value=ok_client)
            res = await svc.generate_image_result(
                "change background",
                tmp_path,
                "out.png",
                {"model": "gemini-web"},
                reference_files=[ref],
                model_override="gemini-web",
            )
            self.assertIsNotNone(res.path)
            self.assertEqual(res.error_code, "")
            kwargs = ok_client.generate_content.await_args.kwargs
            self.assertIsNone(kwargs["model"])
            self.assertEqual(kwargs["files"], [str(ref)])

            svc._ensure_client = AsyncMock(return_value=policy_client)
            res_policy = await svc.generate_image_result("x", tmp_path, "p.png", {})
            self.assertEqual(res_policy.error_code, "content_policy")

            svc._ensure_client = AsyncMock(return_value=empty_client)
            res_empty = await svc.generate_image_result("x", tmp_path, "e.png", {})
            self.assertEqual(res_empty.error_code, "no_image")

            svc._ensure_client = AsyncMock(return_value=timeout_client)
            res_to = await svc.generate_image_result("x", tmp_path, "t.png", {})
            self.assertEqual(res_to.error_code, "timeout")

            svc._ensure_client = AsyncMock(return_value=auth_client)
            res_auth = await svc.generate_image_result("x", tmp_path, "a.png", {})
            self.assertEqual(res_auth.error_code, "auth_error")

    async def test_openai_bridge_endpoints_auth_multipart_and_errors(self):
        captured: list[tuple[str, list[bytes], list[str], str, str]] = []
        mode = {"behavior": "ok"}

        async def fake_gen(prompt: str, *, reference_files=None, model="", size=""):
            if mode["behavior"] != "ok":
                return None, "simulated failure", mode["behavior"]
            raws = [Path(p).read_bytes() for p in (reference_files or [])]
            suffixes = [Path(p).suffix for p in (reference_files or [])]
            captured.append((prompt, raws, suffixes, model, size))
            return b"\x89PNG\r\n\x1a\nbridge_out", "", ""

        server = openai_bridge.OpenAIImageBridgeServer(
            host="127.0.0.1",
            port=0,
            api_key="secret-token",
            generator=fake_gen,
        )
        port = await server.start()
        base_url = f"http://127.0.0.1:{port}"
        try:
            async with aiohttp.ClientSession(trust_env=False) as client:
                async with client.get(f"{base_url}/health") as r:
                    self.assertEqual(r.status, 200)
                    self.assertEqual((await r.json())["status"], "ok")

                async with client.get(f"{base_url}/v1/models") as r:
                    self.assertEqual(r.status, 200)
                    data = await r.json()
                    self.assertEqual(data["data"][0]["id"], "gemini-web")

                # Missing auth -> 401
                async with client.post(
                    f"{base_url}/v1/images/generations",
                    json={"prompt": "cat", "model": "gemini-web"},
                ) as r:
                    self.assertEqual(r.status, 401)

                headers = {"Authorization": "Bearer secret-token"}
                async with client.post(
                    f"{base_url}/v1/images/generations",
                    headers=headers,
                    json={"prompt": "cat", "model": "gemini-web", "size": "768x1024"},
                ) as r:
                    self.assertEqual(r.status, 200)
                    body = await r.json()
                    self.assertEqual(base64.b64decode(body["data"][0]["b64_json"]), b"\x89PNG\r\n\x1a\nbridge_out")

                # Multipart edit with >1MB payload, two image[] fields, and JPEG bytes sent as .png filename
                import io

                big_jpeg = b"\xff\xd8\xff\xe0" + (b"A" * (1_500_000))
                small_webp = b"RIFF1234WEBPdata"
                form = aiohttp.FormData()
                form.add_field("prompt", "edit scene with ref")
                form.add_field("model", "gemini-web")
                form.add_field("image[]", io.BytesIO(big_jpeg), filename="reference-1.png", content_type="image/png")
                form.add_field("image[]", io.BytesIO(small_webp), filename="reference-2.png", content_type="image/png")
                async with client.post(f"{base_url}/v1/images/edits", headers=headers, data=form) as r:
                    self.assertEqual(r.status, 200)
                    body = await r.json()
                    self.assertEqual(base64.b64decode(body["data"][0]["b64_json"]), b"\x89PNG\r\n\x1a\nbridge_out")

                self.assertEqual(captured[1][0], "edit scene with ref")
                self.assertEqual(captured[1][1], [big_jpeg, small_webp])
                self.assertEqual(captured[1][2], [".jpg", ".webp"])

                # Error mappings
                for code, expected_status in (
                    ("content_policy", 400),
                    ("auth_error", 401),
                    ("no_image", 422),
                    ("rate_limited", 429),
                    ("timeout", 504),
                ):
                    mode["behavior"] = code
                    async with client.post(
                        f"{base_url}/v1/images/generations",
                        headers=headers,
                        json={"prompt": "x"},
                    ) as r:
                        self.assertEqual(r.status, expected_status)
                        err_body = await r.json()
                        if code == "content_policy":
                            self.assertIn("content_policy", err_body["error"]["message"].lower())
        finally:
            await server.stop()

    async def test_initialize_starts_bridge_and_deactivates_tool_when_configured(self):
        ctx = types.SimpleNamespace(deactivate_llm_tool_async=AsyncMock(return_value=True))
        plugin = plugin_main.AgyImagePlugin(
            context=ctx,
            config={
                "backend": "gemini_web",
                "openai_bridge_enabled": True,
                "openai_bridge_host": "127.0.0.1",
                "openai_bridge_port": 0,
                "expose_llm_tool": False,
            },
        )
        await plugin.initialize()
        try:
            self.assertIsNotNone(plugin._bridge)
            self.assertTrue(plugin._bridge.is_running)
            self.assertGreater(plugin._bridge.port, 0)
            self.assertNotEqual(plugin._bridge.port, 18765)
            ctx.deactivate_llm_tool_async.assert_awaited_once_with("agy_generate_image")
            async with aiohttp.ClientSession(trust_env=False) as client:
                async with client.get(f"http://127.0.0.1:{plugin._bridge.port}/health") as r:
                    self.assertEqual(r.status, 200)
        finally:
            await plugin.terminate()
        self.assertIsNone(plugin._bridge)

    async def test_companion_online_image_backend_contract_with_bridge(self):
        companion_root = PLUGIN_ROOT.parent / "astrbot_plugin_private_companion"
        if not companion_root.is_dir():
            self.skipTest("astrbot_plugin_private_companion not present")
        if str(companion_root) not in sys.path:
            sys.path.insert(0, str(companion_root))

        import hashlib
        import io
        from PIL import Image
        from astrbot_host.native_extensions.image_native.output_routing import OnlineImageBackend
        from astrbot_host.native_extensions.image_native.types import ImageInvocation, ReferenceImage

        def make_png(color):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), color).save(buf, format="PNG")
            return buf.getvalue()

        def make_jpg(color):
            buf = io.BytesIO()
            Image.new("RGB", (32, 32), color).save(buf, format="JPEG")
            return buf.getvalue()

        out_png = make_png((10, 20, 30))
        ref1_jpg = make_jpg((200, 10, 10))
        ref2_png = make_png((10, 200, 10))

        seen_calls = []

        class GeneratedImage:
            async def save(self, path=".", filename=None, verbose=False, client=None):
                p = Path(path) / (filename or "out.png")
                p.write_bytes(out_png)
                return str(p)

        fake_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(
                side_effect=lambda prompt, **kw: (
                    seen_calls.append((prompt, kw)),
                    types.SimpleNamespace(text="ok", images=[GeneratedImage()]),
                )[1]
            ),
        )
        plugin = plugin_main.AgyImagePlugin(
            context=None,
            config={
                "backend": "gemini_web",
                "openai_bridge_enabled": True,
                "openai_bridge_host": "127.0.0.1",
                "openai_bridge_port": 0,
            },
        )
        plugin._gemini_web._ensure_client = AsyncMock(return_value=fake_client)
        await plugin.initialize()
        try:
            endpoint = {
                "id": "gemini_web_local",
                "name": "Gemini Web",
                "enabled": True,
                "platform": "openai",
                "model": "gemini-web",
                "width": 1024,
                "height": 1024,
                "size_label": "1024x1024",
                "quality": "",
                "response_format": "b64_json",
                "max_reference_images": 4,
                "base_url": f"http://127.0.0.1:{plugin._bridge.port}/v1",
                "api_key": "",
                "timeout_seconds": 60,
            }
            backend = OnlineImageBackend(endpoint)
            self.assertEqual(backend.operations, ("generate", "edit"))

            async def before_dispatch():
                return True

            with tempfile.TemporaryDirectory() as tmp, patch.object(plugin_main, "TEMP_ROOT", Path(tmp)):
                inv_gen = ImageInvocation(
                    "scope", "req1", "inv1", "origin", "generate", "a blue mug on desk"
                )
                res_gen = await backend.execute(inv_gen, (), before_dispatch)
                self.assertEqual(res_gen.images, (out_png,))

                refs = (
                    ReferenceImage(
                        "ref-id", ref1_jpg, "src-1", hashlib.sha256(ref1_jpg).hexdigest(), role="identity"
                    ),
                    ReferenceImage(
                        "ref-pose", ref2_png, "src-2", hashlib.sha256(ref2_png).hexdigest(), role="pose"
                    ),
                )
                inv_edit = ImageInvocation(
                    "scope",
                    "req2",
                    "inv2",
                    "origin",
                    "edit",
                    "selfie in coffee shop",
                    reference_ids=("ref-id", "ref-pose"),
                )
                res_edit = await backend.execute(inv_edit, refs, before_dispatch)
                self.assertEqual(res_edit.images, (out_png,))

            self.assertEqual(len(seen_calls), 2)
            self.assertIsNone(seen_calls[0][1]["model"])
            self.assertNotIn("files", seen_calls[0][1])
            edit_prompt, edit_kw = seen_calls[1]
            self.assertTrue(edit_prompt.startswith("Edit and generate a new image"))
            self.assertIn("image 1 = identity", edit_prompt)
            self.assertIn("image 2 = pose", edit_prompt)
            self.assertEqual(len(edit_kw["files"]), 2)
            self.assertTrue(edit_kw["files"][0].endswith(".jpg"))
            self.assertTrue(edit_kw["files"][1].endswith(".png"))
        finally:
            await plugin.terminate()

    async def test_connectivity_check_ping_and_generate_via_bridge(self):
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (30, 60, 90)).save(buf, format="PNG")
        out_png = buf.getvalue()

        class GeneratedImage:
            async def save(self, path=".", filename=None, verbose=False, client=None):
                p = Path(path) / (filename or "out.png")
                p.write_bytes(out_png)
                return str(p)

        fake_client = types.SimpleNamespace(
            client=None,
            generate_content=AsyncMock(
                return_value=types.SimpleNamespace(text="ok", images=[GeneratedImage()])
            ),
        )
        registered = []
        ctx = types.SimpleNamespace(
            register_web_api=lambda route, fn, methods, desc: registered.append((route, methods)),
            deactivate_llm_tool_async=AsyncMock(return_value=True),
        )
        plugin = plugin_main.AgyImagePlugin(
            context=ctx,
            config={
                "backend": "gemini_web",
                "gemini_web_1psid": "dummy-psid",
                "openai_bridge_enabled": True,
                "openai_bridge_host": "127.0.0.1",
                "openai_bridge_port": 0,
                "expose_llm_tool": False,
            },
        )
        self.assertEqual(
            registered,
            [
                ("/astrbot_plugin_agy_image/status", ["GET"]),
                ("/astrbot_plugin_agy_image/connectivity-test", ["POST"]),
            ],
        )
        plugin._gemini_web._ensure_client = AsyncMock(return_value=fake_client)
        await plugin.initialize()
        try:
            ping_res = await plugin.test_connectivity(generate_image=False)
            self.assertTrue(ping_res["ok"])
            self.assertTrue(ping_res["bridge_health"]["ok"])
            self.assertTrue(ping_res["backend_check"]["ok"])
            self.assertEqual(ping_res["backend_check"]["source"], "manual")
            self.assertNotIn("dummy-psid", json.dumps(ping_res))

            with tempfile.TemporaryDirectory() as tmp, patch.object(plugin_main, "TEMP_ROOT", Path(tmp)):
                gen_res = await plugin.test_connectivity(generate_image=True)
            self.assertTrue(gen_res["ok"])
            self.assertEqual(gen_res["mime_type"], "image/png")
            self.assertEqual(base64.b64decode(gen_res["image_base64"]), out_png)
        finally:
            await plugin.terminate()
