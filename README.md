# astrbot_plugin_agy_image

通过网页版 Gemini（`gemini_web` 自动读取本地 Firefox Cookie）、`gemini-cli`（`nanobanana`）或本地 `agy` CLI 为 AstrBot 提供生图与改图能力的插件，并内置本地 OpenAI 兼容 Images API 桥接服务（支持与 `astrbot_plugin_private_companion` 等插件配合使用）。

## 功能特性

- **网页版 Gemini 生图/改图（推荐，默认）**：通过 `gemini-webapi` 自动发现并安全读取本地 Firefox 的 Google/Gemini Cookie（或手动填写 Cookie），支持文生图与多参考图编辑。
- **本地 OpenAI 兼容桥接**：在 `127.0.0.1:18765` 提供 `/v1/images/generations` 与 `/v1/images/edits`（及 `/v1/models`、`/health`），可直接接入 `astrbot_plugin_private_companion` 的原生生图配置。
- **独立工具与指令**：提供 `/agy生图 <描述>` 手动测试指令，以及可按需开启/关闭的 `agy_generate_image` LLM 工具。
- **WebUI 连通性测试页**：在 AstrBot WebUI 插件卡片中提供 `connectivity` 视图，支持「连通性测试（仅校验登录与桥接）」和「完整生图测试」。

## 依赖安装

```bash
pip install -r requirements.txt
```

## 快速配置

1. 在本地 Firefox 浏览器登录 [https://gemini.google.com](https://gemini.google.com)（或手动在插件配置中填写 Cookie）。
2. 如使用代理，可在插件配置中填写 `proxy_url`（如 `http://127.0.0.1:10808`）。
3. 若配合 `astrbot_plugin_private_companion` 使用，可在配置中关闭 `expose_llm_tool`，保留本地桥接服务，并在陪伴插件的「生图输出」中选择 OpenAI 兼容端点 `http://127.0.0.1:18765/v1`（模型名填 `gemini-web`）。

## 运行测试

```bash
python -m unittest tests/test_agy_image.py -v
```
