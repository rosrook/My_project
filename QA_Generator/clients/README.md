# QA_Generator/clients

本目录包含对外模型调用的同步/异步客户端封装，统一提供图像 + 文本的推理接口。

## 直接使用方式

### 同步客户端（GeminiClient）
```python
from QA_Generator.clients.gemini_client import GeminiClient

client = GeminiClient()
result = client.analyze_image("path/to/image.jpg", "Describe the image")
print(result)
```

### 异步客户端（AsyncGeminiClient）
```python
import asyncio
from QA_Generator.clients.async_client import AsyncGeminiClient

async def run():
    async with AsyncGeminiClient(max_concurrent=3) as client:
        result = await client.analyze_image_async("path/to/image.jpg", "Describe the image")
        print(result)

asyncio.run(run())
```

## 在 QA 生成流程中的接口形式

- `AnswerGenerator` 和 `QuestionGeneratorPrefill` 通过 `GeminiClient` / `AsyncGeminiClient` 调用模型。
- pipeline 中异步答案生成调用 `AsyncGeminiClient` 以支持并发。
- 配置默认读取 `QA_Generator/config/config.py`（如 `SERVICE_NAME / ENV / MODEL_NAME`）。

## 文件说明

### `gemini_client.py`
**同步客户端封装**：
- 提供 `analyze_image(...)`、`filter_image(...)` 等接口
- 支持多种图片输入类型（path / bytes / PIL / base64 / URL）
- 依赖 `QA_Generator/config/config.py` 中的服务配置

### `async_client.py`
**异步客户端封装**：
- 提供 `analyze_image_async(...)`、`filter_image_async(...)`
- 支持并发控制（`max_concurrent`）与请求间隔（`request_delay`）
- 内部实现了 OpenAI 兼容的 `chat.completions.create` 接口，供上层统一调用

## 注意事项

- 客户端默认依赖 `QA_Generator/config/config.py` 中的配置项
- 如果使用 LBOpenAIClient 方式，需要设置正确的 `SERVICE_NAME / ENV / MODEL_NAME`
