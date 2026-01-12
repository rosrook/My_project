# 故障排除指南

## BASE_URL 错误

### 错误信息
```
ValueError: Base URL not set. Please set BASE_URL in MODEL_CONFIG or pass base_url parameter.
```

### 原因
当使用 `LBOpenAIAsyncClient` 时，如果以下条件不满足，代码会回退到需要 `BASE_URL` 的模式：
1. `redeuler` 库未安装（导致 `HAS_LB_CLIENT = False`）
2. 或者 `USE_LB_CLIENT` 环境变量未设置为 `"true"`

### 解决方案

#### 方案 1: 使用 LBOpenAIAsyncClient（推荐，如果使用 Qwen 模型）

1. **安装 redeuler 库**（如果未安装）：
```bash
pip install redeuler
```

2. **设置环境变量**：
```bash
export SERVICE_NAME="mediak8s-editprompt-qwen235b"  # 根据你的服务名称
export ENV="prod"  # 或 "staging"
export API_KEY="1"  # 或你的实际 API key
export USE_LB_CLIENT="true"
```

3. **运行脚本时传递参数**：
```bash
python ProbingFactorGeneration/examples/run_complete_pipeline.py \
    --parquet_dir /path/to/parquet \
    --judge_model_name /workspace/Qwen3-VL-235B-A22B-Instruct \
    --sample_size 10
```

#### 方案 2: 使用标准的 API 模型（如果不需要 LBOpenAIAsyncClient）

如果你使用的是标准的 API（如 Gemini），需要设置 `BASE_URL`：

```bash
export BASE_URL="https://api.example.com"
export API_KEY="your-api-key"
export USE_LB_CLIENT="false"
```

### 检查步骤

1. **检查 redeuler 是否安装**：
```python
python -c "from redeuler.client.openai import LBOpenAIAsyncClient; print('OK')"
```

2. **检查环境变量**：
```bash
echo $USE_LB_CLIENT
echo $SERVICE_NAME
echo $ENV
echo $API_KEY
```

3. **验证配置**：
如果使用 Qwen 模型，确保设置了 `SERVICE_NAME`、`ENV` 和 `USE_LB_CLIENT="true"`。

## 常见问题

### 1. redeuler 未安装
**错误**: `ImportError: cannot import name 'LBOpenAIAsyncClient'`

**解决**: 
```bash
pip install redeuler
```

### 2. SERVICE_NAME 未设置
**错误**: `ValueError: Service Name not set. Please set SERVICE_NAME in MODEL_CONFIG...`

**解决**: 设置环境变量或传递参数：
```bash
export SERVICE_NAME="your-service-name"
```

或在代码中传递：
```python
judge_model = JudgeModel(
    model_name="/workspace/Qwen3-VL-235B-A22B-Instruct",
    use_lb_client=True
)
# 同时需要设置环境变量 SERVICE_NAME, ENV, API_KEY
```

### 3. 配置混淆
**注意**: 
- `LBOpenAIClient`（同步版本，如你的 debug_qwen3vl.py 中使用的）
- `LBOpenAIAsyncClient`（异步版本，代码中使用的）

两者都来自 `redeuler.client.openai`，但我们的代码使用异步版本。
