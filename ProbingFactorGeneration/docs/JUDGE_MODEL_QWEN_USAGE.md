# Judge 模型 Qwen 使用指南

## 概述

JudgeModel 支持使用 Qwen 模型通过 LBOpenAIAsyncClient 进行异步调用。基于您提供的配置示例，以下是正确的配置方式。

## 配置方式

### 使用环境变量配置（推荐）

```bash
# 设置 Qwen 服务配置（使用 LBOpenAIAsyncClient）
export SERVICE_NAME="mediak8s-editprompt-qwen235b"
export ENV="prod"
export API_KEY="1"  # LBOpenAIAsyncClient 通常使用 "1" 作为 api_key
export USE_LB_CLIENT="true"
export JUDGE_MODEL_NAME="/workspace/Qwen3-VL-235B-A22B-Instruct"
```

### 在代码中配置

```python
from ProbingFactorGeneration.models import JudgeModel

# 使用 LBOpenAIAsyncClient 调用 Qwen 模型
judge_model = JudgeModel(
    model_name="/workspace/Qwen3-VL-235B-A22B-Instruct",
    use_lb_client=True,  # 使用 LBOpenAIAsyncClient
    max_concurrent=10,
    model_config={
        "temperature": 0.3,
        "max_tokens": 2000,
    }
)

# SERVICE_NAME 和 ENV 需要从 MODEL_CONFIG 或环境变量读取
# 确保设置了：
# - SERVICE_NAME="mediak8s-editprompt-qwen235b"
# - ENV="prod"
# - API_KEY="1"
# - USE_LB_CLIENT="true"
```

### 完整的 Pipeline 使用示例

```python
from ProbingFactorGeneration.core import ImageLoader, TemplateClaimGenerator
from ProbingFactorGeneration.models import BaselineModel, JudgeModel
from ProbingFactorGeneration.core import FailureAggregator, FilteringFactorMapper
from ProbingFactorGeneration.io import DataSaver
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline

# 初始化组件
image_loader = ImageLoader(image_dir="./data/images")
template_generator = TemplateClaimGenerator(
    config_path="configs/claim_template.example_v1_1.json"
)

# Baseline 模型（可以使用本地模型或 API）
baseline_model = BaselineModel(
    model_path="/path/to/llava/model",  # 或使用 API 模型
    gpu_id=0,
    max_concurrent=1,
)

# Judge 模型（使用 Qwen via LBOpenAIAsyncClient）
judge_model = JudgeModel(
    model_name="/workspace/Qwen3-VL-235B-A22B-Instruct",
    use_lb_client=True,  # 使用 LBOpenAIAsyncClient
    max_concurrent=10,
    model_config={
        "temperature": 0.3,
        "max_tokens": 2000,
    }
)

failure_aggregator = FailureAggregator()
filtering_factor_mapper = FilteringFactorMapper()
data_saver = DataSaver(output_dir="./output")

# 创建 pipeline
pipeline = ProbingFactorPipeline(
    image_loader=image_loader,
    claim_generator=template_generator,
    baseline_model=baseline_model,
    judge_model=judge_model,
    failure_aggregator=failure_aggregator,
    filtering_factor_mapper=filtering_factor_mapper,
    data_saver=data_saver
)

# 处理图像
async with baseline_model, judge_model:
    result = await pipeline.process_single_image_with_templates_async("image.jpg")
    print(result)
```

## 配置说明

### 必需的配置项

1. **SERVICE_NAME**: `"mediak8s-editprompt-qwen235b"`
2. **ENV**: `"prod"`
3. **API_KEY**: `"1"` (LBOpenAIAsyncClient 通常使用 "1")
4. **USE_LB_CLIENT**: `"true"` 或 `True`
5. **MODEL_NAME**: `"/workspace/Qwen3-VL-235B-A22B-Instruct"`

### JudgeModel 参数

- `model_name`: 模型名称，例如 `"/workspace/Qwen3-VL-235B-A22B-Instruct"`
- `use_lb_client`: 必须设置为 `True` 以使用 LBOpenAIAsyncClient
- `max_concurrent`: 最大并发请求数（建议 10）
- `model_config`: 模型配置字典
  - `temperature`: 采样温度（默认 0.3）
  - `max_tokens`: 最大生成 token 数（默认 2000）

### 环境变量配置

在运行代码前，确保设置了以下环境变量：

```bash
export SERVICE_NAME="mediak8s-editprompt-qwen235b"
export ENV="prod"
export API_KEY="1"
export USE_LB_CLIENT="true"
```

或者在 Python 代码中通过 `MODEL_CONFIG` 设置：

```python
import os
os.environ["SERVICE_NAME"] = "mediak8s-editprompt-qwen235b"
os.environ["ENV"] = "prod"
os.environ["API_KEY"] = "1"
os.environ["USE_LB_CLIENT"] = "true"
```

## 依赖安装

确保已安装 `redeuler` 包：

```bash
pip install redeuler
```

如果未安装，代码会自动尝试安装（如果 AsyncGeminiClient 中有相关逻辑）。

## 注意事项

1. **LBOpenAIAsyncClient**: JudgeModel 使用 `AsyncGeminiClient`，它内部使用 `LBOpenAIAsyncClient` 当 `use_lb_client=True` 时
2. **服务名称**: 确保 `SERVICE_NAME` 与您的 redeuler 服务配置一致
3. **模型名称**: 使用完整的模型路径 `/workspace/Qwen3-VL-235B-A22B-Instruct`
4. **并发控制**: 根据服务能力调整 `max_concurrent` 参数
5. **环境**: 确保 `ENV` 设置为正确的环境（"prod" 或 "staging"）

## 验证配置

您可以使用以下代码验证配置是否正确：

```python
import os
from ProbingFactorGeneration.models import JudgeModel

# 设置环境变量
os.environ["SERVICE_NAME"] = "mediak8s-editprompt-qwen235b"
os.environ["ENV"] = "prod"
os.environ["API_KEY"] = "1"
os.environ["USE_LB_CLIENT"] = "true"

# 创建 JudgeModel
judge_model = JudgeModel(
    model_name="/workspace/Qwen3-VL-235B-A22B-Instruct",
    use_lb_client=True,
)

# 测试连接（需要实际的图像和模板）
# async with judge_model:
#     # 进行验证测试
#     pass
```

## 故障排除

### 1. 导入错误

如果遇到 `LBOpenAIAsyncClient` 导入错误：

```bash
pip install redeuler
```

### 2. 服务名称错误

确保 `SERVICE_NAME` 与您的 redeuler 服务配置完全一致。

### 3. 认证错误

LBOpenAIAsyncClient 通常使用 `api_key="1"`，确保环境变量中设置了 `API_KEY="1"`。

### 4. 模型路径错误

确保模型路径 `/workspace/Qwen3-VL-235B-A22B-Instruct` 在服务端是可访问的。
