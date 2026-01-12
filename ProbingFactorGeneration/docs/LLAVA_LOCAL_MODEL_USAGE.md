# 本地 LLaVA 模型使用指南

## 概述

本框架现在支持使用本地的 LLaVA OneVision 模型作为 baseline 模型。通过 `AsyncLLaVAClient`，您可以运行本地模型推理，同时保持与 API 模型相同的接口。

## 安装依赖

使用本地 LLaVA 模型需要安装以下依赖：

```bash
pip install transformers torch torchvision pillow
```

如果使用量化（8-bit 或 4-bit），还需要：

```bash
pip install bitsandbytes
```

## 基本使用

### 1. 直接使用 BaselineModel（推荐）

```python
from ProbingFactorGeneration.models import BaselineModel
from PIL import Image

# 初始化模型，指定模型路径
baseline_model = BaselineModel(
    model_path="/path/to/your/llava/model",  # 您的模型路径
    gpu_id=0,  # 使用的 GPU ID
    max_concurrent=1,  # 本地模型建议并发数为 1
    model_config={
        "temperature": 0.3,
        "max_tokens": 2048,
    }
)

# 使用 async context manager
async with baseline_model:
    # 加载图像
    image = Image.open("test_image.jpg")
    
    # 创建 claim template
    claim_template = {
        "claim_template": "The [OBJECT] is located in the [REGION] of the image.",
        "placeholders": ["OBJECT", "REGION"],
        "slots": {
            "OBJECT": {
                "type": "object_instance",
                "selection_criteria": "Select the most salient object"
            },
            "REGION": {
                "type": "categorical_value",
                "values": ["top", "bottom", "left", "right", "center"]
            }
        },
        "metadata": {
            "baseline_instructions": ["Identify objects in the image"],
            "expected_outputs": ["TRUE", "FALSE", "NOT_RELATED"]
        }
    }
    
    # 完成模板
    completion = await baseline_model.complete_template_async(image, claim_template)
    print(completion)
```

### 2. 在 Pipeline 中使用

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

# 使用本地 LLaVA 模型
baseline_model = BaselineModel(
    model_path="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model",
    gpu_id=0,
    max_concurrent=1,
    model_config={
        "temperature": 0.3,
        "max_tokens": 2048,
    }
)

judge_model = JudgeModel(model_name="gemini-pro-vision")  # Judge 可以使用 API 模型
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
async with baseline_model:
    result = await pipeline.process_single_image_with_templates_async("image.jpg")
    print(result)
```

### 3. 直接使用 AsyncLLaVAClient

```python
from ProbingFactorGeneration.utils import AsyncLLaVAClient
from PIL import Image
import base64
import io

async def test_llava_client():
    # 创建客户端
    client = AsyncLLaVAClient(
        model_path="/path/to/your/llava/model",
        gpu_id=0,
        max_concurrent=1,
    )
    
    async with client:
        # 加载图像
        image = Image.open("test_image.jpg")
        
        # 转换为 base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 调用模型
        response = await client.chat.completions.create(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        
        print(response.choices[0].message.content)

# 运行
import asyncio
asyncio.run(test_llava_client())
```

## 配置选项

### BaselineModel 参数

- `model_path` (str, required): 本地模型目录路径（Hugging Face 格式）
- `gpu_id` (int, default=0): 使用的 GPU ID
- `max_concurrent` (int, default=1): 最大并发请求数（本地模型建议设为 1）
- `model_config` (dict): 模型配置字典，可包含：
  - `temperature` (float): 采样温度
  - `max_tokens` (int): 最大生成 token 数
  - `device` (str): 设备 ("cuda" 或 "cpu")
  - `torch_dtype`: PyTorch 数据类型
  - `load_in_8bit` (bool): 是否使用 8-bit 量化
  - `load_in_4bit` (bool): 是否使用 4-bit 量化

### AsyncLLaVAClient 参数

- `model_path` (str, required): 模型路径
- `gpu_id` (int, optional): GPU ID（设置 CUDA_VISIBLE_DEVICES）
- `max_concurrent` (int, default=1): 最大并发数
- `request_delay` (float, default=0.0): 请求间延迟（秒）
- `device` (str, optional): 设备 ("cuda" 或 "cpu")
- `torch_dtype` (torch.dtype, optional): 数据类型
- `load_in_8bit` (bool, default=False): 8-bit 量化
- `load_in_4bit` (bool, default=False): 4-bit 量化

## 性能优化建议

1. **并发设置**: 本地模型通常建议 `max_concurrent=1`，因为模型推理是计算密集型的
2. **GPU 内存**: 如果 GPU 内存不足，可以考虑：
   - 使用量化 (`load_in_8bit=True` 或 `load_in_4bit=True`)
   - 使用 `torch_dtype=torch.float16`
3. **批处理**: 对于批量处理，建议使用 `complete_template_batch_async` 方法

## 故障排除

### 1. 模型加载失败

如果遇到模型加载问题，检查：
- 模型路径是否正确
- transformers 版本是否兼容
- GPU 内存是否充足

### 2. CUDA 错误

如果遇到 CUDA 错误：
- 检查 GPU 是否可用：`torch.cuda.is_available()`
- 检查 GPU ID 是否正确
- 尝试设置 `device="cpu"` 进行测试

### 3. 内存不足

如果 GPU 内存不足：
- 减少 `max_concurrent` 到 1
- 使用量化：`load_in_8bit=True`
- 使用更小的 `max_tokens`

## 示例：您的模型路径

根据您提供的路径，使用方式如下：

```python
baseline_model = BaselineModel(
    model_path="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model",
    gpu_id=0,
    max_concurrent=1,
)
```

## 注意事项

1. 本地模型首次加载需要时间，建议使用 async context manager 来管理模型生命周期
2. 确保模型路径是有效的 Hugging Face 格式模型目录
3. 本地模型的并发能力有限，建议 `max_concurrent=1`
4. JSON 格式输出：模型会尝试生成 JSON，但可能需要后处理来确保有效性
