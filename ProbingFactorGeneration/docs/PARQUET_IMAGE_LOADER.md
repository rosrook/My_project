# Parquet Image Loader 使用说明

## 概述

`ImageLoader` 现在支持从 Parquet 文件加载图像数据，特别适用于 OpenImages 等大规模数据集。

## 功能特性

- **从 Parquet 文件加载**: 自动读取目录下的所有 `.parquet` 文件
- **采样控制**: 支持设置采样数量 (`sample_size`)
- **随机种子**: 支持设置随机种子以确保可重复的采样
- **多种数据格式**: 支持图像路径和图像字节数据
- **灵活的数据源**: 支持直接文件路径和 Parquet 数据源混合使用

## 使用方法

### 基本使用

```python
from ProbingFactorGeneration.core import ImageLoader

# 从 parquet 目录加载，采样 1000 张图像
loader = ImageLoader(
    parquet_dir="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/",
    sample_size=1000,
    random_seed=42
)

# 获取所有采样后的图像路径
image_paths = loader.get_all_image_paths()
print(f"Loaded {len(image_paths)} images")

# 加载单张图像
image = loader.load(image_paths[0])

# 批量加载
images = loader.load_batch(image_paths[:10])
```

### 使用所有图像（不采样）

```python
loader = ImageLoader(
    parquet_dir="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/",
    sample_size=None  # 或省略此参数，使用所有图像
)

image_paths = loader.get_all_image_paths()
```

### 在 Pipeline 中使用

```python
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline
from ProbingFactorGeneration.core import ImageLoader, TemplateClaimGenerator, FailureAggregator, FilteringFactorMapper
from ProbingFactorGeneration.models import BaselineModel, JudgeModel
from ProbingFactorGeneration.io import DataSaver

# 创建 ImageLoader 并采样 100 张图像
image_loader = ImageLoader(
    parquet_dir="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/",
    sample_size=100,
    random_seed=42
)

# 获取图像路径列表
image_paths = image_loader.get_all_image_paths()

# 创建 pipeline（其他组件初始化略）
pipeline = ProbingFactorPipeline(
    image_loader=image_loader,
    claim_generator=template_generator,
    baseline_model=baseline_model,
    judge_model=judge_model,
    failure_aggregator=FailureAggregator(),
    filtering_factor_mapper=FilteringFactorMapper(),
    data_saver=DataSaver(output_dir="./output")
)

# 处理所有采样的图像
async with baseline_model, judge_model:
    results = await pipeline.process_batch_with_templates_async(image_paths)
```

### 动态调整采样大小

```python
loader = ImageLoader(
    parquet_dir="/path/to/parquet/files",
    sample_size=1000
)

# 获取 1000 张图像
paths_1000 = loader.get_all_image_paths()

# 改为采样 500 张
loader.set_sample_size(500)
paths_500 = loader.get_all_image_paths()
```

## Parquet 文件格式

`ImageLoader` 支持以下 Parquet 文件格式：

### 格式 1: 包含图像路径

Parquet 文件应包含以下列之一：
- `image_path`: 图像文件路径（推荐）
- `path`: 图像文件路径（备用）
- `file_path`: 图像文件路径（备用）

可选列：
- `image_id`: 图像唯一标识符（如果缺失，将从路径提取）
- 其他元数据列（会被保留在 metadata 中）

示例：
```python
import pandas as pd

df = pd.DataFrame({
    'image_path': ['/path/to/image1.jpg', '/path/to/image2.jpg'],
    'image_id': ['img_001', 'img_002'],
    'label': ['cat', 'dog']
})
df.to_parquet('images.parquet')
```

### 格式 2: 包含图像字节数据

Parquet 文件应包含：
- `image_bytes`: 图像的字节数据（bytes 或 base64 编码的字符串）
- `image_id`: 图像唯一标识符（必需）

示例：
```python
import pandas as pd
import base64
from PIL import Image
import io

# 将图像编码为字节
image = Image.open('image.jpg')
buffer = io.BytesIO()
image.save(buffer, format='JPEG')
image_bytes = buffer.getvalue()

df = pd.DataFrame({
    'image_id': ['img_001'],
    'image_bytes': [image_bytes],  # 或 base64.b64encode(image_bytes).decode()
    'label': ['cat']
})
df.to_parquet('images.parquet')
```

## 参数说明

### `__init__` 参数

- `image_dir` (str, Path, optional): 图像文件目录（用于直接文件路径加载）
- `batch_size` (int): 批处理大小（默认: 1）
- `parquet_dir` (str, Path, optional): Parquet 文件目录路径
- `sample_size` (int, optional): 采样数量（None = 使用所有图像）
- `random_seed` (int, optional): 随机种子（用于可重复采样）

### 主要方法

#### `get_all_image_paths() -> List[str]`
获取所有图像路径（采样后）。

#### `load(image_path: Union[str, Path]) -> Image.Image`
加载单张图像。

#### `load_batch(image_paths: List[Union[str, Path]]) -> List[Image.Image]`
批量加载图像。

#### `get_image_id(image_path: Union[str, Path]) -> str`
获取图像 ID。

#### `set_sample_size(sample_size: Optional[int])`
更新采样大小并重新加载。

## 依赖安装

```bash
pip install pandas pyarrow
```

或者从 `requirements.txt` 安装：
```bash
pip install -r requirements.txt
```

## 注意事项

1. **性能**: 首次调用 `get_all_image_paths()` 时会读取所有 parquet 文件，可能需要一些时间。结果会被缓存，后续调用会更快。

2. **内存**: 如果 parquet 文件包含图像字节数据，加载所有数据可能会占用大量内存。建议使用 `sample_size` 限制采样数量。

3. **路径格式**: 
   - 如果 parquet 文件中的路径是相对路径，确保它们相对于 `image_dir`（如果设置了）
   - 绝对路径会直接使用

4. **图像字节格式**: 支持原始字节（bytes）和 base64 编码字符串

5. **错误处理**: 如果某个 parquet 文件读取失败，会打印警告并继续处理其他文件

## 示例：完整工作流

```python
import asyncio
from ProbingFactorGeneration.core import ImageLoader, TemplateClaimGenerator, FailureAggregator, FilteringFactorMapper
from ProbingFactorGeneration.models import BaselineModel, JudgeModel
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline
from ProbingFactorGeneration.io import DataSaver

async def main():
    # 1. 创建 ImageLoader 并采样
    image_loader = ImageLoader(
        parquet_dir="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/",
        sample_size=100,  # 采样 100 张图像
        random_seed=42
    )
    
    # 2. 获取图像路径
    image_paths = image_loader.get_all_image_paths()
    print(f"Processing {len(image_paths)} images")
    
    # 3. 初始化其他组件
    template_generator = TemplateClaimGenerator(
        config_path="configs/claim_template.example_v1_1.json"
    )
    baseline_model = BaselineModel(
        model_path="/path/to/llava/model",
        gpu_id=0
    )
    judge_model = JudgeModel(
        model_name="qwen-model-name"
    )
    
    # 4. 创建 Pipeline
    pipeline = ProbingFactorPipeline(
        image_loader=image_loader,
        claim_generator=template_generator,
        baseline_model=baseline_model,
        judge_model=judge_model,
        failure_aggregator=FailureAggregator(),
        filtering_factor_mapper=FilteringFactorMapper(),
        data_saver=DataSaver(output_dir="./output")
    )
    
    # 5. 处理图像
    async with baseline_model, judge_model:
        results = await pipeline.process_batch_with_templates_async(image_paths)
        
        # 6. 保存结果
        output_path = pipeline.data_saver.save_results(
            results, 
            "probing_results", 
            "json"
        )
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
```
