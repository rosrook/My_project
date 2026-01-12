# 运行完整 Pipeline 指南

## 概述

本文档说明如何运行完整的 Probing Factor Generation Pipeline，包括所需的环境配置、运行指令和预期输出。

## 前置条件

### 1. 安装依赖

```bash
cd ProbingFactorGeneration
pip install -r requirements.txt
```

必需的依赖包括：
- `pandas>=1.5.0`
- `pyarrow>=10.0.0`
- `Pillow>=10.0.0`
- `aiohttp>=3.8.0`
- `pyyaml>=6.0`

### 2. 配置文件

确保以下配置文件存在：
- `configs/claim_template.example_v1_1.json` - Claim template 配置
- `configs/failure_config.example.json` - 失败原因配置

### 3. 环境变量（Judge Model）

如果使用 Qwen 作为 judge 模型，需要设置环境变量：

```bash
export SERVICE_NAME="your_service_name"
export ENV="prod"  # 或 "staging"
export API_KEY="your_api_key"
export USE_LB_CLIENT="true"
```

## 运行方式

### 方式 1: 使用提供的脚本（推荐）

```bash
cd ProbingFactorGeneration
python examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --output_dir ./output \
    --baseline_model_path /path/to/llava/model \
    --judge_model_name /workspace/Qwen3-VL-235B-A22B-Instruct \
    --use_local_baseline \
    --random_seed 42
```

### 方式 2: 使用 API 模型（无需本地模型）

```bash
python examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 10 \
    --output_dir ./output \
    --judge_model_name /workspace/Qwen3-VL-235B-A22B-Instruct
```

### 方式 3: 最小配置（使用默认 API 模型）

```bash
python examples/run_complete_pipeline.py \
    --parquet_dir /mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/ \
    --sample_size 5 \
    --output_dir ./output
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--parquet_dir` | str | `/mnt/tidal-alsh01/.../train/` | Parquet 文件目录路径 |
| `--sample_size` | int | 10 | 采样图像数量 |
| `--output_dir` | str | `./output` | 输出目录 |
| `--baseline_model_path` | str | None | 本地 LLaVA 模型路径（如果使用本地模型） |
| `--judge_model_name` | str | None | Judge 模型名称（如 Qwen 模型） |
| `--claim_template_config` | str | `configs/claim_template.example_v1_1.json` | Claim template 配置文件路径 |
| `--use_local_baseline` | flag | False | 使用本地 baseline 模型（需要 `--baseline_model_path`） |
| `--random_seed` | int | 42 | 随机种子（用于可重复采样） |

## 运行流程

脚本会执行以下步骤：

1. **初始化 ImageLoader**
   - 从 parquet 目录加载所有 `.parquet` 文件
   - 根据 `sample_size` 采样图像路径
   - 打印加载的图像数量

2. **初始化 Claim Generator**
   - 加载 `claim_template.example_v1_1.json` 配置
   - 准备所有 claim templates

3. **初始化 Baseline Model**
   - 如果使用本地模型：加载 LLaVA 模型
   - 如果使用 API：配置 API 客户端

4. **初始化 Judge Model**
   - 配置 Qwen 模型（如果指定）或 API 模型

5. **初始化其他组件**
   - FailureAggregator
   - FilteringFactorMapper
   - DataSaver

6. **创建 Pipeline**
   - 将所有组件组装成完整的 pipeline

7. **处理图像**
   - 对每张图像：
     - 生成 claim templates
     - Baseline 模型完成 templates
     - Judge 模型验证 completions
     - 匹配失败原因并提取筛选要素
     - 聚合失败统计

8. **保存结果**
   - 将结果保存为 JSON 文件
   - 打印统计信息

## 输出结果

### 控制台输出

运行时会显示：

```
================================================================================
Probing Factor Generation Pipeline
================================================================================

Configuration:
  Parquet directory: /mnt/tidal-alsh01/.../train/
  Sample size: 10
  Output directory: ./output
  ...

Step 1: Initializing ImageLoader...
  ✓ Loaded 10 image paths

Step 2: Initializing TemplateClaimGenerator...
  ✓ Claim template loaded from configs/claim_template.example_v1_1.json

...

================================================================================
Processing Images...
================================================================================

✓ Processed 10 images

Step 8: Saving results...
  ✓ Results saved to: ./output/probing_results.json

==================================================
Template-Based Pipeline Statistics
==================================================

Total images processed: 10

Error rates by content type:
  relation: 15.00% (18/120)
  object: 10.00% (12/120)
  ...

'Not Related' Corrections:
  Judge corrected incorrect 'not related': 2/5 (40.00%)

Filtering factor distribution:
  image contains at least one visually identifiable entity: 15
  entity boundaries are perceptually distinguishable: 12
  ...
```

### 输出文件

结果保存在 `{output_dir}/probing_results.json`，格式如下：

```json
{
  "results": [
    {
      "image_id": "image_001",
      "claim_templates": [
        {
          "claim_id": "entity_basic_grounding",
          "claim_template": "The image contains [ENTITY_TYPE] as a visually identifiable element.",
          "metadata": {
            "target_failure_id": "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE",
            ...
          }
        },
        ...
      ],
      "completions": [
        {
          "completed_claim": "The image contains a car as a visually identifiable element.",
          "is_related": true,
          "explanation": "I can see a car in the image.",
          "filled_values": {"ENTITY_TYPE": "car"}
        },
        ...
      ],
      "verifications": [
        {
          "is_correct": false,
          "claim_is_valid": true,
          "failure_id": "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE",
          "failure_category": "visual_grounding",
          "suggested_filtering_factors": [
            "image contains at least one visually identifiable entity",
            "entity boundaries are perceptually distinguishable",
            ...
          ],
          "judge_explanation": "The object is actually a truck, not a car."
        },
        ...
      ],
      "aggregated_failures": {
        "image_id": "image_001",
        "total_claims": 12,
        "failed_claims": 3,
        "success_rate": 0.75,
        "failure_breakdown": {
          "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE": 2,
          "FR_ABSOLUTE_POSITION_ERROR": 1
        },
        "failed_claim_ids": ["entity_basic_grounding", ...]
      },
      "suggested_filtering_factors": [
        "image contains at least one visually identifiable entity",
        "entity boundaries are perceptually distinguishable",
        "At least one visually salient object suitable as a localization target",
        ...
      ]
    },
    ...
  ],
  "summary": {
    "total_images": 10,
    "total_claims": 120,
    "total_failures": 18,
    "overall_success_rate": 0.85,
    "error_rates_by_claim_type": {
      "relation": 0.15,
      "object": 0.10,
      ...
    },
    "filtering_factor_distribution": {
      "image contains at least one visually identifiable entity": 25,
      "entity boundaries are perceptually distinguishable": 20,
      ...
    }
  }
}
```

## 结果解读

### 单个图像结果

- `image_id`: 图像标识符
- `claim_templates`: 生成的 claim templates（从配置文件加载）
- `completions`: Baseline 模型完成的 claims
- `verifications`: Judge 模型的验证结果，包含：
  - `is_correct`: 是否正确
  - `failure_id`: 匹配的失败 ID
  - `suggested_filtering_factors`: 该失败对应的筛选要素
- `aggregated_failures`: 聚合的失败统计
- `suggested_filtering_factors`: **该图像的所有筛选要素**（来自所有失败的 claims）

### 总结统计

- `total_images`: 处理的图像总数
- `total_claims`: 总 claim 数
- `total_failures`: 总失败数
- `overall_success_rate`: 整体成功率
- `error_rates_by_claim_type`: 按内容类型的错误率
- `filtering_factor_distribution`: 筛选要素的分布统计

## 故障排除

### 1. Parquet 文件未找到

```
FileNotFoundError: Parquet directory not found: ...
```

**解决方案**: 检查 `--parquet_dir` 路径是否正确，确保目录存在且包含 `.parquet` 文件。

### 2. 配置文件未找到

```
FileNotFoundError: Claim template config not found: ...
```

**解决方案**: 确保配置文件存在于指定路径，或使用 `--claim_template_config` 指定正确路径。

### 3. 模型加载失败

**本地模型**:
- 检查模型路径是否正确
- 确保有足够的 GPU 内存
- 检查模型格式是否正确（Hugging Face 格式）

**API 模型**:
- 检查环境变量设置
- 检查 API 密钥是否有效
- 检查网络连接

### 4. 内存不足

如果处理大量图像时内存不足：
- 减小 `--sample_size`
- 减小 `max_concurrent` 参数
- 使用批处理方式

### 5. 依赖缺失

```
ImportError: No module named 'pandas'
```

**解决方案**: 安装缺失的依赖：
```bash
pip install -r requirements.txt
```

## 性能优化建议

1. **采样大小**: 从小样本开始（如 10-50 张），验证流程正常后再增加
2. **并发控制**: 根据 API 限制和硬件资源调整 `max_concurrent`
3. **本地模型**: 如果使用本地 LLaVA 模型，确保 GPU 内存充足
4. **批处理**: 对于大量图像，考虑分批处理

## 下一步

运行成功后，你可以：
1. 查看输出 JSON 文件，分析失败模式和筛选要素
2. 根据结果调整 claim templates 或失败原因配置
3. 增加采样大小进行更大规模的评估
4. 使用筛选要素进行数据构建
