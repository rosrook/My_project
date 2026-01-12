# 完整 Pipeline 流程说明

## 概述

本文档说明完整的 pipeline 处理流程，包括 claim 生成、验证、失败匹配和筛选要素提取。

## 完整流程

对于每张图片，pipeline 执行以下步骤：

### 1. 加载图像
- 使用 `ImageLoader` 加载图像文件
- 获取图像 ID

### 2. 生成 Claim Templates
- 使用 `TemplateClaimGenerator` 从 `claim_template.example_v1_1.json` 加载所有 claim schemas
- 每个 claim schema 包含：
  - `claim_id`: 唯一标识符
  - `claim_template`: 模板文本（包含占位符，如 `[OBJECT]`, `[REGION]`）
  - `target_failure_id`: 对应的失败 ID（用于后续匹配）
  - `metadata`: 包含 `baseline_instructions`, `not_related_conditions` 等

### 3. Baseline 模型完成 Templates
- 使用 `BaselineModel.complete_template_batch_async()` 批量完成所有 templates
- 对于每个 template，baseline 模型：
  - 根据图像内容填充占位符
  - 或返回 `NOT_RELATED`（如果图像不适用该 template）
  - 生成解释说明

每个 completion 包含：
- `completed_claim`: 完成的 claim 文本（或 "NOT_RELATED"）
- `is_related`: 是否相关（True/False）
- `explanation`: baseline 模型的解释
- `filled_values`: 填充的值字典

### 4. Judge 模型验证 Completions
- 使用 `JudgeModel.verify_completion_batch_async()` 批量验证所有 completions
- 对于每个 completion，judge 模型检查：
  - 完成的 claim 是否正确（基于图像内容）
  - 解释是否合理
  - 如果 baseline 标记为 `not_related`，判断这个标记是否正确

每个 verification 包含：
- `is_correct`: 整体是否正确（True/False）
- `claim_is_valid`: 完成的 claim 是否有效
- `explanation_is_reasonable`: 解释是否合理
- `not_related_judgment_correct`: not_related 判断是否正确（True/False/None）
- `failure_reason`: judge 提供的失败原因（可选）
- `judge_explanation`: judge 的解释

### 5. 匹配失败原因并提取筛选要素
- 对于每个失败的 claim（`is_correct=False` 或 `not_related_judgment_correct=False`）：
  - 使用 `FailureReasonMatcher` 从 `failure_config.example.json` 匹配最合适的失败原因
  - 匹配策略：
    1. **优先匹配**: 使用 claim template 的 `target_failure_id`（最精确）
    2. **备用匹配**: 如果没有 `target_failure_id`，使用 `failure_category`（基于 capability）
    3. **考虑 not_related**: 如果 baseline 错误地标记为 not_related，匹配相应的失败原因
  
- 从匹配的失败原因中提取 `suggested_filtering_factors`

每个 enhanced verification 包含：
- 原有字段（从 judge 验证结果）
- `failure_id`: 匹配的失败 ID（如 "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE"）
- `failure_category`: 失败类别（如 "visual_grounding"）
- `suggested_filtering_factors`: 建议的筛选要素列表

### 6. 聚合失败统计
- 使用 `FailureAggregator.aggregate()` 聚合失败统计
- 统计信息包括：
  - `total_claims`: 总 claim 数
  - `failed_claims`: 失败 claim 数
  - `success_rate`: 成功率
  - `failure_breakdown`: 按 failure_id 分组的失败计数
  - `failed_claim_ids`: 失败的 claim ID 列表

### 7. 收集筛选要素
- 收集所有失败 claim 的 `suggested_filtering_factors`
- 使用 `FilteringFactorMapper.map_batch()` 合并并去重
- 生成该图像的唯一筛选要素列表

### 8. 构建结果
最终结果包含：
- `image_id`: 图像 ID
- `claim_templates`: 原始 claim templates
- `completions`: Baseline 完成结果
- `verifications`: 增强的验证结果（包含 failure_id 和 suggested_filtering_factors）
- `aggregated_failures`: 聚合的失败统计
- `suggested_filtering_factors`: 该图像的所有筛选要素（来自所有失败的 claims）

## 失败判断逻辑

一个 claim 被认为是失败的，如果满足以下任一条件：

1. **Judge 判断为错误**: `is_correct = False`
   - Judge 模型认为完成的 claim 不正确

2. **错误地标记为 not_related**: `is_related = False` 且 `not_related_judgment_correct = False`
   - Baseline 模型标记为 not_related，但 Judge 认为应该可以回答
   - 这是 baseline 模型的一个失败

## 失败原因匹配逻辑

`FailureReasonMatcher.match_failure_for_claim()` 的匹配策略：

1. **通过 target_failure_id 匹配**（最精确）
   - 从 claim template 的 `metadata.target_failure_id` 直接匹配
   - 例如：`"target_failure_id": "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE"`

2. **通过 failure_category 匹配**（备用）
   - 如果找不到 `target_failure_id`，使用 `capability` 映射到 `failure_category`
   - 例如：`"capability": "visual_entity_recognition"` → `"failure_category": "visual_grounding"`

3. **处理 not_related 情况**
   - 如果 baseline 错误地标记为 not_related，优先使用 claim 的 `target_failure_id`
   - 如果找不到，尝试匹配 task_applicability 类别的失败

## 输出格式示例

```json
{
  "image_id": "image_001",
  "claim_templates": [
    {
      "claim_id": "entity_basic_grounding",
      "claim_template": "The image contains [ENTITY_TYPE] as a visually identifiable element.",
      "metadata": {
        "target_failure_id": "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE",
        "capability": "visual_entity_recognition",
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
      "explanation_is_reasonable": true,
      "not_related_judgment_correct": null,
      "failure_reason": "visual_error",
      "judge_explanation": "The object is actually a truck, not a car.",
      "failure_id": "FR_BASIC_VISUAL_ENTITY_GROUNDING_FAILURE",
      "failure_category": "visual_grounding",
      "suggested_filtering_factors": [
        "image contains at least one visually identifiable entity or structured visual unit",
        "entity can be coarsely described without requiring external context or interpretation",
        ...
      ]
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
    "failed_claim_ids": ["entity_basic_grounding", "entity_basic_grounding", "spatial_absolute_position"]
  },
  "suggested_filtering_factors": [
    "image contains at least one visually identifiable entity or structured visual unit",
    "entity can be coarsely described without requiring external context or interpretation",
    "entity boundaries or presence are perceptually distinguishable from the background",
    "At least one visually salient object suitable as a localization target",
    ...
  ]
}
```

## 使用示例

```python
from ProbingFactorGeneration.core import ImageLoader, TemplateClaimGenerator, FailureAggregator, FilteringFactorMapper
from ProbingFactorGeneration.models import BaselineModel, JudgeModel
from ProbingFactorGeneration.io import DataSaver
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline

# 初始化组件
image_loader = ImageLoader(image_dir="./data/images")
template_generator = TemplateClaimGenerator(
    config_path="configs/claim_template.example_v1_1.json"
)
baseline_model = BaselineModel(
    model_path="/path/to/llava/model",
    gpu_id=0,
    max_concurrent=1
)
judge_model = JudgeModel(
    model_name="/workspace/Qwen3-VL-235B-A22B-Instruct",
    use_lb_client=True
)
failure_aggregator = FailureAggregator()
filtering_factor_mapper = FilteringFactorMapper()
data_saver = DataSaver(output_dir="./output")

# 创建 pipeline（FailureReasonMatcher 会自动创建）
pipeline = ProbingFactorPipeline(
    image_loader=image_loader,
    claim_generator=template_generator,
    baseline_model=baseline_model,
    judge_model=judge_model,
    failure_aggregator=failure_aggregator,
    filtering_factor_mapper=filtering_factor_mapper,
    data_saver=data_saver
)

# 处理单张图像
async with baseline_model, judge_model:
    result = await pipeline.process_single_image_with_templates_async("image.jpg")
    
    # 访问结果
    print(f"Image ID: {result['image_id']}")
    print(f"Failed claims: {result['aggregated_failures']['failed_claims']}")
    print(f"Suggested filtering factors: {result['suggested_filtering_factors']}")
    
    # 保存结果
    pipeline.data_saver.save_results([result], "result", "json")
```

## 配置文件要求

### claim_template.example_v1_1.json
- 必须包含 `claim_schemas` 数组
- 每个 schema 应包含 `target_failure_id`（用于匹配失败原因）

### failure_config.example.json
- 必须包含 `failure_reasons` 数组
- 每个 failure_reason 应包含：
  - `failure_id`: 唯一标识符
  - `failure_category`: 失败类别
  - `suggested_filtering_factors`: 筛选要素列表

## 注意事项

1. **失败判断**: 确保正确理解 `is_correct` 和 `not_related_judgment_correct` 的含义
2. **匹配优先级**: `target_failure_id` 匹配优先于 `failure_category` 匹配
3. **筛选要素合并**: 所有失败 claim 的筛选要素会合并去重，作为图像的筛选要素
4. **性能**: 批量处理时，建议使用 `process_batch_with_templates_async()` 以获得更好的性能
