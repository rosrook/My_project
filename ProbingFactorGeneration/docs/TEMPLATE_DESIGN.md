# Template-Based Claim Generation Design

## 概述

本文档描述了新的基于模板的 claim 生成设计。该设计允许用户预定义包含占位符的 claim 模板，然后由 baseline 模型根据图像内容自动补全这些模板。

## 设计思路

### 核心思想

1. **预定义模板框架**：用户定义包含可替换元素的 claim 模板（如 `"Does this image show a {object}?"`）
2. **Baseline 模型补全**：模型根据图像内容填充占位符，形成具体断言，并提供解释
3. **"Not Related" 机制**：如果模板与图像不相关，模型可以返回 "not related"
4. **Judge 验证与纠正**：Judge 模型评估生成的内容和解释的正确性，对于标记为 "not related" 的情况，如果 Judge 认为其实可以构造回答，则判定为错误

### 设计合理性

✅ **模板 + 模型补全**：结合人工设计结构的可控性和模型的灵活性  
✅ **"Not Related" 机制**：避免强制生成不相关的断言  
✅ **Judge 纠正机制**：提高质量，避免 baseline 模型错误拒绝有效模板  
✅ **解释机制**：有助于理解失败原因，提高可解释性  

## 工作流程

```
1. 加载图像
   ↓
2. 生成 claim 模板（TemplateClaimGenerator）
   - 从 JSON 配置文件加载全局模板
   - 所有模板应用到所有图像
   - 模板包含占位符（如 {object}, {attribute}）
   ↓
3. Baseline 模型补全模板（BaselineModel.complete_template_async）
   - 根据图像内容填充占位符
   - 生成完整断言和解释
   - 如果不相关，返回 "not related"
   ↓
4. Judge 模型验证（JudgeModel.verify_completion_async）
   - 验证生成的断言是否正确
   - 验证解释是否合理
   - 检查 "not related" 判断是否正确
   - 如果 baseline 错误标记为 "not related"，判定为失败
   ↓
5. 聚合失败并映射到过滤因子
   ↓
6. 保存结果
```

## 模块说明

### 1. TemplateClaimGenerator

**位置**: `core/generators/template_claim_generator.py`

**功能**: 从 JSON 配置文件加载全局 claim 模板（应用到所有图像）

**关键方法**:
- `generate(image, image_id)`: 生成模板列表（所有图像使用相同的全局模板）
- `generate_batch(images, image_ids)`: 批量生成
- `get_global_templates_count()`: 获取全局模板数量

**模板格式**:
```json
{
  "global_templates": [
    "Does this image contain any {text_type}?",
    {
      "claim_template": "How many {object_type} are in the image?",
      "content_type": "count",
      "placeholders": ["object_type"]
    },
    {
      "claim_template": "Does this image show a {person_type}?",
      "content_type": "object",
      "placeholders": ["person_type"]
    }
  ]
}
```

**注意**: 所有模板都是全局的，会应用到所有图像上。

### 2. BaselineModel (更新)

**位置**: `models/baseline.py`

**新增方法**:
- `complete_template_async(image, claim_template)`: 补全模板（异步）
- `complete_template(image, claim_template)`: 补全模板（同步包装）
- `complete_template_batch_async(images, claim_templates)`: 批量补全（异步）
- `complete_template_batch(images, claim_templates)`: 批量补全（同步包装）

**返回格式**:
```python
{
    "completed_claim": str,  # 补全后的断言或 "not related"
    "is_related": bool,  # 模板是否与图像相关
    "explanation": str,  # 补全的解释
    "filled_values": Dict[str, str],  # 填充的占位符值
    "claim_id": str,
    "content_type": str,
    "metadata": dict
}
```

**Prompt 设计**:
- 指导模型填充占位符
- 如果不相关，返回 "not related"
- 要求提供解释

### 3. JudgeModel (更新)

**位置**: `models/judge.py`

**新增方法**:
- `verify_completion_async(image, claim_template, completion)`: 验证补全结果（异步）
- `verify_completion(image, claim_template, completion)`: 验证补全结果（同步包装）
- `verify_completion_batch_async(images, claim_templates, completions)`: 批量验证（异步）
- `verify_completion_batch(images, claim_templates, completions)`: 批量验证（同步包装）

**返回格式**:
```python
{
    "is_correct": bool,  # 整体是否正确
    "claim_is_valid": bool,  # 补全的断言是否有效
    "explanation_is_reasonable": bool,  # 解释是否合理
    "not_related_judgment_correct": bool or None,  # "not related" 判断是否正确
    "failure_reason": str or None,  # 失败原因（来自分类法）
    "judge_explanation": str,  # Judge 的解释
    "suggested_correction": str or None,  # 建议的纠正
    "claim_id": str,
    "content_type": str,
    "metadata": dict
}
```

**关键验证逻辑**:
- 验证补全的断言是否正确
- 验证解释是否合理
- **重要**: 如果 baseline 标记为 "not related" 但 Judge 认为可以回答，判定为错误

### 4. Pipeline (更新)

**位置**: `pipeline.py`

**新增方法**:
- `process_single_image_with_templates_async(image_path)`: 处理单张图像（模板流程，异步）
- `process_single_image_with_templates(image_path)`: 处理单张图像（模板流程，同步包装）
- `process_batch_with_templates_async(image_paths)`: 批量处理（模板流程，异步）
- `process_batch_with_templates(image_paths)`: 批量处理（模板流程，同步包装）
- `run_and_save_with_templates_async(image_paths, output_filename, output_format)`: 运行并保存（模板流程，异步）
- `run_and_save_with_templates(image_paths, output_filename, output_format)`: 运行并保存（模板流程，同步包装）

**统计信息**:
- 按内容类型统计错误率
- 统计 "not related" 纠正次数
- 过滤因子分布

## 使用示例

### 基本使用

```python
import asyncio
from ProbingFactorGeneration import (
    ImageLoader, TemplateClaimGenerator,
    BaselineModel, JudgeModel,
    ProbingFactorPipeline
)

# 初始化组件
image_loader = ImageLoader()
template_generator = TemplateClaimGenerator(config_path="claim_template.json")
baseline_model = BaselineModel(model_name="gemini-pro-vision")
judge_model = JudgeModel(model_name="gemini-pro-vision")

# 创建 pipeline
pipeline = ProbingFactorPipeline(
    image_loader=image_loader,
    claim_generator=template_generator,
    baseline_model=baseline_model,
    judge_model=judge_model,
    # ... 其他组件
)

# 处理图像（异步）
async with baseline_model, judge_model:
    result = await pipeline.process_single_image_with_templates_async("image.jpg")
    
    # 查看结果
    print(f"Templates: {len(result['claim_templates'])}")
    print(f"Completions: {len(result['completions'])}")
    print(f"Verifications: {len(result['verifications'])}")
```

### 直接 API 使用

```python
import asyncio
from ProbingFactorGeneration import TemplateClaimGenerator, BaselineModel, JudgeModel

template_generator = TemplateClaimGenerator(config_path="claim_template.json")
baseline_model = BaselineModel()
judge_model = JudgeModel()

image = Image.open("image.jpg")
templates = template_generator.generate(image, image_id="img_001")

async with baseline_model, judge_model:
    # 补全模板
    completions = await baseline_model.complete_template_batch_async(
        [image] * len(templates),
        templates
    )
    
    # 验证补全
    verifications = await judge_model.verify_completion_batch_async(
        [image] * len(templates),
        templates,
        completions
    )
    
    # 检查结果
    for comp, verif in zip(completions, verifications):
        print(f"Template: {comp['metadata']['original_template']}")
        print(f"Completed: {comp['completed_claim']}")
        print(f"Related: {comp['is_related']}")
        print(f"Correct: {verif['is_correct']}")
        if verif.get('not_related_judgment_correct') is False:
            print("⚠️  Judge corrected incorrect 'not related' judgment!")
```

## 配置文件格式

参考 `configs/claim_template.example.json` 了解完整的配置文件格式。

## 优势

1. **可控性**: 用户可以通过模板控制生成的结构和类型
2. **灵活性**: 模型根据图像内容填充占位符，保持灵活性
3. **质量保证**: Judge 模型验证并纠正错误的 "not related" 判断
4. **可解释性**: 每个补全都包含解释，便于理解失败原因
5. **向后兼容**: 保留原有的完整 claim 流程，支持两种模式

## 注意事项

1. **模板设计**: 模板应该清晰明确，占位符名称应该有语义（如 `{object_type}` 而不是 `{x}`）
2. **占位符格式**: 支持 `{placeholder}` 或 `[placeholder]` 格式
3. **"Not Related" 处理**: Judge 模型会严格检查，避免 baseline 错误拒绝有效模板
4. **异步使用**: 推荐使用异步方法以获得更好的性能
5. **资源管理**: 使用 `async with` 确保正确关闭资源

## 后续改进方向

1. **模板验证**: 添加模板语法验证和占位符检查
2. **批量优化**: 进一步优化批量处理的并发控制
3. **错误处理**: 增强错误处理和重试机制
4. **统计分析**: 添加更详细的统计信息（如按模板类型统计）
5. **模板推荐**: 基于图像内容自动推荐相关模板
