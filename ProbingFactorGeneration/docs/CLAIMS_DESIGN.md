# Claims 设计文档

## 1. Claims 的概念与作用

### 什么是 Claims？

Claims（声明/断言）是 VQA（视觉问答）数据构造中的核心元素。它们是针对图像提出的结构化断言，用于：
- **探测模型能力**：测试基线模型对不同类型视觉-语言任务的理解能力
- **识别失败模式**：通过验证 claims 的正确性，发现模型在哪些方面失败
- **生成筛选要素**：将失败模式映射到可复用的数据筛选要素，用于后续 QA 数据构造

### Claims 在 Pipeline 中的位置

```
图像 → Claims → Baseline预测 → Judge验证 → 失败聚合 → 筛选要素映射 → 输出
```

## 2. Claim 的数据结构

### 标准 Claim 格式

每个 Claim 是一个字典，包含以下字段：

```python
{
    "claim_id": str,           # 唯一标识符，格式: "{image_id}_claim_{index}"
    "claim_text": str,         # Claim 的文本内容（必需）
    "content_type": str,       # 内容类型，见 ContentType 枚举
    "metadata": dict           # 可选的元数据信息
}
```

### 字段详细说明

#### 1. `claim_id` (str)
- **用途**：唯一标识每个 claim
- **格式**：`"{image_id}_claim_{index}"` 或自定义
- **示例**：`"image_001_claim_0"`, `"global_claim_1"`

#### 2. `claim_text` (str)
- **用途**：Claim 的核心内容，描述对图像的断言
- **要求**：必需字段，不能为空
- **示例**：
  - `"Does this image show a person?"`
  - `"The image contains text."`
  - `"How many objects are in the image?"`

#### 3. `content_type` (str)
- **用途**：标识 claim 测试的内容类型
- **可选值**：见 `ContentType` 枚举（7种类型）
- **默认行为**：如果不指定，会根据 `claim_text` 自动推断

#### 4. `metadata` (dict)
- **用途**：存储额外的元数据信息
- **常见字段**：
  - `difficulty`: 难度级别（"easy", "medium", "hard"）
  - `category`: 分类标签
  - `source`: 来源（"predefined_config", "generated"）
  - 其他自定义字段

### 完整示例

```python
{
    "claim_id": "image_001_claim_0",
    "claim_text": "Does this image show a person?",
    "content_type": "object",
    "metadata": {
        "difficulty": "easy",
        "category": "person_detection",
        "source": "predefined_config",
        "focus": "human_figure"
    }
}
```

## 3. ContentType 类型系统

### 支持的 ContentType

在 `config.py` 中定义了 7 种内容类型：

| ContentType | 值 | 说明 | 示例 Claims |
|------------|-----|------|------------|
| `OBJECT` | `"object"` | 对象检测 | "Does this image contain a car?", "There is a person in the image." |
| `ATTRIBUTE` | `"attribute"` | 属性识别 | "What color is the car?", "The object is round." |
| `RELATION` | `"relation"` | 关系理解 | "The person is holding an umbrella.", "The dog is next to the cat." |
| `ACTION` | `"action"` | 动作识别 | "Is the person running?", "The dog is playing." |
| `SPATIAL` | `"spatial"` | 空间关系 | "Is the car on the left?", "The tree is behind the house." |
| `COUNT` | `"count"` | 计数任务 | "How many people are in the image?", "Count the number of cars." |
| `TEXT` | `"text"` | 文本识别 | "Does this image contain text?", "What does the sign say?" |

### ContentType 自动推断

如果未指定 `content_type`，`PredefinedClaimGenerator` 会根据 `claim_text` 中的关键词自动推断：

```python
# 推断规则
"how many", "count" → COUNT
"left", "right", "above", "below" → SPATIAL
"doing", "action", "is" → ACTION
"color", "size", "shape" → ATTRIBUTE
"contains", "has", "there is" → OBJECT
"text", "says", "reads" → TEXT
其他 → RELATION (默认)
```

## 4. Claims 的生成方式

### 方式一：ClaimGenerator（自动生成）

适用于自动生成 claims，需要实现具体的生成逻辑。

```python
from ProbingFactorGeneration import ClaimGenerator

generator = ClaimGenerator(content_types=[ContentType.OBJECT, ContentType.COUNT])
claims = generator.generate(image, image_id="img_001")
```

### 方式二：PredefinedClaimGenerator（预定义）

从 JSON 配置文件加载人为设计的 claims。

```python
from ProbingFactorGeneration import PredefinedClaimGenerator

generator = PredefinedClaimGenerator(config_path="claim_config.json")
claims = generator.generate(image, image_id="img_001")
```

## 5. claim_config.json 配置文件格式

### 配置文件结构

```json
{
  "metadata": {
    "description": "配置文件描述",
    "version": "1.0",
    "created": "2024-01-01"
  },
  "global_claims": [...],      // 应用到所有图像的全局 claims
  "claims_by_image": {...}     // 针对特定图像 ID 的 claims
}
```

### global_claims（全局 Claims）

应用到所有图像的 claims。优先级低于图像特定的 claims。

```json
"global_claims": [
  // 方式1: 完整字典格式
  {
    "claim_id": "global_claim_1",
    "claim_text": "Does this image contain any text?",
    "content_type": "text",
    "metadata": {
      "difficulty": "easy",
      "category": "text_detection"
    }
  },
  // 方式2: 简化格式（自动生成 claim_id 和 content_type）
  {
    "claim_text": "Is the image clear?",
    "content_type": "attribute"
  },
  // 方式3: 字符串格式（最简单的写法）
  "This is a simple string claim"
]
```

### claims_by_image（图像特定 Claims）

针对特定图像 ID 的 claims，优先级高于全局 claims。

```json
"claims_by_image": {
  "image_001": [
    {
      "claim_id": "img001_claim_1",
      "claim_text": "Does this image show a person?",
      "content_type": "object",
      "metadata": {
        "difficulty": "easy"
      }
    },
    {
      "claim_text": "How many objects are in the image?",
      "content_type": "count"
    }
  ],
  "image_002": [
    "What is the dominant color in this image?",
    {
      "claim_text": "Is there any text visible?",
      "content_type": "text"
    }
  ]
}
```

### Claims 合并规则

对于给定图像，claims 的加载顺序：
1. **图像特定 claims** (`claims_by_image[image_id]`)
2. **全局 claims** (`global_claims`)

所有 claims 都会被合并到一个列表中。

## 6. Claims 在 Pipeline 中的使用流程

### 完整数据流

```python
# 步骤1: 生成 Claims
claims = claim_generator.generate(image, image_id="img_001")
# 输出: [
#   {"claim_id": "...", "claim_text": "...", "content_type": "...", ...},
#   ...
# ]

# 步骤2: Baseline 模型预测
for claim in claims:
    baseline_answer = baseline_model.predict(image, claim)
    # 输出: {
    #   "prediction": True/False/str,
    #   "confidence": float,
    #   "metadata": {...}
    # }

# 步骤3: Judge 模型验证
for claim, baseline_answer in zip(claims, baseline_answers):
    verification = judge_model.verify(image, claim, baseline_answer)
    # 输出: {
    #   "is_correct": bool,
    #   "failure_reason": str or None,
    #   "confidence": float,
    #   "metadata": {...}
    # }

# 步骤4: 聚合失败
aggregated = failure_aggregator.aggregate(image_id, claims, verifications)

# 步骤5: 映射到筛选要素
filtering_factors = filtering_factor_mapper.map_batch(failure_reasons)
```

### Claim 在预测中的使用

BaselineModel 使用 claim 的 `claim_text` 字段构建 API 请求：

```python
# 在 BaselineModel._build_messages() 中
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": claim["claim_text"]},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
        ]
    }
]
```

## 7. 使用示例

### 示例 1: 基本使用

```python
from ProbingFactorGeneration import PredefinedClaimGenerator
from PIL import Image

# 创建生成器
generator = PredefinedClaimGenerator(config_path="claim_config.json")

# 生成 claims
image = Image.open("image.jpg")
claims = generator.generate(image, image_id="image_001")

# 查看结果
for claim in claims:
    print(f"Claim ID: {claim['claim_id']}")
    print(f"Text: {claim['claim_text']}")
    print(f"Type: {claim['content_type']}")
    print(f"Metadata: {claim.get('metadata', {})}")
    print()
```

### 示例 2: 在 Pipeline 中使用

```python
from ProbingFactorGeneration import (
    ImageLoader, PredefinedClaimGenerator,
    BaselineModel, JudgeModel,
    FailureAggregator, FilteringFactorMapper, DataSaver
)
from ProbingFactorGeneration.pipeline import ProbingFactorPipeline

# 创建组件
image_loader = ImageLoader(image_dir="./data/images")
claim_generator = PredefinedClaimGenerator(config_path="claim_config.json")
baseline_model = BaselineModel(model_name="gemini-pro-vision")
judge_model = JudgeModel()
failure_aggregator = FailureAggregator()
filtering_factor_mapper = FilteringFactorMapper()
data_saver = DataSaver(output_dir="./output")

# 创建 Pipeline
pipeline = ProbingFactorPipeline(
    image_loader=image_loader,
    claim_generator=claim_generator,
    baseline_model=baseline_model,
    judge_model=judge_model,
    failure_aggregator=failure_aggregator,
    filtering_factor_mapper=filtering_factor_mapper,
    data_saver=data_saver
)

# 处理图像
result = pipeline.process_single_image("image_001.jpg")
# result["claims"] 包含该图像的所有 claims
```

### 示例 3: 批量处理

```python
# 批量生成 claims
images = [Image.open(f"img_{i}.jpg") for i in range(10)]
image_ids = [f"image_{i:03d}" for i in range(10)]

claims_dict = generator.generate_batch(images, image_ids)
# 输出: {
#   "image_000": [claim1, claim2, ...],
#   "image_001": [claim3, claim4, ...],
#   ...
# }
```

## 8. 最佳实践

### 设计 Claims 的建议

1. **明确性**：使用清晰、具体的语言
   - ✅ 好: "Does this image show a red car?"
   - ❌ 差: "Car?"

2. **多样性**：覆盖不同的 ContentType
   - 不要只使用一种类型的 claims
   - 混合使用 OBJECT, ATTRIBUTE, RELATION 等

3. **难度梯度**：设计不同难度的 claims
   - Easy: 简单的对象检测
   - Medium: 属性或关系理解
   - Hard: 复杂的推理任务

4. **具体性**：针对图像内容的特定 claims
   - 使用 `claims_by_image` 为特定图像设计针对性 claims

### 配置文件组织

```json
{
  "metadata": {
    "description": "VQA probing claims for dataset X",
    "version": "1.0"
  },
  "global_claims": [
    // 通用的基础 claims（所有图像都测试）
    {
      "claim_text": "Is the image clear and well-lit?",
      "content_type": "attribute",
      "metadata": {"category": "quality_check"}
    }
  ],
  "claims_by_image": {
    // 针对每个图像的特定 claims
    "img_001": [
      // 对象检测
      {"claim_text": "Does this show a person?", "content_type": "object"},
      // 计数任务
      {"claim_text": "How many people are visible?", "content_type": "count"},
      // 空间关系
      {"claim_text": "Is there a car on the left side?", "content_type": "spatial"}
    ]
  }
}
```

## 9. 常见问题

### Q1: claim_id 会自动生成吗？
是的，如果不指定 `claim_id`，会自动生成格式为 `"{image_id}_claim_{index}"` 的 ID。

### Q2: content_type 必须指定吗？
不是必需的。如果不指定，会根据 `claim_text` 中的关键词自动推断。但建议显式指定以获得更准确的结果。

### Q3: 字符串格式和字典格式有什么区别？
- **字符串格式**：简单方便，适合快速创建 claims
- **字典格式**：可以指定更多细节（claim_id, metadata 等），适合复杂场景

### Q4: global_claims 和 claims_by_image 可以同时使用吗？
可以。对于给定图像，会先加载图像特定 claims，然后添加全局 claims。

### Q5: 如何验证 claim_config.json 的格式？
如果格式不正确，`PredefinedClaimGenerator` 会在初始化时抛出 `ValueError` 或 `FileNotFoundError`，并提供详细的错误信息。

## 10. 扩展与自定义

### 扩展 ContentType

在 `config.py` 中添加新的 ContentType：

```python
class ContentType(Enum):
    # ... 现有类型
    TEMPORAL = "temporal"  # 新增：时间相关
    EMOTION = "emotion"    # 新增：情感识别
```

### 自定义 Claim 处理逻辑

继承 `PredefinedClaimGenerator` 并重写方法：

```python
class CustomClaimGenerator(PredefinedClaimGenerator):
    def _infer_content_type(self, claim_text: str) -> str:
        # 自定义推断逻辑
        ...
```
