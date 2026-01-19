# QA_Generator/question/core

本目录包含问题生成流程中的核心基础模块（配置加载、槽位填充、问题验证），
被 `question/prefill` 主流程调用。

## 直接使用方式

这些模块通常由 `VQAGeneratorPrefill` 调用，不建议直接命令行运行。
如果要在代码中直接使用，可以按如下方式：

```python
from QA_Generator.question.core.config_loader import ConfigLoader
from QA_Generator.question.core.slot_filler import SlotFiller
from QA_Generator.question.core.validator import QuestionValidator

loader = ConfigLoader("QA_Generator/question/config/question_config.json")
pipeline_cfg = loader.get_pipeline_config("object_position")
global_constraints = loader.get_global_constraints()

filler = SlotFiller()
slots = filler.fill_slots(image_input=..., pipeline_config=pipeline_cfg, selected_object=...)

validator = QuestionValidator()
ok, reason = validator.validate(
    question="Where is the object located?",
    image_input=...,
    pipeline_config=pipeline_cfg,
    global_constraints=global_constraints
)
```

## 在完整 QA 流程中的接口形式

`QA_Generator/question/prefill/vqa_generator.py` 中会：
- 通过 `ConfigLoader` 读取配置
- 通过 `SlotFiller` 填充必需/可选槽位
- 通过 `QuestionValidator` 做规则+LLM 校验

这些模块提供“规则层”和“约束层”，保证问题质量与一致性。

## 文件说明

### `config_loader.py`
**配置加载器**：
- 读取 `question_config.json`
- 获取 pipeline 配置 / 全局约束 / 生成策略 / 题型比例等

### `slot_filler.py`
**槽位填充器**：
- 填充 `required_slots` 和 `optional_slots`
- 支持随机填充可选槽位增加多样性
- 支持从 `selected_object` 或图像规则中解析

### `validator.py`
**问题验证器**：
- 基础规则检查（空问题、全局限制等）
- Pipeline 级约束检查
- 通过 LLM 进行深度验证（同步/异步）
