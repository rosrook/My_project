# QA_Generator/answer

本目录包含答案生成与校验模块，提供命令行入口和核心逻辑。

## 直接使用方式

```bash
python QA_Generator/answer/main.py questions.json answers.json \
  --config QA_Generator/answer/answer_config.json \
  -n 100
```

- `questions.json`：问题生成阶段输出的 JSON（列表）
- `answers.json`：答案输出文件
- `--config`：可选，答案生成配置（默认已指向 `QA_Generator/answer/answer_config.json`）
- `-n/--max-samples`：可选，只处理前 N 条

## 在完整 QA 流程中的接口形式

在 `QA_Generator/pipeline/pipeline.py` 中，答案模块的调用形式如下：

- **输入**：问题生成模块产出的 `question` 数据列表（内存或中间 JSON）
- **调用**：`AnswerGenerator.generate_answer(...)` 或 `generate_answer_async(...)`
- **输出**：包含 `answer / explanation / full_question / options / correct_option` 等字段的结果

对接字段示例：
- 输入字段：`question`, `question_type`, `image_base64`, `pipeline_name`, `pipeline_intent`, `answer_type`
- 输出字段：`answer`, `explanation`, `full_question`, `options`, `correct_option`（并保留元数据）

答案生成后会交给 `AnswerValidator.validate_and_fix(...)` 做校验与修复。

## 文件说明

### `answer_generator.py`
核心答案生成逻辑：
- 按 `question_type`（选择题/填空题）生成答案
- 选择题会生成错误选项并拼成 `full_question`
- 提供同步/异步接口

常用接口：
- `generate_answer(...)`
- `generate_answer_async(...)`

### `validator.py`
答案校验与修复：
- 格式检查、选项重复、答案一致性验证
- 校验失败时尝试修复
- 输出 `validation_report`

常用接口：
- `validate_and_fix(...)`

### `main.py`
命令行入口：
- 读取问题 JSON → 生成答案 → 校验修复 → 写出结果
- 会输出 `answers.json`，并在需要时输出错误文件

### `answer_config.json`
答案生成配置：
- 选择题错误选项数量范围
- 生成温度、最大 token 等参数
