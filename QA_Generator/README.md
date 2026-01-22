## QA_Generator

本目录提供完整的 QA 生成流程（问题 + 答案）。以下只包含**参数解释**与**指令示例**。

---

### 1) 全流程（问题 + 答案）
入口：`QA_Generator/pipeline/pipeline.py`

#### 必填参数
- `input.json`：输入文件路径（JSON数组，元素需包含 `source_a` 与 `prefill`）。

#### 可选参数
- `--question-config`：问题配置文件路径（默认：`QA_Generator/question/config/question_config.json`）。
- `--answer-config`：答案配置文件路径（默认：`QA_Generator/answer/answer_config.json`）。
- `--pipelines`：仅运行指定 pipeline 列表（空则运行全部）。
- `-n / --max-samples`：限制处理前 N 条样本（调试用）。
- `--no-intermediate`：不保存中间问题/答案文件。
- `--batch-size`：每批处理大小（默认 1000）。
- `--concurrency`：异步并发数（建议 1-5）。
- `--request-delay`：请求间隔（秒）。
- `--no-async`：禁用异步，使用串行处理。
- `--log-file`：日志文件路径（可选）。
- `--debug-questions`：保存问题生成调试信息（包含输入/题型/slots/问题文本等）。
- `--debug-question-dir`：调试信息输出目录（默认：`output_dir/debug/questions`）。
- `--enable-validation-exemptions`：开启验证豁免（`question/visual_recognition/caption/text_association`）。

#### 输出目录
- 自动创建：`vqa_ready4use_YYYYMMDD_HHMMSS`（当前工作目录）
- 内含：成功/失败数据文件、中间文件（可选）、`meta.json`

#### 输出文件
- `vqa_dataset_successful_*.json`：最终可用 VQA 数据
- `question_errors_*.json`：问题生成失败/丢弃样本
- `answer_validation_failed_*.json`：答案生成/校验失败样本
- `meta.json`：本次生成元信息
- 若未加 `--no-intermediate`：
  - `intermediate/questions/batch_*_questions_*.json`
  - `intermediate/answers/batch_*_answers_*.json`
- 若加 `--debug-questions`：
  - `<debug_dir>/<output_stem>_question_debug_*.jsonl`

#### 指令示例（全流程）
- 基本用法：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json
  ```
- 指定 pipelines 与样本数：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json \
    --pipelines question object_counting \
    -n 100
  ```
- 关闭中间文件：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json --no-intermediate
  ```
- 使用串行模式：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json --no-async
  ```
- 加快并发（谨慎）：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json --concurrency 3 --request-delay 0.1
  ```
- 保存问题调试信息：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json \
    --debug-questions \
    --debug-question-dir /path/to/debug/questions
  ```
- 启用验证豁免：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json --enable-validation-exemptions
  ```
- 指定配置：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json \
    --question-config QA_Generator/question/config/question_config.json \
    --answer-config QA_Generator/answer/answer_config.json
  ```
- 指定日志文件：
  ```bash
  python QA_Generator/pipeline/pipeline.py input.json --log-file run.log
  ```

---

### 2) 仅问题生成（prefill）
入口：`QA_Generator/question/prefill/main.py`

#### 必填参数
- `input.json`：输入文件路径（JSON 数组，需包含 `prefill`）
- `output.json`：问题输出路径

#### 可选参数
- `--config`：问题配置文件路径（默认：`QA_Generator/question/config/question_config.json`）
- `--pipelines`：仅运行指定 pipeline 列表
- `-n / --max-samples`：限制处理前 N 条
- `--enable-validation-exemptions`：开启验证豁免（与全流程一致）

#### 指令示例（问题生成）
- 基本用法：
  ```bash
  python QA_Generator/question/prefill/main.py input.json output_questions.json
  ```
- 指定 pipelines 与样本数：
  ```bash
  python QA_Generator/question/prefill/main.py input.json output_questions.json \
    --pipelines question object_counting \
    -n 100
  ```
- 启用验证豁免：
  ```bash
  python QA_Generator/question/prefill/main.py input.json output_questions.json \
    --enable-validation-exemptions
  ```

---

### 3) 仅答案生成
入口：`QA_Generator/answer/main.py`

#### 必填参数
- `questions.json`：问题文件（数组）
- `answers.json`：答案输出路径

#### 可选参数
- `--config`：答案配置文件（默认：`QA_Generator/answer/answer_config.json`）
- `-n / --max-samples`：限制处理前 N 条

#### 指令示例（答案生成）
- 基本用法：
  ```bash
  python QA_Generator/answer/main.py questions.json answers.json
  ```
- 指定配置与样本数：
  ```bash
  python QA_Generator/answer/main.py questions.json answers.json \
    --config QA_Generator/answer/answer_config.json \
    -n 100
  ```

---

### 4) 输入数据格式（摘要）
每条记录至少包含：
- `source_a`：图像信息（需包含 base64）
  - 支持字段：`image_base64` / `jpg` / `img_base64` / `base64` 等
- `prefill`：
  - 方式 1（claim）：
    ```json
    {"prefill": {"claim": "..."}, "source_a": {"jpg": "<base64>"}}
    ```
  - 方式 2（target_object）：
    ```json
    {"prefill": {"target_object": "car"}, "source_a": {"jpg": "<base64>"}}
    ```
