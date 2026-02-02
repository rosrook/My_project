# QA_Generator：问题与答案的生成步骤与校验逻辑

本文档概括 QA_Generator 中 **Question（问题）** 与 **Answer（答案）** 的生成流程及各自校验逻辑，便于向导师或协作方说明。

---

## 一、整体流程

全流程入口：`QA_Generator/pipeline/pipeline.py`。

1. **输入**：JSON 数组，每条含 `source_a`（图像，需 base64）与 `prefill`（预填充：`claim` 或 `target_object`）。
2. **Step 1**：按批次调用 **问题生成**（`VQAGeneratorPrefill`），得到带 `question`、`question_type`、`image_base64` 等的问题列表。
3. **Step 2**：对每条问题调用 **答案生成**（`AnswerGenerator`），再经 **答案校验与修复**（`AnswerValidator.validate_and_fix`），通过则写入成功集，未通过则写入答案校验失败集。
4. **输出**：成功 VQA 数据、问题生成错误、答案校验失败，以及可选中间结果与 meta。

---

## 二、问题（Question）生成步骤

问题生成由 `question/prefill/vqa_generator.py` 的 `VQAGeneratorPrefill` 完成，对「单张图 + 单个 pipeline + 预填充」严格按 **6 步** 执行（当前实现中第 5 步验证被跳过，直接到第 6 步输出）。

### 2.1 六步流程

| 步骤 | 说明 | 模块/逻辑 |
|------|------|-----------|
| **STEP 1** | 加载 Pipeline 规范 | `ConfigLoader.get_pipeline_config(pipeline_name)`，从 `question_config.json` 读取该 pipeline 的 intent、required_slots、optional_slots、example_template、question_constraints、answer_type 等。 |
| **STEP 2** | 处理预填充对象 | `PrefillProcessor.process_prefill(prefill_input, image_input, pipeline_config)`。支持两种输入：**claim**（一句基于该图的 claim，可从中解析目标对象）或 **target_object**（直接给目标对象名）。输出统一结构：`{ name, category?, source: "claim"|"target_object", claim?, confidence }`。若缺失 claim/name 则报错并丢弃。 |
| **STEP 3** | 槽位填充 | `SlotFiller.fill_slots(image_input, pipeline_config, selected_object=prefill_object)`。按 pipeline 的 `required_slots` 与 `optional_slots` 解析槽位：必需槽位从预填充对象或图像中解析，缺一则丢弃；可选槽位按策略（如 50% 概率）填充以增加多样性。 |
| **STEP 4** | 问题生成 | `QuestionGeneratorPrefill.generate_question(image_input, pipeline_config, slots, prefill_object, question_type)`。根据 intent、description、example_template、question_constraints 与 slots、prefill_object 构建 prompt，调用 LLM 生成题干；题型由配置或策略在 `multiple_choice` / `fill_in_blank` 中选择。 |
| **STEP 5** | 验证（当前跳过） | 设计上可由 `QuestionValidator.validate` 做约束与 LLM 深度验证；当前实现为**跳过**该步，直接进入输出。 |
| **STEP 6** | 输出 | 组装结果：`pipeline_name`、`pipeline_intent`、`question`、`question_type`、`answer_type`、`slots`、`prefill_object` 等，写入中间问题文件或进入答案阶段。 |

### 2.2 问题校验逻辑（当启用时）

`question/core/validator.py` 中的 `QuestionValidator` 在 **启用验证** 时负责：

- **基础检查**：问题非空。
- **全局约束** `_check_global_constraints`：禁止某些问题类型（如泛化场景描述、主观偏好、假设性、仅常识、无法从图回答等），通过关键词匹配。
- **Pipeline 约束** `_check_pipeline_constraints`：按 pipeline 的 `question_constraints` 检查（当前实现较简，可扩展）。
- **豁免**：若开启 `enable_validation_exemptions`，对 `question`、`visual_recognition`、`caption`、`text_association` 等 pipeline **跳过 LLM 深度验证**，直接通过。
- **LLM 深度验证** `_validate_with_llm`：用 LLM 结合图像与约束判断问题是否可答、是否合规（未豁免时使用）。

当前全流程默认**不调用**问题验证，以提速；仅问题生成阶段内部可能按配置或后续扩展使用。

---

## 三、答案（Answer）生成步骤

答案生成由 `answer/answer_generator.py` 的 `AnswerGenerator` 完成，按 **题型** 分支：`multiple_choice`（选择题）或 `fill_in_blank`（填空题）。

### 3.1 选择题（multiple_choice）

1. **生成正确答案**：`_generate_correct_answer(question, image_base64, pipeline_info)`，调用 LLM 根据问题与图像得到正确答案文本。
2. **生成错误选项**：按配置（如 2–4 个错误选项）生成干扰项，保证与正确答案可区分。
3. **打乱顺序**：将正确选项与错误选项合并后打乱，得到选项字典（如 A/B/C/D）。
4. **确定正确选项字母**：记录 `correct_option`（如 `"A"`），并组装 `full_question`（题干 + 选项文本）。
5. **返回**：`answer`（正确选项字母）、`full_question`、`options`、`correct_option`、`explanation` 等。

### 3.2 填空题（fill_in_blank）

1. **生成答案**：根据问题与图像调用 LLM 生成填空答案与解释。
2. **返回**：`answer`（填空内容）、`explanation`、`full_question` 等。

支持同步 `generate_answer` 与异步 `generate_answer_async`（与 pipeline 的并发调用对接）。

---

## 四、答案校验逻辑

答案校验由 `answer/validator.py` 的 `AnswerValidator.validate_and_fix` 完成，分为 **格式检查与修复** 和 **VQA 验证** 两步；未通过时可触发**重试生成**或**一次修复再验证**。

### 4.1 Step 1：格式检查与修复（`_format_check_and_fix`）

- **答案格式**：若 `answer` 为 JSON 字符串（如 `{"Answer":"A","Explanation":"..."}`），则解析并提取纯文本 `answer` 与 `explanation`。
- **占位符检查** `_check_placeholders`：检查问题、完整题干、答案、选项是否仍含未填充占位符（如 `[object]`、`{xxx}`、`<xxx>`、`___` 等），若有则记入 issues，并标记未通过。
- **选择题专项**：  
  - **选项重复** `_check_option_duplicates`：选项值是否重复、选项数是否至少 2；若重复则尝试 `_fix_option_duplicates` 修复。  
  - **答案完整性** `_check_answer_completeness`：`answer` 是否存在、是否对应到某选项键、`correct_option` 与 `answer` 是否一致、`correct_option` 是否在 `options` 中；若有问题则尝试 `_fix_answer_completeness` 修复。  
  - **验证修复结果** `_verify_fixes`：对修复后的结果再查一遍，确保无新问题。
- **填空题**：检查答案非空。
- 若格式问题无法自动修复，则 `validation_passed=False`，并可能设置 `should_regenerate=True`，由 pipeline 决定是否重新生成答案。

### 4.2 Step 2：VQA 验证（`_vqa_validation`）

- **困惑度分析**：当前**已禁用**，不再作为通过条件。
- **置信度评估** `_assess_confidence`：用 LLM 评估「模型对答案的置信度」，严格标准为 confidence ≥ 0.7。
- **答案验证** `_validate_answer`：用 LLM 结合图像与题干/选项，判断答案是否与图像内容一致、是否合理。
- **通过条件**：严格标准下，置信度通过且答案验证通过，则 VQA 通过。
- **抢救机制**：若严格未通过，则尝试放宽标准（如置信度在 0.5–0.7 且答案验证有效则视为抢救通过），并记录 `rescue_attempted`、`rescue_successful`。

### 4.3 修复与重试（Pipeline 侧）

- **AnswerRepairer**：当 VQA 未通过时，`AnswerValidator` 可调用 `AnswerRepairer.repair_once`，对当前答案做一次「修复式重判」（如让 LLM 修正答案或解释），再对修复结果重新执行 VQA 验证；若通过则整体视为通过。
- **重试生成**：Pipeline 中若 `validation_report.should_regenerate=True`，会对该条进行**重新生成答案**（最多如 3 次），再再次走校验与修复流程。

---

## 五、小结表

| 阶段 | 步骤概要 | 校验/修复要点 |
|------|----------|----------------|
| **问题生成** | 1 加载 pipeline → 2 预填充(claim/target_object) → 3 槽位填充 → 4 LLM 生成问题 → 5 验证(当前跳过) → 6 输出 | 预填充校验 claim/name 非空；必需槽位缺失则丢弃；可选启用全局/Pipeline 约束与 LLM 问题验证、豁免部分 pipeline。 |
| **答案生成** | 选择题：正确答+错误选项+打乱+correct_option；填空题：直接生成答案 | 无在此阶段校验，由答案校验阶段统一处理。 |
| **答案校验** | 1 格式检查与修复 → 2 VQA 验证(置信度+答案一致性) → 可选修复(Repairer)与重试 | 占位符、选项重复、答案完整性等可自动修；VQA 未过可抢救或重试生成。 |

以上即为 QA_Generator 中问题与答案的生成步骤及校验逻辑的概览；具体配置与开关以 `question_config.json`、`answer_config.json` 及 pipeline 参数为准。
