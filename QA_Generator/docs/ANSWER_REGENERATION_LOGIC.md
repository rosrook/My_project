# 答案重新生成逻辑说明

## 1. 整体流程

```
Pipeline 处理单条记录:
  while retry_count <= max_retries (默认 3):
    ① 调用 generate_answer_async(question, image_base64, question_type, pipeline_info)
    ② 调用 validator.validate_and_fix(result, image_base64)
       - Step 1: 格式检查与修复
       - Step 2: VQA 验证（置信度 + 答案一致性）
       - 若未通过 → 调用 AnswerRepairer.repair_once() 尝试修复
       - 若 repair 成功且再验证通过 → 返回通过
    ③ 若 validation_passed → 成功，退出
    ④ 若 should_regenerate=True → retry_count += 1，回到 ①
```

## 2. 重新生成时的调用

**Pipeline (pipeline.py 第 483 行):**
```python
answer_result = await self.answer_generator.generate_answer_async(
    question=question,
    image_base64=image_base64,
    question_type=question_type,
    pipeline_info=pipeline_info,
    async_client=client,
    model=model_name
)
```

**关键点：每次重试传入的参数完全相同，没有任何变化。**

## 3. 答案生成 Prompt（正确答桉）

**answer_generator.py _generate_correct_answer_async:**
```
Based on the image and the question, provide a concise and accurate answer.

Question: {question}
{target_hint}  # claim + prefilled_values（若有）

Requirements:
1. Provide ONLY the answer text, keep it concise and direct (typically 1-5 words)
2. Do NOT include any analysis, reasoning, or explanation in the answer field
...
Provide your response in the following format:
Answer: [your concise answer here, 1-5 words typically]
Explanation: [optional brief explanation]
```

**Temperature: 0.3**（较低，输出较 deterministic）

## 4. 存在的问题

### 4.1 重试时 prompt 完全一致
- 每次重试的 question、image_base64、pipeline_info 完全相同
- **没有传入**：上一次的答案、验证失败原因（regeneration_reason）
- 模型无法知道「上次生成失败」，也无法针对性改进

### 4.2 高相似度 / 无效重复
- temperature=0.3 较低，同一输入下模型输出高度相似
- 若第一次生成「3」被误判失败（如之前的 len<2 问题），重试很可能再次生成「3」
- 若失败原因是「置信度低」或「答案与图像不符」，重试时模型没有这些反馈，无法调整

### 4.3 重试本质上是「盲试」
- 没有利用 validation_report 中的 regeneration_reason
- 没有利用上一次的 answer_result
- 相当于用相同 prompt 多次采样，成功率提升有限

## 5. 已实现的改进（2025-01）

1. **重试时注入失败反馈**：Pipeline 在重试时传入 `retry_context`，包含：
   - `previous_answer`: 上一次的答案
   - `regeneration_reason`: 验证失败原因（如「置信度评估未通过」「答案验证未通过」）
   - `retry_count`: 当前重试次数
   - Prompt 中增加 `[RETRY]` 段落，告知模型上次答案与失败原因，要求重新审视图像并给出不同/修正后的答案

2. **重试时提高 temperature**：首次 0.3，重试时 0.5~0.7（随 retry_count 递增），增加输出多样性

3. **answer_generator.generate_answer_async** 新增可选参数 `retry_context`
