# 问题重新生成逻辑说明

## 1. 整体流程

```
VQA Generator 异步处理单条记录 (process_one):
  for retry_count in 0..max_retries (默认 3):
    ① 调用 question_generator.generate_question_async(..., retry_context=retry_ctx)
       - 首次: retry_ctx=None
       - 重试: retry_ctx={previous_question, validation_reason, retry_count}
    ② 调用 validator.validate_async(question, ...)
    ③ 若 is_valid → 成功，退出循环
    ④ 若 retry_count < max_retries → 打印重试信息，回到 ①
    ⑤ 若耗尽重试 → 返回错误
```

## 2. 重试时的 Prompt 注入

**question_generator_prefill.py _build_generation_prompt:**

当 `retry_context` 非空时，在 prompt 开头注入：

```
[RETRY] A previous question failed validation.
Previous question was: "{previous_question}".
Failure reason: {validation_reason}.
Please generate a DIFFERENT question that addresses the validation failure. Re-examine the requirements carefully.
```

## 3. 重试时的 Temperature

- 首次: 0.7
- 重试: 0.7 + min(0.15 * retry_count, 0.2)，最高约 0.9，增加多样性

## 4. 与答案重试的对比

| 项目 | 问题生成 | 答案生成 |
|------|----------|----------|
| 重试次数 | 3 | 3 |
| 失败反馈 | previous_question + validation_reason | previous_answer + regeneration_reason |
| Temperature 递增 | 0.7 → 0.9 | 0.3 → 0.7 |
