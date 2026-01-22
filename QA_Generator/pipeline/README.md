# QA_Generator/pipeline

本目录提供完整的 QA 生成流水线（问题 + 答案），用于从输入样本生成完整 VQA 数据集。

## 直接使用方式

```bash
python QA_Generator/pipeline/pipeline.py input.json output_dir/
```

常用参数：
```bash
python QA_Generator/pipeline/pipeline.py input.json output_dir/ \
  --pipelines question object_counting \
  -n 100 \
  --batch-size 1000 \
  --concurrency 3 \
  --request-delay 0.1
```

参数说明（常用）：
- `--pipelines`：只运行指定 pipeline
- `-n/--max-samples`：限制处理样本数
- `--batch-size`：分批处理大小（大文件建议设置）
- `--concurrency`：异步并发数（建议 1-5）
- `--request-delay`：请求间隔，避免 API 限流
- `--no-intermediate`：不保存中间问题/答案文件
- `--log-file`：将日志写入文件

## 在 QA 生成流程中的接口形式

`pipeline.py` 负责：
1. 读取输入 JSON（包含 `prefill` 等字段）
2. 调用 `question/prefill` 生成问题
3. 调用 `answer` 生成答案并校验
4. 输出三类结果：
   - `vqa_dataset_successful_*.json`
   - `question_errors_*.json`
   - `answer_validation_failed_*.json`

## 文件说明

### `pipeline.py`
主流程代码，包含：
- 分批处理逻辑
- 问题生成、答案生成与校验
- 结果聚合与统计输出
- 异步并发与重试控制

## 输入数据格式要点
每条记录应包含：
- `prefill`：包含 `claim` 或 `target_object`
- `source_a`：原始图像/元信息
- 其他必要字段由下游阶段填充

具体输入格式见 `QA_Generator/question/prefill/README.md`。
