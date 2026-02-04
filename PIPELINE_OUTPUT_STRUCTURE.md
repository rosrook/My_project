# Pipeline 输出文件结构说明

运行 `run_full_pipeline.sh` 后，会产生以下文件结构：

## 目录结构总览

```
/home/zhuxuzhou/My_project/
├── data/
│   └── output_1_29_prefill/          # Step 1 输出目录 (OUTPUT_DIR)
│       ├── rank_0/                   # GPU 0 的输出（分布式模式）
│       │   ├── failures/              # 失败案例目录
│       │   │   ├── {image_id}_result.json  # 每个失败图片的详细结果
│       │   │   └── {image_id}.jpg          # 失败图片（JPEG格式）
│       │   ├── {first_image_id}_result.json  # 第一个处理的图片结果（无论是否失败）
│       │   └── {first_image_id}.jpg         # 第一个处理的图片
│       ├── rank_1/                   # GPU 1 的输出
│       │   └── failures/
│       │       └── ...
│       ├── ...                       # rank_2 到 rank_7 (共8个GPU)
│       └── probing_results_failures.json  # 所有失败案例的汇总（如果成功完成）
│
├── FactorFilterAgent/
│   └── failure_key_sampler/
│       └── img_with_pipeline_type_and_prefill/
│           └── third_refined_error_cases.json  # Step 2 输出（ERROR_OUTPUT）
│
├── QA_Generator/
│   └── vqa_ready4use_MMDD_HHMMSS_log.txt  # Step 3 日志文件 (LOG_FILE)
│
└── vqa_ready4use_YYYYMMDD_HHMMSS/    # Step 3 输出目录（在工作目录下）
    ├── vqa_dataset_successful_YYYYMMDD_HHMMSS.json  # ✅ 最终可用的VQA数据集
    ├── question_errors_YYYYMMDD_HHMMSS.json          # 问题生成失败的数据
    ├── answer_validation_failed_YYYYMMDD_HHMMSS.json # 答案校验失败的数据
    ├── meta.json                                      # 元信息（统计、配置等）
    ├── intermediate/                                  # 中间结果（如果 NO_INTERMEDIATE=false）
    │   ├── questions/                                 # 问题生成中间结果
    │   │   └── batch_{N}_questions_{timestamp}.json
    │   └── answers/                                   # 答案生成中间结果
    │       └── batch_{N}_answers_{timestamp}.json
    └── failed_selection/                              # 对象选择失败案例（如果启用debug）
        └── ...
```

---

## Step 1: ProbingFactorGeneration 输出

**输出目录**: `OUTPUT_DIR` (默认: `/home/zhuxuzhou/My_project/data/output_1_29_prefill`)

### 分布式模式（8 GPU）结构

```
output_1_29_prefill/
├── rank_0/                    # GPU 0 的输出
│   ├── failures/              # 失败案例目录（重要！）
│   │   ├── {image_id}_result.json    # 每个失败图片的完整结果
│   │   │                            # 包含：claim_templates, completions, verifications, aggregated_failures
│   │   └── {image_id}.jpg            # 失败图片（JPEG格式，quality=95）
│   ├── {first_image_id}_result.json  # 第一个处理的图片结果（无论是否失败）
│   └── {first_image_id}.jpg         # 第一个处理的图片
│
├── rank_1/                    # GPU 1 的输出
│   └── failures/
│       └── ...
│
├── ...                        # rank_2 到 rank_7
│
└── probing_results_failures.json  # 所有失败案例的汇总JSON（如果成功完成）
                                    # 包含所有 rank 的失败结果
```

### 关键文件说明

1. **`rank_{N}/failures/{image_id}_result.json`**
   - 每个失败图片的详细结果
   - 包含：claim_templates, completions, verifications, aggregated_failures, filtering_factors
   - Step 2 会读取这些文件

2. **`rank_{N}/failures/{image_id}.jpg`**
   - 失败图片的副本（JPEG格式）
   - Step 2 需要这些图片来构建 error_cases.json

3. **`probing_results_failures.json`**
   - 所有失败案例的汇总
   - 包含完整的 results 和 summary
   - 仅在成功完成时生成（达到 target_failure_count 或正常结束）

---

## Step 2: FactorFilterAgent failure_key_sampler 输出

**输出文件**: `ERROR_OUTPUT` (默认: `/home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/img_with_pipeline_type_and_prefill/third_refined_error_cases.json`)

### 文件内容

```json
[
  {
    "sample_index": 20000,
    "id": <record_id>,
    "source_a": {
      "original_id": "<image_id>",
      "jpg": "<base64_encoded_image>"
    },
    "prefill": {
      "claim": "<prefill_claim_text>",
      "target_object": "<target_object_name>"
    },
    "pipeline_type": "<pipeline_type>",
    "pipeline_name": "<pipeline_name>"
  },
  ...
]
```

**说明**:
- 从 Step 1 的 `failures/` 目录读取失败案例
- 为每个失败案例采样一个 failure_key
- 映射到对应的 pipeline_type 和 pipeline_name
- 将图片编码为 base64 格式
- Step 3 会读取这个文件

---

## Step 3: QA_Generator 输出

**输出目录**: 工作目录下的 `vqa_ready4use_YYYYMMDD_HHMMSS/`（自动创建）

### 目录结构

```
vqa_ready4use_YYYYMMDD_HHMMSS/
├── vqa_dataset_successful_YYYYMMDD_HHMMSS.json  # ✅ 最终可用的VQA数据集
├── question_errors_YYYYMMDD_HHMMSS.json          # 问题生成失败的数据
├── answer_validation_failed_YYYYMMDD_HHMMSS.json # 答案校验失败的数据
├── meta.json                                      # 元信息和统计
├── intermediate/                                  # 中间结果（如果 NO_INTERMEDIATE=false）
│   ├── questions/
│   │   └── batch_{N}_questions_{timestamp}.json
│   └── answers/
│       └── batch_{N}_answers_{timestamp}.json
└── failed_selection/                              # 对象选择失败案例（可选）
    └── ...
```

### 关键文件说明

1. **`vqa_dataset_successful_YYYYMMDD_HHMMSS.json`** ⭐ **最重要的文件**
   - 最终可用的 VQA 数据集
   - 包含所有校验通过的问答对
   - 格式：
     ```json
     [
       {
         "question": "<question_text>",
         "question_type": "<type>",
         "image_base64": "<base64_image>",
         "answer": "<answer_text>",
         "explanation": "<explanation>",
         "full_question": "<full_question_with_context>",
         "options": {...},  // 如果是选择题
         "correct_option": "<option_key>",
         "pipeline_name": "<pipeline_name>",
         "pipeline_intent": "<intent>",
         "answer_type": "<type>",
         "sample_index": <index>,
         "id": <id>,
         "source_a_id": "<original_id>",
         "validation_report": {...},
         "timestamp": "<iso_timestamp>",
         "generated_at": "<iso_timestamp>"
       },
       ...
     ]
     ```

2. **`question_errors_YYYYMMDD_HHMMSS.json`**
   - 问题生成阶段失败的数据
   - 包含错误信息和原始输入数据

3. **`answer_validation_failed_YYYYMMDD_HHMMSS.json`**
   - 答案生成或校验失败的数据
   - 包含问题但答案生成失败或校验不通过

4. **`meta.json`**
   - 元信息和统计
   - 包含：总记录数、成功数、失败数、配置参数等

5. **`intermediate/`** (如果 `NO_INTERMEDIATE=false`)
   - 中间结果文件
   - 用于调试和检查中间步骤

---

## 日志文件

**位置**: `LOG_FILE` (默认: `/home/zhuxuzhou/My_project/QA_Generator/vqa_ready4use_MMDD_HHMMSS_log.txt`)

包含 Step 3 的所有输出（stdout 和 stderr），包括：
- 进度信息
- 错误和警告
- 统计信息

---

## 文件大小估算

假设处理 5000 个失败案例：

- **Step 1 输出**:
  - `failures/*.json`: ~5000 个文件，每个 ~50-200 KB = ~250 MB - 1 GB
  - `failures/*.jpg`: ~5000 个文件，每个 ~100-500 KB = ~500 MB - 2.5 GB
  - 总计: ~1-4 GB

- **Step 2 输出**:
  - `error_cases.json`: ~5000 条记录，每条包含 base64 图片 = ~500 MB - 2 GB

- **Step 3 输出**:
  - `vqa_dataset_successful_*.json`: ~5000 条记录 = ~100-500 MB
  - `intermediate/`: 取决于是否保存中间结果 = ~200 MB - 1 GB
  - 总计: ~300 MB - 1.5 GB

**总磁盘空间需求**: 约 **2-8 GB**（取决于图片大小和中间结果）

---

## 注意事项

1. **Step 1 分布式输出**: 8 个 GPU 会创建 8 个 `rank_{N}/` 目录，每个都有独立的 `failures/` 目录

2. **Step 2 输入**: 从 Step 1 的 `failures/` 目录读取，会自动扫描所有 `rank_{N}/failures/` 目录

3. **Step 3 输出位置**: 在工作目录（`PROJECT_ROOT`）下创建，不是固定的输出目录

4. **中间结果**: 如果 `NO_INTERMEDIATE=true`，不会保存中间的问题和答案文件，只保留最终结果

5. **日志文件**: Step 1 和 Step 2 的日志输出到终端，Step 3 的日志保存到 `LOG_FILE`
