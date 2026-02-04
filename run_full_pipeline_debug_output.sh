#!/usr/bin/env bash
set -euo pipefail

# 完整 3 步 Pipeline 错误输出检测脚本
# 仿照 run_full_pipeline.sh，用于记录各阶段模型返回值，便于排查槽位填充、问题生成、答案生成等错误
#
# 记录内容：
# 1. Step 1 - 对象选择：Baseline 模型完成 claim 模板的返回值
# 2. Step 1 - 预填充：Judge 模型 prefill 的 prompt 与返回值
# 3. Step 1 - 验证：Judge 模型 verification 的 prompt 与返回值
# 4. Step 3 - 槽位填充：LLM 填充 required_slots 的 prompt 与返回值
# 5. Step 3 - 问题生成：每次 question 生成时的 prompt 与返回值
# 6. Step 3 - 问题验证：validator 的 prompt 与返回值
# 7. Step 3 - 答案生成：answer generator 各次调用的 prompt 与返回值
# 8. Step 3 - 答案验证：answer validator 的 prompt 与返回值
#
# 输出目录结构：
#   DEBUG_OUTPUT_DIR/
#   ├── step1_baseline_responses.jsonl      # Baseline 对象选择/完成返回值
#   ├── step1_judge_prompts.jsonl          # Judge prefill + verification prompt 与 response
#   ├── probing_output/                    # Step 1 正常输出
#   ├── failure_key_sampler/                # Step 2 输出
#   └── qa_generator/
#       ├── model_responses/               # Step 3 各阶段模型返回值
#       │   ├── slot_filling.jsonl
#       │   ├── question_generation.jsonl
#       │   ├── question_validation.jsonl
#       │   ├── answer_generation.jsonl
#       │   └── answer_validation.jsonl
#       └── ...                             # 正常 VQA 输出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# 调试输出根目录（所有模型返回值记录存放于此）
# =========================
DEBUG_RUN_ID="${DEBUG_RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-${SCRIPT_DIR}/debug_output_$(date +%Y%m%d_%H%M%S)}"

# =========================
# 必需参数（与 run_full_pipeline.sh 对齐）
# =========================
PROJECT_ROOT="${PROJECT_ROOT:-/home/zhuxuzhou/My_project}"

PARQUET_DIR="${PARQUET_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/}"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-VL-235B-A22B-Instruct}"
OUTPUT_DIR="${DEBUG_OUTPUT_DIR}/probing_output"

PIPELINE_CONFIG="${PIPELINE_CONFIG:-${PROJECT_ROOT}/FactorFilterAgent/failure_key_sampler/configs/pipeline_config.example.json}"
ERROR_OUTPUT="${DEBUG_OUTPUT_DIR}/failure_key_sampler/debug_error_cases.json"

# =========================
# 调试模式参数（减少样本量以加快调试）
# =========================
NPROC_PER_NODE=1
TARGET_FAILURE_COUNT=50
FAILURE_BATCH_SIZE=10
MAX_EMPTY_BATCHES=3
PARQUET_SAMPLE_SIZE=4
SEED=42
START_INDEX=0

CONCURRENCY=5
NUM_GPUS=1
ENABLE_VALIDATION_EXEMPTIONS=true
MAX_SAMPLES=20
QA_DEBUG=1
LOG_FILE=""
REQUEST_DELAY=0.1
NO_ASYNC=false
NO_INTERMEDIATE=false
BATCH_SIZE=1000
PIPELINES=()
QUESTION_CONFIG=""
ANSWER_CONFIG=""

# =========================
# 模型调用端点
# =========================
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://10.158.159.139:8000/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-235B-A22B-Instruct}"
USE_LB_CLIENT="${USE_LB_CLIENT:-false}"
API_KEY="${API_KEY:-${OPENAI_API_KEY}}"
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL}}"

# =========================
# 模型返回值记录路径（Step 1）
# =========================
JUDGE_PROMPT_LOG_PATH="${DEBUG_OUTPUT_DIR}/step1_judge_prompts_and_responses.jsonl"
BASELINE_RESPONSE_LOG_PATH="${DEBUG_OUTPUT_DIR}/step1_baseline_responses.jsonl"

# =========================
# 模型返回值记录路径（Step 3）
# =========================
QA_OUTPUT_DIR="${DEBUG_OUTPUT_DIR}/qa_generator"
MODEL_RESPONSE_LOG_DIR="${DEBUG_OUTPUT_DIR}/qa_generator/model_responses"

# =========================
# 可选参数
# =========================
USE_LOCAL_BASELINE=true
SAMPLE_SIZE=""
CLAIM_TEMPLATE_CONFIG=""
INCLUDE_SOURCE_METADATA=false

# Prepare paths
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
PARQUET_DIR="$(cd "${PARQUET_DIR}" && pwd)"
DEBUG_OUTPUT_DIR="$(mkdir -p "${DEBUG_OUTPUT_DIR}" && cd "${DEBUG_OUTPUT_DIR}" && pwd)"
OUTPUT_DIR="${DEBUG_OUTPUT_DIR}/probing_output"
ERROR_OUTPUT_DIR="$(dirname "${ERROR_OUTPUT}")"
mkdir -p "${ERROR_OUTPUT_DIR}"
mkdir -p "${QA_OUTPUT_DIR}"
mkdir -p "${MODEL_RESPONSE_LOG_DIR}"

# Truncate log files for clean run
: > "${JUDGE_PROMPT_LOG_PATH}"
: > "${BASELINE_RESPONSE_LOG_PATH}"

if [[ -z "${LOG_FILE}" ]]; then
  LOG_FILE="${QA_OUTPUT_DIR}/vqa_debug_$(date +%m%d_%H%M%S)_log.txt"
fi
LOG_DIR="$(dirname "${LOG_FILE}")"
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Pipeline 错误输出检测模式"
echo "=========================================="
echo "调试运行 ID (trace_id 前缀): ${DEBUG_RUN_ID}"
echo "调试输出根目录: ${DEBUG_OUTPUT_DIR}"
echo "  - Step 1 Judge 记录: ${JUDGE_PROMPT_LOG_PATH}"
echo "  - Step 1 Baseline 记录: ${BASELINE_RESPONSE_LOG_PATH}"
echo "  - Step 3 模型返回值: ${MODEL_RESPONSE_LOG_DIR}"
echo "  - 样本限制: TARGET_FAILURE_COUNT=${TARGET_FAILURE_COUNT}, MAX_SAMPLES=${MAX_SAMPLES}"
echo "=========================================="
echo ""

echo "== Step 1: ProbingFactorGeneration（对象选择 + 预填充 + 验证）=="
STEP1_ARGS=(
  --parquet_dir "${PARQUET_DIR}"
  --target_failure_count "${TARGET_FAILURE_COUNT}"
  --baseline_model_path "${BASELINE_MODEL_PATH}"
  --failure_batch_size "${FAILURE_BATCH_SIZE}"
  --output_dir "${OUTPUT_DIR}"
  --judge_model_name "${JUDGE_MODEL_NAME}"
  --max_empty_batches "${MAX_EMPTY_BATCHES}"
  --parquet_sample_size "${PARQUET_SAMPLE_SIZE}"
)
if [[ "${USE_LOCAL_BASELINE}" == "true" ]]; then
  STEP1_ARGS+=(--use_local_baseline)
fi
if [[ -n "${SAMPLE_SIZE}" ]]; then
  STEP1_ARGS+=(--sample_size "${SAMPLE_SIZE}")
fi
if [[ -n "${CLAIM_TEMPLATE_CONFIG}" ]]; then
  STEP1_ARGS+=(--claim_template_config "${CLAIM_TEMPLATE_CONFIG}")
fi
if [[ "${INCLUDE_SOURCE_METADATA}" == "true" ]]; then
  STEP1_ARGS+=(--include_source_metadata)
fi

# 启用 Judge 与 Baseline 模型返回值记录
OPENAI_API_KEY="${OPENAI_API_KEY}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
MODEL_NAME="${MODEL_NAME}" \
USE_LB_CLIENT="${USE_LB_CLIENT}" \
API_KEY="${API_KEY}" \
BASE_URL="${BASE_URL}" \
USE_SINGLE_DEVICE_MAP=1 \
JUDGE_PROMPT_LOG_PATH="${JUDGE_PROMPT_LOG_PATH}" \
JUDGE_LOG_RESPONSE=1 \
BASELINE_RESPONSE_LOG_PATH="${BASELINE_RESPONSE_LOG_PATH}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  ProbingFactorGeneration/examples/run_complete_pipeline.py \
  "${STEP1_ARGS[@]}"

echo "== Step 2: FactorFilterAgent failure_key_sampler =="
python -m FactorFilterAgent.failure_key_sampler.main \
  --failure_root "${OUTPUT_DIR}" \
  --pipeline_config "${PIPELINE_CONFIG}" \
  --error_output "${ERROR_OUTPUT}" \
  --seed "${SEED}" \
  --start_index "${START_INDEX}" \
  --embed_images

echo "== Step 3: QA_Generator pipeline（槽位填充 + 问题生成 + 答案生成）=="
STEP3_ARGS=(
  "${ERROR_OUTPUT}"
  --concurrency "${CONCURRENCY}"
  --num-gpus "${NUM_GPUS}"
  --request-delay "${REQUEST_DELAY}"
  --batch-size "${BATCH_SIZE}"
  --max-samples "${MAX_SAMPLES}"
)
if [[ "${ENABLE_VALIDATION_EXEMPTIONS}" == "true" ]]; then
  STEP3_ARGS+=(--enable-validation-exemptions)
fi
if [[ "${NO_ASYNC}" == "true" ]]; then
  STEP3_ARGS+=(--no-async)
fi
if [[ "${NO_INTERMEDIATE}" == "true" ]]; then
  STEP3_ARGS+=(--no-intermediate)
fi
if [[ ${#PIPELINES[@]} -gt 0 ]]; then
  STEP3_ARGS+=(--pipelines "${PIPELINES[@]}")
fi
if [[ -n "${QUESTION_CONFIG}" ]]; then
  STEP3_ARGS+=(--question-config "${QUESTION_CONFIG}")
fi
if [[ -n "${ANSWER_CONFIG}" ]]; then
  STEP3_ARGS+=(--answer-config "${ANSWER_CONFIG}")
fi
STEP3_ARGS+=(--debug-questions)
STEP3_ARGS+=(--debug-question-dir "${MODEL_RESPONSE_LOG_DIR}/questions")

# 启用 Step 3 模型返回值记录（含 trace_id 标号）
STEP3_ENV="OPENAI_API_KEY=${OPENAI_API_KEY} OPENAI_BASE_URL=${OPENAI_BASE_URL} MODEL_NAME=${MODEL_NAME}"
STEP3_ENV="${STEP3_ENV} QA_DEBUG=1"
STEP3_ENV="${STEP3_ENV} MODEL_RESPONSE_LOG_DIR=${MODEL_RESPONSE_LOG_DIR}"
STEP3_ENV="${STEP3_ENV} DEBUG_RUN_ID=${DEBUG_RUN_ID}"

if [[ -n "${LOG_FILE}" ]]; then
  (cd "${QA_OUTPUT_DIR}" && eval "${STEP3_ENV}" python "${PROJECT_ROOT}/QA_Generator/pipeline/pipeline.py" "${STEP3_ARGS[@]}" > "${LOG_FILE}" 2>&1)
  echo "VQA 日志已保存到: ${LOG_FILE}"
else
  (cd "${QA_OUTPUT_DIR}" && eval "${STEP3_ENV}" python "${PROJECT_ROOT}/QA_Generator/pipeline/pipeline.py" "${STEP3_ARGS[@]}")
fi

echo ""
echo "✅ 错误输出检测流程完成。"
echo ""
echo "=========================================="
echo "模型返回值记录汇总（含 trace_id 标号）"
echo "=========================================="
echo "trace_id 格式: [DEBUG_RUN_ID_]B{batch}_R{record}_S{sample}_I{image_id}"
echo "  用于跨 Step 1/2/3 锁定同一条记录，例如: grep 'I<image_id>' *.jsonl"
echo "Step 1 (对象选择 + 预填充 + 验证):"
echo "  - Judge prompt+response: ${JUDGE_PROMPT_LOG_PATH}"
echo "  - Baseline 完成返回值:   ${BASELINE_RESPONSE_LOG_PATH}"
echo ""
echo "Step 3 (问题 + 答案):"
echo "  - 槽位填充:   ${MODEL_RESPONSE_LOG_DIR}/slot_filling.jsonl"
echo "  - 问题生成:   ${MODEL_RESPONSE_LOG_DIR}/question_generation.jsonl"
echo "  - 问题验证:   ${MODEL_RESPONSE_LOG_DIR}/question_validation.jsonl"
echo "  - 答案生成:   ${MODEL_RESPONSE_LOG_DIR}/answer_generation.jsonl"
echo "  - 答案验证:   ${MODEL_RESPONSE_LOG_DIR}/answer_validation.jsonl"
echo ""
echo "调试输出根目录: ${DEBUG_OUTPUT_DIR}"
echo "=========================================="
