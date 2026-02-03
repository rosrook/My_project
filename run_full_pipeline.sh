#!/usr/bin/env bash
set -euo pipefail

# Full 3-step pipeline runner:
# 1) ProbingFactorGeneration -> expanded image data
# 2) FactorFilterAgent failure_key_sampler -> pipeline-mapped samples
# 3) QA_Generator pipeline -> VQA dataset
#
# Parameter fill status:
# - All required and common parameters are filled.
# - Optional (empty) parameters have defaults in downstream code:
#   SAMPLE_SIZE, CLAIM_TEMPLATE_CONFIG, PIPELINES, MAX_SAMPLES, QUESTION_CONFIG, ANSWER_CONFIG.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Required parameters
# =========================
PROJECT_ROOT="/home/zhuxuzhou/My_project"

PARQUET_DIR="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/"
BASELINE_MODEL_PATH="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model"
JUDGE_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
OUTPUT_DIR="/home/zhuxuzhou/My_project/data/output_1_29_prefill"

PIPELINE_CONFIG="/home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/configs/pipeline_config.example.json"
ERROR_OUTPUT="/home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/img_with_pipeline_type_and_prefill/third_refined_error_cases.json"

# =========================
# Common parameters
# =========================
NPROC_PER_NODE=8  # Using 8 GPUs for distributed processing
TARGET_FAILURE_COUNT=5000
# Optimized for 8 GPUs: Each GPU processes this batch size, total = 100 * 8 = 800 images/batch
# Increased from 50 to 100 for better GPU utilization (was 50)
FAILURE_BATCH_SIZE=100
MAX_EMPTY_BATCHES=10
# Optimized: Increased parquet sample size to reduce I/O overhead (was 4)
# With 8 GPUs, we can load more parquet files in parallel
PARQUET_SAMPLE_SIZE=16
SEED=42
START_INDEX=20000

# Optimized for 8 GPUs: Increased concurrency for Step 3 (was 10)
# With 8 GPUs available, we can support higher concurrency (adjust based on API limits)
CONCURRENCY=20
ENABLE_VALIDATION_EXEMPTIONS=true
LOG_FILE="/home/zhuxuzhou/My_project/QA_Generator/vqa_ready4use_$(date +%m%d_%H%M%S)_log.txt"

# =========================
# OpenAI-compatible endpoint (model calls)
# =========================
OPENAI_API_KEY="EMPTY"
OPENAI_BASE_URL="http://10.158.159.139:8000/v1"
MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"

# ProbingFactorGeneration (AsyncGeminiClient) uses these:
USE_LB_CLIENT="false"
API_KEY="${OPENAI_API_KEY}"
BASE_URL="${OPENAI_BASE_URL}"

# =========================
# Less-used parameters (optional - defaults below when empty)
# =========================
USE_LOCAL_BASELINE=true
# SAMPLE_SIZE: empty = not used (Step 1 uses target_failure_count mode; default in Python=10 if fixed sampling)
SAMPLE_SIZE=""
# CLAIM_TEMPLATE_CONFIG: empty = Step 1 uses default "configs/claim_template.example_v1_1.json" (resolved under ProbingFactorGeneration/)
CLAIM_TEMPLATE_CONFIG=""
INCLUDE_SOURCE_METADATA=false

# Optimized: Reduced request delay for Step 3 (was 0.1, adjust if API has rate limits)
REQUEST_DELAY=0.05
NO_ASYNC=false
NO_INTERMEDIATE=false
BATCH_SIZE=1000
# PIPELINES: empty = Step 3 uses all pipelines (default None in Python)
PIPELINES=()
# MAX_SAMPLES: empty = Step 3 processes all records (default None in Python)
MAX_SAMPLES=""
# QUESTION_CONFIG: empty = Step 3 uses QA_Generator/question/config/question_config.json (default None → code uses project path)
QUESTION_CONFIG=""
# ANSWER_CONFIG: empty = Step 3 uses QA_Generator/answer/answer_config.json (default None → code uses project path)
ANSWER_CONFIG=""
# QA_DEBUG: set to 1 or true to enable DEBUG output in QA_Generator (slot_filler, gemini_client, pipeline, validator)
QA_DEBUG=""

# =========================
# Performance optimization notes for 8 GPUs
# =========================
# Step 1 (ProbingFactorGeneration):
#   - Uses torchrun with NPROC_PER_NODE=8 for distributed processing
#   - Each GPU processes FAILURE_BATCH_SIZE images independently
#   - Total throughput = FAILURE_BATCH_SIZE * 8 GPUs = 800 images/batch
#   - Auto-optimization adjusts concurrency per GPU based on batch_size
#   - PARQUET_SAMPLE_SIZE=16 allows loading more data files in parallel across GPUs
#
# Step 3 (QA_Generator):
#   - Currently uses single GPU mode (num_gpus=1 hardcoded)
#   - CONCURRENCY=20 leverages available compute resources
#   - Can be further optimized if code supports multi-GPU mode

# Prepare paths
PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
PARQUET_DIR="$(cd "${PARQUET_DIR}" && pwd)"
OUTPUT_DIR="$(mkdir -p "${OUTPUT_DIR}" && cd "${OUTPUT_DIR}" && pwd)"
ERROR_OUTPUT_DIR="$(dirname "${ERROR_OUTPUT}")"
mkdir -p "${ERROR_OUTPUT_DIR}"

if [[ -n "${LOG_FILE}" ]]; then
  LOG_DIR="$(dirname "${LOG_FILE}")"
  mkdir -p "${LOG_DIR}"
fi

cd "${PROJECT_ROOT}"

echo "== Step 1: ProbingFactorGeneration =="
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

# Step 1 uses auto-optimization internally (request_delay=0.0, auto-concurrency)
OPENAI_API_KEY="${OPENAI_API_KEY}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
MODEL_NAME="${MODEL_NAME}" \
USE_LB_CLIENT="${USE_LB_CLIENT}" \
API_KEY="${API_KEY}" \
BASE_URL="${BASE_URL}" \
torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  ProbingFactorGeneration/examples/run_complete_pipeline.py \
  "${STEP1_ARGS[@]}"

echo "== Step 2: FactorFilterAgent failure_key_sampler =="
python -m FactorFilterAgent.failure_key_sampler.main \
  --failure_root "${OUTPUT_DIR}" \
  --pipeline_config "${PIPELINE_CONFIG}" \
  --error_output "${ERROR_OUTPUT}" \
  --seed "${SEED}" \
  --start_index "${START_INDEX}"

echo "== Step 3: QA_Generator pipeline =="
STEP3_ARGS=(
  "${ERROR_OUTPUT}"
  --concurrency "${CONCURRENCY}"
  --request-delay "${REQUEST_DELAY}"
  --batch-size "${BATCH_SIZE}"
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
if [[ -n "${MAX_SAMPLES}" ]]; then
  STEP3_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${QUESTION_CONFIG}" ]]; then
  STEP3_ARGS+=(--question-config "${QUESTION_CONFIG}")
fi
if [[ -n "${ANSWER_CONFIG}" ]]; then
  STEP3_ARGS+=(--answer-config "${ANSWER_CONFIG}")
fi

# Build Step 3 env (QA_DEBUG controls slot_filler/gemini_client/pipeline DEBUG output)
STEP3_ENV="OPENAI_API_KEY=${OPENAI_API_KEY} OPENAI_BASE_URL=${OPENAI_BASE_URL} MODEL_NAME=${MODEL_NAME}"
if [[ "${QA_DEBUG}" == "1" || "${QA_DEBUG}" == "true" || "${QA_DEBUG}" == "yes" ]]; then
  STEP3_ENV="${STEP3_ENV} QA_DEBUG=1"
fi

if [[ -n "${LOG_FILE}" ]]; then
  eval "${STEP3_ENV}" python QA_Generator/pipeline/pipeline.py "${STEP3_ARGS[@]}" > "${LOG_FILE}" 2>&1
  echo "VQA log saved to: ${LOG_FILE}"
else
  eval "${STEP3_ENV}" python QA_Generator/pipeline/pipeline.py "${STEP3_ARGS[@]}"
fi

echo "✅ All steps completed."
