#!/usr/bin/env bash
set -euo pipefail

# Full 3-step pipeline runner:
# 1) ProbingFactorGeneration -> expanded image data
# 2) FactorFilterAgent failure_key_sampler -> pipeline-mapped samples
# 3) QA_Generator pipeline -> VQA dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# Required parameters
# =========================
PROJECT_ROOT="/home/zhuxuzhou/My_project"

PARQUET_DIR="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--OpenImages/data/train/"
BASELINE_MODEL_PATH="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model"
JUDGE_MODEL_NAME="/workspace/Qwen3-VL-235B-A22B-Instruct"
OUTPUT_DIR="/home/zhuxuzhou/My_project/data/output_1_17_prefill"

PIPELINE_CONFIG="/home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/configs/pipeline_config.example.json"
ERROR_OUTPUT="/home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/img_with_pipeline_type_and_prefill/second_refined_error_cases.json"

# =========================
# Common parameters
# =========================
NPROC_PER_NODE=8
TARGET_FAILURE_COUNT=5000
FAILURE_BATCH_SIZE=50
MAX_EMPTY_BATCHES=10
PARQUET_SAMPLE_SIZE=4
SEED=42
START_INDEX=20000

CONCURRENCY=10
ENABLE_VALIDATION_EXEMPTIONS=true
LOG_FILE="/home/zhuxuzhou/My_project/QA_Generator/vqa_ready4use_$(date +%m%d_%H%M%S)_log.txt"

# =========================
# Less-used parameters
# =========================
USE_LOCAL_BASELINE=true
SAMPLE_SIZE=""
CLAIM_TEMPLATE_CONFIG=""
INCLUDE_SOURCE_METADATA=false

REQUEST_DELAY=0.1
NO_ASYNC=false
NO_INTERMEDIATE=false
BATCH_SIZE=1000
PIPELINES=()
MAX_SAMPLES=""
QUESTION_CONFIG=""
ANSWER_CONFIG=""

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

if [[ -n "${LOG_FILE}" ]]; then
  python QA_Generator/pipeline/pipeline.py "${STEP3_ARGS[@]}" > "${LOG_FILE}" 2>&1
  echo "VQA log saved to: ${LOG_FILE}"
else
  python QA_Generator/pipeline/pipeline.py "${STEP3_ARGS[@]}"
fi

echo "✅ All steps completed."
