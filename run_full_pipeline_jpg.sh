#!/usr/bin/env bash
set -euo pipefail

# Full 3-step pipeline test using a single JPG as input.
# This script converts JPG -> parquet, then runs:
# 1) ProbingFactorGeneration
# 2) FactorFilterAgent/failure_key_sampler
# 3) QA_Generator

# =========================
# CONFIG (edit here)
# =========================
PROJECT_ROOT="/home/zhuxuzhou/My_project"
BASE_OUTPUT_DIR="${PROJECT_ROOT}/debug_jpg_run"

# Input JPG
IMAGE_PATH="/path/to/your.jpg"

# ProbingFactorGeneration (step 1)
OUTPUT_DIR="${BASE_OUTPUT_DIR}/probing_output"
BASELINE_MODEL_PATH="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model"
JUDGE_MODEL_NAME="Qwen3-VL-235B-A22B-Instruct"
CLAIM_TEMPLATE_CONFIG="${PROJECT_ROOT}/ProbingFactorGeneration/configs/claim_template.example_v1_1.json"
USE_LOCAL_BASELINE=true
SAMPLE_SIZE=1
PARQUET_SAMPLE_SIZE=1
RANDOM_SEED=42
# Recommended: include image_path in probing_results.json (helps Step 2 locate images)
INCLUDE_SOURCE_METADATA=true

# Failure mapping (step 2)
PIPELINE_CONFIG="${PROJECT_ROOT}/FactorFilterAgent/failure_key_sampler/configs/pipeline_config.example.json"
FAILURE_CONFIG="${PROJECT_ROOT}/ProbingFactorGeneration/configs/failure_config.example.json"
ERROR_OUTPUT="${BASE_OUTPUT_DIR}/failure_key_sampler/jpg_debug_error_cases.json"
SAMPLER_OUTPUT="${BASE_OUTPUT_DIR}/failure_key_sampler/sampled_failures.jsonl"
SEED=42
START_INDEX=0

# QA_Generator (step 3)
CONCURRENCY=5
ENABLE_VALIDATION_EXEMPTIONS=true
REQUEST_DELAY=0.1
NO_ASYNC=false
NO_INTERMEDIATE=false
BATCH_SIZE=1000
MAX_SAMPLES=""
QUESTION_CONFIG=""
ANSWER_CONFIG=""
QA_OUTPUT_DIR="${BASE_OUTPUT_DIR}/qa_generator"
LOG_FILE="${QA_OUTPUT_DIR}/vqa_jpg_debug_$(date +%m%d_%H%M%S)_log.txt"

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
# Derived paths
# =========================
PARQUET_DIR="${BASE_OUTPUT_DIR}/parquet_input"
PARQUET_FILE="${PARQUET_DIR}/single_image.parquet"
# Probing writes to OUTPUT_DIR in single-rank mode; to OUTPUT_DIR/rank_0 in distributed mode.
FAILURE_ROOT="${OUTPUT_DIR}"

# =========================
# Helpers
# =========================
die() {
  echo "[ERROR] $*" >&2
  exit 1
}

if [[ ! -f "${IMAGE_PATH}" ]]; then
  die "IMAGE_PATH not found: ${IMAGE_PATH}"
fi

mkdir -p "${PARQUET_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${ERROR_OUTPUT}")"
mkdir -p "${QA_OUTPUT_DIR}"

# =========================
# Step 0: JPG -> Parquet
# =========================
IMAGE_PATH="${IMAGE_PATH}" PARQUET_FILE="${PARQUET_FILE}" python - <<'PY'
import os
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    print("[ERROR] pyarrow is required for JPG -> parquet conversion:", e)
    sys.exit(1)

image_path = Path(os.environ["IMAGE_PATH"])
parquet_file = Path(os.environ["PARQUET_FILE"])

if not image_path.exists():
    print(f"[ERROR] IMAGE_PATH not found: {image_path}")
    sys.exit(1)

image_id = image_path.stem
image_bytes = image_path.read_bytes()

table = pa.Table.from_pydict(
    {"image_id": [image_id], "jpg": [image_bytes]},
    schema=pa.schema([("image_id", pa.string()), ("jpg", pa.binary())]),
)

parquet_file.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(table, parquet_file)
print(f"[INFO] Parquet written: {parquet_file}")
PY

# =========================
# Step 1: ProbingFactorGeneration
# =========================
STEP1_ARGS=(
  --parquet_dir "${PARQUET_DIR}"
  --sample_size "${SAMPLE_SIZE}"
  --parquet_sample_size "${PARQUET_SAMPLE_SIZE}"
  --output_dir "${OUTPUT_DIR}"
  --baseline_model_path "${BASELINE_MODEL_PATH}"
  --judge_model_name "${JUDGE_MODEL_NAME}"
  --claim_template_config "${CLAIM_TEMPLATE_CONFIG}"
  --random_seed "${RANDOM_SEED}"
)
if [[ "${USE_LOCAL_BASELINE}" == "true" ]]; then
  STEP1_ARGS+=(--use_local_baseline)
fi
if [[ "${INCLUDE_SOURCE_METADATA}" == "true" ]]; then
  STEP1_ARGS+=(--include_source_metadata)
fi

OPENAI_API_KEY="${OPENAI_API_KEY}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
MODEL_NAME="${MODEL_NAME}" \
USE_LB_CLIENT="${USE_LB_CLIENT}" \
API_KEY="${API_KEY}" \
BASE_URL="${BASE_URL}" \
torchrun --nproc_per_node=1 \
  "${PROJECT_ROOT}/ProbingFactorGeneration/examples/run_complete_pipeline.py" \
  "${STEP1_ARGS[@]}"

# =========================
# Step 2: FactorFilterAgent failure_key_sampler
# =========================
# Determine probing_results.json path (check rank_0 first, then root)
PROBING_RESULTS="${OUTPUT_DIR}/probing_results.json"
if [[ -f "${OUTPUT_DIR}/rank_0/probing_results.json" ]]; then
  PROBING_RESULTS="${OUTPUT_DIR}/rank_0/probing_results.json"
fi

if [[ ! -f "${PROBING_RESULTS}" ]]; then
  echo "[ERROR] probing_results.json not found at: ${PROBING_RESULTS}"
  echo "        Please check Step 1 output."
  exit 1
fi

# Prefer looking for images alongside probing_results.json (run_complete_pipeline.py
# often saves `${image_id}.jpg` into the same output directory).
IMAGE_DIR="$(dirname "${PROBING_RESULTS}")"

python -m FactorFilterAgent.failure_key_sampler.main \
  --input "${PROBING_RESULTS}" \
  --output "${SAMPLER_OUTPUT}" \
  --failure_config "${FAILURE_CONFIG}" \
  --pipeline_config "${PIPELINE_CONFIG}" \
  --error_output "${ERROR_OUTPUT}" \
  --image_dir "${IMAGE_DIR}" \
  --seed "${SEED}" \
  --start_index "${START_INDEX}"

# =========================
# Step 3: QA_Generator pipeline
# =========================
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
if [[ -n "${MAX_SAMPLES}" ]]; then
  STEP3_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi
if [[ -n "${QUESTION_CONFIG}" ]]; then
  STEP3_ARGS+=(--question-config "${QUESTION_CONFIG}")
fi
if [[ -n "${ANSWER_CONFIG}" ]]; then
  STEP3_ARGS+=(--answer-config "${ANSWER_CONFIG}")
fi

(cd "${QA_OUTPUT_DIR}" && \
  OPENAI_API_KEY="${OPENAI_API_KEY}" \
  OPENAI_BASE_URL="${OPENAI_BASE_URL}" \
  MODEL_NAME="${MODEL_NAME}" \
  python "${PROJECT_ROOT}/QA_Generator/pipeline/pipeline.py" "${STEP3_ARGS[@]}" > "${LOG_FILE}" 2>&1)

echo "✅ JPG full pipeline complete."
echo "Log: ${LOG_FILE}"
