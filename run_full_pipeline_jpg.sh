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

# Input JPG
IMAGE_PATH="/path/to/your.jpg"

# ProbingFactorGeneration (step 1)
OUTPUT_DIR="${PROJECT_ROOT}/data/output_jpg_debug"
BASELINE_MODEL_PATH="/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/hf_baseline_model"
JUDGE_MODEL_NAME="/workspace/Qwen3-VL-235B-A22B-Instruct"
CLAIM_TEMPLATE_CONFIG="${PROJECT_ROOT}/ProbingFactorGeneration/configs/claim_template.example_v1_1.json"
USE_LOCAL_BASELINE=true
SAMPLE_SIZE=1
PARQUET_SAMPLE_SIZE=1
RANDOM_SEED=42
INCLUDE_SOURCE_METADATA=false

# Failure mapping (step 2)
PIPELINE_CONFIG="${PROJECT_ROOT}/FactorFilterAgent/failure_key_sampler/configs/pipeline_config.example.json"
ERROR_OUTPUT="${PROJECT_ROOT}/FactorFilterAgent/failure_key_sampler/img_with_pipeline_type_and_prefill/jpg_debug_error_cases.json"
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
LOG_FILE="${PROJECT_ROOT}/QA_Generator/vqa_jpg_debug_$(date +%m%d_%H%M%S)_log.txt"

# =========================
# Derived paths
# =========================
PARQUET_DIR="${PROJECT_ROOT}/debug_jpg_parquet"
PARQUET_FILE="${PARQUET_DIR}/single_image.parquet"
FAILURE_ROOT="${OUTPUT_DIR}/rank_0"

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

torchrun --nproc_per_node=1 \
  "${PROJECT_ROOT}/ProbingFactorGeneration/examples/run_complete_pipeline.py" \
  "${STEP1_ARGS[@]}"

# =========================
# Step 2: FactorFilterAgent failure_key_sampler
# =========================
python -m FactorFilterAgent.failure_key_sampler.main \
  --failure_root "${FAILURE_ROOT}" \
  --pipeline_config "${PIPELINE_CONFIG}" \
  --error_output "${ERROR_OUTPUT}" \
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

python "${PROJECT_ROOT}/QA_Generator/pipeline/pipeline.py" "${STEP3_ARGS[@]}" > "${LOG_FILE}" 2>&1

echo "✅ JPG full pipeline complete."
echo "Log: ${LOG_FILE}"
