#!/usr/bin/env bash
set -euo pipefail

# Wrapper for run_step3_debug_samples.py so you don't need to type CLI args.
# 按需修改下面这些变量即可。

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 2 的输出 JSON（FactorFilterAgent 的 error_output）
INPUT_JSON="/home/zhuxuzhou/My_project/FactorFilterAgent/failure_key_sampler/img_with_pipeline_type_and_prefill/third_more_30k_expanded_error_cases.json"

# 调试输出根目录（每个样本一个子文件夹）
OUTPUT_ROOT="${PROJECT_ROOT}/debug_step3_samples"

# 采样数量
NUM_SAMPLES=10

# 与 run_full_pipeline.sh Step 3 对齐的参数
CONCURRENCY=5
REQUEST_DELAY=0.1
BATCH_SIZE=1000
ENABLE_VALIDATION_EXEMPTIONS=true
NO_ASYNC=false
NO_INTERMEDIATE=false
# 可选：限定只跑某些 pipeline，例如：PIPELINES=("object_relative_position")
PIPELINES=()
QUESTION_CONFIG=""   # 留空则用默认 question_config
ANSWER_CONFIG=""     # 留空则用默认 answer_config
QA_DEBUG=true         # 是否开启 QA 内部 DEBUG 日志

cd "${PROJECT_ROOT}"

ARGS=(
  --input-json "${INPUT_JSON}"
  --output-root "${OUTPUT_ROOT}"
  --num-samples "${NUM_SAMPLES}"
  --concurrency "${CONCURRENCY}"
  --request-delay "${REQUEST_DELAY}"
  --batch-size "${BATCH_SIZE}"
)

if [[ "${ENABLE_VALIDATION_EXEMPTIONS}" == "true" ]]; then
  ARGS+=(--enable-validation-exemptions)
fi
if [[ "${NO_ASYNC}" == "true" ]]; then
  ARGS+=(--no-async)
fi
if [[ "${NO_INTERMEDIATE}" == "true" ]]; then
  ARGS+=(--no-intermediate)
fi
if [[ ${#PIPELINES[@]} -gt 0 ]]; then
  ARGS+=(--pipelines "${PIPELINES[@]}")
fi
if [[ -n "${QUESTION_CONFIG}" ]]; then
  ARGS+=(--question-config "${QUESTION_CONFIG}")
fi
if [[ -n "${ANSWER_CONFIG}" ]]; then
  ARGS+=(--answer-config "${ANSWER_CONFIG}")
fi
if [[ "${QA_DEBUG}" == "true" ]]; then
  ARGS+=(--qa-debug)
fi

echo "Running Step 3 debug samples with arguments:"
echo "  ${ARGS[*]}"

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}" \
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://10.158.159.139:8000/v1}" \
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-235B-A22B-Instruct}" \
python "${PROJECT_ROOT}/run_step3_debug_samples.py" "${ARGS[@]}"

