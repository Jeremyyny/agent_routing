#!/usr/bin/env bash
# Launch vLLM to serve the three frozen subagents on GPU 0.
#
# Prerequisites:
#   conda activate vllm_env   (separate env with: pip install vllm)
#
# Usage:
#   bash scripts/start_subagent_server.sh <base_model> <teacher_id> [output_root] [port]
#
# Example:
#   bash scripts/start_subagent_server.sh Qwen/Qwen3-8B openai_us4_500_runtime_raw
#   bash scripts/start_subagent_server.sh Qwen/Qwen3-8B openai_us4_500_runtime_raw outputs 8000

set -e

BASE_MODEL=${1:?"Usage: $0 <base_model> <teacher_id> [output_root] [port]"}
TEACHER_ID=${2:?"Usage: $0 <base_model> <teacher_id> [output_root] [port]"}
OUTPUT_ROOT=${3:-"outputs"}
PORT=${4:-8000}

ADAPTER_ROOT="${OUTPUT_ROOT}/adapters/${TEACHER_ID}"
EXTRACTOR="${ADAPTER_ROOT}/extractor_adapter"
REASONER="${ADAPTER_ROOT}/reasoner_adapter"
RULE_APPLIER="${ADAPTER_ROOT}/rule_applier_adapter"

echo "[vLLM] base_model   = ${BASE_MODEL}"
echo "[vLLM] teacher_id   = ${TEACHER_ID}"
echo "[vLLM] extractor    = ${EXTRACTOR}"
echo "[vLLM] reasoner     = ${REASONER}"
echo "[vLLM] rule_applier = ${RULE_APPLIER}"
echo "[vLLM] port         = ${PORT}"
echo "[vLLM] GPU          = 0"

# Verify adapter directories exist before launching.
for DIR in "${EXTRACTOR}" "${REASONER}" "${RULE_APPLIER}"; do
    if [ ! -d "${DIR}" ]; then
        echo "[vLLM] ERROR: adapter directory not found: ${DIR}"
        exit 1
    fi
done

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules \
        extractor="${EXTRACTOR}" \
        reasoner="${REASONER}" \
        rule_applier="${RULE_APPLIER}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --trust-remote-code \
    --max-model-len 4096
