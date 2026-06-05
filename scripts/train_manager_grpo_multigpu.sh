#!/usr/bin/env bash
# Full-parameter GRPO on GPUs 1-2-3 with ZeRO Stage 3.
# Subagents are served by a vLLM process on GPU 0 (start_subagent_server.sh).
#
# Prerequisites:
#   1. GPU 0: vLLM subagent server is running and healthy.
#      bash scripts/start_subagent_server.sh <base_model> <teacher_id>
#   2. Training env has: pip install accelerate deepspeed trl torch
#
# Usage:
#   bash scripts/train_manager_grpo_multigpu.sh <teacher_id> [extra cli args...]
#
# Example (MedQA, from cold-start adapter):
#   bash scripts/train_manager_grpo_multigpu.sh openai_us4_500_runtime_raw \
#       --base_model Qwen/Qwen3-8B \
#       --medqa_normalized_cache outputs/data/medqa_us4_normalized.jsonl \
#       --train_size 1200 \
#       --exclude_sft_example_ids outputs/sft_data/openai_us4_500/extractor_runtime_raw_sft.jsonl \
#       --exclude_sft_example_ids outputs/sft_data/openai_us4_500/reasoner_runtime_raw_sft.jsonl \
#       --exclude_sft_example_ids outputs/sft_data/openai_us4_500/rule_applier_runtime_raw_sft.jsonl \
#       --mgr_init_adapter outputs/manager/openai_us4_500_runtime_raw/sft_evolved \
#       --mgr_output_dir outputs/manager/openai_us4_500_runtime_raw/grpo_full \
#       --mgr_max_steps 200 \
#       --mgr_use_wandb \
#       --wandb_project agent_routing \
#       --wandb_run_name openai_us4_500_runtime_raw_grpo_full \
#       --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."

set -e

TEACHER_ID=${1:?"Usage: $0 <teacher_id> [extra cli args...]"}
shift

# Wait for vLLM server to be ready (up to 120s).
SERVER_URL="http://localhost:8000"
echo "[TRAIN] waiting for vLLM server at ${SERVER_URL} ..."
for i in $(seq 1 24); do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "[TRAIN] vLLM server is ready."
        break
    fi
    if [ "${i}" -eq 24 ]; then
        echo "[TRAIN] ERROR: vLLM server did not become ready within 120s."
        exit 1
    fi
    sleep 5
done

CUDA_VISIBLE_DEVICES=1,2,3 \
PYTHONUTF8=1 \
accelerate launch \
    --config_file configs/accelerate_zero3.yaml \
    -m src.pipeline.cli train_manager_grpo \
    --teacher_id "${TEACHER_ID}" \
    --subagent_server_url "${SERVER_URL}" \
    --mgr_full_parameter_rl \
    --mgr_bs 2 \
    --mgr_num_generations 2 \
    --mgr_max_completion_length 2048 \
    --mgr_temperature 1.0 \
    --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    "$@"
