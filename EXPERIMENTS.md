# Experiment Plan: 0.6B / 4B / 8B on MedQA and LegalBench

Six experiments: three model sizes × two benchmarks.

---

## 1. Experiment Matrix

| ID | Model | Benchmark | Hardware | SFT Data | GRPO Mode |
|----|-------|-----------|----------|----------|-----------|
| A | Qwen3-0.6B | MedQA | RTX 5090 (1 GPU, 32 GB) | GPT-4o-mini online | Single-GPU full-param |
| B | Qwen3-0.6B | LegalBench | RTX 5090 (1 GPU, 32 GB) | GPT-4o-mini online | Single-GPU full-param |
| C | Qwen3-4B | MedQA | Cluster (4× H100, 90 GB) | DeepSeek offline | ZeRO3 + vLLM |
| D | Qwen3-4B | LegalBench | Cluster (4× H100, 90 GB) | DeepSeek offline | ZeRO3 + vLLM |
| E | Qwen3-8B | MedQA | Cluster (4× H100, 90 GB) | DeepSeek offline | ZeRO3 + vLLM |
| F | Qwen3-8B | LegalBench | Cluster (4× H100, 90 GB) | DeepSeek offline | ZeRO3 + vLLM |

**Key shortcuts:**
- Experiments C/E share the same DeepSeek-generated SFT data (same JSONL, different base model).
- Experiments D/F share the same DeepSeek-generated SFT data for LegalBench.
- 0.6B runs (A, B) do NOT need vLLM — all 4 models fit on one 5090.

---

## 2. Hyperparameter Design

### Why parameters differ by model size

| Stage | Parameter | 0.6B | 4B | 8B | Reason |
|-------|-----------|------|----|----|--------|
| Subagent SFT | `sft_lr` | 2e-4 | 1e-4 | 5e-5 | Larger models have sharper loss landscape; high lr causes instability above ~4B |
| Subagent SFT | `sft_bs` | 4 | 2 | 1 | Activation memory scales with model size; bs=1 for 8B keeps peak VRAM under 25 GB on a single GPU |
| Subagent SFT | `sft_grad_accum` | 4 | 8 | 8 | Effective batch stays at 16–32 tokens worth of examples regardless of per-device size |
| Subagent SFT | `sft_max_seq_len` | 2048 | 4096 | 4096 | 0.6B rarely generates long chains; 4B/8B benefit from longer context for complex reasoning |
| Manager cold-start SFT | `manager_sft_lr` | 2e-5 | 1e-5 | 5e-6 | Cold-start teaches format, not knowledge — very small lr avoids overwriting pretrained priors |
| Manager GRPO | `mgr_bs` | 4 | 4 | 2 | 4B is fast enough for bs=4 with ZeRO3; 8B is memory-bound, bs=2 keeps KV cache under budget |
| Manager GRPO | `mgr_num_generations` | 4 | 2 | 2 | More rollouts = better reward variance; 0.6B is cheap; 4B/8B trade diversity for VRAM budget. **mgr_bs must be divisible by mgr_num_generations.** |
| Manager GRPO | `max_completion_length` | 1024 | 2048 | 2048 | 0.6B rarely produces long tool chains; 4B/8B generate more verbose reasoning |
| Manager GRPO | `mgr_max_steps` (MedQA) | 200 | 200 | 200 | ~700 manager train rows after exclusions; 200 steps ≈ 2–3 passes |
| Manager GRPO | `mgr_max_steps` (LegalBench) | 80 | 80 | 80 | ~115 manager train rows; fewer steps avoid overfitting small set |

### Output naming convention

| Experiment | teacher_id | base_model |
|------------|-----------|------------|
| A (MedQA 0.6B GPT) | `gpt_medqa_0p6b` | `Qwen/Qwen3-0.6B` |
| B (LegalBench 0.6B GPT) | `gpt_lb_0p6b` | `Qwen/Qwen3-0.6B` |
| C (MedQA 4B DeepSeek) | `ds_medqa_4b` | `Qwen/Qwen3-4B` |
| D (LegalBench 4B DeepSeek) | `ds_lb_4b` | `Qwen/Qwen3-4B` |
| E (MedQA 8B DeepSeek) | `ds_medqa_8b` | `Qwen/Qwen3-8B` |
| F (LegalBench 8B DeepSeek) | `ds_lb_8b` | `Qwen/Qwen3-8B` |

DeepSeek SFT data is generated once and reused:
- MedQA DeepSeek data lives under `outputs/sft_data/ds_medqa_data/`
- LegalBench DeepSeek data lives under `outputs/sft_data/ds_lb_data/`

---

## 3. One-time: Generate DeepSeek SFT Data (Cluster, run before C/D/E/F)

If you already have the prompt files, skip the export step and go straight to import.

### 3a. MedQA DeepSeek data

```bash
export PYTHONUTF8=1

# Load and cache MedQA (once)
python -m src.pipeline.cli load_medqa \
    --base_model Qwen/Qwen3-8B \
    --medqa_normalized_cache outputs/data/medqa_us4_normalized.jsonl \
    --train_size 1200 --dev_size 200 --test_size 1270

# Export prompts for each subagent (skip if you already have these files)
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli export_deepseek_jsonl \
      --teacher_id ds_medqa_data \
      --agent_kind "$KIND" \
      --medqa_normalized_cache outputs/data/medqa_us4_normalized.jsonl \
      --train_size 1200 --test_size 1270 \
      --n_samples 500
done
# Produces:
#   outputs/sft_data/ds_medqa_data/extractor_deepseek_prompts.jsonl
#   outputs/sft_data/ds_medqa_data/reasoner_deepseek_prompts.jsonl
#   outputs/sft_data/ds_medqa_data/rule_applier_deepseek_prompts.jsonl

# ── EXTERNAL STEP: run DeepSeek on the prompts ────────────────────────────────
# Input:  outputs/sft_data/ds_medqa_data/*_deepseek_prompts.jsonl
# Output: outputs/sft_data/ds_medqa_data/*_deepseek_responses.jsonl
# Each response row must have: {"example_id": ..., "response": "..."}
# ─────────────────────────────────────────────────────────────────────────────

# Import DeepSeek responses into SFT format
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli import_deepseek_jsonl \
      --teacher_id ds_medqa_data \
      --agent_kind "$KIND" \
      --deepseek_prompt_jsonl "outputs/sft_data/ds_medqa_data/${KIND}_deepseek_prompts.jsonl" \
      --deepseek_response_jsonl "outputs/sft_data/ds_medqa_data/${KIND}_deepseek_responses.jsonl" \
      --deepseek_teacher_model deepseek-chat \
      --deepseek_import_raw_responses
done
# Produces (reused by both C and E):
#   outputs/sft_data/ds_medqa_data/extractor_sft.jsonl
#   outputs/sft_data/ds_medqa_data/reasoner_sft.jsonl
#   outputs/sft_data/ds_medqa_data/rule_applier_sft.jsonl
```

### 3b. LegalBench DeepSeek data

```bash
export LB_CONFIGS="abercrombie,hearsay,personal_jurisdiction,proa,successor_liability"

# Export prompts (skip if you already have these files)
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli export_legalbench_jsonl \
      --teacher_id ds_lb_data \
      --agent_kind "$KIND" \
      --legalbench_configs "$LB_CONFIGS" \
      --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
      --legalbench_refresh_cache \
      --train_size 215 --test_size 95 \
      --n_samples 100
done
# Drop --legalbench_refresh_cache on re-runs.

# ── EXTERNAL STEP: run DeepSeek on the prompts ────────────────────────────────

# Import responses
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli import_deepseek_jsonl \
      --teacher_id ds_lb_data \
      --agent_kind "$KIND" \
      --deepseek_prompt_jsonl "outputs/sft_data/ds_lb_data/${KIND}_deepseek_prompts.jsonl" \
      --deepseek_response_jsonl "outputs/sft_data/ds_lb_data/${KIND}_deepseek_responses.jsonl" \
      --deepseek_teacher_model deepseek-chat \
      --deepseek_import_raw_responses
done
```

---

## Experiment A: MedQA + Qwen3-0.6B (RTX 5090, GPT)

```bash
# ── env vars ──────────────────────────────────────────────────────────────────
export TID=gpt_medqa_0p6b
export BASE=Qwen/Qwen3-0.6B
export CACHE=outputs/data/medqa_us4_normalized.jsonl
export TASK="You are a manager agent solving USMLE-style medical multiple-choice questions."
export PYTHONUTF8=1

# ── Step 1: cache MedQA ───────────────────────────────────────────────────────
python -m src.pipeline.cli load_medqa \
    --base_model "$BASE" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --dev_size 200 --test_size 1270

# ── Step 2: synthesize subagent SFT data (GPT-4o-mini, online) ───────────────
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli synth_subagent \
      --base_model "$BASE" --teacher_id "$TID" \
      --teacher_provider openai --teacher_model gpt-4o-mini \
      --agent_kind "$KIND" --n_samples 500 \
      --medqa_normalized_cache "$CACHE" \
      --train_size 1200 --test_size 1270 \
      --task_description "$TASK"
done

# ── Step 3: SFT-train subagents ───────────────────────────────────────────────
# lr=2e-4, bs=4, grad_accum=4, effective_batch=16, epochs=3, seq_len=2048
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE" --teacher_id "$TID" --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 2e-4 \
      --sft_bs 4 --sft_grad_accum 4 --sft_max_seq_len 2048
done

# ── Step 4: validate subagents ────────────────────────────────────────────────
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" --eval_n_samples 50
# Target: json_ok_rate > 0.90, schema_ok_rate > 0.85 for all three agents.

# ── Step 5: build cold-start SFT data ────────────────────────────────────────
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --test_size 1270 \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 300 \
    --task_description "$TASK"

# ── Step 6: train manager on cold-start SFT ──────────────────────────────────
# lr=2e-5, bs=4, grad_accum=4, epochs=1
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_train_jsonl "outputs/manager/${TID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 2e-5 \
    --sft_bs 4 --sft_grad_accum 4 --sft_max_seq_len 2048

# ── Step 7: full-param GRPO (single GPU, no vLLM needed for 0.6B) ────────────
# All 4 models (manager + 3 subagents) at 0.6B bf16 = ~5 GB weights total.
# mgr_bs=4, num_generations=4 → 16 rollouts/step, good diversity for small model.
python -m src.pipeline.cli train_manager_grpo \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --test_size 1270 \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TID}/sft_evolved" \
    --mgr_full_parameter_rl \
    --mgr_output_dir "outputs/manager/${TID}/grpo_full" \
    --mgr_bs 4 --mgr_num_generations 4 \
    --mgr_max_completion_length 1024 \
    --mgr_temperature 1.0 --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 200 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TID}_grpo_full" \
    --task_description "$TASK"

# ── Step 8: evaluate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --eval_n_samples 1270 --test_size 1270 \
    --eval_manager_dir "outputs/manager/${TID}/grpo_full" \
    --task_description "$TASK"
# Results: outputs/eval/$TID/manager_tool_eval_report.json
```

---

## Experiment B: LegalBench + Qwen3-0.6B (RTX 5090, GPT)

```bash
export TID=gpt_lb_0p6b
export BASE=Qwen/Qwen3-0.6B
export LB_CACHE=outputs/data/legalbench_5tasks.jsonl
export LB_CONFIGS="abercrombie,hearsay,personal_jurisdiction,proa,successor_liability"
export TASK="You are a manager agent solving LegalBench multiple-choice legal classification tasks."
export PYTHONUTF8=1

# ── Step 2: synthesize (100 samples per subagent — LegalBench train set is small) ──
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli synth_subagent \
      --base_model "$BASE" --teacher_id "$TID" \
      --teacher_provider openai --teacher_model gpt-4o-mini \
      --agent_kind "$KIND" --n_samples 100 \
      --legalbench_configs "$LB_CONFIGS" \
      --legalbench_normalized_cache "$LB_CACHE" --legalbench_refresh_cache \
      --train_size 215 --test_size 95 \
      --task_description "$TASK"
done

# ── Step 3: SFT-train subagents ───────────────────────────────────────────────
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE" --teacher_id "$TID" --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 2e-4 \
      --sft_bs 4 --sft_grad_accum 4 --sft_max_seq_len 2048
done

# ── Step 4: validate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" --eval_n_samples 50

# ── Step 5: cold-start ────────────────────────────────────────────────────────
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --train_size 215 --test_size 95 \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 60 \
    --task_description "$TASK"

# ── Step 6: cold-start SFT ────────────────────────────────────────────────────
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_train_jsonl "outputs/manager/${TID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 2e-5 \
    --sft_bs 4 --sft_grad_accum 4 --sft_max_seq_len 2048

# ── Step 7: full-param GRPO ───────────────────────────────────────────────────
# LegalBench: max_steps=80 (fewer train rows than MedQA)
python -m src.pipeline.cli train_manager_grpo \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --train_size 215 --test_size 95 \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TID}/sft_evolved" \
    --mgr_full_parameter_rl \
    --mgr_output_dir "outputs/manager/${TID}/grpo_full" \
    --mgr_bs 4 --mgr_num_generations 4 \
    --mgr_max_completion_length 1024 \
    --mgr_temperature 1.0 --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 80 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TID}_grpo_full" \
    --task_description "$TASK"

# ── Step 8: evaluate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --eval_n_samples 95 --test_size 95 \
    --eval_manager_dir "outputs/manager/${TID}/grpo_full" \
    --task_description "$TASK"
```

---

## Experiment C: MedQA + Qwen3-4B (Cluster, DeepSeek)

Requires Section 3a (DeepSeek MedQA data) to be complete first.

```bash
export TID=ds_medqa_4b
export BASE=Qwen/Qwen3-4B
export CACHE=outputs/data/medqa_us4_normalized.jsonl
export DATA_TID=ds_medqa_data          # where DeepSeek SFT data lives
export TASK="You are a manager agent solving USMLE-style medical multiple-choice questions."
export PYTHONUTF8=1

# ── Step 3: SFT-train subagents (DeepSeek data already generated) ─────────────
# lr=1e-4, bs=2, grad_accum=8, effective_batch=16, seq_len=4096
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE" --teacher_id "$TID" --agent_kind "$KIND" \
      --sft_train_jsonl "outputs/sft_data/${DATA_TID}/${KIND}_sft.jsonl" \
      --sft_epochs 3 --sft_lr 1e-4 \
      --sft_bs 2 --sft_grad_accum 8 --sft_max_seq_len 4096
done

# ── Step 4: validate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" --eval_n_samples 50

# ── Step 5: cold-start ────────────────────────────────────────────────────────
# Loads 3× 4B subagents locally — fine on a single 90 GB GPU (~24 GB total).
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --test_size 1270 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 300 \
    --task_description "$TASK"

# ── Step 6: cold-start SFT ────────────────────────────────────────────────────
# lr=1e-5, bs=2, grad_accum=8, epochs=1
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_train_jsonl "outputs/manager/${TID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 1e-5 \
    --sft_bs 2 --sft_grad_accum 8 --sft_max_seq_len 4096

# ── Step 7: full-param GRPO on 4 GPUs ────────────────────────────────────────
# Terminal A — start vLLM on GPU 0 (keep running):
#   conda activate vllm_env
#   bash scripts/start_subagent_server.sh "$BASE" "$TID"

# Terminal B — GRPO on GPUs 1-2-3:
# mgr_bs=4, num_generations=2 → 4/2=2 ✓, 8 rollouts/step per GPU, 24 total
bash scripts/train_manager_grpo_multigpu.sh "$TID" \
    --base_model "$BASE" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --test_size 1270 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TID}/sft_evolved" \
    --mgr_output_dir "outputs/manager/${TID}/grpo_full" \
    --mgr_bs 4 --mgr_num_generations 2 \
    --mgr_max_completion_length 2048 \
    --mgr_temperature 1.0 --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 200 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TID}_grpo_full" \
    --task_description "$TASK"
# Note: train_manager_grpo_multigpu.sh already appends --mgr_full_parameter_rl
# and --subagent_server_url http://localhost:8000

# ── Step 8: evaluate (single GPU, stop vLLM server first or keep it) ─────────
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --eval_n_samples 1270 --test_size 1270 \
    --eval_manager_dir "outputs/manager/${TID}/grpo_full" \
    --task_description "$TASK"
```

---

## Experiment D: LegalBench + Qwen3-4B (Cluster, DeepSeek)

Requires Section 3b (DeepSeek LegalBench data) to be complete first.

```bash
export TID=ds_lb_4b
export BASE=Qwen/Qwen3-4B
export LB_CACHE=outputs/data/legalbench_5tasks.jsonl
export LB_CONFIGS="abercrombie,hearsay,personal_jurisdiction,proa,successor_liability"
export DATA_TID=ds_lb_data
export TASK="You are a manager agent solving LegalBench multiple-choice legal classification tasks."
export PYTHONUTF8=1

# ── Step 3: SFT-train subagents ───────────────────────────────────────────────
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE" --teacher_id "$TID" --agent_kind "$KIND" \
      --sft_train_jsonl "outputs/sft_data/${DATA_TID}/${KIND}_sft.jsonl" \
      --sft_epochs 3 --sft_lr 1e-4 \
      --sft_bs 2 --sft_grad_accum 8 --sft_max_seq_len 4096
done

# ── Step 4: validate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" --eval_n_samples 50

# ── Step 5: cold-start ────────────────────────────────────────────────────────
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --train_size 215 --test_size 95 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 60 \
    --task_description "$TASK"

# ── Step 6: cold-start SFT ────────────────────────────────────────────────────
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_train_jsonl "outputs/manager/${TID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 1e-5 \
    --sft_bs 2 --sft_grad_accum 8 --sft_max_seq_len 4096

# ── Step 7: full-param GRPO on 4 GPUs ────────────────────────────────────────
# Terminal A: bash scripts/start_subagent_server.sh "$BASE" "$TID"

bash scripts/train_manager_grpo_multigpu.sh "$TID" \
    --base_model "$BASE" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --train_size 215 --test_size 95 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TID}/sft_evolved" \
    --mgr_output_dir "outputs/manager/${TID}/grpo_full" \
    --mgr_bs 4 --mgr_num_generations 2 \
    --mgr_max_completion_length 2048 \
    --mgr_temperature 1.0 --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 80 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TID}_grpo_full" \
    --task_description "$TASK"

# ── Step 8: evaluate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --eval_n_samples 95 --test_size 95 \
    --eval_manager_dir "outputs/manager/${TID}/grpo_full" \
    --task_description "$TASK"
```

---

## Experiment E: MedQA + Qwen3-8B (Cluster, DeepSeek)

Identical structure to C. Only differences: base_model, lr, bs, lora implicit default.

```bash
export TID=ds_medqa_8b
export BASE=Qwen/Qwen3-8B
export CACHE=outputs/data/medqa_us4_normalized.jsonl
export DATA_TID=ds_medqa_data          # same SFT data files as experiment C
export TASK="You are a manager agent solving USMLE-style medical multiple-choice questions."
export PYTHONUTF8=1

# ── Step 3: SFT-train subagents ───────────────────────────────────────────────
# lr=5e-5, bs=1, grad_accum=8, effective_batch=8, seq_len=4096
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE" --teacher_id "$TID" --agent_kind "$KIND" \
      --sft_train_jsonl "outputs/sft_data/${DATA_TID}/${KIND}_sft.jsonl" \
      --sft_epochs 3 --sft_lr 5e-5 \
      --sft_bs 1 --sft_grad_accum 8 --sft_max_seq_len 4096
done

# ── Step 4: validate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" --eval_n_samples 50

# ── Step 5: cold-start ────────────────────────────────────────────────────────
# 3× 8B subagents loaded locally = ~48 GB — fine on a single 90 GB GPU.
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --test_size 1270 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 300 \
    --task_description "$TASK"

# ── Step 6: cold-start SFT ────────────────────────────────────────────────────
# lr=5e-6, bs=1, grad_accum=8, epochs=1
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_train_jsonl "outputs/manager/${TID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 5e-6 \
    --sft_bs 1 --sft_grad_accum 8 --sft_max_seq_len 4096

# ── Step 7: full-param GRPO on 4 GPUs ────────────────────────────────────────
# mgr_bs=2, num_generations=2 → 2/2=1 ✓, conservative for 8B VRAM budget
# Terminal A: bash scripts/start_subagent_server.sh "$BASE" "$TID"

bash scripts/train_manager_grpo_multigpu.sh "$TID" \
    --base_model "$BASE" \
    --medqa_normalized_cache "$CACHE" \
    --train_size 1200 --test_size 1270 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TID}/sft_evolved" \
    --mgr_output_dir "outputs/manager/${TID}/grpo_full" \
    --mgr_bs 2 --mgr_num_generations 2 \
    --mgr_max_completion_length 2048 \
    --mgr_temperature 1.0 --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 200 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TID}_grpo_full" \
    --task_description "$TASK"

# ── Step 8: evaluate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \
    --eval_n_samples 1270 --test_size 1270 \
    --eval_manager_dir "outputs/manager/${TID}/grpo_full" \
    --task_description "$TASK"
```

---

## Experiment F: LegalBench + Qwen3-8B (Cluster, DeepSeek)

```bash
export TID=ds_lb_8b
export BASE=Qwen/Qwen3-8B
export LB_CACHE=outputs/data/legalbench_5tasks.jsonl
export LB_CONFIGS="abercrombie,hearsay,personal_jurisdiction,proa,successor_liability"
export DATA_TID=ds_lb_data             # same SFT data as experiment D
export TASK="You are a manager agent solving LegalBench multiple-choice legal classification tasks."
export PYTHONUTF8=1

# ── Step 3: SFT-train subagents ───────────────────────────────────────────────
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE" --teacher_id "$TID" --agent_kind "$KIND" \
      --sft_train_jsonl "outputs/sft_data/${DATA_TID}/${KIND}_sft.jsonl" \
      --sft_epochs 3 --sft_lr 5e-5 \
      --sft_bs 1 --sft_grad_accum 8 --sft_max_seq_len 4096
done

# ── Step 4: validate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" --eval_n_samples 50

# ── Step 5: cold-start ────────────────────────────────────────────────────────
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --train_size 215 --test_size 95 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 60 \
    --task_description "$TASK"

# ── Step 6: cold-start SFT ────────────────────────────────────────────────────
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_train_jsonl "outputs/manager/${TID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 5e-6 \
    --sft_bs 1 --sft_grad_accum 8 --sft_max_seq_len 4096

# ── Step 7: full-param GRPO on 4 GPUs ────────────────────────────────────────
# Terminal A: bash scripts/start_subagent_server.sh "$BASE" "$TID"

bash scripts/train_manager_grpo_multigpu.sh "$TID" \
    --base_model "$BASE" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --train_size 215 --test_size 95 \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${DATA_TID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TID}/sft_evolved" \
    --mgr_output_dir "outputs/manager/${TID}/grpo_full" \
    --mgr_bs 2 --mgr_num_generations 2 \
    --mgr_max_completion_length 2048 \
    --mgr_temperature 1.0 --mgr_grpo_beta 0.01 \
    --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 80 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TID}_grpo_full" \
    --task_description "$TASK"

# ── Step 8: evaluate ──────────────────────────────────────────────────────────
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE" --teacher_id "$TID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache "$LB_CACHE" \
    --eval_n_samples 95 --test_size 95 \
    --eval_manager_dir "outputs/manager/${TID}/grpo_full" \
    --task_description "$TASK"
```

---

## 4. Optional: Evolve Round (any experiment)

After Step 7 GRPO, if you want to iterate with failure-driven SFT:

```bash
# Build SFT from GRPO failures
python -m src.pipeline.cli evolve_build_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --medqa_normalized_cache "$CACHE" \        # or --legalbench_* for LegalBench
    --fail_buffer_jsonl "outputs/manager/${TID}/grpo_full/fail_buffer.jsonl" \
    --task_description "$TASK"

# SFT manager on failure trajectories
python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE" --teacher_id "$TID" \
    --manager_sft_epochs 1 --manager_sft_lr 5e-6   # halve lr from cold-start
    # (uses outputs/manager/$TID/evolve/manager_sft_from_failures.jsonl by default)

# Then re-run Step 7 GRPO from the evolved adapter (--mgr_init_adapter outputs/manager/$TID/sft_evolved)
```

---

## 5. Monitoring During GRPO

Watch these metrics in W&B (or terminal logs):

| Metric | Healthy | Problem |
|--------|---------|---------|
| `tools/call_frequency` | > 0.3 after first 20 steps | = 0.0 → cold-start not loaded |
| `reward` / `rewards/binary_outcome_with_format/mean` | climbing from ~0.3 | stuck at 0 → all rollouts wrong |
| `frac_reward_zero_std` | < 0.5 | = 1.0 → all rollouts same reward, no learning signal |
| `tools/failure_frequency` | < 0.1 | high → subagent server down or format broken |

---

## 6. Results Collection

```bash
# Print a summary table across all experiments
for TID in gpt_medqa_0p6b gpt_lb_0p6b ds_medqa_4b ds_lb_4b ds_medqa_8b ds_lb_8b; do
  echo "=== $TID ==="
  python -c "
import json, sys
path = f'outputs/eval/$TID/manager_tool_eval_report.json'
try:
    r = json.load(open(path))
    print(f'  accuracy:       {r[\"accuracy\"]:.3f}')
    print(f'  tool_call_rate: {r[\"tool_call_rate\"]:.3f}')
    print(f'  avg_tool_calls: {r[\"avg_tool_calls\"]:.2f}')
except FileNotFoundError:
    print('  not yet run')
"
done
```

---

## 7. Key Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Subagent SFT on same data for 4B and 8B | Yes | SFT data is task-level, not model-level. Saves generation cost. |
| cold-start before every GRPO | Yes | Without it, `tools/call_frequency=0` — GRPO trains direct-answer only. |
| `--mgr_full_parameter_rl` always | Yes | LoRA GRPO converges slower for instruction-following; full-param is better for learning routing. |
| `mgr_num_generations=4` for 0.6B, 2 for 4B/8B | Memory trade-off | 0.6B is cheap; 4B/8B use KV cache aggressively at 2048 tokens. |
| `max_steps=80` for LegalBench | Fewer train rows | ~115 manager examples after exclusions; 200 steps would be 15+ passes = overfit. |
| DeepSeek offline for cluster | Avoids API latency | Cluster jobs run unattended; online synthesis blocks on API rate limits. |
| GPT-4o-mini online for 5090 | Convenience | Local runs are interactive; API calls are fast enough. |
