# Agent Routing — Metacognitive Routing with Calibrated Confidence

A pipeline for training a manager LLM that learns **when to consult which specialized expert**. The manager decides whether to call 0–3 frozen subagents (extractor / reasoner / rule_applier) before answering a multiple-choice question. Routing is trained with GRPO using a **Calibrated Confidence Routing (CCR)** reward derived from the logarithmic scoring rule — the manager is rewarded not just for being correct but for expressing calibrated uncertainty through its routing choices.

Benchmarks supported: **MedQA-USMLE**, **LegalBench**, **MMLU-Pro**, **GPQA**.

```
                       ┌────────────────────────┐
                       │   Manager (Qwen3)      │
                       │   GRPO + CCR reward    │
                       └────┬────┬───────┬──────┘
                            │    │       │
              ┌─────────────┘    │       └──────────────┐
              ▼                  ▼                      ▼
   ┌──────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
   │  ExtractorAgent  │ │  ReasonerAgent  │ │  RuleApplierAgent    │
   │  (frozen, LoRA)  │ │ (frozen, LoRA)  │ │  (frozen, LoRA)      │
   └──────────────────┘ └─────────────────┘ └──────────────────────┘
            │                    │                     │
            └────────────────────┼─────────────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │   Teacher (Claude / GPT /     │
                 │   DeepSeek) — generates SFT   │
                 │   data for each subagent      │
                 └───────────────────────────────┘
```

**Hard invariant**: subagents never produce the final answer. The manager is the sole authority on the `ANSWER_<TOKEN>` final line. This is enforced by pydantic schemas + a leakage auditor at synthesis time.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set whichever teacher key(s) you'll use:

```bash
export ANTHROPIC_API_KEY=...    # for Claude
export OPENAI_API_KEY=...       # for GPT
export DEEPSEEK_API_KEY=...     # for DeepSeek
```

---

## Data budget and recommended splits

Each benchmark follows the same split structure. The numbers below are the recommended settings for reproducible paper results.

### Why these numbers

| Pool | MedQA | LegalBench (5 tasks) | MMLU-Pro | GPQA-Diamond |
|---|---|---|---|---|
| Total available | ~12 k | ~800–1 500 | ~12 k test | 198 |
| Subagent SFT (per agent) | **500** | **150** | eval only | eval only |
| Manager cold-start SFT | **300** | **80** | eval only | eval only |
| Manager GRPO train | **600** | **200** | eval only | eval only |
| Dev (sanity check) | 200 | 100 | — | — |
| **Test / Eval** | **500** | **100** | **500** | **198 (all)** |

**MMLU-Pro** — The community uses the full `test` split for comparison with other papers; training on any part of it breaks comparability. Use it as a zero-shot generalisation probe only.

**GPQA-Diamond** — Only 198 questions exist; too small and too domain-specific for training. Use all 198 as a hard-case evaluation.

> **Paper claim enabled by this split**: "A routing policy trained on domain X generalises to domains Y and Z without retraining." MedQA/LegalBench train routing; MMLU-Pro/GPQA test whether calibrated uncertainty is domain-agnostic.

### Data overlap rules (critical for fair evaluation)

The three subagent SFT pools, the cold-start pool, and the GRPO pool must be **disjoint** by `example_id`. The `--exclude_sft_example_ids` flag enforces this — pass all three subagent SFT JSONL paths and the GRPO trainer will skip any row whose `example_id` appears in those files.

---

## Synthetic data and CCR: are they in conflict?

No. They operate on completely different parts of the system and solve complementary problems.

```
Stage                │  Data source        │  What it teaches
─────────────────────┼─────────────────────┼────────────────────────────────────
Subagent SFT         │  Synthetic (teacher)│  WHAT to output (JSON schema, facts)
Manager cold-start   │  Synthetic (teacher)│  HOW to format tool calls (format)
Manager GRPO + CCR   │  Outcome (GT label) │  WHEN to call tools (calibration)
```

SFT can replicate the teacher's routing choices, but the teacher's implicit confidence is not observable and cannot be learned from demonstrations alone. CCR fills exactly the gap SFT cannot: it uses the binary outcome signal (correct/incorrect) to shape a calibrated routing policy through RL.

> **One-liner for the paper**: "SFT initialises tool-calling capability; CCR-GRPO optimises calibration through outcome supervision — they are sequential, not competing."

---

## CCR reward explained

When `--mgr_ccr_mode` is set, the binary 0/1 reward is replaced by the **logarithmic scoring rule** applied to the routing decision's implicit confidence:

```
p(k) = p_high + (p_low − p_high) × k / k_max   # linear from confident → uncertain
R    = log p(k)        if correct
R    = log(1 − p(k))   if incorrect
```

With defaults `p_high=0.9`, `p_low=0.2`, `k_max=3`:

| k | Correct | Incorrect | Interpretation |
|---|---|---|---|
| 0 | −0.11 **(best)** | −2.30 **(worst)** | High-stakes: efficient if right, harshly penalised if overconfident |
| 1 | −0.51 | −1.61 | Moderate confidence |
| 2 | −0.92 | −1.20 | Low confidence |
| 3 | −1.61 | −0.22 | Lowest confidence: safe floor when genuinely uncertain |

GRPO normalises rewards within a group (same question, multiple rollouts), so the absolute scale does not matter. What matters is the within-group ordering, which encodes: *"for easy questions, call 0 tools; for hard questions, call more tools."*

---

## Post-training analysis: ECE and routing entropy

After training, compute two calibration metrics from the `train_raw_trace.jsonl` file:

```python
from src.utils.io import read_jsonl
from src.manager.reward import compute_ece, compute_routing_entropy

records = read_jsonl("outputs/manager/<teacher_id>/grpo/train_raw_trace.jsonl")

# Expected Calibration Error: compares p(k) to empirical accuracy per k-bucket
print(compute_ece(records, p_high=0.9, p_low=0.2))
# → {"ece": 0.08, "buckets": {0: {"accuracy": 0.87, "implicit_confidence": 0.90, ...}, ...}}

# Routing entropy: H over the empirical tool-call distribution
print(compute_routing_entropy(records))
# → {"entropy": 0.94, "normalized_entropy": 0.60, "distribution": {"0": 0.45, "1": 0.22, ...}}
```

**Interpreting routing entropy by benchmark** (expected after CCR training):

| Benchmark | Expected normalized entropy | Interpretation |
|---|---|---|
| MedQA (4-choice) | 0.4–0.6 | Medium difficulty; manager confident on ~40–50% |
| LegalBench | 0.3–0.5 | Varies by task; rule-based tasks are easier to route |
| MMLU-Pro (10-choice) | 0.6–0.8 | Higher entropy: more genuine uncertainty |
| GPQA-Diamond | 0.8–1.0 | Expert-level: manager should call tools on almost everything |

---

## Multi-GPU Full-Parameter GRPO (4× H100 / H200)

For 8B+ base models, full-parameter GRPO requires ZeRO Stage 3 and a dedicated GPU for subagent inference:

```
GPU 0  →  vLLM server  (3 subagents, shared base + 3 LoRA adapters, ~27 GB)
GPU 1  ┐
GPU 2  ├→ accelerate + DeepSpeed ZeRO Stage 3  (manager training, ~52 GB/GPU)
GPU 3  ┘
```

### Step 0: install dependencies

```bash
pip install accelerate deepspeed

conda create -n vllm_env python=3.11 -y
conda activate vllm_env
pip install vllm
```

### Step 1: start the subagent vLLM server (GPU 0)

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh Qwen/Qwen3-8B <teacher_id>
```

Verify:
```bash
curl http://localhost:8000/health       # → {"status":"ok"}
curl http://localhost:8000/v1/models | python -m json.tool  # lists extractor, reasoner, rule_applier
```

### Step 2: run GRPO on GPUs 1-2-3

```bash
source .venv/bin/activate
bash scripts/train_manager_grpo_multigpu.sh <teacher_id> \
    --base_model Qwen/Qwen3-8B \
    --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
    --train_size 600 \
    --mgr_ccr_mode --mgr_ccr_p_high 0.9 --mgr_ccr_p_low 0.2 \
    ...
```

### Memory budget (8B, bf16, 4× 90–100 GB cards)

| GPU | Role | VRAM |
|-----|------|------|
| 0 | vLLM: base (16 GB) + 3 adapters + KV cache | ~27 GB |
| 1–3 | ZeRO3 shard | ~52 GB each |

---

## End-to-end on MedQA

```bash
# ── configure once ─────────────────────────────────────────────────────────
export TEACHER_ID=claude_sonnet_4_5
export PROVIDER=anthropic
export MODEL=claude-sonnet-4-5
export BASE_MODEL=Qwen/Qwen3-8B
export TASK_DESC="You are a manager agent solving USMLE-style medical multiple-choice questions."
export PYTHONUTF8=1
# ──────────────────────────────────────────────────────────────────────────

# Step 1 — cache MedQA (idempotent)
python -m src.pipeline.cli load_medqa \
    --base_model "$BASE_MODEL" \
    --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
    --train_size 1400 --dev_size 200 --test_size 500

# Step 2 — synthesize subagent SFT data (500 examples each)
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli synth_subagent \
      --base_model "$BASE_MODEL" \
      --teacher_id "$TEACHER_ID" \
      --teacher_provider "$PROVIDER" --teacher_model "$MODEL" \
      --agent_kind "$KIND" --n_samples 500 \
      --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
      --train_size 1400 --dev_size 200 --test_size 500 \
      --task_description "$TASK_DESC"
done

# Step 3 — SFT-train the three subagents
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE_MODEL" \
      --teacher_id "$TEACHER_ID" --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 2e-4
done

# Step 4 — validate subagents (json_ok_rate and schema_ok_rate should be > 0.9)
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
    --eval_n_samples 50

# Step 5 — build and train manager cold-start SFT (300 examples)
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
    --train_size 1400 --dev_size 200 --test_size 500 \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 300 \
    --task_description "$TASK_DESC"

python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --manager_sft_train_jsonl "outputs/manager/${TEACHER_ID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 2e-5

# Step 6 — GRPO with CCR reward (600 train rows, non-overlapping with SFT data)
# Terminal A — vLLM server:  bash scripts/start_subagent_server.sh "$BASE_MODEL" "$TEACHER_ID"
# Terminal B — training:
bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" \
    --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
    --train_size 600 --dev_size 200 --test_size 500 \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --mgr_ccr_mode --mgr_ccr_p_high 0.9 --mgr_ccr_p_low 0.2 \
    --mgr_bs 2 --mgr_num_generations 6 \
    --mgr_max_steps 300 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_medqa_ccr" \
    --task_description "$TASK_DESC"

# Step 7 — evaluate on 500-example held-out test set
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --medqa_normalized_cache outputs/data/medqa_normalized.jsonl \
    --eval_n_samples 500 --test_size 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --task_description "$TASK_DESC"
```

---

## End-to-end on LegalBench

Uses 5 tasks with small fixed label sets. Total row count is modest (~800–1 500), so all split sizes are scaled down proportionally.

```bash
export TEACHER_ID=legalbench_claude
export PROVIDER=anthropic
export MODEL=claude-sonnet-4-5
export BASE_MODEL=Qwen/Qwen3-8B
export LB_CONFIGS="abercrombie,hearsay,personal_jurisdiction,proa,successor_liability"
export TASK_DESC="You are a manager agent solving LegalBench multiple-choice legal classification tasks."
export PYTHONUTF8=1

# Step 1 — cache LegalBench (idempotent after first run, drop --legalbench_refresh_cache)
# Rows are loaded on-demand; pass --legalbench_normalized_cache to persist them.

# Step 2 — synthesize subagent SFT data (150 examples each)
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli synth_subagent \
      --base_model "$BASE_MODEL" \
      --teacher_id "$TEACHER_ID" \
      --teacher_provider "$PROVIDER" --teacher_model "$MODEL" \
      --agent_kind "$KIND" --n_samples 150 \
      --legalbench_configs "$LB_CONFIGS" \
      --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
      --legalbench_refresh_cache \
      --train_size 530 --dev_size 100 --test_size 100 \
      --task_description "$TASK_DESC"
done
# After first run drop --legalbench_refresh_cache to reuse the cached file.

# Step 3 — SFT subagents
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --base_model "$BASE_MODEL" \
      --teacher_id "$TEACHER_ID" --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 2e-4
done

# Step 4 — validate subagents
python -m src.pipeline.cli eval_subagents \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
    --eval_n_samples 50

# Step 5 — cold-start manager (80 examples)
python -m src.pipeline.cli manager_coldstart_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
    --train_size 530 --dev_size 100 --test_size 100 \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/rule_applier_sft.jsonl" \
    --coldstart_n_samples 80 \
    --task_description "$TASK_DESC"

python -m src.pipeline.cli train_manager_sft \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --manager_sft_train_jsonl "outputs/manager/${TEACHER_ID}/evolve/manager_sft_coldstart.jsonl" \
    --manager_sft_epochs 1 --manager_sft_lr 2e-5

# Step 6 — GRPO with CCR (200 train rows)
bash scripts/train_manager_grpo_multigpu.sh "$TEACHER_ID" \
    --base_model "$BASE_MODEL" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
    --train_size 200 --dev_size 100 --test_size 100 \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/extractor_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/reasoner_sft.jsonl" \
    --exclude_sft_example_ids "outputs/sft_data/${TEACHER_ID}/rule_applier_sft.jsonl" \
    --mgr_init_adapter "outputs/manager/${TEACHER_ID}/sft_coldstart" \
    --mgr_output_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --mgr_ccr_mode --mgr_ccr_p_high 0.9 --mgr_ccr_p_low 0.2 \
    --mgr_bs 2 --mgr_num_generations 6 \
    --mgr_max_steps 100 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name "${TEACHER_ID}_legalbench_ccr" \
    --task_description "$TASK_DESC"

# Step 7 — evaluate on 100-example held-out test set
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --legalbench_configs "$LB_CONFIGS" \
    --legalbench_normalized_cache outputs/data/legalbench_5tasks.jsonl \
    --eval_n_samples 100 --test_size 100 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --task_description "$TASK_DESC"
```

---

## MMLU-Pro: evaluation only (zero-shot generalisation)

MMLU-Pro is used **as an evaluation benchmark only**. Training on any portion of its test set breaks comparability with the literature. The goal is to show that a CCR routing policy trained on MedQA or LegalBench generalises — without any MMLU-Pro training data — to a 10-option broad-knowledge benchmark.

```bash
export BASE_MODEL=Qwen/Qwen3-8B
export TEACHER_ID=claude_sonnet_4_5       # reuse the MedQA-trained manager
export TASK_DESC="You are a manager agent solving multiple-choice questions across diverse academic subjects."
export PYTHONUTF8=1

# Step 1 — download and cache MMLU-Pro test split (500 examples, stratified)
python -m src.pipeline.cli load_mmlu_pro \
    --base_model "$BASE_MODEL" \
    --mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl \
    --mmlu_pro_splits "test" \
    --mmlu_pro_max 500 \
    --test_size 500 --train_size 0 --dev_size 0

# Step 2 — evaluate the MedQA-trained manager on MMLU-Pro (zero-shot transfer)
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --mmlu_pro_normalized_cache outputs/data/mmlu_pro_normalized.jsonl \
    --eval_n_samples 500 --test_size 500 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --task_description "$TASK_DESC"

# Results:
#   outputs/eval/$TEACHER_ID/manager_tool_eval_report.json  ← accuracy, avg_tool_calls
#   outputs/eval/$TEACHER_ID/manager_tool_eval.jsonl        ← per-example detail
```

**What to look for in the results:**

```python
from src.utils.io import read_jsonl
from src.manager.reward import compute_ece, compute_routing_entropy

records = read_jsonl("outputs/eval/<teacher_id>/manager_tool_eval.jsonl")
# eval_manager_tools logs correct/tool_calls per example — compute calibration metrics
ece = compute_ece(records, p_high=0.9, p_low=0.2)
ent = compute_routing_entropy(records)
print(f"ECE = {ece['ece']:.3f}")
print(f"Routing entropy (normalized) = {ent['normalized_entropy']:.3f}")
# Expected: higher entropy than MedQA (10-option questions are harder)
```

> **If you want a per-category breakdown**, filter `records` by `task_subtype` (which contains the MMLU-Pro category name) before calling `compute_ece`.

---

## GPQA-Diamond: expert-level evaluation

GPQA-Diamond has only 198 questions and is used as a hard-case evaluation only. The dataset is **gated on HuggingFace** — you must accept the terms and log in first.

```bash
# One-time setup
huggingface-cli login
# Accept terms at: https://huggingface.co/datasets/Idavidrein/gpqa
```

```bash
export BASE_MODEL=Qwen/Qwen3-8B
export TEACHER_ID=claude_sonnet_4_5       # reuse the MedQA-trained manager
export TASK_DESC="You are a manager agent solving expert-level science multiple-choice questions."
export PYTHONUTF8=1

# Step 1 — download and cache GPQA-Diamond (all 198 examples)
python -m src.pipeline.cli load_gpqa \
    --base_model "$BASE_MODEL" \
    --gpqa_subsets gpqa_diamond \
    --gpqa_normalized_cache outputs/data/gpqa_diamond_normalized.jsonl \
    --test_size 198 --train_size 0 --dev_size 0

# Step 2 — evaluate on all 198 questions
python -m src.pipeline.cli eval_manager_tools \
    --base_model "$BASE_MODEL" \
    --teacher_id "$TEACHER_ID" \
    --gpqa_normalized_cache outputs/data/gpqa_diamond_normalized.jsonl \
    --eval_n_samples 198 --test_size 198 \
    --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
    --task_description "$TASK_DESC"
```

**Expected behavior after CCR training:**
- `avg_tool_calls` should be higher on GPQA-Diamond than on any training benchmark — the manager should have learned that expert-level questions require maximum consultation.
- `routing entropy` should be close to 1.0 (maximum, normalized).
- Accuracy will still be modest (GPQA-Diamond is designed to defeat LLMs), but the routing calibration story is the point.

```bash
# Optional: run all three subsets and compare routing entropy
for SUBSET in gpqa_diamond gpqa_main gpqa_extended; do
  python -m src.pipeline.cli load_gpqa \
      --base_model "$BASE_MODEL" \
      --gpqa_subsets "$SUBSET" \
      --gpqa_normalized_cache "outputs/data/${SUBSET}_normalized.jsonl" \
      --test_size 500 --train_size 0 --dev_size 0
  python -m src.pipeline.cli eval_manager_tools \
      --base_model "$BASE_MODEL" \
      --teacher_id "$TEACHER_ID" \
      --gpqa_normalized_cache "outputs/data/${SUBSET}_normalized.jsonl" \
      --eval_n_samples 500 --test_size 500 \
      --eval_manager_dir "outputs/manager/${TEACHER_ID}/grpo_ccr" \
      --task_description "$TASK_DESC"
done
# Diamond should show the highest routing entropy of the three.
```

---

## Notes on the cold-start step

Cold-start (the `manager_coldstart_sft` + `train_manager_sft` steps) is critical before GRPO. Without it the manager never discovers native tool calls:

```
tools/call_frequency: 0
reward: 0 (or all-negative for CCR mode)
frac_reward_zero_std: 1
```

If you see those after starting GRPO, the cold-start adapter was not loaded. Verify `--mgr_init_adapter` points to a directory containing `adapter_config.json` (LoRA) or `config.json` (full model).

---

## Comparing teachers

Run the full MedQA pipeline three times with different teachers. The `--teacher_id` flag namespaces all outputs so runs never overwrite each other.

| teacher_id | provider | model |
|---|---|---|
| `claude_sonnet_4_5` | anthropic | `claude-sonnet-4-5` |
| `gpt_4o` | openai | `gpt-4o-2024-08-06` |
| `deepseek_v4` | deepseek | `deepseek-chat` |

**Three comparison layers:**

```bash
# Layer 1 — synthesis quality (before any training)
ls outputs/sft_data/<teacher_id>/*_synth_log.jsonl
# Look for: json_parse_fail, schema_fail, leakage_fail rates

# Layer 2 — subagent reliability (after subagent SFT)
cat outputs/eval/<teacher_id>/subagent_eval_report.json
# Look for: json_ok_rate, schema_ok_rate per subagent (want > 0.9)

# Layer 3 — manager accuracy + calibration (final)
cat outputs/eval/<teacher_id>/manager_tool_eval_report.json
# Look for: accuracy, avg_tool_calls, tool_call_rate
```

---

## Output layout

```
outputs/
├── data/
│   ├── medqa_normalized.jsonl
│   ├── legalbench_5tasks.jsonl
│   ├── mmlu_pro_normalized.jsonl
│   └── gpqa_diamond_normalized.jsonl
├── teacher_cache/<teacher_slug>/           # disk-cached teacher API calls
├── sft_data/<teacher_slug>/
│   ├── extractor_sft.jsonl                 # SFT input (prompt + response pairs)
│   ├── extractor_synth_log.jsonl           # per-attempt failure log
│   ├── reasoner_sft.jsonl
│   └── rule_applier_sft.jsonl
├── adapters/<teacher_slug>/
│   ├── extractor_adapter/
│   ├── reasoner_adapter/
│   └── rule_applier_adapter/
├── manager/<teacher_slug>/
│   ├── sft_coldstart/                      # cold-start adapter
│   ├── grpo_ccr/
│   │   ├── fail_buffer.jsonl               # GRPO failures → evolve input
│   │   ├── train_raw_trace.jsonl           # per-completion trace (correct, tool_calls, implicit_confidence)
│   │   └── manager_run_config.json         # includes ccr_mode, p_high, p_low
│   └── evolve/
│       └── manager_sft_from_failures.jsonl
└── eval/<teacher_slug>/
    ├── manager_tool_eval_report.json       # accuracy, avg_tool_calls, calibration metrics
    └── manager_tool_eval.jsonl
```

---

## Available CLI stages

| Stage | What it does |
|---|---|
| `load_medqa` | Download and cache MedQA from HuggingFace |
| `load_gpqa` | Download and cache GPQA subsets (gated — requires HF login) |
| `load_mmlu_pro` | Download and cache MMLU-Pro |
| `export_legalbench_jsonl` | Export LegalBench rows as DeepSeek-style prompt JSONL |
| `synth_subagent` | Generate subagent SFT data via a teacher LLM |
| `export_deepseek_jsonl` | Export synthesis prompts for local DeepSeek inference |
| `import_deepseek_jsonl` | Import and validate local DeepSeek responses |
| `train_subagent` | LoRA-SFT one subagent on its synthesised data |
| `manager_coldstart_sft` | Build + train manager cold-start SFT from teacher demonstrations |
| `train_manager_grpo` | GRPO-train the manager (supports `--mgr_ccr_mode`) |
| `evolve_build_sft` | Build targeted SFT from GRPO fail buffer |
| `train_manager_sft` | LoRA-SFT the manager on evolve or coldstart trajectories |
| `evolve_round` | One full round: GRPO → evolve build → manager SFT |
| `eval_subagents` | Score JSON validity and schema validity per subagent |
| `eval_manager` | Score manager accuracy (no tools, greedy) |
| `eval_manager_tools` | Score manager accuracy with the full tool-calling loop |

Run `python -m src.pipeline.cli --help` for all flags.

---

## Key design decisions

**Why frozen subagents?**
In MAS literature (Stronger-MAS, MetaAgent-X), all agents are trained jointly. Here, subagents are frozen by design: we want to study the routing decision in isolation, not how to train the experts. This is the "consultation" paradigm rather than "collaboration" — the manager consults frozen specialists and takes sole responsibility for the answer.

**Why CCR instead of binary reward?**
Binary reward (0/1) treats "confident and wrong" the same as "uncertain and wrong." CCR makes overconfidence explicitly more costly: a manager that calls 0 tools and gets it wrong receives the worst possible reward, while a manager that called all 3 tools and still got it wrong receives a mild penalty (it was appropriately uncertain). This asymmetry is exactly what the logarithmic scoring rule guarantees.

**Why multiple SFT steps?**
Each SFT step solves a different problem:
- **Subagent SFT**: bootstraps specialist knowledge from a teacher model
- **Cold-start SFT**: teaches the manager the tool-calling interface before RL exploration
- **Evolve SFT**: targeted recovery on hard cases that RL cannot reach efficiently

None of these steps teach calibration — that is the exclusive role of CCR-GRPO.

**GT visibility per subagent**
- **Extractor** — GT hidden from teacher. Extraction should be objective; showing GT biases the teacher to omit counter-evidence.
- **Reasoner / RuleApplier** — GT shown as `PRIVATE_GT` but disclosure is forbidden and audited. These tasks are hard enough that reverse-construction from a known answer is more reliable than unconstrained teacher generation.

**Four synthesis quality gates**
Every teacher response must pass: JSON-parseable → Pydantic schema valid → balance check (Reasoner only, all choices covered) → leakage audit (no GT label in output). Failures are retried up to 2× with bumped temperature, then dropped and logged to `<kind>_synth_log.jsonl`.

---

## Troubleshooting

**`huggingface-cli login` fails for GPQA**
GPQA requires explicitly accepting terms at `https://huggingface.co/datasets/Idavidrein/gpqa`. Log in on the web first, accept the form, then `huggingface-cli login` with a token that has `read` permission.

**MMLU-Pro `validation` split is only 70 rows**
This is expected. The MMLU-Pro `test` split (~12k rows) contains ground-truth answers and is what the paper community evaluates on. We load from `test` by default; pass `--mmlu_pro_splits test` explicitly.

**GRPO `tools/call_frequency = 0` from the start**
Cold-start adapter not loaded. See the "Notes on the cold-start step" section.

**CCR rewards are all negative**
This is expected — log scoring rule rewards are always negative. GRPO normalises within group, so the absolute scale is irrelevant. What matters is the relative ordering within each group.

**`leakage_fail` rate > 20%**
The teacher keeps disclosing the answer. Check `<kind>_synth_log.jsonl` to see the actual matches. Try lowering `--synth_temperature` or switching to a stricter teacher model.

**`schema_ok_rate < 0.7`**
Subagent didn't internalise the schema. Try `--sft_epochs 5` or increase `--n_samples`.

**OOM during GRPO**
Lower `--mgr_bs` to 1, `--mgr_num_generations` to 4, `--mgr_max_completion_length` to 1024. For 8B models, use the multi-GPU setup with a vLLM server.

---

## License & citation

Internal research code. If you build on this, please cite the underlying base model (Qwen3) and TRL appropriately.
