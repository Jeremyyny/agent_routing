# Agent Routing — Manager + Three Subagents on MedQA

A pipeline for training a small manager LLM (Qwen3-0.6B by default) that learns to route between three SFT-trained subagents to solve MedQA. Subagents are trained on data synthesized by a **teacher model** (Claude, GPT, or DeepSeek), then the manager is trained with GRPO using the trained subagents as native tools, with an evolve loop that recycles GRPO failures into manager SFT data.

The whole thing is built for one specific experiment: **comparing how teacher choice (Claude vs GPT vs DeepSeek) affects every layer of the resulting system** — synthesis quality, subagent reliability, and final manager accuracy.

---

## What it builds

```
                       ┌────────────────────────┐
                       │   Manager (Qwen3-0.6B) │
                       │   GRPO + Evolve SFT    │
                       └────┬────┬───────┬──────┘
                            │    │       │
              ┌─────────────┘    │       └──────────────┐
              ▼                  ▼                      ▼
   ┌──────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
   │  ExtractorAgent  │ │  ReasonerAgent  │ │  RuleApplierAgent    │
   │  (frozen, LoRA)  │ │ (frozen, LoRA)  │ │  (frozen, LoRA)      │
   └────────┬─────────┘ └────────┬────────┘ └──────────┬───────────┘  
            │                    │                     │
            └────────────────────┼─────────────────────┘
                                 │
                 ┌───────────────▼───────────────┐
                 │   Teacher (Claude / GPT /     │
                 │   DeepSeek) — generates SFT   │
                 │   data for each subagent      │
                 └───────────────────────────────┘
```

Three subagents, all schema-constrained JSON output:

- **ExtractorAgent** — pulls clinical/factual signals from the question stem (and context, when applicable).
- **ReasonerAgent** — produces a structured reasoning scaffold: sub-questions, required knowledge, per-choice support/against analysis.
- **RuleApplierAgent** — identifies applicable medical decision rules/criteria and maps facts to their elements.

**Hard invariant**: subagents NEVER produce the final answer. The manager is the sole authority on the `ANSWER_<TOKEN>` final line. This is enforced by pydantic schemas + a leakage auditor at synthesis time.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set whichever teacher key(s) you'll use:

```bash
export ANTHROPIC_API_KEY=...    # for Claude
export OPENAI_API_KEY=...       # for GPT
export DEEPSEEK_API_KEY=...     # for DeepSeek
```

You only need the key for the teacher you're currently using — pipelines are run one teacher at a time.

---

## Multi-GPU Full-Parameter GRPO (4× H100 / H200)

For 8B+ base models, full-parameter GRPO requires ZeRO Stage 3 and a dedicated
GPU for subagent inference. The key problem: with standard DDP, every training
process loads the full manager model **and** all three subagents — far too much
VRAM. The solution separates the two concerns:

```
GPU 0  →  vLLM server  (3 subagents, shared base + 3 LoRA adapters, ~27 GB)
GPU 1  ┐
GPU 2  ├→ accelerate + DeepSpeed ZeRO Stage 3  (manager training, ~52 GB/GPU)
GPU 3  ┘
```

### Step 0: Install dependencies in two separate environments

```bash
# Training environment (.venv already has TRL/accelerate/transformers)
pip install accelerate deepspeed

# Separate vLLM environment (avoids version conflicts with TRL)
conda create -n vllm_env python=3.11 -y
conda activate vllm_env
pip install vllm
```

### Step 1: Start the subagent vLLM server on GPU 0

Run in a **separate terminal** (keep it alive for the duration of training):

```bash
conda activate vllm_env
bash scripts/start_subagent_server.sh Qwen/Qwen3-8B openai_us4_500_runtime_raw
```

This loads the base model once (~16 GB) and registers three LoRA adapters (~1 GB
total). Verify it is ready:

```bash
curl http://localhost:8000/health
# → {"status":"ok"}

curl http://localhost:8000/v1/models | python -m json.tool
# should list: extractor, reasoner, rule_applier
```

### Step 2: Run full-parameter GRPO on GPUs 1-2-3

In your training environment:

```bash
source .venv/bin/activate
export PYTHONUTF8=1

bash scripts/train_manager_grpo_multigpu.sh openai_us4_500_runtime_raw \
    --base_model Qwen/Qwen3-8B \
    --medqa_normalized_cache outputs/data/medqa_us4_normalized.jsonl \
    --train_size 1200 \
    --exclude_sft_example_ids outputs/sft_data/openai_us4_500/extractor_runtime_raw_sft.jsonl \
    --exclude_sft_example_ids outputs/sft_data/openai_us4_500/reasoner_runtime_raw_sft.jsonl \
    --exclude_sft_example_ids outputs/sft_data/openai_us4_500/rule_applier_runtime_raw_sft.jsonl \
    --mgr_init_adapter outputs/manager/openai_us4_500_runtime_raw/sft_evolved \
    --mgr_output_dir outputs/manager/openai_us4_500_runtime_raw/grpo_full \
    --mgr_max_steps 200 \
    --mgr_use_wandb --wandb_project agent_routing \
    --wandb_run_name openai_us4_500_runtime_raw_grpo_full_8b \
    --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

The script polls `http://localhost:8000/health` for up to 120 s before launching
training. If it times out, check that `start_subagent_server.sh` is still running.

### Manual equivalent (without the shell scripts)

Terminal 1 — vLLM server:

```bash
conda activate vllm_env
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --enable-lora --max-lora-rank 64 \
    --lora-modules \
        extractor=outputs/adapters/openai_us4_500_runtime_raw/extractor_adapter \
        reasoner=outputs/adapters/openai_us4_500_runtime_raw/reasoner_adapter \
        rule_applier=outputs/adapters/openai_us4_500_runtime_raw/rule_applier_adapter \
    --port 8000 --dtype bfloat16 --trust-remote-code --max-model-len 4096
```

Terminal 2 — training:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1,2,3 PYTHONUTF8=1 \
accelerate launch --config_file configs/accelerate_zero3.yaml \
    -m src.pipeline.cli train_manager_grpo \
    --base_model Qwen/Qwen3-8B \
    --teacher_id openai_us4_500_runtime_raw \
    --subagent_server_url http://localhost:8000 \
    --mgr_full_parameter_rl \
    --mgr_init_adapter outputs/manager/openai_us4_500_runtime_raw/sft_evolved \
    --mgr_output_dir outputs/manager/openai_us4_500_runtime_raw/grpo_full \
    --medqa_normalized_cache outputs/data/medqa_us4_normalized.jsonl \
    --train_size 1200 \
    --exclude_sft_example_ids outputs/sft_data/openai_us4_500/extractor_runtime_raw_sft.jsonl \
    --exclude_sft_example_ids outputs/sft_data/openai_us4_500/reasoner_runtime_raw_sft.jsonl \
    --exclude_sft_example_ids outputs/sft_data/openai_us4_500/rule_applier_runtime_raw_sft.jsonl \
    --mgr_bs 2 --mgr_num_generations 2 \
    --mgr_max_completion_length 2048 --mgr_temperature 1.0 \
    --mgr_grpo_beta 0.01 --mgr_tool_use_bonus 0.2 \
    --mgr_max_steps 200 \
    --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

### Evaluation after multi-GPU GRPO

Evaluation loads subagents locally (single GPU, no vLLM needed):

```bash
python -X utf8 -m src.pipeline.cli eval_manager_tools \
    --base_model Qwen/Qwen3-8B \
    --teacher_id openai_us4_500_runtime_raw \
    --medqa_normalized_cache outputs/data/medqa_us4_normalized.jsonl \
    --eval_n_samples 1270 --test_size 1270 \
    --eval_manager_dir outputs/manager/openai_us4_500_runtime_raw/grpo_full \
    --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

### Memory budget (8B model, bf16, 4× 90–100 GB cards)

| GPU | Role | VRAM used |
|-----|------|-----------|
| 0 | vLLM: base (16 GB) + 3 adapters + KV cache | ~27 GB |
| 1 | ZeRO3 shard: params+grad+Adam+ref (112 GB ÷ 3) | ~52 GB |
| 2 | same | ~52 GB |
| 3 | same | ~52 GB |

No CPU offload required with 90–100 GB cards. See `configs/accelerate_zero3.yaml`
to adjust `num_processes` or enable offload if needed.

---

## End-to-end on MedQA (single teacher)

Each command writes under `outputs/<thing>/<teacher_slug>/`, so artifacts from different teachers never collide.

```bash
TEACHER_ID=claude_sonnet_4_5
PROVIDER=anthropic
MODEL=claude-sonnet-4-5

# 1. Cache MedQA from HF + decide splits (idempotent — re-running is free)
python -m src.pipeline.cli load_medqa \
    --train_size 600 --dev_size 100 --test_size 200

# 2. Synthesize 500 SFT samples per subagent
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli synth_subagent \
      --teacher_id "$TEACHER_ID" \
      --teacher_provider "$PROVIDER" --teacher_model "$MODEL" \
      --agent_kind "$KIND" --n_samples 500
done

# 3. SFT-train each subagent (LoRA r=16, alpha=32)
for KIND in extractor reasoner rule_applier; do
  python -m src.pipeline.cli train_subagent \
      --teacher_id "$TEACHER_ID" --agent_kind "$KIND" \
      --sft_epochs 3 --sft_lr 2e-4
done

# 4. (recommended) Validate that subagents output valid JSON / pass schema
python -m src.pipeline.cli eval_subagents \
    --teacher_id "$TEACHER_ID" --eval_n_samples 50

# 5. Train the manager with GRPO (binary correctness reward)
python -m src.pipeline.cli train_manager_grpo \
    --teacher_id "$TEACHER_ID" \
    --mgr_max_steps 200 \
    --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."

# 6. One evolve round: GRPO failures -> teacher routing plan -> manager SFT
python -m src.pipeline.cli evolve_round \
    --teacher_id "$TEACHER_ID" \
    --teacher_provider "$PROVIDER" --teacher_model "$MODEL" \
    --mgr_max_steps 100

# 7. Evaluate the final manager on the test split
python -m src.pipeline.cli eval_manager \
    --teacher_id "$TEACHER_ID" --eval_n_samples 200
```

A full single-teacher run on Qwen3-0.6B with one A100 takes roughly:

| Stage                    | Time          | Cost (teacher) |
|--------------------------|---------------|----------------|
| Synth (3 × 500 samples)  | 30–90 min     | $5–$30         |
| Subagent SFT (×3)        | 10–20 min     | —              |
| Manager GRPO (200 steps) | 1–3 hours     | —              |
| Evolve round             | 30–60 min     | $1–$5          |
| Eval                     | 5–10 min      | —              |

Teacher cost varies a lot by provider — DeepSeek is the cheapest by ~10×, Claude/GPT comparable.

---

## Comparing teachers (the actual experiment)

Run the full pipeline three times with different teachers. The `--teacher_id` flag controls all output paths, so the three runs never overwrite each other.

| teacher_id           | provider   | model                    |
|----------------------|------------|--------------------------|
| `claude_sonnet_4_5`  | anthropic  | `claude-sonnet-4-5`      |
| `gpt_4o`             | openai     | `gpt-4o-2024-08-06`      |
| `deepseek_v4`        | deepseek   | `deepseek-chat`          |

After all three runs, the three layers of comparison are:

**Layer 1 — synthesis quality** (cheapest signal, available before any training):
```bash
# Per-attempt failure log shows how often each teacher hit each quality gate
ls outputs/sft_data/<teacher_id>/*_synth_log.jsonl
ls outputs/sft_data/<teacher_id>/*.meta.json   # aggregate stats
```
Look for: `json_parse_fail`, `schema_fail`, `leakage_fail`, `balance_fail` rates. A high `leakage_fail` rate means the teacher keeps trying to disclose the answer — that teacher's subagent will likely be worse downstream.

**Layer 2 — subagent reliability** (after subagent SFT):
```bash
cat outputs/eval/<teacher_id>/subagent_eval_report.json
```
Look for: `json_ok_rate` and `schema_ok_rate` per subagent. If they're below ~0.9 the subagent isn't reliable enough to be a tool.

**Layer 3 — manager accuracy** (final number):
```bash
cat outputs/eval/<teacher_id>/manager_eval_report.json
```
This is the headline comparison. But the interesting story is usually in *how* the three teachers fail — pull `outputs/eval/<teacher_id>/manager_eval.jsonl` to see per-example predictions and routing patterns.

---

## Output layout

```
outputs/
├── data/
│   └── medqa_normalized.jsonl              # cached normalized MedQA
├── teacher_cache/<teacher_slug>/           # disk cache for teacher API calls
│   └── <hash>.json                         # avoids re-spending on re-runs
├── sft_data/<teacher_slug>/
│   ├── extractor_sft.jsonl                 # SFT input for the subagent trainer
│   ├── extractor_sft.jsonl.meta.json       # synth stats (acceptance rates etc.)
│   ├── extractor_synth_log.jsonl           # per-attempt failure log
│   ├── reasoner_sft.jsonl
│   ├── reasoner_sft.jsonl.meta.json
│   ├── reasoner_synth_log.jsonl
│   ├── rule_applier_sft.jsonl
│   ├── rule_applier_sft.jsonl.meta.json
│   └── rule_applier_synth_log.jsonl
├── adapters/<teacher_slug>/
│   ├── extractor_adapter/                  # LoRA adapter
│   ├── reasoner_adapter/
│   └── rule_applier_adapter/
├── manager/<teacher_slug>/
│   ├── grpo/
│   │   ├── (model checkpoints)
│   │   ├── fail_buffer.jsonl               # GRPO failures, drives evolve
│   │   ├── train_raw_trace.jsonl           # full per-completion trace
│   │   └── manager_run_config.json
│   ├── evolve/
│   │   ├── manager_sft_from_failures.jsonl # multi-turn SFT trajectories
│   │   └── evolve_run_config.json
│   └── sft_evolved/                        # final manager (post-evolve SFT)
└── eval/<teacher_slug>/
    ├── subagent_eval.jsonl                 # per-example subagent JSON output
    ├── subagent_eval_report.json
    ├── manager_eval.jsonl                  # per-example manager prediction
    └── manager_eval_report.json
```

---

## Available CLI stages

| stage                  | what it does                                                        |
|------------------------|---------------------------------------------------------------------|
| `load_medqa`           | Load + normalize MedQA from HF (or local), cache to disk            |
| `synth_subagent`       | Generate SFT data for one subagent using the chosen teacher         |
| `train_subagent`       | LoRA-SFT one subagent on its synthesized data                       |
| `train_manager_grpo`   | GRPO-train the manager with three subagents as frozen tools         |
| `evolve_build_sft`     | Read GRPO fail buffer, build per-turn manager SFT trajectories      |
| `train_manager_sft`    | LoRA-SFT the manager on the evolve trajectories                     |
| `evolve_round`         | Run all three (GRPO → evolve_build → manager_sft) in sequence       |
| `eval_subagents`       | Score each subagent on JSON validity + schema validity              |
| `eval_manager`         | Score manager final-answer accuracy on a sample                     |

Run `python -m src.pipeline.cli --help` for the full flag list.

---

## Key design decisions worth knowing

**1. Why three subagents instead of two**
The previous PubMedQA-focused version had `reasoning_tool` + `context_tool`, which overlap. Splitting into `Extractor` (objective fact extraction) + `Reasoner` (structured multi-step reasoning) + `RuleApplier` (rule/criterion application) makes the routing decision meaningful. On MedQA specifically, Reasoner is the workhorse because MedQA is closed-book MCQ, but the design generalizes to PubMedQA / LegalBench / GPQA without changes to the schemas.

**2. GT visibility per subagent**
- **Extractor** — GT is hidden from the teacher. Reason: extraction is supposed to be objective; showing GT biases the teacher to only surface evidence supporting the right answer, which teaches the subagent to skip reasonable counter-evidence.
- **Reasoner & RuleApplier** — GT is shown to the teacher (as `PRIVATE_GT`), but the prompt forbids disclosure and the leakage auditor scans every output. Reason: these tasks are hard enough that letting the teacher solve from scratch produces too many wrong samples; reverse-construction from a known answer is more reliable.

**3. Four quality gates at synthesis time**
Every teacher response must pass:
1. JSON-parseable
2. Pydantic schema validation (matches the subagent's output schema)
3. Balance check (Reasoner only — `candidate_analysis` must cover all choice keys)
4. Leakage audit (no GT label, GT choice text, or `ANSWER_X` form anywhere in the output)

Failures are retried up to 2× with bumped temperature, then dropped. All failures are logged to `<kind>_synth_log.jsonl`.

**4. Teacher API call caching**
Teacher responses are cached on disk by `(provider, model, messages, temperature)` hash. Re-running the synthesize stage on the same teacher costs $0 for already-seen prompts. Useful when iterating on the schema or quality gates.

**5. Manager tool binding modes**
TRL's GRPO trainer supports two ways to bind tools to the current example:
- `environment` mode: tools have no `example_id` argument; an `Environment.reset(example_id)` call binds them. Preferred — eliminates manager hallucinating example IDs.
- `argument` mode: tools take `example_id` as an explicit arg, manager must pass it from the user message.

The CLI defaults to `auto`, which picks `environment` if your TRL version supports it. Older TRL versions fall back to `argument`. Both work.

**6. Evolve loop logic**
After GRPO, the `fail_buffer.jsonl` contains every failed example. Evolve does:
1. Read failures → unique example IDs
2. For each failed example: ask teacher to choose a tool sequence (0–3 tools), without showing GT
3. Pre-fetch tool outputs for the chosen sequence
4. Construct multi-turn SFT trajectories: `[user_msg → tool_call_1 → tool_output_1 → tool_call_2 → ... → final ANSWER_<token>]`, split into per-turn `(prompt, response)` pairs
5. SFT-train manager on these trajectories at a low LR (2e-5)

The final answer is constructed from GT (teacher forcing). The teacher only chooses *the routing*, not the answer.

---

## Caveats

**Qwen3-0.6B is for pipeline validation, not production results.**
0.6B is enough to verify the full loop runs and to do the teacher comparison (relative differences between teachers should still surface), but absolute manager accuracy on MedQA will be modest. The full routing dynamic — manager learning when *not* to call tools, learning to combine evidence from multiple tools — only emerges with bigger base models. Set `--base_model Qwen/Qwen3-8B` (or larger) for serious runs; the code is base-model-agnostic.

**The leakage auditor is intentionally strict.**
If you see lots of `leakage_fail` in synth logs (>20% rate), check `outputs/sft_data/<teacher_id>/<kind>_synth_log.jsonl` to see what's tripping it. Most legitimate hits are the teacher writing things like "this option is correct" in `candidate_analysis`. Don't relax the auditor without first verifying the matches are false positives — leakage at SFT time gets baked into the subagent and silently degrades manager accuracy at inference time.

**Subagents are loaded onto the same GPU as the manager.**
Three frozen 0.6B subagents + one training 0.6B manager fits comfortably in 24GB. For 7B+ base models you'll need either gradient checkpointing + 80GB cards, or device sharding (not implemented here).

**The `eval_manager` stage uses simple greedy generation, not real tool-calling rollouts.**
This measures the manager's accuracy when forced to answer without tools — it's a useful baseline but doesn't reflect the trained tool-using behavior. For tool-aware eval, you currently need to re-use the GRPO rollout machinery (a TODO).

---

## Extending to other benchmarks

The schemas, prompts, and training code are benchmark-agnostic. To add PubMedQA / GPQA / LegalBench / LawBench:

1. Add a loader in `src/benchmarks/<name>.py` following `medqa.py`'s shape — must produce `StandardRow` objects.
2. Export it from `src/benchmarks/__init__.py`.
3. Add a `--benchmark` flag in `pipeline/cli.py` and route to the right loader.
4. Use `--task_description` to give the manager an appropriate domain prompt ("You are answering legal reasoning questions.", etc.).

The three subagents map cleanly to the other benchmarks:
- **PubMedQA / LegalBench / LawBench** — Extractor becomes more central (long context to mine).
- **GPQA** — Reasoner stays central; RuleApplier handles formula/principle application.
- **LegalBench / LawBench** — RuleApplier becomes central (statute/doctrine application).

---

## Troubleshooting

**"trl ... environment_factory not supported"**
Upgrade TRL: `pip install -U trl`. Or run with `--binding_mode argument`.

**`leakage_fail` rate is very high for one teacher**
That teacher is consistently disclosing answers despite the prompt. You can:
- Try a stricter system prompt for that teacher (edit `src/subagents/prompts/`).
- Lower `--synth_temperature` to make the teacher less creative.
- Use a different model from that provider (e.g. switch GPT-4o → GPT-4-turbo).
This is itself a finding for the teacher comparison.

**Subagent eval shows `schema_ok_rate < 0.7`**
The subagent didn't internalize the schema. Likely causes:
- Not enough SFT epochs — try `--sft_epochs 5`.
- Synth data is too noisy (check synth logs).
- Base model is too small — schema following improves dramatically with size.

**Manager GRPO loss is flat / reward never improves**
Some debugging steps:
- Check `fail_buffer.jsonl` — is the manager always emitting the wrong format? If so, do a small manager SFT on a few hand-crafted format examples first.
- Lower `--mgr_temperature` from 0.9 to 0.7.
- Increase `--mgr_num_generations` from 6 to 8 (more group diversity for GRPO advantage).
- Check the `train_raw_trace.jsonl` — what is the manager actually outputting?

**Out of memory during GRPO**
- Lower `--mgr_bs` to 1.
- Lower `--mgr_num_generations` to 4.
- Lower `--mgr_max_completion_length` to 1024.
- Last resort: switch subagents to CPU inference (slow but functional).

---

## License & citation

Internal research code. If you build on this, please cite the underlying base model (Qwen3) and TRL appropriately.