# Agent Routing: Manager + Three Subagents

This repo trains a small manager model that can solve multiple-choice tasks by
calling three frozen subagents as native tools:

- `extractor_tool`: extracts decision-relevant facts and evidence.
- `reasoner_tool`: builds a structured reasoning scaffold.
- `rule_applier_tool`: maps facts to rules, criteria, or principles.

The intended training stack is:

1. Build SFT data for each subagent.
2. LoRA-SFT the three subagents.
3. Cold-start the manager on tool-call demonstrations.
4. GRPO-train the manager with the subagents as tools.
5. Optionally build more manager SFT from GRPO failures and iterate.

The manager is the only component allowed to output the final answer line:

```text
ANSWER_<CHOICE_KEY>
```

Subagents should never output the final answer.

---

## Environment Notes

Windows PowerShell examples use the local venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Windows with Chinese locale, run TRL commands with UTF-8 mode:

```powershell
python -X utf8 -m src.pipeline.cli ...
```

or set this once in the shell:

```powershell
$env:PYTHONUTF8="1"
```

For RTX 5090 / CUDA 12.8, keep PyTorch packages on the same cu128 wheel line.
One stable combo is:

```powershell
python -m pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

To use the latest TRL from GitHub without changing your torch install:

```powershell
python -m pip uninstall -y trl
python -m pip install --upgrade --no-deps git+https://github.com/huggingface/trl.git
```

Check GRPO import:

```powershell
python -X utf8 -c "import trl, inspect; from trl import GRPOConfig, GRPOTrainer; print('GRPO ok'); print('environment_factory:', 'environment_factory' in inspect.signature(GRPOTrainer.__init__).parameters)"
```

---

## Output Layout

Important paths:

```text
outputs/
  data/
    medqa_us4_normalized.jsonl
  sft_data/
    openai_us4_500/
      extractor_openai_responses.jsonl
      reasoner_openai_responses.jsonl
      rule_applier_openai_responses.jsonl
      extractor_runtime_raw_sft.jsonl
      reasoner_runtime_raw_sft.jsonl
      rule_applier_runtime_raw_sft.jsonl
  adapters/
    openai_us4_500_runtime_raw/
      extractor_adapter/
      reasoner_adapter/
      rule_applier_adapter/
  manager/
    openai_us4_500_runtime_raw/
      evolve/
        manager_sft_coldstart.jsonl
      sft_evolved/
      grpo/
        fail_buffer.jsonl
        train_raw_trace.jsonl
        manager_run_config.json
  eval/
```

`teacher_id` controls output namespaces. For the current OpenAI raw-response
experiment, use:

```text
openai_us4_500_runtime_raw
```

---

## SFT Data Formats

There are three related JSONL formats. They should not be confused.

### Prompt JSONL

These rows are prompts sent to a teacher model to synthesize subagent outputs.
They contain the full source question and the teacher prompt.

Example shape:

```json
{
  "example_id": 164,
  "benchmark_name": "medqa",
  "agent_kind": "extractor",
  "question": "...",
  "context": "",
  "choices": {"A": "...", "B": "..."},
  "ground_truth": "A",
  "prompt": [
    {"role": "system", "content": "You are an expert annotator..."},
    {"role": "user", "content": "QUESTION:\n..."}
  ]
}
```

These prompts are for generating training data. They are not the runtime
subagent prompts.

### Raw Teacher Response JSONL

OpenAI generation writes rows like:

```json
{
  "example_id": 164,
  "prompt": [...],
  "response": "{\"key_evidence\": [], ...}"
}
```

These files preserve the raw teacher output:

```text
outputs/sft_data/openai_us4_500/extractor_openai_responses.jsonl
outputs/sft_data/openai_us4_500/reasoner_openai_responses.jsonl
outputs/sft_data/openai_us4_500/rule_applier_openai_responses.jsonl
```

Do not train subagents directly on these raw response files unless you
intentionally want the model to learn the teacher-data-generation prompt.

### Runtime Raw SFT JSONL

This is the preferred format for training on raw OpenAI responses:

```json
{
  "example_id": 164,
  "benchmark_name": "medqa",
  "agent_kind": "extractor",
  "teacher_provider": "raw_jsonl",
  "teacher_model": "gpt-4o-mini",
  "prompt": [
    {"role": "system", "content": "You are the Extractor sub-agent..."},
    {"role": "user", "content": "QUESTION:\n..."}
  ],
  "response": "{\"key_evidence\": [], ...}"
}
```

The response is kept raw, but the prompt is rebuilt with the real runtime
subagent prompt from `build_runtime_messages`. This teaches the subagent the
same interface it will see when used as a tool.

Current files:

```text
outputs/sft_data/openai_us4_500/extractor_runtime_raw_sft.jsonl
outputs/sft_data/openai_us4_500/reasoner_runtime_raw_sft.jsonl
outputs/sft_data/openai_us4_500/rule_applier_runtime_raw_sft.jsonl
```

### Validated SFT Alternative

If you import without `--deepseek_import_raw_responses`, the importer parses the
JSON, validates the Pydantic schema, checks choice coverage for the reasoner,
and filters answer leakage. This produces cleaner but smaller files such as:

```text
extractor_sft.jsonl
reasoner_sft.jsonl
rule_applier_sft.jsonl
```

For the raw OpenAI experiment in this README, use `*_runtime_raw_sft.jsonl`.

---

## Build OpenAI Runtime Raw SFT Data

If you already have OpenAI response files, rebuild runtime SFT files with:

```powershell
python -X utf8 -m src.pipeline.cli import_deepseek_jsonl --teacher_id openai_us4_500 --agent_kind extractor --deepseek_prompt_jsonl outputs\sft_data\deepseek_us4_500\extractor_deepseek_prompts.jsonl --deepseek_response_jsonl outputs\sft_data\openai_us4_500\extractor_openai_responses.jsonl --deepseek_sft_jsonl outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --deepseek_teacher_model gpt-4o-mini --deepseek_import_raw_responses
python -X utf8 -m src.pipeline.cli import_deepseek_jsonl --teacher_id openai_us4_500 --agent_kind reasoner --deepseek_prompt_jsonl outputs\sft_data\deepseek_us4_500\reasoner_deepseek_prompts.jsonl --deepseek_response_jsonl outputs\sft_data\openai_us4_500\reasoner_openai_responses.jsonl --deepseek_sft_jsonl outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --deepseek_teacher_model gpt-4o-mini --deepseek_import_raw_responses
python -X utf8 -m src.pipeline.cli import_deepseek_jsonl --teacher_id openai_us4_500 --agent_kind rule_applier --deepseek_prompt_jsonl outputs\sft_data\deepseek_us4_500\rule_applier_deepseek_prompts.jsonl --deepseek_response_jsonl outputs\sft_data\openai_us4_500\rule_applier_openai_responses.jsonl --deepseek_sft_jsonl outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --deepseek_teacher_model gpt-4o-mini --deepseek_import_raw_responses
```

The stage name still says `deepseek_jsonl` because it was originally a bridge
for local DeepSeek generation, but it now works for any prompt/response JSONL.

If you need to generate OpenAI responses from prompt files:

```powershell
$env:OPENAI_API_KEY="..."
python scripts\generate_openai_us4_all.py --input-dir outputs\sft_data\deepseek_us4_500 --output-dir outputs\sft_data\openai_us4_500 --model gpt-4o-mini --resume
```

---

## Train Subagents

Train the three subagents on runtime raw SFT data:

```powershell
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind extractor --sft_train_jsonl outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --sft_epochs 3 --sft_lr 2e-4
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind reasoner --sft_train_jsonl outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --sft_epochs 3 --sft_lr 2e-4
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind rule_applier --sft_train_jsonl outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --sft_epochs 3 --sft_lr 2e-4
```

Adapters are saved under:

```text
outputs/adapters/openai_us4_500_runtime_raw/
```

For a quick smoke test:

```powershell
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind extractor --sft_train_jsonl outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --sft_epochs 1 --sft_max_steps 20
```

---

## Why Manager Cold Start Is Needed

If you run GRPO directly, the manager may never discover native tool calls. You
will see logs like:

```text
tools/call_frequency: 0
reward: 0
frac_reward_zero_std: 1
```

In that state, a reward bonus for using tools does not help, because no rollout
contains a tool call and no trajectory can receive the bonus.

The standard solution is behavior cloning / SFT cold start:

1. Build tool-call demonstrations.
2. SFT the manager on those demonstrations.
3. Start GRPO from that manager SFT adapter.

This teaches the model the native tool-call format before RL tries to optimize
which tool to call and when.

---

## Build Manager Cold-Start SFT

This stage does not require a previous `fail_buffer.jsonl`. It builds tool-call
demonstrations from ordinary training examples.

Command:

```powershell
python -X utf8 -m src.pipeline.cli manager_coldstart_sft --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --train_size 1200 --exclude_sft_example_ids outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --coldstart_n_samples 300 --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

What it does:

1. Loads `outputs/data/medqa_us4_normalized.jsonl`.
2. Splits `train_size=1200` examples.
3. Excludes the 500 examples used for subagent SFT.
4. Samples `coldstart_n_samples=300` remaining manager-only examples.
5. Chooses a simple tool sequence per example:
   - mostly `reasoner_tool`
   - sometimes `extractor_tool -> reasoner_tool`
   - sometimes `rule_applier_tool -> reasoner_tool`
   - long context prefers `extractor_tool -> reasoner_tool`
6. Calls the frozen SFT subagents to get real tool outputs.
7. Writes per-turn manager SFT rows.

Example for one-tool sequence:

```text
row 1:
  prompt = system + user question
  response = assistant native tool_call(reasoner_tool)

row 2:
  prompt = system + user question + assistant tool_call + tool output
  response = assistant final ANSWER_X
```

Example for two-tool sequence:

```text
row 1: prompt -> assistant calls extractor_tool
row 2: prompt + extractor output -> assistant calls reasoner_tool
row 3: prompt + extractor output + reasoner output -> assistant final ANSWER_X
```

Output:

```text
outputs/manager/openai_us4_500_runtime_raw/evolve/manager_sft_coldstart.jsonl
```

If 300 examples produce 720 SFT rows, that is expected: each original example
can produce two or three per-turn SFT rows.

Optional teacher routing:

```powershell
python -X utf8 -m src.pipeline.cli manager_coldstart_sft --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --train_size 1200 --exclude_sft_example_ids outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --coldstart_n_samples 300 --teacher_provider openai --teacher_model gpt-4o-mini --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

With a teacher, the teacher chooses the tool sequence. The final answer is still
constructed from ground truth; the teacher does not generate the answer.

---

## Train Manager On Cold-Start SFT

Train the manager on the generated tool-call demonstrations:

```powershell
python -X utf8 -m src.pipeline.cli train_manager_sft --teacher_id openai_us4_500_runtime_raw --manager_sft_train_jsonl outputs\manager\openai_us4_500_runtime_raw\evolve\manager_sft_coldstart.jsonl --manager_sft_epochs 1 --manager_sft_lr 2e-5
```

Expected progress for 720 SFT rows with batch size 1 and grad accumulation 8:

```text
720 / 8 = 90 optimizer steps
```

The manager SFT adapter is saved at:

```text
outputs/manager/openai_us4_500_runtime_raw/sft_evolved
```

---

## GRPO Manager Training

There are two manager GRPO modes:

- Full-parameter GRPO: pass `--mgr_full_parameter_rl`. The manager SFT adapter
  is merged into the base model, then all manager weights are trainable. This
  saves a full model under the requested output directory.
- LoRA-adapter GRPO: omit `--mgr_full_parameter_rl`. Only the manager LoRA
  adapter is trained and saved.

Recommended full-parameter GRPO from the cold-start manager adapter:

```powershell
python -X utf8 -m src.pipeline.cli train_manager_grpo --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --train_size 1200 --exclude_sft_example_ids outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --mgr_init_adapter outputs\manager\openai_us4_500_runtime_raw\sft_evolved --mgr_full_parameter_rl --mgr_output_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --mgr_bs 2 --mgr_num_generations 2 --mgr_max_completion_length 2048 --mgr_temperature 1.0 --mgr_grpo_beta 0.01 --mgr_tool_use_bonus 0.2 --mgr_max_steps 200 --mgr_use_wandb --wandb_project agent_routing --wandb_run_name openai_us4_500_runtime_raw_grpo_full_after_coldstart --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

Optional LoRA-adapter GRPO if full-parameter training is too expensive:

```powershell
python -X utf8 -m src.pipeline.cli train_manager_grpo --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --train_size 1200 --exclude_sft_example_ids outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --mgr_init_adapter outputs\manager\openai_us4_500_runtime_raw\sft_evolved --mgr_output_dir outputs\manager\openai_us4_500_runtime_raw\grpo_lora --mgr_bs 4 --mgr_num_generations 4 --mgr_max_completion_length 2048 --mgr_temperature 1.0 --mgr_grpo_beta 0.01 --mgr_tool_use_bonus 0.2 --mgr_max_steps 200 --mgr_use_wandb --wandb_project agent_routing --wandb_run_name openai_us4_500_runtime_raw_grpo_lora_after_coldstart --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

Key parameters:

- `--train_size 1200`: size of the MedQA train pool before exclusions.
- `--exclude_sft_example_ids ...`: removes subagent SFT examples from manager GRPO train rows.
- `--mgr_full_parameter_rl`: merge the init adapter and update all manager weights.
- `--mgr_output_dir ...\grpo_full`: keep full-parameter GRPO separate from older adapter runs.
- `--mgr_bs`: GRPO generation batch size. With current TRL this must be divisible by `--mgr_num_generations`.
- `--mgr_num_generations`: rollout count per prompt group.
- `--mgr_max_steps 200`: optimizer update steps, not tool-call steps.
- `--mgr_max_completion_length 2048`: max generated tokens per rollout.
- `--mgr_tool_use_bonus 0.2`: bonus only when the answer is correct and at least one native tool call was used.
- `--mgr_init_adapter ...`: initializes GRPO from the cold-start manager SFT adapter.

After full-parameter GRPO, the output directory should contain `config.json`
and full model weights such as `model.safetensors` or sharded safetensors. If
it only contains `adapter_config.json` and `adapter_model.safetensors`, that run
was adapter GRPO, not full-parameter GRPO.

Per-rollout tool calling is capped in code by:

```text
max_tool_calling_iterations = 3
```

That is the "at most three subagent calls per question" constraint. It is
different from `--mgr_max_steps`.

Watch these metrics:

```text
tools/call_frequency
tools/failure_frequency
reward
rewards/binary_outcome_with_format/mean
frac_reward_zero_std
```

If `tools/call_frequency` stays at 0 after cold start, strengthen the manager
prompt or build more cold-start examples.

---

## W&B

Login once:

```powershell
wandb login
```

Enable W&B for GRPO with:

```text
--mgr_use_wandb
```

Run names are controlled by:

```text
--wandb_run_name <name>
```

SFT stages currently log to terminal only (`report_to=[]`).

---

## Evolve From GRPO Failures

After GRPO, failures are written to:

```text
outputs/manager/<teacher_id>/grpo_full/fail_buffer.jsonl
```

You can turn failures into more manager SFT data:

```powershell
python -X utf8 -m src.pipeline.cli evolve_build_sft --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

If you used a custom GRPO output directory, pass its failure buffer explicitly
when building evolve SFT from that run:

```powershell
python -X utf8 -m src.pipeline.cli evolve_build_sft --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions." --fail_buffer_jsonl outputs\manager\openai_us4_500_runtime_raw\grpo_full\fail_buffer.jsonl
```

Then SFT manager on that failure-driven data:

```powershell
python -X utf8 -m src.pipeline.cli train_manager_sft --teacher_id openai_us4_500_runtime_raw --manager_sft_epochs 1 --manager_sft_lr 2e-5
```

This is different from `manager_coldstart_sft`:

- `manager_coldstart_sft`: no fail buffer needed; builds initial tool-use demonstrations.
- `evolve_build_sft`: requires GRPO failures; builds targeted demonstrations for examples the manager missed.

---

## Evaluation

Evaluation now has two different manager paths:

- `eval_manager_tools`: recommended. This evaluates the trained manager with
  frozen subagents in the loop, using the same tool names as GRPO
  (`extractor_tool`, `reasoner_tool`, `rule_applier_tool`). Use this for final
  accuracy and routing/tool-use metrics.
- `eval_manager`: legacy sanity probe. This does one-shot manager generation
  without executing tools, so it does not measure the routed manager system.

MedQA leakage rule: keep evaluation on the MedQA `test` split. The normalized
cache already contains explicit `train` / `dev` / `test` splits; the CLI honors
those splits. Use `--test_size 1270 --eval_n_samples 1270` to evaluate the full
MedQA test split instead of the default 200-example cap. Subagent SFT rows and
manager GRPO rows come from `train`, so they should not appear in this test
evaluation.

Subagent schema eval checks whether each frozen subagent emits valid outputs:

```powershell
python -X utf8 -m src.pipeline.cli eval_subagents --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --eval_n_samples 50
```

Recommended MedQA manager evaluation after full-parameter GRPO:

```powershell
python -X utf8 -m src.pipeline.cli eval_manager_tools --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --eval_n_samples 1270 --test_size 1270 --eval_manager_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

Outputs:

```text
outputs/eval/openai_us4_500_runtime_raw/manager_tool_eval.jsonl
outputs/eval/openai_us4_500_runtime_raw/manager_tool_eval_report.json
```

Key report fields:

```text
accuracy
valid_answer_rate
tool_call_rate
avg_tool_calls
tool_counts
malformed_tool_calls
```

Use this small smoke test before the full 1270-example run:

```powershell
python -X utf8 -m src.pipeline.cli eval_manager_tools --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --eval_n_samples 20 --test_size 1270 --eval_manager_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

LegalBench manager tool eval:

```powershell
python -X utf8 -m src.pipeline.cli eval_manager_tools --teacher_id openai_us4_500_runtime_raw --legalbench_normalized_cache outputs\data\legalbench_abercrombie.jsonl --eval_n_samples 200 --eval_manager_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --task_description "You are a manager agent solving a LegalBench multiple-choice classification task."
```

Legacy one-shot manager probe, without tool execution:

```powershell
python -X utf8 -m src.pipeline.cli eval_manager --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --eval_n_samples 200 --eval_manager_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```

For training-time rollout behavior, inspect the raw GRPO trace from the run:

```text
outputs/manager/<teacher_id>/grpo_full/train_raw_trace.jsonl
```

---

## CLI Stages

| Stage | Purpose |
| --- | --- |
| `load_medqa` | Load and normalize MedQA. |
| `export_deepseek_jsonl` | Export chat prompts for batch/local teacher generation. |
| `import_deepseek_jsonl` | Convert prompt/response JSONL into subagent SFT. With `--deepseek_import_raw_responses`, keeps response text raw but rebuilds runtime prompts. |
| `synth_subagent` | Online teacher synthesis with validation and leakage checks. |
| `train_subagent` | LoRA-SFT one subagent. |
| `eval_subagents` | Check subagent JSON/schema validity. |
| `manager_coldstart_sft` | Build manager native tool-call SFT demonstrations from ordinary train rows. |
| `train_manager_sft` | LoRA-SFT manager on cold-start or evolve SFT rows. |
| `train_manager_grpo` | GRPO-train manager with subagents as frozen tools. |
| `evolve_build_sft` | Build targeted manager SFT from GRPO failures. |
| `evolve_round` | Run GRPO, build failure SFT, then train manager SFT. |
| `eval_manager` | Simple manager accuracy evaluation. |

Run:

```powershell
python -X utf8 -m src.pipeline.cli --help
```

---

## Troubleshooting

### `trl is required for manager GRPO training`

This can mean TRL import failed, not necessarily that TRL is missing. On Windows
with Chinese locale, TRL may fail reading its jinja templates with GBK. Use:

```powershell
python -X utf8 -c "from trl import GRPOConfig, GRPOTrainer; print('GRPO ok')"
```

and run training with `python -X utf8`.

### `generation_batch_size must be divisible by num_generations`

Newer TRL requires the generation batch size to be divisible by
`num_generations`. Use:

```text
--mgr_bs 2 --mgr_num_generations 2
```

or:

```text
--mgr_bs 4 --mgr_num_generations 4
```

or:

```text
--mgr_bs 8 --mgr_num_generations 4
```

### `tools/call_frequency = 0`

The manager is answering directly and not learning routing. Use the cold-start
flow:

```text
manager_coldstart_sft -> train_manager_sft -> train_manager_grpo --mgr_init_adapter ...
```

### `torchvision::nms does not exist`

Your `torch` and `torchvision` wheels are mismatched. Reinstall a matching
CUDA wheel pair. For cu128:

```powershell
python -m pip install --force-reinstall torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

### Subagent loss looks "not tiny"

For SFT, token-level loss around `0.4` to `0.8` can be fine. Evaluate JSON and
schema validity instead of chasing lower loss.

### Manager GRPO is very fast

If `tools/call_frequency=0` and completions are short, GRPO is only training
direct-answer behavior. Cold-start tool calling before GRPO.

---

## Current Recommended OpenAI US4 Flow

Short version:

```powershell
# 1. Build runtime raw SFT from OpenAI responses
python -X utf8 -m src.pipeline.cli import_deepseek_jsonl --teacher_id openai_us4_500 --agent_kind extractor --deepseek_prompt_jsonl outputs\sft_data\deepseek_us4_500\extractor_deepseek_prompts.jsonl --deepseek_response_jsonl outputs\sft_data\openai_us4_500\extractor_openai_responses.jsonl --deepseek_sft_jsonl outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --deepseek_teacher_model gpt-4o-mini --deepseek_import_raw_responses
python -X utf8 -m src.pipeline.cli import_deepseek_jsonl --teacher_id openai_us4_500 --agent_kind reasoner --deepseek_prompt_jsonl outputs\sft_data\deepseek_us4_500\reasoner_deepseek_prompts.jsonl --deepseek_response_jsonl outputs\sft_data\openai_us4_500\reasoner_openai_responses.jsonl --deepseek_sft_jsonl outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --deepseek_teacher_model gpt-4o-mini --deepseek_import_raw_responses
python -X utf8 -m src.pipeline.cli import_deepseek_jsonl --teacher_id openai_us4_500 --agent_kind rule_applier --deepseek_prompt_jsonl outputs\sft_data\deepseek_us4_500\rule_applier_deepseek_prompts.jsonl --deepseek_response_jsonl outputs\sft_data\openai_us4_500\rule_applier_openai_responses.jsonl --deepseek_sft_jsonl outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --deepseek_teacher_model gpt-4o-mini --deepseek_import_raw_responses

# 2. Train subagents
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind extractor --sft_train_jsonl outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --sft_epochs 3 --sft_lr 2e-4
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind reasoner --sft_train_jsonl outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --sft_epochs 3 --sft_lr 2e-4
python -X utf8 -m src.pipeline.cli train_subagent --teacher_id openai_us4_500_runtime_raw --agent_kind rule_applier --sft_train_jsonl outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --sft_epochs 3 --sft_lr 2e-4

# 3. Build and train manager cold-start SFT
python -X utf8 -m src.pipeline.cli manager_coldstart_sft --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --train_size 1200 --exclude_sft_example_ids outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --coldstart_n_samples 300 --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
python -X utf8 -m src.pipeline.cli train_manager_sft --teacher_id openai_us4_500_runtime_raw --manager_sft_train_jsonl outputs\manager\openai_us4_500_runtime_raw\evolve\manager_sft_coldstart.jsonl --manager_sft_epochs 1 --manager_sft_lr 2e-5

# 4. Full-parameter GRPO from cold-start manager
python -X utf8 -m src.pipeline.cli train_manager_grpo --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --train_size 1200 --exclude_sft_example_ids outputs\sft_data\openai_us4_500\extractor_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\reasoner_runtime_raw_sft.jsonl --exclude_sft_example_ids outputs\sft_data\openai_us4_500\rule_applier_runtime_raw_sft.jsonl --mgr_init_adapter outputs\manager\openai_us4_500_runtime_raw\sft_evolved --mgr_full_parameter_rl --mgr_output_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --mgr_bs 2 --mgr_num_generations 2 --mgr_max_completion_length 2048 --mgr_temperature 1.0 --mgr_grpo_beta 0.01 --mgr_tool_use_bonus 0.2 --mgr_max_steps 200 --mgr_use_wandb --wandb_project agent_routing --wandb_run_name openai_us4_500_runtime_raw_grpo_full --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."

# 5. Evaluate the full-parameter GRPO manager on MedQA test
python -X utf8 -m src.pipeline.cli eval_manager_tools --teacher_id openai_us4_500_runtime_raw --medqa_normalized_cache outputs\data\medqa_us4_normalized.jsonl --eval_manager_dir outputs\manager\openai_us4_500_runtime_raw\grpo_full --eval_n_samples 1270 --test_size 1270 --task_description "You are a manager agent solving USMLE-style medical multiple-choice questions."
```
