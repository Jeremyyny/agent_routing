"""FrozenAgent: load a SFT'd subagent and run it as a non-trainable tool.

Used at:
  - manager GRPO training time (subagents are tools called from manager rollouts)
  - manager evaluation time
  - manager evolve_round (to produce tool outputs for SFT trace construction)

Key behaviors:
  - Loads base model + LoRA adapter (PEFT). If adapter_path points to a full
    save_pretrained dir (no adapter_config.json), loads as a full model.
  - Greedy decoding by default (deterministic tool outputs, important for
    GRPO group-relative advantage computation).
  - Caches outputs by (agent_kind, example_id) so repeated calls on the same
    example during multi-rollout GRPO are free.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts.runtime_prompts import build_runtime_messages
from ..utils.io import read_jsonl
from ..utils.seed import set_seed

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def _render_chat(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass
class FrozenSubagent:
    base_model: str
    adapter_path: Optional[str]
    agent_kind: str             # "extractor" | "reasoner" | "verifier"
    device: str = "cuda"
    max_new_tokens: int = 1024
    dtype_str: str = "bfloat16"

    _tok: Any = field(init=False, default=None)
    _model: Any = field(init=False, default=None)

    def __post_init__(self):
        self._tok = AutoTokenizer.from_pretrained(
            self.adapter_path or self.base_model, trust_remote_code=True
        )
        if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
            self._tok.pad_token_id = self._tok.eos_token_id
        self._tok.padding_side = "left"

        dtype = torch.bfloat16 if self.dtype_str == "bfloat16" and self.device == "cuda" else torch.float32

        is_full_save = (
            self.adapter_path
            and os.path.isdir(self.adapter_path)
            and not os.path.exists(os.path.join(self.adapter_path, "adapter_config.json"))
            and os.path.exists(os.path.join(self.adapter_path, "config.json"))
        )

        if is_full_save:
            model = AutoModelForCausalLM.from_pretrained(
                self.adapter_path, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
            if self.adapter_path:
                if not PEFT_AVAILABLE:
                    raise RuntimeError("peft not available; cannot load adapter.")
                model = PeftModel.from_pretrained(model, self.adapter_path).to(self.device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model

    @torch.no_grad()
    def generate(
        self,
        question: str,
        context: str,
        choices: Dict[str, str],
        temperature: float = 0.0,
    ) -> str:
        messages = build_runtime_messages(
            agent_kind=self.agent_kind,
            question=question,
            context=context,
            choices=choices,
        )
        prompt = _render_chat(self._tok, messages, add_generation_prompt=True)
        inputs = self._tok(prompt, return_tensors="pt").to(self.device)

        do_sample = temperature > 1e-6
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tok.pad_token_id,
            "eos_token_id": self._tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-6)

        out = self._model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()


class SubagentPool:
    """Holds up to three FrozenSubagent instances and routes calls by kind.

    Provides per-(kind, example_id) output caching: during GRPO with N rollouts
    per example, the manager may call the same tool multiple times across
    rollouts; we want the tool output to be deterministic and cheap.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, FrozenSubagent] = {}
        self._cache: Dict[str, str] = {}
        self._call_log: List[Dict[str, Any]] = []

    def register(self, agent: FrozenSubagent) -> None:
        self._agents[agent.agent_kind] = agent

    def has(self, agent_kind: str) -> bool:
        return agent_kind in self._agents

    def call(
        self,
        agent_kind: str,
        example_id: int,
        question: str,
        context: str,
        choices: Dict[str, str],
        cache_namespace: str = "default",
    ) -> str:
        key = f"{cache_namespace}::{agent_kind}::{int(example_id)}"
        if key in self._cache:
            self._call_log.append({
                "ts": int(time.time()),
                "agent_kind": agent_kind,
                "example_id": int(example_id),
                "cache_hit": True,
            })
            return self._cache[key]

        if agent_kind not in self._agents:
            raise KeyError(f"Subagent not registered: {agent_kind}")

        text = self._agents[agent_kind].generate(question, context, choices)
        self._cache[key] = text
        self._call_log.append({
            "ts": int(time.time()),
            "agent_kind": agent_kind,
            "example_id": int(example_id),
            "cache_hit": False,
            "output_len": len(text),
        })
        return text

    def clear_cache(self) -> None:
        self._cache.clear()

    def drain_log(self) -> List[Dict[str, Any]]:
        log = self._call_log
        self._call_log = []
        return log


@dataclass
class SFTConfig:
    base_model: str
    train_jsonl: str
    out_dir: str
    dev_jsonl: Optional[str] = None
    seed: int = 42
    max_seq_len: int = 4096
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_steps: int = -1
    bf16: bool = True


def _tokenize_subagent_sft(rows: List[Dict[str, Any]], tok, max_seq_len: int) -> Any:
    from datasets import Dataset

    eos = tok.eos_token or ""

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt_msgs = ex["prompt"]
        response = ex["response"]
        response_msgs = [{"role": "assistant", "content": response}]

        prompt_text = _render_chat(tok, prompt_msgs, add_generation_prompt=True)
        full_text = _render_chat(tok, prompt_msgs + response_msgs, add_generation_prompt=False) + eos

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full = tok(full_text, add_special_tokens=False)
        input_ids = full["input_ids"][:max_seq_len]
        attention_mask = full["attention_mask"][:max_seq_len]
        plen = min(len(prompt_ids), max_seq_len)
        labels = ([-100] * plen) + input_ids[plen:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    ds = Dataset.from_list(rows)
    return ds.map(_map, remove_columns=ds.column_names)


def _lora_target_modules(model) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    return [name for name in candidates if name in present] or ["q_proj", "v_proj"]


def train_subagent_sft(cfg: SFTConfig) -> None:
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    if cfg.use_lora and not PEFT_AVAILABLE:
        raise RuntimeError("peft is required for LoRA subagent SFT training.")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.bfloat16 if (cfg.bf16 and device == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False

    if cfg.use_lora:
        target = _lora_target_modules(model)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target,
        )
        model = get_peft_model(model, lora_cfg)
        print(f"[SUBAGENT_SFT/LoRA] r={cfg.lora_r} alpha={cfg.lora_alpha} target_modules={target}")

    train_rows = read_jsonl(cfg.train_jsonl)
    if not train_rows:
        raise ValueError(f"No rows in {cfg.train_jsonl}")
    train_ds = _tokenize_subagent_sft(train_rows, tok, cfg.max_seq_len)

    eval_ds = None
    if cfg.dev_jsonl:
        dev_rows = read_jsonl(cfg.dev_jsonl)
        eval_ds = _tokenize_subagent_sft(dev_rows, tok, cfg.max_seq_len) if dev_rows else None

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt")
    args = TrainingArguments(
        output_dir=cfg.out_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        bf16=(cfg.bf16 and device == "cuda"),
        fp16=False,
        report_to=[],
        seed=cfg.seed,
        remove_unused_columns=False,
        max_steps=(cfg.max_steps if cfg.max_steps > 0 else -1),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    os.makedirs(cfg.out_dir, exist_ok=True)
    trainer.model.save_pretrained(cfg.out_dir)
    tok.save_pretrained(cfg.out_dir)
    print(f"[SUBAGENT_SFT] saved -> {cfg.out_dir}")
