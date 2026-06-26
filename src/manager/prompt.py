"""Manager system prompt + final-answer parsing.

The manager is a routing + answering agent. It has three native tools:
  - extractor_tool
  - reasoner_tool
  - verifier_tool

Policy:
  - 0 to 3 tool calls allowed total. Each tool may be called at most once.
  - Manager should answer directly when confident, call 1-2 tools when
    uncertain, all 3 only for hard cases.
  - Final answer ends with exactly one line: ANSWER_<TOKEN>
  - Manager MUST NOT emit tool-call JSON or XML in plain text content;
    it must use the model's native tool-calling interface.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _token_to_label(token: str, choices: Dict[str, str]) -> str:
    """Map ANSWER_<TOKEN> back to canonical choice key."""
    t = token.upper().strip()
    for k in choices.keys():
        if _label_to_token(k) == t:
            return k
    return token


# Compiled per-call (since label space depends on choices), but we expose a
# generic regex for sanity-checking ANY ANSWER_<word> on the last line.
ANSWER_LASTLINE_RE_FOR_KEYS = re.compile(
    r"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_([A-Za-z0-9_]+)\b[^\w]*$",
    re.IGNORECASE,
)


def build_manager_system_prompt(label_keys: List[str], task_description: str = "") -> str:
    """Build the manager's system prompt.

    Args:
        label_keys: choice keys for the current task (e.g. ["A","B","C","D"]).
        task_description: optional one-liner describing the task domain.
    """
    answer_lines = "\n".join(f"  ANSWER_{_label_to_token(k)}" for k in label_keys)
    desc = task_description or "You are a manager agent solving a multiple-choice question."
    return (
        desc + "\n\n"
        "You have THREE tools available via the native tool-calling interface:\n"
        "  - extractor_tool: pulls key signals and structures relevant information from the question (and context, if any).\n"
        "  - reasoner_tool: produces a structured chain-of-thought reasoning scaffold (sub-questions, knowledge, per-choice analysis).\n"
        "  - verifier_tool: identifies relevant domain principles and audits the reasoning for logical or computational errors.\n\n"
        "Routing policy:\n"
        "  - You may call 0 to 3 tools total. Each tool may be used at most once.\n"
        "  - Prefer answering directly if you are confident.\n"
        "  - If uncertain, call ONE tool first, read its output, then decide whether to call another.\n"
        "  - Reserve all three tools for hard cases.\n\n"
        "Output rules:\n"
        "  - When calling a tool, use the native interface. Do NOT write tool calls, XML, or JSON in your text content.\n"
        "  - When calling a tool, do NOT also output a final answer in the same turn.\n"
        "  - When you are ready to answer, your message must end with exactly one of these lines:\n"
        + answer_lines + "\n"
        "  - You may include brief reasoning above the final ANSWER_ line, but nothing after it.\n"
        "  - Do not output <think> tags.\n"
    )


def build_manager_user_message(
    example_id: int,
    question: str,
    context: str,
    choices: Dict[str, str],
    binding_mode: str = "environment",
) -> str:
    lines = [f"Example ID: {example_id}", "", f"Question:\n{question}", ""]
    if choices:
        choices_block = "Choices:\n" + "\n".join(f"  {k}. {v}" for k, v in choices.items())
        lines.append(choices_block)
        lines.append("")
    if context:
        lines.append(f"Context:\n{context}")
        lines.append("")
    if binding_mode == "argument":
        lines.append("If you call a tool, pass the current Example ID as the example_id argument.")
    else:
        lines.append("If you call a tool, the current example is already bound — call without arguments.")
    lines.append("")
    lines.append("If you do not call any tool, answer directly.")
    return "\n".join(lines)


def parse_final_answer(text: str, choice_keys: List[str]) -> Optional[str]:
    """Parse the final ANSWER_<TOKEN> line and map to a canonical choice key.

    Returns the choice key if found and matchable, else None.
    """
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return None
    m = ANSWER_LASTLINE_RE_FOR_KEYS.match(lines[-1])
    if not m:
        return None
    token = m.group(1).upper()
    # Map token back to canonical key
    keys_dict = {k: k for k in choice_keys}
    for k in choice_keys:
        if _label_to_token(k) == token:
            return k
    return None