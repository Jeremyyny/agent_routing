"""Reward functions for manager GRPO.

Two reward modes:

1. Binary mode (default, ccr_mode=False):
   base_reward = 1.0 if correct else 0.0
   + optional routing_efficiency_bonus * saved_tool_calls (when correct)
   + optional tool_use_bonus (when correct and >=1 tool called)

2. Calibrated Confidence Routing mode (ccr_mode=True):
   Frames the routing decision as an implicit confidence expression and
   applies the logarithmic scoring rule (a proper scoring rule). Calling
   fewer tools implicitly claims higher confidence; the reward penalises
   that claim more heavily when wrong.

       p(k) = p_high + (p_low - p_high) * k / k_max   [linear interpolation]
       R    = log(p(k))       if correct
       R    = log(1 - p(k))   if incorrect

   With default p_high=0.9, p_low=0.2, k_max=3:
     k=0 correct  → log(0.90) ≈ -0.11   (best: efficient and right)
     k=3 correct  → log(0.20) ≈ -1.61   (correct but needed all tools)
     k=3 incorrect→ log(0.80) ≈ -0.22   (at least it was uncertain)
     k=0 incorrect→ log(0.10) ≈ -2.30   (worst: confidently wrong)

   All rewards are negative (GRPO normalises within group, so this is fine).

Format guard: penalise any completion that emits plaintext tool-call
artefacts in the final assistant message.

Utilities: compute_ece, compute_routing_entropy — post-hoc analysis on
           raw_trace_jsonl rows.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ..utils.io import append_jsonl
from .prompt import parse_final_answer


# ---------------------------------------------------------------------------
# Plaintext tool-call artefact detection
# ---------------------------------------------------------------------------

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>", re.IGNORECASE)
_TOOLS_TAG_RE = re.compile(r"<tools>", re.IGNORECASE)
_TOOL_CALLS_FIELD_RE = re.compile(r'"tool_calls"\s*:', re.IGNORECASE)
_TOOL_NAMES = ("extractor_tool", "reasoner_tool", "verifier_tool")


def _has_plaintext_tool_artifacts(text: str) -> bool:
    if not text:
        return False
    if _TOOL_CALL_TAG_RE.search(text):
        return True
    if _TOOLS_TAG_RE.search(text):
        return True
    if _TOOL_CALLS_FIELD_RE.search(text):
        return True
    for name in _TOOL_NAMES:
        if re.search(rf"\b{re.escape(name)}\s*[\(\{{:]", text, flags=re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------------------
# Completion parsing helpers
# ---------------------------------------------------------------------------

def _msg_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for blk in content:
            if isinstance(blk, dict) and "text" in blk:
                out.append(str(blk.get("text", "")))
        return "\n".join(out)
    return str(content)


def _extract_completion_stats(completion: Any) -> Dict[str, Any]:
    """Pull routing stats from a completion (TRL message-list format)."""
    if not isinstance(completion, list):
        text = _msg_text(completion)
        return {
            "last_assistant_text": text,
            "tool_calls": 0,
            "tool_msgs": 0,
            "tool_names_called": [],
            "last_msg_has_tool_calls": False,
            "last_msg_has_plaintext_artifacts": _has_plaintext_tool_artifacts(text),
        }

    assistant_msgs = [m for m in completion if isinstance(m, dict) and m.get("role") == "assistant"]
    tool_msgs = [m for m in completion if isinstance(m, dict) and m.get("role") == "tool"]

    tool_calls = 0
    tool_names_called: List[str] = []
    for m in assistant_msgs:
        tc = m.get("tool_calls")
        if isinstance(tc, list):
            tool_calls += len(tc)
            for entry in tc:
                fn = (entry.get("function", {}) or {}).get("name", "") if isinstance(entry, dict) else ""
                if fn:
                    tool_names_called.append(str(fn))

    last_text = ""
    last_has_tc = False
    if assistant_msgs:
        last_text = _msg_text(assistant_msgs[-1].get("content"))
        last_has_tc = bool(assistant_msgs[-1].get("tool_calls"))

    return {
        "last_assistant_text": last_text,
        "tool_calls": tool_calls,
        "tool_msgs": len(tool_msgs),
        "tool_names_called": tool_names_called,
        "last_msg_has_tool_calls": last_has_tc,
        "last_msg_has_plaintext_artifacts": _has_plaintext_tool_artifacts(last_text),
    }


def _ensure_list(x: Any, n: int) -> List[Any]:
    if isinstance(x, list):
        if len(x) == n:
            return x
        if not x:
            return [None] * n
        return (x * ((n // len(x)) + 1))[:n]
    return [x] * n


# ---------------------------------------------------------------------------
# CCR (Calibrated Confidence Routing) helpers
# ---------------------------------------------------------------------------

def _ccr_implicit_confidence(k: int, k_max: int, p_high: float, p_low: float) -> float:
    """Linear interpolation: k=0 → p_high (confident), k=k_max → p_low (uncertain)."""
    if k_max <= 0:
        return max(1e-7, min(1 - 1e-7, p_high))
    t = min(1.0, max(0.0, k / k_max))
    p = p_high + (p_low - p_high) * t
    return max(1e-7, min(1 - 1e-7, p))


def _ccr_log_reward(correct: bool, k: int, k_max: int, p_high: float, p_low: float) -> float:
    """Logarithmic scoring rule applied to implicit confidence from routing."""
    p = _ccr_implicit_confidence(k, k_max, p_high, p_low)
    return math.log(p) if correct else math.log(1.0 - p)


# ---------------------------------------------------------------------------
# Post-hoc analysis utilities
# ---------------------------------------------------------------------------

def compute_ece(
    records: List[Dict[str, Any]],
    k_max: int = 3,
    p_high: float = 0.9,
    p_low: float = 0.2,
) -> Dict[str, Any]:
    """Compute Expected Calibration Error from a list of trace records.

    Groups records by tool_calls count, computes accuracy per group, and
    compares to the implicit confidence p(k) from the CCR mapping.

    Args:
        records: list of dicts with at least "tool_calls" (int) and
                 "correct" (bool or 0/1) fields. Compatible with the
                 rows written to raw_trace_jsonl.
        k_max: maximum tool calls (default 3, matching the manager policy).
        p_high: CCR confidence for k=0 (should match training config).
        p_low: CCR confidence for k=k_max (should match training config).

    Returns:
        dict with "ece" (scalar), "buckets" (per-k stats), "n_total".
    """
    buckets: Dict[int, List[bool]] = defaultdict(list)
    for rec in records:
        k = int(rec.get("tool_calls", 0))
        correct = bool(rec.get("correct", rec.get("reward", 0) > 0))
        buckets[k].append(correct)

    n_total = sum(len(v) for v in buckets.values())
    if n_total == 0:
        return {"ece": 0.0, "buckets": {}, "n_total": 0}

    ece = 0.0
    bucket_stats: Dict[int, Any] = {}
    for k in range(k_max + 1):
        items = buckets.get(k, [])
        if not items:
            bucket_stats[k] = None
            continue
        acc = sum(items) / len(items)
        p = _ccr_implicit_confidence(k, k_max, p_high, p_low)
        weight = len(items) / n_total
        ece += weight * abs(p - acc)
        bucket_stats[k] = {
            "n": len(items),
            "accuracy": round(acc, 4),
            "implicit_confidence": round(p, 4),
            "calibration_gap": round(abs(p - acc), 4),
            "weight": round(weight, 4),
        }

    return {"ece": round(ece, 6), "buckets": bucket_stats, "n_total": n_total}


def compute_routing_entropy(records: List[Dict[str, Any]], k_max: int = 3) -> Dict[str, Any]:
    """Compute routing entropy H over the empirical tool-call distribution.

    Higher entropy = manager is less certain about how many tools to use.
    Very high entropy on a benchmark (like GPQA-Diamond) indicates the
    manager has learned that those questions are genuinely hard.

    Args:
        records: dicts with "tool_calls" int field.
        k_max: maximum tool calls (default 3).

    Returns:
        dict with "entropy" (nats), "distribution" (dict k→fraction),
        "n_total".
    """
    counts: Dict[int, int] = defaultdict(int)
    for rec in records:
        k = min(int(rec.get("tool_calls", 0)), k_max)
        counts[k] += 1

    n_total = sum(counts.values())
    if n_total == 0:
        return {"entropy": 0.0, "distribution": {}, "n_total": 0}

    entropy = 0.0
    distribution: Dict[str, float] = {}
    for k in range(k_max + 1):
        n_k = counts.get(k, 0)
        frac = n_k / n_total
        distribution[str(k)] = round(frac, 4)
        if frac > 0:
            entropy -= frac * math.log(frac)

    return {
        "entropy": round(entropy, 6),
        "max_entropy": round(math.log(k_max + 1), 6),
        "normalized_entropy": round(entropy / math.log(k_max + 1), 4) if k_max > 0 else 1.0,
        "distribution": distribution,
        "n_total": n_total,
    }


# ---------------------------------------------------------------------------
# Reward function builder
# ---------------------------------------------------------------------------

def build_reward_funcs(
    fail_buffer_jsonl: Optional[str] = None,
    raw_trace_jsonl: Optional[str] = None,
    routing_efficiency_bonus: float = 0.0,
    tool_use_bonus: float = 0.0,
    ccr_mode: bool = False,
    ccr_p_high: float = 0.9,
    ccr_p_low: float = 0.2,
    ccr_k_max: int = 3,
    is_main_process: bool = True,
):
    """Construct the reward function list passed to GRPOTrainer.

    Args:
        fail_buffer_jsonl: write every wrong/malformed sample here for the
                           evolve loop (rank 0 only).
        raw_trace_jsonl: log full per-completion stats here (rank 0 only).
        routing_efficiency_bonus: (binary mode only) small bonus per saved
                                   tool call when correct, e.g. 0.05.
        tool_use_bonus: (binary mode only) bonus when correct and >=1 tool used.
        ccr_mode: if True, replace binary reward with CCR log scoring rule.
                  routing_efficiency_bonus and tool_use_bonus are ignored.
        ccr_p_high: CCR implicit confidence when 0 tools called (default 0.9).
        ccr_p_low: CCR implicit confidence when k_max tools called (default 0.2).
        ccr_k_max: maximum tool calls; must match manager policy (default 3).
        is_main_process: only rank 0 should write the side-channel files.
    """

    def reward_fn(
        prompts=None,
        completions=None,
        ground_truth=None,
        example_id=None,
        choice_keys=None,
        **kwargs,
    ) -> List[float]:
        n = len(completions)
        gts = _ensure_list(ground_truth, n)
        eids = _ensure_list(example_id, n)
        ck_lists = _ensure_list(choice_keys, n)

        rewards: List[float] = []
        fail_rows: List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []

        for c, gt, eid, keys in zip(completions, gts, eids, ck_lists):
            stats = _extract_completion_stats(c)
            keys_list = list(keys) if isinstance(keys, (list, tuple)) else []
            pred = parse_final_answer(stats["last_assistant_text"], keys_list)

            valid_format = pred is not None
            no_artifacts = not stats["last_msg_has_plaintext_artifacts"]
            no_tc_in_final = not stats["last_msg_has_tool_calls"]
            base_correct = bool(valid_format and no_artifacts and no_tc_in_final and pred == gt)

            k = int(stats["tool_calls"])

            if ccr_mode:
                reward = _ccr_log_reward(base_correct, k, ccr_k_max, ccr_p_high, ccr_p_low)
                # Apply format penalty on top: malformed completions get floor reward
                if not (valid_format and no_artifacts and no_tc_in_final):
                    # Replace with the worst possible CCR reward (k=0 and wrong)
                    reward = math.log(1.0 - ccr_p_high + 1e-7)
            else:
                reward = 1.0 if base_correct else 0.0
                if base_correct and routing_efficiency_bonus > 0.0:
                    saved = max(0, ccr_k_max - k)
                    reward = reward + routing_efficiency_bonus * saved
                if base_correct and tool_use_bonus > 0.0 and k > 0:
                    reward = reward + tool_use_bonus

            rewards.append(float(reward))

            if not base_correct and is_main_process and fail_buffer_jsonl:
                fail_rows.append({
                    "ts": int(time.time()),
                    "example_id": int(eid) if eid is not None else None,
                    "ground_truth": gt,
                    "pred": pred,
                    "valid_format": bool(valid_format),
                    "no_artifacts": bool(no_artifacts),
                    "no_tc_in_final": bool(no_tc_in_final),
                    "tool_calls": k,
                    "tool_msgs": int(stats["tool_msgs"]),
                    "tool_names_called": list(stats["tool_names_called"]),
                    "last_assistant_text": stats["last_assistant_text"][:2000],
                })

            if is_main_process and raw_trace_jsonl:
                trace_rows.append({
                    "ts": int(time.time()),
                    "example_id": int(eid) if eid is not None else None,
                    "ground_truth": gt,
                    "pred": pred,
                    "correct": bool(base_correct),
                    "reward": float(reward),
                    "tool_calls": k,
                    "tool_msgs": int(stats["tool_msgs"]),
                    "tool_names_called": list(stats["tool_names_called"]),
                    "ccr_mode": bool(ccr_mode),
                    "implicit_confidence": round(
                        _ccr_implicit_confidence(k, ccr_k_max, ccr_p_high, ccr_p_low), 4
                    ) if ccr_mode else None,
                })

        if fail_rows and fail_buffer_jsonl:
            append_jsonl(fail_buffer_jsonl, fail_rows)
        if trace_rows and raw_trace_jsonl:
            append_jsonl(raw_trace_jsonl, trace_rows)

        return rewards

    reward_fn.__name__ = "ccr_log_scoring" if ccr_mode else "binary_outcome_with_format"
    return [reward_fn]


# ---------------------------------------------------------------------------
# Convenience bare-function export
# ---------------------------------------------------------------------------

def binary_outcome_reward(
    prompts=None,
    completions=None,
    ground_truth=None,
    example_id=None,
    choice_keys=None,
    **kwargs,
) -> List[float]:
    fn_list = build_reward_funcs()
    return fn_list[0](
        prompts=prompts,
        completions=completions,
        ground_truth=ground_truth,
        example_id=example_id,
        choice_keys=choice_keys,
        **kwargs,
    )
