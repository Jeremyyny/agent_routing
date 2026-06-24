"""MMLU-Pro loader.

Loads the TIGER-Lab/MMLU-Pro dataset from HuggingFace (public, no gating).

MMLU-Pro has 10 options per question (A–J), compared to 4 in standard MMLU,
making it substantially harder and better suited for evaluating routing
calibration under genuine uncertainty.

Splits: "test" (~12k questions) and "validation" (~70 questions).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .base import StandardRow, normalize_choices

HF_DEFAULT_DATASET = "TIGER-Lab/MMLU-Pro"

# Standard MMLU-Pro category names
ALL_CATEGORIES = (
    "math", "physics", "chemistry", "biology", "computer science",
    "engineering", "economics", "psychology", "history", "philosophy",
    "law", "business", "health", "other",
)


def _from_record(rec: Dict[str, Any], idx: int) -> Optional[StandardRow]:
    question = str(rec.get("question") or "").strip()
    options = rec.get("options") or []
    if not question or not options:
        return None

    choices = normalize_choices(list(options))
    if len(choices) < 2:
        return None

    # answer is a letter like "A"; answer_index is 0-based
    answer_raw = str(rec.get("answer") or "").strip().upper()
    answer_idx = rec.get("answer_index")

    gt = ""
    if answer_raw and answer_raw in choices:
        gt = answer_raw
    elif answer_idx is not None:
        keys = list(choices.keys())
        try:
            i = int(answer_idx)
            if 0 <= i < len(keys):
                gt = keys[i]
        except (TypeError, ValueError):
            pass

    if not gt:
        return None

    category = str(rec.get("category") or "").strip()
    src = str(rec.get("src") or "").strip()
    split = str(rec.get("_source_split") or rec.get("split") or "").lower().strip()
    if split == "validation":
        split = "dev"
    if not split:
        split = "test"

    return StandardRow(
        example_id=idx,
        benchmark_name="mmlu_pro",
        task_subtype=category,
        question=question,
        choices=choices,
        ground_truth=gt,
        context="",
        metadata={"category": category, "src": src, "n_options": len(choices)},
        split=split,
    )


def load_mmlu_pro(
    dataset_name: str = HF_DEFAULT_DATASET,
    categories: "Sequence[str] | str" = (),
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    splits: "Sequence[str] | str" = ("test", "validation"),
) -> List[StandardRow]:
    """Load MMLU-Pro into a list of StandardRow.

    Args:
        dataset_name: HuggingFace dataset id.
        categories: comma string or list of category names to keep,
                    e.g. "math,physics". Empty means all categories.
        hf_cache_dir: optional HuggingFace cache directory.
        max_examples: cap total examples; 0 means no cap.
        splits: which HF splits to load.
    """
    from datasets import load_dataset

    if isinstance(categories, str):
        s = categories.strip()
        cat_filter = {x.strip().lower() for x in s.split(",") if x.strip()}
    else:
        cat_filter = {str(c).strip().lower() for c in categories if str(c).strip()}

    if isinstance(splits, str):
        split_list = [splits]
    else:
        split_list = list(splits)

    ds = load_dataset(dataset_name, cache_dir=hf_cache_dir)

    rows: List[StandardRow] = []
    for split_name in split_list:
        if split_name not in ds:
            # Also try "validation" as alias for "dev"
            alt = "validation" if split_name == "dev" else ("dev" if split_name == "validation" else None)
            if alt and alt in ds:
                split_name = alt
            else:
                continue

        for rec in ds[split_name]:
            rec_dict = dict(rec)
            rec_dict["_source_split"] = split_name
            if cat_filter:
                cat = str(rec_dict.get("category") or "").strip().lower()
                if cat not in cat_filter:
                    continue
            sr = _from_record(rec_dict, len(rows))
            if sr is not None:
                rows.append(sr)
            if max_examples > 0 and len(rows) >= max_examples:
                break
        if max_examples > 0 and len(rows) >= max_examples:
            break

    # Reassign contiguous example_ids
    for new_id, r in enumerate(rows):
        r.example_id = new_id

    return rows
