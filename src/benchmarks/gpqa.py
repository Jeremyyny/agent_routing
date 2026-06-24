"""GPQA loader.

Loads the Idavidrein/gpqa dataset from HuggingFace.

GPQA has three configurations:
  - gpqa_main     (448 questions)
  - gpqa_diamond  (198 questions — hardest subset, recommended for eval)
  - gpqa_extended (546 questions)

NOTE: Idavidrein/gpqa is a gated dataset. You must accept the terms on
HuggingFace (https://huggingface.co/datasets/Idavidrein/gpqa) and run
`huggingface-cli login` before loading.

Each question has one correct answer and three incorrect answers. We
shuffle them into A/B/C/D using a per-example seeded RNG so the mapping
is deterministic but not trivially always "A".
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import StandardRow

HF_DEFAULT_DATASET = "Idavidrein/gpqa"
VALID_SUBSETS = ("gpqa_main", "gpqa_diamond", "gpqa_extended")
DEFAULT_SUBSET = "gpqa_diamond"


def _shuffle_answers(
    correct: str,
    incorrects: List[str],
    seed: int,
) -> Tuple[Dict[str, str], str]:
    """Shuffle correct + incorrect answers into A/B/C/D and return ground truth key."""
    answers = [correct] + incorrects
    rng = random.Random(seed)
    rng.shuffle(answers)
    choices = {chr(ord("A") + i): ans for i, ans in enumerate(answers)}
    gt = next(k for k, v in choices.items() if v == correct)
    return choices, gt


def _from_record(
    rec: Dict[str, Any],
    idx: int,
    subset: str,
    answer_seed: int,
) -> Optional[StandardRow]:
    question = str(rec.get("Question") or "").strip()
    correct = str(rec.get("Correct Answer") or "").strip()
    if not question or not correct:
        return None

    incorrects = []
    for i in range(1, 4):
        val = str(rec.get(f"Incorrect Answer {i}") or "").strip()
        if val:
            incorrects.append(val)

    if len(incorrects) < 3:
        return None

    # Per-example seed = global seed XOR idx, keeps shuffle deterministic
    choices, gt = _shuffle_answers(correct, incorrects, seed=answer_seed ^ idx)

    subdomain = str(rec.get("Subdomain") or rec.get("subdomain") or "").strip()
    domain = str(
        rec.get("High-level domain")
        or rec.get("Domain")
        or rec.get("domain")
        or ""
    ).strip()

    return StandardRow(
        example_id=idx,
        benchmark_name="gpqa",
        task_subtype=subset,
        question=question,
        choices=choices,
        ground_truth=gt,
        context="",
        metadata={"subdomain": subdomain, "domain": domain, "subset": subset},
        split="test",
    )


def load_gpqa(
    dataset_name: str = HF_DEFAULT_DATASET,
    subsets: "Sequence[str] | str" = DEFAULT_SUBSET,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    answer_seed: int = 42,
) -> List[StandardRow]:
    """Load GPQA into a list of StandardRow.

    Args:
        dataset_name: HuggingFace dataset id (must be accepted/gated).
        subsets: one of gpqa_main/gpqa_diamond/gpqa_extended, a comma
                 string of those names, or "all".
        hf_cache_dir: optional HuggingFace cache directory.
        max_examples: cap total examples; 0 means no cap.
        answer_seed: seed for answer shuffling to ensure deterministic A/B/C/D.
    """
    from datasets import load_dataset

    if isinstance(subsets, str):
        s = subsets.strip()
        if s.lower() == "all":
            subset_list = list(VALID_SUBSETS)
        else:
            subset_list = [x.strip() for x in s.split(",") if x.strip()]
    else:
        subset_list = [str(s).strip() for s in subsets if str(s).strip()]

    if not subset_list:
        subset_list = [DEFAULT_SUBSET]

    rows: List[StandardRow] = []
    for subset in subset_list:
        try:
            ds = load_dataset(dataset_name, subset, cache_dir=hf_cache_dir)
        except Exception as e:
            print(
                f"[LOAD_GPQA] WARNING: could not load subset '{subset}' from "
                f"'{dataset_name}'. "
                f"Make sure you accepted the dataset terms on HuggingFace and "
                f"are logged in (`huggingface-cli login`). Error: {e}"
            )
            continue

        for split_name in ds.keys():
            for rec in ds[split_name]:
                sr = _from_record(dict(rec), len(rows), subset, answer_seed)
                if sr is not None:
                    rows.append(sr)
                if max_examples > 0 and len(rows) >= max_examples:
                    break
            if max_examples > 0 and len(rows) >= max_examples:
                break
        if max_examples > 0 and len(rows) >= max_examples:
            break

    # Reassign contiguous example_ids
    for new_id, r in enumerate(rows):
        r.example_id = new_id

    return rows
