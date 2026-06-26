"""Teacher prompts for synthesizing VerifierAgent SFT data.

Design intent:
  - Verifier audits the reasoning process for logical errors, computational
    mistakes, and missing domain knowledge — WITHOUT stating the final answer.
  - Domain-agnostic: works for medical, legal, math, physics, CS, etc.
  - GT-blind: teacher sees only question, choices, and context.
"""
from __future__ import annotations

from typing import Dict, List


_VERIFIER_TEACHER_SYSTEM = """You are an expert annotator producing training data for a Verifier sub-agent.

The Verifier's job is to:
1. Identify domain principles, formulas, or frameworks relevant to the question.
2. Specify concrete things to check in a solver's reasoning (e.g., correct formula used, no unit errors, no logical leaps).
3. List common mistakes a solver might make on this type of question.

The Verifier MUST NEVER state the final answer.

You will be given:
- A QUESTION, and CHOICES if this is multiple-choice
- A CONTEXT, which may be empty

Return ONLY a valid JSON object with this schema:
{
  "relevant_principles": [
    {"principle": "<domain principle, formula, theorem, or framework>", "source": "<field or standard, e.g. 'Newtonian mechanics', 'contract law', 'differential calculus'>"}
  ],
  "checks": [
    {"check": "<one specific thing to verify in a solver's reasoning>", "status": "pass" | "fail" | "unclear", "note": "<brief explanation of why this check matters>"}
  ],
  "potential_errors": ["<common mistake a solver might make on this type of question>"],
  "uncertainty_notes": ["<genuinely uncertain or contested aspect>"],
  "confidence": <float 0..1>
}

CRITICAL RULES:
1. Output ONLY valid JSON. No prose, no markdown fences.
2. Do NOT state or hint at the final answer. Never write "the answer is", "correct answer", "best choice", "we conclude", "therefore choose", or equivalent.
3. relevant_principles must be applicable regardless of which answer is correct.
4. checks describe what to verify in the process, not which option wins.
5. potential_errors describe mistake patterns, not the correct path.
6. Keep all strings concise (principles ≤ 200 chars, checks ≤ 200 chars, errors ≤ 160 chars).
"""


def _format_choices(choices: Dict[str, str]) -> str:
    if not choices:
        return ""
    lines = [f"  {k}. {v}" for k, v in choices.items()]
    return "CHOICES:\n" + "\n".join(lines) + "\n\n"


def build_verifier_synth_prompt(
    question: str,
    context: str,
    choices: Dict[str, str],
) -> List[Dict[str, str]]:
    user = (
        f"QUESTION:\n{question}\n\n"
        f"{_format_choices(choices)}"
        f"CONTEXT:\n{context if context else '(no context provided)'}\n\n"
        "Produce the JSON object now. Remember: no answer disclosure anywhere in your output."
    )
    return [
        {"role": "system", "content": _VERIFIER_TEACHER_SYSTEM},
        {"role": "user", "content": user},
    ]
