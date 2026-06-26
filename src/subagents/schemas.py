"""Pydantic schemas for the three subagents.

Critical invariant: NO subagent output may contain the final answer label
or the choice text of the ground truth choice. The leakage auditor enforces
this at synthesis time. Subagents produce decision-relevant SIGNALS only;
the manager is the sole authority on the final ANSWER_<TOKEN> output.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class AgentKind(str, Enum):
    EXTRACTOR = "extractor"
    REASONER = "reasoner"
    VERIFIER = "verifier"


# ---------- Extractor ----------

class ExtractedEvidence(BaseModel):
    text: str = Field(..., max_length=400)
    relevance: float = Field(..., ge=0.0, le=1.0)
    polarity: str = Field(..., description="support | oppose | neutral")

    @field_validator("polarity")
    @classmethod
    def _check_polarity(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"support", "oppose", "neutral"}:
            return "neutral"
        return v


class ExtractorOutput(BaseModel):
    """Output schema for ExtractorAgent.

    On MedQA (closed-book): `key_evidence` is empty, `extracted_facts` carries
    clinical facts pulled from the question stem.
    On PubMedQA / LegalBench / LawBench: `key_evidence` carries the salient
    sentences from the long context.
    """
    key_evidence: List[ExtractedEvidence] = Field(default_factory=list, max_length=8)
    extracted_facts: List[str] = Field(default_factory=list, max_length=12)
    missing_info: List[str] = Field(default_factory=list, max_length=6)
    context_summary: str = Field(default="", max_length=400)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("extracted_facts", "missing_info")
    @classmethod
    def _trim_lines(cls, v: List[str]) -> List[str]:
        return [str(s)[:240].strip() for s in v if str(s).strip()]


# ---------- Reasoner ----------

class CandidateConsideration(BaseModel):
    """Neutral per-choice conditions, used when the input is MCQ.

    This is intentionally weaker than support/against. The goal is to teach a
    small subagent to map each option to conditions under which it matters,
    without making one option read like the answer.
    """
    choice_key: str = Field(..., max_length=16)
    relevant_if: List[str] = Field(default_factory=list, max_length=3)
    less_relevant_if: List[str] = Field(default_factory=list, max_length=3)

    @field_validator("relevant_if", "less_relevant_if")
    @classmethod
    def _trim_conditionals(cls, v: List[str]) -> List[str]:
        return [str(s)[:180].strip() for s in v if str(s).strip()]


class ReasonerOutput(BaseModel):
    case_facts: List[str] = Field(default_factory=list, max_length=8)
    task_type: str = Field(default="", max_length=64)
    decision_factors: List[str] = Field(default_factory=list, max_length=8)
    knowledge_slots: List[str] = Field(default_factory=list, max_length=6)
    candidate_considerations: List[CandidateConsideration] = Field(default_factory=list, max_length=12)
    missing_information: List[str] = Field(default_factory=list, max_length=4)
    format_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("case_facts", "decision_factors", "knowledge_slots", "missing_information")
    @classmethod
    def _trim_lines(cls, v: List[str]) -> List[str]:
        return [str(s)[:220].strip() for s in v if str(s).strip()]


# ---------- Verifier ----------

class RelevantPrinciple(BaseModel):
    principle: str = Field(..., max_length=200)
    source: str = Field(default="", max_length=120)


class VerifierCheck(BaseModel):
    check: str = Field(..., max_length=200)
    status: str = Field(..., description="pass | fail | unclear")
    note: str = Field(default="", max_length=200)

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        v = (v or "").strip().lower()
        return v if v in {"pass", "fail", "unclear"} else "unclear"


class VerifierOutput(BaseModel):
    relevant_principles: List[RelevantPrinciple] = Field(default_factory=list, max_length=6)
    checks: List[VerifierCheck] = Field(default_factory=list, max_length=8)
    potential_errors: List[str] = Field(default_factory=list, max_length=4)
    uncertainty_notes: List[str] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("potential_errors", "uncertainty_notes")
    @classmethod
    def _trim(cls, v: List[str]) -> List[str]:
        return [str(s)[:200].strip() for s in v if str(s).strip()]


SCHEMA_REGISTRY = {
    AgentKind.EXTRACTOR: ExtractorOutput,
    AgentKind.REASONER: ReasonerOutput,
    AgentKind.VERIFIER: VerifierOutput,
}
