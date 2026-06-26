"""Pydantic schemas for the three subagents.

Critical invariant: NO subagent output may contain the final answer label
or the choice text of the ground truth choice. The leakage auditor enforces
this at synthesis time. Subagents produce decision-relevant SIGNALS only;
the manager is the sole authority on the final ANSWER_<TOKEN> output.
"""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator


class AgentKind(str, Enum):
    EXTRACTOR = "extractor"
    REASONER = "reasoner"
    VERIFIER = "verifier"


# ---------- Extractor ----------

class ExtractedEvidence(BaseModel):
    text: str
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
    key_evidence: List[ExtractedEvidence] = Field(default_factory=list)
    extracted_facts: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    context_summary: str = Field(default="")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------- Reasoner ----------

class CandidateConsideration(BaseModel):
    choice_key: str
    relevant_if: List[str] = Field(default_factory=list)
    less_relevant_if: List[str] = Field(default_factory=list)


class ReasonerOutput(BaseModel):
    case_facts: List[str] = Field(default_factory=list)
    task_type: str = Field(default="")
    decision_factors: List[str] = Field(default_factory=list)
    knowledge_slots: List[str] = Field(default_factory=list)
    candidate_considerations: List[CandidateConsideration] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)
    format_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------- Verifier ----------

class RelevantPrinciple(BaseModel):
    principle: str
    source: str = Field(default="")


class VerifierCheck(BaseModel):
    check: str
    status: str = Field(..., description="pass | fail | unclear")
    note: str = Field(default="")

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        v = (v or "").strip().lower()
        return v if v in {"pass", "fail", "unclear"} else "unclear"


class VerifierOutput(BaseModel):
    relevant_principles: List[RelevantPrinciple] = Field(default_factory=list)
    checks: List[VerifierCheck] = Field(default_factory=list)
    potential_errors: List[str] = Field(default_factory=list)
    uncertainty_notes: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


SCHEMA_REGISTRY = {
    AgentKind.EXTRACTOR: ExtractorOutput,
    AgentKind.REASONER: ReasonerOutput,
    AgentKind.VERIFIER: VerifierOutput,
}
