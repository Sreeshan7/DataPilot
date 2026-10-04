"""Public response models for ``POST /api/v1/datasets/recommend``.

Kept apart from ``schemas/dataset.py`` because a recommendation combines a ranked
``DatasetCandidate`` with its ``DatasetIntelligence``, and ``schemas/intelligence``
already depends on ``schemas/dataset``. Putting both in one module would make the
dependency circular.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.dataset import (
    DatasetCandidate,
    RankingFactor,
    SourceOutcome,
    SourceType,
    VerificationStatus,
)
from app.schemas.intelligence import CompatibilityLevel, DatasetIntelligence
from app.schemas.project import ProjectAnalysisData


class RankedDataset(BaseModel):
    """One entry of the ranked recommendation list."""

    rank: int = Field(..., ge=1)
    dataset: DatasetCandidate
    ranking_score: float = Field(..., ge=0.0, le=100.0)
    verification_status: VerificationStatus
    source_type: SourceType
    reasons: list[str] = Field(default_factory=list)
    ranking_factors: list[RankingFactor] = Field(default_factory=list)
    intelligence: Optional[DatasetIntelligence] = Field(
        None,
        description=(
            "What is known about this dataset and whether it can support the "
            "project: schema state, target state, feature coverage, "
            "compatibility, risks, limitations and a structured explanation."
        ),
    )
    confidence: Optional["RecommendationConfidence"] = Field(
        None,
        description="How sure DataPilot is that this dataset suits the project, and why.",
    )


class DatasetRecommendationResponse(BaseModel):
    """Body of ``POST /api/v1/datasets/recommend``."""

    success: bool = True
    request_id: str
    api_version: str = "v1"
    project_requirements: ProjectAnalysisData
    search_query: str
    count: int
    recommendations: list[RankedDataset]
    sources: list[SourceOutcome] = Field(default_factory=list)
    ranking_weights: dict[str, float] = Field(default_factory=dict)
    final_recommendation: Optional["FinalRecommendation"] = Field(
        None,
        description="The best match, alternatives and what to watch out for.",
    )
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 3C: recommendation confidence and the final recommendation
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """How sure DataPilot is that one dataset suits the project.

    Evidence-based and independent of rank: it answers "can we stand behind this
    dataset?", not "how does it compare?". A dataset can rank first and still be
    LOW because nothing about it could be confirmed.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationConfidence(BaseModel):
    """A confidence level together with every fact that set it."""

    level: ConfidenceLevel
    reasons: list[str] = Field(
        ...,
        min_length=1,
        description="Each reason states a fact that raised or capped the level.",
    )

    @field_validator("reasons")
    @classmethod
    def reasons_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if any(not r or not r.strip() for r in value):
            raise ValueError("a confidence reason must not be blank")
        return value


class RecommendationOutcome(str, Enum):
    RECOMMENDED = "recommended"
    """A registry-confirmed dataset with at least medium confidence."""

    TENTATIVE = "tentative"
    """Something is worth trying, but it is unconfirmed or low confidence."""

    NONE = "none"
    """No candidate could be shown to support the project."""


class RecommendedDataset(BaseModel):
    """The single dataset DataPilot puts forward, with the evidence behind it."""

    dataset_id: str
    name: str
    source: str
    source_type: SourceType
    url: Optional[str] = None
    rank: int = Field(..., ge=1)
    ranking_score: float = Field(..., ge=0.0, le=100.0)
    compatibility: CompatibilityLevel
    confidence: RecommendationConfidence
    why: list[str] = Field(default_factory=list)
    verified: list[str] = Field(default_factory=list)
    unknown: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    watch_out_for: list[str] = Field(default_factory=list)


class AlternativeDataset(BaseModel):
    """A runner-up and the single most important reason it is not the best match."""

    dataset_id: str
    name: str
    source: str
    rank: int = Field(..., ge=1)
    ranking_score: float = Field(..., ge=0.0, le=100.0)
    compatibility: CompatibilityLevel
    verification_status: VerificationStatus
    trade_off: str = Field(..., min_length=1)


class FinalRecommendation(BaseModel):
    """The answer to "which dataset should I use?".

    Rendered from the structured evidence record only; no generative model is
    asked what a dataset contains. ``summary`` is a deterministic sentence built
    from the same fields, so it can never assert more than they do.
    """

    outcome: RecommendationOutcome
    summary: str = Field(..., min_length=1)
    best_match: Optional[RecommendedDataset] = None
    alternatives: list[AlternativeDataset] = Field(default_factory=list)
    margin_to_next: Optional[float] = Field(
        None,
        description="Ranking-score gap between the best match and the next candidate.",
    )
    close_call: bool = False
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def outcome_must_agree_with_the_evidence(self) -> "FinalRecommendation":
        best = self.best_match
        if self.outcome is RecommendationOutcome.NONE:
            if best is not None:
                raise ValueError("a NONE outcome cannot name a best match")
            return self
        if best is None:
            raise ValueError(f"a {self.outcome.value} outcome needs a best match")
        if self.outcome is RecommendationOutcome.RECOMMENDED:
            if best.source_type is SourceType.AI_SUGGESTED:
                raise ValueError("an unconfirmed AI suggestion cannot be RECOMMENDED")
            if best.confidence.level is ConfidenceLevel.LOW:
                raise ValueError("a LOW-confidence dataset cannot be RECOMMENDED")
            if best.compatibility is CompatibilityLevel.LOW:
                raise ValueError("an incompatible dataset cannot be RECOMMENDED")
        return self


RankedDataset.model_rebuild()
DatasetRecommendationResponse.model_rebuild()
