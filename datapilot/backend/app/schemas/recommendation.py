"""Public response models for ``POST /api/v1/datasets/recommend``.

Kept apart from ``schemas/dataset.py`` because a recommendation combines a ranked
``DatasetCandidate`` with its ``DatasetIntelligence``, and ``schemas/intelligence``
already depends on ``schemas/dataset``. Putting both in one module would make the
dependency circular.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.dataset import (
    DatasetCandidate,
    RankingFactor,
    SourceOutcome,
    SourceType,
    VerificationStatus,
)
from app.schemas.intelligence import DatasetIntelligence
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
    meta: dict[str, Any] = Field(default_factory=dict)
