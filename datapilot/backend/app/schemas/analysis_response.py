"""Public response models for dataset analysis (User B)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.dataset_analysis import DatasetAnalysis


class NextAction(BaseModel):
    """Something the client can offer the user next, with the call that performs it."""

    action: str = Field(..., description="Stable identifier, e.g. 'UPLOAD_FILE'.")
    description: str
    endpoint: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)


class DatasetAnalysisResponse(BaseModel):
    """Body of ``POST /api/v1/datasets/analyze``."""

    success: bool = True
    request_id: str
    api_version: str = "v1"
    headline: str = Field(
        ...,
        description="One sentence saying exactly what DataPilot did: analysed your file, or only "
        "found public metadata.",
    )
    evidence_statement: str = Field(
        ..., description="What every figure below rests on, and what was not verified."
    )
    analysis: DatasetAnalysis
    recommended_next_step: str
    next_actions: list[NextAction] = Field(default_factory=list)
