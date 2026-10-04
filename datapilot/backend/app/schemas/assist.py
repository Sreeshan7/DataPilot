"""Response model of the unified workflow, ``POST /api/v1/assist``.

One request in, one journey out, and the response says which journey ran and why, so a
client never has to choose between "find a dataset" and "analyse a dataset".
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.analysis_response import DatasetAnalysisResponse, NextAction
from app.schemas.recommendation import DatasetRecommendationResponse
from app.schemas.routing import IntentResolution, UserIntent


class AssistStatus(str, Enum):
    COMPLETED = "COMPLETED"
    #: Evidence for both journeys was too close; nothing ran, a question is returned.
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    #: The journey is clear but its input is missing (no file/name/link, or no project text).
    NEEDS_INPUT = "NEEDS_INPUT"


class AssistResponse(BaseModel):
    """Body of ``POST /api/v1/assist``."""

    success: bool = True
    request_id: str
    api_version: str = "v1"
    status: AssistStatus
    journey: Optional[UserIntent] = Field(
        None, description="The journey that ran; absent when nothing ran."
    )
    routing: IntentResolution = Field(
        ..., description="Why this journey was chosen, with the evidence and confidence."
    )
    headline: str
    prompt_to_user: Optional[str] = Field(
        None, description="The question or request to show the user when status is not COMPLETED."
    )
    analysis: Optional[DatasetAnalysisResponse] = Field(
        None, description="Set when USER_B_DATASET_ANALYSIS ran."
    )
    discovery: Optional[DatasetRecommendationResponse] = Field(
        None, description="Set when USER_A_DATASET_DISCOVERY ran."
    )
    next_actions: list[NextAction] = Field(
        default_factory=list,
        description="What can follow, including a transition into the other journey.",
    )
