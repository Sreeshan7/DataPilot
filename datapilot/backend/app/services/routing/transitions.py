"""Connections between the two journeys.

User A and User B are not exclusive: someone who analyses their own dataset may then want a
better one, and someone shown a recommendation will want to analyse it. These functions turn
that into concrete next actions the client can offer, each naming the call that performs it.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.analysis_response import NextAction
from app.schemas.dataset import DatasetCandidate
from app.schemas.dataset_analysis import AnalysisMode, DatasetAnalysis
from app.schemas.routing import IntentResolution, UserIntent

RECOMMEND_ENDPOINT = "/api/v1/datasets/recommend"
ANALYZE_ENDPOINT = "/api/v1/datasets/analyze"


def after_analysis(
    analysis: DatasetAnalysis, routing: IntentResolution, goal: Optional[str]
) -> list[NextAction]:
    """User B -> User A: offered when the user said they may want another dataset, or
    when the one they gave could not be verified at all and they have a project to serve."""
    if not goal:
        return []
    reasons: list[str] = []
    if routing.follow_up_intent is UserIntent.USER_A_DATASET_DISCOVERY:
        reasons.append("you said you may want a different dataset")
    if analysis.mode is AnalysisMode.EXTERNAL_METADATA and analysis.external is None:
        reasons.append("that dataset could not be identified in any registry")
    if not reasons:
        return []
    return [NextAction(
        action="FIND_ALTERNATIVE_DATASET",
        description=f"Find other datasets for your project ({' and '.join(reasons)}).",
        endpoint=RECOMMEND_ENDPOINT,
        payload={"description": goal},
    )]


def alternative_dataset_action(goal: str, compatibility) -> NextAction:
    """User B -> User A because the dataset the user brought contradicts their project."""
    return NextAction(
        action="FIND_ALTERNATIVE_DATASET",
        description="This dataset does not fit your project: find datasets that do. " + compatibility.summary,
        endpoint=RECOMMEND_ENDPOINT,
        payload={"description": goal},
    )


def after_discovery(candidates: list[DatasetCandidate]) -> list[NextAction]:
    """User A -> User B: analyse a recommended dataset, or upload the one the user picks."""
    actions: list[NextAction] = []
    for candidate in candidates[:3]:
        if candidate.url and candidate.source_type.value == "verified_external":
            actions.append(NextAction(
                action="ANALYZE_RECOMMENDED_DATASET",
                description=f"Analyze {candidate.name} ({candidate.source}).",
                endpoint=ANALYZE_ENDPOINT,
                payload={"dataset_url": candidate.url},
            ))
    actions.append(NextAction(
        action="UPLOAD_FILE",
        description="Download the dataset you pick and upload it for row-level analysis.",
        endpoint=ANALYZE_ENDPOINT,
        payload={"file": "<the dataset file>"},
    ))
    return actions
