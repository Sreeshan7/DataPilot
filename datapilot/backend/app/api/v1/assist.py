"""The unified workflow: send what the user has, DataPilot picks the journey.

``POST /api/v1/assist`` accepts a message, an optional file, dataset name/link and project goal,
decides whether the user needs a dataset *found* (User A) or *analysed* (User B), runs that
journey, and returns the result together with the routing decision. When the evidence for the
two journeys is too close it runs neither and asks; it never guesses.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from app.api.v1.dataset_analysis import execute_analysis
from app.api.v1.datasets import execute_recommendation
from app.core.logging_config import get_logger
from app.schemas.assist import AssistResponse, AssistStatus
from app.schemas.routing import IntentRequest, UserIntent
from app.services.dataset_analysis.service import AnalysisRequest
from app.services.routing import IntentRouter
from app.services.research import offer as research_offer
from app.services.routing import transitions

logger = get_logger(__name__)

router = APIRouter(prefix="/assist", tags=["Unified Workflow"])

_router = IntentRouter()


@router.post(
    "",
    response_model=AssistResponse,
    status_code=200,
    summary="Describe your situation; DataPilot decides whether to find or analyse a dataset",
    description=(
        "Send any of: `message`, `project_goal`, a `file`, `dataset_name`, `dataset_url`. "
        "DataPilot reads them and picks a journey:\n\n"
        "* **USER_A_DATASET_DISCOVERY** - you need a dataset: it finds, verifies and ranks "
        "candidates for your project.\n"
        "* **USER_B_DATASET_ANALYSIS** - you have (or name) a dataset: it analyses the file, "
        "or researches the public registries when only a name or link is known.\n"
        "* **AMBIGUOUS** - the input could mean either; nothing runs and a question is returned.\n\n"
        "`routing` always explains the decision. `next_actions` carries transitions between the "
        "journeys, for example finding an alternative after analysing a dataset that is not enough."
    ),
    responses={
        413: {"description": "The upload is larger than the limit"},
        415: {"description": "Unsupported file type"},
        422: {"description": "The input could not be used"},
        429: {"description": "A source rate limit was reached"},
        502: {"description": "A source was unavailable"},
        504: {"description": "A source or the analysis timed out"},
    },
)
async def assist(
    message: Optional[str] = Form(None, max_length=5000),
    project_goal: Optional[str] = Form(None, max_length=5000, description="What you want to build."),
    file: Optional[UploadFile] = File(None, description="A .csv, .tsv or .xlsx file."),
    dataset_name: Optional[str] = Form(None, max_length=300),
    dataset_url: Optional[str] = Form(None, max_length=2048),
    source: Optional[str] = Form(None, max_length=50, description="kaggle | huggingface | uci | openml"),
    target: Optional[str] = Form(None, max_length=300),
    sheet_name: Optional[str] = Form(None, max_length=200),
    limit: int = Form(10, ge=1, le=50, description="Recommendations to return when finding a dataset."),
    use_ai: bool = Form(False, description="Let Gemini rephrase project ideas (prose only)."),
) -> AssistResponse:
    request_id = str(uuid.uuid4())
    has_file = file is not None and bool(file.filename)
    try:
        intent_request = IntentRequest(
            message=message,
            has_file=has_file,
            file_name=file.filename if has_file else None,
            dataset_name=dataset_name,
            dataset_url=dataset_url,
            project_goal=project_goal,
        )
        routing = _router.resolve(intent_request)
        # Only the decision is logged: never the user's message or file name.
        logger.info("[%s] POST /api/v1/assist intent=%s confidence=%.2f",
                    request_id, routing.intent.value, routing.confidence)

        if routing.intent is UserIntent.AMBIGUOUS:
            return AssistResponse(
                request_id=request_id, status=AssistStatus.NEEDS_CLARIFICATION, routing=routing,
                headline="DataPilot needs one more detail to choose between finding a dataset and analysing one.",
                prompt_to_user=routing.clarification,
            )

        if routing.required_next_input:
            journey = routing.intent
            headline = (
                "DataPilot understands you want a dataset analysed, but there is nothing to analyse yet."
                if journey is UserIntent.USER_B_DATASET_ANALYSIS
                else "DataPilot understands you need a dataset, but needs a fuller project description."
            )
            return AssistResponse(
                request_id=request_id, status=AssistStatus.NEEDS_INPUT, journey=journey, routing=routing,
                headline=headline, prompt_to_user=routing.required_next_input,
            )

        goal = project_goal or message
        if routing.intent is UserIntent.USER_B_DATASET_ANALYSIS:
            analysis_request = AnalysisRequest(
                stream=file.file if has_file else None,
                file_name=file.filename if has_file else None,
                dataset_name=dataset_name,
                dataset_url=dataset_url,
                message=message,
                source_hint=source,
                target_hint=target,
                sheet_name=sheet_name,
            )
            analysis = await execute_analysis(
                request_id, analysis_request, goal=project_goal, use_ai=use_ai)
            actions = list(analysis.next_actions)
            have = {a.action for a in actions}
            actions += [a for a in transitions.after_analysis(analysis.analysis, routing, project_goal)
                        if a.action not in have]
            return AssistResponse(
                request_id=request_id, status=AssistStatus.COMPLETED, journey=routing.intent, routing=routing,
                headline=analysis.headline, analysis=analysis, next_actions=actions,
            )

        discovery = await execute_recommendation(request_id, project_goal or message, limit)
        count = discovery.count
        return AssistResponse(
            request_id=request_id, status=AssistStatus.COMPLETED, journey=routing.intent, routing=routing,
            headline=(
                f"DataPilot found {count} candidate dataset{'s' if count != 1 else ''} for your project."
                if count else "DataPilot found no matching datasets for that project."
            ),
            discovery=discovery,
            next_actions=transitions.after_discovery([r.dataset for r in discovery.recommendations])
            + [research_offer.offer_from_requirements(project_goal or message, discovery.project_requirements)],
        )
    finally:
        if file is not None:
            await file.close()
