"""User B: analyse a dataset the user already has.

``POST /api/v1/datasets/analyze`` accepts a multipart request carrying any of: an uploaded CSV
or Excel file, a dataset name, a registry link. One endpoint covers every case because the
caller should not have to know which evidence path applies; the response says which one ran.

* with a file, DataPilot parses it and every figure is computed from it;
* with a name or link, DataPilot researches the registries and reports only what they
  published, stating plainly that no rows were read.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core import progress
from app.core.logging_config import get_logger
from app.schemas.analysis_response import DatasetAnalysisResponse
from app.schemas.project import ErrorCode, ErrorDetail, ErrorResponse
from app.services.dataset_analysis import narrative
from app.services.research import offer as research_offer
from app.services.routing import transitions
from app.services.dataset_analysis.errors import AnalysisError
from app.services.dataset_analysis.service import AnalysisRequest, DatasetAnalysisService
from app.services.project_intelligence import ProjectIntelligenceService

logger = get_logger(__name__)

router = APIRouter(prefix="/datasets", tags=["Dataset Analysis (User B)"])

_service: Optional[DatasetAnalysisService] = None


_intelligence: Optional[ProjectIntelligenceService] = None


def get_intelligence_service() -> ProjectIntelligenceService:
    global _intelligence
    if _intelligence is None:
        _intelligence = ProjectIntelligenceService()
    return _intelligence


def get_analysis_service() -> DatasetAnalysisService:
    global _service
    if _service is None:
        _service = DatasetAnalysisService()
    return _service


def error_response(request_id: str, exc: AnalysisError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=ErrorResponse(
            request_id=request_id,
            error=ErrorDetail(code=exc.code, message=exc.message, details=exc.details),
        ).model_dump(),
    )


def internal_error(request_id: str) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail=ErrorResponse(
            request_id=request_id,
            error=ErrorDetail(code=ErrorCode.INTERNAL_ERROR, message="An internal error occurred."),
        ).model_dump(),
    )


async def _refine_target(outcome, goal: str):
    """Goal-aware target suggestion (one small Gemini call, names and types only). Never fails the analysis."""
    try:
        from app.core.config import get_settings
        from app.services.dataset_analysis.ai_target import refine_with_goal
        from app.services.llm_service import LLMService

        settings = get_settings()
        if not settings.analysis_gemini_enabled:
            return outcome, None
        return await refine_with_goal(outcome, goal, settings, LLMService(settings))
    except Exception as exc:
        logger.error("Goal-aware target step failed: %s", type(exc).__name__)
        return outcome, None


async def execute_analysis(
    request_id: str,
    request: AnalysisRequest,
    *,
    goal: Optional[str] = None,
    use_ai: bool = False,
    include_project_intelligence: bool = True,
) -> DatasetAnalysisResponse:
    """Run the User B workflow and package it (also used by /assist). Does not close the upload."""
    try:
        outcome = await get_analysis_service().analyze(request)
    except AnalysisError as exc:
        raise error_response(request_id, exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[%s] Unhandled dataset analysis error: %s", request_id, type(exc).__name__)
        raise internal_error(request_id) from exc

    notices: list[str] = []
    if goal and outcome.parsed is not None:
        progress.start("ai-target", "Matching your goal to a column")
        outcome, picked = await _refine_target(outcome, goal)
        progress.done("ai-target", f"suggested {picked}" if picked else "kept the detected columns")
        if picked:
            notices.append(f"'{picked}' was suggested as the target by an AI model from your goal and the column names; confirm it.")
    analysis = outcome.analysis
    intelligence = None
    if include_project_intelligence:
        try:
            intelligence = await get_intelligence_service().build(outcome, goal=goal, use_ai=use_ai)
        except Exception as exc:  # an enrichment must never fail the analysis the user asked for
            logger.error("[%s] Project intelligence failed: %s", request_id, type(exc).__name__)
            notices.append("Project intelligence could not be produced for this dataset; the analysis above is complete.")
    actions = narrative.next_actions(analysis)
    if (intelligence is not None and intelligence.compatibility is not None
            and intelligence.compatibility.suggests_alternative_dataset and goal):
        actions.append(transitions.alternative_dataset_action(goal, intelligence.compatibility))
    # Research is only OFFERED here; it runs when the user selects it (POST /api/v1/research/discover).
    actions.append(research_offer.offer_from_analysis(analysis, intelligence, goal))
    return DatasetAnalysisResponse(
        request_id=request_id,
        headline=narrative.headline(analysis),
        evidence_statement=narrative.evidence_statement(analysis),
        analysis=analysis,
        project_intelligence=intelligence,
        recommended_next_step=narrative.next_step(analysis),
        next_actions=actions,
        notices=notices + (intelligence.warnings if intelligence else []),
    )


@router.post(
    "/analyze",
    response_model=DatasetAnalysisResponse,
    status_code=200,
    summary="Analyze a dataset you already have (upload, name or link)",
    description=(
        "Send a `.csv`, `.tsv` or `.xlsx` file, **or** a `dataset_name`, **or** a `dataset_url` "
        "(Kaggle, Hugging Face, UCI or OpenML dataset page).\n\n"
        "**With a file** DataPilot parses it: schema, column statistics, data-quality findings, "
        "target candidates and supported ML tasks, all computed from your data "
        "(`evidence = datapilot_parsed`).\n\n"
        "**With a name or link** DataPilot researches the dataset registries and returns the "
        "metadata and schema they publish, with each fact labelled by how it is known "
        "(`registry_verified`, `inferred`, `unknown`). No dataset is downloaded and no row is read, "
        "so there are no statistics or quality checks in that case, and the response says so.\n\n"
        "A file takes precedence over a name or link. `target` names the prediction column "
        "yourself; an unknown column is a 422. `sheet_name` selects a worksheet."
    ),
    responses={
        413: {"description": "The upload is larger than the limit"},
        415: {"description": "Unsupported file type"},
        422: {"description": "The file, reference or target could not be used"},
        429: {"description": "A registry rate limit was reached"},
        502: {"description": "A registry was unavailable"},
        504: {"description": "Analysis or a registry timed out"},
    },
)
async def analyze_dataset(
    file: Optional[UploadFile] = File(None, description="A .csv, .tsv or .xlsx file."),
    dataset_name: Optional[str] = Form(None, max_length=300),
    dataset_url: Optional[str] = Form(None, max_length=2048),
    message: Optional[str] = Form(None, max_length=5000),
    source: Optional[str] = Form(None, max_length=50, description="kaggle | huggingface | uci | openml"),
    target: Optional[str] = Form(None, max_length=300, description="The column to treat as the target."),
    sheet_name: Optional[str] = Form(None, max_length=200),
    project_goal: Optional[str] = Form(
        None, max_length=2000,
        description="What you want to build. When given, the response says whether this dataset fits it."),
    use_ai: bool = Form(
        False, description="Let Gemini rephrase the project ideas (prose only; never changes a column)."),
    include_project_intelligence: bool = Form(
        True, description="Set false to skip feature roles, project ideas and model suggestions."),
) -> DatasetAnalysisResponse:
    request_id = str(uuid.uuid4())
    # Only metadata is logged: never a file name's contents, a cell, or the user's message.
    logger.info("[%s] POST /api/v1/datasets/analyze file=%s", request_id, file is not None and bool(file.filename))

    request = AnalysisRequest(
        stream=file.file if file is not None and file.filename else None,
        file_name=file.filename if file is not None else None,
        dataset_name=dataset_name,
        dataset_url=dataset_url,
        message=message,
        source_hint=source,
        target_hint=target,
        sheet_name=sheet_name,
    )
    try:
        return await execute_analysis(
            request_id, request, goal=project_goal, use_ai=use_ai,
            include_project_intelligence=include_project_intelligence)
    finally:
        if file is not None:
            await file.close()
