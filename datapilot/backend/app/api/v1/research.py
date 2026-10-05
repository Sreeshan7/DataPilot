"""POST /api/v1/research/discover: the explicit, user-selected research step.

Never called by /assist or /analyze: those only *offer* it (a ``RESEARCH_PROBLEM`` next action carrying a
pre-filled request). Running it costs a few web requests and, if ``options.use_ai`` is true, one Gemini call.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from app.core.logging_config import get_logger
from app.schemas.project import ErrorCode, ErrorDetail, ErrorResponse
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.research import ResearchService

logger = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])
_service: Optional[ResearchService] = None


def get_research_service() -> ResearchService:
    global _service
    if _service is None:
        _service = ResearchService()
    return _service


@router.post(
    "/discover",
    response_model=ResearchResponse,
    summary="Research recent papers and methods for this problem (optional, user-selected)",
    description=(
        "Builds 3-5 focused queries from the structured project context, searches the web for academic "
        "sources, deduplicates and ranks them with an explainable score, extracts the methods and reported "
        "figures their retrieved text names, and recommends algorithms by combining literature support with "
        "compatibility to your task. Every paper has a URL and an evidence label. With `options.use_ai`, ONE "
        "Gemini call adds a clearly labelled interpretation; if it fails or is unsupported by the evidence the "
        "verified results are returned without it. Papers are never invented: when search yields nothing the "
        "status says so."
    ),
)
async def discover(body: ResearchRequest) -> ResearchResponse:
    try:
        # Only counts and statuses are logged: never the project text or retrieved content.
        response = await get_research_service().discover(body)
        logger.info("[%s] research status=%s papers=%d gemini_calls=%d", response.request_id, response.status.value,
                    len(response.papers), response.gemini.calls_made)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled research error: %s", type(exc).__name__)
        raise HTTPException(status_code=500, detail=ErrorResponse(
            error=ErrorDetail(code=ErrorCode.INTERNAL_ERROR, message="An internal error occurred.")).model_dump()) from exc
