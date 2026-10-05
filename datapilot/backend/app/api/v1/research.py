"""Research endpoints: the explicit, user-selected research step.

Never called by /assist or /analyze: those only *offer* it (a ``RESEARCH_PROBLEM`` next action carrying a
pre-filled request). Running it costs a few web requests and, with the AI options, a few small Gemini calls.

* ``POST /research/discover``: one JSON answer when everything is done.
* ``POST /research/stream``: the same work as Server-Sent Events: a ``plan`` event, a ``step`` event each time a
  stage really starts/finishes, then one ``result`` event holding the identical response body (or ``error``).
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.logging_config import get_logger
from app.schemas.project import ErrorCode, ErrorDetail, ErrorResponse
from app.schemas.research import ResearchRequest, ResearchResponse
from app.services.research import ResearchService
from app.services.research.progress import Progress

logger = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["Research"])
_service: Optional[ResearchService] = None

HEARTBEAT_SECONDS = 10.0


def get_research_service() -> ResearchService:
    global _service
    if _service is None:
        _service = ResearchService()
    return _service


def _internal_error() -> HTTPException:
    return HTTPException(status_code=500, detail=ErrorResponse(
        error=ErrorDetail(code=ErrorCode.INTERNAL_ERROR, message="An internal error occurred.")).model_dump())


@router.post(
    "/discover",
    response_model=ResearchResponse,
    summary="Research recent papers and methods for this problem (optional, user-selected)",
    description=(
        "Understands the question (core topics, context, typo repairs), searches OpenAlex, arXiv and Crossref "
        "(web search only as a fallback), deduplicates, keeps only papers that are about the subject, ranks them "
        "with an explainable score, extracts the methods and reported figures their retrieved text names, and "
        "recommends algorithms by combining literature support with compatibility to your task. Every paper has a "
        "URL and an evidence label. Optional AI steps (question understanding, relevance judging, interpretation) "
        "are validated and fall back silently. Papers are never invented: when search yields nothing the status "
        "says so."
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
        raise _internal_error() from exc


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/stream",
    summary="Research with live progress (Server-Sent Events)",
    description=(
        "Same work and same final body as `/research/discover`, delivered as a text/event-stream: `plan` (the stages), "
        "`step` (id, state running|done|skipped, detail) as each stage really happens, then `result` (the "
        "ResearchResponse) or `error`. Stage events carry counts only."
    ),
    response_class=StreamingResponse,
)
async def stream(body: ResearchRequest) -> StreamingResponse:
    service = get_research_service()
    queue: asyncio.Queue = asyncio.Queue()

    async def listener(event: dict) -> None:
        await queue.put(("progress", event))

    async def work() -> None:
        try:
            response = await service.discover(body, Progress(listener))
            logger.info("[%s] research(stream) status=%s papers=%d", response.request_id, response.status.value,
                        len(response.papers))
            await queue.put(("result", json.loads(response.model_dump_json())))
        except Exception as exc:
            logger.error("Unhandled research error: %s", type(exc).__name__)
            await queue.put(("error", {"message": "An internal error occurred."}))

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(work())
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # keeps proxies from closing a quiet connection
                    continue
                if kind == "progress":
                    yield _sse(payload["type"], payload)
                else:
                    yield _sse(kind, payload)
                    return
        finally:
            if not task.done():
                task.cancel()  # the client went away: stop working

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
