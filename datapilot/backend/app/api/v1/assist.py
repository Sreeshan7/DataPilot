"""The unified workflow: send what the user has, DataPilot picks the journey.

``POST /api/v1/assist`` accepts a message, an optional file, dataset name/link and project goal,
decides whether the user needs a dataset *found* (User A) or *analysed* (User B), runs that
journey, and returns the result together with the routing decision. When the evidence for the
two journeys is too close it runs neither and asks; it never guesses.

``POST /api/v1/assist/stream`` does exactly the same work and returns the same final body, as
Server-Sent Events: ``plan`` (the stages), ``step`` events as each stage really starts and finishes,
then ``result`` (or ``error``). Stage events carry counts only, never user text or file contents.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.v1.dataset_analysis import execute_analysis
from app.api.v1.datasets import execute_recommendation
from app.core import progress
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

HEARTBEAT_SECONDS = 10.0

_ANALYSIS_WORDS = re.compile(r"\b(?:analy[sz]e|look\s+at|check\s+out|explore|inspect|examine|profile|review|tell\s+me\s+about|understand)\b", re.I)

_JOURNEY_LABEL = {
    UserIntent.USER_A_DATASET_DISCOVERY: "finding a dataset",
    UserIntent.USER_B_DATASET_ANALYSIS: "analysing a dataset",
}


def _plan_for(intent: UserIntent, has_file: bool, has_goal: bool) -> list[tuple[str, str]]:
    """The stages this journey really performs (dynamic ones, such as each source searched, are added as they happen)."""
    if intent is UserIntent.USER_A_DATASET_DISCOVERY:
        return [("requirements", "Extracting your project requirements"), ("merge", "Removing duplicates across sources"),
                ("rank", "Comparing candidates with your project"), ("enrich", "Reading details of the best matches"),
                ("compat", "Checking compatibility with your project")]
    if has_file:
        steps = [("read", "Reading your file"), ("profile", "Profiling every column"),
                 ("targets", "Looking for what could be predicted"), ("quality", "Checking data quality"),
                 ("tasks", "Working out which tasks fit")]
        if has_goal:
            steps.append(("ai-target", "Matching your goal to a column"))
        return steps + [("signal", "Measuring which columns carry signal"), ("ideas", "Drafting project ideas from your columns")]
    return [("lookup", "Looking the dataset up in public registries"), ("ideas", "Drafting project ideas from your columns")]


async def _name_from_message(message: Optional[str]) -> Optional[str]:
    """A dataset name the rules could not find, read by AI from the message (validated against the user's own words)."""
    try:
        from app.core.config import get_settings
        from app.services.dataset_analysis.name_ai import extract_name
        from app.services.llm_service import LLMService

        settings = get_settings()
        if not settings.analysis_gemini_enabled:
            return None
        return await extract_name(LLMService(settings), settings, message or "")
    except Exception as exc:  # optional: the user is simply asked for the name instead
        logger.error("Dataset-name step failed: %s", type(exc).__name__)
        return None


async def _execute(
    message: Optional[str], project_goal: Optional[str], file: Optional[UploadFile], dataset_name: Optional[str],
    dataset_url: Optional[str], source: Optional[str], target: Optional[str], sheet_name: Optional[str], limit: int,
    use_ai: bool,
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
        wants_analysis = bool(message and not project_goal and _ANALYSIS_WORDS.search(message))
        if (not has_file and not dataset_url and not dataset_name and (message or project_goal) and (
                (routing.intent is UserIntent.USER_B_DATASET_ANALYSIS and routing.required_next_input)
                or (routing.intent is UserIntent.AMBIGUOUS and wants_analysis))):  # "analyse LIAR pls"
            progress.start("name", "Working out which dataset you mean")
            found = await _name_from_message(message or project_goal)
            progress.done("name", found if found else "no dataset name found")
            if found:
                dataset_name = found
                intent_request = intent_request.model_copy(update={"dataset_name": found})
                routing = _router.resolve(intent_request)
        # Only the decision is logged: never the user's message or file name.
        logger.info("[%s] POST /api/v1/assist intent=%s confidence=%.2f",
                    request_id, routing.intent.value, routing.confidence)
        label = _JOURNEY_LABEL.get(routing.intent)
        progress.done("route", f"{label} · {round(routing.confidence * 100)}% sure" if label else "needs one more detail",
                      label="Understanding your request")

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
        progress.plan(_plan_for(routing.intent, has_file, bool(project_goal)))
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


_RESPONSES = {
    413: {"description": "The upload is larger than the limit"},
    415: {"description": "Unsupported file type"},
    422: {"description": "The input could not be used"},
    429: {"description": "A source rate limit was reached"},
    502: {"description": "A source was unavailable"},
    504: {"description": "A source or the analysis timed out"},
}


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
    responses=_RESPONSES,
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
    return await _execute(message, project_goal, file, dataset_name, dataset_url, source, target, sheet_name, limit, use_ai)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/stream",
    summary="Same as /assist, with live progress (Server-Sent Events)",
    description=(
        "Same inputs, same work and same final body as `POST /assist`, delivered as a text/event-stream: `plan`, "
        "`step` (id, state running|done|skipped, optional label and detail) as each stage really happens, then "
        "`result` (the AssistResponse) or `error` ({status, detail})."
    ),
    response_class=StreamingResponse,
    responses=_RESPONSES,
)
async def assist_stream(
    message: Optional[str] = Form(None, max_length=5000),
    project_goal: Optional[str] = Form(None, max_length=5000),
    file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None, max_length=300),
    dataset_url: Optional[str] = Form(None, max_length=2048),
    source: Optional[str] = Form(None, max_length=50),
    target: Optional[str] = Form(None, max_length=300),
    sheet_name: Optional[str] = Form(None, max_length=200),
    limit: int = Form(10, ge=1, le=50),
    use_ai: bool = Form(False),
) -> StreamingResponse:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    loop_thread = threading.get_ident()

    def sink(event: dict) -> None:
        # From the loop's own thread the event goes straight in, so it can never be overtaken by the final result;
        # from a worker thread it is handed over through the loop (those events are queued before the worker returns).
        if threading.get_ident() == loop_thread:
            queue.put_nowait(("progress", event))
        else:
            loop.call_soon_threadsafe(queue.put_nowait, ("progress", event))

    async def work() -> None:
        with progress.bind(sink):
            try:
                response = await _execute(message, project_goal, file, dataset_name, dataset_url, source, target,
                                          sheet_name, limit, use_ai)
                await queue.put(("result", json.loads(response.model_dump_json(by_alias=True))))
            except HTTPException as exc:
                await queue.put(("error", {"status": exc.status_code, "detail": exc.detail}))
            except Exception as exc:
                logger.error("Unhandled assist error: %s", type(exc).__name__)
                await queue.put(("error", {"status": 500, "detail": None}))

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(work())
        try:
            yield ": " + " " * 2048 + "\n\n"  # browsers hold back the first ~1 KB of an XHR response: flush it so events show at once
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if kind == "progress":
                    yield _sse(payload["type"], payload)
                else:
                    yield _sse(kind, payload)
                    return
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
