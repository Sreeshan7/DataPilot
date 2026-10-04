"""Intent routing endpoint: which journey does this input belong to?

``POST /api/v1/intent/resolve`` lets a client send whatever the user typed (and say
whether a file came with it) without knowing which workflow to call. It decides only; it
does not run the journey, so it is free, instant and never touches an external service.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.core.logging_config import get_logger
from app.schemas.routing import IntentRequest, IntentResolution
from app.services.routing import IntentRouter

logger = get_logger(__name__)

router = APIRouter(prefix="/intent", tags=["Intent Routing"])

_router = IntentRouter()


class IntentResolveResponse(IntentResolution):
    request_id: str
    api_version: str = "v1"


@router.post(
    "/resolve",
    response_model=IntentResolveResponse,
    summary="Decide whether the user needs a dataset found or a dataset analysed",
    description=(
        "Reads the message, any uploaded file, dataset name/URL and project goal, and "
        "returns `USER_A_DATASET_DISCOVERY` (find me a dataset), "
        "`USER_B_DATASET_ANALYSIS` (analyse the dataset I have) or `AMBIGUOUS` with a "
        "clarifying question. The decision is deterministic and lists its evidence. "
        "`follow_up_intent` is set when one journey may lead to the other."
    ),
)
async def resolve_intent(body: IntentRequest) -> IntentResolveResponse:
    request_id = str(uuid.uuid4())
    resolution = _router.resolve(body)
    # Only the decision is logged: never the user's message.
    logger.info("[%s] intent=%s confidence=%.2f", request_id, resolution.intent.value, resolution.confidence)
    return IntentResolveResponse(request_id=request_id, **resolution.model_dump())
