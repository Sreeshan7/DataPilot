"""Hugging Face Hub dataset discovery.

Public ``GET /api/datasets`` with ``full=true``, which returns each dataset's card:
declared task categories, ``format:``/``modality:`` tags, license, and for some
datasets a structured column schema. No credential is needed and none is sent, no
dataset is downloaded, and one request returns up to ``limit`` rows.

The Hub's ``search`` parameter requires every word to appear, so the long keyword
query built for Kaggle usually matches nothing. The source therefore retries once
with a shorter query (the leading two words, then the first word): the query
builder puts the most project-specific terms first, so the head of the query is the
part worth keeping. At most :data:`MAX_ATTEMPTS` requests are made per search.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import httpx

from app.core.logging_config import get_logger
from app.schemas.dataset import DatasetCandidate, SourceSearchRequest, SourceType
from app.services.datasets.base import DatasetSource, DatasetSourceError
from app.services.datasets.normalizer import normalize_huggingface_records

logger = get_logger(__name__)

_LIST_PATH = "/datasets"

#: Request budget per search: the full query is never tried, only a 2-word head and
#: then a 1-word head. Documented because the Hub's AND matching makes anything
#: longer return nothing, and a third attempt would only ever be a bare common word.
MAX_ATTEMPTS = 2
_HEAD_WORDS = (2, 1)


def query_heads(query: str) -> list[str]:
    """The shortened queries to try, longest first, without repeats."""
    words = query.split()
    heads: list[str] = []
    for count in _HEAD_WORDS[:MAX_ATTEMPTS]:
        head = " ".join(words[:count])
        if head and head not in heads:
            heads.append(head)
    return heads


class HuggingFaceDatasetSource(DatasetSource):
    """Discovers real, externally published Hugging Face datasets."""

    name: ClassVar[str] = "huggingface"
    source_type: ClassVar[SourceType] = SourceType.VERIFIED_EXTERNAL
    default_limit: ClassVar[int] = 10
    max_limit: ClassVar[int] = 50

    def __init__(self, settings=None, client: Optional[httpx.AsyncClient] = None) -> None:
        super().__init__(settings)
        self._client = client

    @property
    def is_enabled(self) -> bool:
        return bool(self._settings.huggingface_enabled)

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.huggingface_enabled)

    @property
    def has_credentials(self) -> bool:
        return False

    async def search(self, request: SourceSearchRequest) -> list[DatasetCandidate]:
        query = (request.query or "").strip()
        if not query:
            raise DatasetSourceError(
                code="DATASET_QUERY_EMPTY",
                message="A non-empty query is required for Hugging Face discovery.",
            )
        limit = max(1, min(int(request.limit), self.max_limit, self._settings.huggingface_max_limit))

        async with self._acquire_client() as client:
            for head in query_heads(query):
                payload = await self._get_json(client, {"search": head, "limit": limit, "full": "true"})
                candidates = normalize_huggingface_records(payload)
                if candidates:
                    logger.info(
                        "Hugging Face discovery for %r returned %d candidates", head, len(candidates)
                    )
                    return candidates[:limit]
        return []

    async def _get_json(self, client: httpx.AsyncClient, params: dict) -> Any:
        url = f"{self._settings.huggingface_api_base_url.rstrip('/')}{_LIST_PATH}"
        try:
            response = await client.get(url, params=params, headers={"Accept": "application/json"})
        except httpx.TimeoutException as exc:
            raise DatasetSourceError(
                code="HUGGINGFACE_TIMEOUT",
                message="Hugging Face did not respond within the configured timeout.",
            ) from exc
        except httpx.HTTPError as exc:
            raise DatasetSourceError(
                code="HUGGINGFACE_UNAVAILABLE", message="Hugging Face could not be reached."
            ) from exc

        status = response.status_code
        if status == 429:
            raise DatasetSourceError(
                code="HUGGINGFACE_RATE_LIMITED",
                message="Hugging Face rate limit exceeded. Try again shortly.",
            )
        if status != 200:
            # The upstream body is never echoed: it can carry internal detail.
            raise DatasetSourceError(
                code="HUGGINGFACE_UNAVAILABLE",
                message=f"Hugging Face returned unexpected status {status}.",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise DatasetSourceError(
                code="HUGGINGFACE_INVALID_RESPONSE",
                message="Hugging Face returned a response that is not valid JSON.",
            ) from exc

    async def healthcheck(self) -> bool:
        if not self.is_enabled:
            return False
        try:
            await self.search(SourceSearchRequest(query="dataset", limit=1))
            return True
        except DatasetSourceError as exc:
            logger.warning("Hugging Face healthcheck failed: %s", exc.code)
            return False

    def _acquire_client(self):
        if self._client is not None:
            return _Borrowed(self._client)
        return httpx.AsyncClient(timeout=self._settings.huggingface_request_timeout)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<HuggingFaceDatasetSource enabled={self.is_enabled}>"


class _Borrowed:
    """Async context manager that adopts, but does not close, an injected client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(self, *exc_info: Any) -> None:
        return None
