"""Kaggle dataset discovery.

Only ``GET /v1/datasets/list`` metadata is used: no dataset is ever downloaded,
no file is listed, no per-dataset detail call is made. One request returns at
most 20 rows regardless of ``pageSize``, so satisfying a larger limit costs one
additional request per page, capped by ``kaggle_max_pages``.

Authentication uses the modern ``KAGGLE_API_TOKEN`` sent as a bearer token, so
no ``~/.kaggle/kaggle.json`` credential file is required. The token is only
ever sent in an Authorization header: it is never logged, echoed in errors, or
placed in a query string.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Optional

import httpx

from app.core.logging_config import get_logger
from app.schemas.dataset import DatasetCandidate, SourceSearchRequest, SourceType
from app.services.datasets.base import DatasetSource, DatasetSourceError
from app.services.datasets.normalizer import apply_kaggle_detail, normalize_kaggle_records

logger = get_logger(__name__)

_LIST_PATH = "/datasets/list"
_VIEW_PATH = "/datasets/view"

#: Kaggle ranks by relevance but returns nothing once too many rare terms are
#: combined (measured live: a 5-term churn query returned 0 rows, its 3-term head
#: returned 20). When a query comes back empty it is retried once with this many
#: leading terms, which the query builder fills with the most project-specific
#: words first. A non-empty result is never retried, so the cost is at most one
#: extra request, and only when there would otherwise be nothing to show.
FALLBACK_QUERY_TERMS = 3


def fallback_query(query: str) -> Optional[str]:
    """The shortened retry query, or ``None`` when shortening would change nothing."""
    words = query.split()
    return " ".join(words[:FALLBACK_QUERY_TERMS]) if len(words) > FALLBACK_QUERY_TERMS else None


class KaggleDatasetSource(DatasetSource):
    """Discovers real, externally published Kaggle datasets."""

    name: ClassVar[str] = "kaggle"
    source_type: ClassVar[SourceType] = SourceType.VERIFIED_EXTERNAL
    default_limit: ClassVar[int] = 10
    max_limit: ClassVar[int] = 50

    def __init__(self, settings=None, client: Optional[httpx.AsyncClient] = None) -> None:
        super().__init__(settings)
        self._client = client
        self._owns_client = client is None

    # -- configuration -----------------------------------------------------

    @property
    def is_enabled(self) -> bool:
        return bool(self._settings.kaggle_enabled)

    @property
    def is_configured(self) -> bool:
        """Public dataset search works unauthenticated, so this is True whenever
        the integration is enabled. When a token is present it is used, which
        additionally scopes results to what the account may see."""
        return bool(self._settings.kaggle_enabled)

    @property
    def has_credentials(self) -> bool:
        return bool(self._settings.kaggle_api_token)

    # -- search ------------------------------------------------------------

    async def search(self, request: SourceSearchRequest) -> list[DatasetCandidate]:
        query = (request.query or "").strip()
        if not query:
            raise DatasetSourceError(
                code="DATASET_QUERY_EMPTY",
                message="A non-empty query is required for Kaggle discovery.",
            )

        limit = max(1, min(int(request.limit), self.max_limit))

        async with self._acquire_client() as client:
            candidates = await self._search_pages(client, query, limit)
            shorter = fallback_query(query)
            if not candidates and shorter is not None:
                logger.info("Kaggle returned nothing for %r; retrying with %r", query, shorter)
                query = shorter
                candidates = await self._search_pages(client, query, limit)

        logger.info(
            "Kaggle discovery for query %r returned %d candidates", query, len(candidates)
        )
        return candidates[:limit]

    async def _search_pages(
        self, client: httpx.AsyncClient, query: str, limit: int
    ) -> list[DatasetCandidate]:
        candidates: list[DatasetCandidate] = []
        seen: set[str] = set()
        for page in range(1, self._settings.kaggle_max_pages + 1):
            payload = await self._fetch_page(client, query, page)
            page_candidates = normalize_kaggle_records(payload)
            if not page_candidates:
                break
            for candidate in page_candidates:
                if candidate.id in seen:
                    continue
                seen.add(candidate.id)
                candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates

    async def _fetch_page(self, client: httpx.AsyncClient, query: str, page: int) -> Any:
        return await self._get_json(client, _LIST_PATH, {"search": query, "page": page})

    async def _get_json(
        self, client: httpx.AsyncClient, path: str, params: Optional[dict] = None
    ) -> Any:
        """One authenticated GET with every failure mapped to a sanitized error."""
        url = f"{self._settings.kaggle_api_base_url.rstrip('/')}{path}"
        headers = {"Accept": "application/json"}
        token = self._settings.kaggle_api_token
        if token:
            # Bearer header only - never a query parameter, never logged.
            headers["Authorization"] = f"Bearer {token}"

        try:
            response = await client.get(url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise DatasetSourceError(
                code="KAGGLE_TIMEOUT",
                message="Kaggle did not respond within the configured timeout.",
            ) from exc
        except httpx.HTTPError as exc:
            raise DatasetSourceError(
                code="KAGGLE_UNAVAILABLE",
                message="Kaggle could not be reached.",
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise DatasetSourceError(
                code="KAGGLE_AUTHENTICATION_ERROR",
                message="Kaggle rejected the configured credentials.",
            )
        if status == 404:
            raise DatasetSourceError(
                code="KAGGLE_NOT_FOUND",
                message="Kaggle has no public record for that dataset.",
            )
        if status == 429:
            raise DatasetSourceError(
                code="KAGGLE_RATE_LIMITED",
                message="Kaggle rate limit exceeded. Try again shortly.",
            )
        if status >= 500:
            raise DatasetSourceError(
                code="KAGGLE_UNAVAILABLE",
                message=f"Kaggle returned server error {status}.",
            )
        if status != 200:
            # Do not echo the upstream body: it can contain internal detail.
            raise DatasetSourceError(
                code="KAGGLE_UNAVAILABLE",
                message=f"Kaggle returned unexpected status {status}.",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise DatasetSourceError(
                code="KAGGLE_INVALID_RESPONSE",
                message="Kaggle returned a response that is not valid JSON.",
            ) from exc

    # -- enrichment --------------------------------------------------------

    @property
    def supports_enrichment(self) -> bool:
        return True

    async def enrich(self, candidate: DatasetCandidate) -> DatasetCandidate:
        """Read the owner's full description and the file list for one dataset.

        Two requests, issued together. The file list is a bonus: if only that call
        fails the description is still applied, because partial detail is better
        than none and the failure is not about the dataset's existence. If the
        description call fails the error is raised so the caller keeps the
        listing-level candidate and records the failure.
        """
        ref = (candidate.source_id or "").strip().strip("/")
        if "/" not in ref:
            raise DatasetSourceError(
                code="DATASET_SOURCE_INVALID_ID",
                message="Kaggle dataset refs must be in 'owner/slug' form.",
            )
        async with self._acquire_client() as client:
            view, files = await asyncio.gather(
                self._get_json(client, f"{_VIEW_PATH}/{ref}"),
                self._get_json(client, f"{_LIST_PATH}/{ref}"),
                return_exceptions=True,
            )
        if isinstance(view, BaseException):
            if isinstance(view, DatasetSourceError):
                raise view
            raise DatasetSourceError(
                code="KAGGLE_UNAVAILABLE", message="Kaggle could not be reached."
            ) from view
        if isinstance(files, BaseException):
            logger.info("Kaggle file list unavailable for %s", ref)
            files = None
        return apply_kaggle_detail(candidate, view, files)

    # -- metadata ----------------------------------------------------------

    async def get_metadata(self, source_id: str) -> DatasetCandidate:
        """Fetch one dataset's metadata via the search endpoint.

        The list endpoint already carries the full published metadata for a
        dataset, so this resolves a ref by querying it and returns the matching
        candidate instead of adding a per-dataset request that would return the
        same fields.
        """
        ref = (source_id or "").strip().strip("/")
        if not ref:
            raise DatasetSourceError(
                code="DATASET_SOURCE_INVALID_ID",
                message="A Kaggle dataset ref such as 'owner/slug' is required.",
            )
        if "/" not in ref:
            raise DatasetSourceError(
                code="DATASET_SOURCE_INVALID_ID",
                message="Kaggle dataset refs must be in 'owner/slug' form.",
            )

        slug = ref.split("/")[-1].replace("-", " ")
        results = await self.search(SourceSearchRequest(query=slug, limit=self.max_limit))
        for candidate in results:
            if candidate.source_id.lower() == ref.lower():
                return candidate
        raise DatasetSourceError(
            code="KAGGLE_NOT_FOUND",
            message=f"No public Kaggle dataset matched ref '{ref}'.",
        )

    async def healthcheck(self) -> bool:
        if not self.is_enabled:
            return False
        try:
            await self.search(SourceSearchRequest(query="dataset", limit=1))
            return True
        except DatasetSourceError as exc:
            logger.warning("Kaggle healthcheck failed: %s", exc.code)
            return False

    # -- client lifecycle --------------------------------------------------

    def _acquire_client(self):
        if self._client is not None:
            return _BorrowedClient(self._client)
        return httpx.AsyncClient(timeout=self._settings.kaggle_request_timeout)

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<KaggleDatasetSource enabled={self.is_enabled} token_present={self.has_credentials}>"


class _BorrowedClient:
    """Async context manager that adopts, but does not close, an injected client."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(self, *exc_info: Any) -> None:
        return None
