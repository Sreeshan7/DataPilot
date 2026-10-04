"""The only door through which dataset research reaches the network.

Every request made while researching a dataset goes through :class:`SafeFetcher`, so the
protections cannot be forgotten by an individual adapter:

* only allowlisted registry hosts, over https (``validate_public_url``);
* redirects are followed by hand, at most ``external_max_redirects`` of them, and every
  hop is validated against the allowlist again - a registry cannot bounce the request
  somewhere else;
* the response is streamed and cut off at ``external_max_response_bytes``, so a hostile or
  broken endpoint cannot exhaust memory;
* a hard timeout on every request;
* a credential is only ever attached to the host it belongs to and is dropped if a redirect
  changes host;
* upstream bodies and error text are never returned, so nothing internal can leak to a user.

Callers get parsed JSON or a :class:`FetchError` carrying a stable code.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.services.dataset_analysis.references import UnsafeURLError, validate_public_url

logger = get_logger(__name__)

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class FetchError(Exception):
    """A research request failed. ``code`` is stable; ``message`` is always safe to show."""

    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    UNAVAILABLE = "UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    UNSAFE = "UNSAFE"

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class SafeFetcher:
    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings
        self._client = client

    async def get_json(
        self,
        url: str,
        params: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        *,
        not_found_statuses: frozenset[int] = frozenset({404}),
    ) -> Any:
        """GET ``url`` and return its parsed JSON body.

        ``not_found_statuses`` lists the HTTP statuses a registry uses to say "no such
        dataset" (OpenML uses 412, Hugging Face answers 401 for a missing or private one).
        """
        try:
            current = validate_public_url(url)
        except UnsafeURLError as exc:
            raise FetchError(FetchError.UNSAFE, f"Refused to contact that address: {exc}") from exc

        owner = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._settings.external_request_timeout)
        try:
            send_headers = dict(headers or {})
            send_headers.setdefault("Accept", "application/json")
            query = params
            for hop in range(self._settings.external_max_redirects + 1):
                status, location, body = await self._request(
                    client, current, query, send_headers, not_found_statuses
                )
                if status in _REDIRECT_STATUSES:
                    if hop >= self._settings.external_max_redirects or not location:
                        raise FetchError(FetchError.UNAVAILABLE, "The registry redirected too many times.")
                    try:
                        target = validate_public_url(urljoin(current, location))
                    except UnsafeURLError as exc:
                        raise FetchError(
                            FetchError.UNSAFE, "The registry redirected to an address that is not allowed."
                        ) from exc
                    if urlsplit(target).hostname != urlsplit(current).hostname:
                        # A credential belongs to one host; never carry it across a redirect.
                        send_headers.pop("Authorization", None)
                    current, query = target, None
                    continue
                return self._parse(body)
            raise FetchError(FetchError.UNAVAILABLE, "The registry redirected too many times.")
        finally:
            if owner:
                await client.aclose()

    async def _request(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: Optional[dict[str, Any]],
        headers: dict[str, str],
        not_found_statuses: frozenset[int],
    ) -> tuple[int, Optional[str], bytes]:
        limit = self._settings.external_max_response_bytes
        try:
            async with client.stream(
                "GET", url, params=params, headers=headers, timeout=self._settings.external_request_timeout,
                follow_redirects=False,
            ) as response:
                status = response.status_code
                location = response.headers.get("location")
                if status in _REDIRECT_STATUSES:
                    return status, location, b""
                if status in not_found_statuses:
                    raise FetchError(FetchError.NOT_FOUND, "The registry has no such dataset.")
                if status == 429:
                    raise FetchError(FetchError.RATE_LIMITED, "The registry's rate limit was reached. Try again shortly.")
                if status >= 500:
                    raise FetchError(FetchError.UNAVAILABLE, f"The registry returned a server error ({status}).")
                if status != 200:
                    raise FetchError(FetchError.UNAVAILABLE, f"The registry refused the request ({status}).")
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > limit:
                        raise FetchError(
                            FetchError.INVALID_RESPONSE,
                            "The registry's response was larger than the allowed size and was discarded.",
                        )
                    chunks.append(chunk)
                return status, None, b"".join(chunks)
        except httpx.TimeoutException as exc:
            raise FetchError(FetchError.TIMEOUT, "The registry did not respond in time.") from exc
        except httpx.HTTPError as exc:
            logger.info("Registry request failed (%s)", type(exc).__name__)
            raise FetchError(FetchError.UNAVAILABLE, "The registry could not be reached.") from exc

    @staticmethod
    def _parse(body: bytes) -> Any:
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError) as exc:
            raise FetchError(FetchError.INVALID_RESPONSE, "The registry returned a response that is not valid JSON.") from exc
