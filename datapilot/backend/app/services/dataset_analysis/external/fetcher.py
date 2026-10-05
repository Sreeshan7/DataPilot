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

    async def get_prefix(
        self,
        url: str,
        headers: Optional[dict[str, str]] = None,
        *,
        max_bytes: int,
        storage_hosts: frozenset[str],
    ) -> tuple[bytes, Optional[int]]:
        """Read only the first ``max_bytes`` of a file: its header and a few rows, never the dataset.

        ``url`` must pass the usual registry validation. The registry may answer with one redirect to a file store; that
        is followed only if the target is https on a host in ``storage_hosts`` (no credentials, no IP literals, default
        port), and the credential is never sent there. The body is read as a stream and abandoned at ``max_bytes`` even
        if the server ignores the Range request. Returns ``(bytes, total_size_if_known)``. The redirect target is a
        signed URL and is never logged or returned.
        """
        try:
            current = validate_public_url(url)
        except UnsafeURLError as exc:
            raise FetchError(FetchError.UNSAFE, f"Refused to contact that address: {exc}") from exc
        owner = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._settings.external_request_timeout)
        send = dict(headers or {})
        try:
            for hop in range(2):
                request_headers = {**send, "Range": f"bytes=0-{max_bytes - 1}", "Accept": "*/*"}
                async with client.stream("GET", current, headers=request_headers, follow_redirects=False,
                                         timeout=self._settings.external_request_timeout) as response:
                    status = response.status_code
                    if status in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if hop or not location:
                            raise FetchError(FetchError.UNAVAILABLE, "The file could not be located.")
                        target = urljoin(current, location)
                        parts = urlsplit(target)
                        host = (parts.hostname or "").lower()
                        if (parts.scheme != "https" or host not in storage_hosts or parts.username or parts.password
                                or parts.port not in (None, 443)):
                            raise FetchError(FetchError.UNSAFE, "The registry redirected to an address that is not allowed.")
                        current = target
                        send.pop("Authorization", None)  # a credential belongs to one host
                        continue
                    if status == 404:
                        raise FetchError(FetchError.NOT_FOUND, "The file was not found.")
                    if status == 429:
                        raise FetchError(FetchError.RATE_LIMITED, "The source's rate limit was reached. Try again shortly.")
                    if status not in (200, 206):
                        raise FetchError(FetchError.UNAVAILABLE, f"The file could not be read ({status}).")
                    total: Optional[int] = None
                    content_range = response.headers.get("content-range", "")
                    if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit():
                        total = int(content_range.rsplit("/", 1)[1])
                    elif status == 200 and response.headers.get("content-length", "").isdigit():
                        total = int(response.headers["content-length"])
                    chunks: list[bytes] = []
                    got = 0
                    async for chunk in response.aiter_bytes():
                        chunks.append(chunk)
                        got += len(chunk)
                        if got >= max_bytes:
                            break  # closing the stream abandons the rest of the file
                    return b"".join(chunks)[:max_bytes], total
            raise FetchError(FetchError.UNAVAILABLE, "The file could not be located.")
        except httpx.TimeoutException as exc:
            raise FetchError(FetchError.TIMEOUT, "The source did not respond in time.") from exc
        except httpx.HTTPError as exc:
            logger.info("File prefix request failed (%s)", type(exc).__name__)
            raise FetchError(FetchError.UNAVAILABLE, "The file could not be reached.") from exc
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
