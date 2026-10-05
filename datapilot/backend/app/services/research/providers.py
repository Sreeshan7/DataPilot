"""Research providers: the interface the service depends on, and the one implementation (web search).

``ResearchProvider`` is the seam for later providers (Semantic Scholar, OpenAlex, Crossref, arXiv): each
returns raw ``SearchHit`` records and the service does everything else. Only ``WebSearchProvider`` exists
today. It uses DuckDuckGo's keyless HTML endpoint through ``httpx`` (no paid API, no new dependency).

Safety properties of the web provider:

* it talks to exactly one configured https host and never fetches a result's URL, so retrieved pages cannot
  steer the server (no SSRF, no crawling, no PDF download); only the search results page is read;
* redirects are not followed, the response size is capped and the request has a timeout;
* everything it returns is untrusted text, and is treated as data downstream.

This is an HTML page, not a documented API: if the site changes its markup or rate-limits the caller, the
provider reports ``ProviderError`` and the service answers ``search_unavailable`` instead of inventing results.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Optional, Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; DataPilotResearch/1.0; +local)"


@dataclass(frozen=True)
class SearchHit:
    """One result. The structured fields are filled only by providers that actually return them (scholarly
    APIs); a plain web-search hit carries just a title, URL and snippet, and the rest stays None."""

    title: str
    url: str
    snippet: str
    query: str
    provider: str
    authors: Optional[tuple[str, ...]] = None
    year: Optional[int] = None
    doi: Optional[str] = None
    venue: Optional[str] = None
    citations: Optional[int] = None
    #: True when the provider is itself a scholarly index, so the host allowlist is not needed.
    academic: bool = False


class ProviderError(Exception):
    """A provider could not answer. ``code`` is stable and ``message`` safe to show."""

    def __init__(self, code: str, message: str) -> None:
        self.code, self.message = code, message
        super().__init__(message)


class ResearchProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> list[SearchHit]: ...


_RESULT = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?:.*?<a[^>]*class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
    re.S,
)
_TAGS = re.compile(r"<[^>]+>")


def _text(fragment: Optional[str]) -> str:
    return " ".join(html.unescape(_TAGS.sub("", fragment or "")).split())


def _real_url(href: str) -> Optional[str]:
    """The result's own URL: unwrap DuckDuckGo's redirect form, and refuse anything but http(s)."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        href = unquote(target)
        parsed = urlparse(href)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.netloc.endswith("duckduckgo.com"):
        return None
    return href


def parse_results(markup: str, query: str, provider: str, limit: int) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for match in _RESULT.finditer(markup):
        url = _real_url(match.group("href"))
        title = _text(match.group("title"))
        if not url or not title:
            continue
        hits.append(SearchHit(title=title, url=url, snippet=_text(match.group("snippet")), query=query, provider=provider))
        if len(hits) >= limit:
            break
    return hits


class WebSearchProvider:
    name = "web_search"
    fallback = True
    delay = 1.0

    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings
        self._client = client

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        url = self._settings.research_search_url
        if not url.startswith("https://"):
            raise ProviderError("UNSAFE", "The search endpoint must be https.")
        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._settings.research_timeout_seconds, follow_redirects=False)
        try:
            response = await client.post(url, data={"q": query}, headers={"User-Agent": _USER_AGENT})
        except httpx.TimeoutException as exc:
            raise ProviderError("TIMEOUT", "The web search timed out.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("UNAVAILABLE", "The web search could not be reached.") from exc
        finally:
            if owns:
                await client.aclose()
        if response.status_code in (202, 429):
            raise ProviderError("RATE_LIMITED", "The web search is rate limiting requests; try again later.")
        if response.status_code != 200:
            raise ProviderError("UNAVAILABLE", f"The web search answered HTTP {response.status_code}.")
        if len(response.content) > self._settings.research_max_response_bytes:
            raise ProviderError("INVALID_RESPONSE", "The web search response was too large to use.")
        hits = parse_results(response.text, query, self.name, limit)
        if not hits and "result__a" not in response.text and "no-results" not in response.text and "No results" not in response.text:
            raise ProviderError("INVALID_RESPONSE", "The web search page was not in the expected form.")
        return hits
