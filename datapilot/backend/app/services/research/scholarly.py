"""Scholarly providers: OpenAlex, arXiv and Crossref. Free, keyless, documented APIs.

Unlike scraping a search page, these return real structured records (title, authors, year, DOI, venue, abstract,
citation count), so a paper's metadata is read, never guessed. Each provider talks to exactly one fixed https host,
follows no redirects off it, caps the response size and times out; none ever fetches a paper's own URL.

Everything returned is still untrusted text (abstracts can contain anything): it is only ever used as data.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

import httpx

from app.core.config import Settings
from app.services.research.providers import ProviderError, SearchHit

_UA = "DataPilot/1.0 (research assistant; scholarly metadata lookup)"
_TAGS = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = {"or", "and", "the", "of", "for", "in", "on", "a", "an", "to"}
MAX_ABSTRACT = 1200


def _clean(text: Optional[str]) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", text or "")).split())


def query_words(query: str, limit: int = 8) -> list[str]:
    """The content words of a query: quotes and boolean words removed, so APIs get plain terms."""
    words = [w for w in _WORD.findall(query.replace('"', " ")) if w.lower() not in _STOP]
    return words[:limit]


def _oa_link(value: object) -> Optional[str]:
    """An open-access link is kept only if it is an https URL; nothing is fetched here."""
    return value if isinstance(value, str) and value.startswith("https://") and len(value) <= 2048 else None


class _Base:
    fallback = False
    delay = 0.2

    def __init__(self, settings: Settings, client: Optional[httpx.AsyncClient] = None) -> None:
        self._settings = settings
        self._client = client

    async def _get(self, url: str, params: dict[str, Any]) -> httpx.Response:
        owns = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self._settings.research_timeout_seconds, follow_redirects=False)
        try:
            response = await client.get(url, params=params, headers={"User-Agent": _UA, "Accept": "application/json, application/atom+xml"})
        except httpx.TimeoutException as exc:
            raise ProviderError("TIMEOUT", f"{self.name} timed out.") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("UNAVAILABLE", f"{self.name} could not be reached.") from exc
        finally:
            if owns:
                await client.aclose()
        if response.status_code == 429:
            raise ProviderError("RATE_LIMITED", f"{self.name} is rate limiting requests.")
        if response.status_code != 200:
            raise ProviderError("UNAVAILABLE", f"{self.name} answered HTTP {response.status_code}.")
        if len(response.content) > self._settings.research_max_response_bytes:
            raise ProviderError("INVALID_RESPONSE", f"{self.name} returned too much data.")
        return response


# ---------------------------------------------------------------------------------------------- OpenAlex


def abstract_from_index(index: Optional[dict[str, list[int]]]) -> str:
    """OpenAlex ships abstracts as an inverted index {word: [positions]}; rebuild the text."""
    if not isinstance(index, dict) or not index:
        return ""
    slots: dict[int, str] = {}
    for word, positions in index.items():
        if isinstance(positions, list):
            for p in positions:
                if isinstance(p, int) and 0 <= p < 5000:
                    slots[p] = word
    return " ".join(slots[i] for i in sorted(slots))[:MAX_ABSTRACT]


class OpenAlexProvider(_Base):
    name = "openalex"
    URL = "https://api.openalex.org/works"

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        params = {
            "search": " ".join(query_words(query, 10)) or query,
            "per-page": max(1, min(limit, 25)),
            "filter": "type:article|preprint|review,has_abstract:true",
            "select": "id,title,display_name,publication_year,doi,authorships,abstract_inverted_index,primary_location,cited_by_count,open_access",
        }
        data = (await self._get(self.URL, params)).json()
        hits: list[SearchHit] = []
        for work in data.get("results", []) if isinstance(data, dict) else []:
            title = _clean(work.get("title") or work.get("display_name"))
            doi = work.get("doi")
            location = work.get("primary_location") or {}
            url = doi or location.get("landing_page_url") or work.get("id")
            if not title or not isinstance(url, str) or not url.startswith("http"):
                continue
            authors = tuple(
                a["author"]["display_name"] for a in (work.get("authorships") or [])[:8]
                if isinstance(a, dict) and isinstance(a.get("author"), dict) and a["author"].get("display_name"))
            hits.append(SearchHit(
                title=title, url=url, snippet=abstract_from_index(work.get("abstract_inverted_index")), query=query,
                provider=self.name, authors=authors or None, year=work.get("publication_year"),
                doi=doi.replace("https://doi.org/", "") if isinstance(doi, str) else None,
                venue=((location.get("source") or {}).get("display_name") if isinstance(location, dict) else None),
                citations=work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else None,
                oa_url=_oa_link((work.get("open_access") or {}).get("oa_url") if isinstance(work.get("open_access"), dict) else None),
                academic=True))
        return hits[:limit]


# ------------------------------------------------------------------------------------------ Semantic Scholar


class SemanticScholarProvider(_Base):
    """Citation counts and full abstracts. The keyless tier is shared and rate limited, so it asks little and a
    refusal just means this source is skipped."""

    name = "semanticscholar"
    URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    max_queries = 2
    delay = 1.2

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        params = {
            "query": " ".join(query_words(query, 10)) or query,
            "limit": max(1, min(limit, 20)),
            "fields": "title,abstract,year,authors,venue,externalIds,citationCount,url,openAccessPdf",
        }
        data = (await self._get(self.URL, params)).json()
        hits: list[SearchHit] = []
        for paper in data.get("data", []) if isinstance(data, dict) else []:
            if not isinstance(paper, dict):
                continue
            title = _clean(paper.get("title"))
            ids = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
            doi = ids.get("DOI") if isinstance(ids.get("DOI"), str) else None
            url = f"https://doi.org/{doi}" if doi else paper.get("url")
            abstract = _clean(paper.get("abstract"))[:MAX_ABSTRACT]
            if not title or not abstract or not isinstance(url, str) or not url.startswith("http"):
                continue
            authors = tuple(a["name"] for a in (paper.get("authors") or [])[:8]
                            if isinstance(a, dict) and isinstance(a.get("name"), str))
            year = paper.get("year") if isinstance(paper.get("year"), int) else None
            citations = paper.get("citationCount") if isinstance(paper.get("citationCount"), int) else None
            hits.append(SearchHit(
                title=title, url=url, snippet=abstract, query=query, provider=self.name, authors=authors or None,
                year=year, doi=doi, venue=_clean(paper.get("venue")) or None, citations=citations,
                oa_url=_oa_link((paper.get("openAccessPdf") or {}).get("url") if isinstance(paper.get("openAccessPdf"), dict) else None),
                academic=True))
        return hits[:limit]


# ------------------------------------------------------------------------------------------------ arXiv

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


class ArxivProvider(_Base):
    name = "arxiv"
    URL = "https://export.arxiv.org/api/query"
    max_queries = 2
    delay = 3.0  # arXiv's API terms ask for a few seconds between requests

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        words = query_words(query, 6)
        if not words:
            return []
        params = {"search_query": " AND ".join(f"all:{w}" for w in words), "max_results": max(1, min(limit, 20)),
                  "sortBy": "relevance"}
        response = await self._get(self.URL, params)
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise ProviderError("INVALID_RESPONSE", "arXiv returned an unreadable response.") from exc
        hits: list[SearchHit] = []
        for entry in root.findall(f"{_ATOM}entry"):
            url = (entry.findtext(f"{_ATOM}id") or "").strip()
            title = _clean(entry.findtext(f"{_ATOM}title"))
            if not title or not url.startswith("http"):
                continue
            published = entry.findtext(f"{_ATOM}published") or ""
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = tuple(_clean(a.findtext(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author")[:8])
            doi = (entry.findtext(f"{_ARXIV}doi") or "").strip() or None
            hits.append(SearchHit(
                title=title, url=url.replace("http://", "https://", 1), snippet=_clean(entry.findtext(f"{_ATOM}summary"))[:MAX_ABSTRACT],
                query=query, provider=self.name, authors=tuple(a for a in authors if a) or None, year=year, doi=doi,
                venue="arXiv", oa_url=_oa_link(url.replace("http://", "https://", 1).replace("/abs/", "/pdf/")),
                academic=True))
        return hits[:limit]


# ---------------------------------------------------------------------------------------------- Crossref


class CrossrefProvider(_Base):
    name = "crossref"
    URL = "https://api.crossref.org/works"
    max_queries = 2

    async def search(self, query: str, limit: int) -> list[SearchHit]:
        params = {"query.bibliographic": " ".join(query_words(query, 10)) or query, "rows": max(1, min(limit, 20)),
                  "select": "title,DOI,author,issued,container-title,abstract,URL,is-referenced-by-count,type",
                  "filter": "type:journal-article,has-abstract:true"}
        data = (await self._get(self.URL, params)).json()
        items = (data.get("message") or {}).get("items", []) if isinstance(data, dict) else []
        hits: list[SearchHit] = []
        for item in items:
            titles = item.get("title") or []
            title = _clean(titles[0]) if titles else ""
            doi = item.get("DOI")
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            if not title or not url.startswith("http"):
                continue
            parts = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
            authors = tuple(
                " ".join(filter(None, [a.get("given"), a.get("family")])) for a in (item.get("author") or [])[:8] if isinstance(a, dict))
            containers = item.get("container-title") or []
            hits.append(SearchHit(
                title=title, url=url, snippet=_clean(item.get("abstract"))[:MAX_ABSTRACT], query=query, provider=self.name,
                authors=tuple(a for a in authors if a) or None, year=parts[0] if parts and isinstance(parts[0], int) else None,
                doi=doi, venue=_clean(containers[0]) if containers else None,
                citations=item.get("is-referenced-by-count") if isinstance(item.get("is-referenced-by-count"), int) else None,
                academic=True))
        return hits[:limit]
