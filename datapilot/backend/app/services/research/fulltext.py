"""Open-access full text for the top papers, read safely.

Titles and abstracts name few methods and almost no figures. For the few papers that publish an open-access PDF, this
reads the first pages so methods and reported figures come from the paper itself (evidence ``paper_text``).

Everything about fetching is defensive, because the URL comes from a search index, not from the user:

* https only, a named public host (no IP literals, no localhost, no credentials, port 443), and the host must RESOLVE
  to public addresses only; this is re-checked on every redirect (at most three, followed manually);
* the body is streamed and abandoned at ``max_bytes``; it must begin with ``%PDF`` whatever the content type claims;
* only the first pages are parsed, the reference list is cut off, text is capped, encrypted files are skipped;
* a PDF is data: nothing in it is executed or followed, and its text is only matched against method/metric patterns.

Any failure leaves that paper at snippet level. Nothing here can fail the research.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import re
import socket
from typing import Awaitable, Callable, Optional
from urllib.parse import urljoin, urlsplit

import httpx

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.schemas.research import EvidenceKind, Paper
from app.services.research import methods as method_tools

logger = get_logger(__name__)

MAX_REDIRECTS = 3
MAX_PAGES = 14
MAX_CHARS = 60_000
_UA = "DataPilot/1.0 (research assistant; open-access full text)"
_REFERENCES = re.compile(r"(?im)^\s*(?:references|bibliography)\s*$")

Resolver = Callable[[str], Awaitable[list[str]]]


async def _system_resolver(host: str) -> list[str]:
    infos = await asyncio.to_thread(socket.getaddrinfo, host, 443, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


async def is_public_https(url: str, resolver: Resolver = _system_resolver) -> bool:
    """True only for an ordinary https URL to a named host that resolves to public addresses."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    if (parts.scheme != "https" or parts.username is not None or parts.password is not None or port not in (None, 443)
            or not host or host == "localhost" or "." not in host or len(url) > 2048
            or any(ord(c) < 33 for c in url)):
        return False
    try:
        ipaddress.ip_address(host)
        return False  # IP literals are refused outright
    except ValueError:
        pass
    try:
        addresses = await resolver(host)
    except Exception:
        return False
    return bool(addresses) and all(ipaddress.ip_address(a).is_global for a in addresses)


async def fetch_pdf(client: httpx.AsyncClient, url: str, max_bytes: int, resolver: Resolver = _system_resolver) -> Optional[bytes]:
    """The PDF's bytes, or None when anything about the URL, the redirects, the size or the content is not acceptable."""
    got = await fetch_limited(client, url, max_bytes, resolver, accept="application/pdf")
    return got[0] if got is not None and got[0].startswith(b"%PDF") else None


async def fetch_limited(
    client: httpx.AsyncClient, url: str, max_bytes: int, resolver: Resolver = _system_resolver, accept: str = "*/*",
) -> Optional[tuple[bytes, str]]:
    """(body, content-type) of a public https page, or None. Same rules as above: named public host resolving to
    public addresses (re-checked on each of <=3 redirects), size capped while streaming, no credentials."""
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        if not await is_public_https(current, resolver):
            return None
        try:
            async with client.stream("GET", current, headers={"User-Agent": _UA, "Accept": accept},
                                     follow_redirects=False) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        return None
                    current = urljoin(current, location)
                    continue
                if response.status_code != 200:
                    return None
                declared = response.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    return None
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > max_bytes:
                        return None  # abandoned, never buffered further
                return bytes(body), response.headers.get("content-type", "").lower()
        except httpx.HTTPError:
            return None
    return None


def extract_text(data: bytes, max_pages: int = MAX_PAGES, max_chars: int = MAX_CHARS) -> str:
    """Text of the first pages, without the reference list. Empty on any problem (encrypted, malformed, no pypdf)."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return ""
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages)
    except Exception:
        return ""
    cut = None
    for found in _REFERENCES.finditer(text):
        if found.start() > len(text) * 0.4:
            cut = found.start()
    if cut is not None:
        text = text[:cut]
    return re.sub(r"[ \t]+", " ", text).strip()[:max_chars]


async def read_full_text(
    papers: list[Paper], settings: Settings, client: Optional[httpx.AsyncClient] = None,
    resolver: Resolver = _system_resolver,
) -> dict[str, str]:
    """{paper id: text} for the top papers that publish an open-access PDF and could be read."""
    targets = [p for p in papers if p.open_access_url][: max(0, settings.research_fulltext_papers)]
    if not targets or not settings.research_fulltext_enabled:
        return {}
    owns = client is None
    http = client or httpx.AsyncClient(timeout=settings.research_fulltext_timeout_seconds)

    async def one(paper: Paper) -> tuple[str, str]:
        data = await fetch_pdf(http, paper.open_access_url or "", settings.research_fulltext_max_bytes, resolver)
        if data is None:
            return paper.id, ""
        return paper.id, await asyncio.to_thread(extract_text, data)

    try:
        results = await asyncio.wait_for(asyncio.gather(*(one(p) for p in targets), return_exceptions=True),
                                         timeout=settings.research_fulltext_timeout_seconds * 2)
    except asyncio.TimeoutError:
        return {}
    finally:
        if owns:
            await http.aclose()
    out: dict[str, str] = {}
    for item in results:
        if isinstance(item, tuple) and len(item[1]) >= 500:  # a stub or scanned page is not a paper
            out[item[0]] = item[1]
    return out


def apply_full_text(papers: list[Paper], texts: dict[str, str]) -> list[Paper]:
    """Papers whose text was read get methods from the full text and the honest label; others are untouched."""
    out = []
    for paper in papers:
        text = texts.get(paper.id)
        if not text:
            out.append(paper)
            continue
        methods = method_tools.methods_in(f"{paper.title} {paper.abstract_snippet or ''} {text}")
        out.append(paper.model_copy(update={
            "methods": methods, "retrieval_status": "full_text", "full_text_chars": len(text),
            "evidence": list(dict.fromkeys([*paper.evidence, EvidenceKind.PAPER_TEXT])),
        }))
    return out
