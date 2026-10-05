"""Turn raw search hits into deduplicated ``Paper`` records, keeping only what the evidence supports.

Only academic-looking sources are kept (a short allowlist of publishers, repositories and university
domains); everything else is counted and dropped. A year, DOI or venue is filled in only when the URL or
the retrieved text states it; authors and citation counts are never inferred (they stay ``None``).

Deduplication is conservative. Two hits are the same paper if they share a DOI, an arXiv id or a canonical
URL, or if their normalised titles are *identical*. Similar-but-different titles are kept apart.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from app.schemas.research import EvidenceKind, Paper
from app.services.research.providers import SearchHit

# host suffix -> venue label
ACADEMIC_HOSTS: dict[str, str] = {
    "arxiv.org": "arXiv", "semanticscholar.org": "Semantic Scholar", "ieeexplore.ieee.org": "IEEE Xplore",
    "ieee.org": "IEEE", "dl.acm.org": "ACM Digital Library", "acm.org": "ACM", "link.springer.com": "Springer",
    "springer.com": "Springer", "springeropen.com": "Springer Open", "sciencedirect.com": "ScienceDirect",
    "pubmed.ncbi.nlm.nih.gov": "PubMed", "ncbi.nlm.nih.gov": "NCBI/PMC", "nature.com": "Nature",
    "mdpi.com": "MDPI", "wiley.com": "Wiley", "tandfonline.com": "Taylor & Francis", "plos.org": "PLOS",
    "frontiersin.org": "Frontiers", "biomedcentral.com": "BioMed Central", "jstor.org": "JSTOR",
    "openreview.net": "OpenReview", "aclanthology.org": "ACL Anthology", "doi.org": "DOI resolver",
    "researchgate.net": "ResearchGate", "sagepub.com": "SAGE", "cambridge.org": "Cambridge", "oup.com": "Oxford",
    "hindawi.com": "Hindawi", "iopscience.iop.org": "IOP", "ssrn.com": "SSRN", "core.ac.uk": "CORE",
    "paperswithcode.com": "Papers with Code", "scholar.google.com": "Google Scholar",
}
_ACADEMIC_TLD = re.compile(r"(?:^|\.)(?:edu|ac\.[a-z]{2})$")
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>?#]+)", re.I)
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4})\.(\d{4,5})(?:v\d+)?(?:\.pdf)?", re.I)
_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_NON_PAPER = re.compile(r"/(?:search|tag|topic|category|author|profile|user|people)/", re.I)


def host_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def venue_for(host: str) -> Optional[str]:
    for suffix, label in ACADEMIC_HOSTS.items():
        if host == suffix or host.endswith("." + suffix):
            return label
    return None


def is_academic(url: str) -> bool:
    if urlparse(url).scheme not in ("http", "https"):
        return False
    host = host_of(url)
    return bool(host) and (venue_for(host) is not None or bool(_ACADEMIC_TLD.search(host))) \
        and not _NON_PAPER.search(urlparse(url).path)


def canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = host_of(url)
    path = re.sub(r"/+$", "", parsed.path)
    arxiv = _ARXIV.search(url)
    if arxiv:
        return f"arxiv.org/abs/{arxiv.group(1)}.{arxiv.group(2)}"
    return f"{host}{path}".lower()


def normalise_title(title: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split())


def _clean_title(title: str) -> str:
    # Search engines append the site name ("... | ScienceDirect", "... - arXiv").
    return re.split(r"\s+[|–—-]\s+(?:[A-Z][\w .&]*)$", title)[0].strip() or title.strip()


@dataclass
class _Entry:
    title: str
    url: str
    snippets: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    arxiv_year: Optional[int] = None
    authors: Optional[tuple[str, ...]] = None
    year: Optional[int] = None
    venue: Optional[str] = None
    citations: Optional[int] = None
    oa_url: Optional[str] = None
    structured: bool = False  # some provider returned real metadata for it


def _doi_of(hit: SearchHit) -> Optional[str]:
    for text in (hit.url, hit.snippet):
        found = _DOI.search(text)
        if found:
            return found.group(1).rstrip(".,;)").lower()
    return None


def _arxiv_year(url: str) -> Optional[int]:
    found = _ARXIV.search(url)
    if not found:
        return None
    yy = int(found.group(1)[:2])
    return 2000 + yy if yy < 90 else 1900 + yy


def normalise(hits: list[SearchHit]) -> tuple[list[Paper], int, int]:
    """(papers, hits_rejected_as_non_academic, duplicates_merged)."""
    entries: dict[str, _Entry] = {}
    alias: dict[str, str] = {}  # secondary key -> primary key
    rejected = merged = 0
    for hit in hits:
        if not hit.academic and not is_academic(hit.url):
            rejected += 1
            continue
        doi = hit.doi.lower() if hit.doi else _doi_of(hit)
        keys = [f"doi:{doi}"] if doi else []
        keys.append(f"url:{canonical_url(hit.url)}")
        title_key = normalise_title(_clean_title(hit.title))
        if title_key:
            keys.append(f"title:{title_key}")
        primary = next((alias[k] for k in keys if k in alias), None)
        if primary is None:
            primary = keys[0]
            entries[primary] = _Entry(title=_clean_title(hit.title), url=hit.url, doi=doi, arxiv_year=_arxiv_year(hit.url))
        else:
            merged += 1
            entry = entries[primary]
            entry.doi = entry.doi or doi
        entry = entries[primary]
        # Keep the richest metadata any provider returned for the same paper (never invent any).
        entry.authors = entry.authors or hit.authors
        entry.year = entry.year or hit.year
        entry.venue = entry.venue or hit.venue
        if hit.citations is not None:
            entry.citations = max(entry.citations or 0, hit.citations)
        entry.structured = entry.structured or hit.academic
        entry.oa_url = entry.oa_url or hit.oa_url
        if hit.academic and entry.url.startswith("https://doi.org") is False and hit.url.startswith("https://doi.org"):
            entry.url = hit.url  # prefer the stable DOI link
        for k in keys:
            alias.setdefault(k, primary)
        entry = entries[primary]
        if hit.snippet and hit.snippet not in entry.snippets:
            entry.snippets.append(hit.snippet)
        if hit.query not in entry.queries:
            entry.queries.append(hit.query)

    papers: list[Paper] = []
    for key, entry in entries.items():
        host = host_of(entry.url)
        snippet = max(entry.snippets, key=len) if entry.snippets else None
        text = f"{entry.title} {snippet or ''}"
        year, year_evidence = None, EvidenceKind.UNKNOWN
        if entry.year:
            year, year_evidence = entry.year, EvidenceKind.SOURCE_METADATA
        elif entry.arxiv_year:
            year, year_evidence = entry.arxiv_year, EvidenceKind.SOURCE_METADATA
        else:
            years = [int(y) for y in _YEAR.findall(text)]
            if years:
                year, year_evidence = max(years), EvidenceKind.WEB_VERIFIED
        papers.append(Paper(
            id=hashlib.sha1(key.encode()).hexdigest()[:12], title=entry.title, url=entry.url, source=host,
            venue=entry.venue or venue_for(host), authors=list(entry.authors) if entry.authors else None,
            publication_year=year, year_evidence=year_evidence, doi=entry.doi, citation_count=entry.citations,
            open_access_url=entry.oa_url,
            abstract_snippet=snippet, matched_queries=entry.queries,
            retrieval_status="snippet_only" if snippet else "title_only",
            evidence=[EvidenceKind.WEB_VERIFIED] + ([EvidenceKind.SOURCE_METADATA] if entry.structured or entry.arxiv_year or entry.doi else []),
        ))
    return papers, rejected, merged
