"""The ONE optional Gemini call, and the validation that decides whether any of its answer is used.

Budget: exactly zero calls unless the caller asks for AI and the feature is enabled and configured; then
exactly one. No per-paper or per-query calls, no loops, no retries here. The key is never read in this module.

A reply is rejected as a whole (and the deterministic wording kept) if any text field contains a link or DOI,
a percentage that no retrieved source reported, an accuracy promise, a claim that experiments were run,
author attribution, a paper id that was not retrieved, or a quoted/backticked name that is neither a
retrieved paper title, a known method nor one of the user's own columns. Failures never raise.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Protocol

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.prompts.research import build_research_prompt
from app.schemas.research import (
    AIInterpretation,
    GeminiResearchUsage,
    LLMResearchSynthesis,
    MethodSupport,
    Paper,
    ReportedMetric,
    ResearchDataset,
    ResearchProject,
)
from app.services.llm_service import LLMServiceError
from app.services.research.methods import CATALOG
from app.services.research.normalize import normalise_title

logger = get_logger(__name__)

MAX_FIELD = 600
MAX_ITEMS = 6
SNIPPET_CHARS = 280
_URL = re.compile(r"https?://|www\.|\b10\.\d{4,9}/|\bdoi\b", re.I)
_PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
_PROMISE = re.compile(r"\bguarantee\w*|\bwill\s+(?:achieve|reach|attain|deliver|get)\b|\bexpect(?:ed)? to (?:achieve|reach)\b", re.I)
_RAN = re.compile(r"\b(?:we|i)\s+(?:ran|trained|tested|evaluated|benchmarked|experimented)\b|\bour (?:experiments?|results)\b|\bwe found\b", re.I)
_AUTHORS = re.compile(r"\bet al\b|\bauthored by\b|\bby (?:Dr|Prof)\b", re.I)
_PAPER_REF = re.compile(r"\[p:([A-Za-z0-9]+)\]")
_QUOTED = re.compile(r"`([^`\n]+)`|\"([^\"\n]{12,})\"")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_KNOWN = {m.name.lower() for m in CATALOG}


class ResearchWriter(Protocol):
    @property
    def is_configured(self) -> bool: ...

    async def generate_research_synthesis(self, user_prompt: str) -> LLMResearchSynthesis: ...


def build_payload(
    project: ResearchProject, dataset: Optional[ResearchDataset], papers: list[Paper],
    methods: list[MethodSupport], metrics: list[ReportedMetric],
) -> dict:
    return {
        "project": {
            "problem": project.problem, "domain": project.domain, "task": project.task, "target": project.target,
            "features": project.features[:20],
            "dataset_rows": dataset.rows if dataset else None,
        },
        "papers": [
            {"id": p.id, "title": p.title[:200], "year": p.publication_year, "source": p.source,
             "snippet": (p.abstract_snippet or "")[:SNIPPET_CHARS], "methods_named": p.methods}
            for p in papers],
        "method_counts": {m.method: m.verified_paper_count for m in methods},
        "reported_figures": [
            {"paper": m.paper_id, "metric": m.metric, "value": m.value, "method": m.method} for m in metrics[:15]],
    }


def _problems(text: str, allowed_percent: set[str], paper_ids: set[str], titles: set[str], own_terms: set[str]) -> Optional[str]:
    if _CONTROL.search(text) or _URL.search(text):
        return "contains a link, DOI or control characters"
    if _PROMISE.search(text):
        return "promises a result"
    if _RAN.search(text):
        return "claims experiments were run"
    if _AUTHORS.search(text):
        return "attributes work to authors"
    for pct in _PERCENT.findall(text):
        if pct.replace(" ", "") not in allowed_percent:
            return "states a percentage no retrieved source reported"
    for ref in _PAPER_REF.findall(text):
        if ref not in paper_ids:
            return "cites a paper that was not retrieved"
    for a, b in _QUOTED.findall(text):
        span = (a or b).strip()
        norm = normalise_title(span)
        if norm in own_terms or span.lower() in _KNOWN:
            continue
        if any(norm and (norm in t or t in norm) for t in titles):
            continue
        return "names something that was not retrieved"
    return None


def validate(
    reply: LLMResearchSynthesis, papers: list[Paper], metrics: list[ReportedMetric], project: ResearchProject,
    dataset: Optional[ResearchDataset],
) -> tuple[Optional[AIInterpretation], Optional[str]]:
    allowed_percent = {m.value.replace(" ", "") for m in metrics if m.value.endswith("%")}
    paper_ids = {p.id for p in papers}
    titles = {normalise_title(p.title) for p in papers}
    own = {normalise_title(c) for c in (project.features + (dataset.columns if dataset else []))}
    own |= {normalise_title(x) for x in (project.target, project.problem, project.domain, project.task) if x}
    fields = {"landscape": reply.landscape, "comparison": reply.comparison, "fit_for_project": reply.fit_for_project}
    for name, text in fields.items():
        if len(text) > MAX_FIELD:
            return None, f"{name} was too long"
        problem = _problems(text, allowed_percent, paper_ids, titles, own)
        if problem:
            return None, f"{name} {problem}"
    lists = {"approaches": reply.approaches, "limitations": reply.limitations}
    cleaned: dict[str, list[str]] = {}
    for name, items in lists.items():
        if len(items) > MAX_ITEMS:
            items = items[:MAX_ITEMS]
        for item in items:
            if len(item) > MAX_FIELD:
                return None, f"{name} item was too long"
            problem = _problems(item, allowed_percent, paper_ids, titles, own)
            if problem:
                return None, f"{name} {problem}"
        cleaned[name] = [i.strip() for i in items if i.strip()]
    if not reply.landscape.strip():
        return None, "landscape was empty"
    named = {m for p in papers for m in p.methods}
    inferred = [m for m in reply.ai_inferred_methods[:MAX_ITEMS] if m.lower() in _KNOWN and m not in named]
    return AIInterpretation(
        landscape=reply.landscape.strip(), approaches=cleaned["approaches"], comparison=reply.comparison.strip(),
        fit_for_project=reply.fit_for_project.strip(), limitations=cleaned["limitations"],
        ai_inferred_methods=inferred), None


async def synthesise(
    writer: Optional[ResearchWriter], settings: Settings, project: ResearchProject, dataset: Optional[ResearchDataset],
    papers: list[Paper], methods: list[MethodSupport], metrics: list[ReportedMetric], *, requested: bool,
) -> tuple[Optional[AIInterpretation], GeminiResearchUsage]:
    if not requested:
        return None, GeminiResearchUsage(status="not_requested")
    if not settings.research_gemini_enabled:
        return None, GeminiResearchUsage(requested=True, status="disabled", note="AI synthesis is turned off on this server.")
    if not papers:
        return None, GeminiResearchUsage(requested=True, status="unavailable", note="There was no retrieved evidence to synthesise.")
    if writer is None or not writer.is_configured:
        return None, GeminiResearchUsage(requested=True, status="unavailable", note="No Gemini key is configured; the verified literature is shown without AI synthesis.")
    prompt = build_research_prompt(build_payload(project, dataset, papers, methods, metrics))
    try:
        reply = await asyncio.wait_for(writer.generate_research_synthesis(prompt), timeout=settings.llm_request_timeout)
    except asyncio.TimeoutError:
        code = "LLM_TIMEOUT"
    except LLMServiceError as exc:
        code = exc.code
    except Exception as exc:  # the optional layer must never break the research result
        code = type(exc).__name__
    else:
        interpretation, why = validate(reply, papers, metrics, project, dataset)
        if interpretation is None:
            logger.info("Research synthesis rejected: %s", why)
            return None, GeminiResearchUsage(
                requested=True, calls_made=1, status="rejected",
                note=f"The AI summary was discarded because it {why}; DataPilot's own summary is shown.")
        return interpretation, GeminiResearchUsage(requested=True, calls_made=1, status="used")
    logger.info("Research synthesis unavailable: %s", code)  # the code only, never the prompt or reply
    return None, GeminiResearchUsage(
        requested=True, calls_made=1, status="failed",
        note=f"The AI service did not respond usefully ({code}); the verified literature is shown without it.")
