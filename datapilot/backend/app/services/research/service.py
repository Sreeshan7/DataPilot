"""Research orchestration: queries -> web search -> normalise -> rank -> methods -> recommendations -> [one Gemini call].

Depends on the ``ResearchProvider`` interface, not on any one provider. Gemini calls: zero during retrieval,
at most one for synthesis, and none unless the caller asked for AI. Whatever happens in the optional layer, the
verified literature is returned.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.research import (
    EvidenceKind,
    ProvenanceNote,
    ResearchRequest,
    ResearchResponse,
    ResearchStatus,
    ResearchSummary,
)
from app.services.research import methods as method_tools
from app.services.research.normalize import normalise
from app.services.research.providers import ProviderError, ResearchProvider, SearchHit, WebSearchProvider
from app.services.research.scholarly import ArxivProvider, CrossrefProvider, OpenAlexProvider
from app.services.research.queries import build_queries, problem_phrase
from app.services.research.ranking import rank
from app.services.research.recommend import recommend
from app.services.research.synthesis import ResearchWriter, synthesise

logger = get_logger(__name__)

MIN_RELEVANCE = 15.0


MIN_PRIMARY_PAPERS = 3  # fewer than this from the scholarly indexes and the web search is tried as well


class ResearchService:
    """Providers are tried in two tiers: scholarly indexes first (real structured metadata), web search only as a
    fallback. The service depends on the ``ResearchProvider`` interface, not on any one source."""

    def __init__(
        self, settings: Optional[Settings] = None, provider: Optional[ResearchProvider] = None,
        writer: Optional[ResearchWriter] = None, providers: Optional[list[ResearchProvider]] = None,
    ) -> None:
        self._settings = settings or get_settings()
        if providers is not None:
            self._providers = providers
        elif provider is not None:
            self._providers = [provider]
        else:
            web = WebSearchProvider(self._settings)
            web.delay = self._settings.research_query_delay_seconds
            self._providers = [OpenAlexProvider(self._settings), ArxivProvider(self._settings),
                               CrossrefProvider(self._settings), web]
        self._writer = writer

    def _research_writer(self) -> Optional[ResearchWriter]:
        if self._writer is None:
            from app.services.llm_service import LLMService  # built only if AI is actually requested

            self._writer = LLMService(self._settings)
        return self._writer

    def _response(self, status: ResearchStatus, message: Optional[str], queries: list[str], **kw) -> ResearchResponse:
        used = kw.pop("used", None) or [p.name for p in self._providers[:1]]
        summary = ResearchSummary(
            queries_used=len(queries), results_retrieved=kw.pop("retrieved", 0), papers_found=kw.pop("found", 0),
            papers_selected=len(kw.get("papers", [])), results_rejected_as_non_academic=kw.pop("rejected", 0),
            search_provider="+".join(used))
        return ResearchResponse(status=status, request_id=str(uuid.uuid4()), message=message, queries=queries,
                                research_summary=summary, **kw)

    async def _search_one(
        self, provider: ResearchProvider, queries: list[str], errors: list[ProviderError]
    ) -> list[SearchHit]:
        s = self._settings
        hits: list[SearchHit] = []
        limit = getattr(provider, "max_queries", None)
        delay = getattr(provider, "delay", 0.0)
        failures_in_a_row = 0
        for index, query in enumerate(queries[:limit] if limit else queries):
            if index and delay:
                await asyncio.sleep(delay)  # bursts get rate limited
            for attempt in (0, 1):
                try:
                    hits.extend(await provider.search(query, s.research_results_per_query))
                    failures_in_a_row = 0
                    break
                except ProviderError as exc:
                    if exc.code == "RATE_LIMITED" and attempt == 0:
                        await asyncio.sleep(s.research_rate_limit_retry_seconds)
                        continue  # one polite retry
                    errors.append(exc)
                    failures_in_a_row += 1
                    break
                except Exception as exc:  # a provider bug is an unavailable source, not a 500
                    logger.error("Research provider %s failed: %s", provider.name, type(exc).__name__)
                    errors.append(ProviderError("UNAVAILABLE", f"{provider.name} failed unexpectedly."))
                    failures_in_a_row += 1
                    break
            if failures_in_a_row >= 2:
                break  # stop asking a service that keeps failing
        return hits

    async def _gather(self, providers: list[ResearchProvider], queries: list[str], errors: list[ProviderError]):
        batches = await asyncio.gather(*(self._search_one(p, queries, errors) for p in providers))
        return [h for batch in batches for h in batch], [p.name for p, b in zip(providers, batches) if b]

    async def discover(self, request: ResearchRequest) -> ResearchResponse:
        s = self._settings
        queries, why_not = build_queries(request.project, request.dataset, s.research_max_queries)
        if not queries:
            return self._response(ResearchStatus.NEEDS_CLARIFICATION, why_not, [])
        if not s.research_enabled:
            return self._response(ResearchStatus.SEARCH_UNAVAILABLE, "Research is turned off on this server.", queries)

        errors: list[ProviderError] = []
        primary = [p for p in self._providers if not getattr(p, "fallback", False)]
        fallback = [p for p in self._providers if getattr(p, "fallback", False)]
        hits, used = await self._gather(primary, queries, errors)
        papers, rejected, _ = normalise(hits)
        if len(papers) < MIN_PRIMARY_PAPERS and fallback:
            more, used_more = await self._gather(fallback, queries[:3], errors)
            hits += more
            used += used_more
            papers, rejected, _ = normalise(hits)
        if not hits and errors:
            return self._response(
                ResearchStatus.SEARCH_UNAVAILABLE,
                f"{errors[0].message} No papers are shown, because none could be verified.", queries,
                limitations=["Every literature source failed or was rate limited; try again in a moment."])

        method_tools.annotate_methods(papers)
        ranked = [p for p in rank(papers, request.project, request.dataset, min_year=request.options.min_year)
                  if p.relevance_score >= MIN_RELEVANCE]
        selected = ranked[: request.options.max_papers]
        if not selected:
            return self._response(
                ResearchStatus.NO_RESULTS,
                "The literature sources returned nothing that matched this problem well enough to show.",
                queries, retrieved=len(hits), found=len(papers), rejected=rejected, used=used,
                limitations=self._limitations(errors, request, 0))

        methods = method_tools.aggregate_methods(selected)
        metrics = method_tools.extract_metrics(selected)
        recommendations = recommend(methods, request.project, request.dataset)
        interpretation, gemini = await synthesise(
            self._research_writer() if request.options.use_ai and s.research_gemini_enabled else None,
            s, request.project, request.dataset, selected, methods, metrics, requested=request.options.use_ai)

        provenance = [
            ProvenanceNote(source="+".join(used) or "search", evidence=EvidenceKind.WEB_VERIFIED,
                           note="Papers, titles, abstracts, authors, years, DOIs and metrics come from the sources named here (scholarly indexes and/or web search)."),
            ProvenanceNote(source="datapilot", evidence=EvidenceKind.SOURCE_METADATA,
                           note="Relevance scores, method counts and recommendations are deterministic computations over that evidence."),
        ]
        if interpretation is not None:
            provenance.append(ProvenanceNote(
                source="gemini", evidence=EvidenceKind.AI_INFERRED,
                note="The AI interpretation is inference over the retrieved evidence and is not verified literature."))
        return self._response(
            ResearchStatus.SUCCESS, None, queries, retrieved=len(hits), found=len(papers), rejected=rejected, used=used,
            datapilot_summary=self._summary(selected, methods, request), papers=selected, methods=methods,
            benchmarks=metrics, algorithm_recommendations=recommendations, ai_interpretation=interpretation,
            gemini=gemini, limitations=self._limitations(errors, request, len(selected)), provenance=provenance)

    @staticmethod
    def _summary(papers, methods, request: ResearchRequest) -> str:
        top = ", ".join(f"{m.method} ({m.verified_paper_count})" for m in methods[:4])
        problem = problem_phrase(request.project) or "this problem"
        return (f"Found {len(papers)} academic sources about {problem}. "
                + (f"Methods named most often in their retrieved text: {top}. " if top else
                   "Their retrieved text names no standard method. ")
                + "This is what the sources say; it is not a measurement on your dataset.")

    @staticmethod
    def _limitations(errors, request: ResearchRequest, selected: int) -> list[str]:
        out = [
            "Evidence is limited to the title and abstract (or snippet) a source returned: full papers were not opened, so "
            "methods and figures not mentioned there are unknown, not absent.",
            "Authors, years and citation counts appear only where a scholarly index (OpenAlex, arXiv, Crossref) supplied them.",
            "Reported figures belong to each source's own experiments and datasets and do not transfer to yours.",
        ]
        if errors:
            out.append(f"{len(errors)} search request(s) failed, so the results may be incomplete.")
        if request.project.task is None and (request.dataset is None or request.dataset.task is None):
            out.append("No task was given, so the queries and the compatibility ratings are broader and less certain.")
        if selected and selected < 3:
            out.append("Very few relevant sources were found; treat the picture as thin.")
        return out
