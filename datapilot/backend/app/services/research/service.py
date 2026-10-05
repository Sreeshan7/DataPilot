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
from app.services.research.queries import build_queries, problem_phrase
from app.services.research.ranking import rank
from app.services.research.recommend import recommend
from app.services.research.synthesis import ResearchWriter, synthesise

logger = get_logger(__name__)

MIN_RELEVANCE = 15.0


class ResearchService:
    def __init__(
        self, settings: Optional[Settings] = None, provider: Optional[ResearchProvider] = None,
        writer: Optional[ResearchWriter] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider or WebSearchProvider(self._settings)
        self._writer = writer

    def _research_writer(self) -> Optional[ResearchWriter]:
        if self._writer is None:
            from app.services.llm_service import LLMService  # built only if AI is actually requested

            self._writer = LLMService(self._settings)
        return self._writer

    def _response(self, status: ResearchStatus, message: Optional[str], queries: list[str], **kw) -> ResearchResponse:
        summary = ResearchSummary(
            queries_used=len(queries), results_retrieved=kw.pop("retrieved", 0), papers_found=kw.pop("found", 0),
            papers_selected=len(kw.get("papers", [])), results_rejected_as_non_academic=kw.pop("rejected", 0),
            search_provider=self._provider.name)
        return ResearchResponse(status=status, request_id=str(uuid.uuid4()), message=message, queries=queries,
                                research_summary=summary, **kw)

    async def discover(self, request: ResearchRequest) -> ResearchResponse:
        s = self._settings
        queries, why_not = build_queries(request.project, request.dataset, s.research_max_queries)
        if not queries:
            return self._response(ResearchStatus.NEEDS_CLARIFICATION, why_not, [])
        if not s.research_enabled:
            return self._response(ResearchStatus.SEARCH_UNAVAILABLE, "Research is turned off on this server.", queries)

        errors: list[ProviderError] = []
        hits: list[SearchHit] = []
        limited_in_a_row = 0
        for index, query in enumerate(queries):
            if index and s.research_query_delay_seconds:
                await asyncio.sleep(s.research_query_delay_seconds)  # bursts get rate limited
            for attempt in (0, 1):
                try:
                    hits.extend(await self._provider.search(query, s.research_results_per_query))
                    limited_in_a_row = 0
                    break
                except ProviderError as exc:
                    if exc.code == "RATE_LIMITED" and attempt == 0:
                        await asyncio.sleep(s.research_rate_limit_retry_seconds)
                        continue  # one polite retry
                    errors.append(exc)
                    limited_in_a_row = limited_in_a_row + 1 if exc.code == "RATE_LIMITED" else 0
                    break
                except Exception as exc:  # a provider bug is an unavailable search, not a 500
                    logger.error("Research provider failed: %s", type(exc).__name__)
                    errors.append(ProviderError("UNAVAILABLE", "The search provider failed unexpectedly."))
                    break
            if limited_in_a_row >= 2:
                break  # stop asking a service that keeps saying no
        if not hits and errors:
            return self._response(
                ResearchStatus.SEARCH_UNAVAILABLE,
                f"{errors[0].message} No papers are shown, because none could be verified.", queries,
                limitations=["Web search is a best-effort, keyless path; it can be rate limited or change its page format."])

        papers, rejected, _ = normalise(hits)
        method_tools.annotate_methods(papers)
        ranked = [p for p in rank(papers, request.project, request.dataset, min_year=request.options.min_year)
                  if p.relevance_score >= MIN_RELEVANCE]
        selected = ranked[: request.options.max_papers]
        if not selected:
            return self._response(
                ResearchStatus.NO_RESULTS,
                "The web search returned no academic sources that matched this problem well enough to show.",
                queries, retrieved=len(hits), found=len(papers), rejected=rejected,
                limitations=self._limitations(errors, request, 0))

        methods = method_tools.aggregate_methods(selected)
        metrics = method_tools.extract_metrics(selected)
        recommendations = recommend(methods, request.project, request.dataset)
        interpretation, gemini = await synthesise(
            self._research_writer() if request.options.use_ai and s.research_gemini_enabled else None,
            s, request.project, request.dataset, selected, methods, metrics, requested=request.options.use_ai)

        provenance = [
            ProvenanceNote(source=self._provider.name, evidence=EvidenceKind.WEB_VERIFIED,
                           note="Papers, titles, snippets, years and metrics come from retrieved web results."),
            ProvenanceNote(source="datapilot", evidence=EvidenceKind.SOURCE_METADATA,
                           note="Relevance scores, method counts and recommendations are deterministic computations over that evidence."),
        ]
        if interpretation is not None:
            provenance.append(ProvenanceNote(
                source="gemini", evidence=EvidenceKind.AI_INFERRED,
                note="The AI interpretation is inference over the retrieved evidence and is not verified literature."))
        return self._response(
            ResearchStatus.SUCCESS, None, queries, retrieved=len(hits), found=len(papers), rejected=rejected,
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
            "Evidence is limited to the title and snippet the web search returned: full papers were not opened, so "
            "methods and figures not mentioned there are unknown, not absent.",
            "Authors and citation counts are not retrieved and are left unknown.",
            "Reported figures belong to each source's own experiments and datasets and do not transfer to yours.",
        ]
        if errors:
            out.append(f"{len(errors)} search request(s) failed, so the results may be incomplete.")
        if request.project.task is None and (request.dataset is None or request.dataset.task is None):
            out.append("No task was given, so the queries and the compatibility ratings are broader and less certain.")
        if selected and selected < 3:
            out.append("Very few relevant sources were found; treat the picture as thin.")
        return out
