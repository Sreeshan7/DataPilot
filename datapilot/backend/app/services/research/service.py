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
    GeminiResearchUsage,
    Interpretation,
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
from app.services.research import assist
from app.services.research.progress import Progress
from app.services.research.queries import QueryPlan, build_queries, understand
from app.services.research.ranking import rank
from app.services.research.recommend import recommend
from app.services.research.synthesis import ResearchWriter, synthesise

logger = get_logger(__name__)

MIN_RELEVANCE = 30.0
ASSIST_MIN_HEADROOM = 3  # leave calls for the rest of the app: assist only when the budget is comfortable


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
            excluded_off_topic=kw.pop("off_topic", 0),
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

    async def discover(self, request: ResearchRequest, progress: Optional[Progress] = None) -> ResearchResponse:
        s = self._settings
        pg = progress or Progress()
        plan = understand(request.project, request.dataset)
        assisted: list[str] = []
        notes: list[str] = []
        wants_assist = request.options.ai_assist and s.research_ai_assist_enabled and s.research_gemini_enabled
        writer = self._research_writer() if wants_assist else None
        can_assist = writer is not None and writer.is_configured and getattr(writer, "headroom", 99) >= ASSIST_MIN_HEADROOM
        extra: list[str] = []
        steps = [("understand", "Reading your question")]
        if can_assist:
            steps.append(("ai-question", "Asking AI for synonyms and related terms"))
        steps += [("queries", "Building search queries"), ("search", "Searching OpenAlex, arXiv and Crossref"),
                  ("dedupe", "Removing duplicates and non-academic pages"),
                  ("gate", "Checking each paper is really about your subject")]
        if can_assist:
            steps.append(("ai-relevance", "AI is reading titles and abstracts"))
        steps += [("methods", "Comparing methods and reported results"), ("recommend", "Matching methods to your task")]
        if request.options.use_ai and s.research_gemini_enabled:
            steps.append(("synthesis", "Writing the AI interpretation"))
        steps.append(("finish", "Preparing your findings"))
        await pg.plan(steps)
        await pg.start("understand")
        await pg.done("understand", ", ".join(plan.core_terms) if plan.problem else "needs more detail")
        if plan.problem and can_assist:
            await pg.start("ai-question")
            understood, why = await assist.understand_question(writer, s, request.project, request.dataset, plan)
            if understood:
                plan.core_groups, plan.ai_assisted = understood.core_groups, True
                plan.context_terms = understood.context_terms or plan.context_terms
                plan.problem = understood.problem
                extra = understood.queries[:2]
                assisted.append("question")
                await pg.done("ai-question", f"{len(plan.all_core_words)} terms")
            elif why:
                notes.append(f"AI question understanding skipped: {why}.")
                await pg.skip("ai-question", "used the built-in reading")
        await pg.start("queries")
        queries, why_not = build_queries(request.project, request.dataset, s.research_max_queries, plan=plan)
        if not queries:
            return self._response(ResearchStatus.NEEDS_CLARIFICATION, why_not, [])
        queries = list(dict.fromkeys(queries[: max(1, s.research_max_queries - len(extra))] + extra))
        await pg.done("queries", f"{len(queries)} queries")
        shown = Interpretation(problem=plan.problem, core_terms=plan.core_terms, context_terms=plan.context_terms,
                               corrections=plan.corrections, ai_assisted=plan.ai_assisted)
        if not s.research_enabled:
            return self._response(ResearchStatus.SEARCH_UNAVAILABLE, "Research is turned off on this server.", queries,
                                  interpretation=shown)

        errors: list[ProviderError] = []
        await pg.start("search", f"{len(queries)} queries")
        primary = [p for p in self._providers if not getattr(p, "fallback", False)]
        fallback = [p for p in self._providers if getattr(p, "fallback", False)]
        hits, used = await self._gather(primary, queries, errors)
        papers, rejected, _ = normalise(hits)
        if len(papers) < MIN_PRIMARY_PAPERS and fallback:
            more, used_more = await self._gather(fallback, queries[:3], errors)
            hits += more
            used += used_more
            papers, rejected, _ = normalise(hits)
        await pg.done("search", f"{len(hits)} results from {len(used)} source{'s' if len(used) != 1 else ''}")
        await pg.start("dedupe")
        await pg.done("dedupe", f"{len(papers)} distinct papers")
        if not hits and errors:
            return self._response(
                ResearchStatus.SEARCH_UNAVAILABLE,
                f"{errors[0].message} No papers are shown, because none could be verified.", queries,
                limitations=["Every literature source failed or was rate limited; try again in a moment."],
                interpretation=shown)

        await pg.start("gate")
        method_tools.annotate_methods(papers)
        gated = rank(papers, request.project, request.dataset, min_year=request.options.min_year, plan=plan)
        ranked = [p for p in gated if p.relevance_score >= MIN_RELEVANCE]
        await pg.done("gate", f"{len(ranked)} kept, {max(0, len(papers) - len(ranked))} left out")
        judge_ok = writer is not None and writer.is_configured and getattr(writer, "headroom", 99) >= ASSIST_MIN_HEADROOM - 1
        if can_assist and not (ranked and judge_ok):
            await pg.skip("ai-relevance", "nothing to judge or budget low")
        if ranked and judge_ok:
            await pg.start("ai-relevance", f"{min(len(ranked), assist.JUDGE_BATCH)} papers")
            judged, why = await assist.judge_relevance(writer, s, request.project, plan, ranked)
            if judged:
                ranked = self._apply_judgements(ranked, judged)
                assisted.append("relevance")
                await pg.done("ai-relevance", f"{len(judged)} papers judged")
            elif why:
                notes.append(f"AI relevance check skipped: {why}.")
                await pg.skip("ai-relevance", "used the built-in ranking")
        off_topic = max(0, len(papers) - len(ranked))
        selected = ranked[: request.options.max_papers]
        if not selected:
            return self._response(
                ResearchStatus.NO_RESULTS,
                "The literature sources returned nothing that was clearly about this problem, so nothing is shown "
                "rather than showing loosely related papers. Try naming the subject more specifically.",
                queries, retrieved=len(hits), found=len(papers), rejected=rejected, used=used, off_topic=off_topic,
                limitations=self._limitations(errors, request, 0) + notes, interpretation=shown,
                gemini=GeminiResearchUsage(requested=request.options.use_ai, calls_made=len(assisted),
                                           assisted=assisted))

        await pg.start("methods")
        methods = method_tools.aggregate_methods(selected)
        metrics = method_tools.extract_metrics(selected)
        await pg.done("methods", f"{len(methods)} methods, {len(metrics)} reported figures")
        await pg.start("recommend")
        recommendations = recommend(methods, request.project, request.dataset)
        await pg.done("recommend", f"{len(recommendations)} ranked")
        if request.options.use_ai and s.research_gemini_enabled:
            await pg.start("synthesis")
        interpretation, gemini = await synthesise(
            self._research_writer() if request.options.use_ai and s.research_gemini_enabled else None,
            s, request.project, request.dataset, selected, methods, metrics, requested=request.options.use_ai)
        gemini.calls_made += len(assisted)
        gemini.assisted = assisted
        if request.options.use_ai and s.research_gemini_enabled:
            await (pg.done("synthesis", "checked against the evidence") if interpretation is not None
                   else pg.skip("synthesis", "unavailable; showing the verified literature"))
        await pg.start("finish")

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
            off_topic=off_topic, interpretation=shown,
            datapilot_summary=self._summary(selected, methods, plan), papers=selected, methods=methods,
            benchmarks=metrics, algorithm_recommendations=recommendations, ai_interpretation=interpretation,
            gemini=gemini, limitations=self._limitations(errors, request, len(selected)) + notes, provenance=provenance)

    @staticmethod
    def _apply_judgements(ranked, judged):
        """Drop papers the AI called unrelated (0) and, if enough remain, loosely related ones (1); order by AI fit."""
        out = []
        for p in ranked:
            j = judged.get(p.id)
            out.append(p.model_copy(update={"ai_relevance": j.relevance, "ai_reason": j.reason or None}) if j else p)
        keep = [p for p in out if p.ai_relevance is None or p.ai_relevance >= 2]
        if len(keep) < 3:
            keep = [p for p in out if p.ai_relevance is None or p.ai_relevance >= 1]
        keep.sort(key=lambda p: (-(p.ai_relevance if p.ai_relevance is not None else 1), -p.relevance_score, p.url))
        return keep

    @staticmethod
    def _summary(papers, methods, plan: QueryPlan) -> str:
        top = ", ".join(f"{m.method} ({m.verified_paper_count})" for m in methods[:4])
        problem = plan.problem or "this problem"
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
