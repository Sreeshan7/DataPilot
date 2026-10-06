"""Unified, source-agnostic dataset discovery.

    Project Requirements
            |
    DatasetDiscoveryService
        /          \\
  Kaggle          Gemini
  (real results)  (AI suggestions)
        \\          /
          Normalize  (done inside each source)
               |
          Deduplicate
               |
               Rank  (deterministic)
               |
       Unified ranked result

The service never branches on which source produced a candidate. Adding UCI,
Hugging Face or research-paper discovery means registering one more
:class:`~app.services.datasets.base.DatasetSource` in
``build_default_sources`` and changing nothing else.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Sequence

from app.core.config import Settings, get_settings
from app.core import progress
from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    MetadataDepth,
    SourceOutcome,
    SourceSearchRequest,
    SourceStatus,
    SourceType,
)
from app.schemas.project import ProjectAnalysisData, ProjectAnalysisRequest
from app.services.datasets.base import DatasetSource, DatasetSourceError, redact_secrets
from app.services.datasets.deduplicator import DatasetDeduplicator, name_token_key
from app.services.datasets.query_builder import build_dataset_search_query
from app.services.datasets.ranking import DatasetRankingEngine
from app.services.project_analyzer import ProjectAnalyzer

logger = get_logger(__name__)


@dataclass
class DatasetDiscoveryResult:
    """Outcome of one discovery run."""

    query: str
    candidates: list[DatasetCandidate] = field(default_factory=list)
    sources: list[SourceOutcome] = field(default_factory=list)
    requirements: Optional[ProjectAnalysisData] = None
    raw_candidate_count: int = 0
    duration_ms: float = 0.0
    enrichment_attempted: int = 0
    enrichment_succeeded: int = 0
    enrichment_error_codes: list[str] = field(default_factory=list)
    confirmed_suggestions: int = 0
    fit_judged: int = 0
    fit_reordered: bool = False
    fit_note: Optional[str] = None

    def count_by_source_type(self, source_type: SourceType) -> int:
        return sum(1 for c in self.candidates if c.source_type is source_type)

    @property
    def verified_count(self) -> int:
        return self.count_by_source_type(SourceType.VERIFIED_EXTERNAL)

    @property
    def unverified_count(self) -> int:
        return self.count_by_source_type(SourceType.AI_SUGGESTED)


class DatasetDiscoveryService:
    """Queries every registered source, then dedupes and ranks the results."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        sources: Optional[Sequence[DatasetSource]] = None,
        ranking_engine: Optional[DatasetRankingEngine] = None,
        project_analyzer: Optional[ProjectAnalyzer] = None,
        fit_writer=None,
    ) -> None:
        from app.services.datasets import build_default_sources

        self._settings = settings or get_settings()
        self._sources: list[DatasetSource] = (
            list(sources) if sources is not None else build_default_sources(self._settings)
        )
        self._ranking = ranking_engine or DatasetRankingEngine(self._settings)
        self._project_analyzer = project_analyzer
        self._fit_writer = fit_writer

    @property
    def sources(self) -> list[DatasetSource]:
        return list(self._sources)

    def ranking_weights(self) -> dict[str, float]:
        """Active ranking weights, so a response documents its own scoring."""
        return self._ranking.weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int,
        request_id: Optional[str] = None,
    ) -> DatasetDiscoveryResult:
        """Discover datasets for a bare text query.

        No project requirements exist, so sources that require them (GenAI) are
        reported as ``skipped`` rather than being called with an empty context.
        This is what keeps ``/datasets/search`` free of Gemini calls.
        """
        return await self._discover(
            query=query,
            limit=limit,
            requirements=None,
            request_id=request_id,
        )

    async def discover(
        self,
        requirements: ProjectAnalysisData,
        limit: int,
        request_id: Optional[str] = None,
        query: Optional[str] = None,
    ) -> DatasetDiscoveryResult:
        """Discover datasets for already-structured project requirements."""
        return await self._discover(
            query=query
            or build_dataset_search_query(
                requirements,
                max_terms=self._settings.dataset_discovery_max_query_terms,
            ),
            limit=limit,
            requirements=requirements,
            request_id=request_id,
        )

    async def recommend(
        self,
        description: str,
        limit: int,
        request_id: Optional[str] = None,
    ) -> tuple[ProjectAnalysisData, DatasetDiscoveryResult]:
        """Full DataPilot workflow: analyze -> discover -> rank.

        Returns the structured requirements together with the ranked dataset
        result so the caller can present both.
        """
        rid = request_id or str(uuid.uuid4())
        analyzer = self._project_analyzer or ProjectAnalyzer(self._settings)
        progress.start("requirements", "Extracting your project requirements")
        analysis = await analyzer.analyze(
            ProjectAnalysisRequest(description=description),
            request_id=rid,
        )
        requirements = analysis.data
        understanding = requirements.project_understanding
        bits = [b for b in (getattr(getattr(understanding, "domain", None), "value", None),
                            (understanding.ml_tasks[0].task if getattr(understanding, "ml_tasks", None) else None)) if b]
        progress.done("requirements", " · ".join(str(b) for b in bits) or None)
        result = await self.discover(requirements, limit=limit, request_id=rid)
        return requirements, await self._apply_fit_judge(description, requirements, result)

    async def _apply_fit_judge(self, description: str, requirements: ProjectAnalysisData, result: DatasetDiscoveryResult) -> DatasetDiscoveryResult:
        """One optional AI call that can reorder (never rescore) the ranked datasets by how well they match the user's setting."""
        if not self._settings.dataset_fit_judge_enabled or len(result.candidates) < 2:
            return result
        from app.services.datasets import fit_judge

        writer = self._fit_writer
        if writer is None:
            from app.services.llm_service import LLMService  # built only when needed; shares the process-wide key budget

            writer = self._fit_writer = LLMService(self._settings)
        understanding = requirements.project_understanding
        project = {
            "description": " ".join((description or "").split())[:600],
            "objective": getattr(understanding, "objective", None),
            "target": getattr(getattr(understanding, "target", None), "value", None),
            "domain": getattr(getattr(understanding, "domain", None), "value", None),
        }
        progress.start("fit", "Checking each dataset against your specific setting")
        fits, why = await fit_judge.judge_fit(writer, self._settings, project, result.candidates)
        if not fits:
            result.fit_note = f"AI setting check skipped: {why}." if why else None
            progress.skip("fit", "used the built-in ranking")
            return result
        result.candidates, result.fit_reordered = fit_judge.reorder(result.candidates, fits)
        result.fit_judged = len(fits)
        progress.done("fit", f"{len(fits)} datasets checked" + (", order adjusted" if result.fit_reordered else ""))
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _discover(
        self,
        query: str,
        limit: int,
        requirements: Optional[ProjectAnalysisData],
        request_id: Optional[str],
    ) -> DatasetDiscoveryResult:
        rid = request_id or str(uuid.uuid4())
        start = time.monotonic()
        normalized_query = " ".join((query or "").split()).strip()

        if not normalized_query:
            raise DatasetSourceError(
                code="DATASET_QUERY_EMPTY",
                message="A non-empty search query is required.",
            )

        limit = max(1, min(int(limit), self._settings.kaggle_max_limit))
        base_request = SourceSearchRequest(
            query=normalized_query,
            limit=limit,
            project_requirements=requirements,
        )

        verified_sources, suggestion_sources = self._partition_sources()
        outcomes: list[SourceOutcome] = []
        collected: list[DatasetCandidate] = []

        # Pass 1: verified external sources. Each source is queried exactly once.
        verified_names: list[str] = []
        for source in verified_sources:
            outcome, candidates = await self._run_source(source, base_request, rid)
            outcomes.append(outcome)
            collected.extend(candidates)
            verified_names.extend(c.name for c in candidates)

        # Pass 2: suggestion-based sources, told what verified sources already
        # found so they propose complements rather than repeats. Still one call
        # per source.
        suggestion_request = base_request.model_copy(
            update={"known_dataset_names": verified_names[:20]}
        )
        for source in suggestion_sources:
            outcome, candidates = await self._run_source(source, suggestion_request, rid)
            outcomes.append(outcome)
            collected.extend(candidates)

        raw_count = len(collected)
        progress.start("merge", "Removing duplicates across sources")

        # A fresh deduplicator per run: the index holds candidate references, so
        # reusing one across requests would leak results between callers.
        deduplicator = DatasetDeduplicator()
        deduplicated = deduplicator.deduplicate(collected)

        progress.done("merge", f"{len(deduplicated)} distinct datasets")
        confirmed = 0
        if requirements is not None:
            progress.start("confirm", "Checking AI suggestions against the registries")
            deduplicated, confirmed = await self._confirm_suggestions(
                deduplicated, deduplicator, verified_sources, rid
            )

        if requirements is not None:
            progress.done("confirm", f"{confirmed} confirmed" if confirmed else "none needed confirming")
        progress.start("rank", "Comparing candidates with your project")
        ranked = (
            self._ranking.rank(deduplicated, requirements)
            if requirements is not None
            else list(deduplicated)
        )

        progress.done("rank", f"{len(ranked)} ranked")
        enrichment = _EnrichmentStats()
        if requirements is not None:
            progress.start("enrich", "Reading details of the best matches")
            ranked, enrichment = await self._enrich_top(ranked, requirements, rid)
            progress.done("enrich", f"{enrichment.succeeded} of {enrichment.attempted} read" if enrichment.attempted else "already detailed")
        final = ranked[:limit]

        duration_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info(
            "[%s] Discovery for %r: %d raw -> %d unique -> %d returned "
            "(%d verified, %d unverified) in %.1fms",
            rid,
            normalized_query,
            raw_count,
            len(deduplicated),
            len(final),
            sum(1 for c in final if c.source_type is SourceType.VERIFIED_EXTERNAL),
            sum(1 for c in final if c.source_type is SourceType.AI_SUGGESTED),
            duration_ms,
        )

        if not final and outcomes:
            failed = [o for o in outcomes if o.status is SourceStatus.FAILED]
            if failed and all(o.status is not SourceStatus.OK for o in outcomes):
                # Every source failed: surface the real cause instead of an
                # empty list that reads like "no datasets exist".
                raise DatasetSourceError(
                    code=failed[0].error_code or "DATASET_DISCOVERY_FAILED",
                    message=failed[0].message or "All dataset sources failed.",
                )

        return DatasetDiscoveryResult(
            query=normalized_query,
            candidates=final,
            sources=outcomes,
            requirements=requirements,
            raw_candidate_count=raw_count,
            duration_ms=duration_ms,
            enrichment_attempted=enrichment.attempted,
            enrichment_succeeded=enrichment.succeeded,
            enrichment_error_codes=sorted(set(enrichment.error_codes)),
            confirmed_suggestions=confirmed,
        )

    # ------------------------------------------------------------------
    # Phase 3B: enrichment
    # ------------------------------------------------------------------

    async def _enrich_top(
        self,
        ranked: list[DatasetCandidate],
        requirements: ProjectAnalysisData,
        rid: str,
    ) -> tuple[list[DatasetCandidate], "_EnrichmentStats"]:
        """Re-read the best verified candidates from their registry, then re-rank.

        A registry listing is often too thin to judge a dataset (Kaggle's list
        endpoint returns no description at all), so a first-pass ranking says more
        about titles than about data. Only the top ``dataset_enrichment_top_k``
        candidates are enriched, which keeps the extra traffic bounded and spends
        it where it can change the answer.

        An enrichment failure never fails the request: that candidate stays at
        listing depth and says so.
        """
        stats = _EnrichmentStats()
        top_k = self._settings.dataset_enrichment_top_k
        if not self._settings.dataset_enrichment_enabled or top_k <= 0:
            return ranked, stats

        by_name = {source.name: source for source in self._sources}
        targets: list[tuple[DatasetCandidate, DatasetSource]] = []
        for candidate in ranked:
            if len(targets) >= top_k:
                break
            source = by_name.get(candidate.source)
            if (
                candidate.source_type is SourceType.VERIFIED_EXTERNAL
                and source is not None
                and source.supports_enrichment
                and candidate.metadata_depth is MetadataDepth.LISTING
            ):
                targets.append((candidate, source))
        if not targets:
            return ranked, stats

        stats.attempted = len(targets)
        outcomes = await asyncio.gather(
            *(self._enrich_one(candidate, source, rid) for candidate, source in targets)
        )

        replacements: dict[str, DatasetCandidate] = {}
        for (candidate, _), (enriched, error_code) in zip(targets, outcomes):
            if error_code:
                stats.error_codes.append(error_code)
            elif enriched is not candidate:
                replacements[candidate.id] = enriched
        stats.succeeded = len(replacements)
        if not replacements:
            return ranked, stats

        updated = [replacements.get(candidate.id, candidate) for candidate in ranked]
        return self._ranking.rank(updated, requirements), stats

    async def _enrich_one(
        self, candidate: DatasetCandidate, source: DatasetSource, rid: str
    ) -> tuple[DatasetCandidate, Optional[str]]:
        try:
            return await source.enrich(candidate), None
        except DatasetSourceError as exc:
            logger.warning("[%s] Enrichment of %s failed: %s", rid, candidate.id, exc.code)
            return candidate, exc.code
        except Exception as exc:
            logger.error(
                "[%s] Enrichment of %s raised %s", rid, candidate.id, type(exc).__name__
            )
            return candidate, "DATASET_SOURCE_UNEXPECTED_ERROR"

    # ------------------------------------------------------------------
    # Phase 3C: registry confirmation of AI suggestions
    # ------------------------------------------------------------------

    async def _confirm_suggestions(
        self,
        candidates: list[DatasetCandidate],
        deduplicator: DatasetDeduplicator,
        verified_sources: Sequence[DatasetSource],
        rid: str,
    ) -> tuple[list[DatasetCandidate], int]:
        """Look a few AI suggestions up by name in the verified registries.

        A suggestion becomes a registry record only when a registry returns a
        dataset whose *identifying name tokens are identical* to the suggestion's.
        The registry's record then wins the merge, and the model's own claims are
        kept apart as ``ai_suggested_claims``. A near-match, a miss, or a failed
        lookup leaves the suggestion unverified: confirming the wrong dataset is
        worse than confirming none.

        Bounded by ``dataset_verification_lookups`` suggestions, well-known and
        higher-confidence ones first, so the extra requests stay predictable.
        """
        budget = self._settings.dataset_verification_lookups
        if budget <= 0 or not verified_sources:
            return candidates, 0

        pending = [c for c in candidates if c.source_type is SourceType.AI_SUGGESTED]
        pending.sort(
            key=lambda c: (
                0 if c.is_well_known else 1,
                -float(c.confidence_score or 0.0),
                c.name.lower(),
            )
        )

        result = list(candidates)
        confirmed = 0
        for suggestion in pending[:budget]:
            key = name_token_key(suggestion.name)
            if not key:
                continue
            for source in verified_sources:
                if not source.is_configured:
                    continue
                try:
                    hits = await source.search(
                        SourceSearchRequest(query=suggestion.name, limit=5)
                    )
                except DatasetSourceError as exc:
                    logger.info(
                        "[%s] Confirmation lookup on %s failed: %s", rid, source.name, exc.code
                    )
                    continue
                except Exception as exc:
                    logger.error(
                        "[%s] Confirmation lookup on %s raised %s",
                        rid,
                        source.name,
                        type(exc).__name__,
                    )
                    continue
                match = next((h for h in hits if name_token_key(h.name) == key), None)
                if match is None:
                    continue
                merged = deduplicator.absorb(match)
                if merged is None or merged is suggestion:
                    break
                position = next(i for i, c in enumerate(result) if c is suggestion)
                if any(c is merged for c in result):
                    # The registry record was already in the list; drop the duplicate.
                    del result[position]
                else:
                    result[position] = merged
                confirmed += 1
                break
        return result, confirmed

    def _partition_sources(
        self,
    ) -> tuple[list[DatasetSource], list[DatasetSource]]:
        verified = [s for s in self._sources if s.source_type is SourceType.VERIFIED_EXTERNAL]
        suggestions = [s for s in self._sources if s.source_type is SourceType.AI_SUGGESTED]
        return verified, suggestions

    async def _run_source(
        self,
        source: DatasetSource,
        request: SourceSearchRequest,
        rid: str,
    ) -> tuple[SourceOutcome, list[DatasetCandidate]]:
        """Query one source, converting any failure into a reported outcome."""
        if not source.is_configured:
            return (
                SourceOutcome(
                    source=source.name,
                    source_type=source.source_type,
                    status=SourceStatus.SKIPPED,
                    message="Source is not configured",
                ),
                [],
            )

        if source.supports_project_requirements and request.project_requirements is None:
            return (
                SourceOutcome(
                    source=source.name,
                    source_type=source.source_type,
                    status=SourceStatus.SKIPPED,
                    message="No structured project requirements supplied for this query",
                ),
                [],
            )

        start = time.monotonic()
        progress.start(f"source-{source.name}", f"Searching {source.name}")
        try:
            candidates = await source.search(request)
        except DatasetSourceError as exc:
            logger.warning("[%s] Source %s failed: %s", rid, source.name, exc.code)
            outcome = SourceOutcome(
                source=source.name,
                source_type=source.source_type,
                status=SourceStatus.FAILED,
                error_code=exc.code,
                message=redact_secrets(exc.message),
            )
        except Exception as exc:
            # Never leak an upstream exception body into a response.
            logger.error("[%s] Source %s raised %s", rid, source.name, type(exc).__name__)
            outcome = SourceOutcome(
                source=source.name,
                source_type=source.source_type,
                status=SourceStatus.FAILED,
                error_code="DATASET_SOURCE_UNEXPECTED_ERROR",
                message="The dataset source failed unexpectedly.",
            )
        else:
            outcome = SourceOutcome(
                source=source.name,
                source_type=source.source_type,
                status=SourceStatus.OK,
                candidate_count=len(candidates),
            )
        outcome.duration_ms = round((time.monotonic() - start) * 1000, 2)
        if outcome.status is SourceStatus.OK:
            progress.done(f"source-{source.name}", f"{len(candidates)} candidates")
        else:
            progress.skip(f"source-{source.name}", "unavailable right now")
        return outcome, candidates if outcome.status is SourceStatus.OK else []


@dataclass
class _EnrichmentStats:
    attempted: int = 0
    succeeded: int = 0
    error_codes: list[str] = field(default_factory=list)
