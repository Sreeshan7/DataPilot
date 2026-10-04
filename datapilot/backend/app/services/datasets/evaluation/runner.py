"""The evaluation runner: ``evaluate_ranker(ranker, benchmark)``.

This is the only function that produces a complete report, and it is deliberately
generic over the ranker. A ranker is anything exposing
``rank(candidates, requirements) -> ranked candidates``. Today that is
``DatasetRankingEngine``; tomorrow it could be an embedding reranker or a learned
model, and the benchmark would judge it the same way. That is what makes the
framework a measurement instrument rather than a test of one implementation.

Separation enforced here:

* the runner never mutates ranking behaviour and never adjusts a weight;
* grades are attached by candidate **id**, so a ranker that reorders, filters or
  duplicates candidates cannot mis-associate a grade;
* retrieval and ranking are reported as separate numbers, because a benchmark
  cannot tell you which of the two is broken on its own.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional, Protocol, Sequence

from app.schemas.dataset import DatasetCandidate
from app.schemas.project import ProjectAnalysisData
from app.services.datasets.evaluation.benchmark import BENCHMARK_VERSION, load_benchmark
from app.services.datasets.evaluation.diagnostics import (
    build_ranked_items,
    suspected_unknown_as_mismatch,
)
from app.services.datasets.evaluation.fixtures import compile_case
from app.services.datasets.evaluation.labels import (
    HIGHLY_RELEVANT_THRESHOLD,
    RELEVANT_THRESHOLD,
)
from app.services.datasets.evaluation.metrics import (
    aggregate_metrics,
    case_metrics,
    incompatible_top_rate,
)
from app.services.datasets.evaluation.models import (
    BenchmarkCase,
    CaseEvaluation,
    CompiledCase,
    EvaluationReport,
    FailureKind,
)
from app.services.datasets.evaluation.taxonomy import classify_failures


class CandidateRanker(Protocol):
    """The interface every ranker must satisfy to be evaluated."""

    def rank(
        self, candidates: list[DatasetCandidate], requirements: ProjectAnalysisData
    ) -> list[DatasetCandidate]:  # pragma: no cover - protocol
        ...


def evaluate_case(
    case: BenchmarkCase, ranker: CandidateRanker
) -> CaseEvaluation:
    """Run one case and produce its metrics plus diagnostics.

    Candidate inputs are deep-copied so a ranker that mutates its inputs cannot
    contaminate a later ablation run of the same case.
    """
    compiled: CompiledCase = compile_case(case)
    ranked_candidates = ranker.rank(list(compiled.candidates), compiled.requirements)
    items = build_ranked_items(ranked_candidates, compiled.grades_by_id)

    pool_grades = [compiled.grades_by_id[c.id] for c in compiled.candidates]
    ranked_grades = [item.grade for item in items]

    failures = classify_failures(compiled, items)
    best = _highest_grade(items)
    pool_best = max(pool_grades) if pool_grades else 0

    # Retrieval vs ranking: a pool without the best available dataset cannot
    # produce a good ranking, and scoring it as a ranking failure is a lie about
    # where the defect is.
    if not case.retrieval_complete:
        kind = FailureKind.RETRIEVAL
    elif failures:
        kind = FailureKind.RANKING
    else:
        kind = FailureKind.NONE

    query = _describe_query(case)

    return CaseEvaluation(
        case_id=case.id,
        category=case.category,
        difficulty=case.difficulty,
        description=case.description,
        query=query,
        grades=dict(compiled.grades_by_id),
        ranked=items,
        metrics=case_metrics(ranked_grades, pool_grades),
        failures=failures,
        failure_kind=kind,
        best_available_retrieved=pool_best >= HIGHLY_RELEVANT_THRESHOLD,
        best_available_rank=_rank_of_grade(items, max(HIGHLY_RELEVANT_THRESHOLD, 0)),
        suspected_unknown_as_mismatch=suspected_unknown_as_mismatch(items),
    )


def _highest_grade(items) -> Optional[int]:
    return max((item.grade for item in items), default=None)


def _rank_of_grade(items, grade: int) -> Optional[int]:
    for item in items:
        if item.grade >= grade:
            return item.rank
    return None


def _describe_query(case: BenchmarkCase) -> str:
    """A one-line, human-readable summary of what the project asked for.

    Used as the report's "query" column so a reader can see what was being asked
    without cross-referencing the case definition.
    """
    req = case.requirements
    parts: list[str] = []
    if req.domain:
        parts.append(f"domain={req.domain}")
    if req.tasks:
        parts.append(f"task={'/'.join(req.tasks)}")
    if req.target:
        parts.append(f"target={req.target!r}")
    if req.features:
        parts.append(f"features={', '.join(req.features[:4])}")
    return "; ".join(parts) or "(no requirements)"


def evaluate_ranker(
    ranker: CandidateRanker,
    benchmark: Optional[Sequence[BenchmarkCase]] = None,
) -> EvaluationReport:
    """Evaluate ``ranker`` over the whole benchmark and return a report.

    :param ranker: anything with ``rank(candidates, requirements)``.
    :param benchmark: cases to run; defaults to the full corpus.
    :returns: a report carrying aggregate metrics, diagnostics, retrieval recall,
        the failure taxonomy, and per-case detail.
    """
    cases = list(benchmark) if benchmark is not None else load_benchmark()
    per_case: list[CaseEvaluation] = [evaluate_case(case, ranker) for case in cases]

    per_case_metrics = {evaluation.case_id: evaluation.metrics for evaluation in per_case}
    metrics = aggregate_metrics(per_case_metrics)

    diagnostics = _aggregate_diagnostics(per_case)
    retrieval_recall = _retrieval_recall(per_case)
    ranking_quality = _ranking_quality(per_case)
    taxonomy_counts = _taxonomy_counts(per_case)
    kind_counts = _failure_kind_counts(per_case)

    return EvaluationReport(
        benchmark_version=BENCHMARK_VERSION,
        ranker=type(ranker).__name__,
        case_count=len(per_case),
        candidate_count=sum(len(c.candidates) for c in cases),
        graded_candidate_count=sum(len(c.grades) for c in cases),
        metrics=metrics,
        diagnostics=diagnostics,
        retrieval_recall=retrieval_recall,
        ranking_quality=ranking_quality,
        failure_taxonomy=taxonomy_counts,
        failure_kind_counts=kind_counts,
        generated_at=datetime.now(tz=timezone.utc),
        per_case=per_case,
    )


def _aggregate_diagnostics(per_case: Sequence[CaseEvaluation]) -> dict[str, float]:
    """The headline diagnostic rates: incompatible-in-top-3 and UNKNOWN-as-MISMATCH."""
    n = len(per_case)
    if n == 0:
        return {}
    incompatible = sum(
        incompatible_top_rate([item.grade for item in e.ranked], k=3) for e in per_case
    ) / n
    unknown_rate = (
        sum(e.suspected_unknown_as_mismatch for e in per_case)
        / max(1, sum(len(e.ranked) for e in per_case))
    )
    top1_miss = sum(
        1.0 for e in per_case if not e.ranked or e.ranked[0].grade < HIGHLY_RELEVANT_THRESHOLD
    ) / n
    top3_miss = sum(
        1.0 for e in per_case if not any(item.grade >= HIGHLY_RELEVANT_THRESHOLD for item in e.ranked[:3])
    ) / n
    return {
        "incompatible_in_top3": incompatible,
        "unknown_treated_as_mismatch": unknown_rate,
        "no_grade3_at_rank1": top1_miss,
        "no_grade3_in_top3": top3_miss,
    }


def _retrieval_recall(per_case: Sequence[CaseEvaluation]) -> float:
    if not per_case:
        return 0.0
    return sum(1.0 for e in per_case if e.best_available_retrieved) / len(per_case)


def _ranking_quality(per_case: Sequence[CaseEvaluation]) -> dict[str, float]:
    """Ranking quality restricted to cases where retrieval succeeded.

    Averaging over all cases mixes "the query missed" with "the ranker was wrong";
    these numbers are the second without the first.
    """
    answered = [e for e in per_case if e.best_available_retrieved]
    if not answered:
        return {}
    return aggregate_metrics({e.case_id: e.metrics for e in answered})


def _taxonomy_counts(per_case: Sequence[CaseEvaluation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evaluation in per_case:
        for failure in evaluation.failures:
            counts[failure.category.value] = counts.get(failure.category.value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _failure_kind_counts(per_case: Sequence[CaseEvaluation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for evaluation in per_case:
        counts[evaluation.failure_kind.value] = counts.get(evaluation.failure_kind.value, 0) + 1
    return counts


__all__ = [
    "CandidateRanker",
    "evaluate_case",
    "evaluate_ranker",
]