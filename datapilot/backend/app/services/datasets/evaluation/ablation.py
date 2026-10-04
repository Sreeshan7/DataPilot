"""Ablation: what does each signal actually contribute to ranking quality?

An ablation answers a question a weight cannot: *if this signal vanished, would
ranking get worse?* A heavy weight that changes no ranking is dead weight; a light
weight that flips many rankings is doing more work than its number suggests.

Implementation note, because it matters for interpretation: a signal is removed
by setting its weight to zero and re-running the whole engine. The ranker itself
is never modified, patched or re-implemented, so an ablation measures the real
system. A weight of zero also drops the signal out of the weighted mean's
denominator, which is the correct reading of "remove this signal" rather than
"keep it and multiply by zero".
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from app.core.config import Settings
from app.services.datasets.evaluation.metrics import aggregate_metrics, metric_deltas
from app.services.datasets.evaluation.models import (
    AblationResult,
    BenchmarkCase,
    EvaluationReport,
)
from app.services.datasets.evaluation.runner import evaluate_case

#: Signal -> the Settings field that carries its weight.
ABLATION_TARGETS: dict[str, str] = {
    "popularity": "ranking_weight_popularity",
    "recency": "ranking_weight_recency",
    "quality": "ranking_weight_quality",
    "verification": "ranking_weight_verification",
    "domain_relevance": "ranking_weight_domain_relevance",
    "keyword_overlap": "ranking_weight_keyword_overlap",
    "feature_match": "ranking_weight_feature_match",
    "target_match": "ranking_weight_target_match",
    "task_match": "ranking_weight_task_match",
    "compatibility": "ranking_weight_compatibility",
}

#: Metrics reported for every ablation, in report order. A spread of headline
#: metrics rather than one number, because a signal can improve top-1 while
#: wrecking the tail and a single average hides that.
REPORTED_METRICS: tuple[str, ...] = (
    "precision@1",
    "precision@3",
    "recall@5",
    "mrr",
    "mrr_highly_relevant",
    "ndcg@5",
    "ndcg@10",
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
)


def _subset(report: EvaluationReport, names: Sequence[str]) -> dict[str, float]:
    return {name: report.metrics[name] for name in names if name in report.metrics}


def run_ablation(
    engine_factory: Callable[[Settings], object],
    base_settings: Settings,
    benchmark: Sequence[BenchmarkCase],
    baseline: EvaluationReport,
    signals: Optional[Sequence[str]] = None,
) -> list[AblationResult]:
    """Re-run the benchmark with each signal's weight set to zero.

    :param engine_factory: builds a fresh engine from settings. Required rather
        than reusing one engine because the engine holds its settings.
    :param baseline: the unmodified run to diff against.
    """
    targets = dict(ABLATION_TARGETS)
    selected = list(signals) if signals is not None else list(targets)

    baseline_metrics = _subset(baseline, REPORTED_METRICS)
    baseline_tops = {e.case_id: _top_id(e) for e in baseline.per_case}

    results: list[AblationResult] = []
    for signal in selected:
        field = targets.get(signal)
        if field is None:
            raise KeyError(f"unknown ablation target: {signal!r}")
        variant_settings = base_settings.model_copy(update={field: 0})
        engine = engine_factory(variant_settings)

        per_case = [evaluate_case(case, engine) for case in benchmark]
        aggregated = aggregate_metrics({e.case_id: e.metrics for e in per_case})
        variant_metrics = {name: aggregated[name] for name in REPORTED_METRICS if name in aggregated}

        rank_changes = sum(
            1 for e in per_case if _top_id(e) != baseline_tops.get(e.case_id)
        )

        results.append(
            AblationResult(
                disabled=signal,
                metrics=variant_metrics,
                deltas=metric_deltas(baseline_metrics, variant_metrics),
                rank_changes=rank_changes,
            )
        )
    return results


def _top_id(evaluation) -> Optional[str]:
    if not evaluation.ranked:
        return None
    return evaluation.ranked[0].candidate_id


def render_ablation_table(
    baseline: EvaluationReport, results: Sequence[AblationResult]
) -> str:
    """Markdown table for the report's ablation section."""
    lines = [
        "| signal removed | d(NDCG@5) | d(NDCG@10) | d(MRR) | d(P@1) | d(hit@3) | top-1 changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        d = result.deltas
        lines.append(
            f"| `{result.disabled}` | {_signed(d.get('ndcg@5'))} | {_signed(d.get('ndcg@10'))} | "
            f"{_signed(d.get('mrr'))} | {_signed(d.get('precision@1'))} | "
            f"{_signed(d.get('hit_rate@3'))} | {result.rank_changes} |"
        )
    return "\n".join(lines)


def _signed(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"