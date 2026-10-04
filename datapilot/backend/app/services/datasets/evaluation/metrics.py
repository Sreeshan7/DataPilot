"""Ranking metrics over graded relevance.

Pure functions on a grade sequence: no engine, no schemas, no fixtures. That is
deliberate - a metric that can only be computed by running the whole pipeline
cannot be unit-tested against hand-computable examples, and the mission's first
requirement is that these numbers mean exactly what their names say.

Convention choices, all stated here because metrics are only comparable when
their conventions are:

* **Relevance cut.** Precision and recall count a candidate as relevant at grade
  ``>= 2``. Grade 1 is a "maybe after transformation" signal and belongs in NDCG,
  where graded gains are the point, not in the binary cut.
* **Recall denominator.** Recall is over *all* relevant candidates in the
  graded pool, not over the pool's top-k. A ranker that returns three of ten
  relevant datasets scores 0.3, which is the number a reader expects.
* **Gains.** ``2**max(grade, 0) - 1``. Grade -1 therefore contributes zero gain
  rather than negative gain, so an actively incompatible dataset is "worth
  nothing" instead of "worth less than putting nothing there", which would let a
  single wrong result push a score below a shorter correct ranking.
* **Short lists.** A list shorter than k is padded with non-relevant entries, so
  P@5 of a two-item result is 0.4 and not 1.0.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from app.services.datasets.evaluation.labels import (
    HIGHLY_RELEVANT_THRESHOLD,
    RELEVANT_THRESHOLD,
)

#: Metrics computed for every case at every cut.
DEFAULT_CUTS: tuple[int, ...] = (1, 3, 5, 10)


def _ranked_grades_at(grades: Sequence[int], k: int) -> list[int]:
    """The top-k slice, conceptually padded with non-relevant entries."""
    return [grades[i] if i < len(grades) else 0 for i in range(k)]


def precision_at_k(
    grades: Sequence[int], k: int, threshold: int = RELEVANT_THRESHOLD
) -> float:
    """Share of the top k positions holding a relevant candidate."""
    if k <= 0:
        return 0.0
    window = _ranked_grades_at(grades, k)
    hits = sum(1 for g in window if g >= threshold)
    return hits / k


def recall_at_k(
    grades: Sequence[int],
    total_relevant: int,
    k: int,
    threshold: int = RELEVANT_THRESHOLD,
) -> float:
    """Share of the relevant pool found within the top k positions."""
    if total_relevant <= 0:
        return 0.0
    window = _ranked_grades_at(grades, k)
    hits = sum(1 for g in window if g >= threshold)
    return hits / total_relevant


def reciprocal_rank(
    grades: Sequence[int], threshold: int = RELEVANT_THRESHOLD
) -> float:
    """1 / rank of the first relevant candidate, 0.0 if none is in the list."""
    for index, grade in enumerate(grades, start=1):
        if grade >= threshold:
            return 1.0 / index
    return 0.0


def highly_relevant_reciprocal_rank(grades: Sequence[int]) -> float:
    """Reciprocal rank of the first *grade-3* candidate.

    Reported next to MRR because a ranker that puts a grade-2 dataset first has a
    perfect MRR and a useless ranking; separating the two is the only way that
    difference shows up in a summary number.
    """
    for index, grade in enumerate(grades, start=1):
        if grade >= HIGHLY_RELEVANT_THRESHOLD:
            return 1.0 / index
    return 0.0


def _gain(grade: int) -> float:
    return (2.0 ** max(grade, 0)) - 1.0


def dcg(grades: Sequence[int]) -> float:
    """Discounted cumulative gain of one ordering."""
    return sum(_gain(g) / math.log2(index + 1) for index, g in enumerate(grades, start=1))


def ndcg_at_k(grades: Sequence[int], k: int) -> float:
    """Normalised DCG against the best possible ordering of the same pool."""
    window = _ranked_grades_at(grades, k)
    ideal = sorted((g for g in grades), reverse=True)[:k]
    if not ideal:
        return 0.0
    best = dcg(ideal)
    if best <= 0.0:
        return 0.0
    return dcg(window) / best


def top_k_hit_rate(
    grades: Sequence[int], k: int, threshold: int = HIGHLY_RELEVANT_THRESHOLD
) -> float:
    """1.0 when at least one candidate at or above ``threshold`` is in the top k."""
    return 1.0 if any(g >= threshold for g in _ranked_grades_at(grades, k)) else 0.0


def case_metrics(
    ranked_grades: Sequence[int], pool_grades: Sequence[int]
) -> dict[str, float]:
    """Every per-case metric in one dict.

    ``ranked_grades`` is the engine's output order; ``pool_grades`` is every
    graded candidate in the pool, which is the recall denominator.
    """
    relevant_in_pool = sum(1 for g in pool_grades if g >= RELEVANT_THRESHOLD)
    metrics: dict[str, float] = {}
    for k in DEFAULT_CUTS:
        metrics[f"precision@{k}"] = precision_at_k(ranked_grades, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked_grades, relevant_in_pool, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_grades, k)
    metrics["mrr"] = reciprocal_rank(ranked_grades)
    metrics["mrr_highly_relevant"] = highly_relevant_reciprocal_rank(ranked_grades)
    for k in DEFAULT_CUTS:
        metrics[f"hit_rate@{k}"] = top_k_hit_rate(ranked_grades, k)
    metrics["pool_size"] = float(len(pool_grades))
    metrics["relevant_in_pool"] = float(relevant_in_pool)
    return metrics


def aggregate_metrics(
    per_case: Mapping[str, Sequence[float]]
) -> dict[str, float]:
    """Macro-average each metric across cases.

    Macro rather than micro on purpose: one case with a 60-candidate pool must
    not outvote nine small ones.
    """
    if not per_case:
        return {}
    names = sorted({name for values in per_case.values() for name in values})
    out: dict[str, float] = {}
    for name in names:
        values = [values[name] for values in per_case.values() if name in values]
        out[name] = sum(values) / len(values) if values else 0.0
    return out


def incompatible_top_rate(
    ranked_grades: Sequence[int], k: int = 3
) -> float:
    """Share of top-k positions holding a grade -1 (clearly unusable) dataset."""
    window = _ranked_grades_at(grades=ranked_grades, k=k)
    if not window:
        return 0.0
    return sum(1 for g in window if g <= -1) / len(window)


def worst_rank_of(grades: Sequence[int], predicate) -> Optional[int]:  # type: ignore[type-arg]
    """1-based rank of the first candidate matching ``predicate``, if any."""
    for index, grade in enumerate(grades, start=1):
        if predicate(grade):
            return index
    return None


def metric_deltas(
    baseline: Mapping[str, float], variant: Mapping[str, float]
) -> dict[str, float]:
    """Signed change of every shared metric, variant minus baseline."""
    return {
        name: variant[name] - baseline[name]
        for name in sorted(set(baseline) & set(variant))
        if not name.startswith("pool_")
    }


def iter_cut_values(cuts: Iterable[int] = DEFAULT_CUTS) -> list[int]:
    return sorted(set(int(c) for c in cuts))