"""Sensitivity: what happens to a ranking when one thing changes?

Ablation answers "what if this signal were gone". Sensitivity answers the question
a practitioner actually asks: "what if the dataset had told me about its target?"
"what if it had 10x the downloads?" "what if it were unverified?"

Method, and why it is this method:

* Perturbation is applied to **candidate metadata**, not to the factor arithmetic.
  Re-deriving a modified factor and re-aggregating it by hand would only prove
  that arithmetic works; re-running the engine over changed input measures the
  system as it behaves.
* Every scenario re-runs the full engine on the same project. Nothing is
  short-circuited.
* A scenario that changes nothing is still reported. "Popularity tripled and the
  rank did not move" is a result worth printing, because it bounds how much that
  signal can hurt.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, Sequence

from app.core.config import Settings
from app.schemas.dataset import DatasetCandidate, SourceType, VerificationStatus
from app.schemas.project import ProjectAnalysisData
from app.services.datasets.evaluation.fixtures import compile_case
from app.services.datasets.evaluation.models import (
    BenchmarkCase,
    SensitivityScenario,
)

Scenario = tuple[str, str, Callable[[DatasetCandidate], None]]

#: Metrics reported per scenario, before and after.
REPORTED = ("score", "rank", "relevance_score")


def _triple_popularity(candidate: DatasetCandidate) -> None:
    if candidate.download_count is not None:
        candidate.download_count *= 3


def _make_it_verified(candidate: DatasetCandidate) -> None:
    candidate.verification_status = VerificationStatus.VERIFIED
    candidate.source_type = SourceType.VERIFIED_EXTERNAL


def _make_it_unverified(candidate: DatasetCandidate) -> None:
    candidate.verification_status = VerificationStatus.UNVERIFIED


def _strip_target(candidate: DatasetCandidate) -> None:
    candidate.target_description = None


def _strip_features(candidate: DatasetCandidate) -> None:
    candidate.feature_description = None


def _strip_tags(candidate: DatasetCandidate) -> None:
    candidate.tags = []
    candidate.domain = None


def _strip_task_types(candidate: DatasetCandidate) -> None:
    candidate.task_types = []


def _mark_stale(candidate: DatasetCandidate) -> None:
    if candidate.updated_at is not None:
        candidate.updated_at = candidate.updated_at.replace(year=candidate.updated_at.year - 5)


def _change_target_label(candidate: DatasetCandidate) -> None:
    candidate.target_description = "unrelated label vocabulary"


def scenarios_for(
    candidate: DatasetCandidate,
) -> list[tuple[str, str, Callable[[DatasetCandidate], None]]]:
    """The perturbation set applied to one candidate."""
    return [
        ("popularity x3", "triple the download count", _triple_popularity),
        ("evidence quality", "mark the candidate verified", _make_it_verified),
        ("verification removed", "mark the candidate unverified", _make_it_unverified),
        ("target evidence removed", "delete the published target description", _strip_target),
        ("feature evidence removed", "delete the published feature description", _strip_features),
        ("domain evidence removed", "delete domain and tags", _strip_tags),
        ("task evidence removed", "delete declared task types", _strip_task_types),
        ("recency penalty", "age the last update by five years", _mark_stale),
        ("target contradicted", "replace the target with an unrelated label", _change_target_label),
    ]


def run_sensitivity(
    engine_factory: Callable[[Settings], object],
    base_settings: Settings,
    benchmark: Sequence[BenchmarkCase],
    *,
    max_cases: int = 8,
) -> list[SensitivityScenario]:
    """Run perturbations on a bounded sample of cases.

    Bounded because this re-runs the engine once per scenario per case, and the
    point is to characterise factor influence, not to produce another benchmark
    score. Case selection is the first ``max_cases`` in corpus order, which is
    stable across runs.
    """
    engine = engine_factory(base_settings)
    selected = list(benchmark)[:max_cases]
    results: list[SensitivityScenario] = []

    for case in selected:
        baseline = compile_case(case)
        ordered_before = engine.rank(list(baseline.candidates), baseline.requirements)
        baseline_ranks = {c.id: index + 1 for index, c in enumerate(ordered_before)}
        baseline_scores = {c.id: c.ranking_score or 0.0 for c in ordered_before}

        for candidate_spec in case.candidates:
            for name, description, mutate in scenarios_for(candidate_spec):
                compiled = compile_case(case)
                target = next(
                    c for c in compiled.candidates if c.id == f"benchmark:{case.id}/{candidate_spec.key}"
                )
                before_state = target.model_dump_json()
                mutate(target)
                # Skip perturbations that do not apply to this candidate rather
                # than recording a zero delta. "Delete the published target" on a
                # candidate that never declared one changes nothing, and reporting
                # that as score_delta 0 would claim the ranker is robust to losing
                # evidence it was never given. Every scenario in the output has to
                # be a perturbation that actually happened.
                if target.model_dump_json() == before_state:
                    continue
                ordered_after = engine.rank(list(compiled.candidates), compiled.requirements)
                after_ranks = {c.id: index + 1 for index, c in enumerate(ordered_after)}
                after_scores = {c.id: c.ranking_score or 0.0 for c in ordered_after}

                cid = target.id
                if cid not in baseline_ranks:
                    continue
                rank_before = baseline_ranks[cid]
                rank_after = after_ranks.get(cid, rank_before)
                score_before = baseline_scores[cid]
                score_after = after_scores.get(cid, score_before)

                results.append(
                    SensitivityScenario(
                        name=name,
                        description=description,
                        case_id=case.id,
                        candidate_id=cid,
                        before=_snapshot(rank_before, score_before),
                        after=_snapshot(rank_after, score_after),
                        rank_before=rank_before,
                        rank_after=rank_after,
                        score_before=round(score_before, 4),
                        score_after=round(score_after, 4),
                        score_delta=round(score_after - score_before, 4),
                        rank_delta=rank_after - rank_before,
                    )
                )
    return results


def _snapshot(rank: int, score: float) -> dict[str, float]:
    return {"rank": float(rank), "score": round(score, 4)}


def summarise_sensitivity(scenarios: Sequence[SensitivityScenario]) -> dict[str, dict[str, float]]:
    """Per-scenario aggregate: mean |score delta|, mean |rank delta|, movers."""
    grouped: dict[str, list[SensitivityScenario]] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario.name, []).append(scenario)

    summary: dict[str, dict[str, float]] = {}
    for name, group in grouped.items():
        n = len(group)
        summary[name] = {
            "count": float(n),
            "mean_abs_score_delta": round(
                sum(abs(s.score_delta) for s in group) / n, 4
            ),
            "max_abs_score_delta": round(
                max(abs(s.score_delta) for s in group), 4
            ),
            "mean_abs_rank_delta": round(
                sum(abs(s.rank_delta) for s in group) / n, 4
            ),
            "moved_cases": float(sum(1 for s in group if s.rank_delta != 0)),
            "max_abs_rank_delta": float(max(abs(s.rank_delta) for s in group)),
        }
    return summary


def render_sensitivity_table(summary: dict[str, dict[str, float]]) -> str:
    """Markdown table for the report's sensitivity section."""
    lines = [
        "| perturbation | runs | mean abs score delta | max abs score delta | mean abs rank delta | runs that moved |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in summary.items():
        lines.append(
            f"| `{name}` | {int(stats['count'])} | {stats['mean_abs_score_delta']:.4f} | "
            f"{stats['max_abs_score_delta']:.4f} | {stats['mean_abs_rank_delta']:.4f} | "
            f"{int(stats['moved_cases'])} |"
        )
    return "\n".join(lines)