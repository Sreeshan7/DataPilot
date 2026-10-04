"""Does the *final recommendation* put forward the right dataset?

The ranking benchmark measures whether the best dataset is ranked first. The thing a
user actually acts on is one step later: ``FinalRecommendation.best_match``. That step
adds its own rules (an incompatible dataset is never put forward, a confirmed dataset
beats an unconfirmed one, a close call goes to the more certain dataset), and until
this module nothing measured whether those rules help or hurt.

Every figure here is measured on the same hand-graded corpus as the ranking benchmark,
which this project wrote. It says how the recommendation layer behaves on that corpus;
it is not real-world accuracy.

Outcomes of one case, in the vocabulary of the grades:

* the best match is a ``HIGHLY_RELEVANT`` dataset            -> correct
* it is ``RELEVANT``                                          -> acceptable
* it is ``WEAK`` or ``IRRELEVANT``                            -> poor
* it is ``INCOMPATIBLE``                                      -> a safety failure
* nothing is put forward and nothing in the pool was good     -> correct abstention
* nothing is put forward but a good dataset was available     -> false abstention
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.schemas.dataset import SourceType
from app.schemas.recommendation import RecommendationOutcome
from app.services.datasets.evaluation.fixtures import BENCHMARK_NOW, compile_case
from app.services.datasets.evaluation.labels import (
    HIGHLY_RELEVANT_THRESHOLD,
    RELEVANT_THRESHOLD,
    is_incompatible,
)
from app.services.datasets.evaluation.models import BenchmarkCase
from app.services.datasets.intelligence import build_intelligence_report
from app.services.datasets.ranking import DatasetRankingEngine
from app.services.datasets.recommendation import build_recommendations


class CaseVerdict(str):
    CORRECT = "correct"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNSAFE = "unsafe"
    CORRECT_ABSTENTION = "correct_abstention"
    FALSE_ABSTENTION = "false_abstention"


class RecommendationCaseResult(BaseModel):
    case_id: str
    category: str
    outcome: RecommendationOutcome
    best_name: Optional[str] = None
    best_grade: Optional[int] = None
    best_confidence: Optional[str] = None
    pool_best_grade: int
    verdict: str
    close_call: bool = False
    ranked_first_grade: Optional[int] = None


class RecommendationReport(BaseModel):
    cases: list[RecommendationCaseResult] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    safety_violations: list[str] = Field(default_factory=list)


def _verdict(
    outcome: RecommendationOutcome, best_grade: Optional[int], pool_best: int
) -> str:
    if outcome is RecommendationOutcome.NONE or best_grade is None:
        return (
            CaseVerdict.FALSE_ABSTENTION
            if pool_best >= RELEVANT_THRESHOLD
            else CaseVerdict.CORRECT_ABSTENTION
        )
    if is_incompatible(best_grade):
        return CaseVerdict.UNSAFE
    if best_grade >= HIGHLY_RELEVANT_THRESHOLD:
        return CaseVerdict.CORRECT
    if best_grade >= RELEVANT_THRESHOLD:
        return CaseVerdict.ACCEPTABLE
    return CaseVerdict.POOR


def evaluate_recommendations(
    cases: Sequence[BenchmarkCase], settings: Settings
) -> RecommendationReport:
    """Run ranking -> intelligence -> final recommendation on every case."""
    engine = DatasetRankingEngine(settings=settings, now=BENCHMARK_NOW)
    results: list[RecommendationCaseResult] = []
    violations: list[str] = []

    for case in cases:
        compiled = compile_case(case)
        ranked = engine.rank(list(compiled.candidates), compiled.requirements)
        intelligence = build_intelligence_report(ranked, compiled.requirements)
        _, final = build_recommendations(ranked, intelligence)

        grades = compiled.grades_by_id
        pool_best = max(grades.values()) if grades else 0
        best = final.best_match
        best_grade = grades.get(best.dataset_id) if best else None

        # Safety rules the schema promises; checked against the graded truth too.
        if best and best.source_type is SourceType.AI_SUGGESTED:
            if final.outcome is RecommendationOutcome.RECOMMENDED:
                violations.append(f"{case.id}: an AI suggestion was RECOMMENDED")
        if best_grade is not None and is_incompatible(best_grade):
            violations.append(f"{case.id}: best match {best.name!r} is graded INCOMPATIBLE")

        results.append(
            RecommendationCaseResult(
                case_id=case.id,
                category=case.category,
                outcome=final.outcome,
                best_name=best.name if best else None,
                best_grade=best_grade,
                best_confidence=best.confidence.level.value if best else None,
                pool_best_grade=pool_best,
                verdict=_verdict(final.outcome, best_grade, pool_best),
                close_call=final.close_call,
                ranked_first_grade=grades.get(ranked[0].id) if ranked else None,
            )
        )

    return RecommendationReport(
        cases=results,
        metrics=_metrics(results),
        counts=dict(sorted(Counter(r.verdict for r in results).items())),
        safety_violations=violations,
    )


def _metrics(results: Sequence[RecommendationCaseResult]) -> dict[str, float]:
    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    answerable = [r for r in results if r.pool_best_grade >= HIGHLY_RELEVANT_THRESHOLD]
    usable = [r for r in results if r.pool_best_grade >= RELEVANT_THRESHOLD]
    unanswerable = [r for r in results if r.pool_best_grade < RELEVANT_THRESHOLD]
    put_forward = [r for r in results if r.best_grade is not None]

    correct = sum(1 for r in answerable if (r.best_grade or -9) >= HIGHLY_RELEVANT_THRESHOLD)
    acceptable = sum(1 for r in usable if (r.best_grade or -9) >= RELEVANT_THRESHOLD)
    ranked_first_correct = sum(
        1 for r in answerable if (r.ranked_first_grade or -9) >= HIGHLY_RELEVANT_THRESHOLD
    )
    return {
        "cases": float(len(results)),
        "best_match_correct_rate": rate(correct, len(answerable)),
        "best_match_acceptable_rate": rate(acceptable, len(usable)),
        "ranked_first_correct_rate": rate(ranked_first_correct, len(answerable)),
        "incompatible_best_rate": rate(
            sum(1 for r in put_forward if is_incompatible(r.best_grade or 0)), len(put_forward)
        ),
        "false_abstention_rate": rate(
            sum(1 for r in usable if r.verdict == CaseVerdict.FALSE_ABSTENTION), len(usable)
        ),
        "correct_abstention_rate": rate(
            sum(1 for r in unanswerable if r.verdict == CaseVerdict.CORRECT_ABSTENTION),
            len(unanswerable),
        ),
        "recommended_ai_suggestions": float(
            sum(
                1
                for r in results
                if r.outcome is RecommendationOutcome.RECOMMENDED and r.best_grade is None
            )
        ),
    }


@dataclass
class Comparison:
    """How the recommendation layer changes the answer relative to rank 1."""

    rescued: list[str] = field(default_factory=list)
    harmed: list[str] = field(default_factory=list)


def compare_with_rank_one(report: RecommendationReport) -> Comparison:
    """Cases where the layer picked something other than rank 1, and whether it helped."""
    out = Comparison()
    for r in report.cases:
        if r.best_grade is None or r.ranked_first_grade is None:
            continue
        if r.best_grade > r.ranked_first_grade:
            out.rescued.append(r.case_id)
        elif r.best_grade < r.ranked_first_grade:
            out.harmed.append(r.case_id)
    return out
