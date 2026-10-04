"""Per-case diagnostics: why each candidate got the rank it got.

A metric says *that* a case failed. This module exists to say *why*, down to the
factor that contributed the points, because a failure report that stops at
"ndcg@5 = 0.0" is not actionable.

Three rules shape the output:

* **Same arithmetic as the ranker.** Every number here is read back from the
  factor the ranker actually computed. Nothing is re-derived, so a diagnostic can
  never disagree with the score it explains.
* **Evidence, not conclusions.** The report states what evidence existed and what
  verdict followed. It does not assert that a dataset is good.
* **UNKNOWN is visible.** A candidate whose ranking rests on nothing is reported
  as unknown-driven rather than silently averaged away.
"""

from __future__ import annotations

from typing import Optional, Sequence

from app.schemas.dataset import DatasetCandidate, RankingFactorName
from app.services.datasets.evaluation.labels import RelevanceGrade
from app.services.datasets.evaluation.models import (
    CompiledCase,
    FactorContribution,
    RankedItem,
)

#: Factors whose evidence may be absent. A project-specific signal in this set is
#: the difference between "absent" and "unknown", and the difference between a
#: metric that measures relevance and one that measures metadata volume.
_EVIDENCE_SENSITIVE = (
    RankingFactorName.FEATURE_MATCH,
    RankingFactorName.TARGET_MATCH,
    RankingFactorName.TASK_MATCH,
)

#: Publisher signals: never evidence of relevance, only of adoption.
_PUBLISHER_FACTORS = (
    RankingFactorName.POPULARITY,
    RankingFactorName.RECENCY,
    RankingFactorName.QUALITY,
    RankingFactorName.VERIFICATION,
)


def build_factor_views(
    candidate: DatasetCandidate,
) -> list[FactorContribution]:
    """Flatten a scored candidate's factors into the diagnostic view."""
    views: list[FactorContribution] = []
    for factor in candidate.ranking_factors:
        views.append(
            FactorContribution(
                name=factor.name.value,
                weight=factor.weight,
                score=factor.score,
                contribution=factor.contribution,
                detail=factor.detail,
                matched_terms=list(factor.matched_terms),
                unmatched_terms=list(factor.unmatched_terms),
                unknown_terms=list(factor.unknown_terms),
                contradictions=list(factor.contradictions or []),
            )
        )
    # Stable order by contribution, descending, so the report leads with what
    # actually decided the rank.
    views.sort(key=lambda v: (-v.contribution, v.name))
    return views


def build_ranked_items(
    ranked: Sequence[DatasetCandidate],
    grades_by_id: dict[str, int],
) -> list[RankedItem]:
    """Attach grades and factor views to the engine's output order."""
    items: list[RankedItem] = []
    for rank, candidate in enumerate(ranked, start=1):
        items.append(
            RankedItem(
                rank=rank,
                candidate_id=candidate.id,
                name=candidate.name,
                grade=grades_by_id.get(candidate.id, 0),
                score=candidate.ranking_score or 0.0,
                relevance_score=candidate.relevance_score,
                quality_score=candidate.quality_score,
                source_type=candidate.source_type,
                verification_status=candidate.verification_status,
                factors=build_factor_views(candidate),
                reasons=list(candidate.ranking_reasons or []),
            )
        )
    return items


def best_available(ranked: Sequence[RankedItem]) -> Optional[RankedItem]:
    """The highest-graded candidate actually present in the ranked list."""
    if not ranked:
        return None
    return max(
        ranked,
        key=lambda item: (item.grade, -item.rank),
    )


def suspected_unknown_as_mismatch(ranked: Sequence[RankedItem]) -> int:
    """Count requirement verdicts that read as MISMATCH without evidence.

    A factor score of 0.0 with ``unknown_terms`` non-empty is the fingerprint of
    "we could not tell, so we scored zero". That is only a *suspicion* - the
    ranker may legitimately reach 0.0 for other reasons - but a large count
    across the benchmark is exactly the "UNKNOWN is not mismatch" failure the
    evaluation is meant to catch, and it is not visible anywhere else.
    """
    suspicious = 0
    for item in ranked:
        for factor in item.factors:
            if factor.name not in {f.value for f in _EVIDENCE_SENSITIVE}:
                continue
            if factor.score == 0.0 and factor.unknown_terms:
                suspicious += 1
    return suspicious


def evidence_quality(item: RankedItem) -> dict[str, float]:
    """How much of the candidate's score rests on declared vs assumed evidence.

    Reported per candidate because "the top hit is mostly UNKNOWN" is a different
    and more actionable statement than "the top hit scored 71".
    """
    evidence_driven = 0.0
    unknown_driven = 0.0
    publisher_driven = 0.0
    publisher_names = {f.value for f in _PUBLISHER_FACTORS}
    for factor in item.factors:
        if factor.name in publisher_names:
            publisher_driven += factor.contribution
        elif factor.score is None or factor.unknown_terms:
            unknown_driven += factor.contribution
        else:
            evidence_driven += factor.contribution
    total = evidence_driven + unknown_driven + publisher_driven
    if total <= 0:
        return {"evidence": 0.0, "unknown": 0.0, "publisher": 0.0}
    return {
        "evidence": evidence_driven / total,
        "unknown": unknown_driven / total,
        "publisher": publisher_driven / total,
    }


def render_case_report(case: CompiledCase, items: Sequence[RankedItem]) -> str:
    """A human-readable per-case diagnostic block for the Markdown report."""
    lines: list[str] = []
    lines.append(f"### `{case.case.id}` - {case.case.description}")
    lines.append("")
    lines.append(
        f"- category: `{case.case.category}` / difficulty: `{case.case.difficulty.value}`"
    )
    lines.append(f"- pooled candidates: {len(case.candidates)}")
    if case.case.notes:
        lines.append(f"- note: {case.case.notes}")
    lines.append("")
    lines.append("| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | --- |")
    for item in items:
        mix = evidence_quality(item)
        lines.append(
            f"| {item.rank} | {item.name} | {item.grade} | {item.score:.2f} | "
            f"{_fmt(item.relevance_score)} | {_fmt(item.quality_score)} | "
            f"{mix['evidence']:.2f}/{mix['unknown']:.2f}/{mix['publisher']:.2f} |"
        )
    lines.append("")
    for item in items:
        lines.append(f"**{item.rank}. {item.name}** (grade {item.grade})")
        parts = [
            f"`{factor.name}` {factor.display} w={factor.weight:g} -> {factor.contribution:+.2f}"
            for factor in item.factors
        ]
        lines.append(f"- factors: {'; '.join(parts)}")
        if item.reasons:
            lines.append(f"- reasons: {'; '.join(item.reasons)}")
        if item.verification_status.value != "verified":
            lines.append(
                "- evidence note: unverified, so declared-field rules were not applied"
            )
    lines.append("")
    return "\n".join(lines)


def render_factor_table(ranked: Sequence[RankedItem], limit: int = 3) -> str:
    """A compact top-N factor table, used by the report's failure sections."""
    lines = [
        "| candidate | grade | factor | score | weight | contribution |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for item in ranked[:limit]:
        for factor in item.factors:
            score = "unknown" if factor.score is None else f"{factor.score:.2f}"
            lines.append(
                f"| {item.name} | {item.grade} | `{factor.name}` | {score} | "
                f"{factor.weight:g} | {factor.contribution:+.2f} |"
            )
    return "\n".join(lines)


def grade_band(grade: int) -> str:
    """Human label for a grade, used in report prose."""
    try:
        level = RelevanceGrade(grade)
    except ValueError:
        # Report prose must survive a malformed label. Raising here would abort a
        # whole run over one bad grade, and silently calling it "irrelevant" would
        # quietly change what the report counts as a miss - so say "invalid" and
        # let the corpus invariant test be the thing that complains.
        return "invalid"
    return {
        RelevanceGrade.HIGHLY_RELEVANT: "highly relevant",
        RelevanceGrade.RELEVANT: "relevant",
        RelevanceGrade.WEAK: "weak",
        RelevanceGrade.IRRELEVANT: "irrelevant",
        RelevanceGrade.INCOMPATIBLE: "incompatible",
    }[level]


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"