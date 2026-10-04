"""Executable ranking invariants.

These are the properties that must hold for *every* ranking the system produces,
not for one scenario. Each returns a list of violations rather than asserting, so
the same code serves three callers: a test that requires the list to be empty, the
benchmark runner that reports them as findings, and the report generator that prints
them.

They are written as pure functions over ``(candidates, requirements, ranked)``
tuples. That keeps them free of engine internals: if an invariant depends on how
the ranker works rather than on what it must produce, it does not belong here.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from app.schemas.dataset import DatasetCandidate, MatchStatus, RankingFactorName
from app.schemas.project import ProjectAnalysisData
from app.services.datasets.evaluation.labels import (
    HIGHLY_RELEVANT_THRESHOLD,
    is_incompatible,
    is_relevant,
)

SUM_TOLERANCE = 0.05


def _violations(condition: bool, message: str) -> list[str]:
    return [] if condition else [message]


# 1 -------------------------------------------------------------------------


def property_within_weight_bounds(ranked: Sequence[DatasetCandidate]) -> list[str]:
    """No final score may leave 0.0-100.0."""
    out: list[str] = []
    for candidate in ranked:
        score = candidate.ranking_score
        if score is None or not (0.0 <= score <= 100.0):
            out.append(f"{candidate.id}: ranking_score {score} outside 0.0-100.0")
    return out


# 2 -------------------------------------------------------------------------


def property_contributions_sum_to_score(ranked: Sequence[DatasetCandidate]) -> list[str]:
    """Every displayed explanation must add up to the score it explains."""
    out: list[str] = []
    for candidate in ranked:
        total = sum(f.contribution for f in candidate.ranking_factors)
        score = candidate.ranking_score or 0.0
        if abs(total - score) > SUM_TOLERANCE:
            out.append(
                f"{candidate.id}: contributions {total:.2f} != score {score:.2f}"
            )
    return out


# 3 -------------------------------------------------------------------------


def property_no_factor_duplication(ranked: Sequence[DatasetCandidate]) -> list[str]:
    """Each factor appears at most once, or its contribution is double-counted."""
    out: list[str] = []
    for candidate in ranked:
        seen: set[str] = set()
        for factor in candidate.ranking_factors:
            if factor.name.value in seen:
                out.append(f"{candidate.id}: factor {factor.name.value} repeated")
            seen.add(factor.name.value)
    return out


# 4 -------------------------------------------------------------------------


def property_requirement_factors_within_bounds(ranked: Sequence[DatasetCandidate]) -> list[str]:
    """A judged requirement factor must lie in 0.0-1.0; an unjudged one must be None.

    The second half is the invariant that makes "UNKNOWN is not mismatch"
    structural rather than aspirational: an unjudgeable factor is represented as
    ``None``, which the aggregation excludes, rather than as a 0.0 that looks like
    evidence of absence.
    """
    out: list[str] = []
    requirement_names = {
        RankingFactorName.DOMAIN_RELEVANCE,
        RankingFactorName.KEYWORD_OVERLAP,
        RankingFactorName.FEATURE_MATCH,
        RankingFactorName.TARGET_MATCH,
        RankingFactorName.TASK_MATCH,
        RankingFactorName.COMPATIBILITY,
    }
    for candidate in ranked:
        for factor in candidate.ranking_factors:
            if factor.name not in requirement_names:
                continue
            if factor.score is None:
                if factor.contribution != 0.0:
                    out.append(
                        f"{candidate.id}:{factor.name.value}: unjudged but contributed "
                        f"{factor.contribution}"
                    )
                continue
            if not (0.0 <= factor.score <= 1.0):
                out.append(
                    f"{candidate.id}:{factor.name.value}: score {factor.score} outside 0.0-1.0"
                )
    return out


# 5 -------------------------------------------------------------------------


def property_publisher_factors_absent_when_unpublished(
    ranked: Sequence[DatasetCandidate],
) -> list[str]:
    """A dataset the registry never gave a usability rating or download count to is
    not penalised for it."""
    out: list[str] = []
    for candidate in ranked:
        for name in (RankingFactorName.QUALITY, RankingFactorName.POPULARITY, RankingFactorName.RECENCY):
            factor = next(
                (f for f in candidate.ranking_factors if f.name is name), None
            )
            if factor is None:
                continue
            if candidate.download_count is None and name is RankingFactorName.POPULARITY:
                if factor.score is not None and factor.score > 0.0:
                    out.append(
                        f"{candidate.id}: popularity scored {factor.score} with no "
                        "published download count"
                    )
            if candidate.usability is None and name is RankingFactorName.QUALITY:
                if factor.score is not None and factor.score > 0.0:
                    out.append(
                        f"{candidate.id}: quality scored {factor.score} with no published "
                        "usability rating"
                    )
            if candidate.updated_at is None and name is RankingFactorName.RECENCY:
                if factor.score is not None and factor.score > 0.0:
                    out.append(
                        f"{candidate.id}: recency scored {factor.score} with no published "
                        "update date"
                    )
    return out


# 6 -------------------------------------------------------------------------


def property_unknown_is_not_mismatch(ranked: Sequence[DatasetCandidate]) -> list[str]:
    """No requirement may be recorded as MISMATCH while it was also UNKNOWN.

    This is a check on the requirement records themselves rather than on the factor
    score, because the two can disagree in exactly the way this project cares
    about: a factor can average to 0.0 across several requirements while every
    individual requirement is honestly UNKNOWN.
    """
    out: list[str] = []
    for candidate in ranked:
        for factor in candidate.ranking_factors:
            for requirement in factor.requirements:
                if requirement.status is MatchStatus.UNKNOWN and requirement.detail:
                    if "mismatch" in requirement.detail.lower():
                        out.append(
                            f"{candidate.id}:{factor.name.value}: UNKNOWN requirement "
                            f"described as a mismatch: {requirement.detail!r}"
                        )
    return out


# 7 -------------------------------------------------------------------------


def property_incompatible_not_above_relevant(
    ranked: Sequence[DatasetCandidate], grades: dict[str, int]
) -> list[str]:
    """A grade -1 dataset never ranks above a grade >= 2 dataset.

    This is the single most important ordering property: a result set that puts an
    unusable dataset first is worse than one that returns nothing.
    """
    out: list[str] = []
    for upper in range(len(ranked)):
        for lower in range(upper + 1, len(ranked)):
            above, below = ranked[upper], ranked[lower]
            above_grade = grades.get(above.id)
            below_grade = grades.get(below.id)
            if above_grade is None or below_grade is None:
                continue
            if is_incompatible(above_grade) and is_relevant(below_grade):
                out.append(
                    f"incompatible '{above.name}' (grade {above_grade}) at rank "
                    f"{upper + 1} outranks relevant '{below.name}' (grade {below_grade}) "
                    f"at rank {lower + 1}"
                )
    return out


# 8 -------------------------------------------------------------------------


def property_determinism(
    candidates: Sequence[DatasetCandidate],
    requirements: ProjectAnalysisData,
    rank: Callable,
    runs: int = 3,
) -> list[str]:
    """The same input in the same order produces byte-identical rankings."""
    out: list[str] = []
    reference: Optional[list[str]] = None
    reference_scores: Optional[list[Optional[float]]] = None
    for attempt in range(runs):
        ordered = rank(list(candidates), requirements)
        fingerprint = [c.id for c in ordered]
        scores = [c.ranking_score for c in ordered]
        if reference is None:
            reference = fingerprint
            reference_scores = scores
        elif fingerprint != reference or scores != reference_scores:
            out.append(
                f"run {attempt + 1} differs from run 1: {fingerprint} vs {reference}"
            )
    return out


# 9 -------------------------------------------------------------------------


def property_input_order_independence(
    candidates: Sequence[DatasetCandidate],
    requirements: ProjectAnalysisData,
    rank: Callable,
) -> list[str]:
    """Shuffling the input pool must not change the ranking's content.

    Tie-breaking is allowed to prefer the original order for candidates that are
    genuinely indistinguishable, so this compares the score profile of each
    candidate, not the raw order.
    """
    forward = rank(list(candidates), requirements)
    backward = rank(list(reversed(list(candidates))), requirements)
    forward_profile = sorted((c.id, c.ranking_score) for c in forward)
    backward_profile = sorted((c.id, c.ranking_score) for c in backward)
    out: list[str] = []
    if forward_profile != backward_profile:
        out.append("reversing the input pool changed the score profile")
    return out


# 10 ------------------------------------------------------------------------


def property_unverified_not_penalised_for_missing_evidence(
    ranked: Sequence[DatasetCandidate],
) -> list[str]:
    """An unverified candidate is never marked as contradicting a requirement.

    Contradiction detection is evidence-based, and an inferred field is not
    evidence. Penalising an AI suggestion for contradicting a requirement using the
    same unverified authority that was used to satisfy one would be incoherent.
    """
    from app.schemas.dataset import VerificationStatus

    out: list[str] = []
    for candidate in ranked:
        if candidate.verification_status is not VerificationStatus.VERIFIED:
            for factor in candidate.ranking_factors:
                if factor.contradictions:
                    out.append(
                        f"{candidate.id}: unverified candidate carries contradictions "
                        f"on {factor.name.value}: {factor.contradictions}"
                    )
    return out


def property_grade_three_present_when_retrievable(
    ranked: Sequence[DatasetCandidate], grades: dict[str, int]
) -> list[str]:
    """If a grade-3 dataset was in the pool, it must be ranked somewhere."""
    out: list[str] = []
    has_three = any(grade >= HIGHLY_RELEVANT_THRESHOLD for grade in grades.values())
    if not has_three:
        return out
    ranked_ids = {c.id for c in ranked}
    missing = [
        cid for cid, grade in grades.items()
        if grade >= HIGHLY_RELEVANT_THRESHOLD and cid not in ranked_ids
    ]
    for cid in missing:
        out.append(f"grade-3 candidate {cid} was dropped from the ranking entirely")
    return out


ALL_PROPERTIES = (
    property_within_weight_bounds,
    property_contributions_sum_to_score,
    property_no_factor_duplication,
    property_requirement_factors_within_bounds,
    property_publisher_factors_absent_when_unpublished,
    property_unknown_is_not_mismatch,
    property_input_order_independence,
    property_unverified_not_penalised_for_missing_evidence,
)

__all__ = [name for name in dir() if name.startswith(("property_", "ALL_"))]