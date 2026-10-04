"""Failure taxonomy: classify *why* a benchmark case underperformed.

A ranked list is wrong in a small number of recognisable ways, and each way has a
different owner. "Ranking is bad" sends the next person nowhere; "a grade-2
dataset outranked the grade-3 dataset because its 1.4M downloads outscored the
target match" points at a specific factor.

Classification is deterministic and rule-based. It is a diagnostic, not a claim:
every rule states the condition it fired on, and the report shows the condition so
a reviewer can disagree with the label.
"""

from __future__ import annotations

from typing import Optional, Sequence

from app.schemas.dataset import RankingFactorName, VerificationStatus
from app.services.datasets.evaluation.labels import (
    HIGHLY_RELEVANT_THRESHOLD,
    RELEVANT_THRESHOLD,
)
from app.services.datasets.evaluation.models import (
    CaseFailure,
    CompiledCase,
    FailureCategory,
    RankedItem,
)

_EVIDENCE_SENSITIVE = {
    RankingFactorName.FEATURE_MATCH.value,
    RankingFactorName.TARGET_MATCH.value,
    RankingFactorName.TASK_MATCH.value,
}


def _score_of(item: RankedItem, name: str) -> Optional[float]:
    factor = item.factor(name)
    return None if factor is None else factor.score


def _contribution_of(item: RankedItem, name: str) -> float:
    factor = item.factor(name)
    return 0.0 if factor is None else factor.contribution


def _best(ranked: Sequence[RankedItem], minimum: int) -> Optional[RankedItem]:
    """Highest-graded item in the list, or None if nothing clears ``minimum``."""
    pool = [item for item in ranked if item.grade >= minimum]
    if not pool:
        return None
    return min(pool, key=lambda item: item.rank)


def _worst_preferred(ranked: Sequence[RankedItem], ceiling: int) -> Optional[RankedItem]:
    """Lowest-graded item in the list, or None if nothing is at or below ``ceiling``."""
    pool = [item for item in ranked if item.grade <= ceiling]
    if not pool:
        return None
    return min(pool, key=lambda item: item.rank)


def classify_failures(
    case: CompiledCase, ranked: Sequence[RankedItem]
) -> list[CaseFailure]:
    """Every failure condition detected for one case."""
    failures: list[CaseFailure] = []
    if not ranked:
        return [CaseFailure(category=FailureCategory.METADATA_INSUFFICIENT, detail="no candidates ranked")]

    # A pool that never contained the best available answer is a retrieval
    # problem. Reporting it as a ranking failure would send the fix to the
    # ranker, where it cannot help.
    pool_best = max(item.grade for item in ranked)
    if pool_best < HIGHLY_RELEVANT_THRESHOLD:
        failures.append(
            CaseFailure(
                category=FailureCategory.QUERY_RECALL_FAILURE,
                detail=(
                    f"pool contained no grade-{HIGHLY_RELEVANT_THRESHOLD} dataset "
                    f"(best in pool: grade {pool_best})"
                ),
            )
        )

    best = _best(ranked, RELEVANT_THRESHOLD)
    worst = _worst_preferred(ranked, 0)
    top = ranked[0]

    # Inversion: a usable dataset ranked below an unusable one.
    if best is not None and worst is not None and worst.rank < best.rank:
        failures.append(
            CaseFailure(
                category=FailureCategory.INVERSION,
                detail=(
                    f"grade {worst.grade} '{worst.name}' (rank {worst.rank}) outranked "
                    f"grade {best.grade} '{best.name}' (rank {best.rank})"
                ),
                candidate_ids=[worst.candidate_id, best.candidate_id],
            )
        )

    # Publisher domination: an inversion where a publisher signal is the larger
    # contribution gap between the two.
    if best is not None and worst is not None and worst.rank < best.rank:
        for publisher in ("popularity", "recency"):
            gap = _contribution_of(worst, publisher) - _contribution_of(best, publisher)
            if gap > 0:
                category = (
                    FailureCategory.POPULARITY_DOMINATION
                    if publisher == "popularity"
                    else FailureCategory.RECENCY_DOMINATION
                )
                failures.append(
                    CaseFailure(
                        category=category,
                        detail=(
                            f"'{worst.name}' gained +{gap:.2f} more {publisher} than "
                            f"'{best.name}' and outranked it"
                        ),
                        candidate_ids=[worst.candidate_id, best.candidate_id],
                    )
                )

    # Dimension mismatches on the best available answer. Each rule asks: does the
    # ranker have evidence that this dataset is about something else?
    if best is not None:
        target = _score_of(best, RankingFactorName.TARGET_MATCH.value)
        if target is not None and target <= 0.2 and best.grade >= RELEVANT_THRESHOLD:
            failures.append(
                CaseFailure(
                    category=FailureCategory.WRONG_TARGET,
                    detail=f"target match score {target:.2f} on grade {best.grade} '{best.name}'",
                    candidate_ids=[best.candidate_id],
                )
            )
        elif target is None and best.grade >= RELEVANT_THRESHOLD:
            failures.append(
                CaseFailure(
                    category=FailureCategory.MISSING_TARGET_EVIDENCE,
                    detail=f"no target evidence at all for grade {best.grade} '{best.name}'",
                    candidate_ids=[best.candidate_id],
                )
            )

        feature = _score_of(best, RankingFactorName.FEATURE_MATCH.value)
        if feature is not None and feature <= 0.2 and best.grade >= RELEVANT_THRESHOLD:
            failures.append(
                CaseFailure(
                    category=FailureCategory.FEATURE_MISMATCH,
                    detail=f"feature match score {feature:.2f} on grade {best.grade} '{best.name}'",
                    candidate_ids=[best.candidate_id],
                )
            )
        elif feature is None and best.grade >= RELEVANT_THRESHOLD:
            failures.append(
                CaseFailure(
                    category=FailureCategory.FEATURE_UNKNOWN,
                    detail=f"no feature evidence at all for grade {best.grade} '{best.name}'",
                    candidate_ids=[best.candidate_id],
                )
            )

    # Task mismatch: best answer declared a different task.
    if best is not None:
        task = _score_of(best, RankingFactorName.TASK_MATCH.value)
        if task is not None and task <= 0.2 and best.grade >= RELEVANT_THRESHOLD:
            failures.append(
                CaseFailure(
                    category=FailureCategory.WRONG_TASK,
                    detail=f"task match score {task:.2f} on grade {best.grade} '{best.name}'",
                    candidate_ids=[best.candidate_id],
                )
            )

    # Domain mismatch on the top hit.
    domain = _score_of(top, RankingFactorName.DOMAIN_RELEVANCE.value)
    if domain is not None and domain <= 0.2 and top.grade >= RELEVANT_THRESHOLD:
        failures.append(
            CaseFailure(
                category=FailureCategory.WRONG_DOMAIN,
                detail=f"domain relevance {domain:.2f} on top hit '{top.name}'",
                candidate_ids=[top.candidate_id],
            )
        )

    # Generic-term domination: the top hit won on keyword overlap while its
    # project-specific evidence factors are absent or near-zero.
    if (
        top.grade <= RelevanceBand.IRRELEVANT
        and (_score_of(top, RankingFactorName.FEATURE_MATCH.value) or 0.0) <= 0.2
        and (_score_of(top, RankingFactorName.TARGET_MATCH.value) or 0.0) <= 0.2
        and (_score_of(top, RankingFactorName.KEYWORD_OVERLAP.value) or 0.0) >= 0.4
    ):
        failures.append(
            CaseFailure(
                category=FailureCategory.GENERIC_TERM_DOMINATION,
                detail=(
                    f"'{top.name}' reached rank 1 on keyword overlap while its feature "
                    f"and target evidence are absent"
                ),
                candidate_ids=[top.candidate_id],
            )
        )

    # AI evidence overtrust: an unverified candidate reached the top three on the
    # strength of prose alone. Its factors should be UNKNOWN, not credited.
    ai_top = [
        item
        for item in ranked[:3]
        if item.verification_status is not VerificationStatus.VERIFIED
    ]
    for item in ai_top:
        credited = [
            factor
            for factor in item.factors
            if factor.name in _EVIDENCE_SENSITIVE
            and factor.score is not None
            and factor.score > 0.5
        ]
        if credited:
            failures.append(
                CaseFailure(
                    category=FailureCategory.AI_EVIDENCE_OVERTRUST,
                    detail=(
                        f"unverified '{item.name}' (rank {item.rank}) was credited "
                        f"{', '.join(f.name for f in credited)} above 0.5 from prose"
                    ),
                    candidate_ids=[item.candidate_id],
                )
            )

    # Verification confusion: an unverified candidate outranked a verified one of
    # equal or better grade.
    verified_best = _best(
        [i for i in ranked if i.verification_status is VerificationStatus.VERIFIED],
        RELEVANT_THRESHOLD,
    )
    if verified_best is not None:
        for item in ranked:
            if (
                item.verification_status is not VerificationStatus.VERIFIED
                and item.rank < verified_best.rank
                and item.grade <= verified_best.grade
            ):
                failures.append(
                    CaseFailure(
                        category=FailureCategory.VERIFICATION_CONFUSION,
                        detail=(
                            f"unverified grade {item.grade} '{item.name}' (rank {item.rank}) "
                            f"outranked verified grade {verified_best.grade} "
                            f"'{verified_best.name}' (rank {verified_best.rank})"
                        ),
                        candidate_ids=[item.candidate_id, verified_best.candidate_id],
                    )
                )
                break

    # Contradiction missed: a clearly incompatible dataset in the top three with
    # a high compatibility score. A grade -1 dataset that the ranker scored as
    # compatible is a concrete detection gap, not a taste disagreement.
    for item in ranked[:3]:
        if item.grade <= -1:
            compat = _score_of(item, RankingFactorName.COMPATIBILITY.value)
            if compat is not None and compat >= 0.7:
                failures.append(
                    CaseFailure(
                        category=FailureCategory.CONTRADICTION_MISSED,
                        detail=(
                            f"grade -1 '{item.name}' in top 3 with compatibility "
                            f"{compat:.2f} - contradiction not detected"
                        ),
                        candidate_ids=[item.candidate_id],
                    )
                )

    # Semantic equivalence missed: grade >= 2 candidate whose target factor is
    # unknown (None) *and* whose target dimension is the one the case is graded
    # on. Distinguished from MISSING_TARGET_EVIDENCE by the corpus author having
    # supplied a dictionary term that the ranker did not resolve.
    if best is not None and _score_of(best, RankingFactorName.TARGET_MATCH.value) is None:
        failures.append(
            CaseFailure(
                category=FailureCategory.SEMANTIC_EQUIVALENCE_MISSED,
                detail=(
                    f"'{best.name}' is graded relevant but its target factor is unknown - "
                    "a synonym or label-space form the ranker did not resolve"
                ),
                candidate_ids=[best.candidate_id],
            )
        )

    # Metadata insufficiency: three or more of the top hit's requirement factors
    # are unknown, meaning most of its score is not evidence-based.
    unknown_requirement_factors = [
        factor
        for factor in top.factors
        if factor.name in _EVIDENCE_SENSITIVE and factor.score is None
    ]
    if len(unknown_requirement_factors) >= 3:
        failures.append(
            CaseFailure(
                category=FailureCategory.METADATA_INSUFFICIENT,
                detail=(
                    f"top hit '{top.name}' has {len(unknown_requirement_factors)} "
                    "unjudgeable requirement factors"
                ),
                candidate_ids=[top.candidate_id],
            )
        )

    return failures


class RelevanceBand:
    """Grade cut-offs used by the classification rules.

    Named constants rather than bare numbers so the rules above read as
    statements about grade bands instead of as magic numbers.
    """

    HIGHLY_RELEVANT = HIGHLY_RELEVANT_THRESHOLD
    RELEVANT = RELEVANT_THRESHOLD
    WEAK = 1
    IRRELEVANT = 0
    INCOMPATIBLE = -1