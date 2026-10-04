"""Score decomposition audit.

Every explanation this system shows a user is generated from factor
contributions. That makes the contributions the most safety-critical piece of
arithmetic in the codebase: if a contribution is wrong, the system does not merely
score badly, it *lies about why*. So the decomposition is audited rather than
trusted.

Two halves:

* **Reference table** - what each factor means, its raw range, weight, maximum
  point contribution, whether it can be unknown or mismatched, and whether it can
  dominate a decision. This is the document a reviewer reads.
* **Runtime arithmetic checks** - does the sum of contributions actually equal the
  displayed score, are any contributions impossible, is any factor counted twice,
  does any single factor carry enough weight to flip a ranking on its own.

Nothing here changes a weight. The audit reports what it finds.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from app.core.config import Settings
from app.schemas.dataset import DatasetCandidate, RankingFactorName
from app.services.datasets.evaluation.models import (
    AuditFinding,
    AuditReport,
    CaseEvaluation,
    FactorAuditEntry,
)

#: Points of tolerance when checking that contributions sum to the score. The
#: ranker rounds contributions and the score independently, so an exact equality
#: test would fail on rounding alone.
SUM_TOLERANCE = 0.05


def _factor_reference(settings: Settings) -> list[FactorAuditEntry]:
    """Static description of every ranking factor, with live weights."""
    return [
        FactorAuditEntry(
            name=RankingFactorName.DOMAIN_RELEVANCE.value,
            semantic_meaning=(
                "Does the dataset's declared subject matter overlap the project's "
                "domain and subdomain?"
            ),
            raw_range="0.0-1.0 from declared domain and tag overlap",
            weight=settings.ranking_weight_domain_relevance,
            nominal_share=settings.ranking_weight_domain_relevance / 100.0,
            max_contribution=settings.ranking_weight_domain_relevance,
            higher_is_better=True,
            unknown_possible=False,
            mismatch_possible=False,
            can_dominate=False,
            findings=[],
        ),
        FactorAuditEntry(
            name=RankingFactorName.KEYWORD_OVERLAP.value,
            semantic_meaning=(
                "How many of the project's query terms appear in the dataset's "
                "published text? Vocabulary overlap, not semantic equivalence."
            ),
            raw_range="0.0-1.0 matched_term_count / total required terms",
            weight=settings.ranking_weight_keyword_overlap,
            nominal_share=settings.ranking_weight_keyword_overlap / 100.0,
            max_contribution=settings.ranking_weight_keyword_overlap,
            higher_is_better=True,
            unknown_possible=False,
            mismatch_possible=False,
            can_dominate=False,
            findings=[
                "Generic terms such as 'students' or 'data' can match many datasets; "
                "this factor is why GENERIC_TERM_DOMINATION is a tracked failure."
            ],
        ),
        FactorAuditEntry(
            name=RankingFactorName.FEATURE_MATCH.value,
            semantic_meaning=(
                "Which required features are visible in the dataset's published "
                "schema, feature description or prose?"
            ),
            raw_range="0.0-1.0 covered_features / required_features",
            weight=settings.ranking_weight_feature_match,
            nominal_share=settings.ranking_weight_feature_match / 100.0,
            max_contribution=settings.ranking_weight_feature_match,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=True,
            can_dominate=False,
            findings=[
                "UNKNOWN when no schema evidence is published: silence proves "
                "absence of a declaration, not absence of the feature."
            ],
        ),
        FactorAuditEntry(
            name=RankingFactorName.TARGET_MATCH.value,
            semantic_meaning=(
                "Does the dataset publish a label space compatible with the "
                "project's target concept?"
            ),
            raw_range="0.0-1.0 dictionary label-space coverage",
            weight=settings.ranking_weight_target_match,
            nominal_share=settings.ranking_weight_target_match / 100.0,
            max_contribution=settings.ranking_weight_target_match,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=True,
            can_dominate=False,
            findings=[
                "Relies on the target-concept dictionary, so a valid synonym "
                "outside the dictionary reads as unknown or mismatched."
            ],
        ),
        FactorAuditEntry(
            name=RankingFactorName.TASK_MATCH.value,
            semantic_meaning="Is the requested ML task among the dataset's declared task types?",
            raw_range="0.0-1.0 declared task overlap",
            weight=settings.ranking_weight_task_match,
            nominal_share=settings.ranking_weight_task_match / 100.0,
            max_contribution=settings.ranking_weight_task_match,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=True,
            can_dominate=False,
        ),
        FactorAuditEntry(
            name=RankingFactorName.COMPATIBILITY.value,
            semantic_meaning=(
                "Positive evidence that the dataset is something the project "
                "cannot use. Scored down from 1.0 by typed contradictions."
            ),
            raw_range="0.0-1.0 (1.0 minus summed contradiction penalties, capped)",
            weight=settings.ranking_weight_compatibility,
            nominal_share=settings.ranking_weight_compatibility / 100.0,
            max_contribution=settings.ranking_weight_compatibility,
            higher_is_better=True,
            unknown_possible=False,
            mismatch_possible=True,
            can_dominate=False,
            findings=[
                "Detected only from declared metadata on verified candidates; "
                "unverified candidates are never contradiction-penalised."
            ],
        ),
        FactorAuditEntry(
            name=RankingFactorName.QUALITY.value,
            semantic_meaning="Registry usability rating for the dataset.",
            raw_range="0.0-1.0, None when the registry publishes no rating",
            weight=settings.ranking_weight_quality,
            nominal_share=settings.ranking_weight_quality / 100.0,
            max_contribution=settings.ranking_weight_quality,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=False,
            can_dominate=False,
        ),
        FactorAuditEntry(
            name=RankingFactorName.POPULARITY.value,
            semantic_meaning="Adoption signals: download and vote volume, log-scaled.",
            raw_range="0.0-1.0, None when the registry publishes no counts",
            weight=settings.ranking_weight_popularity,
            nominal_share=settings.ranking_weight_popularity / 100.0,
            max_contribution=settings.ranking_weight_popularity,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=False,
            can_dominate=False,
            findings=[
                "Deliberately small: popularity is evidence of adoption, never of "
                "fit. POPULARITY_DOMINATION is tracked as a failure."
            ],
        ),
        FactorAuditEntry(
            name=RankingFactorName.RECENCY.value,
            semantic_meaning="How recently the dataset was updated.",
            raw_range="0.0-1.0 decay over update age, None when no update date is published",
            weight=settings.ranking_weight_recency,
            nominal_share=settings.ranking_weight_recency / 100.0,
            max_contribution=settings.ranking_weight_recency,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=False,
            can_dominate=False,
            findings=[
                "Deliberately the smallest weight; recency is not fitness."
            ],
        ),
        FactorAuditEntry(
            name=RankingFactorName.VERIFICATION.value,
            semantic_meaning=(
                "How the candidate was obtained: confirmed registry record versus "
                "a model's suggestion."
            ),
            raw_range="0.0-1.0, None for unverified AI suggestions",
            weight=settings.ranking_weight_verification,
            nominal_share=settings.ranking_weight_verification / 100.0,
            max_contribution=settings.ranking_weight_verification,
            higher_is_better=True,
            unknown_possible=True,
            mismatch_possible=False,
            can_dominate=True,
            findings=[
                "Largest single weight, so it can outvote a large relevance gap. "
                "This is the intended trade-off and the reason "
                "VERIFICATION_CONFUSION is tracked; the sensitivity report measures "
                "what removing it would cost."
            ],
        ),
    ]


def _check_contributions(candidate: DatasetCandidate) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    factors = candidate.ranking_factors
    if not factors:
        return findings

    names = Counter(f.name for f in factors)
    for name, count in names.items():
        if count > 1:
            findings.append(
                AuditFinding(
                    severity="error",
                    subject=f"{candidate.id}:{name.value}",
                    detail=f"factor appears {count} times; contribution would double-count",
                )
            )

    total = sum(f.contribution for f in factors)
    displayed = candidate.ranking_score or 0.0
    if abs(total - displayed) > SUM_TOLERANCE:
        findings.append(
            AuditFinding(
                severity="error",
                subject=candidate.id,
                detail=(
                    f"contributions sum to {total:.2f} but the displayed score is "
                    f"{displayed:.2f}; the explanation does not match the score"
                ),
            )
        )

    for factor in factors:
        if factor.contribution < -1e-9:
            findings.append(
                AuditFinding(
                    severity="error",
                    subject=f"{candidate.id}:{factor.name.value}",
                    detail=f"negative contribution {factor.contribution:.2f}",
                )
            )
        if factor.score is not None and not (0.0 <= factor.score <= 1.0):
            findings.append(
                AuditFinding(
                    severity="error",
                    subject=f"{candidate.id}:{factor.name.value}",
                    detail=f"score {factor.score:.3f} outside the documented 0.0-1.0 range",
                )
            )
        if factor.score is None and factor.contribution != 0.0:
            findings.append(
                AuditFinding(
                    severity="error",
                    subject=f"{candidate.id}:{factor.name.value}",
                    detail="unjudgeable factor must contribute exactly zero",
                )
            )
    return findings


#: How far above its nominal weight a contribution may run before the audit treats
#: it as denominator inflation rather than as rounding noise.
#:
#: The ratio under test is ``contribution / weight``, which is exactly 1.0 for a
#: full-evidence candidate and cannot exceed 1.0 at all while the score mean
#: divides by the nominal factor weight. Contributions are rounded to two
#: decimals, so the largest artifact this comparison can see is ``0.005 / 2`` on
#: the smallest factor - well under a percent. 0.02 leaves room for that rounding
#: while still catching any factor that is paid even slightly above its weight.
#:
#: This constant was 0.20 when the audit compared a factor's share of the achieved
#: score against ``weight / 100``. That comparison also fired on every candidate
#: simply for scoring below 100, so it reported ten inflated factors on arithmetic
#: that was already correct, and the real defect - a sparse candidate's survivors
#: reaching 1.59x their weight - was diluted among the false positives.
INFLATION_TOLERANCE = 0.02


def _observe(
    entry: FactorAuditEntry,
    contribution: float,
    share: float,
    ratio: float,
) -> None:
    entry.observed_max_contribution = max(entry.observed_max_contribution, round(contribution, 4))
    entry.observed_max_share = max(entry.observed_max_share, round(share, 4))
    entry.observed_max_ratio = max(entry.observed_max_ratio, round(ratio, 4))


def _is_inflated(entry: FactorAuditEntry) -> bool:
    """Whether a factor was ever paid more than its nominal weight.

    Tested against ``contribution / weight`` rather than against the share of the
    achieved score. The share test was a proxy that also fires on every candidate
    simply for scoring below 100, which made a pinned denominator look broken.
    """
    return entry.observed_max_ratio > 1.0 + INFLATION_TOLERANCE


def _factor_entries_by_name(settings: Settings) -> dict[str, FactorAuditEntry]:
    return {entry.name: entry for entry in _factor_reference(settings)}


def audit_scores(
    evaluations: Sequence[CaseEvaluation], settings: Settings
) -> AuditReport:
    """Run the arithmetic checks over an entire evaluation run.

    Two things are checked, and they are deliberately different in kind:

    * **Arithmetic invariants** - contributions sum to the score, no negative or
      out-of-range values, no factor counted twice, unjudgeable factors contribute
      nothing. A violation of any of these is a bug and is reported as an error.
    * **Observed influence** - the largest contribution and largest share of a
      final score each factor actually reached across the run. This is reported
      rather than asserted, because the point ceiling of a factor is
      ``weight / counted_weight * 100``: when other factors are dropped from the
      denominator the survivors are inflated in proportion. A factor that reached
      a materially larger share than its nominal weight is worth a reviewer's
      attention, and the table below shows exactly which factors those are.
    """
    findings: list[AuditFinding] = []
    max_error = 0.0
    duplicated = 0
    impossible = 0
    entries = _factor_entries_by_name(settings)

    for evaluation in evaluations:
        for item in evaluation.ranked:
            case_findings = _check_contributions(_AsCandidateView(item))
            findings.extend(case_findings)
            max_error = max(
                max_error,
                abs(sum(f.contribution for f in item.factors) - item.score),
            )
            duplicated += sum(1 for f in case_findings if "appears" in f.detail)
            impossible += sum(
                1
                for f in case_findings
                if f.severity == "error" and "appears" not in f.detail
            )
            if item.score <= 0:
                continue
            for factor in item.factors:
                entry = entries.get(factor.name)
                if entry is not None:
                    _observe(
                        entry,
                        factor.contribution,
                        factor.contribution / item.score,
                        factor.contribution / factor.weight if factor.weight else 0.0,
                    )

    inflated = sum(1 for entry in entries.values() if _is_inflated(entry))

    factors = list(entries.values())
    total_weight = sum(f.weight for f in factors)
    for entry in factors:
        if entry.weight > 20.0:
            entry.findings.append(
                f"weight {entry.weight:g} is more than 20 points; this factor can "
                "decide a ranking on its own"
            )
        if entry.can_dominate:
            entry.findings.append(
                "can dominate: single-signal decisions are possible and are "
                "checked by the ablation and sensitivity reports"
            )
        if _is_inflated(entry):
            entry.findings.append(
                f"paid up to {entry.observed_max_ratio:.3f}x its nominal weight, so the "
                "score mean rescaled the factor set"
            )
            findings.append(
                AuditFinding(
                    severity="warning",
                    subject=f"factor:{entry.name}",
                    detail=(
                        f"denominator inflation: reached "
                        f"{entry.observed_max_contribution:.2f} points against a "
                        f"nominal weight of {entry.weight:g} "
                        f"({entry.observed_max_ratio:.3f}x)"
                    ),
                )
            )

    if findings:
        findings.append(
            AuditFinding(
                severity="summary",
                subject="score decomposition",
                detail=(
                    f"{sum(1 for f in findings if f.severity == 'error')} error(s), "
                    f"{sum(1 for f in findings if f.severity == 'warning')} warning(s); "
                    f"largest contribution-sum error {max_error:.4f}; "
                    f"{inflated} of {len(entries)} factors exceeded their nominal share"
                ),
            )
        )

    return AuditReport(
        max_total_weight=total_weight,
        contributions_sum_matches_score=max_error <= SUM_TOLERANCE,
        max_abs_contribution_error=round(max_error, 4),
        duplicated_contributions=duplicated,
        impossible_contributions=impossible,
        inflated_contributions=inflated,
        factors=factors,
        findings=findings,
    )


class _AsCandidateView:
    """Adapts a :class:`RankedItem` to the attribute shape ``_check_contributions`` reads.

    The audit works on diagnostic views so it can run without the original
    candidate objects, which keeps it usable against a stored report.
    """

    __slots__ = ("id", "ranking_factors", "ranking_score")

    def __init__(self, item) -> None:
        from app.schemas.dataset import RankingFactor

        self.id = item.candidate_id
        self.ranking_score = item.score
        self.ranking_factors = [
            RankingFactor(
                name=RankingFactorName(factor.name),
                weight=factor.weight,
                score=factor.score,
                contribution=factor.contribution,
                detail=factor.detail,
                matched_terms=list(factor.matched_terms),
                unmatched_terms=list(factor.unmatched_terms),
                unknown_terms=list(factor.unknown_terms),
                contradictions=list(factor.contradictions),
            )
            for factor in item.factors
        ]


def dominance_table(evaluation: CaseEvaluation, limit: int = 5) -> list[tuple[str, str, float]]:
    """Per-factor share of the top hit's score, for the report's influence section."""
    if not evaluation.ranked:
        return []
    top = evaluation.ranked[0]
    total = sum(f.contribution for f in top.factors) or 1.0
    rows = [
        (factor.name, factor.display, factor.contribution / total)
        for factor in top.factors
    ]
    rows.sort(key=lambda row: -row[2])
    return rows[:limit]


def factor_influence_over_benchmark(evaluation: CaseEvaluation) -> dict[str, float]:
    """Average share of score per factor across every ranked candidate in a run.

    This is the aggregate answer to "which factors actually decide the ranking",
    which is a different question from "which factor is heaviest".
    """
    shares: dict[str, list[float]] = {}
    for evaluation_case in evaluation.per_case:
        for item in evaluation_case.ranked:
            total = sum(f.contribution for f in item.factors)
            if total <= 0:
                continue
            for factor in item.factors:
                shares.setdefault(factor.name, []).append(factor.contribution / total)
    return {
        name: sum(values) / len(values) for name, values in sorted(shares.items())
    }


def summarise_findings(report: AuditReport) -> Iterable[str]:
    for finding in report.findings:
        yield f"[{finding.severity}] {finding.subject}: {finding.detail}"