"""The final answer to "which dataset should I use?".

Everything here is derived from the ``DatasetIntelligence`` record that already
exists for each candidate. Nothing is asked of a generative model: a model asked to
explain a recommendation will happily describe columns it has never seen, whereas a
sentence rendered from typed evidence can only say what that evidence says.

    candidate + DatasetIntelligence
        -> RecommendationConfidence   (can we stand behind this dataset?)
        -> FinalRecommendation        (best match, alternatives, what to watch for)

Two rules shape the result and are enforced by the schema, not only by this module:

* An unconfirmed AI suggestion is never RECOMMENDED. At most it is TENTATIVE.
* A dataset found incompatible is never put forward, however highly it ranks.
  Ranking asks "how highly should we recommend this?"; compatibility asks "can it
  actually support the project?". The recommendation needs both answers.
"""

from __future__ import annotations

from typing import Optional, Sequence

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    MatchQuality,
    MetadataDepth,
    SourceType,
    VerificationStatus,
)
from app.schemas.intelligence import (
    CompatibilityLevel,
    DatasetIntelligence,
    FeatureMatchState,
    RiskSeverity,
    TargetState,
)
from app.schemas.recommendation import (
    AlternativeDataset,
    ConfidenceLevel,
    FinalRecommendation,
    RecommendationConfidence,
    RecommendationOutcome,
    RecommendedDataset,
)

logger = get_logger(__name__)

#: Two datasets whose ranking scores differ by less than this are a close call.
#: 5.0 is just under 5.33, the smallest controlled relevance gap the ranking stress
#: test measured (evaluation/RANKING_STRESS_TEST.md): anything narrower is within the
#: swing that a single factor change can produce, so a stated "best" would overstate
#: the lead.
CLOSE_CALL_MARGIN = 5.0

#: Runners-up shown with the best match.
MAX_ALTERNATIVES = 3

#: Lines kept per evidence list in the recommended dataset's summary.
MAX_LINES = 6

#: Cautions listed for the best match.
MAX_WATCH_OUTS = 5

_LEVEL_ORDER = {ConfidenceLevel.LOW: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.HIGH: 2}
_COMPATIBILITY_ORDER = {
    CompatibilityLevel.LOW: 0,
    CompatibilityLevel.UNKNOWN: 1,
    CompatibilityLevel.MEDIUM: 2,
    CompatibilityLevel.HIGH: 3,
}
_SEVERITY_ORDER = {RiskSeverity.CRITICAL: 0, RiskSeverity.WARNING: 1, RiskSeverity.INFO: 2}


def _cap(level: ConfidenceLevel, ceiling: ConfidenceLevel) -> ConfidenceLevel:
    return level if _LEVEL_ORDER[level] <= _LEVEL_ORDER[ceiling] else ceiling


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def build_confidence(
    candidate: DatasetCandidate, intelligence: DatasetIntelligence
) -> RecommendationConfidence:
    """How sure we are that one dataset suits the project, and every reason why.

    Starts from the compatibility level (UNKNOWN counts as LOW: not knowing is not a
    reason for confidence) and is only ever *capped* from there, never raised. Each
    cap is a fact about the evidence, and each is reported.
    """
    compatibility = intelligence.compatibility.overall
    level = {
        CompatibilityLevel.HIGH: ConfidenceLevel.HIGH,
        CompatibilityLevel.MEDIUM: ConfidenceLevel.MEDIUM,
        CompatibilityLevel.LOW: ConfidenceLevel.LOW,
        CompatibilityLevel.UNKNOWN: ConfidenceLevel.LOW,
    }[compatibility]
    reasons = [f"compatibility with the project is {compatibility.value}"]

    if candidate.source_type is SourceType.AI_SUGGESTED:
        level = _cap(level, ConfidenceLevel.LOW)
        reasons.append("existence is not confirmed by any registry")
    elif candidate.verification_status is VerificationStatus.VERIFIED:
        reasons.append(f"confirmed to exist in {candidate.source}")
        if candidate.metadata_depth is MetadataDepth.LISTING:
            level = _cap(level, ConfidenceLevel.MEDIUM)
            reasons.append("only the registry's search listing was read")

    target = intelligence.target_analysis
    if target.state is TargetState.UNKNOWN:
        level = _cap(level, ConfidenceLevel.MEDIUM)
        reasons.append("the target could not be confirmed in the dataset")
    elif target.state is TargetState.DERIVABLE:
        level = _cap(level, ConfidenceLevel.MEDIUM)
        reasons.append("the target is not stored and must be derived")
    elif target.state is TargetState.PRESENT and target.match_quality is MatchQuality.SEMANTIC:
        level = _cap(level, ConfidenceLevel.MEDIUM)
        reasons.append("the target matched only through a dictionary equivalent")

    coverage = intelligence.feature_coverage
    inferred = [f for f in coverage.matched if f.state is FeatureMatchState.INFERRED]
    if inferred:
        level = _cap(level, ConfidenceLevel.MEDIUM)
        reasons.append(
            f"{len(inferred)} required "
            f"{'feature' if len(inferred) == 1 else 'features'} matched only by shared subject area"
        )
    if coverage.unknown_count:
        level = _cap(level, ConfidenceLevel.MEDIUM)
        reasons.append(f"{coverage.unknown_count} required features could not be verified")
    if coverage.missing_count:
        reasons.append(f"{coverage.missing_count} required features are missing")

    return RecommendationConfidence(level=level, reasons=reasons)


# ---------------------------------------------------------------------------
# Final recommendation
# ---------------------------------------------------------------------------


def _watch_out_for(intelligence: DatasetIntelligence) -> list[str]:
    """Cautions for the best match: risks by severity, then limitations."""
    lines: list[str] = []
    risks = sorted(intelligence.risks, key=lambda r: _SEVERITY_ORDER[r.severity])
    for item in [*(r.detail for r in risks), *(l.detail for l in intelligence.limitations)]:
        if item not in lines:
            lines.append(item)
    return lines[:MAX_WATCH_OUTS]


def _trade_off(
    best: DatasetCandidate,
    best_intel: DatasetIntelligence,
    other: DatasetCandidate,
    other_intel: DatasetIntelligence,
) -> str:
    """The single most important reason ``other`` is not the best match.

    Checked in a fixed order, strongest first, so the same pair always yields the
    same sentence.
    """
    compat = other_intel.compatibility
    if compat.overall is CompatibilityLevel.LOW:
        why = (
            compat.contradictions[0]
            if compat.contradictions
            else next((r.detail for r in other_intel.risks), "a requirement is not met")
        )
        return f"found incompatible with the project: {why}"
    if (
        other.source_type is SourceType.AI_SUGGESTED
        and best.source_type is SourceType.VERIFIED_EXTERNAL
    ):
        return "suggested by a model and not confirmed by any registry"
    if _COMPATIBILITY_ORDER[compat.overall] < _COMPATIBILITY_ORDER[best_intel.compatibility.overall]:
        return (
            f"lower compatibility ({compat.overall.value} against "
            f"{best_intel.compatibility.overall.value})"
        )
    other_cov, best_cov = other_intel.feature_coverage, best_intel.feature_coverage
    if other_cov.missing_count > best_cov.missing_count:
        return f"{other_cov.missing_count} required features missing (against {best_cov.missing_count})"
    if other_cov.unknown_count > best_cov.unknown_count:
        return (
            f"{other_cov.unknown_count} required features unverified "
            f"(against {best_cov.unknown_count})"
        )
    gap = float(best.ranking_score or 0.0) - float(other.ranking_score or 0.0)
    if gap > 0:
        return f"scores {gap:.1f} points lower overall"
    return "comparable on every requirement; ranked lower on the tie-break"


def _eligible(intelligence: DatasetIntelligence) -> bool:
    """A dataset found incompatible is never put forward."""
    return intelligence.compatibility.overall is not CompatibilityLevel.LOW


def _cap_lines(lines: Sequence[str]) -> list[str]:
    return list(lines)[:MAX_LINES]


def build_final_recommendation(
    candidates: Sequence[DatasetCandidate],
    intelligence: Sequence[DatasetIntelligence],
    confidences: Sequence[RecommendationConfidence],
) -> FinalRecommendation:
    """Choose the best match, the runners-up and the cautions, from evidence only.

    ``candidates`` must be in ranking order and aligned with ``intelligence`` and
    ``confidences``.
    """
    if not (len(candidates) == len(intelligence) == len(confidences)):
        raise ValueError("candidates, intelligence and confidences must be aligned")

    if not candidates:
        return FinalRecommendation(
            outcome=RecommendationOutcome.NONE,
            summary="No candidate datasets were found for this project.",
        )

    rows = list(zip(range(1, len(candidates) + 1), candidates, intelligence, confidences))
    eligible = [row for row in rows if _eligible(row[2])]
    verified = [row for row in eligible if row[1].source_type is SourceType.VERIFIED_EXTERNAL]

    # A registry-confirmed dataset is always preferred to an unconfirmed one that
    # happens to rank a few points higher: confirmed existence is the one thing an
    # AI suggestion cannot offer.
    chosen = (verified or eligible or [None])[0]

    if chosen is None:
        closest = rows[0]
        summary = (
            f"None of the {len(candidates)} candidate datasets could be shown to support "
            f"this project. The closest, {closest[1].name}, was {_incompatibility(closest[2])}."
        )
        alternatives = [
            _alternative(rank, c, i, closest[1], closest[2])
            for rank, c, i, _ in rows[:MAX_ALTERNATIVES]
        ]
        return FinalRecommendation(
            outcome=RecommendationOutcome.NONE,
            summary=summary,
            alternatives=alternatives,
            caveats=["Every candidate was found incompatible with at least one requirement."],
        )

    rank, best, best_intel, best_conf = chosen
    others = [row for row in rows if row[0] != rank]
    runner = next((row for row in others if _eligible(row[2])), None)

    margin: Optional[float] = None
    close = False
    if runner is not None:
        margin = round(float(best.ranking_score or 0.0) - float(runner[1].ranking_score or 0.0), 2)
        close = margin < CLOSE_CALL_MARGIN

    watch = _watch_out_for(best_intel)
    explanation = best_intel.explanation
    why = list(explanation.why_recommended)
    if explanation.requirement_coverage:
        why.append(explanation.requirement_coverage[0])

    recommended = RecommendedDataset(
        dataset_id=best.id,
        name=best.name,
        source=best.source,
        source_type=best.source_type,
        url=best.url,
        rank=rank,
        ranking_score=float(best.ranking_score or 0.0),
        compatibility=best_intel.compatibility.overall,
        confidence=best_conf,
        why=_cap_lines(why),
        verified=_cap_lines(explanation.what_is_verified),
        unknown=_cap_lines(explanation.what_is_unknown),
        missing=_cap_lines(explanation.what_is_missing),
        watch_out_for=watch,
    )

    confirmed = best.source_type is SourceType.VERIFIED_EXTERNAL
    outcome = (
        RecommendationOutcome.RECOMMENDED
        if confirmed and best_conf.level is not ConfidenceLevel.LOW
        else RecommendationOutcome.TENTATIVE
    )

    caveats: list[str] = []
    if not verified:
        caveats.append(
            "No registry-confirmed dataset was found, so this suggestion rests on a "
            "model's description and has not been checked against any registry."
        )
    if rank > 1:
        caveats.append(
            f"Ranked #{rank}: higher-ranked candidates were passed over because they "
            "were found incompatible or are unconfirmed."
        )

    alternatives = [
        _alternative(r, c, i, best, best_intel) for r, c, i, _ in others[:MAX_ALTERNATIVES]
    ]

    return FinalRecommendation(
        outcome=outcome,
        summary=_summary(outcome, best, best_intel, best_conf, watch, runner, margin, close),
        best_match=recommended,
        alternatives=alternatives,
        margin_to_next=margin,
        close_call=close,
        caveats=caveats,
    )


def _incompatibility(intelligence: DatasetIntelligence) -> str:
    compat = intelligence.compatibility
    if compat.contradictions:
        return f"found incompatible: {compat.contradictions[0]}"
    for risk in intelligence.risks:
        if risk.severity is not RiskSeverity.INFO:
            return f"found incompatible: {risk.detail}"
    return "found incompatible with the project"


def _alternative(
    rank: int,
    candidate: DatasetCandidate,
    intelligence: DatasetIntelligence,
    best: DatasetCandidate,
    best_intel: DatasetIntelligence,
) -> AlternativeDataset:
    return AlternativeDataset(
        dataset_id=candidate.id,
        name=candidate.name,
        source=candidate.source,
        rank=rank,
        ranking_score=float(candidate.ranking_score or 0.0),
        compatibility=intelligence.compatibility.overall,
        verification_status=candidate.verification_status,
        trade_off=_trade_off(best, best_intel, candidate, intelligence),
    )


def _summary(
    outcome: RecommendationOutcome,
    best: DatasetCandidate,
    intel: DatasetIntelligence,
    confidence: RecommendationConfidence,
    watch: Sequence[str],
    runner,
    margin: Optional[float],
    close: bool,
) -> str:
    coverage = intel.feature_coverage
    target = intel.target_analysis
    if outcome is RecommendationOutcome.RECOMMENDED:
        head = f"{best.name} ({best.source}) is the best match"
    else:
        head = f"{best.name} ({best.source}) is worth trying but is not a confident recommendation"

    parts = [
        f"{head}: compatibility {intel.compatibility.overall.value}, "
        f"confidence {confidence.level.value}.",
        f"Features: {coverage.summary()}.",
        f"Target: {target.state.value}.",
    ]
    if outcome is RecommendationOutcome.TENTATIVE:
        parts.append(f"Reason for caution: {confidence.reasons[-1]}.")
    if watch:
        parts.append(f"Main caution: {watch[0]}")
    if close and runner is not None:
        parts.append(
            f"The next candidate, {runner[1].name}, is within {margin:.1f} points, so "
            "treat this as a close call."
        )
    return " ".join(parts)


def build_recommendations(
    candidates: Sequence[DatasetCandidate],
    intelligence: Sequence[DatasetIntelligence],
) -> tuple[list[RecommendationConfidence], FinalRecommendation]:
    """Per-dataset confidence plus the final recommendation, in one call."""
    confidences = [build_confidence(c, i) for c, i in zip(candidates, intelligence)]
    return confidences, build_final_recommendation(candidates, intelligence, confidences)
