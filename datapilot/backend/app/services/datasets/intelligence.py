"""Builds a ``DatasetIntelligence`` record for one already-ranked candidate.

Design rule: **this module never re-decides a match.** Every MATCH / UNKNOWN /
MISMATCH verdict it reports is read out of the ranking engine's own output
(``candidate.ranking_factors[].requirements``). Compatibility and ranking then
answer different questions from the *same* facts, instead of two independent
implementations drifting apart and disagreeing about whether a feature is
present.

The ranking engine is not modified, imported for behaviour, or reweighted. Two
pure text helpers are reused from it because they encode project knowledge that
must not be duplicated here: ``_declares_schema`` (does this free text enumerate
the dataset's own columns?) and ``_TARGET_NOT_PROVIDED`` (does this text declare
the label absent?). Duplicating either would produce a second, drifting answer
to the same question.

What this module adds is the layer ranking does not have: a schema *state*, a
four-way target state, per-feature provenance, categorical compatibility, risks
and limitations - all derived from evidence already on the candidate.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    MatchQuality,
    MatchStatus,
    ProjectAnalysisData,
    RankingFactor,
    RankingFactorName,
    RequirementMatch,
    MetadataDepth,
    SourceType,
    VerificationStatus,
)
from app.schemas.intelligence import (
    DECLARING_LAYERS,
    EMITTABLE_RISK_CATEGORIES,
    CompatibilityAssessment,
    CompatibilityLevel,
    DatasetIntelligence,
    DatasetLimitation,
    DatasetRisk,
    DimensionCompatibility,
    EvidenceClaim,
    EvidenceLayer,
    FeatureCompatibility,
    FeatureCoverage,
    FeatureMatchState,
    RecommendationExplanation,
    RiskCategory,
    RiskSeverity,
    SchemaState,
    TargetAnalysis,
    TargetState,
    strongest_layer,
)
from app.services.datasets.target_concepts import value_tokens
from app.services.datasets.compatibility import (
    _candidate_modalities,
    _project_modalities,
    find_contradictions,
)
from app.services.datasets.ranking import (
    DatasetRankingEngine,
    _declares_schema,
    _TARGET_NOT_PROVIDED,
    build_project_profile,
    get_target_concept_index,
)

logger = get_logger(__name__)

#: Fields whose presence means the registry published a statement about them.
_DECLARED_EVIDENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("description", "description"),
    ("domain", "domain"),
    ("tags", "tags"),
    ("task_types", "ml_tasks"),
    ("target_description", "target_description"),
    ("feature_description", "feature_description"),
    ("license", "license"),
)

#: A free-text schema enumeration can show that a feature is present, but it is
#: prose: such lists are often partial, so it is not a verified schema. It yields
#: PARTIAL, never VERIFIED. Only a structured column list that the registry itself
#: published (``DatasetCandidate.columns``) produces VERIFIED / SCHEMA_VERIFIED.
_TEXT_ENUMERATED_SCHEMA = SchemaState.PARTIAL

#: A construction route: the candidate says a label would have to be *made*.
#: This is the second half of DERIVABLE, and it is required - the first half is
#: an admission that no label is published. Deliberately narrow: "requires
#: pre-processing" is not a route to a target.
_DERIVATION_ROUTE = re.compile(
    r"\b(?:can|could|would|must|may|should)\s+be\s+"
    r"(?:derived|computed|calculated|inferred|constructed|engineered|created|"
    r"generated|assigned|labelled|labeled|annotated)\b"
    r"|\b(?:derive|deriving|compute|calculating|infer|inferring|construct|"
    r"engineering)\b[^.;]{0,40}?\b(?:target|label|outcome|ground[- ]truth)\b",
    re.IGNORECASE,
)

#: The subject the route has to be about. Guards against a candidate that
#: mentions both a missing target and an unrelated computation and would
#: otherwise be read as promising a derivation.
_DERIVATION_SUBJECT = re.compile(
    r"\b(?:target|label|outcome|class|ground[- ]truth|annotation)\b", re.IGNORECASE
)

_STATES_FOR_QUALITY: dict[MatchQuality, FeatureMatchState] = {
    MatchQuality.EXACT: FeatureMatchState.EXACT,
    MatchQuality.ALIAS: FeatureMatchState.ALIAS,
    MatchQuality.SEMANTIC: FeatureMatchState.SEMANTIC,
    MatchQuality.INFERRED: FeatureMatchState.INFERRED,
}


def _factor(candidate: DatasetCandidate, name: RankingFactorName) -> Optional[RankingFactor]:
    for factor in candidate.ranking_factors:
        if factor.name is name:
            return factor
    return None


def _requirements(
    candidate: DatasetCandidate, name: RankingFactorName
) -> list[RequirementMatch]:
    factor = _factor(candidate, name)
    if factor is None:
        return []
    if factor.requirements:
        return list(factor.requirements)
    # A factor that scored without per-term verdicts (no requirements, or an
    # early return such as "no required features identified"). Fall back to the
    # term lists so a match is still reported rather than silently dropped.
    out: list[RequirementMatch] = []
    for term in factor.matched_terms:
        out.append(RequirementMatch(term=term, status=MatchStatus.MATCH))
    for term in factor.unknown_terms:
        out.append(RequirementMatch(term=term, status=MatchStatus.UNKNOWN))
    for term in factor.unmatched_terms:
        out.append(RequirementMatch(term=term, status=MatchStatus.MISMATCH))
    return out


def _publishes(candidate: DatasetCandidate) -> bool:
    """Whether the candidate's own fields are registry statements, not model output.

    Same rule the ranker uses: a model may satisfy a requirement, but it can
    never *prove absence*, because its silence is not a statement.
    """
    return candidate.verification_status is VerificationStatus.VERIFIED


def _candidate_layer(candidate: DatasetCandidate) -> EvidenceLayer:
    """The strongest evidence layer available anywhere on this candidate."""
    if candidate.source_type is SourceType.AI_SUGGESTED:
        # A model's text is the origin of every present field here. No registry
        # published any of it, so the record cannot exceed AI_SUGGESTED however
        # confident the model claimed to be.
        return EvidenceLayer.AI_SUGGESTED
    if _publishes(candidate):
        return EvidenceLayer.SOURCE_DECLARED
    if candidate.evidence_text().strip():
        return EvidenceLayer.INFERRED
    return EvidenceLayer.UNKNOWN


def _field_layer(candidate: DatasetCandidate, field: str) -> EvidenceLayer:
    """Provenance of one metadata field, as a claim layer."""
    value = getattr(candidate, field, None)
    empty = value is None or (isinstance(value, (str, list, tuple, set, dict)) and not value)
    if empty:
        return EvidenceLayer.UNKNOWN
    if field == "domain":
        # ``domain`` is always DataPilot's own tag mapping; no registry publishes
        # it. Claiming it as declared would assert verification never performed.
        return (
            EvidenceLayer.AI_SUGGESTED
            if candidate.source_type is SourceType.AI_SUGGESTED
            else EvidenceLayer.INFERRED
        )
    if candidate.source_type is SourceType.AI_SUGGESTED:
        return EvidenceLayer.AI_SUGGESTED
    return EvidenceLayer.SOURCE_DECLARED


def _evidence_claims(candidate: DatasetCandidate) -> list[EvidenceClaim]:
    """One claim per published metadata field, in a deterministic order."""
    claims: list[EvidenceClaim] = []
    for field, subject in _DECLARED_EVIDENCE_FIELDS:
        layer = _field_layer(candidate, field)
        raw = getattr(candidate, field, None)
        if isinstance(raw, (list, tuple, set)):
            value: Optional[str] = ", ".join(sorted(str(v) for v in raw)) or None
        else:
            value = str(raw).strip() if raw is not None else None
        claims.append(
            EvidenceClaim(
                subject=subject,
                value=value,
                layer=layer,
                source=candidate.source if layer is not EvidenceLayer.UNKNOWN else "none",
            )
        )
    if candidate.columns:
        # A structured column list the registry published. It is the only thing in
        # this record that is machine-generated schema rather than prose, so it is
        # the only thing allowed to carry the SCHEMA_VERIFIED layer - and only
        # when a registry, not a model, published it.
        shown = ", ".join(candidate.columns[:40]) + (
            f" (+{len(candidate.columns) - 40} more)" if len(candidate.columns) > 40 else ""
        )
        claims.append(
            EvidenceClaim(
                subject="columns",
                value=shown,
                layer=(
                    EvidenceLayer.SCHEMA_VERIFIED
                    if _publishes(candidate)
                    else EvidenceLayer.AI_SUGGESTED
                ),
                source=candidate.source,
            )
        )
    for field, subject in (("file_types", "file_types"), ("modalities", "modalities")):
        values = getattr(candidate, field)
        if values:
            claims.append(
                EvidenceClaim(
                    subject=subject,
                    value=", ".join(sorted(values)),
                    layer=_field_layer(candidate, field),
                    source=candidate.source,
                )
            )
    return claims


def _schema_state(candidate: DatasetCandidate) -> SchemaState:
    """How much trustworthy schema information this candidate carries.

    ``VERIFIED`` requires a *structured* column list published by the registry
    itself (Hugging Face's ``dataset_info.features``). DataPilot does not download
    or parse dataset files, so ``VERIFIED`` means "the registry's own machine-made
    schema", not "we read the file". Prose that enumerates columns is ``PARTIAL``:
    it can show a feature is present, but a prose list may be incomplete, so it
    cannot on its own settle a dataset's overall compatibility.
    """
    if not _publishes(candidate):
        # An AI candidate's prose is a model's guess about columns. It cannot
        # prove anything about absence, so it is not a schema at all.
        return SchemaState.UNKNOWN
    if candidate.columns:
        # The registry published a structured column list. That is a schema, not
        # prose about one, so absence can be concluded from it.
        return SchemaState.VERIFIED
    enumerates = _declares_schema(candidate.description) or _declares_schema(
        candidate.feature_description
    )
    if enumerates:
        return _TEXT_ENUMERATED_SCHEMA
    if candidate.feature_description:
        # The registry published a feature field. It exists, but free text is
        # not a column list, so absence cannot be proven from it.
        return SchemaState.PARTIAL
    return SchemaState.UNKNOWN


def _target_analysis(
    candidate: DatasetCandidate,
    contradictions: Optional[dict[str, list[str]]] = None,
) -> TargetAnalysis:
    """Four-way target state, decided from declared evidence only.

    Gemini speculation is not sufficient for ``PRESENT`` or ``DERIVABLE``; the
    model validator on ``TargetAnalysis`` refuses to construct either on a
    non-declaring layer, so the rule cannot be bypassed by a future caller.

    A declared target contradiction outranks a keyword match. The ranker scores
    the target factor by finding requirement words in the candidate's text, so a
    dataset about "whether the student passed the final examination" registers a
    hit on a project target of "hospital readmission rate" purely through the
    word "examination". The contradiction detector compares the two label spaces
    directly and concludes they differ, which is the more specific statement
    about the same evidence, so it is the one reported here. This resolves the
    conflict in favour of the more specific finding rather than re-deciding the
    match on our own terms.
    """
    verdict = _first_target_requirement(candidate)
    requirement = verdict.term if verdict else None
    factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
    declared_conflict = (contradictions or {}).get("target") or []
    layer = _target_layer(candidate)
    status = verdict.status if verdict else MatchStatus.UNKNOWN

    # An admission that no label is published is the most specific thing the
    # candidate can say about its target, so it is settled before anything else.
    #
    # It has to come first because the shared contradiction detector compares
    # label *families*, and a sentence like "no target column is provided"
    # contains the word "no" next to "target" - which reads to the family index
    # as a target concept and produces a spurious "declares a different label
    # space" conflict. That detector is the frozen ranker's, so rather than
    # change it here, the more specific statement wins: a dataset that says it
    # has no label has not disagreed with the project about which label.
    declared_absence = _declared_target_absence(candidate)
    if declared_absence:
        route = _declared_derivation_route(candidate)
        if route:
            return TargetAnalysis(
                requirement=requirement,
                state=TargetState.DERIVABLE,
                layer=layer,
                evidence=f'the dataset states: "{route}"',
                detail=(
                    "The dataset declares the label is not provided and names what "
                    f"it would take to build it. Treat as {requirement!r} derivable, "
                    "not present."
                ),
            )
        return TargetAnalysis(
            requirement=requirement,
            state=TargetState.UNKNOWN,
            layer=layer,
            evidence=f'the dataset states: "{declared_absence}"',
            detail=(
                f"The dataset says no {requirement!r} label is published, but names "
                "no way to build one, so whether it can be derived is unresolved."
            ),
        )

    if requirement is None or factor is None:
        return TargetAnalysis(
            requirement=None,
            state=TargetState.UNKNOWN,
            layer=EvidenceLayer.UNKNOWN,
            evidence="the project named no target to check",
            detail="No prediction target was identified in the project requirements.",
        )

    # A model may *suggest* a target; it may not establish one. Every state
    # other than UNKNOWN is a claim about the dataset, so all of them require a
    # declaring layer. An AI candidate's match is kept in the evidence text as an
    # explicit suggestion, which is the difference between "AI-suggested: churn"
    # and "the dataset contains a churn label".
    if layer not in DECLARING_LAYERS:
        suggestion = verdict.via or f"the candidate's metadata names {requirement!r}"
        return TargetAnalysis(
            requirement=requirement,
            state=TargetState.UNKNOWN,
            layer=layer,
            evidence=f"AI-suggested only: {suggestion}",
            detail=(
                f"A generative model suggests this dataset provides {requirement!r}, "
                "but no external registry confirms it. Reported as unknown, not as "
                "a present target."
            ),
        )

    # A weak match (a semantic or inferred resolution) is outweighed by a declared
    # label-space contradiction on the same evidence: the detector compared the two
    # label spaces directly, whereas the weak match only found a shared word. An
    # exact or alias match names the target outright, so it still stands.
    weak_match = verdict.quality in (MatchQuality.SEMANTIC, MatchQuality.INFERRED)
    if (
        status is MatchStatus.MATCH
        and verdict.quality is MatchQuality.INFERRED
        and not declared_conflict
    ):
        # The weakest positive evidence the ranker accepts is a shared subject
        # area: real, but it does not say the dataset has *this* label. Reporting
        # it as PRESENT would claim a column nobody named.
        return TargetAnalysis(
            requirement=requirement,
            state=TargetState.UNKNOWN,
            layer=layer,
            evidence=(
                f"only a shared subject area links the dataset to {requirement!r}"
                + (f" ({verdict.via})" if verdict.via else "")
            ),
            detail=(
                f"The dataset covers a related subject but nothing in its metadata "
                f"names {requirement!r} as a label. This is an unknown, not a present target."
            ),
            match_quality=verdict.quality,
        )
    if status is MatchStatus.MATCH and not (declared_conflict and weak_match):
        return TargetAnalysis(
            requirement=requirement,
            state=TargetState.PRESENT,
            layer=layer,
            evidence=verdict.via or f"the dataset's own metadata names {requirement!r}",
            detail=(
                f"{requirement!r} is supported by "
                f"{'declared metadata' if layer in DECLARING_LAYERS else 'inferred metadata'}"
                + (f" ({verdict.quality.value})" if verdict.quality else "")
            ),
            match_quality=verdict.quality,
        )

    if declared_conflict or status is MatchStatus.MISMATCH:
        route = _declared_derivation_route(candidate)
        if route:
            return TargetAnalysis(
                requirement=requirement,
                state=TargetState.DERIVABLE,
                layer=layer,
                evidence=f'the dataset states: "{route}"',
                detail=(
                    "The dataset says the label is not stored and names how it "
                    f"could be built, so {requirement!r} is derivable rather than "
                    "present or incompatible."
                ),
            )
        if declared_conflict:
            return TargetAnalysis(
                requirement=requirement,
                state=TargetState.INCOMPATIBLE,
                layer=layer,
                evidence=declared_conflict[0],
                detail=(
                    f"The dataset's own metadata contradicts {requirement!r}: it declares "
                    "a different label space, not merely different wording."
                ),
            )
        # The ranker's MISMATCH says the declared target text does not match the
        # project's wording. That alone is a wording gap, not a contradiction:
        # "number of borrows per title" does not contain "borrow count" yet is exactly
        # that target, and calling every such case INCOMPATIBLE condemned the best
        # dataset in two benchmark cases.
        #
        # What separates a wording gap from a different outcome is shared vocabulary:
        # the wording gaps still share a content word ("borrow", "adverse"), whereas a
        # dataset declaring "readmission within 30 days" for a "mortality" project
        # shares none, and no dictionary equivalence was found either (the ranker
        # tried those before reporting MISMATCH). That combination is reported as
        # INCOMPATIBLE, worded as the lexical judgment it is.
        declared_target = (candidate.target_description or "").strip()
        wanted = set(value_tokens(requirement))
        shared = wanted & set(value_tokens(declared_target))
        if declared_target and wanted and not shared:
            return TargetAnalysis(
                requirement=requirement,
                state=TargetState.INCOMPATIBLE,
                layer=layer,
                evidence=(
                    f"the dataset declares its target as {declared_target!r}, which shares "
                    f"no words with {requirement!r}, and no dictionary equivalence links them"
                ),
                detail=(
                    f"The dataset declares a target that appears to be a different outcome "
                    f"from {requirement!r}. This is a vocabulary judgment, not a read of the "
                    "label column: confirm against the dataset's own label definition."
                ),
            )
        return TargetAnalysis(
            requirement=requirement,
            state=TargetState.UNKNOWN,
            layer=layer,
            evidence=(
                f"the dataset declares its target as {declared_target!r}, which does not "
                f"use the wording of {requirement!r}"
                if declared_target
                else verdict.detail or "no target evidence was published"
            ),
            detail=(
                f"The dataset's declared target is worded differently from {requirement!r}. "
                "That is not evidence they differ, so compatibility is unknown rather than "
                "incompatible; check the label definition before relying on it."
            ),
        )

    return TargetAnalysis(
        requirement=requirement,
        state=TargetState.UNKNOWN,
        layer=layer,
        evidence=verdict.detail or "no target evidence was published",
        detail=(
            f"Nothing in the dataset's metadata establishes whether {requirement!r} "
            "is available. This is an unknown, not a missing label."
        ),
    )


def _target_layer(candidate: DatasetCandidate) -> EvidenceLayer:
    """Layer for a *positive or negative* target claim.

    A target found in a registry candidate's prose is still declared: the
    registry published that description on purpose, which is the same authority
    as a dedicated ``target_description`` field and the same authority
    ``evidence.py`` gives it. Prose versus a named field changes how strongly
    *absence* can be concluded - a miss in prose is UNKNOWN, a miss in a declared
    field is a real absence - but it does not weaken a positive match, which
    cannot happen by accident.

    An AI candidate's target text is the model's own guess, so its layer is
    capped at AI_SUGGESTED and the model validator refuses to let it establish
    PRESENT, DERIVABLE or INCOMPATIBLE.
    """
    if candidate.source_type is SourceType.AI_SUGGESTED:
        return EvidenceLayer.AI_SUGGESTED
    if candidate.source_type is SourceType.VERIFIED_EXTERNAL:
        return EvidenceLayer.SOURCE_DECLARED
    return EvidenceLayer.INFERRED


def _target_text(candidate: DatasetCandidate) -> str:
    """The candidate's own statements about its target, in one string."""
    return " ".join(
        part
        for part in (candidate.target_description, candidate.description)
        if part
    )


def _declared_target_absence(candidate: DatasetCandidate) -> Optional[str]:
    """The phrase in which the candidate says its own label is absent, if any.

    An admission of absence is not a declaration of a *different* label, so this
    is checked before the contradiction detector: a registry that says "no
    target column is provided" is telling us the dataset has no label, not that
    it disagrees about which label. Reporting that as INCOMPATIBLE would tell
    the user their dataset actively disagrees with them, when in fact it simply
    has nothing to offer.
    """
    text = _target_text(candidate)
    match = _TARGET_NOT_PROVIDED.search(text)
    return match.group(0).strip() if match else None


def _declared_derivation_route(candidate: DatasetCandidate) -> Optional[str]:
    """A phrase where the candidate says how a label could be built, if any.

    "No label provided" is an admission of absence. "A label can be derived by
    thresholding the recorded values" is an admission of absence *plus* a
    construction route. Only the second supports ``DERIVABLE``; conflating them
    would let any dataset that mentions a missing column be reported as
    derivable, which is the claim the spec warns against.
    """
    text = _target_text(candidate)
    for match in _DERIVATION_ROUTE.finditer(text):
        sentence = _sentence_around(text, match)
        # Require the route to be about producing a label or target, not merely
        # about some other computation the dataset mentions.
        if _DERIVATION_SUBJECT.search(sentence):
            return sentence.strip()
    return None


def _sentence_around(text: str, match: re.Match[str]) -> str:
    """The sentence containing ``match``, so the quoted evidence reads cleanly."""
    start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start()))
    end_candidates = [
        index
        for index in (text.find(".", match.end()), text.find("\n", match.end()))
        if index != -1
    ]
    end = min(end_candidates) if end_candidates else len(text)
    return text[start + 1 : end] if start != -1 else text[:end]


def _first_target_requirement(
    candidate: DatasetCandidate,
) -> Optional[RequirementMatch]:
    verdicts = _requirements(candidate, RankingFactorName.TARGET_MATCH)
    return verdicts[0] if verdicts else None


def _feature_coverage(candidate: DatasetCandidate, schema_state: SchemaState) -> FeatureCoverage:
    """Partition required features into matched / unknown / missing.

    The bucket a feature lands in depends on the schema state, which is the whole
    point: the ranker records ``MISMATCH`` for a feature that a registry-published
    schema does not contain, and ``UNKNOWN`` when nobody published one. Here that
    ``MISMATCH`` is relabelled ``MISSING`` only when the schema can actually prove
    absence, and stays ``UNKNOWN`` otherwise.
    """
    can_prove_absence = schema_state in {SchemaState.VERIFIED, SchemaState.PARTIAL}
    matched: list[FeatureCompatibility] = []
    unknown: list[FeatureCompatibility] = []
    missing: list[FeatureCompatibility] = []
    seen: set[str] = set()

    for verdict in _requirements(candidate, RankingFactorName.FEATURE_MATCH):
        key = verdict.term.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)

        if verdict.status is MatchStatus.MATCH and verdict.quality is not None:
            state = _STATES_FOR_QUALITY[verdict.quality]
            matched.append(
                FeatureCompatibility(
                    requirement=verdict.term,
                    state=state,
                    layer=_match_layer(candidate, verdict),
                    via=verdict.via,
                    detail=(
                        f"{verdict.term!r} is supported by a {state.value} match"
                        + (f" ({verdict.via})" if verdict.via else "")
                    ),
                )
            )
        elif verdict.status is MatchStatus.MISMATCH and can_prove_absence:
            missing.append(
                FeatureCompatibility(
                    requirement=verdict.term,
                    state=FeatureMatchState.MISSING,
                    layer=EvidenceLayer.INFERRED
                    if schema_state is SchemaState.PARTIAL
                    else EvidenceLayer.SCHEMA_VERIFIED,
                    detail=(
                        f"{verdict.term!r} is absent from a "
                        f"{schema_state.value} description of the dataset's own "
                        "columns"
                    ),
                )
            )
        else:
            # Either the ranker said UNKNOWN, or it said MISMATCH against a
            # schema nobody could see. Both are unknown: absence of evidence is
            # not evidence of absence.
            unknown.append(
                FeatureCompatibility(
                    requirement=verdict.term,
                    state=FeatureMatchState.UNKNOWN,
                    layer=EvidenceLayer.UNKNOWN
                    if schema_state is SchemaState.UNKNOWN
                    else EvidenceLayer.INFERRED,
                    detail=(
                        f"no trustworthy evidence establishes whether {verdict.term!r} "
                        "is present"
                    ),
                )
            )

    ordered = sorted(matched, key=lambda f: f.requirement.lower())
    return FeatureCoverage(
        required_count=len(ordered) + len(unknown) + len(missing),
        matched=ordered,
        unknown=sorted(unknown, key=lambda f: f.requirement.lower()),
        missing=sorted(missing, key=lambda f: f.requirement.lower()),
    )


def _match_layer(candidate: DatasetCandidate, verdict: RequirementMatch) -> EvidenceLayer:
    """Provenance of one confirmed feature match.

    An ``INFERRED``-quality match is weak evidence by definition: it reached the
    feature without naming it. Everything else the ranker accepted came from the
    candidate's own published text.
    """
    if verdict.quality is MatchQuality.INFERRED:
        return (
            EvidenceLayer.AI_SUGGESTED
            if candidate.source_type is SourceType.AI_SUGGESTED
            else EvidenceLayer.INFERRED
        )
    if candidate.source_type is SourceType.AI_SUGGESTED:
        return EvidenceLayer.AI_SUGGESTED
    return EvidenceLayer.SOURCE_DECLARED


#: How a contradiction dimension in ``compatibility.py`` maps onto the
#: compatibility dimensions this module reports. Reusing the existing detector
#: rather than re-deriving mismatches keeps one answer to "does this dataset
#: contradict the project", so ranking and compatibility cannot disagree.
_CONTRADICTION_DIMENSIONS: dict[str, str] = {
    "domain": "domain",
    "ml_task": "task",
    "target": "target",
    "modality": "data_type",
}


def contradictions_by_dimension(
    candidate: DatasetCandidate, requirements: Optional[ProjectAnalysisData]
) -> dict[str, list[str]]:
    """Re-run the existing contradiction detector, keeping the dimension label.

    The ranker stores only the human-readable messages on its compatibility
    factor, which loses which dimension each one belongs to. Re-calling
    ``find_contradictions`` with the ranker's own inputs recovers that label
    without reimplementing a single detection rule - and because the inputs and
    the function are shared, the answer is identical by construction.

    Returns an empty mapping for an AI suggestion, which the detector declines to
    judge at all.
    """
    if requirements is None or candidate.source_type is not SourceType.VERIFIED_EXTERNAL:
        return {}
    profile = build_project_profile(requirements)
    found = find_contradictions(
        candidate,
        project_domain=profile.project_domain_key,
        project_tasks=profile.task_terms,
        project_targets=profile.target_labels,
        project_data_types=profile.data_type_notes,
        candidate_domains=DatasetRankingEngine._candidate_domains(candidate),
        target_index=get_target_concept_index(),
    )
    out: dict[str, list[str]] = {}
    for contradiction in found:
        dimension = _CONTRADICTION_DIMENSIONS.get(contradiction.dimension)
        if dimension:
            out.setdefault(dimension, []).append(contradiction.message)
    return {key: sorted(value) for key, value in sorted(out.items())}


def _target_dimension(
    candidate: DatasetCandidate,
    analysis: TargetAnalysis,
    contradictions: Optional[dict[str, list[str]]] = None,
) -> DimensionCompatibility:
    """Target compatibility, from the richer target analysis rather than the verdict.

    The ranker's target verdict has three outcomes, but the target analysis has
    four: it separates a target that is *positively incompatible* from one that
    is merely *not there yet*. That distinction matters at this level, because a
    derivable target is usable with engineering work while an incompatible one is
    not, and reporting both as MISMATCH would tell the user to skip a dataset
    that can serve them. A derivable target is therefore UNKNOWN here - the
    requirement is not currently met - and the reason travels in the evidence and
    lifts the overall level to MEDIUM rather than LOW.
    """
    declared = (contradictions or {}).get("target") or []
    if analysis.state is TargetState.INCOMPATIBLE:
        return DimensionCompatibility(
            dimension="target",
            status=MatchStatus.MISMATCH,
            evidence=declared[0] if declared else analysis.evidence,
            layer=analysis.layer,
        )
    if analysis.state is TargetState.PRESENT:
        return DimensionCompatibility(
            dimension="target",
            status=MatchStatus.MATCH,
            evidence=analysis.evidence,
            layer=analysis.layer,
        )
    return DimensionCompatibility(
        dimension="target",
        status=MatchStatus.UNKNOWN,
        evidence=analysis.evidence,
        layer=analysis.layer,
    )


def _dimension(
    candidate: DatasetCandidate,
    factor_name: RankingFactorName,
    dimension: str,
    contradictions: Optional[dict[str, list[str]]] = None,
) -> DimensionCompatibility:
    """Verdict for one dimension, from the ranker's own verdicts plus contradictions.

    A contradiction is the strongest signal available: it means the candidate's
    published metadata positively states something incompatible. Absent one, the
    per-requirement verdicts decide. Absent both, the dimension is UNKNOWN - not
    MISMATCH, because a candidate that merely mentions nothing relevant has not
    declared itself incompatible.
    """
    declared = (contradictions or {}).get(dimension) or []

    factor = _factor(candidate, factor_name)
    verdicts = _requirements(candidate, factor_name)

    if declared:
        return DimensionCompatibility(
            dimension=dimension,
            status=MatchStatus.MISMATCH,
            evidence=declared[0],
            layer=EvidenceLayer.SOURCE_DECLARED,
        )

    if factor is None:
        return DimensionCompatibility(
            dimension=dimension,
            status=MatchStatus.UNKNOWN,
            evidence="this dimension was not evaluated",
            layer=EvidenceLayer.UNKNOWN,
        )

    # The terms of one dimension are alternatives, not a checklist: a project that
    # lists "classification, regression" is served by a dataset supporting either,
    # and a domain plus its subdomain are two views of one subject. So any MATCH
    # settles the dimension, exactly as the contradiction detector treats an
    # overlapping task list; MISMATCH applies only when nothing matched. (Features
    # are different - each is a separate requirement - and have their own coverage.)
    statuses = {v.status for v in verdicts}
    if MatchStatus.MATCH in statuses:
        status = MatchStatus.MATCH
    elif MatchStatus.MISMATCH in statuses:
        status = MatchStatus.MISMATCH
    else:
        status = MatchStatus.UNKNOWN

    evidence = factor.detail
    if not evidence and verdicts and verdicts[0].detail:
        evidence = verdicts[0].detail
    if factor.contradictions and status is MatchStatus.MISMATCH:
        evidence = factor.contradictions[0]
    return DimensionCompatibility(
        dimension=dimension,
        status=status,
        evidence=evidence or "no evidence recorded",
        layer=_candidate_layer(candidate),
    )


#: A project needs the time dimension when its own task, target or concepts say
#: so. Plain words, not concept ids: this is about the project's wording.
_TEMPORAL_PROJECT = re.compile(
    r"\b(?:forecast\w*|time[- ]?series|temporal|sequence prediction|trend prediction)\b",
    re.IGNORECASE,
)

#: Evidence in a dataset's own published text that it has a time axis. Deliberately
#: specific: a bare "date" would match every "last updated" sentence.
_TEMPORAL_EVIDENCE = re.compile(
    r"\b(?:time[- ]?series|timestamps?|date[- ]?time|hourly|daily|weekly|monthly|"
    r"quarterly|yearly|annual|historical|temporal|over time)\b",
    re.IGNORECASE,
)


def _project_needs_temporal(profile) -> bool:
    if profile is None:
        return False
    text = " ".join([*profile.task_terms, *profile.target_terms, *profile.concept_terms])
    return bool(_TEMPORAL_PROJECT.search(text))


def _data_type_dimension(
    candidate: DatasetCandidate,
    profile,
    contradictions: Optional[dict[str, list[str]]] = None,
) -> DimensionCompatibility:
    """Data modality: the project's stated type against published file extensions.

    Only published facts decide it. A file extension is something the registry
    states about the dataset, so ``.csv`` against a "tabular" project is a MATCH and
    ``.png`` only is a MISMATCH. Not knowing the files, or the project stating no
    data type, is UNKNOWN: it is not a contradiction.
    """
    declared = (contradictions or {}).get("data_type") or []
    if declared:
        return DimensionCompatibility(
            dimension="data_type",
            status=MatchStatus.MISMATCH,
            evidence=declared[0],
            layer=EvidenceLayer.SOURCE_DECLARED,
        )
    required = _project_modalities(profile.data_type_notes) if profile else frozenset()
    if not required:
        return DimensionCompatibility(
            dimension="data_type",
            status=MatchStatus.UNKNOWN,
            evidence="the project did not state a data type",
            layer=EvidenceLayer.UNKNOWN,
        )
    offered = _candidate_modalities(candidate)
    if not offered:
        return DimensionCompatibility(
            dimension="data_type",
            status=MatchStatus.UNKNOWN,
            evidence="no file types were published, so the data modality is unknown",
            layer=EvidenceLayer.UNKNOWN,
        )
    if offered & required:
        files = ", ".join(candidate.file_types) or ", ".join(candidate.file_names[:3])
        # The wording follows the authority: a registry publishes files, a model
        # only suggests them, and the sentence must not state a suggestion as fact.
        verb = "publishes" if _publishes(candidate) else "is suggested by a model to have"
        return DimensionCompatibility(
            dimension="data_type",
            status=MatchStatus.MATCH,
            evidence=(
                f"{verb} {', '.join(sorted(offered & required))} files ({files})"
            ),
            layer=(
                EvidenceLayer.SOURCE_DECLARED
                if _publishes(candidate)
                else EvidenceLayer.AI_SUGGESTED
            ),
        )
    # Published modality with no overlap. The contradiction detector reports it for
    # a verified candidate; for an unverified one the file claim is a model's guess.
    return DimensionCompatibility(
        dimension="data_type",
        status=MatchStatus.UNKNOWN,
        evidence=(
            f"the files suggest {', '.join(sorted(offered))} data, but a model "
            "supplied that and it cannot be treated as a published fact"
        ),
        layer=EvidenceLayer.AI_SUGGESTED,
    )


def _temporal_dimension(candidate: DatasetCandidate, profile) -> Optional[DimensionCompatibility]:
    """Whether a forecasting project can see a time axis in the dataset's own text.

    ``None`` when the project has no temporal requirement, so a dimension nobody
    asked about is never reported as unknown. Never MISMATCH: a dataset that does
    not mention time has not said it lacks it.
    """
    if not _project_needs_temporal(profile):
        return None
    text = " ".join(
        part
        for part in (
            candidate.name,
            candidate.description,
            " ".join(candidate.tags),
            candidate.feature_description,
            " ".join(candidate.file_names),
        )
        if part
    )
    found = _TEMPORAL_EVIDENCE.search(text)
    if not found:
        return DimensionCompatibility(
            dimension="temporal",
            status=MatchStatus.UNKNOWN,
            evidence="nothing published says whether the data has a time axis",
            layer=EvidenceLayer.UNKNOWN,
        )
    source = "the dataset's own text" if _publishes(candidate) else "a model's description"
    return DimensionCompatibility(
        dimension="temporal",
        status=MatchStatus.MATCH,
        evidence=f"{source} mentions {found.group(0)!r}",
        layer=(
            EvidenceLayer.SOURCE_DECLARED
            if _publishes(candidate)
            else EvidenceLayer.AI_SUGGESTED
        ),
    )


def _required_dimensions(profile) -> frozenset[str]:
    """Dimensions the project actually asked about.

    Without requirements nothing can be told apart, so the three always-present
    dimensions are assumed. A dimension the project never stated is never reported
    as unknown: an unanswered question nobody asked is not a gap.
    """
    if profile is None:
        return frozenset({"domain", "task", "target"})
    required: set[str] = set()
    if profile.project_domain_key or profile.domain_label:
        required.add("domain")
    if profile.task_terms:
        required.add("task")
    if profile.target_terms or profile.raw_target:
        required.add("target")
    if _project_modalities(profile.data_type_notes):
        required.add("data_type")
    if _project_needs_temporal(profile):
        required.add("temporal")
    return frozenset(required)


def _overall_level(
    target_state: TargetState,
    coverage: FeatureCoverage,
    domain: MatchStatus,
    task: MatchStatus,
    contradictions: list[str],
    schema_state: SchemaState,
) -> CompatibilityLevel:
    """One of HIGH, MEDIUM, LOW or UNKNOWN - never a percentage.

    A stated incompatibility decides the answer on its own, because no amount of
    feature overlap makes a dataset whose label space contradicts the project
    usable. MEDIUM is "supported, but with work" and covers both a target that
    must be derived and requirements that cannot be shown either way. HIGH is only
    reachable when every required feature is positively matched, the target is
    present and nothing is unknown: an unproven requirement keeps a record out of
    HIGH, rather than a penalty being applied afterwards. A dataset nothing is
    known about is UNKNOWN, not LOW: LOW is a statement about the dataset, UNKNOWN
    is a statement about our knowledge of it.
    """
    if domain is MatchStatus.MISMATCH or task is MatchStatus.MISMATCH:
        return CompatibilityLevel.LOW
    if contradictions:
        return CompatibilityLevel.LOW

    undecidable = (
        target_state is TargetState.UNKNOWN
        or coverage.unknown_count > 0
        or schema_state is SchemaState.UNKNOWN
    )
    # A feature absent from a *structured* schema is a real gap. One absent from a
    # prose column list is only "not named there", because such lists are often
    # partial, so it keeps the dataset out of HIGH without condemning it as LOW.
    structured_gap = coverage.missing_count > 0 and schema_state is SchemaState.VERIFIED
    undecidable = undecidable or (coverage.missing_count > 0 and not structured_gap)
    blocking = target_state is TargetState.INCOMPATIBLE or structured_gap
    if not undecidable and not blocking:
        return CompatibilityLevel.HIGH
    if coverage.required_count == 0 and target_state is TargetState.UNKNOWN:
        # Nothing was ever checkable, so nothing supports a grade.
        return CompatibilityLevel.UNKNOWN
    if blocking:
        return CompatibilityLevel.LOW
    if coverage.required_count and coverage.unknown_count == coverage.required_count:
        return CompatibilityLevel.UNKNOWN
    return CompatibilityLevel.MEDIUM


def _risks(
    candidate: DatasetCandidate,
    schema_state: SchemaState,
    target_state: TargetState,
    coverage: FeatureCoverage,
    layer: EvidenceLayer,
    domain: MatchStatus,
    task: MatchStatus,
) -> list[DatasetRisk]:
    """Only risks the current metadata actually supports.

    Every ``RiskCategory`` outside ``EMITTABLE_RISK_CATEGORIES`` needs the raw
    file - class counts, null rates, row-level duplicates - and is deliberately
    absent here rather than guessed at.
    """
    risks: list[DatasetRisk] = []

    def add(
        category: RiskCategory,
        severity: RiskSeverity,
        detail: str,
        subject: Optional[str] = None,
        risk_layer: EvidenceLayer = layer,
    ) -> None:
        if category not in EMITTABLE_RISK_CATEGORIES:  # pragma: no cover - guard
            raise ValueError(f"{category.value} needs raw dataset profiling")
        risks.append(
            DatasetRisk(
                category=category,
                severity=severity,
                detail=detail,
                subject=subject,
                layer=risk_layer,
            )
        )

    if candidate.source_type is SourceType.AI_SUGGESTED:
        add(
            RiskCategory.AI_UNVERIFIED,
            RiskSeverity.CRITICAL,
            (
                "This dataset was suggested by a generative model. Its existence "
                "and metadata have not been confirmed by any external registry."
            ),
            risk_layer=EvidenceLayer.AI_SUGGESTED,
        )
    if schema_state is SchemaState.UNKNOWN:
        add(
            RiskCategory.SCHEMA_INCOMPLETE,
            RiskSeverity.WARNING,
            (
                "No trustworthy schema information is available, so required "
                "features can be neither confirmed nor ruled out."
            ),
        )
    elif schema_state is SchemaState.PARTIAL:
        add(
            RiskCategory.SCHEMA_INCOMPLETE,
            RiskSeverity.INFO,
            (
                "Schema information comes from published prose, not from "
                "verifying the dataset's own columns."
            ),
        )
    if target_state is TargetState.UNKNOWN:
        add(
            RiskCategory.TARGET_UNKNOWN,
            RiskSeverity.WARNING,
            "The project's target could not be established from the available evidence.",
        )
    elif target_state is TargetState.DERIVABLE:
        add(
            RiskCategory.TARGET_DERIVATION_REQUIRED,
            RiskSeverity.WARNING,
            (
                "The target is not stored in this dataset; it must be derived "
                "before modelling."
            ),
        )
    elif target_state is TargetState.INCOMPATIBLE:
        add(
            RiskCategory.TARGET_MISMATCH,
            RiskSeverity.CRITICAL,
            "The dataset's declared target contradicts the project's target.",
        )
    for feature in coverage.unknown:
        add(
            RiskCategory.FEATURE_UNKNOWN,
            RiskSeverity.INFO,
            f"Could not determine whether the required feature {feature.requirement!r} is present.",
            subject=feature.requirement,
            risk_layer=feature.layer,
        )
    for feature in coverage.missing:
        add(
            RiskCategory.FEATURE_MISSING,
            RiskSeverity.WARNING,
            (
                f"The required feature {feature.requirement!r} is absent from the "
                "registry's published schema."
                if schema_state is SchemaState.VERIFIED
                else f"The required feature {feature.requirement!r} is not named in the "
                "dataset's own column description, which may be incomplete."
            ),
            subject=feature.requirement,
            risk_layer=feature.layer,
        )
    if domain is MatchStatus.MISMATCH:
        add(
            RiskCategory.DOMAIN_MISMATCH,
            RiskSeverity.WARNING,
            "The dataset's subject matter conflicts with the project's.",
        )
    if task is MatchStatus.MISMATCH:
        add(
            RiskCategory.TASK_MISMATCH,
            RiskSeverity.WARNING,
            "The dataset declares only different ML tasks than the project requires.",
        )
    return risks


def _limitations(
    candidate: DatasetCandidate,
    schema_state: SchemaState,
    target_state: TargetState,
    coverage: FeatureCoverage,
    layer: EvidenceLayer,
) -> list[DatasetLimitation]:
    """Evidence-backed caveats. Nothing here is speculative."""
    out: list[DatasetLimitation] = []

    def add(code: str, detail: str, limitation_layer: EvidenceLayer = layer) -> None:
        out.append(DatasetLimitation(code=code, detail=detail, layer=limitation_layer))

    if schema_state is SchemaState.UNKNOWN:
        add(
            "schema_not_available",
            "The dataset's schema is not publicly available, so its columns could not be checked.",
            EvidenceLayer.UNKNOWN,
        )
    elif schema_state is SchemaState.PARTIAL:
        add(
            "schema_from_prose",
            "Column information comes from published prose rather than a verified schema.",
            EvidenceLayer.INFERRED,
        )
    if target_state is TargetState.UNKNOWN:
        add(
            "target_not_verified",
            "The project target could not be verified against this dataset.",
            EvidenceLayer.UNKNOWN,
        )
    elif target_state is TargetState.DERIVABLE:
        add(
            "target_requires_derivation",
            "The target is not stored in the dataset and must be derived.",
        )
    if coverage.unknown_count:
        add(
            "features_unverifiable",
            f"{coverage.unknown_count} required "
            f"{'feature' if coverage.unknown_count == 1 else 'features'} could not be "
            "verified either way.",
            EvidenceLayer.UNKNOWN,
        )
    if (
        candidate.source_type is SourceType.VERIFIED_EXTERNAL
        and candidate.metadata_depth is MetadataDepth.LISTING
    ):
        add(
            "listing_metadata_only",
            "Only the registry's search listing was read, so the owner's full "
            "description and file list were not considered.",
            EvidenceLayer.UNKNOWN,
        )
    if candidate.source_type is SourceType.AI_SUGGESTED:
        add(
            "ai_generated_candidate",
            "This candidate was generated by a model and has not been externally verified.",
            EvidenceLayer.AI_SUGGESTED,
        )
    if target_state is TargetState.PRESENT and schema_state is not SchemaState.VERIFIED:
        add(
            "target_definition_unconfirmed",
            "The target appears compatible, but its exact definition could not be "
            "confirmed against a verified schema.",
            EvidenceLayer.INFERRED,
        )
    return out


def build_recommendation_explanation(
    *,
    candidate: DatasetCandidate,
    layer: EvidenceLayer,
    schema_state: SchemaState,
    target: TargetAnalysis,
    coverage: FeatureCoverage,
    domain: DimensionCompatibility,
    task: DimensionCompatibility,
    risks: Sequence[DatasetRisk],
    limitations: Sequence[DatasetLimitation],
    data_type: Optional[DimensionCompatibility] = None,
    temporal: Optional[DimensionCompatibility] = None,
    required: frozenset[str] = frozenset(),
) -> RecommendationExplanation:
    """Render the structured explanation from typed state.

    Every line is derived from a state computed above. There is no path by which
    prose can assert a fact the evidence does not carry: a feature appears under
    ``what_matches`` only when its state is one of the four satisfied states, and
    under ``what_is_missing`` only when its state is MISSING.
    """
    target_state = target.state
    target_name = target.requirement or "the project target"

    why: list[str] = []
    if domain.status is MatchStatus.MATCH:
        why.append(f"Strong {domain.evidence.split(',')[0]} domain match")
    if target_state is TargetState.PRESENT:
        why.append(f"Compatible target: {target_name}")
    elif target_state is TargetState.DERIVABLE:
        why.append(f"The project target ({target_name}) can be derived from published fields")
    if task.status is MatchStatus.MATCH:
        why.append(f"Suitable prediction task ({task.evidence})")
    satisfied = [f.requirement for f in coverage.matched]
    if satisfied:
        listed = ", ".join(satisfied[:5])
        more = f" and {len(satisfied) - 5} more" if len(satisfied) > 5 else ""
        why.append(f"Required features supported: {listed}{more}")
    if not why:
        why.append("No requirement could be confirmed from the available evidence")

    matches = [f.detail for f in coverage.matched]
    if domain.status is MatchStatus.MATCH:
        matches.append(f"Domain: {domain.evidence}")
    if task.status is MatchStatus.MATCH:
        matches.append(f"ML task: {task.evidence}")
    for extra, label in ((data_type, "Data type"), (temporal, "Time dimension")):
        if extra is not None and extra.status is MatchStatus.MATCH:
            matches.append(f"{label}: {extra.evidence}")

    coverage_lines = [f"Features: {coverage.summary()}"]
    coverage_lines.append(
        f"Target: {target_state.value}"
        + (f" ({target.match_quality.value} match)" if target.match_quality else "")
    )
    for dimension, label in (
        (domain, "Domain"),
        (task, "ML task"),
        (data_type, "Data type"),
        (temporal, "Time dimension"),
    ):
        if dimension is None:
            continue
        if dimension.dimension in required or dimension.status is not MatchStatus.UNKNOWN:
            coverage_lines.append(f"{label}: {dimension.status.value}")

    unknown = [f.detail for f in coverage.unknown]
    if target_state is TargetState.UNKNOWN:
        unknown.append(f"Target: nothing establishes whether {target_name} is available")
    for extra, label in ((data_type, "Data type"), (temporal, "Time dimension")):
        if (
            extra is not None
            and extra.dimension in required
            and extra.status is MatchStatus.UNKNOWN
        ):
            unknown.append(f"{label}: {extra.evidence}")

    verified = [
        f"{f.requirement} ({f.state.value})"
        for f in coverage.matched
        if f.layer in DECLARING_LAYERS
    ]
    inferred = [
        f"{f.requirement} ({f.state.value})"
        for f in coverage.matched
        if f.layer not in DECLARING_LAYERS
    ]
    if target_state is TargetState.PRESENT:
        # Only a match that names the target is "verified"; a dictionary-equivalent
        # match is DataPilot's own reasoning over published text, so it is inferred.
        if target.match_quality in (MatchQuality.EXACT, MatchQuality.ALIAS):
            verified.append(f"Target {target_name} named in the dataset's own metadata")
        else:
            inferred.append(
                f"Target {target_name} matched through a dictionary equivalent"
                + (f" ({target.evidence})" if target.evidence else "")
            )
    if layer is EvidenceLayer.AI_SUGGESTED:
        inferred.append("Candidate metadata is model-generated, not externally published")
    claims = candidate.ai_suggested_claims
    if claims is not None:
        inferred.append(
            f"A model also made unconfirmed claims about this dataset ({claims.source}); "
            "none were used as evidence"
        )

    return RecommendationExplanation(
        why_recommended=why,
        requirement_coverage=coverage_lines,
        what_matches=matches,
        what_is_unknown=unknown,
        what_is_missing=[f.detail for f in coverage.missing],
        what_is_verified=verified,
        what_is_inferred=inferred,
        risks=[f"{r.category.value}: {r.detail}" for r in risks],
        limitations=[f"{lim.code}: {lim.detail}" for lim in limitations],
    )


def build_dataset_intelligence(
    candidate: DatasetCandidate,
    requirements: Optional[ProjectAnalysisData] = None,
) -> DatasetIntelligence:
    """Assemble the intelligence record for one already-ranked candidate.

    Requires ``candidate`` to have been through ``DatasetRankingEngine.rank``:
    the verdicts are read from its ranking factors rather than recomputed, so
    compatibility and ranking cannot disagree.

    ``requirements`` is optional and only widens the result: it lets the existing
    contradiction detector be re-run so each contradiction can be attributed to a
    dimension. Without it, dimension mismatches fall back to the per-requirement
    verdicts, which is coarser but never wrong.
    """
    layer = _candidate_layer(candidate)
    claims = _evidence_claims(candidate)
    schema_state = _schema_state(candidate)
    coverage = _feature_coverage(candidate, schema_state)

    # Contradictions are resolved first: the target analysis needs them, because a
    # declared label-space conflict outranks a keyword match on the same evidence.
    by_dimension = contradictions_by_dimension(candidate, requirements)
    target_state_analysis = _target_analysis(candidate, by_dimension)

    domain = _dimension(
        candidate, RankingFactorName.DOMAIN_RELEVANCE, "domain", by_dimension
    )
    task = _dimension(candidate, RankingFactorName.TASK_MATCH, "task", by_dimension)
    target = _target_dimension(candidate, target_state_analysis, by_dimension)
    profile = build_project_profile(requirements) if requirements is not None else None
    data_type = _data_type_dimension(candidate, profile, by_dimension)
    temporal = _temporal_dimension(candidate, profile)
    required = _required_dimensions(profile)

    compatibility_factor = _factor(candidate, RankingFactorName.COMPATIBILITY)
    contradictions = list(compatibility_factor.contradictions) if compatibility_factor else []

    unknown_dimensions = sorted(
        dimension.dimension
        for dimension in (domain, task, target, data_type, temporal)
        if dimension is not None
        and dimension.dimension in required
        and dimension.status is MatchStatus.UNKNOWN
    )

    overall = _overall_level(
        target_state_analysis.state,
        coverage,
        domain.status,
        task.status,
        contradictions,
        schema_state,
    )
    risks = _risks(
        candidate,
        schema_state,
        target_state_analysis.state,
        coverage,
        layer,
        domain.status,
        task.status,
    )
    limitations = _limitations(
        candidate, schema_state, target_state_analysis.state, coverage, layer
    )

    compatibility = CompatibilityAssessment(
        overall=overall,
        domain=domain.status,
        task=task.status,
        target=target.status,
        data_type=data_type.status,
        temporal=temporal.status if temporal is not None else None,
        feature_coverage=coverage,
        contradictions=sorted(contradictions),
        unknown_dimensions=unknown_dimensions,
    )

    explanation = build_recommendation_explanation(
        candidate=candidate,
        layer=layer,
        schema_state=schema_state,
        target=target_state_analysis,
        coverage=coverage,
        domain=domain,
        task=task,
        risks=risks,
        limitations=limitations,
        data_type=data_type,
        temporal=temporal,
        required=required,
    )

    # The record's own layer must not overstate its claims, so it is derived from
    # the claims rather than from the candidate.
    record_layer = strongest_layer([claim.layer for claim in claims] + [layer])

    return DatasetIntelligence(
        candidate_id=candidate.id,
        evidence_layer=record_layer,
        evidence=claims,
        schema_state=schema_state,
        target_analysis=target_state_analysis,
        feature_coverage=coverage,
        compatibility=compatibility,
        risks=risks,
        limitations=limitations,
        explanation=explanation,
    )


def build_intelligence_report(
    candidates: Sequence[DatasetCandidate],
    requirements: Optional[ProjectAnalysisData] = None,
) -> list[DatasetIntelligence]:
    """Intelligence for a ranked list, in the caller's order."""
    return [
        build_dataset_intelligence(candidate, requirements) for candidate in candidates
    ]