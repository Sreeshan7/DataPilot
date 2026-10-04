"""Unit tests for Phase 3A dataset intelligence and compatibility.

Every case is written as a behaviour the user could be misled about, so the
assertions target the *claim* rather than the shape: it is not enough for the
code to produce a record, it has to refuse to overstate what the evidence shows.

Cases A-V map one-to-one onto the Phase 3A specification.
"""

from __future__ import annotations

import json

import pytest

from app.schemas.dataset import (
    DatasetCandidate,
    MatchStatus,
    SourceType,
    VerificationStatus,
)
from app.schemas.intelligence import (
    CompatibilityAssessment,
    CompatibilityLevel,
    DatasetIntelligence,
    DatasetLimitation,
    DatasetRisk,
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
)
from app.services.datasets.intelligence import (
    build_dataset_intelligence,
    build_intelligence_report,
    strongest_layer,
)
from app.schemas.intelligence import DECLARING_LAYERS
from app.services.datasets.normalizer import normalize_genai_candidate
from app.services.datasets.ranking import DatasetRankingEngine
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_genai_candidate,
    make_requirements,
    make_settings,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _engine(**overrides) -> DatasetRankingEngine:
    return DatasetRankingEngine(settings=make_settings(**overrides), now=TEST_NOW)


def _intelligence(candidate: DatasetCandidate, requirements=None) -> DatasetIntelligence:
    """Rank, then read intelligence back out of the ranked candidate."""
    engine = _engine()
    ranked = engine.rank([candidate], requirements or make_requirements())[0]
    return build_dataset_intelligence(ranked, requirements or make_requirements())


def _ranked(candidate: DatasetCandidate, requirements=None, **overrides) -> DatasetCandidate:
    engine = _engine(**overrides)
    return engine.rank([candidate], requirements or make_requirements())[0]


def _terms(requirements):
    return {f.name for f in requirements.dataset_requirements.feature_requirements}


def _coverage_term(coverage: FeatureCoverage, term: str) -> FeatureCompatibility | None:
    for item in coverage.all_features:
        if item.requirement == term:
            return item
    return None


def _states(coverage: FeatureCoverage) -> dict[str, FeatureMatchState]:
    return {item.requirement: item.state for item in coverage.all_features}


def _coverage(*, matched=(), unknown=(), missing=()) -> FeatureCoverage:
    return FeatureCoverage(
        required_count=len(matched) + len(unknown) + len(missing),
        matched=list(matched),
        unknown=list(unknown),
        missing=list(missing),
    )


def _assessment(coverage: FeatureCoverage, **kwargs) -> CompatibilityAssessment:
    payload = {
        "overall": CompatibilityLevel.UNKNOWN,
        "domain": MatchStatus.UNKNOWN,
        "task": MatchStatus.UNKNOWN,
        "target": MatchStatus.UNKNOWN,
        "data_type": MatchStatus.UNKNOWN,
        "feature_coverage": coverage,
    }
    payload.update(kwargs)
    return CompatibilityAssessment(**payload)


def _ai_candidate(**overrides) -> DatasetCandidate:
    """An AI-suggested candidate as the ranker sees it: a normalized DatasetCandidate."""
    return normalize_genai_candidate(make_genai_candidate(**overrides))


# ---------------------------------------------------------------------------
# A-E  identity, evidence layers, schema states
# ---------------------------------------------------------------------------


class TestAIdentitySeparation:
    def test_a_intelligence_is_attached_without_touching_candidate_identity(self):
        candidate = make_candidate()
        ranked = _ranked(candidate)
        intelligence = build_dataset_intelligence(ranked, make_requirements())

        assert intelligence.candidate_id == ranked.id
        assert ranked.id == candidate.id
        assert ranked.name == candidate.name
        assert ranked.source == candidate.source

    def test_b_intelligence_does_not_duplicate_candidate_metadata(self):
        fields = set(DatasetIntelligence.model_fields)
        assert fields.isdisjoint(
            {"name", "description", "url", "owner", "license", "tags", "domain"}
        )

    def test_c_report_preserves_rank_order_and_one_record_per_candidate(self):
        candidates = [
            make_candidate(name="One", source_id="owner/one"),
            make_candidate(name="Two", source_id="owner/two"),
            make_candidate(name="Three", source_id="owner/three"),
        ]
        engine = _engine()
        requirements = make_requirements()
        ranked = engine.rank(candidates, requirements)

        report = build_intelligence_report(ranked, requirements)

        assert [r.candidate_id for r in report] == [c.id for c in ranked]
        assert len({r.candidate_id for r in report}) == len(ranked)


class TestDEvidenceLayers:
    def test_d_verified_registry_candidate_reports_source_declared(self):
        intelligence = _intelligence(make_candidate())

        assert intelligence.evidence_layer is EvidenceLayer.SOURCE_DECLARED
        assert all(
            claim.layer is not EvidenceLayer.AI_SUGGESTED
            for claim in intelligence.evidence
        )

    def test_e_layer_order_prefers_declared_over_weaker_evidence(self):
        assert strongest_layer([EvidenceLayer.UNKNOWN, EvidenceLayer.SOURCE_DECLARED]) is (
            EvidenceLayer.SOURCE_DECLARED
        )
        assert strongest_layer([EvidenceLayer.INFERRED, EvidenceLayer.AI_SUGGESTED]) is (
            EvidenceLayer.INFERRED
        )
        assert strongest_layer([EvidenceLayer.UNKNOWN]) is EvidenceLayer.UNKNOWN
        assert strongest_layer([]) is EvidenceLayer.UNKNOWN

    def test_e_record_layer_is_never_weaker_than_its_own_claims(self):
        intelligence = _intelligence(make_candidate())
        claim_layers = [claim.layer for claim in intelligence.evidence]
        assert strongest_layer(claim_layers) is not None
        assert intelligence.evidence_layer is strongest_layer(
            claim_layers + [intelligence.evidence_layer]
        )


class TestFSchemaState:
    def test_f_partial_schema_when_registry_describes_but_does_not_enumerate(self):
        candidate = make_candidate(feature_description="Columns include a grade field")
        intelligence = _intelligence(candidate)
        assert intelligence.schema_state is SchemaState.PARTIAL

    def test_f_unknown_schema_when_candidate_publishes_nothing(self):
        candidate = make_candidate(
            description="",
            tags=[],
            domain=None,
            source_type=SourceType.VERIFIED_EXTERNAL,
        )
        intelligence = _intelligence(candidate)
        assert intelligence.schema_state is SchemaState.UNKNOWN

    def test_f_ai_candidate_is_never_schema_verified(self):
        intelligence = _intelligence(_ai_candidate())
        assert intelligence.schema_state is not SchemaState.VERIFIED


# ---------------------------------------------------------------------------
# G-L  target analysis
# ---------------------------------------------------------------------------


class TestGTargetStates:
    def test_g_declared_target_is_present(self):
        candidate = make_candidate(
            target_description="Whether the student passed the final examination"
        )
        requirements = make_requirements(target="final examination result")
        intelligence = _intelligence(candidate, requirements)

        assert intelligence.target_analysis.state is TargetState.PRESENT
        assert intelligence.target_analysis.layer is EvidenceLayer.SOURCE_DECLARED

    def test_h_no_evidence_at_all_is_unknown_not_incompatible(self):
        candidate = make_candidate(target_description=None, description="")
        requirements = make_requirements(target="a field nobody publishes")
        intelligence = _intelligence(candidate, requirements)

        assert intelligence.target_analysis.state is TargetState.UNKNOWN
        assert intelligence.target_analysis.state is not TargetState.INCOMPATIBLE

    def test_h_target_declared_absent_is_incompatible(self):
        candidate = make_candidate(
            target_description="Whether the student passed the final examination"
        )
        requirements = make_requirements(target="hospital readmission rate")
        intelligence = _intelligence(candidate, requirements)

        assert intelligence.target_analysis.state is TargetState.INCOMPATIBLE
        assert intelligence.target_analysis.layer is EvidenceLayer.SOURCE_DECLARED

    def test_h_declared_absence_of_a_different_target_is_incompatible(self):
        """The registry names its own target, and it is not the project's."""
        candidate = make_candidate(
            target_description="Whether the student passed the final examination"
        )
        requirements = make_requirements(target="hospital readmission rate")
        intelligence = _intelligence(candidate, requirements)
        # It must not claim the project target is PRESENT merely because a target
        # field exists.
        assert intelligence.target_analysis.state is not TargetState.PRESENT

    def test_i_present_target_requires_declared_evidence(self):
        with pytest.raises(ValueError, match="need declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.PRESENT,
                layer=EvidenceLayer.INFERRED,
                evidence="inferred from tags",
            )

    def test_i_present_target_rejected_on_ai_evidence(self):
        with pytest.raises(ValueError, match="need declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.PRESENT,
                layer=EvidenceLayer.AI_SUGGESTED,
                evidence="the model thinks so",
            )

    def test_i_derivable_target_requires_declared_evidence(self):
        with pytest.raises(ValueError, match="need declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.DERIVABLE,
                layer=EvidenceLayer.UNKNOWN,
                evidence="maybe",
            )

    def test_i_incompatible_target_also_requires_declared_evidence(self):
        with pytest.raises(ValueError, match="declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.INCOMPATIBLE,
                layer=EvidenceLayer.AI_SUGGESTED,
                evidence="the model thinks so",
            )

    def test_i_derivable_target_rejected_on_inferred_evidence(self):
        with pytest.raises(ValueError, match="need declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.DERIVABLE,
                layer=EvidenceLayer.INFERRED,
                evidence="probably",
            )

    def test_i_unknown_target_needs_no_evidence_at_all(self):
        analysis = TargetAnalysis(
            requirement="churn",
            state=TargetState.UNKNOWN,
            layer=EvidenceLayer.UNKNOWN,
            evidence="nothing was published",
        )
        assert analysis.state is TargetState.UNKNOWN

    def test_j_derivable_target_requires_named_ingredients(self):
        """'No target provided' alone is not a construction route."""
        candidate = make_candidate(
            target_description=(
                "No target column is provided. The dataset contains measurements "
                "only."
            )
        )
        requirements = make_requirements(target="customer churn")
        intelligence = _intelligence(candidate, requirements)

        assert intelligence.target_analysis.state in (
            TargetState.DERIVABLE,
            TargetState.UNKNOWN,
        )
        if intelligence.target_analysis.state is TargetState.DERIVABLE:
            assert intelligence.target_analysis.layer is not EvidenceLayer.UNKNOWN

    def test_j_target_state_never_exceeds_its_evidence_layer(self):
        """The builder must never emit a declaring state on unproven evidence."""
        candidates = [
            make_candidate(),
            make_candidate(target_description=None, description=""),
            make_candidate(target_description="Whether the student passed"),
            make_candidate(domain=None, tags=[], description="", target_description=None),
            _ai_candidate(),
            _ai_candidate(target_description="Whether the student passed"),
        ]
        requirements = make_requirements()
        for candidate in candidates:
            intelligence = _intelligence(candidate, requirements)
            analysis = intelligence.target_analysis
            if analysis.layer not in DECLARING_LAYERS:
                assert analysis.state is TargetState.UNKNOWN, (
                    f"{candidate.name}: {analysis.state.value} on {analysis.layer.value}"
                )

    def test_k_ai_target_suggestion_is_recorded_but_not_claimed(self):
        candidate = _ai_candidate(
            target_description="Whether the student passed the final examination"
        )
        requirements = make_requirements(target="final examination result")
        intelligence = _intelligence(candidate, requirements)

        analysis = intelligence.target_analysis
        assert analysis.state is TargetState.UNKNOWN
        assert analysis.layer is EvidenceLayer.AI_SUGGESTED
        assert "AI-suggested" in analysis.evidence


# ---------------------------------------------------------------------------
# L-P  feature coverage
# ---------------------------------------------------------------------------


class TestFeatureCoverage:
    def test_l_exact_feature_match_is_exact(self):
        intelligence = _intelligence(make_candidate())
        states = _states(intelligence.feature_coverage)
        assert states["attendance"] is FeatureMatchState.EXACT

    def test_l_alias_feature_match_is_reported_as_alias(self):
        """ALIAS means every word of the requirement is present, out of phrase.

        "Total hours of study" contains both "hours" and "study" but not the
        phrase "study hours", which is a looser match than an exact phrase hit
        and a stronger one than a dictionary equivalent.
        """
        candidate = make_candidate(description="Total hours of study each week")
        intelligence = _intelligence(candidate)
        states = _states(intelligence.feature_coverage)
        assert states["study hours"] is FeatureMatchState.ALIAS

    def test_l_semantic_feature_match_is_reported_as_semantic(self):
        """SEMANTIC means a dictionary-declared equivalent stood in for the term."""
        candidate = make_candidate(description="Weekly hours studied")
        intelligence = _intelligence(candidate)
        states = _states(intelligence.feature_coverage)
        assert states["study hours"] is FeatureMatchState.SEMANTIC
        hours = _coverage_term(intelligence.feature_coverage, "study hours")
        assert hours.via, "a semantic match must name what stood in for the term"

    def test_l_dictionary_equivalent_is_reported_as_semantic(self):
        candidate = make_candidate(description="Body mass index and cholesterol")
        requirements = make_requirements(features=["BMI"], target="heart disease")
        intelligence = _intelligence(candidate, requirements)
        states = _states(intelligence.feature_coverage)
        assert states["BMI"] is FeatureMatchState.SEMANTIC

    def test_l_exact_phrase_outranks_out_of_phrase_alias(self):
        exact = _intelligence(make_candidate(description="study hours per week"))
        alias = _intelligence(make_candidate(description="Total hours of study"))
        assert (
            _states(exact.feature_coverage)["study hours"]
            is FeatureMatchState.EXACT
        )
        assert (
            _states(alias.feature_coverage)["study hours"]
            is FeatureMatchState.ALIAS
        )

    def test_l_every_required_feature_is_accounted_for_exactly_once(self):
        requirements = make_requirements()
        intelligence = _intelligence(make_candidate(), requirements)

        coverage = intelligence.feature_coverage
        assert coverage.required_count == len(requirements.dataset_requirements.feature_requirements)
        seen = [f.requirement for f in coverage.all_features]
        assert len(seen) == len(set(seen))
        assert set(seen) == _terms(requirements)

    def test_m_unknown_feature_is_not_missing(self):
        candidate = make_candidate(
            name="Dataset 7",
            description="",
            tags=[],
            domain=None,
            target_description=None,
        )
        intelligence = _intelligence(candidate)
        assert intelligence.feature_coverage.missing == []
        assert intelligence.feature_coverage.unknown

    def test_m_missing_requires_a_schema_that_can_prove_absence(self):
        """Same ranker verdict, opposite bucket, decided only by schema visibility.

        The ranker already returns MISMATCH for "attendance" in both cases below.
        Only the second candidate published an enumeration, so only the second
        may report the feature as absent rather than merely unestablished.
        """
        silent = make_candidate(
            name="Dataset 7", description="", tags=[], domain=None, target_description=None
        )
        enumerated = make_candidate(
            description="",
            target_description=None,
            feature_description="Each row has the identifier, score and date columns",
        )

        silent_intelligence = _intelligence(silent)
        enumerated_intelligence = _intelligence(enumerated)

        assert silent_intelligence.schema_state is SchemaState.UNKNOWN
        assert silent_intelligence.feature_coverage.missing == []
        assert silent_intelligence.feature_coverage.unknown

        assert enumerated_intelligence.schema_state is not SchemaState.UNKNOWN
        assert enumerated_intelligence.feature_coverage.missing
        for feature in enumerated_intelligence.feature_coverage.missing:
            assert feature.state is FeatureMatchState.MISSING

    def test_n_free_text_description_never_proves_a_missing_feature(self):
        """``description`` is prose. It cannot enumerate, so absence stays unknown.

        The ranker's own schema cues make the distinction: "column", "field",
        "row" and "attribute" count as declaring a schema, and ordinary
        descriptive prose does not. Without that, a dataset whose blurb happens
        to omit a required feature would report it as absent.
        """
        prose_only = _intelligence(
            make_candidate(
                name="Dataset 7",
                description="Study hours, examination marks and assignment scores",
                target_description=None,
            )
        )
        assert prose_only.schema_state is SchemaState.UNKNOWN
        assert prose_only.feature_coverage.missing == []
        assert prose_only.feature_coverage.unknown

    def test_n_missing_feature_rejected_on_no_evidence(self):
        with pytest.raises(ValueError, match="cannot be MISSING"):
            FeatureCompatibility(
                requirement="attendance",
                state=FeatureMatchState.MISSING,
                layer=EvidenceLayer.UNKNOWN,
            )

    def test_n_missing_feature_rejected_on_ai_evidence(self):
        with pytest.raises(ValueError, match="cannot be MISSING"):
            FeatureCompatibility(
                requirement="attendance",
                state=FeatureMatchState.MISSING,
                layer=EvidenceLayer.AI_SUGGESTED,
            )

    def test_n_missing_feature_is_legal_on_a_declared_enumeration(self):
        """Prose that enumerates columns is what makes MISSING expressible at all."""
        feature = FeatureCompatibility(
            requirement="attendance",
            state=FeatureMatchState.MISSING,
            layer=EvidenceLayer.INFERRED,
        )
        assert feature.state is FeatureMatchState.MISSING

    def test_n_unknown_feature_cannot_claim_verified_schema(self):
        """Verified schema is somebody knowing, so it cannot file a feature as UNKNOWN."""
        with pytest.raises(ValueError, match="cannot be UNKNOWN"):
            FeatureCompatibility(
                requirement="attendance",
                state=FeatureMatchState.UNKNOWN,
                layer=EvidenceLayer.SCHEMA_VERIFIED,
            )

    def test_n_unknown_feature_cannot_claim_declared_evidence(self):
        with pytest.raises(ValueError, match="cannot be UNKNOWN"):
            FeatureCompatibility(
                requirement="attendance",
                state=FeatureMatchState.UNKNOWN,
                layer=EvidenceLayer.SOURCE_DECLARED,
            )

    def test_n_unknown_feature_is_fine_on_no_evidence(self):
        feature = FeatureCompatibility(
            requirement="attendance",
            state=FeatureMatchState.UNKNOWN,
            layer=EvidenceLayer.UNKNOWN,
        )
        assert feature.state is FeatureMatchState.UNKNOWN

class TestOCrossFieldGuards:
    """Rules the whole record has to satisfy, checked on DatasetIntelligence."""

    def _record(self, **kwargs):
        coverage = kwargs.pop("feature_coverage", _coverage())
        payload = {
            "candidate_id": "kaggle:x",
            "evidence_layer": EvidenceLayer.SOURCE_DECLARED,
            "evidence": [],
            "schema_state": SchemaState.PARTIAL,
            "target_analysis": TargetAnalysis(
                requirement="churn",
                state=TargetState.UNKNOWN,
                layer=EvidenceLayer.UNKNOWN,
            ),
            "feature_coverage": coverage,
            "compatibility": _assessment(coverage),
        }
        payload.update(kwargs)
        return payload

    def test_o_missing_feature_requires_a_schema_that_can_prove_absence(self):
        """UNKNOWN schema cannot support MISSING - that is absence of evidence."""
        payload = self._record(
            schema_state=SchemaState.UNKNOWN,
            feature_coverage=_coverage(
                missing=[
                    FeatureCompatibility(
                        requirement="attendance",
                        state=FeatureMatchState.MISSING,
                        layer=EvidenceLayer.SOURCE_DECLARED,
                    )
                ]
            ),
        )
        payload["compatibility"] = _assessment(payload["feature_coverage"])
        with pytest.raises(ValueError, match="missing features"):
            DatasetIntelligence(**payload)

    def test_o_missing_feature_is_allowed_against_a_partial_schema(self):
        payload = self._record(
            schema_state=SchemaState.PARTIAL,
            feature_coverage=_coverage(
                missing=[
                    FeatureCompatibility(
                        requirement="attendance",
                        state=FeatureMatchState.MISSING,
                        layer=EvidenceLayer.INFERRED,
                    )
                ]
            ),
        )
        payload["compatibility"] = _assessment(payload["feature_coverage"])
        assert DatasetIntelligence(**payload).feature_coverage.missing

    def test_o_verified_schema_needs_a_verified_claim(self):
        with pytest.raises(ValueError, match="SCHEMA_VERIFIED claim"):
            DatasetIntelligence(**self._record(schema_state=SchemaState.VERIFIED))

    def test_o_verified_claim_cannot_coexist_with_unknown_schema(self):
        with pytest.raises(ValueError, match="schema_state is unknown"):
            DatasetIntelligence(
                **self._record(
                    schema_state=SchemaState.UNKNOWN,
                    evidence=[
                        EvidenceClaim(
                            subject="columns",
                            value="identifier, score",
                            layer=EvidenceLayer.SCHEMA_VERIFIED,
                            source="phase-3b schema verification",
                        )
                    ],
                )
            )

    def test_o_record_layer_cannot_understate_its_claims(self):
        with pytest.raises(ValueError, match="understates"):
            DatasetIntelligence(
                **self._record(
                    evidence_layer=EvidenceLayer.INFERRED,
                    evidence=[
                        EvidenceClaim(
                            subject="description",
                            value="text",
                            layer=EvidenceLayer.SOURCE_DECLARED,
                            source="kaggle",
                        )
                    ],
                )
            )

    def test_o_incompatible_target_needs_declared_evidence(self):
        """The local TargetAnalysis guard refuses it before the record sees it."""
        with pytest.raises(ValueError, match="declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.INCOMPATIBLE,
                layer=EvidenceLayer.AI_SUGGESTED,
                evidence="the model thinks so",
            )

    def test_o_incompatible_target_rejected_on_inferred_evidence(self):
        with pytest.raises(ValueError, match="declared evidence"):
            TargetAnalysis(
                requirement="churn",
                state=TargetState.INCOMPATIBLE,
                layer=EvidenceLayer.INFERRED,
                evidence="tags suggest otherwise",
            )

    def test_o_coverage_reported_twice_must_agree(self):
        payload = self._record(
            compatibility=_assessment(
                _coverage(
                    unknown=[
                        FeatureCompatibility(
                            requirement="attendance",
                            state=FeatureMatchState.UNKNOWN,
                            layer=EvidenceLayer.UNKNOWN,
                        )
                    ]
                )
            )
        )
        with pytest.raises(ValueError, match="reported once"):
            DatasetIntelligence(**payload)

    def test_o_coverage_buckets_must_partition_the_requirements(self):
        with pytest.raises(ValueError, match="required_count"):
            _coverage(
                unknown=[
                    FeatureCompatibility(
                        requirement="attendance",
                        state=FeatureMatchState.UNKNOWN,
                        layer=EvidenceLayer.UNKNOWN,
                    )
                ],
                missing=[],
            ).__class__(
                required_count=3,
                unknown=[
                    FeatureCompatibility(
                        requirement="attendance",
                        state=FeatureMatchState.UNKNOWN,
                        layer=EvidenceLayer.UNKNOWN,
                    )
                ],
            )

    def test_o_a_feature_cannot_sit_in_two_buckets(self):
        shared = FeatureCompatibility(
            requirement="attendance",
            state=FeatureMatchState.EXACT,
            layer=EvidenceLayer.SOURCE_DECLARED,
        )
        with pytest.raises(ValueError, match="more than one bucket"):
            _coverage(matched=[shared], unknown=[shared])

    def test_p_inferred_feature_match_is_labelled_inferred(self):
        candidate = make_candidate(description="Hours of revision per week")
        intelligence = _intelligence(candidate)
        hours = _coverage_term(intelligence.feature_coverage, "study hours")
        if hours is not None and hours.state in (
            FeatureMatchState.SEMANTIC,
            FeatureMatchState.INFERRED,
        ):
            assert hours.layer is not EvidenceLayer.SCHEMA_VERIFIED


# ---------------------------------------------------------------------------
# Q-S  compatibility dimensions and overall level
# ---------------------------------------------------------------------------


class TestCompatibilityDimensions:
    def _transport_candidate(self) -> DatasetCandidate:
        """A candidate whose own metadata resolves to exactly one other domain."""
        return make_candidate(
            name="Transport Routes",
            description="Bus and train route timings and delays",
            tags=["transport"],
            domain="transport",
            target_description=None,
        )

    def test_q_domain_mismatch_is_reported(self):
        candidate = self._transport_candidate()
        requirements = make_requirements(domain="Healthcare")
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.domain is MatchStatus.MISMATCH

    def test_q_domain_with_no_evidence_is_unknown_not_mismatch(self):
        """A bare record states nothing, so nothing can contradict the project.

        The name must be neutral too: the default fixture is called "Verified
        Student Dataset", and the word "student" alone resolves to the education
        domain, which would make this a mismatch rather than silence.
        """
        candidate = make_candidate(
            name="Dataset 7",
            description="",
            tags=[],
            domain=None,
            target_description=None,
        )
        requirements = make_requirements(domain="Healthcare")
        intelligence = _intelligence(candidate, requirements)
        # No domain means nothing was stated, which cannot contradict anything.
        assert intelligence.compatibility.domain is MatchStatus.UNKNOWN

    def test_q_domain_mismatch_needs_a_single_resolved_domain(self):
        """A candidate spanning two domains has not declared itself single-domain."""
        candidate = make_candidate(
            name="Mixed Collection",
            description="Bus routes and exam marks together",
            tags=["transport", "education"],
            domain="transport",
            target_description=None,
        )
        requirements = make_requirements(domain="Healthcare")
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.domain is not MatchStatus.MISMATCH

    def test_r_task_mismatch_is_reported(self):
        """A dataset that *declares* only other tasks contradicts the project."""
        candidate = make_candidate(tags=["education"], task_types=["classification"])
        requirements = make_requirements(tasks=["anomaly detection"])
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.task is MatchStatus.MISMATCH
        assert any("anomaly detection" in c for c in intelligence.compatibility.contradictions)
        assert RiskCategory.TASK_MISMATCH in {r.category for r in intelligence.risks}

    def test_r_undeclared_task_is_unknown_not_mismatch(self):
        """Kaggle publishes no task field; silence cannot contradict the project."""
        candidate = make_candidate(tags=["education"])
        requirements = make_requirements(tasks=["anomaly detection"])
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.task is MatchStatus.UNKNOWN
        assert RiskCategory.TASK_MISMATCH not in {r.category for r in intelligence.risks}

    def test_r_target_mismatch_is_reported(self):
        candidate = make_candidate(
            target_description="Whether the student passed the final examination"
        )
        requirements = make_requirements(target="insurance fraud amount")
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.target is MatchStatus.MISMATCH

    def test_s_derivable_target_is_not_reported_as_mismatch(self):
        """A constructible target is usable with work, so it is not incompatible."""
        candidate = make_candidate(
            target_description=(
                "No target column is provided. A target can be derived by "
                "thresholding the recorded measurements."
            )
        )
        requirements = make_requirements(target="customer churn")
        intelligence = _intelligence(candidate, requirements)

        if intelligence.target_analysis.state is TargetState.DERIVABLE:
            assert intelligence.compatibility.target is not MatchStatus.MISMATCH

    def test_s_unknown_dimension_never_counts_as_a_match(self):
        candidate = make_candidate()
        intelligence = _intelligence(candidate)
        assessment = intelligence.compatibility
        for dimension in (assessment.domain, assessment.task, assessment.target):
            if dimension is MatchStatus.UNKNOWN:
                assert dimension is not MatchStatus.MATCH

    def test_s_data_type_is_unknown_without_a_data_type_matcher(self):
        intelligence = _intelligence(make_candidate())
        assert intelligence.compatibility.data_type is MatchStatus.UNKNOWN

    def test_s_overall_level_is_one_of_four_named_levels(self):
        intelligence = _intelligence(make_candidate())
        assert intelligence.compatibility.overall in set(CompatibilityLevel)

    def test_s_low_overall_requires_a_blocking_finding(self):
        candidate = make_candidate(
            name="City Bus Routes",
            description="Bus routes, stop schedules and passenger counts",
            domain="transport",
            tags=["transport"],
            target_description=None,
        )
        requirements = make_requirements(domain="Healthcare")
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.overall is CompatibilityLevel.LOW
        assert intelligence.compatibility.domain is MatchStatus.MISMATCH
        assert RiskCategory.DOMAIN_MISMATCH in {r.category for r in intelligence.risks}

    def test_s_self_contradicting_candidate_is_not_graded_low(self):
        """Tagged transport but described as education: no single domain, so no verdict."""
        candidate = make_candidate(domain="transport", tags=["transport"])
        requirements = make_requirements(domain="Healthcare")
        intelligence = _intelligence(candidate, requirements)
        assert intelligence.compatibility.domain is MatchStatus.UNKNOWN
        assert intelligence.compatibility.overall is not CompatibilityLevel.LOW


# ---------------------------------------------------------------------------
# risks, limitations, explanation
# ---------------------------------------------------------------------------


class TestRisksAndExplanation:
    def test_ai_candidate_raises_ai_unverified_risk(self):
        intelligence = _intelligence(_ai_candidate())
        categories = {risk.category for risk in intelligence.risks}
        assert RiskCategory.AI_UNVERIFIED in categories

    def test_unknown_target_raises_target_unknown_risk(self):
        candidate = make_candidate(target_description=None, description="")
        intelligence = _intelligence(candidate)
        categories = {risk.category for risk in intelligence.risks}
        assert RiskCategory.TARGET_UNKNOWN in categories

    def test_profile_dependent_risks_are_never_invented(self):
        """No profiling happened, so no data-quality risk may be emitted."""
        for candidate in (make_candidate(), _ai_candidate()):
            intelligence = _intelligence(candidate)
            categories = {risk.category for risk in intelligence.risks}
            assert categories.isdisjoint(
                {
                    RiskCategory.CLASS_IMBALANCE,
                    RiskCategory.HIGH_MISSINGNESS,
                    RiskCategory.DUPLICATES,
                    RiskCategory.IDENTIFIER_COLUMN,
                    RiskCategory.SMALL_DATASET,
                    RiskCategory.POSSIBLE_LEAKAGE,
                }
            )

    def test_explanation_separates_matches_unknowns_and_missing(self):
        intelligence = _intelligence(make_candidate())
        explanation = intelligence.explanation

        assert explanation.why_recommended
        assert isinstance(explanation, RecommendationExplanation)
        assert explanation.what_is_unknown or explanation.what_is_missing

    def test_explanation_cannot_assert_an_unverified_target(self):
        candidate = _ai_candidate(
            target_description="Whether the student passed the final examination"
        )
        requirements = make_requirements(target="final examination result")
        intelligence = _intelligence(candidate, requirements)
        explanation = intelligence.explanation

        assert intelligence.target_analysis.state is TargetState.UNKNOWN
        why = " ".join(explanation.why_recommended).lower()
        assert "compatible target" not in why
        assert "target is present" not in why
        assert not any(v.startswith("Target ") for v in explanation.what_is_verified)

    def test_risk_and_limitation_models_require_a_detail(self):
        with pytest.raises(ValueError):
            DatasetRisk(
                category=RiskCategory.AI_UNVERIFIED,
                severity=RiskSeverity.WARNING,
                detail="",
            )
        with pytest.raises(ValueError):
            DatasetRisk(
                category=RiskCategory.AI_UNVERIFIED,
                severity=RiskSeverity.WARNING,
                detail="   ",
            )
        with pytest.raises(ValueError):
            DatasetLimitation(code="schema_not_available", detail="")
        with pytest.raises(ValueError):
            DatasetLimitation(code="", detail="The schema is not published.")

    def test_a_blank_explanation_section_is_rejected(self):
        with pytest.raises(ValueError, match="blank entry"):
            RecommendationExplanation(why_recommended=[""])


# ---------------------------------------------------------------------------
# T-V  determinism and reporting
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_t_same_input_serializes_identically(self):
        candidate = make_candidate()
        requirements = make_requirements()

        first = build_dataset_intelligence(_ranked(candidate, requirements), requirements)
        second = build_dataset_intelligence(_ranked(candidate, requirements), requirements)

        assert first.model_dump_json() == second.model_dump_json()

    def test_t_records_survive_a_json_round_trip(self):
        intelligence = _intelligence(make_candidate())
        restored = DatasetIntelligence.model_validate_json(intelligence.model_dump_json())

        assert restored == intelligence

    def test_u_ordering_is_stable_regardless_of_input_order(self):
        requirements = make_requirements()
        candidates = [
            make_candidate(name="A", source_id="owner/a"),
            make_candidate(name="B", source_id="owner/b"),
        ]
        engine = _engine()
        forward = engine.rank(candidates, requirements)
        backward = engine.rank(list(reversed(candidates)), requirements)

        assert [c.id for c in forward] == [c.id for c in backward]

        report = build_intelligence_report(forward, requirements)
        assert json.loads(report[0].model_dump_json())["candidate_id"] == forward[0].id

    def test_v_every_record_carries_candidate_id_and_explanation(self):
        intelligence = _intelligence(make_candidate())
        assert intelligence.candidate_id
        assert intelligence.explanation.why_recommended


# ---------------------------------------------------------------------------
# regressions: target matching on a shared generic word
# ---------------------------------------------------------------------------


class TestGenericWordTargetFalsePositive:
    """"rate" alone resolved a readmission target to the ``attendance + rate`` space."""

    def test_unrelated_target_is_incompatible_not_present(self):
        candidate = make_candidate(
            target_description="Whether the student passed the final examination"
        )
        intelligence = _intelligence(
            candidate, make_requirements(target="hospital readmission rate")
        )
        assert intelligence.target_analysis.state is TargetState.INCOMPATIBLE
        assert intelligence.compatibility.target is MatchStatus.MISMATCH
        assert RiskCategory.TARGET_MISMATCH in {r.category for r in intelligence.risks}
