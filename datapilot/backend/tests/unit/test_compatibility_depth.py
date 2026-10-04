"""Phase 3B: requirement coverage and the dimensions beyond features and target.

Every case is a way the compatibility record could overstate or understate what is
known: a modality claimed without a published file, a time axis invented, a
dimension reported as unknown that nobody asked about, or "4/6" hiding the unknown.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import MatchStatus, MetadataDepth
from app.schemas.intelligence import (
    CompatibilityLevel,
    FeatureCompatibility,
    FeatureCoverage,
    FeatureMatchState,
    EvidenceLayer,
    RecommendationExplanation,
)
from app.services.datasets.intelligence import build_dataset_intelligence
from app.services.datasets.normalizer import normalize_genai_candidate
from app.services.datasets.ranking import DatasetRankingEngine
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_genai_candidate,
    make_requirements,
    make_settings,
)


def _intel(candidate, requirements=None):
    requirements = requirements or make_requirements()
    ranked = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW).rank(
        [candidate], requirements
    )[0]
    return build_dataset_intelligence(ranked, requirements)


def _feature(name, state, layer):
    return FeatureCompatibility(requirement=name, state=state, layer=layer)


# ---------------------------------------------------------------------------
# Requirement coverage reads matched / unknown / missing separately
# ---------------------------------------------------------------------------


class TestCoverageSummary:
    def test_summary_keeps_the_three_buckets_apart(self):
        coverage = FeatureCoverage(
            required_count=6,
            matched=[
                _feature(n, FeatureMatchState.EXACT, EvidenceLayer.SOURCE_DECLARED)
                for n in "abcd"
            ],
            unknown=[_feature("e", FeatureMatchState.UNKNOWN, EvidenceLayer.UNKNOWN)],
            missing=[_feature("f", FeatureMatchState.MISSING, EvidenceLayer.INFERRED)],
        )
        assert coverage.summary() == "6 required: 4 matched, 1 unknown, 1 missing"
        assert "4/6" not in coverage.summary()

    def test_no_required_features_is_stated_not_reported_as_zero_of_zero(self):
        assert FeatureCoverage(required_count=0).summary() == (
            "no required features were identified"
        )

    def test_the_explanation_leads_with_a_coverage_line_per_requirement(self):
        explanation = _intel(make_candidate()).explanation
        lines = explanation.requirement_coverage
        assert lines[0].startswith("Features: ")
        assert any(line.startswith("Target: ") for line in lines)
        assert any(line.startswith("Domain: ") for line in lines)

    def test_a_blank_coverage_line_is_rejected(self):
        with pytest.raises(ValueError, match="blank entry"):
            RecommendationExplanation(requirement_coverage=["  "])

    def test_coverage_lines_match_the_feature_buckets(self):
        intel = _intel(make_candidate())
        assert intel.explanation.requirement_coverage[0] == (
            f"Features: {intel.feature_coverage.summary()}"
        )


# ---------------------------------------------------------------------------
# Data type: published file extensions against the stated modality
# ---------------------------------------------------------------------------


class TestDataType:
    def test_published_csv_matches_a_tabular_project(self):
        candidate = make_candidate(file_types=[".csv"], file_names=["marks.csv"])
        intel = _intel(candidate, make_requirements(data_types=["tabular records"]))
        assert intel.compatibility.data_type is MatchStatus.MATCH
        assert any("Data type" in m for m in intel.explanation.what_matches)

    def test_only_images_against_a_tabular_project_is_a_mismatch(self):
        candidate = make_candidate(file_types=[".png"], file_names=["a.png", "b.png"])
        intel = _intel(candidate, make_requirements(data_types=["tabular measurements"]))
        assert intel.compatibility.data_type is MatchStatus.MISMATCH
        assert intel.compatibility.overall is CompatibilityLevel.LOW
        assert any("image" in c for c in intel.compatibility.contradictions)

    def test_unknown_files_are_unknown_not_a_mismatch(self):
        intel = _intel(make_candidate(), make_requirements(data_types=["tabular records"]))
        assert intel.compatibility.data_type is MatchStatus.UNKNOWN
        assert "data_type" in intel.compatibility.unknown_dimensions
        assert any("Data type" in u for u in intel.explanation.what_is_unknown)

    def test_a_project_that_names_no_data_type_is_not_told_it_is_unknown(self):
        intel = _intel(make_candidate(file_types=[".csv"]))
        assert intel.compatibility.data_type is MatchStatus.UNKNOWN
        assert "data_type" not in intel.compatibility.unknown_dimensions

    def test_an_ai_candidates_file_claim_is_labelled_as_a_claim_not_a_fact(self):
        ai = normalize_genai_candidate(make_genai_candidate())
        ai.file_types = [".csv"]
        intel = _intel(ai, make_requirements(data_types=["tabular records"]))
        assert intel.compatibility.data_type is MatchStatus.MATCH
        line = next(m for m in intel.explanation.what_matches if m.startswith("Data type"))
        assert "suggested by a model" in line
        assert "publishes" not in line
        assert not any("Data type" in v for v in intel.explanation.what_is_verified)

    def test_an_ai_candidates_conflicting_file_claim_cannot_veto(self):
        ai = normalize_genai_candidate(make_genai_candidate())
        ai.file_types = [".png"]
        intel = _intel(ai, make_requirements(data_types=["tabular records"]))
        assert intel.compatibility.data_type is not MatchStatus.MISMATCH
        assert intel.compatibility.contradictions == []


# ---------------------------------------------------------------------------
# Temporal: a time axis for a forecasting project
# ---------------------------------------------------------------------------


FORECAST = dict(
    domain="Energy",
    subdomain="Forecasting",
    tasks=["time series forecasting"],
    target="electricity consumption",
    features=["temperature"],
    keywords=["energy forecasting", "electricity consumption"],
)


class TestTemporal:
    def test_a_project_without_a_time_requirement_has_no_temporal_dimension(self):
        intel = _intel(make_candidate())
        assert intel.compatibility.temporal is None
        assert "temporal" not in intel.compatibility.unknown_dimensions

    def test_hourly_data_matches_a_forecasting_project(self):
        candidate = make_candidate(
            name="Household Power",
            description="Hourly electricity consumption with temperature readings",
            domain="energy",
            tags=["energy"],
        )
        intel = _intel(candidate, make_requirements(**FORECAST))
        assert intel.compatibility.temporal is MatchStatus.MATCH
        assert any("Time dimension" in m for m in intel.explanation.what_matches)

    def test_no_time_word_is_unknown_never_a_mismatch(self):
        candidate = make_candidate(
            name="Household Power",
            description="Electricity consumption with temperature readings",
            domain="energy",
            tags=["energy"],
        )
        intel = _intel(candidate, make_requirements(**FORECAST))
        assert intel.compatibility.temporal is MatchStatus.UNKNOWN
        assert "temporal" in intel.compatibility.unknown_dimensions
        assert intel.compatibility.contradictions == []

    def test_a_last_updated_date_is_not_a_time_axis(self):
        candidate = make_candidate(
            name="Household Power",
            description="Electricity consumption. Last updated on 2025-01-01.",
            domain="energy",
            tags=["energy"],
        )
        intel = _intel(candidate, make_requirements(**FORECAST))
        assert intel.compatibility.temporal is MatchStatus.UNKNOWN

    def test_an_ai_candidates_time_claim_is_not_listed_as_verified(self):
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Smart Meter Series", description="Hourly smart meter readings"
            )
        )
        intel = _intel(ai, make_requirements(**FORECAST))
        assert not any("Time dimension" in v for v in intel.explanation.what_is_verified)
        line = next(
            m for m in intel.explanation.what_matches if m.startswith("Time dimension")
        )
        assert "a model's description" in line


# ---------------------------------------------------------------------------
# Unknown dimensions name only what the project asked about
# ---------------------------------------------------------------------------


class TestUnknownDimensions:
    def test_a_project_with_no_domain_is_not_told_the_domain_is_unknown(self):
        intel = _intel(make_candidate(), make_requirements(domain=None, subdomain=None))
        assert "domain" not in intel.compatibility.unknown_dimensions

    def test_an_undeclared_task_is_listed_when_the_project_requires_one(self):
        intel = _intel(make_candidate(), make_requirements(tasks=["classification"]))
        assert "task" in intel.compatibility.unknown_dimensions

    def test_no_tasks_means_the_task_dimension_is_not_listed(self):
        intel = _intel(make_candidate(), make_requirements(tasks=[]))
        assert "task" not in intel.compatibility.unknown_dimensions

    def test_the_list_is_sorted_and_free_of_duplicates(self):
        dims = _intel(make_candidate()).compatibility.unknown_dimensions
        assert dims == sorted(set(dims))


# ---------------------------------------------------------------------------
# Listing-only records say so
# ---------------------------------------------------------------------------


class TestListingDepthLimitation:
    def test_a_listing_level_registry_record_carries_the_limitation(self):
        intel = _intel(make_candidate())
        assert "listing_metadata_only" in {lim.code for lim in intel.limitations}

    def test_a_detailed_record_does_not(self):
        candidate = make_candidate(metadata_depth=MetadataDepth.DETAILED)
        assert "listing_metadata_only" not in {
            lim.code for lim in _intel(candidate).limitations
        }

    def test_an_ai_candidate_is_not_described_as_a_listing(self):
        ai = normalize_genai_candidate(make_genai_candidate())
        assert "listing_metadata_only" not in {lim.code for lim in _intel(ai).limitations}


# ---------------------------------------------------------------------------
# Alternatives within one dimension: any match settles it
# ---------------------------------------------------------------------------


class TestAlternativesWithinADimension:
    """A project that lists several acceptable tasks is served by a dataset
    supporting any of them. It was previously marked MISMATCH for the others."""

    def test_supporting_one_of_several_listed_tasks_is_a_match(self):
        candidate = make_candidate(task_types=["classification"])
        intel = _intel(candidate, make_requirements(tasks=["classification", "regression"]))
        assert intel.compatibility.task is MatchStatus.MATCH
        assert intel.compatibility.contradictions == []

    def test_supporting_none_of_the_listed_tasks_is_still_a_mismatch(self):
        candidate = make_candidate(task_types=["time series forecasting"])
        intel = _intel(candidate, make_requirements(tasks=["classification", "regression"]))
        assert intel.compatibility.task is MatchStatus.MISMATCH
        assert intel.compatibility.overall is CompatibilityLevel.LOW

    def test_the_dimension_agrees_with_the_contradiction_detector(self):
        for declared, expect in (
            (["classification"], MatchStatus.MATCH),
            (["regression"], MatchStatus.MATCH),
            (["clustering"], MatchStatus.MISMATCH),
        ):
            intel = _intel(
                make_candidate(task_types=declared),
                make_requirements(tasks=["classification", "regression"]),
            )
            assert intel.compatibility.task is expect
            assert bool(intel.compatibility.contradictions) is (expect is MatchStatus.MISMATCH)


# ---------------------------------------------------------------------------
# Regression (found by the live end-to-end run): a generic word is not a modality
# ---------------------------------------------------------------------------


class TestProjectModalitiesMatchWholePhrases:
    """"image data" used to be matched on its token "data", so any note containing
    the word "data" was read as requiring *image* data, and every tabular dataset
    was reported as contradicting a project about network flow data."""

    @pytest.mark.parametrize(
        "note",
        [
            "Likely time-series or sequential network flow data.",
            "inferred Network traffic data",
            "customer data",
            "sensor readings over time",
            "",
        ],
    )
    def test_a_generic_data_note_names_no_modality(self, note):
        from app.services.datasets.compatibility import _project_modalities

        assert _project_modalities([note]) == frozenset()

    @pytest.mark.parametrize(
        ("note", "expected"),
        [
            ("image data", {"image"}),
            ("labelled images of chest x-rays", {"image"}),
            ("tabular records", {"tabular"}),
            ("a spreadsheet of marks", {"tabular"}),
            ("speech recordings", {"audio"}),
            ("short video clips", {"video"}),
            ("text documents", {"text"}),
            ("tabular data and text documents", {"tabular", "text"}),
        ],
    )
    def test_a_stated_modality_is_recognised(self, note, expected):
        from app.services.datasets.compatibility import _project_modalities

        assert _project_modalities([note]) == frozenset(expected)

    def test_structured_is_not_found_inside_unstructured(self):
        from app.services.datasets.compatibility import _project_modalities

        assert "tabular" not in _project_modalities(["unstructured text documents"])
        assert "text" in _project_modalities(["unstructured text documents"])

    def test_a_tabular_dataset_does_not_contradict_a_network_flow_project(self):
        candidate = make_candidate(file_types=[".csv"], file_names=["flows.csv"])
        req = make_requirements(data_types=["Likely time-series or sequential network flow data."])
        intel = _intel(candidate, req)
        assert intel.compatibility.contradictions == []
        assert intel.compatibility.data_type is not MatchStatus.MISMATCH
        assert "data_type" not in intel.compatibility.unknown_dimensions

    def test_a_real_image_requirement_still_contradicts_tabular_files(self):
        candidate = make_candidate(file_types=[".csv"], file_names=["marks.csv"])
        intel = _intel(candidate, make_requirements(data_types=["image data of x-rays"]))
        assert intel.compatibility.data_type is MatchStatus.MISMATCH


# ---------------------------------------------------------------------------
# Found by the recommendation benchmark: wording gap vs a different outcome
# ---------------------------------------------------------------------------


class TestTargetWordingVersusDifferentOutcome:
    """The ranker's target MISMATCH only says the declared target text does not use
    the project's words. INCOMPATIBLE must mean a different outcome, not different
    phrasing: it condemned the best dataset in two benchmark cases."""

    def _intel_for(self, declared, wanted):
        from app.schemas.intelligence import TargetState  # noqa: F401

        candidate = make_candidate(
            target_description=declared, description="Dated records", feature_description=None
        )
        return _intel(candidate, make_requirements(target=wanted))

    def test_a_wording_gap_that_shares_a_word_is_unknown_not_incompatible(self):
        from app.schemas.intelligence import TargetState

        intel = self._intel_for("number of borrows per title", "borrow count")
        assert intel.target_analysis.state is TargetState.UNKNOWN
        assert intel.compatibility.target is not MatchStatus.MISMATCH
        assert intel.compatibility.overall is not CompatibilityLevel.LOW
        assert "does not use the wording" in intel.target_analysis.evidence

    def test_a_declared_target_sharing_no_word_is_a_different_outcome(self):
        from app.schemas.intelligence import TargetState

        intel = self._intel_for("readmission within 30 days", "patient mortality")
        assert intel.target_analysis.state is TargetState.INCOMPATIBLE
        assert "shares no words" in intel.target_analysis.evidence
        assert "vocabulary judgment" in intel.target_analysis.detail

    def test_the_incompatible_wording_admits_it_is_lexical(self):
        intel = self._intel_for("loan default", "approved")
        assert intel.target_analysis.state.value == "incompatible"
        assert "confirm against the dataset's own label definition" in intel.target_analysis.detail

    def test_no_declared_target_is_never_incompatible(self):
        from app.schemas.intelligence import TargetState

        candidate = make_candidate(target_description=None, description="Records")
        intel = _intel(candidate, make_requirements(target="patient mortality"))
        assert intel.target_analysis.state is not TargetState.INCOMPATIBLE

    def test_an_ai_candidates_declared_target_can_never_be_incompatible(self):
        from app.schemas.intelligence import TargetState

        ai = normalize_genai_candidate(
            make_genai_candidate(target_description="readmission within 30 days")
        )
        intel = _intel(ai, make_requirements(target="patient mortality"))
        assert intel.target_analysis.state is not TargetState.INCOMPATIBLE
