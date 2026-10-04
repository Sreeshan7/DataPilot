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
