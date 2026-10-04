"""Phase 3B: reading the registry's own per-dataset record.

Kaggle's listing returns an empty description for every dataset, so a listing-level
candidate can only be judged on a title, a subtitle and tags. These tests pin what
the enrichment adds, that it adds only what the registry states, and that the
result changes what the intelligence layer is allowed to conclude.
"""

from __future__ import annotations

import httpx
import pytest

from app.schemas.dataset import MetadataDepth
from app.schemas.intelligence import FeatureMatchState, SchemaState
from app.services.datasets.base import DatasetSourceError
from app.services.datasets.deduplicator import DatasetDeduplicator
from app.services.datasets.intelligence import build_dataset_intelligence
from app.services.datasets.kaggle_source import KaggleDatasetSource
from app.services.datasets.normalizer import (
    MAX_DESCRIPTION_CHARS,
    MAX_FILE_NAMES,
    apply_kaggle_detail,
    normalize_kaggle_record,
)
from app.services.datasets.ranking import DatasetRankingEngine
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_kaggle_record,
    make_requirements,
    make_settings,
)

VIEW = {
    "subtitleNullable": "Predict student performance in secondary education",
    "descriptionNullable": (
        "Student grades and school data. Columns: attendance, study hours, "
        "assignment scores, final grade."
    ),
    "totalBytesNullable": 130131,
    "licenseNameNullable": "CC BY 4.0",
}
FILES = {
    "datasetFiles": [
        {"name": "Maths.csv", "totalBytes": 50669},
        {"name": "Portuguese.csv", "totalBytes": 79462},
    ]
}


def _listing(**overrides):
    record = make_kaggle_record(description="", **overrides)
    candidate = normalize_kaggle_record(record)
    assert candidate is not None
    return candidate


# ---------------------------------------------------------------------------
# apply_kaggle_detail (pure)
# ---------------------------------------------------------------------------


class TestApplyKaggleDetail:
    def test_a_listing_candidate_starts_at_listing_depth(self):
        assert _listing().metadata_depth is MetadataDepth.LISTING

    def test_the_owners_description_is_added_and_depth_is_detailed(self):
        enriched = apply_kaggle_detail(_listing(), VIEW, FILES)
        assert enriched.metadata_depth is MetadataDepth.DETAILED
        assert "Columns: attendance" in enriched.description
        assert enriched.description.startswith("Predict student performance")

    def test_the_input_candidate_is_never_mutated(self):
        original = _listing()
        before = original.model_dump()
        apply_kaggle_detail(original, VIEW, FILES)
        assert original.model_dump() == before

    def test_file_names_extensions_and_count_come_straight_from_the_registry(self):
        enriched = apply_kaggle_detail(_listing(), VIEW, FILES)
        assert enriched.file_count == 2
        assert enriched.file_names == ["Maths.csv", "Portuguese.csv"]
        assert enriched.file_types == [".csv"]

    def test_file_names_are_capped_but_the_count_is_not(self):
        many = {"datasetFiles": [{"name": f"part{i}.csv"} for i in range(MAX_FILE_NAMES + 15)]}
        enriched = apply_kaggle_detail(_listing(), VIEW, many)
        assert len(enriched.file_names) == MAX_FILE_NAMES
        assert enriched.file_count == MAX_FILE_NAMES + 15

    def test_an_overlong_description_is_capped(self):
        view = {"descriptionNullable": "x" * (MAX_DESCRIPTION_CHARS + 500)}
        enriched = apply_kaggle_detail(_listing(), view, None)
        assert len(enriched.description) == MAX_DESCRIPTION_CHARS

    def test_a_subtitle_already_inside_the_description_is_not_repeated(self):
        view = {
            "subtitleNullable": "Student grades",
            "descriptionNullable": "Student grades and attendance for two schools.",
        }
        enriched = apply_kaggle_detail(_listing(), view, None)
        assert enriched.description.count("Student grades") == 1

    def test_existing_size_and_license_are_not_overwritten(self):
        listing = _listing(total_bytes=999, license_name="Apache 2.0")
        enriched = apply_kaggle_detail(listing, VIEW, FILES)
        assert enriched.size_bytes == 999
        assert enriched.license == "Apache 2.0"

    def test_missing_size_and_license_are_filled_from_the_record(self):
        listing = _listing()
        listing.size_bytes = None
        listing.license = None
        enriched = apply_kaggle_detail(listing, VIEW, FILES)
        assert enriched.size_bytes == 130131
        assert enriched.license == "CC BY 4.0"

    @pytest.mark.parametrize("view,files", [(None, None), ("junk", []), ({}, {}), ([], 5)])
    def test_nothing_usable_returns_the_same_candidate_at_listing_depth(self, view, files):
        listing = _listing()
        result = apply_kaggle_detail(listing, view, files)
        assert result is listing
        assert result.metadata_depth is MetadataDepth.LISTING

    def test_files_alone_still_count_as_detail(self):
        enriched = apply_kaggle_detail(_listing(), None, FILES)
        assert enriched.metadata_depth is MetadataDepth.DETAILED
        assert enriched.file_types == [".csv"]

    def test_malformed_file_entries_are_skipped(self):
        files = {"datasetFiles": ["junk", {"name": None}, {"name": "ok.csv"}, 7]}
        enriched = apply_kaggle_detail(_listing(), None, files)
        assert enriched.file_names == ["ok.csv"]

    def test_enrichment_refreshes_the_evidence_assessment(self):
        listing = _listing()
        assert listing.description  # subtitle fallback only
        enriched = apply_kaggle_detail(listing, VIEW, FILES)
        assert enriched.evidence is not None
        assert "description" in enriched.evidence.external_fields

    def test_enrichment_never_changes_verification_or_source_type(self):
        enriched = apply_kaggle_detail(_listing(), VIEW, FILES)
        assert enriched.verification_status.value == "verified"
        assert enriched.source_type.value == "verified_external"
        assert enriched.target_description is None
        assert enriched.task_types == []


# ---------------------------------------------------------------------------
# KaggleDatasetSource.enrich (mock transport, path-routed)
# ---------------------------------------------------------------------------


def _routed_source(routes: dict[str, httpx.Response | Exception], seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        for suffix, response in routes.items():
            if request.url.path.endswith(suffix):
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError(f"unexpected request {request.url}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return KaggleDatasetSource(settings=make_settings(kaggle_api_token="KGAT_secret"), client=client)


REF = "owner/student-performance"


@pytest.mark.asyncio
class TestKaggleEnrich:
    async def test_reads_the_description_and_the_file_list(self):
        source = _routed_source(
            {
                f"/datasets/view/{REF}": httpx.Response(200, json=VIEW),
                f"/datasets/list/{REF}": httpx.Response(200, json=FILES),
            }
        )
        enriched = await source.enrich(_listing())
        assert enriched.metadata_depth is MetadataDepth.DETAILED
        assert enriched.file_types == [".csv"]
        assert "Columns:" in enriched.description

    async def test_a_failed_file_list_still_applies_the_description(self):
        source = _routed_source(
            {
                f"/datasets/view/{REF}": httpx.Response(200, json=VIEW),
                f"/datasets/list/{REF}": httpx.Response(500),
            }
        )
        enriched = await source.enrich(_listing())
        assert enriched.metadata_depth is MetadataDepth.DETAILED
        assert enriched.file_types == []
        assert "Columns:" in enriched.description

    async def test_a_failed_description_call_raises_a_sanitized_error(self):
        source = _routed_source(
            {
                f"/datasets/view/{REF}": httpx.Response(503, text="KGAT_secret internal trace"),
                f"/datasets/list/{REF}": httpx.Response(200, json=FILES),
            }
        )
        with pytest.raises(DatasetSourceError) as raised:
            await source.enrich(_listing())
        assert raised.value.code == "KAGGLE_UNAVAILABLE"
        assert "KGAT_secret" not in raised.value.message
        assert "internal trace" not in raised.value.message

    async def test_not_found_has_its_own_code(self):
        source = _routed_source(
            {
                f"/datasets/view/{REF}": httpx.Response(404),
                f"/datasets/list/{REF}": httpx.Response(404),
            }
        )
        with pytest.raises(DatasetSourceError) as raised:
            await source.enrich(_listing())
        assert raised.value.code == "KAGGLE_NOT_FOUND"

    async def test_timeout_maps_to_the_timeout_code(self):
        source = _routed_source(
            {
                f"/datasets/view/{REF}": httpx.ReadTimeout("slow"),
                f"/datasets/list/{REF}": httpx.ReadTimeout("slow"),
            }
        )
        with pytest.raises(DatasetSourceError) as raised:
            await source.enrich(_listing())
        assert raised.value.code == "KAGGLE_TIMEOUT"

    async def test_a_malformed_ref_is_rejected_before_any_request(self):
        seen: list = []
        source = _routed_source({}, seen)
        bad = _listing()
        bad.source_id = "no-slash"
        with pytest.raises(DatasetSourceError) as raised:
            await source.enrich(bad)
        assert raised.value.code == "DATASET_SOURCE_INVALID_ID"
        assert seen == []

    async def test_the_token_is_a_header_and_never_in_the_url(self):
        seen: list = []
        source = _routed_source(
            {
                f"/datasets/view/{REF}": httpx.Response(200, json=VIEW),
                f"/datasets/list/{REF}": httpx.Response(200, json=FILES),
            },
            seen,
        )
        await source.enrich(_listing())
        assert len(seen) == 2
        assert all("KGAT_secret" not in str(r.url) for r in seen)
        assert all(r.headers["authorization"] == "Bearer KGAT_secret" for r in seen)

    async def test_the_source_advertises_enrichment(self):
        assert _routed_source({}).supports_enrichment is True


# ---------------------------------------------------------------------------
# What enrichment changes about the conclusions
# ---------------------------------------------------------------------------


def _intel(candidate, requirements):
    ranked = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW).rank(
        [candidate], requirements
    )[0]
    return build_dataset_intelligence(ranked, requirements)


class TestEnrichmentChangesTheConclusions:
    REQ = dict(features=["attendance", "bmi"], target="final grade", tasks=["regression"])

    def test_before_enrichment_nothing_about_columns_is_known(self):
        listing = _listing()
        intel = _intel(listing, make_requirements(**self.REQ))
        assert intel.schema_state is SchemaState.UNKNOWN
        assert FeatureMatchState.MISSING not in {
            f.state for f in intel.feature_coverage.all_features
        }

    def test_after_enrichment_a_published_column_list_can_prove_a_feature_present_or_absent(self):
        enriched = apply_kaggle_detail(_listing(), VIEW, FILES)
        intel = _intel(enriched, make_requirements(**self.REQ))
        states = {f.requirement: f.state for f in intel.feature_coverage.all_features}
        assert intel.schema_state is SchemaState.PARTIAL
        assert states["attendance"] in {FeatureMatchState.EXACT, FeatureMatchState.ALIAS}
        assert states["bmi"] is FeatureMatchState.MISSING

    def test_the_schema_is_never_reported_as_verified_from_prose(self):
        enriched = apply_kaggle_detail(_listing(), VIEW, FILES)
        assert _intel(enriched, make_requirements(**self.REQ)).schema_state is not SchemaState.VERIFIED

    def test_published_file_extensions_decide_modality(self):
        image_only = {"datasetFiles": [{"name": "scan1.png"}, {"name": "scan2.png"}]}
        enriched = apply_kaggle_detail(_listing(), VIEW, image_only)
        req = make_requirements(data_types=["tabular measurements"])
        ranked = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW).rank([enriched], req)[0]
        factor = next(f for f in ranked.ranking_factors if f.name.value == "compatibility")
        assert any("image" in c for c in factor.contradictions)


# ---------------------------------------------------------------------------
# DatasetDeduplicator.absorb
# ---------------------------------------------------------------------------


class TestAbsorb:
    def test_a_later_duplicate_is_merged_into_the_indexed_record(self):
        dedup = DatasetDeduplicator()
        first = make_candidate(name="Student Performance", url="https://x.com/a")
        dedup.deduplicate([first])
        later = make_candidate(name="Student Performance", url="https://x.com/a", source_id="o/b")
        survivor = dedup.absorb(later)
        assert survivor is first
        assert first.merged_candidate_count == 2

    def test_a_non_duplicate_is_not_indexed_and_returns_none(self):
        dedup = DatasetDeduplicator()
        dedup.deduplicate([make_candidate(name="Student Performance")])
        stranger = make_candidate(name="Heart Disease", source_id="o/h")
        assert dedup.absorb(stranger) is None
        # and it was not silently indexed
        assert dedup.absorb(stranger.model_copy(deep=True)) is None

    def test_the_verified_record_wins_when_it_arrives_second(self):
        from app.services.datasets.normalizer import normalize_genai_candidate
        from tests.conftest import make_genai_candidate

        dedup = DatasetDeduplicator()
        ai = normalize_genai_candidate(make_genai_candidate(name="Student Performance"))
        dedup.deduplicate([ai])
        verified = make_candidate(name="Student Performance", source_id="o/v")
        survivor = dedup.absorb(verified)
        assert survivor is verified
        assert survivor.ai_suggested_claims is not None
        # the surviving record is what later arrivals now match against
        again = make_candidate(name="Student Performance", source_id="o/w")
        assert dedup.absorb(again) is verified
