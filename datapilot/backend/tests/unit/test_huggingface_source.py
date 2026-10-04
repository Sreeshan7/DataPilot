"""Phase 3C: the Hugging Face Hub as a second verified registry.

Payload shapes are taken from live ``/api/datasets?full=true`` responses. What
matters here is not that rows parse, but that every claim a Hub row makes is mapped
to the right authority: task categories become declared tasks, a structured
``dataset_info.features`` list becomes the only thing allowed to be a verified
schema, and nothing is inferred beyond the topic domain.
"""

from __future__ import annotations

import httpx
import pytest

from app.schemas.dataset import (
    MetadataDepth,
    SourceSearchRequest,
    SourceType,
    VerificationStatus,
)
from app.schemas.intelligence import (
    CompatibilityLevel,
    EvidenceLayer,
    FeatureMatchState,
    SchemaState,
)
from app.services.datasets import build_default_sources
from app.services.datasets.base import DatasetSourceError
from app.services.datasets.deduplicator import DatasetDeduplicator
from app.services.datasets.huggingface_source import (
    HuggingFaceDatasetSource,
    query_heads,
)
from app.services.datasets.intelligence import build_dataset_intelligence
from app.services.datasets.normalizer import (
    normalize_huggingface_record,
    normalize_huggingface_records,
)
from app.services.datasets.ranking import DatasetRankingEngine
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_requirements,
    make_settings,
)


def hf_record(**overrides) -> dict:
    """A realistic Hub row that publishes a card, tasks, formats and a schema."""
    record = {
        "id": "riyadahmadov/StudentPerformance",
        "author": "riyadahmadov",
        "downloads": 20,
        "likes": 2,
        "private": False,
        "disabled": False,
        "gated": False,
        "lastModified": "2024-04-23T10:00:00.000Z",
        "createdAt": "2024-04-20T10:00:00.000Z",
        "description": "\n\t\n\t\tStudent performance\n\t\nGrades and attendance for two schools.",
        "tags": [
            "task_categories:tabular-classification",
            "language:en",
            "license:cc-by-4.0",
            "size_categories:1K<n<10K",
            "format:csv",
            "modality:tabular",
            "modality:text",
            "library:datasets",
            "region:us",
            "education",
        ],
        "cardData": {
            "pretty_name": "Student Performance",
            "dataset_info": {
                "features": [
                    {"name": "attendance", "dtype": "float64"},
                    {"name": "study_hours", "dtype": "float64"},
                    {"name": "final_grade", "dtype": "int64"},
                ],
                "dataset_size": 48213,
            },
        },
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_a_row_becomes_a_verified_external_candidate(self):
        c = normalize_huggingface_record(hf_record())
        assert c.id == "huggingface:riyadahmadov/StudentPerformance"
        assert c.source == "huggingface"
        assert c.source_type is SourceType.VERIFIED_EXTERNAL
        assert c.verification_status is VerificationStatus.VERIFIED
        assert c.url == "https://huggingface.co/datasets/riyadahmadov/StudentPerformance"
        assert c.owner == "riyadahmadov"
        assert c.name == "Student Performance"

    def test_task_categories_become_declared_task_types(self):
        c = normalize_huggingface_record(hf_record())
        assert c.task_types == ["tabular classification"]

    def test_format_and_modality_tags_become_file_types_and_modalities(self):
        c = normalize_huggingface_record(hf_record())
        assert c.file_types == [".csv"]
        assert c.modalities == ["tabular", "text"]

    def test_license_downloads_likes_and_dates_are_read_from_the_record(self):
        c = normalize_huggingface_record(hf_record())
        assert c.license == "cc-by-4.0"
        assert c.download_count == 20 and c.vote_count == 2
        assert c.updated_at.year == 2024 and c.created_at.year == 2024
        assert c.size_bytes == 48213

    def test_a_structured_schema_becomes_columns_and_a_declared_feature_description(self):
        c = normalize_huggingface_record(hf_record())
        assert c.columns == ["attendance", "study_hours", "final_grade"]
        assert c.feature_description == "columns: attendance, study_hours, final_grade"

    def test_no_schema_means_no_columns_and_no_feature_description(self):
        c = normalize_huggingface_record(hf_record(cardData={"pretty_name": "X"}))
        assert c.columns == []
        assert c.feature_description is None

    def test_dataset_info_as_a_list_of_configs_is_read(self):
        card = {"dataset_info": [{"features": [{"name": "a"}]}, {"features": [{"name": "b"}, {"name": "a"}]}]}
        assert normalize_huggingface_record(hf_record(cardData=card)).columns == ["a", "b"]

    def test_packaging_tags_are_not_topics(self):
        c = normalize_huggingface_record(hf_record())
        assert c.tags == ["education"]

    def test_the_topic_domain_is_derived_not_published(self):
        c = normalize_huggingface_record(hf_record())
        assert c.domain == "education"
        assert "domain" in c.evidence.inferred_fields

    def test_the_card_is_already_in_the_search_response_so_depth_is_detailed(self):
        assert normalize_huggingface_record(hf_record()).metadata_depth is MetadataDepth.DETAILED

    def test_target_description_is_never_invented(self):
        assert normalize_huggingface_record(hf_record()).target_description is None

    def test_an_unpublished_target_is_explained_not_reported_as_a_gap(self):
        notes = " ".join(normalize_huggingface_record(hf_record()).evidence.notes)
        assert "huggingface does not publish: target_description" in notes

    def test_description_whitespace_is_collapsed(self):
        c = normalize_huggingface_record(hf_record())
        assert c.description == "Student performance Grades and attendance for two schools."

    @pytest.mark.parametrize("flag", ["private", "disabled"])
    def test_private_and_disabled_datasets_are_skipped(self, flag):
        assert normalize_huggingface_record(hf_record(**{flag: True})) is None

    @pytest.mark.parametrize("bad", [None, "x", 5, [], {}, {"id": ""}, {"id": None}])
    def test_malformed_rows_are_skipped(self, bad):
        assert normalize_huggingface_record(bad) is None

    def test_a_legacy_single_segment_id_has_no_owner_and_a_readable_name(self):
        c = normalize_huggingface_record(hf_record(id="squad_v2", author=None, cardData={}))
        assert c.owner is None
        assert c.name == "squad v2"

    def test_one_bad_row_does_not_discard_the_rest(self):
        out = normalize_huggingface_records([hf_record(), "junk", hf_record(id="a/b")])
        assert [c.source_id for c in out] == ["riyadahmadov/StudentPerformance", "a/b"]

    def test_a_non_list_payload_yields_nothing(self):
        assert normalize_huggingface_records({"error": "x"}) == []

    def test_every_candidate_is_serializable_with_evidence_attached(self):
        c = normalize_huggingface_record(hf_record())
        assert c.evidence is not None
        c.model_dump_json()


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------


def _source(handler, **settings):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HuggingFaceDatasetSource(settings=make_settings(**settings), client=client)


def _request(query="student performance attendance examination marks assignment", limit=10):
    return SourceSearchRequest(query=query, limit=limit)


class TestQueryHeads:
    def test_a_long_query_is_tried_as_two_words_then_one(self):
        assert query_heads("student performance attendance marks") == [
            "student performance",
            "student",
        ]

    def test_a_single_word_query_is_not_repeated(self):
        assert query_heads("diabetes") == ["diabetes"]

    def test_the_full_query_is_never_sent(self):
        assert "student performance attendance marks" not in query_heads(
            "student performance attendance marks"
        )


@pytest.mark.asyncio
class TestSearch:
    async def test_returns_normalized_verified_candidates(self):
        source = _source(lambda r: httpx.Response(200, json=[hf_record()]))
        out = await source.search(_request())
        assert [c.source for c in out] == ["huggingface"]

    async def test_falls_back_to_a_shorter_query_when_the_first_finds_nothing(self):
        seen: list[str] = []

        def handler(request):
            seen.append(request.url.params["search"])
            return httpx.Response(200, json=[] if len(seen) == 1 else [hf_record()])

        out = await _source(handler).search(_request())
        assert seen == ["student performance", "student"]
        assert len(out) == 1

    async def test_stops_at_the_first_query_that_returns_rows(self):
        seen: list[str] = []

        def handler(request):
            seen.append(request.url.params["search"])
            return httpx.Response(200, json=[hf_record()])

        await _source(handler).search(_request())
        assert len(seen) == 1

    async def test_never_makes_more_than_the_attempt_budget(self):
        seen: list = []

        def handler(request):
            seen.append(1)
            return httpx.Response(200, json=[])

        assert await _source(handler).search(_request()) == []
        assert len(seen) == 2

    async def test_the_card_is_requested_and_the_limit_is_capped(self):
        captured = {}

        def handler(request):
            captured.update(dict(request.url.params))
            return httpx.Response(200, json=[hf_record()])

        await _source(handler, huggingface_max_limit=7).search(_request(limit=50))
        assert captured["full"] == "true"
        assert captured["limit"] == "7"

    async def test_results_are_capped_at_the_requested_limit(self):
        rows = [hf_record(id=f"o/d{i}") for i in range(10)]
        out = await _source(lambda r: httpx.Response(200, json=rows)).search(_request(limit=3))
        assert len(out) == 3

    async def test_no_credential_is_ever_sent(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=[hf_record()])

        await _source(handler, kaggle_api_token="KGAT_secret").search(_request())
        assert "authorization" not in seen[0].headers
        assert "KGAT_secret" not in str(seen[0].url)

    async def test_an_empty_query_is_rejected(self):
        with pytest.raises(DatasetSourceError) as raised:
            await _source(lambda r: httpx.Response(200, json=[])).search(_request(query="  "))
        assert raised.value.code == "DATASET_QUERY_EMPTY"

    @pytest.mark.parametrize(
        "response,code",
        [
            (httpx.Response(429), "HUGGINGFACE_RATE_LIMITED"),
            (httpx.Response(500), "HUGGINGFACE_UNAVAILABLE"),
            (httpx.Response(503, text="internal trace KGAT_secret"), "HUGGINGFACE_UNAVAILABLE"),
            (httpx.Response(404), "HUGGINGFACE_UNAVAILABLE"),
            (httpx.Response(200, text="not json"), "HUGGINGFACE_INVALID_RESPONSE"),
        ],
    )
    async def test_failures_map_to_sanitized_codes(self, response, code):
        with pytest.raises(DatasetSourceError) as raised:
            await _source(lambda r: response).search(_request())
        assert raised.value.code == code
        assert "internal trace" not in raised.value.message
        assert "KGAT_secret" not in raised.value.message

    async def test_a_timeout_and_a_connection_error_are_mapped(self):
        def slow(request):
            raise httpx.ReadTimeout("slow")

        def down(request):
            raise httpx.ConnectError("down")

        with pytest.raises(DatasetSourceError) as t:
            await _source(slow).search(_request())
        with pytest.raises(DatasetSourceError) as d:
            await _source(down).search(_request())
        assert t.value.code == "HUGGINGFACE_TIMEOUT"
        assert d.value.code == "HUGGINGFACE_UNAVAILABLE"


class TestRegistry:
    def test_it_is_registered_by_default_and_can_be_disabled(self):
        names = [s.name for s in build_default_sources(make_settings())]
        assert "huggingface" in names
        names = [s.name for s in build_default_sources(make_settings(huggingface_enabled=False))]
        assert "huggingface" not in names

    def test_it_is_a_verified_source_without_credentials(self):
        source = HuggingFaceDatasetSource(make_settings())
        assert source.source_type is SourceType.VERIFIED_EXTERNAL
        assert source.has_credentials is False
        assert source.supports_enrichment is False


# ---------------------------------------------------------------------------
# What a Hub record lets the system conclude
# ---------------------------------------------------------------------------


def _intel(candidate, requirements):
    ranked = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW).rank(
        [candidate], requirements
    )[0]
    return ranked, build_dataset_intelligence(ranked, requirements)


class TestWhatTheRecordProves:
    REQ = dict(
        features=["attendance", "study hours", "bmi"],
        target="final grade",
        tasks=["classification"],
    )

    def test_a_published_column_list_makes_the_schema_verified(self):
        _, intel = _intel(normalize_huggingface_record(hf_record()), make_requirements(**self.REQ))
        assert intel.schema_state is SchemaState.VERIFIED
        assert any(c.layer is EvidenceLayer.SCHEMA_VERIFIED and c.subject == "columns" for c in intel.evidence)

    def test_a_feature_absent_from_a_verified_schema_is_missing_and_blocks(self):
        _, intel = _intel(normalize_huggingface_record(hf_record()), make_requirements(**self.REQ))
        states = {f.requirement: f.state for f in intel.feature_coverage.all_features}
        assert states["bmi"] is FeatureMatchState.MISSING
        assert states["attendance"] in {FeatureMatchState.EXACT, FeatureMatchState.ALIAS}
        assert states["study hours"] in {FeatureMatchState.EXACT, FeatureMatchState.ALIAS}
        missing = next(f for f in intel.feature_coverage.missing if f.requirement == "bmi")
        assert missing.layer is EvidenceLayer.SCHEMA_VERIFIED
        assert intel.compatibility.overall is CompatibilityLevel.LOW

    def test_a_dataset_with_no_published_schema_is_never_verified(self):
        c = normalize_huggingface_record(hf_record(cardData={"pretty_name": "Student Performance"}))
        _, intel = _intel(c, make_requirements(**self.REQ))
        assert intel.schema_state is not SchemaState.VERIFIED
        assert FeatureMatchState.MISSING not in {f.state for f in intel.feature_coverage.all_features}

    def test_declared_task_categories_decide_task_compatibility(self):
        c = normalize_huggingface_record(hf_record())
        _, same = _intel(c, make_requirements(tasks=["classification"]))
        _, other = _intel(c, make_requirements(tasks=["time series forecasting"]))
        assert same.compatibility.task.value == "match"
        assert other.compatibility.task.value == "mismatch"

    def test_a_published_image_modality_contradicts_a_tabular_project(self):
        c = normalize_huggingface_record(
            hf_record(tags=["modality:image", "format:parquet"], cardData={})
        )
        _, intel = _intel(c, make_requirements(data_types=["tabular records"]))
        assert intel.compatibility.data_type.value == "mismatch"

    def test_a_published_tabular_modality_matches(self):
        c = normalize_huggingface_record(hf_record(tags=["modality:tabular"], cardData={}))
        _, intel = _intel(c, make_requirements(data_types=["tabular records"]))
        assert intel.compatibility.data_type.value == "match"

    def test_an_ai_candidate_can_never_carry_a_verified_schema(self):
        from app.services.datasets.normalizer import normalize_genai_candidate
        from tests.conftest import make_genai_candidate

        ai = normalize_genai_candidate(make_genai_candidate())
        ai.columns = ["attendance", "study_hours"]  # a forged claim
        _, intel = _intel(ai, make_requirements(**self.REQ))
        assert intel.schema_state is SchemaState.UNKNOWN
        assert all(c.layer is not EvidenceLayer.SCHEMA_VERIFIED for c in intel.evidence)


class TestProseListsAreNotStructuredSchemas:
    """A prose column list can show a feature is present; it cannot sink a dataset."""

    def test_a_feature_missing_from_prose_keeps_the_dataset_out_of_high_but_not_low(self):
        c = make_candidate(
            description="Student records. Columns: attendance, study hours, assignment scores.",
            target_description=None,
            file_types=[".csv"],
            metadata_depth=MetadataDepth.DETAILED,
        )
        _, intel = _intel(c, make_requirements(features=["attendance", "bmi"]))
        assert intel.schema_state is SchemaState.PARTIAL
        assert intel.feature_coverage.missing_count == 1
        assert intel.compatibility.overall is CompatibilityLevel.MEDIUM

    def test_the_risk_says_the_description_may_be_incomplete(self):
        c = make_candidate(
            description="Student records. Columns: attendance, study hours.",
            target_description=None,
        )
        _, intel = _intel(c, make_requirements(features=["attendance", "bmi"]))
        risk = next(r for r in intel.risks if r.category.value == "feature_missing")
        assert "may be incomplete" in risk.detail
        assert "published schema" not in risk.detail


# ---------------------------------------------------------------------------
# Cross-source deduplication
# ---------------------------------------------------------------------------


class TestCrossSourceDedup:
    def test_the_same_dataset_on_two_registries_collapses_to_one(self):
        kaggle = make_candidate(name="Student Performance", source_id="o/student-performance")
        hub = normalize_huggingface_record(hf_record(id="o/student_performance"))
        merged = DatasetDeduplicator().deduplicate([kaggle, hub])
        assert len(merged) == 1
        assert merged[0].merged_candidate_count == 2
        assert "huggingface" in merged[0].duplicate_sources

    def test_a_merge_keeps_both_registries_published_facts(self):
        kaggle = make_candidate(name="Student Performance", source_id="o/student-performance")
        hub = normalize_huggingface_record(hf_record(id="o/student_performance"))
        merged = DatasetDeduplicator().deduplicate([kaggle, hub])[0]
        assert merged.columns == ["attendance", "study_hours", "final_grade"]
        assert merged.task_types == ["tabular classification"]
        assert merged.modalities == ["tabular", "text"]

    def test_different_datasets_with_similar_names_stay_separate(self):
        kaggle = make_candidate(name="Student Performance in Mathematics", source_id="o/maths")
        hub = normalize_huggingface_record(hf_record(id="o/student_performance"))
        assert len(DatasetDeduplicator().deduplicate([kaggle, hub])) == 2

    def test_an_ai_suggestion_merged_into_a_hub_record_adds_no_columns(self):
        from app.services.datasets.normalizer import normalize_genai_candidate
        from tests.conftest import make_genai_candidate

        hub = normalize_huggingface_record(
            hf_record(id="o/student_performance", cardData={"pretty_name": "Student Performance"})
        )
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", features=["attendance", "bmi"])
        )
        merged = DatasetDeduplicator().deduplicate([hub, ai])[0]
        assert merged.columns == []
        assert merged.feature_description is None
        assert "bmi" in merged.ai_suggested_claims.feature_description


class TestStatedModalityBeatsExtension:
    def test_parquet_is_a_container_not_a_modality(self):
        from app.services.datasets.compatibility import _candidate_modalities

        image = normalize_huggingface_record(
            hf_record(tags=["modality:image", "format:parquet"], cardData={})
        )
        assert _candidate_modalities(image) == frozenset({"image"})

    def test_extensions_are_used_only_when_no_modality_was_stated(self):
        from app.services.datasets.compatibility import _candidate_modalities

        assert _candidate_modalities(make_candidate(file_types=[".csv"])) == frozenset({"tabular"})
        assert _candidate_modalities(make_candidate(file_types=[".png"])) == frozenset({"image"})

    def test_unrecognised_modality_labels_fall_back_to_extensions(self):
        from app.services.datasets.compatibility import _candidate_modalities

        c = make_candidate(file_types=[".csv"], modalities=["geospatial", "3d"])
        assert _candidate_modalities(c) == frozenset({"tabular"})
