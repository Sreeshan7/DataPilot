"""Resolving what the user pointed at, then researching it in the registries.

Covers the product's trust rules for the "dataset known only by name or URL" path: an exact
match resolves, several matches are reported as probable with alternatives (never silently
collapsed), a near match is ambiguous, a missing dataset is UNRESOLVED rather than invented,
and the registry's word is never presented as something DataPilot parsed.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.schemas.dataset_analysis import EvidenceLevel, InputType, ResolutionStatus
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisError
from app.services.dataset_analysis.external.service import (
    CONFIDENCE_AUTHORITATIVE_AMONG_MANY,
    CONFIDENCE_SINGLE_MATCH,
    CONFIDENCE_USER_UPLOADED_AMONG_MANY,
    ExternalDatasetMetadataService,
)
from app.services.dataset_analysis.input_resolver import DatasetInputResolver, normalize_source
from tests.unit.registry_mock import (
    MockRegistries,
    hf_record,
    kaggle_item,
    kaggle_view,
    openml_desc,
    openml_features,
    openml_list,
    openml_qualities,
    uci_record,
)

resolver = DatasetInputResolver()


def settings(**kw) -> Settings:
    token = kw.pop("kaggle_api_token", "")
    return Settings(gemini_api_key="", kaggle_api_token=token, app_env="testing", **kw)


def service(mock: MockRegistries, **kw) -> ExternalDatasetMetadataService:
    s = settings(**kw)
    return ExternalDatasetMetadataService(s, mock.fetcher(s))


def no_results(mock):
    """Every registry answers a search with nothing."""
    return (mock.route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": []})
            .route("openml.org/api/v1/json/data/list", lambda r: httpx.Response(412, json={"error": {"code": "372"}}))
            .route("huggingface.co/api/datasets?", [])
            .route("kaggle.com/api/v1/datasets/list?", []))


# ---------------------------------------------------------------------------
# The offline input resolver
# ---------------------------------------------------------------------------


class TestInputResolver:
    def test_an_uploaded_file_needs_no_external_resolution(self):
        r = resolver.resolve(has_file=True, file_name="../../etc/Student Marks.csv")
        assert r.input_type is InputType.UPLOADED_FILE and r.raw_file_available is True
        assert r.resolution_status is ResolutionStatus.NOT_APPLICABLE and r.resolution_confidence == 1.0
        assert r.normalized_dataset_name == "Student Marks" and r.source == "upload"
        assert r.schema_available is False  # only parsing may say otherwise

    def test_the_file_wins_over_a_name_and_says_so(self):
        r = resolver.resolve(has_file=True, file_name="a.csv", dataset_name="Titanic")
        assert r.input_type is InputType.UPLOADED_FILE and r.normalized_dataset_name == "Titanic"
        assert any("the uploaded file was analysed, not the registry" in e for e in r.evidence)

    def test_the_file_wins_over_a_url_and_says_so(self):
        r = resolver.resolve(has_file=True, file_name="a.csv", dataset_url="https://www.kaggle.com/datasets/a/b")
        assert r.input_type is InputType.UPLOADED_FILE and r.source == "upload"

    @pytest.mark.parametrize(
        ("url", "source", "identifier"),
        [("https://www.kaggle.com/datasets/heptapod/titanic", "kaggle", "heptapod/titanic"),
         ("https://huggingface.co/datasets/phihung/titanic", "huggingface", "phihung/titanic"),
         ("https://archive.ics.uci.edu/dataset/53/iris", "uci", "53"),
         ("https://www.openml.org/d/61", "openml", "61")],
    )
    def test_a_registry_url_is_resolved_syntactically(self, url, source, identifier):
        r = resolver.resolve(dataset_url=url)
        assert r.input_type is InputType.DATASET_URL and (r.source, r.source_identifier) == (source, identifier)
        assert r.resolution_status is ResolutionStatus.RESOLVED and r.resolution_confidence == 0.97

    def test_a_url_inside_the_message_is_found(self):
        r = resolver.resolve(message="Analyze https://www.openml.org/d/61 please.")
        assert r.source == "openml" and r.source_identifier == "61"

    @pytest.mark.parametrize(
        "url",
        ["https://evil.example/data.csv", "https://127.0.0.1/x", "https://www.kaggle.com/competitions/titanic",
         "https://user:pw@www.kaggle.com/datasets/a/b", "https://www.kaggle.com/someone"],
    )
    def test_an_unsafe_or_unsupported_url_is_unsupported_with_advice(self, url):
        r = resolver.resolve(dataset_url=url)
        assert r.input_type is InputType.DATASET_URL and r.resolution_status is ResolutionStatus.UNSUPPORTED
        assert r.clarification and r.source is None

    def test_a_dataset_name_field_is_a_public_dataset_name(self):
        r = resolver.resolve(dataset_name="Titanic")
        assert r.input_type is InputType.PUBLIC_DATASET_NAME and r.normalized_dataset_name == "titanic"
        assert r.resolution_status is ResolutionStatus.UNRESOLVED  # nothing has confirmed it yet

    def test_a_name_in_the_message_is_extracted(self):
        r = resolver.resolve(message="I have the Titanic dataset. What can I build?")
        assert r.input_type is InputType.PUBLIC_DATASET_NAME and r.normalized_dataset_name == "titanic"

    def test_a_source_named_in_the_message_restricts_the_search(self):
        r = resolver.resolve(message="Analyze the Kaggle Titanic dataset")
        assert r.source == "kaggle"

    def test_an_explicit_source_hint_is_normalised(self):
        assert resolver.resolve(dataset_name="Iris", source_hint="Hugging Face").source == "huggingface"
        assert normalize_source("OpenML") == "openml" and normalize_source("nonsense") is None

    @pytest.mark.parametrize("name", ["student", "Customers", "sales", "health"])
    def test_a_bare_topic_word_is_ambiguous_and_asks(self, name):
        r = resolver.resolve(dataset_name=name)
        assert r.input_type is InputType.AMBIGUOUS_DATASET_REFERENCE
        assert r.needs_confirmation and "too vague" in " ".join(r.evidence)

    def test_a_vague_owned_dataset_in_the_message_is_ambiguous(self):
        r = resolver.resolve(message="I downloaded a student dataset")
        assert r.input_type is InputType.AMBIGUOUS_DATASET_REFERENCE

    def test_nothing_at_all_asks_for_something(self):
        r = resolver.resolve()
        assert r.input_type is InputType.NO_DATASET_REFERENCE and r.clarification

    def test_a_specific_multiword_name_is_not_vague(self):
        r = resolver.resolve(dataset_name="Student Performance")
        assert r.input_type is InputType.PUBLIC_DATASET_NAME


# ---------------------------------------------------------------------------
# Research by link
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResearchByUrl:
    async def test_a_uci_link_returns_registry_verified_metadata(self):
        mock = MockRegistries().route("archive.ics.uci.edu/api/dataset", uci_record())
        research = await service(mock).research(resolver.resolve(dataset_url="https://archive.ics.uci.edu/dataset/53/iris"))
        r = research.resolution
        assert r.resolution_status is ResolutionStatus.RESOLVED and r.schema_available
        assert r.schema_evidence_level is EvidenceLevel.REGISTRY_VERIFIED and r.raw_file_available is False
        assert {"description", "schema", "row_count"} <= set(r.available_metadata)
        assert research.metadata.row_count == 150

    async def test_a_kaggle_link_without_a_structured_schema_is_inferred_at_best(self):
        mock = (MockRegistries().route("/datasets/view/", kaggle_view())
                .route("/datasets/list/owner/titanic", {"datasetFiles": [{"name": "t.csv"}]}))
        research = await service(mock).research(resolver.resolve(dataset_url="https://www.kaggle.com/datasets/owner/titanic"))
        assert research.resolution.schema_evidence_level is EvidenceLevel.INFERRED
        assert research.resolution.schema_evidence_level is not EvidenceLevel.REGISTRY_VERIFIED

    async def test_a_kaggle_link_with_no_column_list_has_no_schema(self):
        mock = (MockRegistries().route("/datasets/view/", kaggle_view(description="A famous dataset."))
                .route("/datasets/list/owner/titanic", {"datasetFiles": []}))
        research = await service(mock).research(resolver.resolve(dataset_url="https://www.kaggle.com/datasets/owner/titanic"))
        assert research.resolution.schema_available is False
        assert research.resolution.schema_evidence_level is EvidenceLevel.UNKNOWN
        assert "schema" not in research.resolution.available_metadata

    async def test_a_dataset_that_does_not_exist_is_unresolved_not_an_error(self):
        mock = MockRegistries().route("kaggle.com", lambda r: httpx.Response(404))
        research = await service(mock).research(resolver.resolve(dataset_url="https://www.kaggle.com/datasets/a/none"))
        assert research.metadata is None
        assert research.resolution.resolution_status is ResolutionStatus.UNRESOLVED
        assert any("nothing about its structure could be verified" in l for l in research.limitations)

    @pytest.mark.parametrize(
        ("status", "code", "http"),
        [(429, ErrorCode.DATASET_RATE_LIMITED, 429), (500, ErrorCode.DATASET_SOURCE_UNAVAILABLE, 502)],
    )
    async def test_an_upstream_failure_is_a_mapped_error(self, status, code, http):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(status, text="secret internals"))
        with pytest.raises(AnalysisError) as err:
            await service(mock).research(resolver.resolve(dataset_url="https://huggingface.co/datasets/o/x"))
        assert err.value.code is code and err.value.status_code == http
        assert "secret internals" not in err.value.message

    async def test_a_timeout_is_a_504(self):
        def slow(request):
            raise httpx.ReadTimeout("slow")

        mock = MockRegistries().route("openml.org", slow)
        with pytest.raises(AnalysisError) as err:
            await service(mock).research(resolver.resolve(dataset_url="https://www.openml.org/d/61"))
        assert err.value.code is ErrorCode.DATASET_SOURCE_TIMEOUT and err.value.status_code == 504

    async def test_a_disabled_registry_is_a_clear_error(self):
        mock = MockRegistries()
        with pytest.raises(AnalysisError) as err:
            await service(mock, huggingface_enabled=False).research(
                resolver.resolve(dataset_url="https://huggingface.co/datasets/o/x"))
        assert err.value.code is ErrorCode.DATASET_URL_UNSUPPORTED and mock.requests == []

    async def test_research_can_be_switched_off_and_makes_no_request(self):
        mock = MockRegistries()
        research = await service(mock, external_enabled=False).research(
            resolver.resolve(dataset_url="https://www.openml.org/d/61"))
        assert research.metadata is None and mock.requests == []
        assert any("disabled" in l for l in research.limitations)

    async def test_an_ambiguous_reference_triggers_no_research(self):
        mock = MockRegistries()
        research = await service(mock).research(resolver.resolve(dataset_name="student"))
        assert research.metadata is None and mock.requests == []


# ---------------------------------------------------------------------------
# Research by name
# ---------------------------------------------------------------------------


def one_iris_everywhere(mock):
    return (no_results(mock)
            .route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
            .route("archive.ics.uci.edu/api/dataset?", uci_record()))


@pytest.mark.asyncio
class TestResearchByName:
    async def test_a_single_exact_match_is_resolved(self):
        mock = one_iris_everywhere(MockRegistries())
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        r = research.resolution
        assert r.resolution_status is ResolutionStatus.RESOLVED and r.resolution_confidence == CONFIDENCE_SINGLE_MATCH
        assert r.source == "uci" and r.source_identifier == "53" and r.alternatives == []
        assert not r.needs_confirmation and research.conflicts == []

    async def test_a_name_search_never_downloads_the_data(self):
        mock = one_iris_everywhere(MockRegistries())
        await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert not any(u.endswith(".csv") or "static/public" in u for u in mock.urls())

    async def test_several_exact_matches_are_probable_with_alternatives_and_a_clarification(self):
        mock = (no_results(MockRegistries())
                .route("openml.org/api/v1/json/data/list", openml_list((40704, "Titanic", 2, 2201, 4), (42437, "titanic", 6, 891, 12)))
                .route("/data/features/40704", openml_features(("Class", "numeric", False), ("Age", "numeric", False), ("Sex", "numeric", False), ("class", "nominal", True)))
                .route("/data/features/42437", openml_features(("a", "numeric", False), ("class", "nominal", True)))
                .route("/data/qualities/", openml_qualities(NumberOfInstances=2201))
                .route("/data/40704", openml_desc(40704, "Titanic", 2))
                .route("/data/42437", openml_desc(42437, "titanic", 6)))
        research = await service(mock).research(resolver.resolve(dataset_name="Titanic"))
        r = research.resolution
        assert r.resolution_status is ResolutionStatus.PROBABLE and r.needs_confirmation
        assert r.resolution_confidence == CONFIDENCE_AUTHORITATIVE_AMONG_MANY
        assert len(r.alternatives) == 1 and r.alternatives[0].row_count in (891, 2201)
        assert "share the name" in r.clarification
        assert any("confirm which one you mean" in l for l in research.limitations)

    async def test_the_most_authoritative_registry_wins_among_exact_matches(self):
        mock = (no_results(MockRegistries())
                .route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
                .route("archive.ics.uci.edu/api/dataset?", uci_record())
                .route("huggingface.co/api/datasets?", [{"id": "o/iris", "downloads": 10**7}])
                .route("huggingface.co/api/datasets/o/iris", hf_record("o/iris"))
                .route("kaggle.com/api/v1/datasets/list?", [kaggle_item("o/iris", "Iris", 10**8)]))
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert research.resolution.source == "uci"  # popularity does not outrank origin
        assert {a.source for a in research.resolution.alternatives} == {"huggingface", "kaggle"}

    async def test_among_user_uploaded_hubs_the_confidence_is_lower(self):
        mock = (no_results(MockRegistries())
                .route("huggingface.co/api/datasets?", [{"id": "o/titanic", "downloads": 500}, {"id": "p/titanic", "downloads": 9000}])
                .route("huggingface.co/api/datasets/p/titanic", hf_record("p/titanic"))
                .route("huggingface.co/api/datasets/o/titanic", hf_record("o/titanic")))
        research = await service(mock).research(resolver.resolve(dataset_name="Titanic"))
        assert research.resolution.resolution_confidence == CONFIDENCE_USER_UPLOADED_AMONG_MANY
        assert research.resolution.source_identifier == "p/titanic"  # popularity breaks ties within a registry

    async def test_a_near_match_is_ambiguous_with_candidates_and_nothing_analysed(self):
        mock = (no_results(MockRegistries())
                .route("kaggle.com/api/v1/datasets/list?", [kaggle_item("a/b", "Titanic Survival Prediction"), kaggle_item("c/d", "Titanic Passenger Data")]))
        research = await service(mock).research(resolver.resolve(dataset_name="Titanic"))
        r = research.resolution
        assert r.resolution_status is ResolutionStatus.AMBIGUOUS and research.metadata is None
        assert r.input_type is InputType.AMBIGUOUS_DATASET_REFERENCE and len(r.alternatives) == 2
        assert r.needs_confirmation and r.clarification

    async def test_an_unknown_name_is_unresolved_and_says_nothing_was_verified(self):
        research = await service(no_results(MockRegistries())).research(resolver.resolve(dataset_name="Zzyzx Totally Fake"))
        assert research.resolution.resolution_status is ResolutionStatus.UNRESOLVED and research.metadata is None
        assert research.resolution.schema_available is False
        assert any("nothing about its structure could be verified" in l for l in research.limitations)

    async def test_a_source_hint_searches_only_that_registry(self):
        mock = (no_results(MockRegistries())
                .route("kaggle.com/api/v1/datasets/list?", [kaggle_item("o/titanic", "Titanic")])
                .route("/datasets/view/", kaggle_view()).route("/datasets/list/o/titanic", {"datasetFiles": []}))
        research = await service(mock).research(resolver.resolve(message="Analyze the Kaggle Titanic dataset"))
        assert research.resolution.source == "kaggle" and set(mock.hosts()) == {"www.kaggle.com"}

    async def test_a_partial_registry_failure_is_reported_but_does_not_stop_the_research(self):
        mock = (one_iris_everywhere(MockRegistries())
                .route("huggingface.co", lambda r: httpx.Response(500)))
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert research.metadata is not None
        assert any("huggingface" in l and "could not be searched" in l for l in research.limitations)

    async def test_when_every_registry_fails_the_error_is_surfaced(self):
        mock = MockRegistries()  # default: 500 everywhere
        with pytest.raises(AnalysisError) as err:
            await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert err.value.status_code == 502

    async def test_if_the_best_match_cannot_be_read_the_next_is_tried(self):
        mock = (no_results(MockRegistries())
                .route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
                .route("archive.ics.uci.edu/api/dataset?", {"status": 404})
                .route("huggingface.co/api/datasets?", [{"id": "o/iris", "downloads": 5}])
                .route("huggingface.co/api/datasets/o/iris", hf_record("o/iris")))
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert research.resolution.source == "huggingface"

    async def test_a_listed_dataset_whose_record_is_gone_is_unresolved(self):
        mock = (no_results(MockRegistries())
                .route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
                .route("archive.ics.uci.edu/api/dataset?", {"status": 404}))
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert research.metadata is None and research.resolution.resolution_status is ResolutionStatus.UNRESOLVED


@pytest.mark.asyncio
class TestConflicts:
    def two_registries(self, hf_features, uci_variables=None):
        mock = (no_results(MockRegistries())
                .route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 9, "name": "Titanic"}]})
                .route("archive.ics.uci.edu/api/dataset?", uci_record(9, "Titanic", 2201, variables=uci_variables or [
                    {"name": "class", "role": "Feature", "type": "Categorical"}, {"name": "age", "role": "Feature", "type": "Categorical"},
                    {"name": "survived", "role": "Target", "type": "Binary"}]))
                .route("huggingface.co/api/datasets?", [{"id": "o/titanic", "downloads": 5}])
                .route("huggingface.co/api/datasets/o/titanic", hf_record("o/titanic", features=hf_features, rows=891)))
        return mock

    async def test_different_column_sets_are_reported_as_a_conflict(self):
        mock = self.two_registries([{"name": "survived", "dtype": "int64"}, {"name": "fare", "dtype": "float64"}, {"name": "cabin", "dtype": "string"}])
        research = await service(mock).research(resolver.resolve(dataset_name="Titanic"))
        columns = [c for c in research.conflicts if c.field == "columns"]
        assert columns and "different columns" in columns[0].note and len(columns[0].values) == 2

    async def test_identical_column_sets_are_not_a_conflict(self):
        mock = self.two_registries([{"name": "Class", "dtype": "string"}, {"name": "AGE", "dtype": "string"}, {"name": "survived", "dtype": "int64"}])
        research = await service(mock).research(resolver.resolve(dataset_name="Titanic"))
        assert not [c for c in research.conflicts if c.field == "columns"]

    async def test_different_row_counts_are_reported(self):
        mock = (no_results(MockRegistries())
                .route("openml.org/api/v1/json/data/list", openml_list((1, "Titanic", 1, 2201, 4), (2, "Titanic", 2, 891, 12)))
                .route("/data/features/", openml_features(("a", "numeric", False), ("y", "nominal", True)))
                .route("/data/qualities/1", openml_qualities(NumberOfInstances=2201))
                .route("/data/qualities/2", openml_qualities(NumberOfInstances=891))
                .route("/data/1", openml_desc(1, "Titanic", 1)).route("/data/2", openml_desc(2, "Titanic", 2)))
        research = await service(mock).research(resolver.resolve(dataset_name="Titanic"))
        assert [c.field for c in research.conflicts] == ["row_count"]


@pytest.mark.asyncio
class TestNeverClaimsRawData:
    async def test_no_statistics_are_attached_to_researched_metadata(self):
        mock = one_iris_everywhere(MockRegistries())
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert research.resolution.raw_file_available is False
        dumped = research.metadata.model_dump_json()
        for forbidden in ("mean", "median", "std", "quantile", "missing_pct", "uniqueness_ratio"):
            assert forbidden not in dumped

    async def test_every_schema_column_carries_a_registry_or_weaker_level(self):
        mock = one_iris_everywhere(MockRegistries())
        research = await service(mock).research(resolver.resolve(dataset_name="Iris"))
        assert all(c.evidence is not EvidenceLevel.DATAPILOT_PARSED for c in research.metadata.schema_columns)

    async def test_no_credential_appears_in_any_research_output(self):
        mock = (MockRegistries().route("/datasets/view/", kaggle_view())
                .route("/datasets/list/owner/titanic", {"datasetFiles": []}))
        research = await service(mock, kaggle_api_token="KGAT_super_secret").research(
            resolver.resolve(dataset_url="https://www.kaggle.com/datasets/owner/titanic"))
        assert "KGAT_super_secret" not in research.resolution.model_dump_json() + research.metadata.model_dump_json()
