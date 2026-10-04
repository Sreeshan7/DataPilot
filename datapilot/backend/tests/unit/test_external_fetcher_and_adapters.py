"""The safe fetcher and the four registry adapters, against a fake registry.

The fetcher is a security boundary, so most of its tests are refusals. The adapter tests pin
what evidence level each registry's data may carry: structured schema is REGISTRY_VERIFIED,
text read out of a description is only INFERRED, and nothing is ever filled in.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.schemas.dataset_analysis import EvidenceLevel
from app.services.dataset_analysis.external.adapters import (
    HuggingFaceAdapter,
    KaggleAdapter,
    OpenMLAdapter,
    UCIAdapter,
    build_adapters,
)
from app.services.dataset_analysis.external.fetcher import FetchError, SafeFetcher
from app.services.dataset_analysis.external.prose import extract_columns
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


def settings(**kw) -> Settings:
    return Settings(gemini_api_key="", kaggle_api_token=kw.pop("kaggle_api_token", ""), app_env="testing", **kw)


def fetcher_for(mock: MockRegistries, **kw) -> SafeFetcher:
    return mock.fetcher(settings(**kw))


async def get(fetcher, url, **kw):
    return await fetcher.get_json(url, **kw)


# ---------------------------------------------------------------------------
# SafeFetcher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFetcherSafety:
    async def test_a_registry_url_is_fetched_and_parsed(self):
        mock = MockRegistries().route("huggingface.co/api/datasets", {"ok": True})
        assert await get(fetcher_for(mock), "https://huggingface.co/api/datasets") == {"ok": True}

    @pytest.mark.parametrize(
        "url",
        ["https://evil.example/data", "http://127.0.0.1/x", "https://169.254.169.254/latest/meta-data",
         "https://user:pw@huggingface.co/api/datasets", "file:///etc/passwd", "https://localhost/x",
         "https://huggingface.co:8443/api/datasets"],
    )
    async def test_an_unsafe_address_is_refused_before_any_request_is_made(self, url):
        mock = MockRegistries()
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock), url)
        assert err.value.code == FetchError.UNSAFE and mock.requests == []

    async def test_plain_text_json_is_accepted(self):
        mock = MockRegistries().route(
            "archive.ics.uci.edu", lambda r: MockRegistries._json({"status": 200}, content_type="text/plain;charset=UTF-8")
        )
        assert (await get(fetcher_for(mock), "https://archive.ics.uci.edu/api/dataset"))["status"] == 200

    async def test_a_relative_redirect_is_followed_on_the_same_host(self):
        mock = (MockRegistries()
                .route("/api/datasets/mnist", lambda r: httpx.Response(307, headers={"location": "/api/datasets/ylecun/mnist"}))
                .route("/api/datasets/ylecun/mnist", {"id": "ylecun/mnist"}))
        assert (await get(fetcher_for(mock), "https://huggingface.co/api/datasets/mnist"))["id"] == "ylecun/mnist"
        assert len(mock.requests) == 2

    async def test_a_redirect_to_a_disallowed_host_is_refused(self):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(302, headers={"location": "https://evil.example/steal"}))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock), "https://huggingface.co/api/datasets/x")
        assert err.value.code == FetchError.UNSAFE and mock.hosts() == ["huggingface.co"]

    async def test_a_redirect_to_an_internal_address_is_refused(self):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/"}))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock), "https://huggingface.co/api/datasets/x")
        assert err.value.code == FetchError.UNSAFE

    async def test_a_redirect_loop_is_cut_off(self):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(302, headers={"location": "/api/datasets/loop"}))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock, external_max_redirects=2), "https://huggingface.co/api/datasets/loop")
        assert err.value.code == FetchError.UNAVAILABLE and len(mock.requests) == 3

    async def test_a_credential_is_dropped_when_a_redirect_changes_host(self):
        seen = []

        def responder(request):
            seen.append((request.url.host, request.headers.get("authorization")))
            if request.url.host == "www.kaggle.com":
                return httpx.Response(302, headers={"location": "https://huggingface.co/api/datasets/x"})
            return MockRegistries._json({"ok": 1})

        mock = MockRegistries().route("kaggle.com", responder).route("huggingface.co", responder)
        await get(fetcher_for(mock), "https://www.kaggle.com/api/v1/datasets/list", headers={"Authorization": "Bearer KGAT_secret"})
        assert seen[0] == ("www.kaggle.com", "Bearer KGAT_secret")
        assert seen[1] == ("huggingface.co", None)

    async def test_an_oversized_response_is_discarded(self):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(200, content=b"[" + b"1," * 5000 + b"1]"))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock, external_max_response_bytes=1000), "https://huggingface.co/api/datasets")
        assert err.value.code == FetchError.INVALID_RESPONSE and "larger than" in err.value.message

    async def test_a_response_just_under_the_cap_is_accepted(self):
        body = b'{"a":"' + b"x" * 100 + b'"}'
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(200, content=body))
        assert await get(fetcher_for(mock, external_max_response_bytes=len(body)), "https://huggingface.co/api/x")

    @pytest.mark.parametrize(
        ("status", "code"),
        [(404, FetchError.NOT_FOUND), (429, FetchError.RATE_LIMITED), (500, FetchError.UNAVAILABLE),
         (503, FetchError.UNAVAILABLE), (403, FetchError.UNAVAILABLE), (418, FetchError.UNAVAILABLE)],
    )
    async def test_statuses_map_to_stable_codes(self, status, code):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(status, text="internal trace KGAT_secret /srv/app.py"))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock), "https://huggingface.co/api/x")
        assert err.value.code == code
        assert "KGAT_secret" not in err.value.message and "/srv" not in err.value.message

    async def test_registry_specific_not_found_statuses(self):
        mock = MockRegistries().route("openml.org", lambda r: httpx.Response(412, json={"error": {"code": "372"}}))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock), "https://www.openml.org/api/x", not_found_statuses=frozenset({412}))
        assert err.value.code == FetchError.NOT_FOUND

    async def test_timeouts_and_connection_failures(self):
        def slow(request):
            raise httpx.ReadTimeout("slow")

        def down(request):
            raise httpx.ConnectError("down")

        for responder, code in ((slow, FetchError.TIMEOUT), (down, FetchError.UNAVAILABLE)):
            mock = MockRegistries().route("huggingface.co", responder)
            with pytest.raises(FetchError) as err:
                await get(fetcher_for(mock), "https://huggingface.co/api/x")
            assert err.value.code == code

    async def test_invalid_json_is_an_invalid_response(self):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(200, text="<html>not json</html>"))
        with pytest.raises(FetchError) as err:
            await get(fetcher_for(mock), "https://huggingface.co/api/x")
        assert err.value.code == FetchError.INVALID_RESPONSE and "<html>" not in err.value.message

    async def test_the_configured_timeout_is_applied_to_each_request(self):
        seen = {}

        def responder(request):
            seen["timeout"] = request.extensions.get("timeout")
            return MockRegistries._json({})

        mock = MockRegistries().route("huggingface.co", responder)
        await get(fetcher_for(mock, external_request_timeout=7.0), "https://huggingface.co/api/x")
        assert seen["timeout"]["read"] == 7.0


# ---------------------------------------------------------------------------
# Prose
# ---------------------------------------------------------------------------


class TestProseExtraction:
    def test_an_inline_list(self):
        names, method = extract_columns("Columns: PassengerId, Survived, Pclass, Name, Sex, Age, Fare.")
        assert names[:3] == ["PassengerId", "Survived", "Pclass"] and method == "inline list"

    def test_a_bullet_list_under_a_heading(self):
        text = "## Features\n- age: years\n- study_hours: per week\n- final_grade: 0-100\n"
        assert extract_columns(text) == (["age", "study_hours", "final_grade"], "bullet list")

    def test_a_markdown_table(self):
        text = "| Column | Meaning |\n|---|---|\n| age | years |\n| bmi | index |\n| glucose | mg/dl |\n"
        assert extract_columns(text) == (["age", "bmi", "glucose"], "markdown table")

    def test_underscores_inside_names_survive(self):
        names, _ = extract_columns("Columns: study_hours, final_grade, attendance_rate.")
        assert names == ["study_hours", "final_grade", "attendance_rate"]

    @pytest.mark.parametrize(
        "text",
        ["", "short", "A lovely dataset about students and their lives in school. Enjoy using it.",
         "Columns: a, b", "This describes features of the data in a long sentence with no list at all really."],
    )
    def test_prose_without_a_list_yields_nothing_not_a_guess(self, text):
        assert extract_columns(text) == ([], "")

    def test_duplicates_make_a_list_implausible(self):
        assert extract_columns("Columns: age, age, age, age.") == ([], "")

    def test_a_long_sentence_is_not_a_column_list(self):
        text = "Columns: " + ", ".join(["this is a very long sentence that is clearly not a name"] * 5) + "."
        assert extract_columns(text) == ([], "")


# ---------------------------------------------------------------------------
# UCI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUCI:
    def adapter(self, mock, **kw):
        return UCIAdapter(settings(**kw), fetcher_for(mock, **kw))

    async def test_variables_become_a_registry_verified_schema_with_roles(self):
        mock = MockRegistries().route("archive.ics.uci.edu/api/dataset", uci_record())
        meta = await self.adapter(mock).fetch("53")
        assert meta.schema_level is EvidenceLevel.REGISTRY_VERIFIED and meta.schema_source == "UCI variables"
        assert [c.original_name for c in meta.schema_columns][-1] == "class"
        assert meta.registry_target_columns == ["class"] and meta.schema_columns[-1].role == "target"
        assert all(c.evidence is EvidenceLevel.REGISTRY_VERIFIED for c in meta.schema_columns)
        assert meta.row_count == 150 and meta.column_count == 5 and meta.tasks == ["Classification"]

    async def test_a_numeric_id_is_queried_by_id_and_a_name_by_name(self):
        mock = MockRegistries().route("archive.ics.uci.edu/api/dataset", uci_record())
        await self.adapter(mock).fetch("53")
        await self.adapter(mock).fetch("Heart Disease")
        assert "id=53" in mock.urls()[0] and "name=Heart" in mock.urls()[1]

    async def test_a_404_inside_a_200_body_is_not_found(self):
        mock = MockRegistries().route("archive.ics.uci.edu/api/dataset", {"status": 404, "message": "not found"})
        with pytest.raises(FetchError) as err:
            await self.adapter(mock).fetch("999999")
        assert err.value.code == FetchError.NOT_FOUND

    async def test_a_dataset_with_no_variable_list_is_unknown_not_empty(self):
        mock = MockRegistries().route("archive.ics.uci.edu/api/dataset", uci_record(variables=[]))
        meta = await self.adapter(mock).fetch("53")
        assert meta.schema_level is EvidenceLevel.UNKNOWN and meta.schema_columns == []
        assert "no variable list" in meta.schema_note

    async def test_search_returns_hits(self):
        mock = MockRegistries().route("datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}, {"id": 99}]})
        hits = await self.adapter(mock).search("iris")
        assert [(h.identifier, h.name) for h in hits] == [("53", "Iris")]

    async def test_a_failed_search_status_is_empty(self):
        mock = MockRegistries().route("datasets/list", {"status": 500})
        assert await self.adapter(mock).search("iris") == []


# ---------------------------------------------------------------------------
# OpenML
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenML:
    def adapter(self, mock, **kw):
        return OpenMLAdapter(settings(**kw), fetcher_for(mock, **kw))

    def full_mock(self, **q):
        return (MockRegistries()
                .route("/data/features/", openml_features(("sepallength", "numeric", False), ("class", "nominal", True)))
                .route("/data/qualities/", openml_qualities(NumberOfInstances=150, NumberOfMissingValues=0, NumberOfClasses=3,
                                                              MajorityClassSize=50, **q))
                .route("/data/61", openml_desc()))

    async def test_features_and_qualities_become_registry_facts(self):
        meta = await self.adapter(self.full_mock()).fetch("61")
        assert meta.schema_level is EvidenceLevel.REGISTRY_VERIFIED and meta.registry_target_columns == ["class"]
        assert meta.row_count == 150 and meta.column_count == 2 and meta.license == "Public"
        assert meta.registry_statistics["NumberOfClasses"] == 3.0 and meta.registry_statistics["MajorityClassSize"] == 50.0
        assert [c.role for c in meta.schema_columns] == ["feature", "target"]

    async def test_a_single_feature_returned_as_an_object_not_a_list(self):
        mock = (MockRegistries().route("/data/features/", openml_features(("only", "numeric", False)))
                .route("/data/qualities/", openml_qualities(NumberOfInstances=10))
                .route("/data/61", openml_desc(target="only")))
        meta = await self.adapter(mock).fetch("61")
        assert [c.original_name for c in meta.schema_columns] == ["only"]

    async def test_failed_qualities_do_not_lose_the_schema(self):
        mock = (MockRegistries().route("/data/features/", openml_features(("a", "numeric", False), ("y", "nominal", True)))
                .route("/data/qualities/", lambda r: httpx.Response(500)).route("/data/61", openml_desc()))
        meta = await self.adapter(mock).fetch("61")
        assert meta.schema_columns and meta.row_count is None and meta.registry_statistics == {}

    async def test_an_unknown_dataset_is_not_found(self):
        mock = MockRegistries().route("openml.org", lambda r: httpx.Response(412, json={"error": {"code": "111"}}))
        with pytest.raises(FetchError) as err:
            await self.adapter(mock).fetch("9999999")
        assert err.value.code == FetchError.NOT_FOUND

    async def test_search_lists_active_versions_with_counts(self):
        mock = MockRegistries().route("/data/list/data_name/titanic", openml_list((40704, "Titanic", 2, 2201, 4), (42437, "titanic", 6, 891, 12)))
        hits = await self.adapter(mock).search("Titanic")
        assert {(h.identifier, h.row_count, h.column_count) for h in hits} == {("40704", 2201, 4), ("42437", 891, 12)}

    async def test_no_results_is_an_empty_list_not_an_error(self):
        mock = MockRegistries().route("openml.org", lambda r: httpx.Response(412, json={"error": {"code": "372", "message": "No results"}}))
        assert await self.adapter(mock).search("zzz") == []

    async def test_the_name_is_path_encoded(self):
        mock = MockRegistries().route("openml.org", lambda r: httpx.Response(412, json={}))
        await self.adapter(mock).search("a/b c")
        assert "a%2Fb%20c" in mock.urls()[0]


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHuggingFace:
    def adapter(self, mock, **kw):
        return HuggingFaceAdapter(settings(**kw), fetcher_for(mock, **kw))

    async def test_structured_features_are_registry_verified_with_normalised_dtypes(self):
        mock = MockRegistries().route("/api/datasets/owner/titanic", hf_record())
        meta = await self.adapter(mock).fetch("owner/titanic")
        assert meta.schema_level is EvidenceLevel.REGISTRY_VERIFIED
        types = {c.original_name: c.data_type for c in meta.schema_columns}
        assert types["age"] == "float64" and types["label"] == "class_label"
        assert meta.row_count == 891 and meta.size_bytes == 12345 and meta.license == "cc-by-4.0"
        assert meta.tasks == ["tabular classification"] and meta.file_types == [".csv"]
        assert "train.csv" in meta.files and meta.tags == ["survival"]

    async def test_a_redirect_to_the_canonical_id_is_followed(self):
        mock = (MockRegistries()
                .route("/api/datasets/mnist", lambda r: httpx.Response(307, headers={"location": "/api/datasets/ylecun/mnist"}))
                .route("/api/datasets/ylecun/mnist", hf_record("ylecun/mnist")))
        meta = await self.adapter(mock).fetch("mnist")
        assert meta.identifier == "ylecun/mnist"

    @pytest.mark.parametrize("status", [401, 403, 404])
    async def test_missing_or_private_is_not_found(self, status):
        mock = MockRegistries().route("huggingface.co", lambda r: httpx.Response(status, text="Invalid username or password."))
        with pytest.raises(FetchError) as err:
            await self.adapter(mock).fetch("zz/nope")
        assert err.value.code == FetchError.NOT_FOUND and "username" not in err.value.message

    async def test_a_private_dataset_is_never_described(self):
        mock = MockRegistries().route("/api/datasets/o/x", hf_record("o/x", private=True))
        with pytest.raises(FetchError):
            await self.adapter(mock).fetch("o/x")

    async def test_without_features_the_schema_falls_back_to_the_description_as_inferred(self):
        mock = MockRegistries().route(
            "/api/datasets/o/x", hf_record("o/x", features=False, description="Columns: age, income, tenure, churn flag.")
        )
        meta = await self.adapter(mock).fetch("o/x")
        assert meta.schema_level is EvidenceLevel.INFERRED
        assert all(c.evidence is EvidenceLevel.INFERRED for c in meta.schema_columns)
        assert "inline list" in meta.schema_source and "incomplete" in meta.schema_note

    async def test_without_features_or_a_list_the_schema_is_unknown(self):
        mock = MockRegistries().route("/api/datasets/o/x", hf_record("o/x", features=False, description="A pleasant dataset."))
        meta = await self.adapter(mock).fetch("o/x")
        assert meta.schema_level is EvidenceLevel.UNKNOWN and meta.schema_columns == []

    async def test_several_configurations_are_noted_and_only_the_first_is_listed(self):
        record = hf_record("o/x")
        record["cardData"]["dataset_info"] = [
            {"features": [{"name": "a", "dtype": "int64"}], "splits": [{"num_examples": 10}]},
            {"features": [{"name": "b", "dtype": "int64"}], "splits": [{"num_examples": 99}]},
        ]
        mock = MockRegistries().route("/api/datasets/o/x", record)
        meta = await self.adapter(mock).fetch("o/x")
        assert [c.original_name for c in meta.schema_columns] == ["a"] and meta.row_count == 10
        assert "several configurations" in meta.schema_note

    async def test_search_skips_private_datasets(self):
        mock = MockRegistries().route("/api/datasets?", [{"id": "a/one", "downloads": 5}, {"id": "a/two", "private": True}])
        assert [h.identifier for h in await self.adapter(mock).search("x")] == ["a/one"]


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestKaggle:
    def adapter(self, mock, **kw):
        return KaggleAdapter(settings(**kw), fetcher_for(mock, **kw))

    def mock(self, description=None):
        view = kaggle_view() if description is None else kaggle_view(description=description)
        return (MockRegistries().route("/datasets/view/", view)
                .route("/datasets/list/owner/titanic", {"datasetFiles": [{"name": "train.csv"}, {"name": "test.csv"}]}))

    async def test_a_column_list_in_the_description_is_inferred_never_verified(self):
        meta = await self.adapter(self.mock()).fetch("owner/titanic")
        assert meta.schema_level is EvidenceLevel.INFERRED and meta.schema_source.startswith("description")
        assert all(c.evidence is EvidenceLevel.INFERRED for c in meta.schema_columns)
        assert "Survived" in [c.original_name for c in meta.schema_columns]
        assert "incomplete" in meta.schema_note and meta.registry_target_columns == []

    async def test_no_list_in_the_description_means_an_unknown_schema(self):
        meta = await self.adapter(self.mock("A famous dataset of ship passengers.")).fetch("owner/titanic")
        assert meta.schema_level is EvidenceLevel.UNKNOWN and meta.schema_columns == []
        assert "does not list the columns" in meta.schema_note

    async def test_files_license_and_size_come_from_the_registry(self):
        meta = await self.adapter(self.mock()).fetch("owner/titanic")
        assert meta.files == ["train.csv", "test.csv"] and meta.file_types == [".csv"]
        assert meta.license == "CC0" and meta.size_bytes == 61194 and meta.row_count is None

    async def test_row_count_is_never_invented_for_kaggle(self):
        assert (await self.adapter(self.mock()).fetch("owner/titanic")).row_count is None

    async def test_the_token_is_sent_only_as_a_header_to_kaggle(self):
        mock = self.mock()
        await self.adapter(mock, kaggle_api_token="KGAT_tok").fetch("owner/titanic")
        assert all(r.headers.get("authorization") == "Bearer KGAT_tok" for r in mock.requests)
        assert all("KGAT_tok" not in str(r.url) for r in mock.requests)

    async def test_no_token_means_no_authorization_header(self):
        mock = self.mock()
        await self.adapter(mock).fetch("owner/titanic")
        assert all("authorization" not in r.headers for r in mock.requests)

    async def test_a_failed_file_list_does_not_lose_the_description(self):
        mock = MockRegistries().route("/datasets/view/", kaggle_view()).route("/datasets/list/owner/titanic", lambda r: httpx.Response(500))
        meta = await self.adapter(mock).fetch("owner/titanic")
        assert meta.files == [] and meta.schema_level is EvidenceLevel.INFERRED

    async def test_a_failed_view_is_an_error(self):
        mock = MockRegistries().route("kaggle.com", lambda r: httpx.Response(404))
        with pytest.raises(FetchError) as err:
            await self.adapter(mock).fetch("owner/none")
        assert err.value.code == FetchError.NOT_FOUND

    async def test_search_maps_nullable_fields(self):
        mock = MockRegistries().route("/datasets/list?", [kaggle_item("a/b", "Titanic", 9)])
        hits = await self.adapter(mock).search("titanic")
        assert (hits[0].identifier, hits[0].name, hits[0].popularity) == ("a/b", "Titanic", 9.0)


class TestAdapterRegistry:
    def test_enabled_adapters_are_ordered_by_authority(self):
        mock = MockRegistries()
        names = [a.name for a in build_adapters(settings(), fetcher_for(mock))]
        assert names == ["uci", "openml", "huggingface", "kaggle"]

    def test_adapters_can_be_disabled(self):
        mock = MockRegistries()
        names = [a.name for a in build_adapters(settings(kaggle_enabled=False, huggingface_enabled=False), fetcher_for(mock))]
        assert names == ["uci", "openml"]

    def test_disabling_external_research_disables_every_adapter(self):
        assert build_adapters(settings(external_enabled=False), fetcher_for(MockRegistries())) == []
