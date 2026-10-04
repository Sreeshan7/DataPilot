"""POST /api/v1/datasets/analyze, end to end through the real app.

Each class is one promise to the user: an uploaded file is analysed from its own data; a
name or link is researched without ever claiming rows were read; anything unsafe, oversized or
unsupported is refused with a clear, safe error; and nothing is left behind on the server.
"""

from __future__ import annotations

import glob
import io
import json
import os
import tempfile

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import BodySizeLimitMiddleware
from app.api.v1 import dataset_analysis as api
from app.core.config import Settings
from app.services.dataset_analysis.evaluation.datasets import GENERATORS
from app.services.dataset_analysis.external.service import ExternalDatasetMetadataService
from app.services.dataset_analysis.service import DatasetAnalysisService
from tests.unit.registry_mock import (
    MockRegistries,
    hf_record,
    kaggle_view,
    openml_desc,
    openml_features,
    openml_qualities,
    uci_record,
)

URL = "/api/v1/datasets/analyze"


def make_settings(**kw) -> Settings:
    token = kw.pop("kaggle_api_token", "")
    return Settings(gemini_api_key="", kaggle_api_token=token, app_env="testing", **kw)


def build_client(monkeypatch, mock: MockRegistries | None = None, **settings_kw) -> TestClient:
    settings = make_settings(**settings_kw)
    mock = mock or MockRegistries()
    external = ExternalDatasetMetadataService(settings, mock.fetcher(settings))
    monkeypatch.setattr(api, "_service", DatasetAnalysisService(settings, external=external))
    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


def csv_file(name="churn", filename="data.csv"):
    return {"file": (filename, GENERATORS[name]().csv_bytes(), "text/csv")}


def err(response):
    return response.json()["detail"]["error"]


def leftover_uploads() -> set[str]:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "datapilot_*")))


@pytest.fixture
def client(monkeypatch):
    return build_client(monkeypatch)


# ---------------------------------------------------------------------------
# An uploaded file: the strongest evidence
# ---------------------------------------------------------------------------


class TestUploadedFile:
    def test_a_csv_is_analysed_from_its_own_data(self, client):
        response = client.post(URL, files=csv_file("customer_churn"))
        assert response.status_code == 200
        body = response.json()
        analysis = body["analysis"]
        assert analysis["mode"] == "file_parsed"
        assert body["headline"] == "DataPilot analyzed your actual file."
        assert "computed from your file" in body["evidence_statement"]
        assert analysis["schema"]["level"] == "datapilot_parsed"
        assert analysis["schema"]["row_count"] == 600 and analysis["schema"]["column_count"] == 7
        assert analysis["input_resolution"]["raw_file_available"] is True
        assert analysis["input_resolution"]["schema_evidence_level"] == "datapilot_parsed"
        assert {c["evidence_level"] for c in analysis["column_profiles"]} == {"datapilot_parsed"}

    def test_the_schema_is_exposed_under_its_public_name(self, client):
        body = client.post(URL, files=csv_file("diabetes")).json()
        assert "schema" in body["analysis"] and "schema_info" not in body["analysis"]

    def test_targets_tasks_and_quality_are_returned(self, client):
        analysis = client.post(URL, files=csv_file("customer_churn")).json()["analysis"]
        assert analysis["target_candidates"][0]["column"] == "churn"
        assert analysis["task_candidates"][0]["task"] == "binary_classification"
        codes = {f["code"] for f in analysis["quality"]["findings"]}
        assert {"possible_identifier", "possible_sensitive_field"} <= codes

    def test_a_workbook_is_analysed_and_a_sheet_can_be_chosen(self, client, tmp_path):
        from openpyxl import Workbook

        wb = Workbook()
        wb.active.title = "Notes"
        wb.active.append(["x"])
        wb.active.append([1])
        other = wb.create_sheet("Data")
        other.append(["a", "b", "label"])
        for i in range(40):
            other.append([i, i * 2, "p" if i % 2 else "q"])
        path = tmp_path / "book.xlsx"
        wb.save(path)
        files = {"file": ("book.xlsx", path.read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        body = client.post(URL, files=files, data={"sheet_name": "Data"}).json()
        assert body["analysis"]["ingestion"]["sheet_name"] == "Data"
        assert body["analysis"]["schema"]["column_count"] == 3

    def test_a_missing_sheet_is_a_422_listing_the_sheets(self, client, tmp_path):
        from openpyxl import Workbook

        wb = Workbook()
        wb.active.title = "Only"
        wb.active.append(["a"])
        wb.active.append([1])
        path = tmp_path / "b.xlsx"
        wb.save(path)
        response = client.post(URL, files={"file": ("b.xlsx", path.read_bytes())}, data={"sheet_name": "Nope"})
        assert response.status_code == 422 and err(response)["code"] == "EXCEL_SHEET_NOT_FOUND"
        assert err(response)["details"]["sheets_available"] == ["Only"]

    def test_a_named_target_is_honoured(self, client):
        analysis = client.post(URL, files=csv_file("customer_churn"), data={"target": "support_calls"}).json()["analysis"]
        assert analysis["target_candidates"][0]["column"] == "support_calls"
        assert analysis["target_candidates"][0]["user_specified"] is True

    def test_an_unknown_target_is_a_422_with_the_available_columns(self, client):
        response = client.post(URL, files=csv_file("customer_churn"), data={"target": "nope"})
        assert response.status_code == 422 and err(response)["code"] == "TARGET_COLUMN_NOT_FOUND"
        assert "churn" in err(response)["details"]["columns_available"]

    def test_a_file_wins_over_a_dataset_name(self, client):
        analysis = client.post(URL, files=csv_file("diabetes"), data={"dataset_name": "Iris"}).json()["analysis"]
        assert analysis["mode"] == "file_parsed" and analysis["external"] is None
        assert any("uploaded file was analysed" in e for e in analysis["input_resolution"]["evidence"])

    def test_the_analysis_is_deterministic(self, client):
        def strip(body):
            a = body["analysis"]
            a["metadata"].pop("analysis_id"), a["metadata"].pop("started_at"), a["metadata"].pop("duration_ms")
            a["ingestion"].pop("duration_ms")
            body.pop("request_id")
            return json.dumps(body, sort_keys=True)

        first = client.post(URL, files=csv_file("house_prices")).json()
        second = client.post(URL, files=csv_file("house_prices")).json()
        assert strip(first) == strip(second)

    def test_the_limits_in_force_are_reported(self, client):
        limits = client.post(URL, files=csv_file("diabetes")).json()["analysis"]["metadata"]["limits"]
        assert limits["max_rows"] == 200_000 and limits["random_seed"] == 42


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        ("filename", "status"),
        [("evil.exe", 415), ("data.json", 415), ("old.xls", 415), ("macro.xlsm", 415), ("noext", 415), ("a.csv.exe", 415)],
    )
    def test_unsupported_file_types(self, client, filename, status):
        response = client.post(URL, files={"file": (filename, b"a,b\n1,2\n")})
        assert response.status_code == status and err(response)["code"] == "FILE_TYPE_UNSUPPORTED"

    def test_an_empty_file(self, client):
        response = client.post(URL, files={"file": ("e.csv", b"")})
        assert response.status_code == 422 and err(response)["code"] == "FILE_EMPTY"

    def test_a_header_only_file(self, client):
        response = client.post(URL, files={"file": ("h.csv", b"a,b,c\n")})
        assert err(response)["code"] == "FILE_NO_DATA_ROWS"

    def test_a_binary_file_named_csv(self, client):
        response = client.post(URL, files={"file": ("b.csv", b"%PDF-1.4\x00\x01\x02")})
        assert response.status_code == 422 and err(response)["code"] == "FILE_MALFORMED"

    def test_a_corrupt_workbook(self, client):
        response = client.post(URL, files={"file": ("b.xlsx", b"not a workbook" * 20)})
        assert response.status_code == 422 and err(response)["code"] == "FILE_MALFORMED"

    def test_nothing_supplied(self, client):
        response = client.post(URL, data={})
        assert response.status_code == 422 and err(response)["code"] == "DATASET_INPUT_REQUIRED"

    def test_errors_never_contain_paths_cell_values_or_tracebacks(self, client):
        for response in (
            client.post(URL, files={"file": ("b.csv", b"%PDF-1.4\x00\x01")}),
            client.post(URL, files={"file": ("b.xlsx", b"junk" * 50)}),
            client.post(URL, files=csv_file("customer_churn"), data={"target": "nope"}),
        ):
            text = response.text
            assert tempfile.gettempdir() not in text and "Traceback" not in text
            assert "datapilot_" not in text and "user17@example.com" not in text

    def test_the_error_envelope_matches_the_other_endpoints(self, client):
        body = client.post(URL, files={"file": ("e.csv", b"")}).json()
        assert body["detail"]["success"] is False and body["detail"]["request_id"]


class TestSizeLimits:
    def test_an_oversized_upload_is_refused_by_the_service_limit(self, monkeypatch):
        client = build_client(monkeypatch, analysis_max_upload_bytes=2000)
        response = client.post(URL, files={"file": ("big.csv", GENERATORS["house_prices"]().csv_bytes())})
        assert response.status_code == 413 and err(response)["code"] == "FILE_TOO_LARGE"

    def test_nothing_is_left_on_disk_after_a_refusal(self, monkeypatch):
        before = leftover_uploads()
        client = build_client(monkeypatch, analysis_max_upload_bytes=2000)
        client.post(URL, files={"file": ("big.csv", GENERATORS["house_prices"]().csv_bytes())})
        client.post(URL, files={"file": ("bad.csv", b"%PDF-1.4\x00\x01")})
        assert leftover_uploads() == before

    def test_nothing_is_left_on_disk_after_a_success(self, client):
        before = leftover_uploads()
        assert client.post(URL, files=csv_file("diabetes")).status_code == 200
        assert leftover_uploads() == before

    def test_a_cleanup_failure_is_reported_without_the_path(self, client, monkeypatch):
        real_remove = os.remove
        removed: list[str] = []

        def refuse(path):
            removed.append(path)
            raise PermissionError(path)

        monkeypatch.setattr(os, "remove", refuse)
        response = client.post(URL, files=csv_file("diabetes"))
        monkeypatch.setattr(os, "remove", real_remove)
        for path in removed:
            real_remove(path)
        body = response.json()
        assert response.status_code == 200
        assert any("could not be removed" in w for w in body["analysis"]["warnings"])
        assert tempfile.gettempdir() not in response.text

    def test_a_malicious_filename_is_only_ever_a_label(self, client):
        files = {"file": ("../../../etc/passwd.csv", GENERATORS["diabetes"]().csv_bytes())}
        body = client.post(URL, files=files).json()
        assert body["analysis"]["ingestion"]["display_name"] == "passwd.csv"
        assert ".." not in body["analysis"]["ingestion"]["display_name"]

    def test_the_analysis_time_budget_is_enforced(self, monkeypatch):
        client = build_client(monkeypatch, analysis_time_limit_seconds=0.0)
        response = client.post(URL, files=csv_file("house_prices"))
        assert response.status_code == 504 and err(response)["code"] == "ANALYSIS_TIMEOUT"


class TestBodySizeMiddleware:
    """The cap must hold before the multipart body reaches disk, and without trusting headers."""

    def app(self, limit):
        app = FastAPI()
        app.add_middleware(BodySizeLimitMiddleware, limits={"/datasets/analyze": limit})
        app.include_router(api.router, prefix="/api/v1")
        return app

    def test_an_honest_content_length_over_the_limit_is_refused_immediately(self):
        client = TestClient(self.app(10_000))
        response = client.post(URL, files={"file": ("a.csv", b"x" * 50_000)})
        assert response.status_code == 413 and err(response)["code"] == "FILE_TOO_LARGE"

    def test_a_lying_or_absent_content_length_is_still_stopped_as_bytes_arrive(self):
        import asyncio

        sent: list = []

        async def inner_app(scope, receive, send):
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break

        middleware = BodySizeLimitMiddleware(inner_app, {"/datasets/analyze": 1000})
        chunks = iter([{"type": "http.request", "body": b"x" * 600, "more_body": True},
                       {"type": "http.request", "body": b"x" * 600, "more_body": False}])

        async def receive():
            return next(chunks)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "path": "/api/v1/datasets/analyze", "headers": []}
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as raised:
            asyncio.run(middleware(scope, receive, send))
        assert raised.value.status_code == 413

    def test_other_paths_and_methods_are_untouched(self):
        client = TestClient(self.app(10))
        assert client.get("/api/v1/datasets/analyze").status_code in (404, 405)

    def test_a_request_under_the_limit_passes_through(self):
        app = self.app(1_000_000)
        monkey = TestClient(app, raise_server_exceptions=False)
        response = monkey.post(URL, data={})
        assert response.status_code == 422  # reaches the route: nothing supplied


# ---------------------------------------------------------------------------
# A name or a link: registry metadata only
# ---------------------------------------------------------------------------


def uci_iris(mock: MockRegistries) -> MockRegistries:
    return (mock.route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
            .route("archive.ics.uci.edu/api/dataset?", uci_record())
            .route("openml.org/api/v1/json/data/list", lambda r: httpx.Response(412, json={}))
            .route("huggingface.co/api/datasets?", [])
            .route("kaggle.com/api/v1/datasets/list?", []))


class TestRegistryReference:
    def test_a_dataset_name_is_researched_and_never_claims_rows_were_read(self, monkeypatch):
        client = build_client(monkeypatch, uci_iris(MockRegistries()))
        body = client.post(URL, data={"dataset_name": "Iris"}).json()
        analysis = body["analysis"]
        assert analysis["mode"] == "external_metadata"
        assert body["headline"] == "DataPilot found public metadata for this dataset, but has not parsed the raw file."
        assert "no rows were read" in body["evidence_statement"]
        assert analysis["schema"]["level"] == "registry_verified"
        assert analysis["input_resolution"]["raw_file_available"] is False
        assert analysis["column_profiles"] == [] and analysis["quality"] is None and analysis["ingestion"] is None
        assert any("has not parsed the raw file" in l for l in analysis["limitations"])

    def test_nothing_in_a_reference_analysis_is_marked_as_parsed(self, monkeypatch):
        client = build_client(monkeypatch, uci_iris(MockRegistries()))
        text = client.post(URL, data={"dataset_name": "Iris"}).text
        assert "datapilot_parsed" not in text

    def test_a_registry_designated_target_is_reported_with_its_source(self, monkeypatch):
        client = build_client(monkeypatch, uci_iris(MockRegistries()))
        analysis = client.post(URL, data={"dataset_name": "Iris"}).json()["analysis"]
        target = analysis["target_candidates"][0]
        assert target["column"] == "class" and target["state"] == "strong_candidate"
        assert any(e["level"] == "registry_verified" and "designates" in e["statement"] for e in target["evidence"])
        assert all(e["level"] != "datapilot_parsed" for e in target["evidence"])
        assert any("no rows were read" in a for a in target["against"])

    def test_the_next_step_asks_for_the_file_and_offers_to_upload_it(self, monkeypatch):
        client = build_client(monkeypatch, uci_iris(MockRegistries()))
        body = client.post(URL, data={"dataset_name": "Iris"}).json()
        assert "Upload the actual file" in body["recommended_next_step"]
        assert body["next_actions"][0]["action"] == "UPLOAD_FILE"

    def test_a_dataset_url_is_resolved_exactly(self, monkeypatch):
        mock = (MockRegistries().route("/data/features/", openml_features(("a", "numeric", False), ("y", "nominal", True)))
                .route("/data/qualities/", openml_qualities(NumberOfInstances=100, NumberOfClasses=2))
                .route("/data/61", openml_desc()))
        client = build_client(monkeypatch, mock)
        analysis = client.post(URL, data={"dataset_url": "https://www.openml.org/d/61"}).json()["analysis"]
        assert analysis["input_resolution"]["resolution_status"] == "resolved"
        assert analysis["dataset_identity"]["identifier"] == "61" and analysis["schema"]["row_count"] == 100
        assert analysis["schema"]["row_count_evidence"] == "registry_verified"

    def test_a_kaggle_dataset_without_a_published_schema_says_so(self, monkeypatch):
        mock = (MockRegistries().route("/datasets/view/", kaggle_view(description="A famous dataset."))
                .route("/datasets/list/owner/titanic", {"datasetFiles": []}))
        client = build_client(monkeypatch, mock)
        body = client.post(URL, data={"dataset_url": "https://www.kaggle.com/datasets/owner/titanic"}).json()
        assert body["analysis"]["schema"]["level"] == "unknown" and body["analysis"]["schema"]["columns"] == []
        assert "does not expose enough schema information" in body["headline"]
        assert any("does not expose enough schema information" in l for l in body["analysis"]["limitations"])
        assert body["analysis"]["target_candidates"] == []

    def test_a_kaggle_column_list_read_from_text_is_flagged_inferred(self, monkeypatch):
        mock = (MockRegistries().route("/datasets/view/", kaggle_view())
                .route("/datasets/list/owner/titanic", {"datasetFiles": []}))
        client = build_client(monkeypatch, mock)
        body = client.post(URL, data={"dataset_url": "https://www.kaggle.com/datasets/owner/titanic"}).json()
        assert body["analysis"]["schema"]["level"] == "inferred"
        assert "only read from the description" in body["headline"]
        assert any("may be incomplete or inexact" in l for l in body["analysis"]["limitations"])

    def test_an_ambiguous_name_analyses_nothing_and_asks(self, monkeypatch):
        client = build_client(monkeypatch)
        body = client.post(URL, data={"dataset_name": "student"}).json()
        assert body["analysis"]["input_resolution"]["input_type"] == "ambiguous_dataset_reference"
        assert body["analysis"]["external"] is None and "could be thousands" in body["recommended_next_step"]

    def test_an_unknown_name_is_unresolved_not_invented(self, monkeypatch):
        mock = (MockRegistries().route("archive.ics.uci.edu", {"status": 200, "data": []})
                .route("openml.org", lambda r: httpx.Response(412, json={}))
                .route("huggingface.co", []).route("kaggle.com", []))
        client = build_client(monkeypatch, mock)
        body = client.post(URL, data={"dataset_name": "Zzyzx Totally Fake Data"}).json()
        assert body["analysis"]["input_resolution"]["resolution_status"] == "unresolved"
        assert body["analysis"]["schema"]["level"] == "unknown" and body["analysis"]["external"] is None
        assert "could not identify" in body["headline"]

    def test_several_matching_datasets_are_reported_with_alternatives(self, monkeypatch):
        mock = (MockRegistries().route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": []})
                .route("openml.org", lambda r: httpx.Response(412, json={}))
                .route("kaggle.com/api/v1/datasets/list?", [])
                .route("huggingface.co/api/datasets?", [{"id": "o/titanic", "downloads": 1}, {"id": "p/titanic", "downloads": 9}])
                .route("huggingface.co/api/datasets/p/titanic", hf_record("p/titanic"))
                .route("huggingface.co/api/datasets/o/titanic", hf_record("o/titanic")))
        client = build_client(monkeypatch, mock)
        body = client.post(URL, data={"dataset_name": "Titanic"}).json()
        resolution = body["analysis"]["input_resolution"]
        assert resolution["resolution_status"] == "probable" and resolution["needs_confirmation"]
        assert len(resolution["alternatives"]) == 1 and "share the name" in body["recommended_next_step"]
        assert any(a["action"] == "ANALYZE_ALTERNATIVE" for a in body["next_actions"])

    @pytest.mark.parametrize(
        "url",
        ["https://evil.example/data.csv", "https://127.0.0.1/x", "https://www.kaggle.com/competitions/titanic",
         "ftp://www.kaggle.com/datasets/a/b", "https://user:pw@www.kaggle.com/datasets/a/b"],
    )
    def test_an_unsafe_or_unsupported_link_is_a_422_and_no_request_is_made(self, monkeypatch, url):
        mock = MockRegistries()
        client = build_client(monkeypatch, mock)
        response = client.post(URL, data={"dataset_url": url})
        assert response.status_code == 422 and err(response)["code"] == "DATASET_URL_UNSUPPORTED"
        assert mock.requests == []

    def test_a_registry_outage_is_a_502_and_a_timeout_a_504(self, monkeypatch):
        down = MockRegistries().route("huggingface.co", lambda r: httpx.Response(500, text="internal trace"))
        response = build_client(monkeypatch, down).post(URL, data={"dataset_url": "https://huggingface.co/datasets/o/x"})
        assert response.status_code == 502 and "internal trace" not in response.text

        def slow(request):
            raise httpx.ReadTimeout("slow")

        timeout = MockRegistries().route("openml.org", slow)
        response = build_client(monkeypatch, timeout).post(URL, data={"dataset_url": "https://www.openml.org/d/61"})
        assert response.status_code == 504 and err(response)["code"] == "DATASET_SOURCE_TIMEOUT"

    def test_a_target_hint_is_validated_against_the_published_schema(self, monkeypatch):
        client = build_client(monkeypatch, uci_iris(MockRegistries()))
        response = client.post(URL, data={"dataset_name": "Iris", "target": "nope"})
        assert response.status_code == 422 and err(response)["code"] == "TARGET_COLUMN_NOT_FOUND"
        ok = client.post(URL, data={"dataset_name": "Iris", "target": "sepal width"}).json()["analysis"]
        assert ok["target_candidates"][0]["column"] == "sepal_width" and ok["target_candidates"][0]["user_specified"]

    def test_the_kaggle_token_never_appears_in_a_response(self, monkeypatch):
        mock = (MockRegistries().route("/datasets/view/", kaggle_view())
                .route("/datasets/list/owner/titanic", {"datasetFiles": []}))
        client = build_client(monkeypatch, mock, kaggle_api_token="KGAT_very_secret")
        response = client.post(URL, data={"dataset_url": "https://www.kaggle.com/datasets/owner/titanic"})
        assert "KGAT_very_secret" not in response.text


class TestRobustness:
    def test_an_unexpected_failure_is_a_generic_500(self, client, monkeypatch):
        async def boom(self, request):
            raise RuntimeError("secret internal detail /srv/app.py")

        monkeypatch.setattr(DatasetAnalysisService, "analyze", boom)
        response = client.post(URL, files=csv_file("diabetes"))
        assert response.status_code == 500 and err(response)["code"] == "INTERNAL_ERROR"
        assert "secret" not in response.text and "/srv" not in response.text

    def test_the_endpoint_is_documented_and_existing_ones_still_work(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert URL in paths and "/api/v1/datasets/recommend" in paths and "/api/v1/projects/analyze" in paths
        assert client.get("/health").status_code == 200

    def test_the_request_log_never_contains_file_contents(self, client, caplog):
        with caplog.at_level("INFO"):
            client.post(URL, files={"file": ("secret_name.csv", b"name,ssn\nAlice,123-45-6789\nBob,987-65-4321\n")})
        assert "123-45-6789" not in caplog.text and "Alice" not in caplog.text

    def test_a_blank_filename_does_not_count_as_a_file(self, monkeypatch):
        client = build_client(monkeypatch, uci_iris(MockRegistries()))
        response = client.post(URL, files={"file": ("", b"")}, data={"dataset_name": "Iris"})
        assert response.status_code == 200 and response.json()["analysis"]["mode"] == "external_metadata"
