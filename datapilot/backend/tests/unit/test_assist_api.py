"""POST /api/v1/assist: one entry point that picks the journey itself.

Promises under test: a client never has to choose between "find" and "analyse"; the response
always explains the decision; unclear input runs nothing and asks; a missing input is asked for
instead of failing; the journeys connect (analyse -> find another, find -> analyse); and every
safety property of the individual endpoints still holds behind this one.
"""

from __future__ import annotations

import glob
import os
import tempfile

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import assist as assist_module
from app.api.v1 import dataset_analysis as analysis_api
from app.api.v1 import datasets as datasets_api
from app.core.config import Settings
from app.main import app
from app.services.dataset_analysis.evaluation.datasets import GENERATORS
from app.services.dataset_analysis.external.service import ExternalDatasetMetadataService
from app.services.dataset_analysis.service import DatasetAnalysisService
from tests.unit.registry_mock import MockRegistries, uci_record
from tests.unit.test_dataset_api import _StubService

URL = "/api/v1/assist"
GOAL = "I want to build a model that predicts which customers will cancel their subscription next month"


@pytest.fixture
def stub(monkeypatch):
    service = _StubService()
    monkeypatch.setattr(datasets_api, "_service", service)
    return service


@pytest.fixture
def client(monkeypatch, stub):
    settings = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing")
    mock = (MockRegistries()
            .route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
            .route("archive.ics.uci.edu/api/dataset?", uci_record())
            .route("openml.org/api/v1/json/data/list", lambda r: httpx.Response(412, json={}))
            .route("huggingface.co/api/datasets?", [])
            .route("kaggle.com/api/v1/datasets/list?", []))
    external = ExternalDatasetMetadataService(settings, mock.fetcher(settings))
    monkeypatch.setattr(analysis_api, "_service", DatasetAnalysisService(settings, external=external))
    return TestClient(app, raise_server_exceptions=False)


def csv_file(name="customer_churn"):
    return {"file": ("data.csv", GENERATORS[name]().csv_bytes(), "text/csv")}


def leftover_uploads() -> set[str]:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "datapilot_*")))


class TestUserBRouting:
    def test_an_uploaded_file_is_analysed_without_choosing_a_journey(self, client):
        r = client.post(URL, data={"message": "here is my data"}, files=csv_file())
        body = r.json()
        assert r.status_code == 200 and body["status"] == "COMPLETED"
        assert body["journey"] == "USER_B_DATASET_ANALYSIS"
        assert body["analysis"]["analysis"]["mode"] == "file_parsed"
        assert body["discovery"] is None
        assert body["routing"]["intent"] == "USER_B_DATASET_ANALYSIS" and body["routing"]["evidence"]

    def test_a_file_beats_an_accompanying_wish_for_a_better_dataset(self, client):
        r = client.post(URL, data={"message": "analyze this, but I might need a better dataset", "project_goal": GOAL},
                        files=csv_file())
        body = r.json()
        assert body["journey"] == "USER_B_DATASET_ANALYSIS"
        assert body["routing"]["follow_up_intent"] == "USER_A_DATASET_DISCOVERY"
        actions = {a["action"]: a for a in body["next_actions"]}
        assert actions["FIND_ALTERNATIVE_DATASET"]["endpoint"] == "/api/v1/datasets/recommend"
        assert actions["FIND_ALTERNATIVE_DATASET"]["payload"]["description"] == GOAL

    def test_no_transition_is_offered_without_a_project_to_serve(self, client):
        r = client.post(URL, data={"message": "analyze this, but I might need a better dataset"}, files=csv_file())
        assert "FIND_ALTERNATIVE_DATASET" not in {a["action"] for a in r.json()["next_actions"]}

    def test_a_registry_link_is_researched_not_downloaded(self, client):
        r = client.post(URL, data={"dataset_url": "https://archive.ics.uci.edu/dataset/53/iris"})
        body = r.json()
        assert body["journey"] == "USER_B_DATASET_ANALYSIS"
        assert body["analysis"]["analysis"]["mode"] == "external_metadata"
        assert body["analysis"]["analysis"]["ingestion"] is None

    def test_a_dataset_named_in_the_message_is_analysed(self, client):
        r = client.post(URL, data={"message": "I already have the Iris dataset and want to understand it"})
        body = r.json()
        assert body["journey"] == "USER_B_DATASET_ANALYSIS" and body["status"] == "COMPLETED"
        assert body["analysis"] is not None

    def test_a_claim_of_ownership_without_a_dataset_asks_for_it(self, client, stub):
        r = client.post(URL, data={"message": "I have a dataset and want to analyze it"})
        body = r.json()
        assert r.status_code == 200 and body["status"] == "NEEDS_INPUT"
        assert body["journey"] == "USER_B_DATASET_ANALYSIS"
        assert "Upload" in body["prompt_to_user"]
        assert body["analysis"] is None and stub.calls == []


class TestUserARouting:
    def test_a_project_description_finds_datasets(self, client, stub):
        r = client.post(URL, data={"project_goal": GOAL})
        body = r.json()
        assert r.status_code == 200 and body["status"] == "COMPLETED"
        assert body["journey"] == "USER_A_DATASET_DISCOVERY"
        assert body["discovery"]["count"] == 1 and body["analysis"] is None
        assert stub.calls and stub.calls[0]["description"] == GOAL

    def test_the_message_is_the_description_when_there_is_no_separate_goal(self, client, stub):
        client.post(URL, data={"message": "Find me a dataset to predict which customers will cancel their subscription"})
        assert "cancel their subscription" in stub.calls[0]["description"]

    def test_the_limit_is_passed_through_and_validated(self, client, stub):
        client.post(URL, data={"project_goal": GOAL, "limit": "3"})
        assert stub.calls[0]["limit"] == 3
        assert client.post(URL, data={"project_goal": GOAL, "limit": "500"}).status_code == 422

    def test_discovery_offers_a_way_into_analysis(self, client):
        actions = {a["action"] for a in client.post(URL, data={"project_goal": GOAL}).json()["next_actions"]}
        assert "UPLOAD_FILE" in actions

    def test_a_too_short_description_asks_for_more_instead_of_searching(self, client, stub):
        r = client.post(URL, data={"message": "find me a dataset"})
        body = r.json()
        assert body["status"] == "NEEDS_INPUT" and body["journey"] == "USER_A_DATASET_DISCOVERY"
        assert body["prompt_to_user"] and stub.calls == []

    def test_a_source_failure_uses_the_same_error_envelope_as_recommend(self, monkeypatch, client):
        from app.services.datasets.base import DatasetSourceError

        monkeypatch.setattr(datasets_api, "_service", _StubService(error=DatasetSourceError("KAGGLE_TIMEOUT", "x")))
        r = client.post(URL, data={"project_goal": GOAL})
        assert r.status_code == 504 and r.json()["detail"]["error"]["code"] == "DATASET_SOURCE_TIMEOUT"


class TestAmbiguity:
    def test_unclear_input_runs_nothing_and_asks(self, client, stub):
        r = client.post(URL, data={"message": "help"})
        body = r.json()
        assert r.status_code == 200 and body["status"] == "NEEDS_CLARIFICATION"
        assert body["journey"] is None and body["routing"]["intent"] == "AMBIGUOUS"
        assert body["prompt_to_user"] == body["routing"]["clarification"]
        assert body["analysis"] is None and body["discovery"] is None and stub.calls == []

    def test_an_empty_request_is_ambiguous_not_an_error(self, client):
        r = client.post(URL, data={})
        assert r.status_code == 200 and r.json()["status"] == "NEEDS_CLARIFICATION"

    def test_conflicting_signals_ask_which_journey(self, client, stub):
        r = client.post(URL, data={"message": "I have a dataset. Find me a dataset for churn prediction"})
        body = r.json()
        assert body["status"] == "NEEDS_CLARIFICATION" and body["routing"]["intent"] == "AMBIGUOUS"
        assert "analyze" in body["prompt_to_user"] and stub.calls == []


class TestSafetyBehindTheUnifiedEndpoint:
    def test_an_unsupported_file_is_refused(self, client):
        r = client.post(URL, data={"message": "analyze my file"}, files={"file": ("evil.exe", b"MZ....", "application/octet-stream")})
        assert r.status_code == 415

    def test_an_oversized_upload_is_refused_before_it_is_read(self, monkeypatch, client):
        too_big = b"a,b\n" + b"1,2\n" * (30 * 1024 * 1024 // 4)
        r = client.post(URL, data={"message": "analyze my file"}, files={"file": ("big.csv", too_big, "text/csv")})
        assert r.status_code == 413

    def test_nothing_is_left_on_the_server_after_an_upload(self, client):
        before = leftover_uploads()
        client.post(URL, data={"message": "here is my data"}, files=csv_file())
        client.post(URL, data={"message": "here is my data"}, files={"file": ("x.csv", b"", "text/csv")})
        assert leftover_uploads() == before

    def test_an_unsafe_url_is_refused_not_fetched(self, client):
        r = client.post(URL, data={"dataset_url": "http://169.254.169.254/latest/meta-data"})
        assert r.status_code == 422
        error = r.json()["detail"]["error"]
        assert error["code"] == "DATASET_URL_UNSUPPORTED" and "IP-address" in error["details"]["reason"]

    def test_the_message_is_never_echoed_in_logs(self, client, caplog):
        secret = "ZEBRA-QUOKKA-7731"
        with caplog.at_level("DEBUG"):
            client.post(URL, data={"message": f"I have a dataset about {secret}, analyze it"}, files=csv_file())
        assert secret not in caplog.text

    def test_the_route_is_in_the_openapi_schema(self, client):
        assert "/api/v1/assist" in client.get("/openapi.json").json()["paths"]


def test_the_module_exports_a_router():
    assert assist_module.router.prefix == "/assist"
    assert uci_record  # imported helper is the shared mock record builder
