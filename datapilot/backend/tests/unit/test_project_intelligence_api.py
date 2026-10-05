"""project_intelligence through the real endpoints: /datasets/analyze and /assist."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dataset_analysis as api
from app.main import app
from app.services.dataset_analysis.evaluation.datasets import GENERATORS
from app.services.dataset_analysis.service import DatasetAnalysisService
from app.services.project_intelligence import ProjectIntelligenceService
from tests.unit.pi_helpers import SETTINGS
from tests.unit.test_project_compatibility_gemini_external import FakeWriter, good_reply

ANALYZE = "/api/v1/datasets/analyze"
ASSIST = "/api/v1/assist"


def file(name="customer_churn"):
    return {"file": (f"{name}.csv", GENERATORS[name]().csv_bytes(), "text/csv")}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_service", DatasetAnalysisService(SETTINGS))
    monkeypatch.setattr(api, "_intelligence", ProjectIntelligenceService(SETTINGS))
    return TestClient(app, raise_server_exceptions=False)


def test_a_file_analysis_carries_project_intelligence(client):
    body = client.post(ANALYZE, files=file()).json()
    pi = body["project_intelligence"]
    assert pi["understanding"]["best_target"]["column"] == "churn"
    assert pi["feature_intelligence"]["available"] and pi["project_ideas"] and pi["model_recommendations"]
    assert pi["compatibility"] is None and pi["gemini"]["status"] == "not_requested"
    assert "No score is implied" in " ".join(pi["limitations"])


def test_the_existing_analysis_fields_are_unchanged(client):
    body = client.post(ANALYZE, files=file()).json()
    assert body["analysis"]["mode"] == "file_parsed" and body["headline"] == "DataPilot analyzed your actual file."


def test_it_can_be_switched_off(client):
    body = client.post(ANALYZE, files=file(), data={"include_project_intelligence": "false"}).json()
    assert body["project_intelligence"] is None and body["analysis"]["mode"] == "file_parsed"


def test_a_project_goal_adds_compatibility(client):
    body = client.post(ANALYZE, files=file(), data={"project_goal": "Predict customer churn using tenure and contract"}).json()
    assert body["project_intelligence"]["compatibility"]["status"] == "MATCH"


def test_a_contradicting_goal_offers_to_find_another_dataset(client):
    body = client.post(ANALYZE, files=file("house_prices"),
                       data={"project_goal": "Classify whether a patient has diabetes using BMI and glucose"}).json()
    assert body["project_intelligence"]["compatibility"]["status"] == "MISMATCH"
    action = next(a for a in body["next_actions"] if a["action"] == "FIND_ALTERNATIVE_DATASET")
    assert action["endpoint"] == "/api/v1/datasets/recommend"


def test_the_user_target_override_is_validated(client):
    ok = client.post(ANALYZE, files=file("house_prices"), data={"target": "bedrooms"}).json()
    assert ok["project_intelligence"]["understanding"]["target_source"] == "user_specified"
    bad = client.post(ANALYZE, files=file("house_prices"), data={"target": "no_such_column"})
    assert bad.status_code == 422 and bad.json()["detail"]["error"]["code"] == "TARGET_COLUMN_NOT_FOUND"


def test_use_ai_without_a_key_degrades_and_never_fails(client):
    r = client.post(ANALYZE, files=file(), data={"use_ai": "true"})
    assert r.status_code == 200 and r.json()["project_intelligence"]["gemini"]["status"] == "not_configured"


def test_use_ai_with_a_writer_marks_the_ideas(monkeypatch, client):
    monkeypatch.setattr(api, "_intelligence", ProjectIntelligenceService(SETTINGS, writer=FakeWriter(good_reply)))
    body = client.post(ANALYZE, files=file(), data={"use_ai": "true"}).json()
    assert body["project_intelligence"]["gemini"]["status"] == "used"
    assert {i["source"] for i in body["project_intelligence"]["project_ideas"]} == {"gemini_enhanced"}


def test_a_failure_in_project_intelligence_never_fails_the_analysis(monkeypatch, client):
    class Boom:
        async def build(self, *a, **k):
            raise RuntimeError("secret detail")

    monkeypatch.setattr(api, "_intelligence", Boom())
    r = client.post(ANALYZE, files=file())
    assert r.status_code == 200 and r.json()["project_intelligence"] is None
    assert any("could not be produced" in n for n in r.json()["notices"]) and "secret detail" not in r.text


def test_assist_passes_the_goal_and_returns_compatibility(client):
    body = client.post(ASSIST, files=file(), data={
        "message": "here is my data", "project_goal": "Predict customer churn using tenure and contract"}).json()
    assert body["journey"] == "USER_B_DATASET_ANALYSIS"
    assert body["analysis"]["project_intelligence"]["compatibility"]["status"] == "MATCH"


def test_assist_adds_the_alternative_dataset_action_once(client):
    body = client.post(ASSIST, files=file("house_prices"), data={
        "message": "analyze my file, if it isn't suitable find a different dataset",
        "project_goal": "Classify whether a patient has diabetes using BMI and glucose"}).json()
    assert [a["action"] for a in body["next_actions"]].count("FIND_ALTERNATIVE_DATASET") == 1


def test_the_response_never_contains_a_cell_value(client):
    text = client.post(ANALYZE, files=file(), data={"use_ai": "false"}).text
    assert "user1@example.com" not in text and "C20000" not in text
