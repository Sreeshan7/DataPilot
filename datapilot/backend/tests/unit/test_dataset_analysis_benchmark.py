"""The Phase 4 benchmark, pinned.

Thresholds are set at the values measured when the benchmark was introduced (after the defects it
exposed were fixed), so a regression fails here instead of passing silently. They describe behaviour
on this project's own synthetic data and hand-written phrasings; they are not real-world accuracy.

The second half is the rule that matters most for a dataset known only by name or link: whatever
the registry says, the analysis must never carry statistics, quality findings, parsed-file evidence
or a target stated more strongly than the registry supports.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.v1 import dataset_analysis as analysis_api
from app.core.config import Settings
from app.schemas.dataset_analysis import CandidateState, EvidenceLevel, QualityCode
from app.services.dataset_analysis.evaluation import benchmark as bench
from app.services.dataset_analysis.evaluation.datasets import GENERATORS, HELDOUT_GENERATORS
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


@pytest.fixture(scope="module")
def report():
    return bench.run_benchmark()


class TestSafetyRules:
    def test_no_safety_rule_is_violated(self, report):
        assert report.safety_violations == []

    def test_no_strong_target_is_claimed_where_none_exists(self, report):
        assert report.metrics["strong_target_false_positive_rate"] == 0.0

    def test_no_task_is_offered_that_the_data_cannot_support(self, report):
        assert report.metrics["unsupported_tasks_offered"] == 0.0

    def test_no_request_is_sent_to_the_wrong_journey_on_any_batch(self, report):
        for batch in ("routing", "routing_heldout", "routing_round2", "routing_unseen"):
            assert report.metrics[f"{batch}_wrong_journey"] == 0.0, batch


class TestAnalysisQuality:
    def test_the_right_target_leads_on_every_dataset_that_has_one(self, report):
        assert report.metrics["target_top1_accuracy"] == 1.0
        assert report.metrics["heldout_target_top1_accuracy"] == 1.0

    def test_identifiers_are_found_without_false_alarms(self, report):
        assert report.metrics["identifier_recall"] == 1.0
        assert report.metrics["identifier_false_positives"] == 0.0

    def test_sensitive_and_free_text_columns_are_found(self, report):
        assert report.metrics["sensitive_recall"] == 1.0
        assert report.metrics["free_text_recall"] == 1.0

    def test_every_planted_problem_is_reported(self, report):
        assert report.metrics["planted_finding_recall"] == 1.0

    def test_clean_datasets_get_no_false_alarms(self, report):
        assert report.metrics["false_alarm_findings"] == 0.0

    def test_every_expected_task_is_offered(self, report):
        assert report.metrics["task_recall"] == 1.0

    def test_the_result_does_not_depend_on_how_the_file_was_written(self, report):
        """Same table as semicolon / tab / CRLF / BOM files gives the same conclusions."""
        assert report.variant_mismatches == []

    def test_a_very_small_dataset_is_never_presented_as_a_supported_task(self, report):
        tiny = next(d for d in report.datasets if d.name == "tiny_table")
        assert tiny.tasks_forbidden_present == []


class TestRouting:
    def test_the_tuned_and_held_out_batches_are_resolved(self, report):
        for batch in ("routing", "routing_heldout", "routing_round2"):
            assert report.metrics[f"{batch}_accuracy"] == 1.0, batch

    def test_follow_up_journeys_are_recognised(self, report):
        for batch in ("routing", "routing_heldout", "routing_unseen"):
            assert report.metrics[f"{batch}_follow_up_recall"] == 1.0, batch

    def test_unseen_phrasings_resolve_about_two_times_in_three_and_the_rest_ask(self, report):
        """Measured once, never tuned. Falling below is a regression; every miss must be a question."""
        assert report.metrics["routing_unseen_accuracy"] >= 0.65
        assert all(m.got == "AMBIGUOUS" for m in report.unseen_routing_misses)


class TestTheBenchmarkIsDeterministic:
    def test_two_runs_agree_on_every_digest_and_metric(self, report):
        again = bench.run_benchmark(include_variants=False)
        assert [d.digest for d in again.datasets] == [d.digest for d in report.datasets]
        assert {k: v for k, v in again.metrics.items() if not k.startswith("format")} == {
            k: v for k, v in report.metrics.items() if not k.startswith("format")}

    def test_the_digest_ignores_clocks_but_not_conclusions(self):
        content = GENERATORS["customer_churn"]().csv_bytes()
        first, second = bench.analyze_bytes(content), bench.analyze_bytes(content)
        assert first.metadata.analysis_id != second.metadata.analysis_id
        assert bench.analysis_digest(first) == bench.analysis_digest(second)
        other = bench.analyze_bytes(GENERATORS["house_prices"]().csv_bytes())
        assert bench.analysis_digest(first) != bench.analysis_digest(other)


class TestTheBenchmarkCoversWhatItClaims:
    def test_every_generated_dataset_has_ground_truth(self):
        assert set(GENERATORS) == set(bench.TRUTH)
        assert set(HELDOUT_GENERATORS) == set(bench.HELDOUT_TRUTH)

    def test_the_ground_truth_names_real_columns(self):
        for name, factory in {**GENERATORS, **HELDOUT_GENERATORS}.items():
            columns = set(factory().columns)
            truth = {**bench.TRUTH, **bench.HELDOUT_TRUTH}[name]
            named = (truth.targets | truth.identifiers | truth.sensitive | truth.free_text
                     | truth.ambiguous_identifiers | {c for _, c in truth.findings if c})
            assert named <= columns, (name, named - columns)

    def test_the_planted_leak_is_in_the_leaky_dataset_only(self, report):
        by_name = {d.name: d for d in report.datasets}
        assert by_name["student_performance_leaky"].findings_missed == []
        assert QualityCode.LEAKAGE_SIGNAL.value not in by_name["student_performance"].forbidden_findings_present

    def test_the_held_out_sets_are_marked_as_such(self, report):
        assert {d.name for d in report.datasets if d.heldout} == set(HELDOUT_GENERATORS)

    def test_a_deliberately_wrong_truth_is_caught(self):
        """The benchmark must be able to fail: claim a different target and it reports a miss."""
        dataset = GENERATORS["house_prices"]()
        analysis = bench.analyze_bytes(dataset.csv_bytes())
        wrong = bench.Truth(targets=frozenset({"bedrooms"}), identifiers=frozenset({"year_built"}))
        result = bench.evaluate_dataset("house_prices", dataset, wrong, analysis)
        assert result.target_top1_correct is False and result.identifiers_missed == ["year_built"]


# ---------------------------------------------------------------------------
# External-only honesty
# ---------------------------------------------------------------------------

URL = "/api/v1/datasets/analyze"


def _iris(mock):
    return (mock.route("archive.ics.uci.edu/api/datasets/list", {"status": 200, "data": [{"id": 53, "name": "Iris"}]})
            .route("archive.ics.uci.edu/api/dataset?", uci_record())
            .route("openml.org/api/v1/json/data/list", lambda r: httpx.Response(412, json={}))
            .route("huggingface.co/api/datasets?", []).route("kaggle.com/api/v1/datasets/list?", []))


def _empty(mock):
    return (mock.route("archive.ics.uci.edu", {"status": 200, "data": []})
            .route("openml.org", lambda r: httpx.Response(412, json={}))
            .route("huggingface.co", []).route("kaggle.com", []))


def _openml_link(mock):
    return (mock.route("/data/features/", openml_features(("a", "numeric", False), ("y", "nominal", True)))
            .route("/data/qualities/", openml_qualities(NumberOfInstances=100, NumberOfClasses=2))
            .route("/data/61", openml_desc()))


def _kaggle_no_schema(mock):
    return (mock.route("/datasets/view/", kaggle_view(description="A famous dataset."))
            .route("/datasets/list/owner/titanic", {"datasetFiles": []}))


def _kaggle_text_columns(mock):
    return (mock.route("/datasets/view/", kaggle_view())
            .route("/datasets/list/owner/titanic", {"datasetFiles": []}))


def _hf_link(mock):
    return mock.route("huggingface.co/api/datasets/owner/titanic", hf_record("owner/titanic"))


def _hf_no_features(mock):
    return mock.route("huggingface.co/api/datasets/owner/titanic", hf_record("owner/titanic", features=False))


EXTERNAL_CASES = {
    "uci_name_with_schema": (_iris, {"dataset_name": "Iris"}),
    "openml_link": (_openml_link, {"dataset_url": "https://www.openml.org/d/61"}),
    "huggingface_link_with_features": (_hf_link, {"dataset_url": "https://huggingface.co/datasets/owner/titanic"}),
    "huggingface_link_without_features": (_hf_no_features, {"dataset_url": "https://huggingface.co/datasets/owner/titanic"}),
    "kaggle_link_without_schema": (_kaggle_no_schema, {"dataset_url": "https://www.kaggle.com/datasets/owner/titanic"}),
    "kaggle_link_columns_from_text": (_kaggle_text_columns, {"dataset_url": "https://www.kaggle.com/datasets/owner/titanic"}),
    "unknown_name": (_empty, {"dataset_name": "Zzyzx Totally Fake Data"}),
    "ambiguous_name": (lambda m: m, {"dataset_name": "student"}),
}


@pytest.fixture(params=sorted(EXTERNAL_CASES))
def external(request, monkeypatch):
    configure, form = EXTERNAL_CASES[request.param]
    settings = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing")
    mock = configure(MockRegistries())
    external_service = ExternalDatasetMetadataService(settings, mock.fetcher(settings))
    monkeypatch.setattr(analysis_api, "_service", DatasetAnalysisService(settings, external=external_service))
    from app.main import app

    response = TestClient(app, raise_server_exceptions=False).post(URL, data=form)
    assert response.status_code == 200, response.text
    return request.param, response


class TestExternalOnlyNeverPretends:
    def test_no_statistic_finding_or_row_is_present(self, external):
        analysis = external[1].json()["analysis"]
        assert analysis["mode"] == "external_metadata"
        assert analysis["column_profiles"] == [] and analysis["quality"] is None and analysis["ingestion"] is None

    def test_nothing_anywhere_in_the_response_is_marked_as_parsed(self, external):
        assert "datapilot_parsed" not in external[1].text

    def test_the_response_says_no_rows_were_read(self, external):
        body = external[1].json()
        analysis = body["analysis"]
        said = (
            "no rows were read" in body["evidence_statement"]
            or "could not identify" in body["headline"]
            or "matches several datasets" in body["headline"]
            or analysis["external"] is None
        )
        assert said, body["headline"]
        assert analysis["input_resolution"]["raw_file_available"] is False

    def test_a_target_is_strong_only_when_the_registry_designates_it(self, external):
        for target in external[1].json()["analysis"]["target_candidates"]:
            if target["state"] == "strong_candidate":
                assert any("designates" in e["statement"] and e["level"] == "registry_verified"
                           for e in target["evidence"]), target
            assert all(e["level"] in ("registry_verified", "heuristic") for e in target["evidence"])

    def test_every_task_admits_that_no_rows_were_read(self, external):
        for task in external[1].json()["analysis"]["task_candidates"]:
            assert any("no rows were read" in l for l in task["limitations"]), task
            assert task["state"] in ("possible", "weak")

    def test_without_a_schema_there_is_no_target_and_no_task(self, external):
        analysis = external[1].json()["analysis"]
        if analysis["schema"]["level"] == "unknown":
            assert analysis["target_candidates"] == [] and analysis["task_candidates"] == []

    def test_an_unidentified_dataset_has_no_invented_identity(self, external):
        name, response = external
        analysis = response.json()["analysis"]
        if name in ("unknown_name", "ambiguous_name"):
            assert analysis["external"] is None and analysis["schema"]["columns"] == []
            assert analysis["schema"]["level"] == "unknown"

    def test_a_next_step_is_always_offered(self, external):
        assert external[1].json()["recommended_next_step"]

    def test_the_schema_level_is_never_stronger_than_its_source(self, external):
        name, response = external
        level = response.json()["analysis"]["schema"]["level"]
        if name == "kaggle_link_columns_from_text":
            assert level == "inferred"
        if name in ("kaggle_link_without_schema", "huggingface_link_without_features"):
            assert level == "unknown"
        assert level != "datapilot_parsed"


def test_registry_evidence_is_never_confused_with_parsed_evidence():
    assert EvidenceLevel.REGISTRY_VERIFIED is not EvidenceLevel.DATAPILOT_PARSED
    assert CandidateState.STRONG_CANDIDATE.value == "strong_candidate"
