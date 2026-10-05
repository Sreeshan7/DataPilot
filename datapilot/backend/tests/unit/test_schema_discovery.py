"""Finding a dataset's structure when no registry publishes it.

Order of trust: a registry's structured schema > the real header read from the dataset's own file (first KB only) >
an AI model's reading of cited web pages (a lead to confirm). Each is labelled as what it is, none is ever shown as
parsed rows, and nothing here may hide a failure or invent a column.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.schemas.dataset_analysis import AnalysisMode, EvidenceLevel, ResolutionStatus
from app.services.dataset_analysis.external import web_knowledge as wk
from app.services.dataset_analysis.external.fetcher import FetchError, SafeFetcher
from app.services.dataset_analysis.external.peek import PEEK_BYTES, choose_file, sample_schema
from app.services.dataset_analysis.external.service import ExternalDatasetMetadataService
from app.services.dataset_analysis.external.web_knowledge import WebDatasetResearcher
from app.services.dataset_analysis.service import AnalysisRequest, DatasetAnalysisService
from app.services.llm_service import LLMServiceError
from tests.unit.pi_helpers import build, light_settings, run
from tests.unit.registry_mock import MockRegistries, hf_record, kaggle_view

KAGGLE_URL = "https://www.kaggle.com/datasets/owner/students"
HEADER = b"Hours_Studied,Attendance,Gender,Passed,Notes,Exam_Score\n" + b"".join(
    f"{i % 12},{50 + i % 50},{'M' if i % 2 else 'F'},{'yes' if i % 3 else 'no'},some long free text {'x' * 45} {i},{40 + (i * 7) % 60}\n".encode()
    for i in range(120))


def settings(**kw):
    return light_settings(external_enabled=True, kaggle_api_token="KGAT_SECRET_TOKEN", **kw)


# ----------------------------------------------------------------------------------------------- the sampler


class TestSampleSchema:
    def test_names_and_kinds_come_from_the_header_and_a_few_rows(self):
        s = sample_schema(HEADER, total_bytes=100_000)
        kinds = {c.name: c.kind for c in s.columns}
        assert list(kinds) == ["Hours_Studied", "Attendance", "Gender", "Passed", "Notes", "Exam_Score"]
        assert kinds["Exam_Score"] == "Continuous" and kinds["Gender"] == "Binary" == kinds["Passed"]
        assert kinds["Notes"] == "Text" and kinds["Hours_Studied"] == "Integer" and s.sample_rows == 120

    def test_a_row_estimate_is_only_given_with_enough_sample_and_a_known_size(self):
        assert 500 < sample_schema(HEADER, total_bytes=100_000).estimated_rows < 2500
        assert sample_schema(HEADER, total_bytes=None).estimated_rows is None
        assert sample_schema(b"a,b\n1,2\n3,4\n", total_bytes=10_000).estimated_rows is None  # too few rows to trust

    def test_other_delimiters_bom_and_a_cut_off_last_row(self):
        assert [c.name for c in sample_schema("﻿a;b;c\n1;2;3\n4;5;6\n".encode("utf-8")).columns] == ["a", "b", "c"]
        cut = HEADER[: PEEK_BYTES - 1] if len(HEADER) >= PEEK_BYTES else HEADER[:-7]
        assert sample_schema(cut) is not None

    def test_binary_and_non_tables_are_refused(self):
        assert sample_schema(b"PK\x03\x04\x00\x00binary") is None
        assert sample_schema(b"just one line of prose with no newline") is None
        assert sample_schema(b"1,2,3\n4,5,6\n") is None  # a header of numbers is data, not names

    def test_unsafe_column_names_are_replaced_not_passed_on(self):
        s = sample_schema(b"ok,<script>alert(1)</script>,fine\n1,2,3\n4,5,6\n")
        assert [c.name for c in s.columns] == ["ok", "column_2", "fine"]

    def test_the_sample_never_carries_cell_values(self):
        s = sample_schema(HEADER, 1000)
        assert "some long free text" not in repr(s)

    def test_choosing_the_main_file(self):
        assert choose_file(["sample_submission.csv", "test.csv", "train.csv", "readme.md"]) == "train.csv"
        assert choose_file(["a.png", "b.zip"]) is None
        assert choose_file(["data.csv"]) == "data.csv"


# ------------------------------------------------------------------------------- the scoped prefix reader


def storage_mock(body=HEADER, status=206, location="https://storage.googleapis.com:443/kagglesdsdata/x.csv?X-Goog-Signature=SECRETSIG"):
    mock = MockRegistries()
    seen = {"storage_headers": None, "kaggle_range": None}

    def kaggle(request):
        seen["kaggle_range"] = request.headers.get("range")
        return httpx.Response(302, headers={"location": location})

    def storage(request):
        seen["storage_headers"] = dict(request.headers)
        full = len(body)
        return httpx.Response(status, content=body, headers={"content-range": f"bytes 0-{full - 1}/{full * 50}"} if status == 206 else {})

    mock.route("kaggle.com/api/v1/datasets/download", kaggle).route("storage.googleapis.com", storage)
    return mock, seen


HOSTS = frozenset({"storage.googleapis.com"})
DL = "https://www.kaggle.com/api/v1/datasets/download/owner/students?fileName=data.csv"


def prefix(mock, max_bytes=4096, **kw):
    fetcher = mock.fetcher(settings())
    return run(fetcher.get_prefix(DL, {"Authorization": "Bearer KGAT_SECRET_TOKEN"}, max_bytes=max_bytes, storage_hosts=HOSTS, **kw))


class TestGetPrefix:
    def test_reads_only_the_first_bytes_and_reports_the_full_size(self):
        mock, seen = storage_mock()
        data, total = prefix(mock, max_bytes=1000)
        assert len(data) == 1000 and data == HEADER[:1000] and total == len(HEADER) * 50
        assert seen["kaggle_range"] == "bytes=0-999"

    def test_stops_at_the_cap_even_if_the_server_ignores_range(self):
        mock, _ = storage_mock(status=200)
        data, _ = prefix(mock, max_bytes=500)
        assert len(data) == 500

    def test_the_credential_goes_to_kaggle_and_never_to_the_file_store(self):
        mock, seen = storage_mock()
        prefix(mock)
        assert "authorization" not in {k.lower() for k in seen["storage_headers"]}

    @pytest.mark.parametrize("location", [
        "https://evil.example/x.csv", "http://storage.googleapis.com/x.csv", "https://169.254.169.254/latest",
        "https://storage.googleapis.com:8443/x.csv", "https://user:pw@storage.googleapis.com/x.csv"])
    def test_a_redirect_anywhere_else_is_refused(self, location):
        mock, _ = storage_mock(location=location)
        with pytest.raises(FetchError) as e:
            prefix(mock)
        assert e.value.code == FetchError.UNSAFE

    def test_a_non_registry_url_is_refused_before_any_request(self):
        mock, _ = storage_mock()
        with pytest.raises(FetchError):
            run(mock.fetcher(settings()).get_prefix("https://evil.example/f.csv", {}, max_bytes=100, storage_hosts=HOSTS))
        assert mock.requests == []

    def test_the_signed_url_never_appears_in_an_error(self):
        mock, _ = storage_mock(status=500)
        with pytest.raises(FetchError) as e:
            prefix(mock)
        assert "SECRETSIG" not in e.value.message and "X-Goog" not in e.value.message


# ------------------------------------------------------------------------------------- Kaggle end to end


def kaggle_mock(body=HEADER, files=("data.csv",), description="A famous dataset."):
    mock, seen = storage_mock(body)
    (mock.route("/datasets/view/", kaggle_view(description=description))
     .route("/datasets/list/owner/students", {"datasetFiles": [{"name": n} for n in files]}))
    return mock, seen


def analyse(mock, **kw):
    st = settings(**kw)
    external = ExternalDatasetMetadataService(st, mock.fetcher(st), web_researcher=kw.pop("web", None) or WebDatasetResearcher(st, None))
    return run(DatasetAnalysisService(st, external=external).analyze(AnalysisRequest(dataset_url=KAGGLE_URL)))


class TestKaggleHeaderRead:
    def test_columns_types_and_a_row_estimate_come_from_the_real_file(self):
        a = analyse(kaggle_mock()[0]).analysis
        assert a.schema_info.level is EvidenceLevel.REGISTRY_VERIFIED and a.schema_info.source.startswith("file header")
        assert [c.normalized_name for c in a.schema_info.columns][:3] == ["hours_studied", "attendance", "gender"]
        assert {c.normalized_name: c.data_type for c in a.schema_info.columns}["exam_score"] == "Continuous"
        assert "estimated from its size" in a.schema_info.note and "Missingness" in a.schema_info.note

    def test_the_row_count_is_an_estimate_and_is_labelled_as_one(self):
        a = analyse(kaggle_mock()[0]).analysis
        assert a.schema_info.row_count and 3000 < a.schema_info.row_count < 12000  # the mock file is 50x the sampled bytes
        assert a.schema_info.row_count_evidence is EvidenceLevel.INFERRED

    def test_it_is_still_clearly_not_an_analysis_of_the_data(self):
        a = analyse(kaggle_mock()[0]).analysis
        assert a.mode is AnalysisMode.EXTERNAL_METADATA and a.column_profiles == [] and a.quality is None and a.ingestion is None
        assert "datapilot_parsed" not in a.model_dump_json()

    def test_a_target_and_task_are_suggested_but_never_strong(self):
        a = analyse(kaggle_mock()[0]).analysis
        best = a.target_candidates[0]
        assert best.column == "exam_score" and best.state.value == "possible"
        assert any(t.task.value == "regression" and t.target == "exam_score" for t in a.task_candidates)
        assert all(t.state.value != "strong_candidate" for t in a.target_candidates)

    def test_project_ideas_are_now_grounded_in_the_real_columns(self):
        pi = build(analyse(kaggle_mock()[0]), goal="Predict the exam score using hours studied and attendance")
        assert pi.project_ideas and pi.project_ideas[0].target == "exam_score"
        assert {"hours_studied", "attendance"} <= set(pi.project_ideas[0].required_features + pi.project_ideas[0].optional_features)
        assert pi.compatibility.status.value == "MATCH"
        assert pi.feature_intelligence.available is False  # importance still needs the rows

    def test_a_failed_peek_falls_back_to_the_description_and_says_why(self):
        mock = kaggle_mock(description="Columns: Alpha, Beta, Gamma, Delta.")[0]
        mock.route("storage.googleapis.com", lambda r: httpx.Response(500))  # the last route wins
        a = analyse(mock).analysis
        assert a.schema_info.level is EvidenceLevel.INFERRED and "Reading the file's header was not possible" in a.schema_info.note

    def test_no_peek_without_a_token_or_when_switched_off(self):
        mock, _ = kaggle_mock()
        a = analyse(mock, external_peek_enabled=False).analysis
        assert a.schema_info.level is EvidenceLevel.UNKNOWN
        assert not any("storage.googleapis" in str(r.url) for r in mock.requests)

    def test_an_archive_only_dataset_is_not_peeked(self):
        a = analyse(kaggle_mock(files=("images.zip",))[0]).analysis
        assert a.schema_info.level is EvidenceLevel.UNKNOWN


# ----------------------------------------------------------------------------------- Hugging Face server


class TestHuggingFaceServerSchema:
    INFO = {"dataset_info": {"default": {"features": {
        "text": {"dtype": "string", "_type": "Value"}, "label": {"names": ["neg", "pos"], "_type": "ClassLabel"}},
        "splits": {"train": {"num_examples": 25000}, "test": {"num_examples": 25000}}}}}

    def test_a_card_without_features_is_completed_from_the_datasets_server(self):
        mock = (MockRegistries().route("huggingface.co/api/datasets/owner/reviews", hf_record("owner/reviews", features=False))
                .route("datasets-server.huggingface.co/info", self.INFO))
        st = settings()
        out = run(DatasetAnalysisService(st, external=ExternalDatasetMetadataService(st, mock.fetcher(st), web_researcher=WebDatasetResearcher(st, None)))
                  .analyze(AnalysisRequest(dataset_url="https://huggingface.co/datasets/owner/reviews")))
        a = out.analysis
        assert a.schema_info.source == "Hugging Face datasets-server info" and a.schema_info.row_count == 50000
        assert {c.normalized_name: c.data_type for c in a.schema_info.columns} == {"text": "string", "label": "class_label"}
        assert a.target_candidates[0].column == "label" and a.target_candidates[0].target_kind.value == "binary"


# ------------------------------------------------------------------------------------- the web lookup


GOOD = ('{"found": true, "dataset_name": "Shop Churn", "description": "Customer churn at a shop.", "row_count": 7043, '
        '"columns": [{"name": "tenure", "kind": "integer"}, {"name": "monthly_charges", "kind": "numeric"}, {"name": "churn", "kind": "binary"}]}')
SOURCES = [("kaggle.com", "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc")]


class FakeSearcher:
    def __init__(self, text=GOOD, sources=SOURCES, error=None, configured=True):
        self.text, self.sources, self.error, self._c, self.calls = text, sources, error, configured, 0

    @property
    def is_configured(self):
        return self._c

    async def search_grounded(self, prompt, system):
        self.calls += 1
        self.prompt = prompt
        if self.error:
            raise self.error
        return self.text, self.sources


class TestWebKnowledgeValidation:
    def test_a_grounded_valid_reply_is_accepted(self):
        k = wk.validate(GOOD, SOURCES)
        assert [c.name for c in k.columns] == ["tenure", "monthly_charges", "churn"] and k.row_count == 7043
        assert [c.kind for c in k.columns] == ["Integer", "Continuous", "Binary"] and k.sources == [("kaggle.com", SOURCES[0][1])]

    def test_an_answer_with_no_cited_sources_is_discarded_however_confident(self):
        assert wk.validate(GOOD, []) is None
        assert wk.validate(GOOD, [("x", "http://not-https.example")]) is None

    @pytest.mark.parametrize("text", [
        "not json at all", '{"found": false}', '{"found": true, "columns": "tenure"}',
        '{"found": true, "columns": [{"name": "only_one"}]}', "```json\n{broken\n```"])
    def test_malformed_or_negative_replies_are_discarded(self, text):
        assert wk.validate(text, SOURCES) is None

    def test_unsafe_names_duplicates_and_silly_numbers_are_dropped(self):
        text = ('{"found": true, "row_count": -5, "dataset_name": "X", "columns": [{"name": "ok"}, {"name": "OK"}, '
                '{"name": "<img onerror=1>"}, {"name": "fine_two"}, {"name": "' + "x" * 200 + '"}]}')
        k = wk.validate(text, SOURCES)
        assert [c.name for c in k.columns] == ["ok", "fine_two"] and k.row_count is None

    def test_a_fenced_json_reply_is_read(self):
        assert wk.validate("Here you go:\n```json\n" + GOOD + "\n```", SOURCES) is not None

    def test_the_description_is_capped_and_cleaned(self):
        text = '{"found": true, "description": "' + "a\\u0007 " * 400 + '", "columns": [{"name": "a1"}, {"name": "b2"}]}'
        assert len(wk.validate(text, SOURCES).description) <= wk.MAX_DESCRIPTION


class TestWebDatasetResearcher:
    def researcher(self, searcher, **kw):
        return WebDatasetResearcher(settings(**kw), searcher)

    def test_one_grounded_call_then_the_cache(self):
        s = FakeSearcher()
        r = self.researcher(s)
        first, _ = run(r.lookup("Shop Churn"))
        second, _ = run(r.lookup("shop churn"))
        assert first is second and s.calls == 1

    def test_quota_and_auth_errors_mean_no_lookup_not_a_crash_and_are_not_cached(self):
        s = FakeSearcher(error=LLMServiceError("LLM_RATE_LIMITED", "quota"))
        r = self.researcher(s)
        k, why = run(r.lookup("Shop Churn"))
        assert k is None and "LLM_RATE_LIMITED" in why
        s.error = None
        assert run(r.lookup("Shop Churn"))[0] is not None  # recovered: the failure was not remembered

    def test_no_key_and_switched_off_do_nothing(self):
        assert "no Gemini key" in run(self.researcher(FakeSearcher(configured=False)).lookup("x"))[1]
        s = FakeSearcher()
        assert run(self.researcher(s, web_lookup_enabled=False).lookup("x"))[0] is None and s.calls == 0

    def test_a_slow_lookup_times_out(self):
        class Slow(FakeSearcher):
            async def search_grounded(self, prompt, system):
                await asyncio.sleep(2)

        k, why = run(self.researcher(Slow(), web_lookup_timeout_seconds=0.05).lookup("x"))
        assert k is None and "timed out" in why

    def test_the_prompt_is_only_the_name_or_link(self):
        s = FakeSearcher()
        run(self.researcher(s).lookup("Shop Churn", "https://www.kaggle.com/datasets/a/b", "kaggle"))
        assert "https://www.kaggle.com/datasets/a/b" in s.prompt and "KGAT" not in s.prompt


class TestWebFallbackInTheAnalysis:
    def unknown_name_mock(self):
        return (MockRegistries().route("archive.ics.uci.edu", {"status": 200, "data": []})
                .route("openml.org", lambda r: httpx.Response(412, json={}))
                .route("huggingface.co", []).route("kaggle.com", []))

    def analyse_name(self, searcher, name="Shop Churn Telco"):
        st = settings()
        external = ExternalDatasetMetadataService(st, self.unknown_name_mock().fetcher(st), web_researcher=WebDatasetResearcher(st, searcher))
        return run(DatasetAnalysisService(st, external=external).analyze(AnalysisRequest(dataset_name=name)))

    def test_a_name_no_registry_knows_is_found_on_the_web_and_labelled_ai(self):
        out = self.analyse_name(FakeSearcher())
        a = out.analysis
        assert a.schema_info.level is EvidenceLevel.AI_INFERRED and a.schema_info.row_count_evidence is EvidenceLevel.AI_INFERRED
        assert a.external.source == "web" and a.external.web_sources[0].url == SOURCES[0][1]
        res = a.input_resolution
        assert res.resolution_status is ResolutionStatus.PROBABLE and res.needs_confirmation and res.resolution_confidence <= 0.5
        assert all(c.evidence is EvidenceLevel.AI_INFERRED for c in a.schema_info.columns)
        assert any("AI model reading web pages" in lim for lim in a.limitations)

    def test_ai_found_structure_is_never_verified_parsed_or_a_strong_target(self):
        a = self.analyse_name(FakeSearcher()).analysis
        assert "datapilot_parsed" not in a.model_dump_json() and "registry_verified" not in a.schema_info.model_dump_json()
        assert all(t.state.value != "strong_candidate" for t in a.target_candidates)
        assert a.target_candidates[0].column == "churn"
        assert any("read by AI" in e.statement for e in a.target_candidates[0].evidence)

    def test_it_can_ground_ideas_but_cannot_prove_a_mismatch(self):
        pi = build(self.analyse_name(FakeSearcher()), goal="Predict hospital readmission using age and diagnosis")
        assert pi.compatibility.status.value != "MISMATCH" and not pi.compatibility.suggests_alternative_dataset
        assert all(i.evidence_level is EvidenceLevel.AI_INFERRED for i in pi.project_ideas)

    def test_when_the_web_cannot_help_the_name_stays_unresolved_and_says_why(self):
        a = self.analyse_name(FakeSearcher(error=LLMServiceError("LLM_RATE_LIMITED", "q"))).analysis
        assert a.external is None and a.schema_info.level is EvidenceLevel.UNKNOWN
        assert any("web lookup" in lim and "LLM_RATE_LIMITED" in lim for lim in a.limitations)

    def test_an_ungrounded_answer_leaves_it_unresolved(self):
        a = self.analyse_name(FakeSearcher(sources=[])).analysis
        assert a.external is None and a.input_resolution.resolution_status is ResolutionStatus.UNRESOLVED

    def test_an_ambiguous_generic_name_never_triggers_a_web_lookup(self):
        s = FakeSearcher()
        out = self.analyse_name(s, name="student")
        assert s.calls == 0 and out.analysis.external is None

    def test_a_registry_dataset_with_a_real_schema_is_not_second_guessed(self):
        s = FakeSearcher()
        mock = kaggle_mock()[0]
        st = settings()
        external = ExternalDatasetMetadataService(st, mock.fetcher(st), web_researcher=WebDatasetResearcher(st, s))
        run(DatasetAnalysisService(st, external=external).analyze(AnalysisRequest(dataset_url=KAGGLE_URL)))
        assert s.calls == 0

    def test_a_registry_dataset_with_no_columns_anywhere_is_completed_by_the_web(self):
        s = FakeSearcher()
        mock = kaggle_mock(files=("images.zip",))[0]
        st = settings()
        external = ExternalDatasetMetadataService(st, mock.fetcher(st), web_researcher=WebDatasetResearcher(st, s))
        a = run(DatasetAnalysisService(st, external=external).analyze(AnalysisRequest(dataset_url=KAGGLE_URL))).analysis
        assert s.calls == 1 and a.schema_info.level is EvidenceLevel.AI_INFERRED and a.external.source == "kaggle"
        assert a.input_resolution.resolution_status is ResolutionStatus.RESOLVED


class TestIdentifiersInTheSample:
    def table(self, extra_name="student_id", unique=True, rows=120):
        body = "".join(f"{i if unique else i % 5},{i % 7},{50 + i % 40}\n" for i in range(rows))
        return f"{extra_name},hours,score\n{body}".encode()

    def role(self, raw, name="student_id"):
        return {c.name: c.role for c in sample_schema(raw).columns}[name]

    def test_a_unique_id_column_is_marked_as_a_row_identifier(self):
        assert self.role(self.table()) == "row_id"

    def test_a_repeating_id_named_column_is_not(self):
        assert self.role(self.table(unique=False)) is None

    def test_a_word_that_merely_ends_in_id_is_not_an_identifier(self):
        assert self.role(self.table("paid"), "paid") is None

    def test_too_few_rows_prove_nothing(self):
        assert self.role(self.table(rows=12)) is None

    def test_identifiers_are_kept_out_of_ideas_and_target_matching(self):
        mock = kaggle_mock(body=self.table() + b"")[0]
        pi = build(analyse(mock), goal="Predict the score for each student id using hours")
        used = set()
        for idea in pi.project_ideas:
            used |= set(idea.required_features + idea.optional_features)
        assert "student_id" not in used
        features = {f.column: f.role.value for f in pi.feature_intelligence.features}
        assert features.get("student_id") == "IDENTIFIER"
