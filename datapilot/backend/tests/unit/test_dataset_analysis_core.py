"""Targets, tasks and quality findings, end to end through real CSV ingestion.

The datasets are deterministic and shaped like the ones people bring. Each test asserts a
behaviour a user would be misled about if it failed: a wrong target, a supervised task with
no target, a leak that goes unreported, a finding with no numbers behind it.
"""

from __future__ import annotations

import json
import os

import pytest

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    CandidateState,
    ColumnFlag,
    EvidenceLevel,
    MLTask,
    QualityCode,
    TargetKind,
)
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisError
from app.services.dataset_analysis.evaluation.datasets import GENERATORS, SyntheticDataset
from app.services.dataset_analysis.ingestion import parse_csv
from app.services.dataset_analysis.pipeline import analyze_table, resolve_target_hint

SETTINGS = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing")


def run(dataset: SyntheticDataset | str, tmp_path, settings: Settings = SETTINGS, **kw):
    if isinstance(dataset, str):
        dataset = GENERATORS[dataset]()
    path = tmp_path / f"{dataset.name}.csv"
    path.write_bytes(dataset.csv_bytes())
    table = parse_csv(str(path), path.name, os.path.getsize(path), settings)
    return analyze_table(table, settings, **kw)


def target(result, column):
    return next((t for t in result.targets if t.column == column), None)


def task_names(result):
    return {(t.task, t.target) for t in result.tasks}


def findings(result, code):
    return result.quality.by_code(code)


class TestTargetCandidates:
    @pytest.mark.parametrize(
        ("dataset", "column", "kind"),
        [
            ("student_performance", "final_grade", TargetKind.CONTINUOUS),
            ("customer_churn", "churn", TargetKind.BINARY),
            ("house_prices", "price", TargetKind.CONTINUOUS),
            ("diabetes", "outcome", TargetKind.BINARY),
            ("bank_marketing", "y", TargetKind.BINARY),
            ("fraud_transactions", "is_fraud", TargetKind.BINARY),
        ],
    )
    def test_the_obvious_target_is_found_first(self, tmp_path, dataset, column, kind):
        result = run(dataset, tmp_path)
        best = result.targets[0]
        assert best.column == column and best.target_kind is kind
        assert best.state in (CandidateState.STRONG_CANDIDATE, CandidateState.POSSIBLE)

    def test_every_candidate_shows_the_case_for_it(self, tmp_path):
        for candidate in run("customer_churn", tmp_path).targets:
            assert candidate.evidence and candidate.why_target

    def test_evidence_is_labelled_by_how_it_is_known(self, tmp_path):
        best = run("customer_churn", tmp_path).targets[0]
        levels = {e.level for e in best.evidence}
        assert EvidenceLevel.DATAPILOT_PARSED in levels and EvidenceLevel.HEURISTIC in levels
        assert not any(e.level is EvidenceLevel.AI_INFERRED for e in best.evidence)

    def test_weak_candidates_come_with_the_case_against(self, tmp_path):
        result = run("student_performance", tmp_path)
        weak = [t for t in result.targets if t.state is CandidateState.WEAK]
        assert weak and all(t.score < 2.5 for t in weak)

    def test_an_identifier_is_never_a_target_candidate(self, tmp_path):
        for name in ("student_performance", "customer_churn", "diabetes", "fraud_transactions"):
            result = run(name, tmp_path)
            identifiers = {c.name for c in result.columns if ColumnFlag.POSSIBLE_IDENTIFIER in c.flags
                           and c.identifier_evidence and c.identifier_evidence[0] >= 0.9}
            assert not identifiers & {t.column for t in result.targets}, name

    def test_a_sensitive_column_is_never_a_target_candidate(self, tmp_path):
        assert target(run("customer_churn", tmp_path), "email") is None

    def test_a_dataset_with_no_plausible_target_says_so(self, tmp_path):
        assert run("sensor_readings", tmp_path).targets == []

    def test_scores_map_to_states_monotonically(self, tmp_path):
        order = {CandidateState.STRONG_CANDIDATE: 0, CandidateState.POSSIBLE: 1, CandidateState.WEAK: 2}
        result = run("bank_marketing", tmp_path)
        ranked = [t for t in result.targets if not t.user_specified]
        assert [order[t.state] for t in ranked] == sorted(order[t.state] for t in ranked)

    def test_candidates_are_sorted_best_first(self, tmp_path):
        scores = [t.score for t in run("bank_marketing", tmp_path).targets]
        assert scores == sorted(scores, reverse=True)

    def test_a_binary_last_column_without_a_name_cue_is_not_strong(self, tmp_path):
        ds = SyntheticDataset("anon", {
            "f1": [str(i % 7) for i in range(80)],
            "f2": [str((i * 3) % 11) for i in range(80)],
            "zz": ["p" if i % 2 else "q" for i in range(80)],
        })
        best = run(ds, tmp_path).targets
        assert not best or best[0].state is not CandidateState.STRONG_CANDIDATE

    def test_a_time_indexed_quantity_is_a_forecasting_target(self, tmp_path):
        result = run("sales_timeseries", tmp_path)
        assert result.targets[0].column == "units_sold"
        assert any("indexed by the regular date/time column" in e.statement for e in result.targets[0].evidence)

    def test_the_confidence_orders_candidates_and_never_reaches_one_unprompted(self, tmp_path):
        for candidate in run("student_performance", tmp_path).targets:
            assert 0.0 < candidate.confidence < 1.0


class TestUserSpecifiedTarget:
    def test_a_named_column_becomes_a_strong_user_specified_candidate(self, tmp_path):
        result = run("student_performance", tmp_path, target_hint="study_hours")
        first = result.targets[0]
        assert first.column == "study_hours" and first.user_specified
        assert first.state is CandidateState.STRONG_CANDIDATE and first.confidence == 1.0

    def test_the_other_candidates_remain_visible(self, tmp_path):
        result = run("student_performance", tmp_path, target_hint="study_hours")
        assert any(t.column == "final_grade" for t in result.targets)

    @pytest.mark.parametrize("hint", ["Final Grade", "FINAL_GRADE", " final_grade ", "final-grade"])
    def test_the_hint_matches_by_original_or_normalised_name(self, tmp_path, hint):
        result = run("student_performance", tmp_path, target_hint=hint)
        assert result.targets[0].column == "final_grade" and result.targets[0].user_specified

    def test_a_missing_column_is_an_error_not_a_silent_fallback(self, tmp_path):
        with pytest.raises(AnalysisError) as err:
            run("student_performance", tmp_path, target_hint="does_not_exist")
        assert err.value.code is ErrorCode.TARGET_COLUMN_NOT_FOUND and err.value.status_code == 422
        assert "final_grade" in err.value.details["columns_available"]

    def test_a_blank_hint_means_no_hint(self, tmp_path):
        table = parse_csv(
            str(_write(tmp_path, "customer_churn")), "x.csv", 1000, SETTINGS)
        columns = analyze_table(table, SETTINGS, target_hint="  ").columns
        assert resolve_target_hint(columns, "   ") is None and resolve_target_hint(columns, None) is None

    def test_a_hint_for_a_constant_column_is_honoured_but_gets_no_task(self, tmp_path):
        result = run("sales_timeseries", tmp_path, target_hint="store")
        chosen = result.targets[0]
        assert chosen.column == "store" and chosen.user_specified
        assert not any(t.target == "store" for t in result.tasks)


def _write(tmp_path, name):
    path = tmp_path / f"{name}.csv"
    path.write_bytes(GENERATORS[name]().csv_bytes())
    return path


class TestTasks:
    def test_binary_classification(self, tmp_path):
        assert (MLTask.BINARY_CLASSIFICATION, "churn") in task_names(run("customer_churn", tmp_path))

    def test_regression(self, tmp_path):
        assert (MLTask.REGRESSION, "price") in task_names(run("house_prices", tmp_path))

    def test_time_series_forecasting_needs_a_date_axis(self, tmp_path):
        assert (MLTask.TIME_SERIES_FORECASTING, "units_sold") in task_names(run("sales_timeseries", tmp_path))
        assert not any(t.task is MLTask.TIME_SERIES_FORECASTING for t in run("house_prices", tmp_path).tasks)

    def test_multiclass_classification(self, tmp_path):
        ds = SyntheticDataset("species", {
            "sepal": [str(i % 13) for i in range(150)],
            "petal": [str((i * 7) % 17) for i in range(150)],
            "species": ["setosa", "versicolor", "virginica"] * 50,
        })
        assert (MLTask.MULTICLASS_CLASSIFICATION, "species") in task_names(run(ds, tmp_path))

    def test_clustering_and_anomaly_detection_need_no_target(self, tmp_path):
        result = run("sensor_readings", tmp_path)
        names = {t.task for t in result.tasks}
        assert names == {MLTask.CLUSTERING, MLTask.ANOMALY_DETECTION}
        assert all(t.target is None for t in result.tasks)

    def test_no_supervised_task_without_a_target_candidate(self, tmp_path):
        from app.schemas.dataset_analysis import SUPERVISED_TASKS

        for task in run("sensor_readings", tmp_path).tasks:
            assert task.task not in SUPERVISED_TASKS

    def test_every_supervised_task_has_a_target_that_is_a_real_column(self, tmp_path):
        from app.schemas.dataset_analysis import SUPERVISED_TASKS

        for name in GENERATORS:
            result = run(name, tmp_path)
            columns = {c.name for c in result.columns}
            for task in result.tasks:
                if task.task in SUPERVISED_TASKS:
                    assert task.target in columns, (name, task)

    def test_a_rare_positive_class_is_also_framed_as_anomaly_detection(self, tmp_path):
        result = run("fraud_transactions", tmp_path)
        assert (MLTask.ANOMALY_DETECTION, "is_fraud") in task_names(result)

    def test_a_tiny_dataset_downgrades_the_task_and_says_why(self, tmp_path):
        task = next(t for t in run("tiny_table", tmp_path).tasks if t.target == "label")
        # Too few rows to train and validate: named, but never presented as a supported task.
        assert task.state is CandidateState.WEAK
        assert task.confidence <= 0.35
        assert any("only 12 rows" in limit for limit in task.limitations)

    def test_a_dominant_class_is_flagged_as_a_limit_on_the_task(self, tmp_path):
        task = next(t for t in run("fraud_transactions", tmp_path).tasks
                    if t.task is MLTask.BINARY_CLASSIFICATION)
        assert any("minority class" in limit for limit in task.limitations)

    def test_a_class_too_small_to_validate_is_reported(self, tmp_path):
        ds = SyntheticDataset("rare", {
            "x": [str(i % 9) for i in range(200)],
            "y": [str((i * 5) % 13) for i in range(200)],
            "label": ["a"] * 197 + ["b"] * 3,
        })
        task = next(t for t in run(ds, tmp_path).tasks if t.target == "label")
        assert any("smallest class has only 3" in limit for limit in task.limitations)

    def test_multilabel_is_detected_from_prefixed_boolean_columns(self, tmp_path):
        ds = SyntheticDataset("tags", {
            "doc": [str(i) for i in range(60)],
            "genre_action": ["1" if i % 2 else "0" for i in range(60)],
            "genre_comedy": ["1" if i % 3 else "0" for i in range(60)],
            "genre_drama": ["1" if i % 5 else "0" for i in range(60)],
        })
        assert any(t.task is MLTask.MULTILABEL_CLASSIFICATION for t in run(ds, tmp_path).tasks)

    def test_tasks_are_sorted_strongest_first(self, tmp_path):
        order = {CandidateState.STRONG_CANDIDATE: 0, CandidateState.POSSIBLE: 1, CandidateState.WEAK: 2}
        states = [order[t.state] for t in run("fraud_transactions", tmp_path).tasks]
        assert states == sorted(states)


class TestQualityFindings:
    def test_a_messy_file_reports_each_problem_with_numbers(self, tmp_path):
        result = run("messy_table", tmp_path)
        assert findings(result, QualityCode.DUPLICATE_ROWS)[0].evidence["duplicate_rows"] == 10
        assert findings(result, QualityCode.DUPLICATE_COLUMNS)[0].columns == ["value", "value_copy"]
        assert findings(result, QualityCode.CONSTANT_COLUMN)[0].column == "constant"
        assert findings(result, QualityCode.HIGH_MISSINGNESS)[0].column == "mostly_missing"
        assert findings(result, QualityCode.INVALID_NUMERIC_VALUES)[0].column == "mixed"

    def test_every_finding_has_evidence_a_severity_and_a_confidence(self, tmp_path):
        for name in GENERATORS:
            for finding in run(name, tmp_path).quality.findings:
                assert finding.detail and 0.0 <= finding.confidence <= 1.0
                assert finding.evidence_level in (EvidenceLevel.DATAPILOT_PARSED, EvidenceLevel.HEURISTIC)

    def test_high_missingness_states_the_threshold_it_used(self, tmp_path):
        finding = findings(run("messy_table", tmp_path), QualityCode.HIGH_MISSINGNESS)[0]
        assert finding.evidence["threshold_pct"] == 40.0 and "rule-of-thumb" in finding.detail

    def test_moderate_missingness_is_not_called_high(self, tmp_path):
        result = run("diabetes", tmp_path)
        assert [f.column for f in findings(result, QualityCode.MISSING_VALUES)] == ["insulin"]
        assert not findings(result, QualityCode.HIGH_MISSINGNESS)

    def test_a_clean_dataset_has_no_duplicate_or_missing_findings(self, tmp_path):
        result = run("house_prices", tmp_path)
        for code in (QualityCode.DUPLICATE_ROWS, QualityCode.MISSING_VALUES, QualityCode.HIGH_MISSINGNESS,
                     QualityCode.CONSTANT_COLUMN, QualityCode.INVALID_NUMERIC_VALUES):
            assert not findings(result, code), code

    def test_class_imbalance_is_reported_for_a_rare_target(self, tmp_path):
        finding = findings(run("fraud_transactions", tmp_path), QualityCode.CLASS_IMBALANCE)[0]
        assert finding.column == "is_fraud" and finding.evidence["minority_share"] < 0.1
        assert finding.evidence["imbalance_ratio"] > 10

    def test_a_balanced_target_is_not_reported(self, tmp_path):
        assert not findings(run("bank_marketing", tmp_path), QualityCode.CLASS_IMBALANCE)

    def test_a_possible_outlier_states_its_fences(self, tmp_path):
        finding = next(f for f in findings(run("sensor_readings", tmp_path), QualityCode.POSSIBLE_OUTLIER)
                       if f.column == "sensor_a")
        assert finding.evidence["lower_fence"] < finding.evidence["upper_fence"]
        assert finding.evidence["iqr_multiplier"] == 1.5 and "may be errors or genuine" in finding.detail

    def test_identifiers_and_sensitive_fields_are_reported_not_removed(self, tmp_path):
        result = run("customer_churn", tmp_path)
        assert findings(result, QualityCode.POSSIBLE_IDENTIFIER)
        sensitive = findings(result, QualityCode.POSSIBLE_SENSITIVE_FIELD)[0]
        assert sensitive.column == "email" and sensitive.evidence_level is EvidenceLevel.HEURISTIC
        assert "email" in {c.name for c in result.columns}

    def test_a_truncated_file_says_every_figure_is_partial(self, tmp_path):
        small = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing", analysis_max_rows=100)
        result = run("house_prices", tmp_path, small)
        assert findings(result, QualityCode.TRUNCATED)
        assert result.schema.row_count == 100 and "rows analysed" in result.schema.note

    def test_malformed_rows_surface_in_quality(self, tmp_path):
        ds = SyntheticDataset("ragged", {"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
        path = tmp_path / "ragged.csv"
        path.write_bytes(ds.csv_bytes() + b"9\n")
        table = parse_csv(str(path), "ragged.csv", path.stat().st_size, SETTINGS)
        result = analyze_table(table, SETTINGS)
        assert findings(result, QualityCode.MALFORMED_ROWS)

    def test_the_thresholds_used_are_reported(self, tmp_path):
        thresholds = run("house_prices", tmp_path).quality.thresholds
        assert thresholds["leakage_association"] == 0.98 and thresholds["high_missing_ratio"] == 0.4

    def test_findings_are_sorted_most_severe_first(self, tmp_path):
        order = {"critical": 0, "warning": 1, "info": 2}
        severities = [order[f.severity.value] for f in run("messy_table", tmp_path).quality.findings]
        assert severities == sorted(severities)

    def test_overall_missing_percentage_is_exact(self, tmp_path):
        ds = SyntheticDataset("m", {"a": ["1", None, "3", None], "b": ["x", "y", None, "z"]})
        result = run(ds, tmp_path)
        assert result.quality.overall_missing_pct == pytest.approx(37.5)


class TestLeakage:
    def test_a_near_perfect_predictor_is_a_leakage_signal(self, tmp_path):
        result = run("student_performance_leaky", tmp_path)
        leak = findings(result, QualityCode.LEAKAGE_SIGNAL)
        assert [f.column for f in leak] == ["final_exam_score"]
        assert leak[0].severity.value == "critical" and leak[0].evidence_level is EvidenceLevel.DATAPILOT_PARSED
        assert leak[0].evidence["association"] >= 0.98 and leak[0].evidence["target"] == "final_grade"

    def test_the_wording_is_a_check_not_a_verdict(self, tmp_path):
        leak = findings(run("student_performance_leaky", tmp_path), QualityCode.LEAKAGE_SIGNAL)[0]
        assert "Check whether" in leak.detail and "predicts" in leak.detail

    def test_legitimate_strong_features_are_not_flagged(self, tmp_path):
        assert not findings(run("student_performance", tmp_path), QualityCode.LEAKAGE_SIGNAL)
        assert not findings(run("diabetes", tmp_path), QualityCode.LEAKAGE_SIGNAL)

    def test_an_identifier_is_not_reported_as_leakage(self, tmp_path):
        result = run("student_performance", tmp_path)
        assert "student_id" not in {f.column for f in findings(result, QualityCode.LEAKAGE_SIGNAL)}

    def test_a_categorical_feature_that_determines_the_class_is_a_leakage_signal(self, tmp_path):
        ds = SyntheticDataset("cat_leak", {
            "age": [str(20 + i % 40) for i in range(300)],
            "refund_status": ["refunded" if i % 4 == 0 else "none" for i in range(300)],
            "cancelled": ["yes" if i % 4 == 0 else "no" for i in range(300)],
        })
        leaks = findings(run(ds, tmp_path), QualityCode.LEAKAGE_SIGNAL)
        assert leaks and leaks[0].column == "refund_status"

    def test_a_suspicious_name_with_strong_association_is_only_potential(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(3)
        labels = ["a" if v else "b" for v in rng.random(400) < 0.5]
        noise = rng.integers(0, 10, 400)
        ds = SyntheticDataset("potential", {
            "x": [str(v) for v in rng.integers(0, 11, 400)],
            # overlapping ranges (a: 0-9, b: 5-14): informative about the label, not decisive
            "outcome_score": [str(int(n) + (5 if lab == "b" else 0)) for n, lab in zip(noise, labels)],
            "label": labels,
        })
        result = run(ds, tmp_path)
        assert not [f for f in findings(result, QualityCode.LEAKAGE_SIGNAL) if f.column == "outcome_score"]
        potential = [f for f in findings(result, QualityCode.POTENTIAL_LEAKAGE) if f.column == "outcome_score"]
        assert potential and potential[0].confidence == 0.5 and potential[0].evidence_level is EvidenceLevel.HEURISTIC
        assert "hint, not a finding" in potential[0].detail


class TestPipeline:
    def test_the_schema_is_datapilot_parsed_with_originals_kept(self, tmp_path):
        result = run("customer_churn", tmp_path)
        assert result.schema.level is EvidenceLevel.DATAPILOT_PARSED
        assert result.schema.row_count == 600 and result.schema.row_count_evidence is EvidenceLevel.DATAPILOT_PARSED
        names = [c.normalized_name for c in result.schema.columns]
        assert names[0] == "customer_id" and "churn" in names

    def test_the_analysis_is_deterministic(self, tmp_path):
        def snapshot():
            r = run("student_performance_leaky", tmp_path)
            return json.dumps(
                [[t.model_dump(mode="json") for t in r.targets], [t.model_dump(mode="json") for t in r.tasks],
                 r.quality.model_dump(mode="json")], sort_keys=True)
        assert snapshot() == snapshot()

    def test_withheld_values_never_appear_in_the_output(self, tmp_path):
        result = run("customer_churn", tmp_path)
        dumped = json.dumps(
            [c.profile.model_dump(mode="json") for c in result.columns]
            + [f.model_dump(mode="json") for f in result.quality.findings]
        )
        assert "@example.com" not in dumped and "user17@" not in dumped

    def test_a_large_table_is_analysed_on_a_seeded_sample_reproducibly(self, tmp_path):
        small = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing", analysis_sample_rows=200)
        a, b = run("fraud_transactions", tmp_path, small), run("fraud_transactions", tmp_path, small)
        assert [t.score for t in a.targets] == [t.score for t in b.targets]

    def test_the_time_budget_is_enforced(self, tmp_path):
        from app.services.dataset_analysis.errors import AnalysisTimeout
        from app.services.dataset_analysis.safety import Deadline

        ticks = iter([0.0] + [99.0] * 1000)
        deadline = Deadline(1.0, clock=lambda: next(ticks))
        with pytest.raises(AnalysisTimeout):
            run("house_prices", tmp_path, deadline=deadline)


class TestLeakageIsNotAStrongPredictor:
    """A feature that separates the classes well is a good predictor, not necessarily a leak."""

    def test_a_correlation_ratio_alone_is_not_enough(self, tmp_path):
        from app.services.dataset_analysis.evaluation.datasets import HELDOUT_GENERATORS

        result = run(HELDOUT_GENERATORS["iris_like"](), tmp_path)
        assert findings(result, QualityCode.LEAKAGE_SIGNAL) == []  # petal length: ratio 0.985, rule 96.7%

    def test_a_feature_that_is_the_target_in_disguise_is_still_flagged(self, tmp_path):
        ds = SyntheticDataset("copycat", {
            "x": [str((i * 7) % 11) for i in range(300)],
            "level": [str(i % 3 * 10 + (i % 7) * 0.01) for i in range(300)],
            "label": ["low", "mid", "high"] * 100,
        })
        leak = findings(run(ds, tmp_path), QualityCode.LEAKAGE_SIGNAL)
        assert [f.column for f in leak] == ["level"]
        assert leak[0].evidence["single_feature_accuracy"] >= 0.98
