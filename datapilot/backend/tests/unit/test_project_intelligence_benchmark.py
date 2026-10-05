"""The Phase 5 benchmark, pinned at the values measured when it was introduced."""

from __future__ import annotations

import pytest

from app.services.dataset_analysis.evaluation import project_benchmark as pb


@pytest.fixture(scope="module")
def report():
    return pb.run_project_benchmark()


class TestSafety:
    def test_no_safety_rule_is_violated(self, report):
        assert report.safety_violations == []

    def test_no_idea_ever_names_a_column_that_does_not_exist(self, report):
        assert report.metrics["invented_columns"] == 0.0

    def test_a_leaking_column_is_never_recommended(self, report):
        assert report.metrics["leaks_recommended"] == 0.0

    def test_a_contradicting_goal_is_never_called_a_match(self, report):
        assert report.metrics["goal_false_match"] == 0.0


class TestQuality:
    def test_every_strong_driver_is_found(self, report):
        assert report.metrics["strong_driver_recall"] == 1.0

    def test_every_planted_leak_and_identifier_is_recognised(self, report):
        assert report.metrics["leak_detection_recall"] == 1.0 and report.metrics["identifier_recall"] == 1.0

    def test_recommendations_are_mostly_real_drivers(self, report):
        assert report.metrics["recommended_precision"] >= 0.9

    def test_pure_noise_columns_are_rarely_recommended(self, report):
        """Two of 118 noise columns on the 120-column dataset, none elsewhere."""
        assert report.metrics["noise_recommended"] <= 3
        assert all(not r.noise_recommended for r in report.datasets if r.name != "wide_numeric")

    def test_goal_compatibility_is_right_on_the_graded_cases(self, report):
        assert report.metrics["goal_accuracy"] == 1.0 and report.metrics["goal_false_mismatch"] == 0.0

    def test_every_rich_dataset_gets_at_least_three_ideas(self, report):
        for r in report.datasets:
            assert r.ideas >= 3, r.name

    def test_the_importance_model_beat_the_baseline_where_signal_exists(self, report):
        assert all(r.reliable for r in report.datasets)


def test_the_truth_names_real_columns():
    from app.services.dataset_analysis.evaluation.datasets import GENERATORS, HELDOUT_GENERATORS

    sources = {**GENERATORS, **HELDOUT_GENERATORS}
    for name, truth in pb.TRUTH.items():
        columns = set(sources[name]().columns)
        assert (truth.drivers | truth.noise | truth.must_find | truth.ambiguous | truth.leaks | truth.identifiers) <= columns, name


def test_a_wrong_expectation_is_caught():
    """The benchmark can fail: a forged leak that is not a leak would be reported as missed."""
    forged = pb.FeatureTruth(drivers=frozenset(), noise=frozenset(), leaks=frozenset({"previous_marks"}))
    original = pb.TRUTH["student_performance"]
    pb.TRUTH["student_performance"] = forged
    try:
        result = pb.run_project_benchmark()
    finally:
        pb.TRUTH["student_performance"] = original
    assert result.metrics["leak_detection_recall"] < 1.0
