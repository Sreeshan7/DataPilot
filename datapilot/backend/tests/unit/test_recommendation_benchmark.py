"""The final recommendation, measured on the graded corpus.

Thresholds are pinned at the values measured when this benchmark was introduced
(after the two defects it exposed were fixed), so a regression in the recommendation
layer fails here instead of passing silently. They describe behaviour on a corpus this
project wrote; they are not real-world accuracy.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.schemas.recommendation import RecommendationOutcome
from app.services.datasets.evaluation.benchmark import load_benchmark
from app.services.datasets.evaluation.recommendation_benchmark import (
    CaseVerdict,
    compare_with_rank_one,
    evaluate_recommendations,
)


@pytest.fixture(scope="module")
def report():
    settings = Settings(
        gemini_api_key="",
        kaggle_api_token="",
        kaggle_enabled=False,
        dataset_discovery_genai_enabled=False,
        app_env="testing",
    )
    return evaluate_recommendations(load_benchmark(), settings)


class TestSafety:
    def test_no_safety_rule_is_violated_against_the_graded_truth(self, report):
        assert report.safety_violations == []

    def test_an_incompatible_dataset_is_never_the_best_match(self, report):
        assert report.metrics["incompatible_best_rate"] == 0.0
        assert not [r for r in report.cases if r.verdict == CaseVerdict.UNSAFE]

    def test_an_ai_suggestion_is_never_recommended(self, report):
        assert report.metrics["recommended_ai_suggestions"] == 0.0

    def test_the_layer_never_picks_something_worse_than_rank_one(self, report):
        assert compare_with_rank_one(report).harmed == []


class TestQuality:
    def test_best_match_is_the_best_available_dataset_almost_always(self, report):
        assert report.metrics["best_match_correct_rate"] >= 0.96

    def test_best_match_is_at_least_relevant_almost_always(self, report):
        assert report.metrics["best_match_acceptable_rate"] >= 0.98

    def test_it_never_abstains_when_a_usable_dataset_exists(self, report):
        assert report.metrics["false_abstention_rate"] == 0.0

    def test_it_is_at_least_as_accurate_as_taking_rank_one(self, report):
        assert report.metrics["best_match_correct_rate"] >= report.metrics["ranked_first_correct_rate"]

    def test_it_rescues_the_known_target_conflict_cases(self, report):
        rescued = compare_with_rank_one(report).rescued
        assert {"health-target-conflict", "finance-target-conflict-approved"} <= set(rescued)

    def test_it_abstains_when_everything_is_incompatible(self, report):
        case = next(r for r in report.cases if r.case_id == "edge-all-incompatible")
        assert case.outcome is RecommendationOutcome.NONE
        assert case.verdict == CaseVerdict.CORRECT_ABSTENTION


class TestRegressionsTheBenchmarkFound:
    """Each was a defect in the finished recommendation layer, invisible to the
    unit tests, that this benchmark exposed."""

    def test_the_best_dataset_is_not_condemned_for_a_wording_gap(self, report):
        for case_id in ("edu-library-demand", "health-adverse-effect-nlp"):
            case = next(r for r in report.cases if r.case_id == case_id)
            assert case.best_grade == 3, case_id

    def test_a_confirmed_irrelevant_dataset_is_not_put_forward_over_a_relevant_suggestion(self, report):
        case = next(r for r in report.cases if r.case_id == "cyber-ai-surfaced-dataset")
        assert case.best_grade == 3
        assert case.outcome is RecommendationOutcome.TENTATIVE


class TestReportIntegrity:
    def test_every_case_is_accounted_for(self, report):
        assert len(report.cases) == 62
        assert sum(report.counts.values()) == 62

    def test_the_run_is_deterministic(self, report):
        settings = Settings(
            gemini_api_key="", kaggle_api_token="", kaggle_enabled=False,
            dataset_discovery_genai_enabled=False, app_env="testing",
        )
        again = evaluate_recommendations(load_benchmark(), settings)
        assert again.model_dump_json() == report.model_dump_json()

    def test_every_verdict_is_one_of_the_named_ones(self, report):
        allowed = {
            CaseVerdict.CORRECT, CaseVerdict.ACCEPTABLE, CaseVerdict.POOR, CaseVerdict.UNSAFE,
            CaseVerdict.CORRECT_ABSTENTION, CaseVerdict.FALSE_ABSTENTION,
        }
        assert {r.verdict for r in report.cases} <= allowed
