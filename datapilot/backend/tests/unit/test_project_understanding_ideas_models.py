"""Dataset -> ML understanding, grounded project ideas and model recommendations.

The rules: a target is only as confident as its evidence; every idea names columns that exist; ideas
that the data cannot support are not generated; model advice cites facts about this dataset and never
promises a score.
"""

from __future__ import annotations

import re

import pytest

from app.schemas.dataset_analysis import (
    AnalysisMode,
    CandidateState,
    EvidenceLevel,
    MLTask,
    QualityCode,
)
from app.schemas.project_intelligence import (
    Difficulty,
    LearningSetting,
    ModelTier,
    ProjectIdea,
    ProjectIntelligence,
    TargetSource,
)
from app.services.project_intelligence.ideas import generate_ideas, validate_ideas
from app.services.project_intelligence.models import NO_PROMISE, recommend_models
from app.services.project_intelligence.shape import shape_of
from app.services.project_intelligence.understanding import understand
from tests.unit.pi_helpers import intelligence_of, outcome_of


def u(name, target=None):
    return understand(outcome_of(name, target).analysis)


class TestUnderstanding:
    def test_a_strong_detected_target_is_confident_and_supervised(self):
        x = u("customer_churn")
        assert x.learning_setting is LearningSetting.SUPERVISED and x.target_is_confident
        assert x.best_target.column == "churn" and x.target_source is TargetSource.DETECTED
        assert x.primary_task.task is MLTask.BINARY_CLASSIFICATION

    def test_a_user_named_target_wins_and_says_so(self):
        x = u("house_prices", target="bedrooms")
        assert x.best_target.column == "bedrooms" and x.target_source is TargetSource.USER_SPECIFIED
        assert x.target_is_confident and "you named" in x.summary

    def test_no_target_means_unsupervised_only(self):
        x = u("sensor_readings")
        assert x.learning_setting is LearningSetting.UNSUPERVISED_ONLY and x.best_target is None
        assert x.primary_task.task in (MLTask.CLUSTERING, MLTask.ANOMALY_DETECTION)
        assert "no column looks like a prediction target" in x.summary

    def test_a_weak_target_is_visible_but_not_called_confident(self):
        x = u("messy_table")
        assert x.best_target is not None and not x.target_is_confident
        assert any("only a weak target" in lim for lim in x.limitations)
        assert x.learning_setting is LearningSetting.UNSUPERVISED_ONLY

    def test_a_regular_time_series_is_forecast_not_merely_regressed(self):
        assert u("sales_timeseries").primary_task.task is MLTask.TIME_SERIES_FORECASTING

    def test_alternatives_are_listed_after_the_best(self):
        x = u("student_performance")
        assert all(t.column != x.best_target.column for t in x.alternative_targets)
        assert len(x.alternative_targets) <= 3

    def test_critical_findings_are_blockers(self):
        x = u("student_performance_leaky")
        assert x.blockers and any("final_exam_score" in b for b in x.blockers)

    def test_a_tiny_dataset_keeps_its_warning(self):
        x = u("tiny_table")
        assert any("only 12 rows" in lim for lim in x.limitations)

    def test_a_supervised_setting_without_a_target_is_rejected_by_the_model(self):
        from app.schemas.project_intelligence import DatasetMLUnderstanding

        with pytest.raises(ValueError, match="needs a target"):
            DatasetMLUnderstanding(learning_setting=LearningSetting.SUPERVISED, summary="x")
        with pytest.raises(ValueError, match="must exist"):
            DatasetMLUnderstanding(learning_setting=LearningSetting.UNKNOWN, target_is_confident=True, summary="x")


ALL_NAMES = ("student_performance", "student_performance_leaky", "customer_churn", "house_prices", "diabetes",
             "sales_timeseries", "sensor_readings", "bank_marketing", "fraud_transactions", "review_text",
             "tiny_table", "messy_table", "iris_like", "employee_attrition", "wide_numeric")


class TestIdeasAreGrounded:
    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_every_idea_names_only_columns_that_exist(self, name):
        pi = intelligence_of(name)
        columns = shape_of(outcome_of(name).analysis).names
        for idea in pi.project_ideas:
            referenced = set(idea.required_features) | set(idea.optional_features) | ({idea.target} - {None})
            assert referenced <= columns, (name, idea.title, referenced - columns)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_every_backticked_name_in_the_text_is_a_real_column(self, name):
        columns = shape_of(outcome_of(name).analysis).names
        for idea in intelligence_of(name).project_ideas:
            text = " ".join([idea.title, idea.problem_statement, idea.why_it_fits, *idea.challenges, *idea.limitations])
            for span in re.findall(r"`([^`]*)`", text):
                assert span in columns, (name, idea.title, span)

    @pytest.mark.parametrize("name", ALL_NAMES)
    def test_between_zero_and_five_ideas_each_complete(self, name):
        ideas = intelligence_of(name).project_ideas
        assert len(ideas) <= 5
        for idea in ideas:
            assert idea.title and idea.problem_statement and idea.why_it_fits
            assert idea.evaluation_metrics and idea.baseline_models
            assert idea.difficulty in Difficulty and idea.limitations

    def test_the_rich_datasets_get_at_least_three_ideas(self):
        for name in ("student_performance", "customer_churn", "diabetes", "iris_like", "employee_attrition"):
            assert len(intelligence_of(name).project_ideas) >= 3, name

    def test_ideas_never_use_a_leaking_column_or_an_identifier(self):
        for idea in intelligence_of("student_performance_leaky").project_ideas:
            assert "final_exam_score" not in idea.required_features + idea.optional_features
            assert "student_id" not in idea.required_features + idea.optional_features
        for idea in intelligence_of("customer_churn").project_ideas:
            assert not ({"customer_id", "email"} & set(idea.required_features + idea.optional_features))

    def test_the_target_is_never_among_its_own_features(self):
        for name in ALL_NAMES:
            for idea in intelligence_of(name).project_ideas:
                assert idea.target not in idea.required_features + idea.optional_features

    def test_the_first_idea_is_the_primary_prediction_task(self):
        first = intelligence_of("customer_churn").project_ideas[0]
        assert first.target == "churn" and first.task is MLTask.BINARY_CLASSIFICATION
        assert first.state is CandidateState.STRONG_CANDIDATE and first.evidence_level is EvidenceLevel.DATAPILOT_PARSED

    def test_the_required_features_are_the_recommended_ones(self):
        pi = intelligence_of("student_performance")
        recommended = {f.column for f in pi.feature_intelligence.features if f.role.value == "RECOMMENDED_FEATURE"}
        assert set(pi.project_ideas[0].required_features) == recommended

    def test_a_text_dataset_gets_a_text_idea_even_without_tabular_features(self):
        titles = [i.title for i in intelligence_of("review_text").project_ideas]
        assert any("text in `review_text`" in t for t in titles)

    def test_unsupervised_ideas_leave_out_duplicates_and_mostly_empty_columns(self):
        for idea in intelligence_of("messy_table").project_ideas:
            used = idea.required_features + idea.optional_features
            assert "value_copy" not in used and "mostly_missing" not in used and "id" not in used

    def test_a_forecasting_dataset_does_not_get_an_unsupportable_segmentation(self):
        assert all(i.task is not MLTask.CLUSTERING for i in intelligence_of("sales_timeseries").project_ideas)

    def test_a_dataset_with_no_usable_columns_gets_no_ideas_and_says_why(self):
        pi = intelligence_of("tiny_table")
        assert pi.project_ideas == []
        assert any("could be grounded" in lim for lim in pi.limitations)

    def test_challenges_reflect_what_was_found_in_the_data(self):
        fraud = intelligence_of("fraud_transactions").project_ideas[0]
        assert any("class imbalance" in c for c in fraud.challenges)
        leaky = intelligence_of("student_performance_leaky").project_ideas[0]
        assert any("final_exam_score" in c and "leak" in c for c in leaky.challenges)

    def test_difficulty_rises_with_the_trouble_in_the_data(self):
        easy = intelligence_of("house_prices").project_ideas[0].difficulty
        hard = intelligence_of("fraud_transactions").project_ideas[0].difficulty
        order = [Difficulty.BEGINNER, Difficulty.INTERMEDIATE, Difficulty.ADVANCED]
        assert order.index(hard) >= order.index(easy)

    def test_no_idea_promises_an_accuracy(self):
        promise = re.compile(r"\d\s*%|guarantee|will achieve|accuracy of", re.I)
        for name in ALL_NAMES:
            for idea in intelligence_of(name).project_ideas:
                blob = " ".join([idea.title, idea.problem_statement, idea.why_it_fits, idea.extension_potential,
                                 *idea.challenges, *idea.limitations])
                assert not promise.search(blob), (name, idea.title)

    def test_ideas_describe_association_never_cause(self):
        for name in ALL_NAMES:
            for idea in intelligence_of(name).project_ideas:
                blob = " ".join([idea.title, idea.problem_statement, idea.why_it_fits]).lower()
                assert not re.search(r"\bcauses?\b|\bdrives?\b", blob)

    def test_validate_ideas_drops_an_idea_that_invents_a_column(self):
        idea = intelligence_of("customer_churn").project_ideas[0]
        forged = idea.model_copy(update={"required_features": idea.required_features + ["invented_column"]})
        assert validate_ideas([idea, forged], shape_of(outcome_of("customer_churn").analysis).names) == [idea]

    def test_the_project_idea_model_rejects_a_target_used_as_a_feature(self):
        with pytest.raises(ValueError, match="own features"):
            ProjectIdea(id="x", title="t", problem_statement="p", task=MLTask.REGRESSION, target="a",
                        required_features=["a"], why_it_fits="w", difficulty=Difficulty.BEGINNER,
                        evidence_level=EvidenceLevel.DATAPILOT_PARSED)

    def test_the_container_refuses_more_than_five_ideas(self):
        pi = intelligence_of("iris_like")
        with pytest.raises(ValueError, match="at most five"):
            ProjectIntelligence(understanding=pi.understanding, feature_intelligence=pi.feature_intelligence,
                                project_ideas=pi.project_ideas * 3)

    def test_ideas_are_deterministic(self):
        outcome = outcome_of("diabetes")
        pi = intelligence_of("diabetes")
        shape = shape_of(outcome.analysis)
        again = generate_ideas(outcome.analysis, pi.understanding, pi.feature_intelligence, shape)
        # the service attaches a research request, a blueprint and the "recommended" flag, and puts the best first project
        # first; every other field of every idea must match the generator's exactly
        added = {"research_request", "blueprint", "recommended"}
        strip = lambda ideas: sorted(({k: v for k, v in i.model_dump().items() if k not in added} for i in ideas), key=lambda d: d["id"])
        assert strip(again) == strip(pi.project_ideas)
        assert all(i.blueprint and i.blueprint.build_steps and i.blueprint.source == "deterministic" for i in pi.project_ideas)
        assert sum(i.recommended for i in pi.project_ideas) <= 1
        assert all(i.research_request and i.research_request["project"]["problem"] for i in pi.project_ideas)
        assert all(i.feasibility is None or i.feasibility.level in ("high", "medium", "low") for i in pi.project_ideas)


class TestModelRecommendations:
    def test_every_task_gets_a_baseline_first(self):
        for name in ("customer_churn", "house_prices", "sales_timeseries", "sensor_readings", "iris_like"):
            models = intelligence_of(name).model_recommendations
            assert models and models[0].tier is ModelTier.BASELINE, name

    def test_classification_includes_a_trivial_baseline_and_a_stronger_model(self):
        names = [m.model for m in intelligence_of("customer_churn").model_recommendations]
        assert any("Majority-class" in n for n in names) and any("Gradient-boosted" in n for n in names)

    def test_imbalance_changes_the_advice(self):
        fraud = intelligence_of("fraud_transactions").model_recommendations
        assert any("class weights" in c or "imbalanced" in c for m in fraud for c in m.cautions + [m.why])
        assert any("threshold" in m.model.lower() for m in fraud)

    def test_a_small_dataset_is_steered_to_simple_models(self):
        small = [m for m in intelligence_of("review_text").model_recommendations if m.tier is ModelTier.STRONGER]
        assert small and "Regularised" in small[0].model

    def test_text_columns_trigger_a_text_model(self):
        names = [m.model for m in intelligence_of("review_text").model_recommendations]
        assert any("TF-IDF" in n for n in names)

    def test_forecasting_demands_chronological_validation(self):
        cautions = " ".join(c for m in intelligence_of("sales_timeseries").model_recommendations for c in m.cautions)
        assert "chronologically" in cautions and "extrapolate" in cautions

    def test_clustering_and_anomaly_advice_exist(self):
        models = [m.model for m in intelligence_of("sensor_readings").model_recommendations]
        assert any("k-means" in m or "Isolation" in m for m in models)

    def test_identifier_columns_do_not_trigger_high_cardinality_advice(self):
        for name in ("student_performance", "customer_churn"):
            assert not any("high-cardinality" in m.model for m in intelligence_of(name).model_recommendations)

    def test_no_recommendation_promises_a_score(self):
        promise = re.compile(r"\d\s*%|guarantee|will achieve|accuracy of", re.I)
        for name in ALL_NAMES:
            for m in intelligence_of(name).model_recommendations:
                assert not promise.search(" ".join([m.model, m.why, *m.cautions, *m.preprocessing])), (name, m.model)

    def test_the_no_promise_statement_is_always_in_the_limitations(self):
        for name in ALL_NAMES:
            assert NO_PROMISE in intelligence_of(name).limitations

    def test_no_task_means_no_recommendations(self):
        x = u("customer_churn").model_copy(update={"primary_task": None})
        assert recommend_models(x, shape_of(outcome_of("customer_churn").analysis)) == []

    def test_a_recommendation_cites_this_datasets_row_count(self):
        why = " ".join(m.why for m in intelligence_of("house_prices").model_recommendations)
        assert "400 rows" in why

    def test_missing_values_are_reflected_in_preprocessing(self):
        diabetes = intelligence_of("diabetes").model_recommendations
        assert any("impute" in step for m in diabetes for step in m.preprocessing)


def test_the_understanding_matches_the_analysis_mode():
    assert outcome_of("customer_churn").analysis.mode is AnalysisMode.FILE_PARSED
    assert QualityCode.CLASS_IMBALANCE.value == "class_imbalance"
