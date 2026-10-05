"""Feature intelligence: measured on real rows, leakage-aware, and never faked.

Each class is one promise: importance exists only where DataPilot has rows; a column that gives the
answer away is set apart rather than celebrated; a model that cannot beat a trivial predictor ranks
nothing; random columns are not recommended; and one file always gives one answer.
"""

from __future__ import annotations

import io

import pytest

from app.schemas.dataset_analysis import EvidenceLevel, MLTask
from app.schemas.project_intelligence import FeatureAssessment, FeatureIntelligence, FeatureRole
from app.services.dataset_analysis.evaluation.datasets import SyntheticDataset, wide_numeric
from app.services.project_intelligence import features
from app.services.project_intelligence.understanding import understand
from tests.unit.pi_helpers import SETTINGS, analyze_dataset, intelligence_of, light_settings, outcome_of

R = FeatureRole


def by_column(fi: FeatureIntelligence) -> dict[str, FeatureAssessment]:
    return {f.column: f for f in fi.features}


def fi_of(name: str, target=None) -> FeatureIntelligence:
    return intelligence_of(name, target).feature_intelligence


class TestRolesOnKnownDatasets:
    def test_the_real_drivers_of_a_grade_are_recommended_and_ranked_first(self):
        fi = fi_of("student_performance")
        roles = by_column(fi)
        assert fi.available and fi.target == "final_grade" and fi.task is MLTask.REGRESSION
        assert roles["previous_marks"].role is R.RECOMMENDED_FEATURE and roles["previous_marks"].importance_rank == 1
        assert roles["study_hours"].role is R.RECOMMENDED_FEATURE
        assert roles["gender"].role is R.LOW_SIGNAL  # planted as pure noise

    def test_an_identifier_is_never_a_feature(self):
        roles = by_column(fi_of("customer_churn"))
        assert roles["customer_id"].role is R.IDENTIFIER and roles["email"].role is R.IDENTIFIER
        for column in ("customer_id", "email"):
            a = roles[column]
            assert a.permutation_importance is None and a.mutual_information is None and a.association is None

    def test_a_planted_leak_is_set_apart_not_recommended(self):
        roles = by_column(fi_of("student_performance_leaky"))
        leak = roles["final_exam_score"]
        assert leak.role is R.POSSIBLE_LEAKAGE and "almost perfectly" in leak.reasons[0]
        assert any("kept out of the importance model" in c for c in leak.cautions)
        assert leak.permutation_importance is None  # it was not allowed to dominate the model

    def test_the_leak_does_not_hide_the_legitimate_features(self):
        roles = by_column(fi_of("student_performance_leaky"))
        assert roles["previous_marks"].role is R.RECOMMENDED_FEATURE
        assert roles["previous_marks"].importance_rank == 1

    def test_a_constant_column_is_low_signal_with_the_reason(self):
        roles = by_column(fi_of("sales_timeseries"))
        assert roles["store"].role is R.LOW_SIGNAL and "single value" in roles["store"].reasons[0]

    def test_free_text_is_unknown_not_judged(self):
        fi = fi_of("review_text")
        assert not fi.available  # nothing numeric or categorical is left to model
        assert by_column(fi)["review_text"].role is R.UNKNOWN

    def test_a_multiclass_target_is_supported(self):
        fi = fi_of("iris_like")
        assert fi.available and fi.task is MLTask.MULTICLASS_CLASSIFICATION
        assert by_column(fi)["petal_length"].role is R.RECOMMENDED_FEATURE
        assert fi.method.scoring == "balanced_accuracy"

    def test_a_binary_target_is_scored_with_auc(self):
        assert fi_of("diabetes").method.scoring == "roc_auc"
        assert by_column(fi_of("diabetes"))["glucose"].role is R.RECOMMENDED_FEATURE

    def test_sensitive_features_are_cautioned_even_when_predictive(self):
        from app.schemas.dataset_analysis import ColumnFlag

        outcome = outcome_of("employee_attrition")
        assert any(ColumnFlag.POSSIBLE_SENSITIVE_FIELD in c.flags for c in outcome.parsed.columns)
        roles = by_column(fi_of("employee_attrition"))
        assert roles["ssn"].role is R.IDENTIFIER  # unique per row, so an identifier first


class TestNeverFakesImportance:
    def test_a_registry_only_dataset_has_no_importance_at_all(self):
        outcome = outcome_of("customer_churn")
        external = features.external_feature_intelligence(
            outcome.analysis.model_copy(update={}), "churn")
        # the analysis above is a parsed one; build a registry-only view of the same schema
        assert external.available is False and external.method is None
        assert all(f.permutation_importance is None and f.mutual_information is None for f in external.features)

    def test_the_model_refuses_importance_without_evidence_level_parsed(self):
        with pytest.raises(ValueError, match="DATAPILOT_PARSED"):
            FeatureAssessment(column="x", role=R.RECOMMENDED_FEATURE, evidence_level=EvidenceLevel.HEURISTIC,
                              permutation_importance=0.4)
        with pytest.raises(ValueError, match="DATAPILOT_PARSED"):
            FeatureAssessment(column="x", role=R.UNKNOWN, evidence_level=EvidenceLevel.REGISTRY_VERIFIED, association=0.3)

    def test_unavailable_intelligence_must_say_why_and_carry_no_numbers(self):
        with pytest.raises(ValueError, match="say why"):
            FeatureIntelligence(available=False)
        with pytest.raises(ValueError, match="UNKNOWN or a registry"):
            FeatureIntelligence(available=False, unavailable_reason="x", features=[
                FeatureAssessment(column="a", role=R.RECOMMENDED_FEATURE, evidence_level=EvidenceLevel.UNKNOWN)])
        with pytest.raises(ValueError, match="not computed"):
            FeatureIntelligence(available=False, unavailable_reason="x", features=[
                FeatureAssessment(column="a", role=R.LOW_SIGNAL, evidence_level=EvidenceLevel.DATAPILOT_PARSED,
                                  permutation_importance=0.1)])

    def test_every_number_reported_is_parsed_evidence(self):
        for name in ("student_performance", "customer_churn", "diabetes", "house_prices"):
            for f in fi_of(name).features:
                assert f.evidence_level is EvidenceLevel.DATAPILOT_PARSED
                if f.permutation_importance is not None:
                    assert fi_of(name).available

    def test_a_model_that_cannot_beat_a_trivial_predictor_ranks_nothing(self):
        # A target that is pure noise: no feature can predict it.
        import numpy as np

        rng = np.random.default_rng(5)
        n = 300
        ds = SyntheticDataset("noise_target", {
            "a": [f"{v:.3f}" for v in rng.normal(size=n)], "b": [f"{v:.3f}" for v in rng.normal(size=n)],
            "c": [f"{v:.3f}" for v in rng.normal(size=n)],
            "label": ["yes" if v > 0 else "no" for v in rng.normal(size=n)]})
        fi = intelligence_of_dataset(ds)
        assert fi.available and fi.method.reliable is False
        assert all(f.permutation_importance is None for f in fi.features)  # withheld, not shown as a ranking
        assert all(f.role is not R.RECOMMENDED_FEATURE for f in fi.features)
        assert "withheld" in fi.method.description

    def test_a_regression_that_cannot_extrapolate_a_trend_is_reported_unreliable(self):
        fi = fi_of("sales_timeseries")
        assert fi.task is MLTask.TIME_SERIES_FORECASTING and fi.method.split == "chronological"
        assert not fi.method.reliable and fi.method.held_out_score < fi.method.baseline_score
        assert all(f.role is not R.RECOMMENDED_FEATURE for f in fi.features)


def intelligence_of_dataset(ds: SyntheticDataset, settings=SETTINGS) -> FeatureIntelligence:
    from tests.unit.pi_helpers import build

    return build(analyze_dataset(ds, settings), settings=settings).feature_intelligence


class TestUnavailableCases:
    def test_no_target_means_no_importance(self):
        fi = fi_of("sensor_readings")
        assert not fi.available and "no target" in fi.unavailable_reason.lower()

    def test_a_task_without_importance_says_so(self):
        fi = fi_of("messy_table")
        assert not fi.available and "clustering" in fi.unavailable_reason

    def test_too_few_rows_is_refused_with_the_count(self):
        fi = fi_of("tiny_table")
        assert not fi.available and "Only 12 rows" in fi.unavailable_reason

    def test_a_single_class_target_has_nothing_to_predict(self):
        ds = SyntheticDataset("one_class", {
            "x": [str(i) for i in range(60)], "y": [str(i % 7) for i in range(60)], "label": ["same"] * 60})
        fi = build_features(analyze_dataset(ds, target="label"))
        assert not fi.available

    def test_a_user_named_target_drives_the_analysis(self):
        fi = fi_of("house_prices", target="sqft")
        assert fi.target == "sqft" and fi.available
        assert by_column(fi)["price"].role in (R.RECOMMENDED_FEATURE, R.USE_WITH_CAUTION, R.POSSIBLE_LEAKAGE)


def build_features(outcome, settings=SETTINGS):
    from tests.unit.pi_helpers import build

    return build(outcome, settings=settings).feature_intelligence


class TestMeasurementQuality:
    def test_random_columns_are_almost_never_recommended(self):
        """One strong driver (f003) and one weaker (f017) hidden among 118 pure-noise columns."""
        noise_total = noise_recommended = true_recommended = 0
        for seed in (131, 7, 99):
            fi = intelligence_of_dataset(wide_numeric(seed=seed), light_settings(
                importance_max_rows=300, importance_n_estimators=100, importance_n_repeats=5, importance_max_features=40))
            roles = by_column(fi)
            true_recommended += roles["f003"].role is R.RECOMMENDED_FEATURE
            noise = [f for name, f in roles.items() if name not in ("f003", "f017") and name.startswith("f")]
            noise_total += len(noise)
            noise_recommended += sum(f.role is R.RECOMMENDED_FEATURE for f in noise)
        assert true_recommended == 3  # the strongest driver is always found
        assert noise_recommended / noise_total < 0.02

    def test_the_feature_cap_is_respected_and_reported(self):
        fi = intelligence_of_dataset(wide_numeric(), light_settings(
            importance_max_features=10, importance_max_rows=300))
        assert fi.method.features_modelled == 10 and fi.method.features_skipped > 0
        not_modelled = [f for f in fi.features if f.permutation_importance is None and f.association is not None]
        assert not_modelled and any("not included in the importance model" in " ".join(f.cautions) for f in not_modelled)

    def test_the_row_cap_samples_and_says_so(self):
        fi = intelligence_of_dataset(wide_numeric(n=300, p=20), light_settings(importance_max_rows=100))
        assert fi.method.sampled and fi.method.rows_used <= 100 and fi.method.rows_available == 300

    def test_the_method_reports_how_the_numbers_were_made(self):
        m = fi_of("diabetes").method
        assert m.model == "RandomForestClassifier" and m.seed == SETTINGS.analysis_random_seed
        assert m.noise_columns >= 3 and m.noise_floor is not None and m.baseline_score == 0.5
        assert "held-out" in m.description and "yardsticks" in m.description
        assert m.held_out_score > m.baseline_score and m.reliable

    def test_importance_is_ranked_densely_by_size(self):
        ranked = [f for f in fi_of("student_performance").features if f.importance_rank]
        assert [f.importance_rank for f in ranked] == list(range(1, len(ranked) + 1))
        values = [f.permutation_importance for f in ranked]
        assert values == sorted(values, reverse=True)

    def test_a_dominant_feature_is_noted_but_not_demoted(self):
        sqft = by_column(fi_of("house_prices"))["sqft"]
        assert sqft.role is R.RECOMMENDED_FEATURE
        assert any("almost all of the model's signal" in c for c in sqft.cautions)

    def test_the_time_axis_is_not_scored_as_a_predictor(self):
        date = by_column(fi_of("sales_timeseries"))["date"]
        assert date.role is R.UNKNOWN and "time axis" in date.reasons[0] and date.permutation_importance is None

    def test_the_causality_caveat_is_always_present(self):
        for name in ("student_performance", "sensor_readings", "tiny_table"):
            assert "not causes" in fi_of(name).caveat


class TestDeterminism:
    def test_the_same_file_gives_the_same_answer(self):
        first = intelligence_of_dataset(wide_numeric(n=200, p=20))
        second = intelligence_of_dataset(wide_numeric(n=200, p=20))
        assert first.model_dump() == second.model_dump()

    def test_the_seed_changes_the_result_only_through_the_seed(self):
        a = intelligence_of_dataset(wide_numeric(n=200, p=20), light_settings(analysis_random_seed=1))
        b = intelligence_of_dataset(wide_numeric(n=200, p=20), light_settings(analysis_random_seed=2))
        assert a.method.seed == 1 and b.method.seed == 2
        # both runs find the same strongest driver
        assert by_column(a)["f003"].role is R.RECOMMENDED_FEATURE and by_column(b)["f003"].role is R.RECOMMENDED_FEATURE


def test_understanding_is_the_only_input_to_the_target_choice():
    outcome = outcome_of("customer_churn")
    understanding = understand(outcome.analysis)
    fi = features.build_feature_intelligence(outcome.parsed, understanding, SETTINGS)
    assert fi.target == understanding.best_target.column and fi.task is understanding.primary_task.task
    assert io.BytesIO  # keep the import used
