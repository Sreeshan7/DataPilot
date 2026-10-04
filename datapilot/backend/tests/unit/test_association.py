"""Association measures against values that can be derived independently.

The cases that matter most are the ones that must return ``None``: a measure that invents a
number for a constant column or a one-row-per-group feature would turn "cannot tell" into a
confident-looking statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.services.dataset_analysis.association import (
    association,
    is_categorical_like,
    sample_indices,
)
from app.services.dataset_analysis.ingestion import ParsedTable
from app.services.dataset_analysis.profiler import profile_table
from app.schemas.dataset_analysis import IngestionInfo

SETTINGS = Settings(gemini_api_key="", kaggle_api_token="", app_env="testing")


def cols(**data):
    n = len(next(iter(data.values())))
    frame = pd.DataFrame({k: [None if v is None else v for v in vals] for k, vals in data.items()}, dtype=object)
    info = IngestionInfo(file_type="csv", display_name="t.csv", size_bytes=1, rows_read=n)
    typed = profile_table(ParsedTable(frame=frame, original_names=list(data), info=info), SETTINGS)
    return {c.name: c for c in typed}, np.arange(n)


def assoc(feature_values, target_values, **kw):
    c, idx = cols(f=[None if v is None else str(v) for v in feature_values],
                  t=[None if v is None else str(v) for v in target_values])
    return association(c["f"], c["t"], idx, **kw)


RNG = np.random.default_rng(0)


class TestSpearman:
    def test_a_monotone_relationship_is_one_even_when_not_linear(self):
        x = list(range(1, 41))
        result = assoc(x, [v**3 for v in x])
        assert result.method == "spearman" and result.value == pytest.approx(1.0)

    def test_a_decreasing_relationship_is_also_one(self):
        x = list(range(1, 41))
        assert assoc(x, [-v for v in x]).value == pytest.approx(1.0)

    def test_independent_columns_are_near_zero(self):
        x = RNG.normal(size=400).round(4).tolist()
        y = RNG.normal(size=400).round(4).tolist()
        assert assoc(x, y).value < 0.15

    def test_a_constant_column_has_no_defined_association(self):
        assert assoc([5] * 30, list(range(30))) is None
        assert assoc(list(range(30)), [5] * 30) is None

    def test_fewer_than_three_pairs_has_no_defined_association(self):
        assert assoc([1, 2], [3, 4]) is None

    def test_missing_values_are_dropped_pairwise(self):
        x = list(range(1, 31)) + [None] * 5
        y = [v * 2 for v in range(1, 31)] + [7] * 5
        result = assoc(x, y)
        assert result.n == 30 and result.value == pytest.approx(1.0)


class TestCorrelationRatio:
    def test_a_categorical_feature_that_fully_determines_a_number(self):
        feature = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
        target = [1.5] * 20 + [5.5] * 20 + [9.5] * 20
        result = assoc(feature, target)
        assert result.method == "correlation_ratio" and result.value == pytest.approx(1.0)

    def test_a_numeric_feature_separating_classes(self):
        feature = list(range(0, 20)) + list(range(100, 120))
        target = ["lo"] * 20 + ["hi"] * 20
        result = assoc(feature, target)
        assert result.method == "correlation_ratio" and result.value > 0.95

    def test_no_difference_between_groups_is_zero(self):
        feature = ["a", "a", "b", "b"] * 15
        target = [1.5, 2.5, 1.5, 2.5] * 15  # both groups hold 1.5 and 2.5: identical means
        assert assoc(feature, target).value == pytest.approx(0.0, abs=1e-9)

    def test_eta_matches_a_hand_computation(self):
        feature = ["a", "a", "b", "b"]
        target = [1.5, 3.5, 5.5, 7.5]
        # group means 2 and 6, grand mean 4: between = 2*(2^2)+2*(2^2)=16, total = 9+1+1+9=20
        assert assoc(feature, target).value == pytest.approx(np.sqrt(16 / 20), rel=1e-4)


class TestCramersV:
    def test_a_perfect_one_to_one_mapping_is_one(self):
        feature = ["x", "y", "z"] * 30
        target = ["p", "q", "r"] * 30
        result = assoc(feature, target)
        assert result.method == "cramers_v" and result.value == pytest.approx(1.0, abs=1e-6)

    def test_independent_categories_are_near_zero(self):
        feature = RNG.choice(["a", "b", "c"], size=600).tolist()
        target = RNG.choice(["u", "v"], size=600).tolist()
        assert assoc(feature, target).value < 0.12

    def test_bias_correction_keeps_small_noisy_tables_from_looking_associated(self):
        feature = RNG.choice(list("abcdefghij"), size=40).tolist()
        target = RNG.choice(["u", "v", "w"], size=40).tolist()
        assert assoc(feature, target).value < 0.3

    def test_a_single_class_has_no_defined_association(self):
        assert assoc(["a", "b"] * 20, ["same"] * 40) is None

    def test_values_stay_within_zero_and_one(self):
        for _ in range(5):
            f = RNG.choice(["a", "b", "c"], size=100).tolist()
            t = RNG.choice(["x", "y"], size=100).tolist()
            assert 0.0 <= assoc(f, t).value <= 1.0


class TestSingleFeatureAccuracy:
    def test_a_feature_that_determines_the_class_scores_perfectly(self):
        feature = ["a"] * 25 + ["b"] * 25
        target = ["yes"] * 25 + ["no"] * 25
        r = assoc(feature, target)
        assert r.single_feature_accuracy == 1.0 and r.majority_baseline == 0.5

    def test_an_uninformative_feature_scores_at_baseline(self):
        feature = ["a", "b"] * 40
        target = ["yes"] * 60 + ["no"] * 20
        r = assoc(feature, target)
        assert r.single_feature_accuracy == pytest.approx(r.majority_baseline, abs=0.1)

    def test_a_numeric_feature_is_binned_so_it_cannot_memorise(self):
        feature = list(range(100))
        target = ["lo"] * 50 + ["hi"] * 50
        r = assoc(feature, target)
        assert r.single_feature_accuracy >= 0.95

    def test_one_row_per_group_gives_no_accuracy_at_all(self):
        feature = [f"id{i}" for i in range(60)]
        target = ["a", "b"] * 30
        r = assoc(feature, target)
        assert r is None or r.single_feature_accuracy is None


class TestEdgeCases:
    def test_a_high_cardinality_feature_is_not_grouped(self):
        feature = [f"v{i}" for i in range(200)]
        target = ["a", "b"] * 100
        assert assoc(feature, target) is None

    def test_free_text_is_never_associated(self):
        text = ["the quick brown fox jumps over the lazy dog number " + str(i) for i in range(40)]
        assert assoc(text, ["a", "b"] * 20) is None

    def test_a_column_is_not_associated_with_itself(self):
        c, idx = cols(f=[str(i) for i in range(20)])
        assert association(c["f"], c["f"], idx) is None

    def test_dates_are_treated_as_a_quantity(self):
        dates = [f"2024-01-{d:02d}" for d in range(1, 29)]
        c, idx = cols(d=dates, y=[str(i * 2) for i in range(28)])
        r = association(c["d"], c["y"], idx)
        assert r is not None and r.value == pytest.approx(1.0)

    def test_an_integer_target_with_few_values_is_treated_as_classes(self):
        c, _ = cols(t=[str(i % 3) for i in range(60)])
        assert is_categorical_like(c["t"], 20)
        c2, _ = cols(t=[str(i) for i in range(60)])
        assert not is_categorical_like(c2["t"], 20)

    def test_the_result_is_deterministic(self):
        f = [str(v) for v in RNG.normal(size=100).round(3)]
        t = [str(v) for v in RNG.normal(size=100).round(3)]
        assert assoc(f, t) == assoc(f, t)


class TestSampling:
    def test_a_small_table_is_used_whole(self):
        assert list(sample_indices(10, 100, 42)) == list(range(10))

    def test_a_large_table_is_sampled_reproducibly_in_order(self):
        a = sample_indices(10_000, 500, 42)
        b = sample_indices(10_000, 500, 42)
        assert len(a) == 500 and np.array_equal(a, b) and list(a) == sorted(a)

    def test_a_different_seed_gives_a_different_sample(self):
        assert not np.array_equal(sample_indices(10_000, 500, 1), sample_indices(10_000, 500, 2))
