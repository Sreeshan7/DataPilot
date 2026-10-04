"""Unit tests for the evaluation metrics.

These test the metric functions against hand-computed values. If a metric cannot be
verified against a worked example, it cannot be trusted in a report - and a
plausible-looking but wrong NDCG is the kind of defect that survives for months
because the numbers it produces still look reasonable.
"""

from __future__ import annotations

import math

import pytest

from app.services.datasets.evaluation.labels import RelevanceGrade
from app.services.datasets.evaluation.metrics import (
    aggregate_metrics,
    case_metrics,
    dcg,
    highly_relevant_reciprocal_rank,
    incompatible_top_rate,
    metric_deltas,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    top_k_hit_rate,
)


class TestPrecisionAtK:
    def test_perfect_precision_at_one(self):
        assert precision_at_k([3, 2, 1, 0], 1) == 1.0

    def test_grade_one_does_not_count_as_relevant(self):
        # Grade 1 is "weak", below the grade >= 2 cut, so P@1 is 0.
        assert precision_at_k([1, 3], 1) == 0.0

    def test_two_of_three(self):
        assert precision_at_k([3, 0, 2, -1], 3) == pytest.approx(2 / 3)

    def test_short_list_is_padded_not_credited(self):
        # A two-item list at k=5 is 2/5, not 1.0.
        assert precision_at_k([3, 3], 5) == pytest.approx(0.4)

    def test_empty_list(self):
        assert precision_at_k([], 3) == 0.0

    def test_zero_k(self):
        assert precision_at_k([3], 0) == 0.0

    def test_threshold_is_configurable(self):
        assert precision_at_k([1, 1], 2, threshold=1) == 1.0

    def test_incompatible_never_counts_as_relevant(self):
        assert precision_at_k([-1, -1, -1], 3) == 0.0


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k([3, 2, 1], total_relevant=2, k=3) == 1.0

    def test_partial_recall(self):
        # 1 of 4 relevant found in the top 2.
        assert recall_at_k([3, 0, 0, 2], total_relevant=4, k=2) == 0.25

    def test_recall_exceeds_precision_by_design(self):
        # Grades 3 and 2 are relevant, grade 1 is not. In a five-item pool that is
        # P@5 = 2/5 but R@5 = 2/2 = 1.0.
        grades = [3, 2, 1, 0, -1]
        assert precision_at_k(grades, 5) == pytest.approx(0.4)
        assert recall_at_k(grades, total_relevant=2, k=5) == 1.0

    def test_no_relevant_in_pool_is_zero_not_error(self):
        assert recall_at_k([0, 0], total_relevant=0, k=2) == 0.0

    def test_negative_total_is_zero(self):
        assert recall_at_k([3], total_relevant=-1, k=1) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([3, 0, 2]) == 1.0

    def test_third_position(self):
        assert reciprocal_rank([0, 1, 3]) == pytest.approx(1 / 3)

    def test_none_relevant(self):
        assert reciprocal_rank([0, 0, -1]) == 0.0

    def test_highly_relevant_variant_ignores_grade_two(self):
        grades = [2, 3]
        assert reciprocal_rank(grades) == 1.0
        assert highly_relevant_reciprocal_rank(grades) == pytest.approx(0.5)


class TestNdcg:
    def test_perfect_ordering_is_one(self):
        grades = [3, 2, 1, 0]
        assert ndcg_at_k(grades, 4) == pytest.approx(1.0)

    def test_incompatible_at_top_reduces_score_below_perfect(self):
        perfect = ndcg_at_k([3, 2, 1, 0], 4)
        # Grade -1 has zero gain (clamped), so it acts like a 0 in the gain
        # function, not like a negative gain.
        assert ndcg_at_k([-1, 2, 1, 0], 4) < perfect
        assert ndcg_at_k([-1, 2, 1, 0], 4) >= 0.0

    def test_reversed_ordering_is_worse_than_sorted(self):
        ascending = ndcg_at_k([0, 1, 2, 3], 4)
        descending = ndcg_at_k([3, 2, 1, 0], 4)
        assert ascending < descending

    def test_all_incompatible_pool_is_zero_not_nan(self):
        assert ndcg_at_k([-1, -1, -1], 3) == 0.0

    def test_empty_pool_is_zero(self):
        assert ndcg_at_k([], 3) == 0.0

    def test_matches_manual_dcg_ratio(self):
        grades = [3, 0, 2, 0]
        expected = dcg([3, 0, 2, 0]) / dcg([3, 2, 0, 0])
        assert ndcg_at_k(grades, 4) == pytest.approx(expected)

    def test_gain_is_exponential_and_clamped_at_zero(self):
        # Standard DCG gain, 2**g - 1: grade 3 gains 7, not 3. The exponential is
        # the point - it is what makes a grade-3 hit worth so much more than a
        # grade-2 hit that the ordering of those two is not a coin flip.
        assert dcg([3]) == pytest.approx(7.0)
        assert dcg([2]) == pytest.approx(3.0)
        # Negative grades must not subtract: they are clamped to a zero gain.
        assert dcg([-1]) == 0.0
        assert dcg([0]) == 0.0

    def test_dcg_uses_log_discount(self):
        # A single relevant hit at rank 1 vs rank 2: 7/1 vs 7/log2(3).
        assert dcg([3, 0]) == pytest.approx(7.0)
        assert dcg([0, 3]) == pytest.approx(7 / math.log2(3))


class TestTopKHitRate:
    def test_hit_at_one(self):
        assert top_k_hit_rate([3, 0, 0], 1) == 1.0

    def test_miss_at_one_but_hit_at_five(self):
        grades = [0, 2, 0, 0, 3]
        assert top_k_hit_rate(grades, 1) == 0.0
        assert top_k_hit_rate(grades, 5) == 1.0

    def test_grade_two_is_not_a_highly_relevant_hit(self):
        assert top_k_hit_rate([2, 2], 2) == 0.0
        assert top_k_hit_rate([2, 2], 2, threshold=RelevanceGrade.RELEVANT) == 1.0


class TestIncompatibleTopRate:
    def test_one_of_three(self):
        assert incompatible_top_rate([-1, 3, 2], k=3) == pytest.approx(1 / 3)

    def test_zero_when_clean(self):
        assert incompatible_top_rate([3, 2, 1], k=3) == 0.0

    def test_grade_zero_is_not_incompatible(self):
        assert incompatible_top_rate([0, 0, 0], k=3) == 0.0

    def test_empty_list(self):
        assert incompatible_top_rate([], k=3) == 0.0


class TestCaseMetrics:
    def test_all_keys_present(self):
        metrics = case_metrics([3, 1, 0], [3, 1, 0])
        for key in (
            "precision@1",
            "precision@3",
            "precision@5",
            "precision@10",
            "recall@5",
            "recall@10",
            "mrr",
            "ndcg@5",
            "ndcg@10",
            "hit_rate@1",
            "hit_rate@3",
            "hit_rate@5",
            "mrr_highly_relevant",
            "pool_size",
            "relevant_in_pool",
        ):
            assert key in metrics, key

    def test_recall_denominator_is_the_whole_pool(self):
        # One relevant in the top 1 out of three relevant overall.
        metrics = case_metrics([3, 0, 0], [3, 2, 2])
        assert metrics["recall@3"] == pytest.approx(1 / 3)

    def test_pool_size_recorded(self):
        assert case_metrics([3], [3, 0, 0, 0])["pool_size"] == 4.0

    def test_relevant_in_pool_counts_grade_two_and_above(self):
        assert case_metrics([0], [3, 2, 1, 0, -1])["relevant_in_pool"] == 2.0


class TestAggregate:
    def test_macro_average(self):
        aggregate = aggregate_metrics(
            {
                "a": {"ndcg@5": 1.0, "pool_size": 2.0},
                "b": {"ndcg@5": 0.0, "pool_size": 10.0},
            }
        )
        assert aggregate["ndcg@5"] == 0.5
        # Macro, not micro: the 10-item pool does not dominate.
        assert aggregate["pool_size"] == 6.0

    def test_empty_input(self):
        assert aggregate_metrics({}) == {}

    def test_missing_key_is_skipped_not_fatal(self):
        aggregate = aggregate_metrics({"a": {"ndcg@5": 1.0}, "b": {}})
        assert aggregate["ndcg@5"] == 1.0


class TestMetricDeltas:
    def test_signed_difference(self):
        deltas = metric_deltas({"ndcg@5": 0.9}, {"ndcg@5": 0.8})
        assert deltas["ndcg@5"] == pytest.approx(-0.1)

    def test_pool_metrics_excluded(self):
        deltas = metric_deltas({"ndcg@5": 1.0, "pool_size": 2.0}, {"ndcg@5": 1.0, "pool_size": 9.0})
        assert "pool_size" not in deltas

    def test_only_shared_keys(self):
        assert metric_deltas({"a": 1.0}, {"b": 1.0}) == {}