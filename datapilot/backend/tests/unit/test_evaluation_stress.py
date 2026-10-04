"""Tests for the controlled ranking stress harness.

The harness is measurement code, so these tests do not assert what the ranker
*should* do - that is what the report is for. They assert the things that must be
true for the measurements to mean anything:

- a controlled pair really did move only the factor it claims to move;
- the published contributions actually reconstruct the published score;
- the UNKNOWN/MATCH/MISMATCH distinction survives into the reported status;
- the sweeps are pinned and reproducible.

If the harness silently stopped being controlled, every conclusion drawn from it
would still look perfectly reasonable while being wrong. These are the guards.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from app.core.config import Settings
from app.schemas.dataset import SourceType, VerificationStatus
from app.services.datasets.evaluation import stress as S
from app.services.datasets.evaluation.fixtures import BENCHMARK_NOW


@pytest.fixture(scope="module")
def offline_settings():
    return Settings(
        gemini_api_key="",
        kaggle_api_token="",
        kaggle_enabled=False,
        dataset_discovery_genai_enabled=False,
        app_env="testing",
    )


@pytest.fixture(scope="module")
def engine(offline_settings):
    return S.make_engine(offline_settings)


@pytest.fixture(scope="module")
def pairs(engine):
    return S.controlled_pairs(engine)


class TestControlledPairs:
    def test_every_dimension_is_measured(self, pairs):
        assert {p.dimension for p in pairs} == {
            "popularity", "recency", "verification", "quality",
            "domain_relevance", "target_match", "feature_match", "task_match",
        }

    def test_each_pair_is_single_dimension_or_documented_coupling(self, pairs):
        """A pair that moved an undeclared factor is not evidence of anything."""
        for pair in pairs:
            assert pair.controlled, (
                f"{pair.dimension} moved {pair.moved_factors}, "
                f"which is neither isolated nor a documented coupling"
            )

    def test_only_task_match_carries_a_documented_coupling(self, pairs):
        coupled = {p.dimension for p in pairs if not p.isolated}
        assert coupled == set(S.COUPLED_DIMENSIONS)

    def test_task_mismatch_costs_task_and_compatibility_together(self, pairs):
        task = next(p for p in pairs if p.dimension == "task_match")
        assert "compatibility" in task.moved_factors
        # The two penalties stack, so the observed swing exceeds the nominal weight.
        assert task.contribution_delta == pytest.approx(8.0)
        assert task.score_delta > task.contribution_delta

    def test_bigger_download_count_raises_popularity_and_nothing_else(self, pairs):
        pop = next(p for p in pairs if p.dimension == "popularity")
        assert pop.isolated
        assert pop.factor_score_delta > 0
        assert pop.contribution_delta == pytest.approx(0.58, abs=0.02)

    def test_wrong_domain_scores_partially_rather_than_zero(self, pairs):
        domain = next(p for p in pairs if p.dimension == "domain_relevance")
        assert domain.low.factor("domain_relevance").score == pytest.approx(0.5)
        assert domain.high.factor("domain_relevance").score >= 0.85

    def test_declared_wrong_target_scores_zero_not_unknown(self, pairs):
        target = next(p for p in pairs if p.dimension == "target_match")
        assert target.low.factor("target_match").score == 0.0
        assert target.high.factor("target_match").score == 1.0
        assert target.low.factor("target_match").unmatched_terms == ("churn",)


class TestScoreReconstruction:
    """The published breakdown must add up to the published score."""

    def test_contributions_sum_to_ranking_score(self, engine):
        for fields in ({}, dict(download_count=None, usability=None, updated_at=None)):
            probe = S.rank_all([S.make_candidate("sum", **fields)], engine)[0]
            assert probe.contributions_sum == pytest.approx(probe.score, abs=0.02)

    def test_contributions_sum_to_score_across_every_renormalisation_case(self, engine):
        candidate = S.make_candidate(
            "renorm",
            download_count=None,
            vote_count=None,
            usability=None,
            updated_at=None,
            description=S.TARGET_FREE_DESCRIPTION,
            tags=S.TARGET_FREE_TAGS,
            target_description=None,
        )
        probe = S.rank_all([candidate], engine)[0]
        assert probe.contributions_sum == pytest.approx(probe.score, abs=0.02)

    def test_contributions_are_never_negative(self, pairs):
        for pair in pairs:
            for probe in (pair.low, pair.high):
                for factor in probe.factors.values():
                    assert factor.contribution >= 0.0

    def test_no_factor_exceeds_its_weight_times_normalisation(self, engine):
        candidate = S.make_candidate(
            "cap",
            download_count=None,
            vote_count=None,
            usability=None,
            updated_at=None,
        )
        probe = S.rank_all([candidate], engine)[0]
        ceiling = probe.factor("verification").weight * probe.normalisation
        assert probe.factor("verification").contribution <= ceiling + 0.01


class TestKeywordOverlapIsolation:
    def test_only_keyword_overlap_moves_when_concepts_change(self, engine):
        sweep = S.keyword_project_sweep(engine)
        for earlier, later in zip(sweep, sweep[1:]):
            for name in S.ALL_FEATURE_NAMES:
                if name == "keyword_overlap":
                    continue
                assert earlier.factors[name].score == later.factors[name].score, name

    def test_keyword_points_track_the_overlap_score(self, engine):
        for probe in S.keyword_project_sweep(engine):
            factor = probe.factor("keyword_overlap")
            assert factor.contribution == pytest.approx(factor.score * factor.weight, abs=0.01)


class TestFactorCeilings:
    @pytest.fixture(scope="class")
    def ceilings(self, engine):
        return {row["factor"]: row for row in S.factor_ceilings(engine)}

    def test_popularity_cannot_reach_its_nominal_weight(self, ceilings):
        row = ceilings["popularity (votes free)"]
        assert row["max_points"] <= 5.0
        assert row["total_swing"] < 5.0

    def test_popability_is_bounded_by_log_compression(self, ceilings):
        """A billion-fold difference in downloads cannot produce a large swing."""
        row = ceilings["popularity"]
        assert row["total_swing"] < 3.5

    def test_recency_is_bounded_by_its_weight_and_clips_at_the_horizon(self, ceilings):
        row = ceilings["recency"]
        assert row["max_points"] == pytest.approx(2.0)
        assert row["min_points"] == pytest.approx(0.0)

    def test_verification_has_exactly_two_reachable_values(self, ceilings):
        row = ceilings["verification"]
        assert row["min_points"] == pytest.approx(3.75)
        assert row["max_points"] == pytest.approx(15.0)

    def test_ai_suggested_cannot_reach_full_verification_credit(self, engine):
        """Provenance caps verification even when the status claims otherwise."""
        candidate = S.make_candidate(
            "ai-verified",
            source_type=SourceType.AI_SUGGESTED,
            verification_status=VerificationStatus.VERIFIED,
        )
        probe = S.rank_all([candidate], engine)[0]
        assert probe.factor("verification").score == pytest.approx(0.25)


class TestTippingPoints:
    def test_relevance_ladder_is_strictly_decreasing(self, engine):
        ladder = S.relevance_ladder(engine)
        relevances = [p.relevance for _, p in ladder]
        assert relevances == sorted(relevances, reverse=True)
        assert len(set(relevances)) == len(relevances)

    def test_popularity_cannot_flip_any_relevance_gap_in_the_ladder(self, engine):
        for row in S.popularity_tipping(engine):
            assert row["ratio_needed"] is None, row

    def test_recency_cannot_flip_any_relevance_gap_in_the_ladder(self, engine):
        for row in S.recency_tipping(engine):
            assert row["age_at_flip_days"] is None, row

    def test_smallest_relevance_gap_exceeds_both_publisher_swings(self, engine):
        """The reason no tipping point exists, stated as an inequality."""
        smallest_gap = min(r["score_gap"] for r in S.popularity_tipping(engine))
        ceilings = {r["factor"]: r for r in S.factor_ceilings(engine)}
        assert smallest_gap > ceilings["popularity (votes free)"]["total_swing"]
        assert smallest_gap > ceilings["recency"]["total_swing"]

    def test_verification_survives_every_relevance_rung(self, engine):
        for row in S.verification_tipping(engine):
            assert row["verified_wins"], row

    def test_verification_step_exceeds_its_nominal_step_when_prose_hides_a_miss(self, engine):
        """Losing provenance also lowers judged coverage, so the step grows."""
        steps = {row["rung"]: row["step"] for row in S.verification_tipping(engine)}
        assert steps["full"] == pytest.approx(11.25)
        assert steps["two-of-three"] > steps["full"]


class TestRenormalisation:
    """The denominator is pinned, so silence cannot inflate anything."""

    @pytest.fixture(scope="class")
    def cases(self, engine):
        return {row["case"]: row for row in S.renormalisation_cases(engine)}

    def test_full_evidence_has_no_inflation(self, cases):
        row = cases["A-all-judgeable"]
        assert row["denominator"] == pytest.approx(100.0)
        assert row["normalisation"] == pytest.approx(1.0)

    def test_denominator_is_identical_however_little_evidence_there_is(self, cases):
        """The property the old denominator broke: the mean never shrinks."""
        for row in cases.values():
            assert row["denominator"] == pytest.approx(100.0), row["case"]
            assert row["normalisation"] == pytest.approx(1.0), row["case"]

    def test_absent_publisher_metadata_does_not_shrink_the_denominator(self, cases):
        assert cases["B-publisher-absent"]["denominator"] == pytest.approx(100.0)

    def test_silent_factors_keep_their_weight_but_earn_nothing(self, cases):
        row = cases["B-publisher-absent"]
        for name in ("popularity", "quality", "recency"):
            assert name in row["silent_factors"]
        # verification is paid its nominal weight, not an inflated one
        assert row["verification_points"] == pytest.approx(15.0)

    def test_no_factor_is_paid_more_than_its_nominal_weight(self, cases):
        for row in cases.values():
            assert row["max_inflation"] <= 1.0 + 1e-9, row["case"]

    def test_no_factor_is_paid_more_than_its_nominal_weight_in_the_profile(self, engine):
        for row in S.inflation_profile(engine):
            assert row["multiple_of_weight"] <= 1.0 + 1e-9, row["factor"]

    def test_absent_metadata_ranks_below_complete_metadata(self, cases):
        """The inversion this fix removes: less evidence used to score higher."""
        assert cases["B-publisher-absent"]["score"] < cases["A-all-judgeable"]["score"]

    def test_unknown_target_costs_its_own_points_and_nobody_elses(self, cases):
        row = cases["C-publisher-absent+target-unknown"]
        assert row["target_points"] == pytest.approx(0.0)
        assert row["verification_points"] == pytest.approx(15.0)
        assert row["denominator"] == pytest.approx(100.0)


class TestEvidenceRemovalIsMonotonic:
    """Removing evidence must never raise a score."""

    @pytest.fixture(scope="class")
    def steps(self, engine):
        return S.evidence_removal_monotonicity(engine)

    def test_no_step_ever_increases_the_score(self, steps):
        offenders = [s["step"] for s in steps if s["increased"]]
        assert not offenders, f"removing evidence raised the score at: {offenders}"

    def test_score_is_non_increasing_end_to_end(self, steps):
        scores = [s["score"] for s in steps]
        assert scores == sorted(scores, reverse=True)

    def test_the_denominator_never_moves(self, steps):
        assert {s["denominator"] for s in steps} == {100.0}

    def test_inflation_stays_at_one_throughout(self, steps):
        assert {s["max_inflation"] for s in steps} == {1.0}

    def test_the_first_step_is_the_complete_candidate(self, steps):
        assert steps[0]["score"] == pytest.approx(90.4)
        assert steps[0]["silent_factors"] == "-"

    def test_the_last_step_is_the_silentest_candidate(self, steps):
        assert steps[-1]["score"] < steps[0]["score"]
        assert "task_match" in steps[-1]["silent_factors"]


class TestUnknownMatchMismatch:
    @pytest.fixture(scope="class")
    def rows(self, engine):
        return {r["case"]: r for r in S.unknown_vs_match_vs_mismatch(engine)}

    def test_match_earns_points_and_the_others_earn_none(self, rows):
        assert rows["MATCH"]["target_points"] > 0
        for case in (
            "MISMATCH-resolvable",
            "MISMATCH-unresolved",
            "UNKNOWN-absent",
            "UNKNOWN-declared-absence",
        ):
            assert rows[case]["target_points"] == pytest.approx(0.0)

    def test_mismatch_and_unknown_are_different_reported_states(self, rows):
        assert rows["MISMATCH-resolvable"]["target_score"] == 0.0
        assert rows["UNKNOWN-absent"]["target_score"] is None

    def test_unknown_is_not_reported_as_a_mismatch(self, rows):
        assert "unknown" in rows["UNKNOWN-absent"]["statuses"]
        assert "mismatch" not in rows["UNKNOWN-absent"]["statuses"]

    def test_unknown_never_earns_positive_credit(self, rows):
        for case, row in rows.items():
            if "UNKNOWN" in case:
                assert row["target_score"] is None or row["target_score"] == 0.0
                assert row["target_points"] == pytest.approx(0.0)

    def test_declared_absence_is_read_as_absence_not_contradiction(self, rows):
        assert rows["UNKNOWN-declared-absence"]["statuses"] == "churn=unknown"

    def test_match_outranks_unknown_outranks_mismatch(self, rows):
        """The required ordering, with all surrounding evidence held equal.

        The separation between UNKNOWN and MISMATCH is not produced by the target
        factor, which pays both zero. It comes from the existing contradiction
        channel: a target the concept index resolves to a different label space
        costs compatibility, which silence does not. No factor weight changes to
        achieve it.
        """
        match = rows["MATCH"]["dataset_score"]
        unknown = rows["UNKNOWN-absent"]["dataset_score"]
        mismatch = rows["MISMATCH-resolvable"]["dataset_score"]
        assert match > unknown > mismatch
        assert rows["MISMATCH-resolvable"]["separating_factors"] == "compatibility"
        assert rows["MISMATCH-resolvable"]["compatibility"] < rows["UNKNOWN-absent"]["compatibility"]

    def test_unresolvable_mismatch_is_not_separated_and_is_recorded_as_such(self, rows):
        """The honest limit: a wrong target outside the index ties with silence.

        "the number of support tickets opened" is genuinely not churn, but no
        label space matches it, so the contradiction rule declines to fire and the
        two candidates tie. The case is kept in the report so the boundary is
        visible rather than hidden behind a more convenient example.
        """
        assert rows["MISMATCH-unresolved"]["dataset_score"] == pytest.approx(
            rows["UNKNOWN-absent"]["dataset_score"]
        )
        assert rows["MISMATCH-unresolved"]["separating_factors"] == "none"
        assert rows["MISMATCH-resolvable"]["dataset_score"] < (
            rows["MISMATCH-unresolved"]["dataset_score"]
        )

    def test_keyword_overlap_holds_still_across_all_three_states(self, rows):
        values = {row["keyword_overlap"] for row in rows.values()}
        assert len(values) == 1


class TestKeywordAdversarial:
    @pytest.fixture(scope="class")
    def rows(self, engine):
        return {r["candidate"]: r for r in S.keyword_adversarial(engine)}

    def test_stuffing_reaches_full_keyword_credit(self, rows):
        assert rows["stuffed"]["keyword_overlap"] == pytest.approx(1.0)
        assert rows["honest"]["keyword_overlap"] < 1.0

    def test_stuffing_costs_domain_relevance_completely(self, rows):
        assert rows["stuffed"]["domain_relevance"] == pytest.approx(0.0)
        assert rows["honest"]["domain_relevance"] > 0.0

    def test_stuffing_earns_more_keyword_points_but_still_loses(self, rows):
        assert rows["stuffed"]["keyword_contribution"] > rows["honest"]["keyword_contribution"]
        assert rows["stuffed"]["score"] < rows["honest"]["score"]
        assert rows["honest"]["rank"] == 1


class TestSameRelevanceTieBreaks:
    @pytest.fixture(scope="class")
    def rows(self, engine):
        return S.same_relevance_tiebreaks(engine)

    def test_relevance_is_exactly_equal_in_every_scenario(self, rows):
        for row in rows:
            assert row["relevance_equal"], row
            assert row["relevance_delta"] == pytest.approx(0.0)

    def test_each_scenario_moves_exactly_one_factor(self, rows):
        for row in rows:
            assert len(row["moved_factors"].split(",")) == 1, row

    def test_the_single_moved_factor_decides_the_order(self, rows):
        for row in rows:
            assert row["deciding_factor"] in row["moved_factors"]

    def test_tie_break_strength_follows_the_measured_order(self, rows):
        """verification > recency > quality > popularity, by measured swing."""
        by_factor = {
            r["deciding_factor"]: max(
                abs(x["score_delta"]) for x in rows if x["deciding_factor"] == r["deciding_factor"]
            )
            for r in rows
        }
        assert (
            by_factor["verification"]
            > by_factor["recency"]
            > by_factor["quality"]
            > by_factor["popularity"]
        )

    def test_recency_tie_break_never_exceeds_two_points(self, rows):
        recency = [r for r in rows if r["deciding_factor"] == "recency"]
        assert max(abs(r["score_delta"]) for r in recency) <= 2.0


class TestDeterminism:
    def test_two_runs_agree_exactly(self, offline_settings):
        first = S.rank_all([S.make_candidate("det")], S.make_engine(offline_settings))[0]
        second = S.rank_all([S.make_candidate("det")], S.make_engine(offline_settings))[0]
        assert first.score == second.score
        assert {n: f.contribution for n, f in first.factors.items()} == {
            n: f.contribution for n, f in second.factors.items()
        }

    def test_clock_is_pinned_to_the_benchmark_instant(self, offline_settings):
        engine = S.make_engine(offline_settings)
        probe = S.rank_all([S.make_candidate("clock", updated_at=BENCHMARK_NOW)], engine)[0]
        assert probe.factor("recency").score == pytest.approx(1.0)

    def test_an_absent_date_is_absent_rather_than_defaulted(self, offline_settings):
        engine = S.make_engine(offline_settings)
        absent = S.rank_all([S.make_candidate("gone", updated_at=None)], engine)[0]
        dated = S.rank_all(
            [S.make_candidate("here", updated_at=BENCHMARK_NOW - timedelta(days=90))], engine
        )[0]
        assert absent.factor("recency").score is None
        assert dated.factor("recency").score is not None

    def test_sweeps_are_stable_across_engines(self, offline_settings):
        first = S.relevance_ladder(S.make_engine(offline_settings))
        second = S.relevance_ladder(S.make_engine(offline_settings))
        assert [p.score for _, p in first] == [p.score for _, p in second]


class TestHarnessDoesNotMutateTheProduct:
    WEIGHT_FIELDS = (
        "ranking_weight_domain_relevance",
        "ranking_weight_keyword_overlap",
        "ranking_weight_feature_match",
        "ranking_weight_target_match",
        "ranking_weight_task_match",
        "ranking_weight_compatibility",
        "ranking_weight_quality",
        "ranking_weight_popularity",
        "ranking_weight_recency",
        "ranking_weight_verification",
    )

    def test_stress_module_declares_no_weights_of_its_own(self):
        """Weights belong to config; the harness must not carry a second set."""
        import inspect

        for line in inspect.getsource(S).splitlines():
            stripped = line.strip()
            assert not stripped.startswith("RANKING_WEIGHTS")
            assert "ranking_weight_" not in stripped

    def test_engine_weights_are_unchanged_after_a_full_sweep(self, offline_settings):
        engine = S.make_engine(offline_settings)
        before = {name: getattr(offline_settings, name) for name in self.WEIGHT_FIELDS}
        S.controlled_pairs(engine)
        S.renormalisation_cases(engine)
        S.factor_ceilings(engine)
        S.unknown_vs_match_vs_mismatch(engine)
        assert {name: getattr(offline_settings, name) for name in self.WEIGHT_FIELDS} == before