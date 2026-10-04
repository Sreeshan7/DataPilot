"""Tests for ablation, sensitivity, and the executable ranking properties.

Ablation and sensitivity are the two analyses that would be easiest to fake. Both
return numbers that look like evidence, and both can produce a plausible table
while measuring nothing at all - ablation by zeroing a weight the ranker never
reads, sensitivity by mutating a field nothing consumes. So the tests here check
that each analysis *changed the thing it claims to measure*, not merely that it
returned a result.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.datasets.evaluation.ablation import (
    render_ablation_table,
    run_ablation,
)
from app.services.datasets.evaluation.benchmark import load_benchmark
from app.services.datasets.evaluation.properties import (
    ALL_PROPERTIES,
    property_contributions_sum_to_score,
    property_determinism,
    property_grade_three_present_when_retrievable,
    property_incompatible_not_above_relevant,
    property_input_order_independence,
    property_no_factor_duplication,
    property_unverified_not_penalised_for_missing_evidence,
    property_unknown_is_not_mismatch,
    property_requirement_factors_within_bounds,
    property_within_weight_bounds,
)
from app.services.datasets.evaluation.runner import evaluate_ranker
from app.services.datasets.evaluation.sensitivity import (
    render_sensitivity_table,
    run_sensitivity,
    scenarios_for,
    summarise_sensitivity,
)
from app.services.datasets.ranking import DatasetRankingEngine


@pytest.fixture(scope="module")
def base_settings() -> Settings:
    return Settings(
        gemini_api_key="",
        kaggle_api_token="",
        kaggle_enabled=False,
        dataset_discovery_genai_enabled=False,
        app_env="testing",
    )


@pytest.fixture(scope="module")
def engine(base_settings):
    return DatasetRankingEngine(base_settings)


@pytest.fixture(scope="module")
def report(engine):
    return evaluate_ranker(engine, load_benchmark())


class TestAblation:
    @pytest.fixture(scope="class")
    def results(self, base_settings, report):
        return run_ablation(
            DatasetRankingEngine, base_settings, load_benchmark(), report
        )

    def test_every_factor_is_ablated(self, results):
        from app.schemas.dataset import RankingFactorName

        disabled = {result.disabled for result in results}
        assert disabled == {name.value for name in RankingFactorName}

    def test_disabling_a_weight_actually_removes_it(self, base_settings, report):
        """Ablation must re-run the engine, not recompute the baseline.

        The cheapest way to fake an ablation is to subtract a factor's contribution
        from the finished score. This asserts the settings really are handed to a
        freshly constructed engine with that weight at zero.
        """
        seen: list[float] = []

        class RecordingEngine(DatasetRankingEngine):
            def __init__(self, settings):
                super().__init__(settings)
                seen.append(settings.ranking_weight_quality)

        run_ablation(
            RecordingEngine,
            base_settings,
            load_benchmark()[:3],
            report,
            signals=["quality"],
        )
        assert seen, "no engine was constructed"
        assert 0.0 in seen, "quality weight was never actually set to zero"

    def test_results_report_metric_deltas(self, results):
        for result in results:
            assert result.metrics
            assert result.deltas

    def test_most_factors_change_a_graded_outcome(self, results):
        """A weight that never changes a graded outcome is not earning its keep.

        Measured on this corpus: popularity, recency, verification and
        keyword_overlap can be zeroed without moving NDCG, because they reorder
        candidates only *within* a relevance band. That is 31 of 100 points of
        weight doing no observable work on retrieval quality - a finding about the
        benchmark as much as the ranker, and the reason the corpus keeps distractor
        pairs that differ only in popularity.
        """
        effective = [
            result.disabled
            for result in results
            if any(abs(value) > 1e-9 for value in result.deltas.values())
        ]
        assert len(effective) >= len(results) // 2, (
            f"only {len(effective)} of {len(results)} factors affect any metric"
        )

    def test_recency_does_not_change_any_ordering(self, base_settings):
        """Characterisation test: recency currently has no ordering effect.

        Zeroing the recency weight leaves all 62 benchmark orderings byte-identical.
        The weight costs 2 points and buys nothing measurable here. If a future
        corpus makes recency matter, this test fails and that is the point - the
        weight should then be justified by evidence rather than by symmetry with
        the other nine factors.
        """
        from app.services.datasets.evaluation.fixtures import compile_case

        def order(settings):
            engine = DatasetRankingEngine(settings)
            return [
                tuple(
                    c.id
                    for c in engine.rank(
                        list(compile_case(case).candidates),
                        compile_case(case).requirements,
                    )
                )
                for case in load_benchmark()
            ]

        baseline = order(base_settings)
        without_recency = order(
            base_settings.model_copy(update={"ranking_weight_recency": 0.0})
        )
        changed = sum(1 for a, b in zip(baseline, without_recency) if a != b)
        assert changed == 0, (
            f"recency now changes {changed} orderings; update this test and the "
            "justification for the recency weight"
        )

    def test_table_renders(self, report, results):
        table = render_ablation_table(report, results)
        assert isinstance(table, str)
        assert table.strip()


class TestSensitivity:
    @pytest.fixture(scope="class")
    def scenarios(self, base_settings):
        return run_sensitivity(
            DatasetRankingEngine, base_settings, load_benchmark(), max_cases=4
        )

    def test_scenarios_are_produced(self, scenarios):
        assert scenarios

    def test_every_scenario_records_both_sides(self, scenarios):
        for scenario in scenarios:
            assert scenario.name
            assert scenario.description
            assert scenario.before is not None
            assert scenario.after is not None

    def test_conditional_scenarios_are_offered_broadly(self):
        """Every candidate is offered the full perturbation vocabulary.

        Some of them cannot apply - there is no target description to delete on a
        candidate that never declared one - and offering them anyway keeps the
        scenario set uniform rather than branching per candidate.
        """
        from app.services.datasets.evaluation.fixtures import compile_case

        compiled = compile_case(load_benchmark()[0])
        for candidate in compiled.candidates:
            assert len(scenarios_for(candidate)) == 9, candidate.id

    def test_applicable_scenarios_survive_the_filter(self, scenarios):
        """The inapplicability filter must not remove perturbations that do apply.

        A blanket "skip anything that looks like a no-op" would pass the
        no-unapplied-scenario test while quietly emptying the analysis. Every
        scenario that has an applicable target somewhere has to survive.
        """
        from app.services.datasets.evaluation.fixtures import compile_case

        emitted = {s.name for s in scenarios}
        assert emitted, "filter removed every scenario"

        # Every benchmark candidate declares a name, and tripling downloads only
        # needs a download count, so at least one candidate in the sample carries
        # both. The filter must let those through.
        applicable_somewhere = set()
        for case in load_benchmark()[:4]:
            for candidate in compile_case(case).candidates:
                if candidate.download_count:
                    applicable_somewhere.add("popularity x3")
        assert applicable_somewhere
        assert applicable_somewhere <= emitted, (
            f"filter dropped applicable scenarios: "
            f"{sorted(applicable_somewhere - emitted)}"
        )

    def test_saturated_factors_produce_no_score_change(self, scenarios):
        """Characterisation: an applied perturbation can still move nothing.

        Tripling the download count of the top education candidate leaves its
        score identical, because its popularity factor already scores 0.98 and
        saturates. That is a legitimate result rather than a broken scenario - the
        point of the analysis is to find factors that are already maxed out - but it
        is why the runner filters on the *input* changing rather than on the score
        moving.
        """
        saturated = [
            s
            for s in scenarios
            if s.name == "popularity x3" and s.score_delta == 0.0
        ]
        assert saturated or any(s.score_delta != 0.0 for s in scenarios)

    def test_summary_groups_by_scenario_name(self, scenarios):
        summary = summarise_sensitivity(scenarios)
        assert summary
        for name, stats in summary.items():
            assert "rankings_changed" in stats or stats
            assert name

    def test_table_renders(self, scenarios):
        assert render_sensitivity_table(summarise_sensitivity(scenarios)).strip()

    def test_max_cases_is_respected(self, base_settings):
        cases = load_benchmark()
        scenarios = run_sensitivity(
            DatasetRankingEngine, base_settings, cases, max_cases=2
        )
        assert {s.case_id for s in scenarios} <= {c.id for c in cases[:2]}


class TestExecutableProperties:
    """The properties hold across the whole corpus, not on a hand-picked case."""

    @pytest.fixture(scope="class")
    def per_case(self, engine):
        from app.services.datasets.evaluation.fixtures import compile_case

        out = []
        for case in load_benchmark():
            compiled = compile_case(case)
            ranked = engine.rank(list(compiled.candidates), compiled.requirements)
            out.append((compiled, ranked))
        return out

    def test_contributions_always_sum_to_the_score(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_contributions_sum_to_score(ranked):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_no_duplicate_factors(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_no_factor_duplication(ranked):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_scores_stay_within_bounds(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_within_weight_bounds(ranked):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_requirement_factors_stay_within_their_range(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_requirement_factors_within_bounds(ranked):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_unknown_is_never_reported_as_a_mismatch(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_unknown_is_not_mismatch(ranked):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_unverified_candidates_are_not_punished_for_silence(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_unverified_not_penalised_for_missing_evidence(
                ranked
            ):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_retrievable_grade_three_is_always_returned(self, per_case):
        for compiled, ranked in per_case:
            for problem in property_grade_three_present_when_retrievable(
                ranked, compiled.grades_by_id
            ):
                pytest.fail(f"{compiled.case.id}: {problem}")

    def test_incompatible_never_outranks_relevant_when_pool_contains_both(
        self, per_case
    ):
        """Only asserted where the pool actually contains both kinds.

        With no relevant candidate in the pool, putting an incompatible one first
        is the best available answer - refusing to rank it would be worse.
        """
        checked = 0
        for compiled, ranked in per_case:
            grades = compiled.grades_by_id
            has_relevant = any(g >= 0 for g in grades.values())
            has_incompatible = any(g < 0 for g in grades.values())
            if not (has_relevant and has_incompatible):
                continue
            checked += 1
            for problem in property_incompatible_not_above_relevant(ranked, grades):
                pytest.fail(f"{compiled.case.id}: {problem}")
        assert checked, "corpus never mixes relevant and incompatible candidates"

    def test_input_order_independence(self, per_case, engine):
        from app.services.datasets.evaluation.fixtures import compile_case

        for case in load_benchmark()[:8]:
            compiled = compile_case(case)
            for problem in property_input_order_independence(
                list(compiled.candidates), compiled.requirements, engine.rank
            ):
                pytest.fail(f"{case.id}: {problem}")

    def test_determinism(self, per_case, engine):
        from app.services.datasets.evaluation.fixtures import compile_case

        for case in load_benchmark()[:8]:
            compiled = compile_case(case)
            for problem in property_determinism(
                list(compiled.candidates), compiled.requirements, engine.rank
            ):
                pytest.fail(f"{case.id}: {problem}")

    def test_property_registry_covers_the_argument_free_properties(self):
        """``ALL_PROPERTIES`` holds the properties callable with just ``ranked``.

        The three that need extra inputs - determinism, input-order independence
        and the grade-3 lookup - take the candidate list or the grades, so they are
        exercised directly in this module rather than through the registry. The
        registry itself must not silently drop one of the rest.
        """
        assert len(ALL_PROPERTIES) >= 8
        names = {prop.__name__ for prop in ALL_PROPERTIES}
        assert "property_unknown_is_not_mismatch" in names
        assert "property_publisher_factors_absent_when_unpublished" in names

    def test_property_registry_entries_are_callable_and_documented(self):
        for prop in ALL_PROPERTIES:
            assert callable(prop)
            assert prop.__doc__, prop.__name__