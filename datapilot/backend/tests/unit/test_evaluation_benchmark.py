"""Tests for the benchmark corpus and the golden regression set.

The corpus is the foundation every other metric rests on. A benchmark with a
mislabelled grade does not fail loudly - it quietly shifts NDCG and produces a
failure taxonomy that blames the ranker for the benchmark author's mistake. So
these tests assert structural invariants over the whole corpus rather than
spot-checking a few cases.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.schemas.dataset import SourceType
from app.services.datasets.evaluation.benchmark import (
    BENCHMARK_VERSION,
    load_benchmark,
)
from app.services.datasets.evaluation.fixtures import BENCHMARK_NOW, compile_case
from app.services.datasets.evaluation.golden_set import (
    GOLDEN_CASES,
    KNOWN_FAILURES,
    check_golden_case,
    known_failure_for,
    load_golden_cases,
    run_golden_set,
)
from app.services.datasets.evaluation.labels import RelevanceGrade
from app.services.datasets.ranking import DatasetRankingEngine


@pytest.fixture(scope="module")
def cases():
    return load_benchmark()


@pytest.fixture(scope="module")
def offline_settings():
    return Settings(
        gemini_api_key="",
        kaggle_api_token="",
        kaggle_enabled=False,
        dataset_discovery_genai_enabled=False,
        app_env="testing",
    )


class TestBenchmarkCorpus:
    def test_corpus_is_not_trivially_small(self, cases):
        assert len(cases) >= 50

    def test_case_ids_are_unique(self, cases):
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))

    def test_every_case_has_the_required_fields(self, cases):
        for case in cases:
            assert case.id, case
            assert case.description, case.id
            assert case.requirements, case.id

    def test_every_candidate_is_graded(self, cases):
        for case in cases:
            for spec in case.candidates:
                assert spec.key in case.grades, f"{case.id}/{spec.key}"

    def test_grades_use_the_five_declared_levels(self, cases):
        allowed = {int(g) for g in RelevanceGrade}
        for case in cases:
            for key, grade in case.grades.items():
                assert grade in allowed, f"{case.id}/{key}={grade}"

    def test_every_case_has_something_to_find(self, cases):
        # A case where nothing is relevant still has value - it proves the ranker
        # does not invent relevance - but it must be a deliberate probe, not an
        # accident of grading.
        for case in cases:
            if any(g >= 1 for g in case.grades.values()):
                continue
            assert set(case.grades.values()) == {-1}, (
                f"{case.id} has no relevant candidate but is not an "
                "all-incompatible probe"
            )

    def test_all_incompatible_probe_exists(self, cases):
        all_bad = [c for c in cases if set(c.grades.values()) == {-1}]
        assert all_bad, "corpus no longer covers the nothing-relevant case"

    def test_most_cases_have_a_highly_relevant_candidate(self, cases):
        with_grade3 = [c for c in cases if any(g == 3 for g in c.grades.values())]
        assert len(with_grade3) / len(cases) > 0.8

    def test_most_cases_are_retrieval_complete(self, cases):
        # `retrieval_complete=False` means the corpus deliberately models a
        # partial pool. That is the point of such a case, but it must stay rare,
        # because an incomplete pool lowers recall for reasons that have nothing
        # to do with the ranker.
        incomplete = [c for c in cases if not c.retrieval_complete]
        assert len(incomplete) / len(cases) < 0.1

    def test_candidate_keys_are_unique_within_a_case(self, cases):
        for case in cases:
            keys = [s.key for s in case.candidates]
            assert len(keys) == len(set(keys)), case.id

    def test_ids_are_traceable_and_not_fabricated(self, cases):
        # The corpus is offline fixtures, not a snapshot of Kaggle. Ids and URLs
        # have to make that obvious to a reader without opening the source,
        # otherwise someone will later cite them as real dataset references.
        for case in cases:
            compiled = compile_case(case)
            for spec, candidate in zip(case.candidates, compiled.candidates):
                assert candidate.id == f"benchmark:{case.id}/{spec.key}"
                assert candidate.source == "kaggle-fixture"
                assert candidate.url and ".invalid/" in candidate.url

    def test_required_categories_are_present(self, cases):
        covered = {c.category for c in cases}
        for expected in (
            "education",
            "healthcare",
            "cybersecurity",
            "finance",
            "retail",
            "nlp",
            "computer_vision",
            "time_series",
            "edge_case",
        ):
            assert expected in covered, f"missing benchmark category: {expected}"

    def test_clock_is_fixed(self):
        # Metrics like recency are a function of the evaluation date; letting it
        # float would make two runs of the same code disagree.
        assert BENCHMARK_NOW.year == 2026
        assert BENCHMARK_NOW.tzinfo is not None

    def test_version_is_declared(self):
        assert BENCHMARK_VERSION


class TestCompilation:
    def test_compile_is_repeatable(self, cases):
        case = cases[0]
        first = compile_case(case)
        second = compile_case(case)
        assert [c.id for c in first.candidates] == [c.id for c in second.candidates]

    def test_compile_does_not_mutate_the_spec(self, cases):
        case = cases[0]
        before = [
            (s.key, s.name, s.description, s.target_description, s.domain)
            for s in case.candidates
        ]
        compile_case(case)
        after = [
            (s.key, s.name, s.description, s.target_description, s.domain)
            for s in case.candidates
        ]
        assert before == after

    def test_candidate_ids_follow_the_generated_pattern(self, cases):
        case = cases[0]
        compiled = compile_case(case)
        assert [c.id for c in compiled.candidates] == [
            f"benchmark:{case.id}/{s.key}" for s in case.candidates
        ]

    def test_ai_rationale_is_never_candidate_published_evidence(self, cases):
        """The corpus must not leak AI prose into the evidence the ranker reads.

        An AI-suggested candidate's rationale is exactly the text the ranker is
        supposed to distrust. If the description also carries the tempting target,
        the case stops testing AI-evidence handling and starts testing the
        description - and the trap silently stops being a trap.
        """
        checked = 0
        for case in cases:
            for spec in case.candidates:
                if spec.source_type is not SourceType.AI_SUGGESTED:
                    continue
                checked += 1
                # An AI suggestion with no prose at all is normal in production,
                # so it is not a corpus defect. What must hold is that the cases
                # built specifically to test rationale handling do carry prose.
                if "rationale" in spec.key or "inferred" in case.id:
                    assert spec.rationale, (
                        f"{spec.key}: rationale-handling case has no rationale"
                    )
                    published = " ".join(
                        filter(
                            None,
                            [
                                spec.description,
                                spec.feature_description,
                                spec.target_description,
                                " ".join(spec.tags),
                            ],
                        )
                    ).lower()
                    # The trap works by the rationale claiming something the
                    # published metadata does not. If the rationale's distinctive
                    # content is already published, the trap is disarmed.
                    distinctive = [
                        token
                        for token in spec.rationale.lower().replace(",", " ").split()
                        if len(token) > 6 and token not in published
                    ]
                    assert distinctive, (
                        f"{spec.key}: rationale is fully contained in the published "
                        "metadata, so it no longer tests rationale distrust"
                    )
        assert checked >= 4, "corpus lost its AI-suggested cases"

    def test_ai_rationale_cases_exist(self, cases):
        traps = [
            spec
            for case in cases
            for spec in case.candidates
            if spec.source_type is SourceType.AI_SUGGESTED and spec.rationale
        ]
        assert len(traps) >= 3

    def test_ai_flagged_candidates_carry_a_provenance_hint(self, cases):
        for case in cases:
            for spec in case.candidates:
                if spec.source_type is SourceType.AI_SUGGESTED:
                    assert spec.possible_source, spec.key

    def test_non_ai_candidates_are_not_marked_as_suggestions(self, cases):
        for case in cases:
            for spec in case.candidates:
                if spec.source_type is not SourceType.AI_SUGGESTED:
                    assert not spec.rationale, (
                        f"{spec.key}: a verified candidate must not carry a "
                        "model rationale"
                    )


class TestGoldenSet:
    def test_golden_set_loads(self):
        loaded = load_golden_cases()
        assert len(loaded) >= 10

    def test_every_golden_case_is_in_the_benchmark(self):
        benchmark_ids = {c.id for c in load_benchmark()}
        for case in load_golden_cases():
            assert case.id in benchmark_ids, case.id

    def test_every_expectation_names_a_real_candidate(self):
        by_case = {c.id: c for c in load_benchmark()}
        for golden in GOLDEN_CASES:
            case = by_case[golden.case_id]
            keys = {s.key for s in case.candidates}
            if golden.expect_top_key is not None:
                assert golden.expect_top_key in keys, golden.case_id
            for key in golden.forbidden_keys:
                assert key in keys, f"{golden.case_id}/{key}"

    def test_expectations_and_forbidden_keys_never_overlap(self):
        for golden in GOLDEN_CASES:
            if golden.expect_top_key is not None:
                assert golden.expect_top_key not in golden.forbidden_keys, golden.case_id

    def test_required_scenarios_are_covered(self):
        # The mission named these explicitly; a golden set that quietly drops one
        # of them is worse than one that never had it.
        ids = {g.case_id for g in GOLDEN_CASES}
        for required in (
            "edu-student-performance",
            "cyber-iot-intrusion",
            "health-diabetes-bmi",
            "finance-churn",
            "health-target-conflict",
            "health-readmission-sparse",
            "nlp-multilingual-sentiment",
            "retail-demand-forecast",
        ):
            assert required in ids, f"golden set missing {required}"

    def test_every_expectation_documents_itself(self):
        for golden in GOLDEN_CASES:
            assert golden.why, golden.case_id
            if golden.forbidden_keys:
                assert golden.forbidden_top_n >= 1, golden.case_id

    def test_every_case_declares_at_least_one_expectation(self):
        for golden in GOLDEN_CASES:
            assert golden.expect_top_key is not None or golden.forbidden_keys, golden.case_id

    def test_known_failures_reference_real_golden_cases(self):
        golden_ids = {g.case_id for g in GOLDEN_CASES}
        for failure in KNOWN_FAILURES:
            assert failure.case_id in golden_ids, failure.case_id
            assert failure.description, failure.case_id
            assert failure.mechanism, failure.case_id
            assert failure.fix_requires, failure.case_id

    def test_known_failure_lookup(self):
        assert known_failure_for("health-target-conflict") is not None
        assert known_failure_for("edu-student-performance") is None


class TestGoldenSuiteBehaviour:
    def test_suite_is_green_on_the_current_ranker(self, offline_settings):
        engine = DatasetRankingEngine(offline_settings)
        assert run_golden_set(engine.rank) == []

    def test_known_failures_are_actually_failing(self, offline_settings):
        """Guards against a known-failure list that has quietly gone stale.

        A known failure that stops failing should be deleted from the list, and
        this test is what makes that a visible event rather than a silent one.
        """
        engine = DatasetRankingEngine(offline_settings)
        strict = run_golden_set(engine.rank, enforce_known_failures=True)
        for failure in KNOWN_FAILURES:
            assert any(failure.case_id in problem for problem in strict), (
                f"known failure {failure.case_id} no longer reproduces; "
                "delete it from KNOWN_FAILURES"
            )

    def test_known_failures_are_suppressed_by_default(self, offline_settings):
        engine = DatasetRankingEngine(offline_settings)
        suppressed = run_golden_set(engine.rank)
        strict = run_golden_set(engine.rank, enforce_known_failures=True)
        assert len(strict) > len(suppressed)

    def test_check_detects_a_violated_expectation(self):
        """The checker itself must fail when it should.

        A golden checker that returns an empty list no matter what is worse than no
        checker, because it reports a green suite forever.
        """
        case = next(c for c in load_golden_cases() if c.id == "edu-student-performance")
        compiled = compile_case(case)
        ranked = sorted(compiled.candidates, key=lambda c: c.name, reverse=True)
        problems = check_golden_case(case, ranked)
        assert problems, "checker missed an obviously wrong ordering"
        assert "edu-student-performance" in problems[0]

    def test_check_reports_empty_ranking(self):
        case = load_golden_cases()[0]
        problems = check_golden_case(case, [])
        assert problems
        assert "no candidates" in problems[0]

    def test_golden_ranking_is_deterministic(self, offline_settings):
        engine = DatasetRankingEngine(offline_settings)
        case = load_golden_cases()[0]
        compiled = compile_case(case)
        first = [c.id for c in engine.rank(list(compiled.candidates), compiled.requirements)]
        second = [c.id for c in engine.rank(list(compiled.candidates), compiled.requirements)]
        assert first == second