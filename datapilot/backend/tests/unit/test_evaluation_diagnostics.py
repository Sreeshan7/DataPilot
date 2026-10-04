"""Tests for diagnostics, the failure taxonomy, and the score audit.

Three different jobs, three different failure modes:

* Diagnostics turns ranked output into per-factor numbers. If it silently drops a
  factor, every downstream report understates that factor's influence.
* The taxonomy assigns a name to a miss. A misclassification is not cosmetic - it
  sends the next round of work at the wrong part of the system.
* The audit checks arithmetic the ranker itself claims is correct. A ranker that
  displays contributions summing to the wrong total is lying in the UI, which is
  worse than being imprecise.

Note on tolerances: the ranker rounds each factor to 2 decimals for display, so
"contributions sum to the score" can only ever hold within a rounding budget. The
tests below use the same documented tolerance the audit uses rather than inventing
a tighter one that would fail for a legitimate reason.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.services.datasets.evaluation.audit import (
    INFLATION_TOLERANCE,
    SUM_TOLERANCE,
    audit_scores,
    dominance_table,
    factor_influence_over_benchmark,
    summarise_findings,
)
from app.services.datasets.evaluation.benchmark import load_benchmark
from app.services.datasets.evaluation.diagnostics import (
    best_available,
    build_factor_views,
    build_ranked_items,
    evidence_quality,
    grade_band,
    render_case_report,
    render_factor_table,
    suspected_unknown_as_mismatch,
)
from app.services.datasets.evaluation.fixtures import compile_case
from app.services.datasets.evaluation.runner import evaluate_ranker
from app.services.datasets.evaluation.taxonomy import FailureCategory, classify_failures
from app.services.datasets.ranking import DatasetRankingEngine


@pytest.fixture(scope="module")
def settings_() -> Settings:
    return Settings(
        gemini_api_key="",
        kaggle_api_token="",
        kaggle_enabled=False,
        dataset_discovery_genai_enabled=False,
        app_env="testing",
    )


@pytest.fixture(scope="module")
def engine(settings_) -> DatasetRankingEngine:
    return DatasetRankingEngine(settings_)


@pytest.fixture(scope="module")
def report(engine):
    return evaluate_ranker(engine, load_benchmark())


@pytest.fixture(scope="module")
def compiled_edu():
    return compile_case(
        next(c for c in load_benchmark() if c.id == "edu-student-performance")
    )


@pytest.fixture(scope="module")
def ranked_edu(engine, compiled_edu):
    return engine.rank(list(compiled_edu.candidates), compiled_edu.requirements)


class TestGradeBands:
    @pytest.mark.parametrize(
        ("grade", "band"),
        [
            (3, "highly relevant"),
            (2, "relevant"),
            (1, "weak"),
            (0, "irrelevant"),
            (-1, "incompatible"),
        ],
    )
    def test_bands_match_the_label_definitions(self, grade, band):
        assert grade_band(grade) == band

    def test_out_of_scale_grade_is_reported_not_guessed(self):
        # A grade outside the declared scale has to announce itself. Folding it
        # into "irrelevant" would quietly change what a report counts as a miss,
        # and raising would abort a whole run over one bad label.
        assert grade_band(99) == "invalid"
        assert grade_band(-7) == "invalid"


class TestFactorViews:
    def test_every_factor_appears_exactly_once(self, ranked_edu):
        for candidate in ranked_edu:
            names = [f.name.value for f in candidate.ranking_factors]
            assert len(names) == len(set(names)), candidate.id
            assert len(names) >= 10, candidate.id

    def test_contributions_sum_to_the_displayed_score(self, ranked_edu):
        for candidate in ranked_edu:
            total = sum(f.contribution for f in candidate.ranking_factors)
            assert total == pytest.approx(
                candidate.ranking_score, abs=SUM_TOLERANCE
            ), candidate.id

    def test_no_factor_can_outweigh_its_nominal_weight(self, report):
        """The denominator is pinned, so no factor can be paid more than it weighs.

        This is the invariant that replaced the old renormalisation. Previously a
        sparse candidate's surviving factors were rescaled by
        ``100 / judged_weight`` and peaked around 1.59x nominal, which meant a
        candidate declaring three confident factors could outrank a fully
        documented one - and a candidate with *no* publisher metadata scored
        93.65 against a complete candidate's 90.40. Any ratio above 1.0 here
        means the denominator is moving again.
        """
        ratios: dict[str, float] = {}
        for evaluation in report.per_case:
            for item in evaluation.ranked:
                for factor in item.factors:
                    if factor.score is None or not factor.weight:
                        continue
                    ratio = factor.contribution / factor.weight
                    ratios[factor.name] = max(ratios.get(factor.name, 0.0), ratio)
        assert ratios, "no factors were measured at all"
        for name, ratio in ratios.items():
            assert ratio <= 1.0 + 1e-6, f"{name} contributed {ratio:.4f}x its weight"

    def test_inflation_audit_reports_no_structural_inflation(self, report, settings_):
        """With the denominator pinned, no factor is ever paid more than it weighs.

        Tested on ``contribution / weight``. The audit used to compare the share of
        the achieved score against ``weight / 100``, which fires on every candidate
        that scores below 100 and so reported all ten factors as inflated even
        once the arithmetic was sound.
        """
        audit = audit_scores(report.per_case, settings_)
        flagged = [e for e in audit.factors if e.observed_max_ratio > 1.0 + INFLATION_TOLERANCE]
        assert not flagged, f"factors still paid above their weight: {[e.name for e in flagged]}"

    def test_scores_track_the_pinned_weighted_mean(self, ranked_edu):
        """The displayed score is the mean over *all* factors, rescaled to 100.

        Silent factors keep their weight in the denominator and contribute
        nothing, so a candidate cannot present a high score on the little it did
        declare. This catches the failure mode that matters most for the audit: a
        candidate that stays quiet about most requirements must not be able to
        present a high score on the little it did declare.
        """
        for candidate in ranked_edu:
            scored = [f for f in candidate.ranking_factors if f.score is not None]
            if not scored:
                assert candidate.ranking_score == 0.0, candidate.id
                continue
            total_weight = sum(f.weight for f in candidate.ranking_factors)
            ceiling = sum(f.weight * (f.score or 0.0) for f in candidate.ranking_factors)
            assert candidate.ranking_score <= ceiling + 0.01, candidate.id
            assert candidate.ranking_score <= 100.0 + 0.01, candidate.id
            assert total_weight == pytest.approx(100.0), candidate.id

    def test_full_evidence_candidates_use_a_unit_scale(self, ranked_edu):
        """With every factor judged, contribution is exactly ``weight * score``."""
        unit_scale = []
        for candidate in ranked_edu:
            scored = [f for f in candidate.ranking_factors if f.score is not None]
            if len(scored) != len(candidate.ranking_factors):
                continue
            for factor in scored:
                if factor.weight * factor.score:
                    unit_scale.append(
                        factor.contribution / (factor.weight * factor.score)
                    )
        assert unit_scale
        for ratio in unit_scale:
            assert ratio == pytest.approx(1.0, abs=0.01), ratio

    def test_sparse_candidates_are_penalised_not_scaled_up(self, ranked_edu):
        """A silent factor keeps its weight in the denominator and earns nothing.

        The old rule renormalised over the judged weight alone, so the five points
        a missing factor would have taken were redistributed to the factors that
        were scored and a sparse candidate gained up to 1.59x its nominal
        contribution. Silence now costs exactly the weight it was assigned, which
        is why the corpus's sparse candidates score below the complete ones.
        """
        sparse = [
            candidate
            for candidate in ranked_edu
            if any(f.score is None for f in candidate.ranking_factors)
        ]
        assert sparse, "corpus no longer contains a sparse-metadata candidate"
        for candidate in sparse:
            for factor in candidate.ranking_factors:
                if factor.score is None:
                    assert factor.contribution == 0.0
                else:
                    assert factor.contribution <= factor.weight * factor.score + 0.01, (
                        f"{candidate.id}/{factor.name.value}: a judged factor was paid "
                        "more than weight * score"
                    )

    def test_unscored_factors_contribute_nothing(self, ranked_edu):
        for candidate in ranked_edu:
            for factor in candidate.ranking_factors:
                if factor.score is None:
                    assert factor.contribution == 0.0, (
                        f"{candidate.id}/{factor.name.value}: an UNKNOWN factor "
                        "must contribute nothing rather than a guess"
                    )

    def test_factor_view_preserves_the_evidence_detail(self, ranked_edu):
        views = build_factor_views(ranked_edu[0])
        assert views
        for view in views:
            assert isinstance(view.contradictions, list)
            assert isinstance(view.unknown_terms, list)
            assert isinstance(view.unmatched_terms, list)
            assert isinstance(view.matched_terms, list)
            assert view.detail, view.name

    def test_views_are_ordered_by_influence(self, ranked_edu):
        views = build_factor_views(ranked_edu[0])
        contributions = [v.contribution for v in views]
        assert contributions == sorted(contributions, reverse=True)

    def test_every_factor_has_a_detail_string(self, ranked_edu):
        for candidate in ranked_edu:
            for factor in candidate.ranking_factors:
                assert factor.detail, f"{candidate.id}/{factor.name.value}"


class TestRankedItems:
    def test_ranks_are_one_based_and_dense(self, report):
        for evaluation in report.per_case:
            ranks = [item.rank for item in evaluation.ranked]
            assert ranks == list(range(1, len(ranks) + 1)), evaluation.case_id

    def test_scores_are_non_increasing_by_rank(self, report):
        for evaluation in report.per_case:
            scores = [item.score for item in evaluation.ranked]
            assert scores == sorted(scores, reverse=True), evaluation.case_id

    def test_best_available_prefers_the_highest_grade(self, report):
        for evaluation in report.per_case:
            best = best_available(evaluation.ranked)
            if best is None:
                continue
            assert best.grade == max(item.grade for item in evaluation.ranked)

    def test_best_available_of_empty_is_none(self):
        assert best_available([]) is None

    def test_build_ranked_items_carries_grades_through(self, compiled_edu):
        items = build_ranked_items(
            list(compiled_edu.candidates),
            {c.id: 3 for c in compiled_edu.candidates},
        )
        assert len(items) == len(compiled_edu.candidates)
        assert all(item.grade == 3 for item in items)

    def test_ungraded_candidate_defaults_to_irrelevant(self, compiled_edu):
        items = build_ranked_items(list(compiled_edu.candidates), {})
        assert all(item.grade == 0 for item in items)


class TestEvidenceQuality:
    def test_reports_bounded_fractions(self, report):
        seen = False
        for evaluation in report.per_case:
            for item in evaluation.ranked:
                quality = evidence_quality(item)
                seen = seen or bool(quality)
                for name, value in quality.items():
                    assert 0.0 <= value <= 1.0, (item.candidate_id, name, value)
        assert seen, "evidence_quality never produced a value"


class TestUnknownAsMismatch:
    def test_clean_ranker_reports_none(self, report):
        # The headline claim of the whole exercise is that UNKNOWN is not silently
        # converted into a mismatch. If this is non-zero, the claim is false.
        assert report.diagnostics["unknown_treated_as_mismatch"] == 0.0

    def test_counter_agrees_with_per_case_values(self, report):
        total = sum(e.suspected_unknown_as_mismatch for e in report.per_case)
        assert total / len(report.per_case) == report.diagnostics[
            "unknown_treated_as_mismatch"
        ]

    def test_counter_is_zero_on_a_healthy_case(self, ranked_edu, compiled_edu):
        items = build_ranked_items(ranked_edu, compiled_edu.grades_by_id)
        assert suspected_unknown_as_mismatch(items) == 0

    def test_counter_flags_an_injected_fake_mismatch(self, compiled_edu, ranked_edu):
        """The detector has to be able to fire, or its zero proves nothing.

        A negative-only test is satisfied by a function that always returns 0. The
        fingerprint is score 0.0 together with non-empty ``unknown_terms`` on an
        evidence-sensitive factor, so that is what gets injected.
        """
        from dataclasses import replace

        from app.services.datasets.evaluation.models import FactorContribution

        items = build_ranked_items(ranked_edu, compiled_edu.grades_by_id)
        victim = items[0]
        factors = []
        touched = 0
        for factor in victim.factors:
            values = factor.model_dump()
            if factor.name in {"feature_match", "target_match"}:
                values["score"] = 0.0
                values["unknown_terms"] = ["some undeclared requirement"]
                touched += 1
            factors.append(FactorContribution(**values))
        assert touched, "no evidence-sensitive factor available to poison"
        poisoned = victim.model_copy(update={"factors": factors})
        assert suspected_unknown_as_mismatch([poisoned]) > 0


class TestFailureTaxonomy:
    def test_inversion_is_detected(self, compiled_edu, ranked_edu):
        items = build_ranked_items(
            list(reversed(ranked_edu)), compiled_edu.grades_by_id
        )
        failures = classify_failures(compiled_edu, items)
        assert any(f.category == FailureCategory.INVERSION for f in failures)

    def test_ordered_ranking_is_not_called_an_inversion(self, compiled_edu, ranked_edu):
        items = build_ranked_items(ranked_edu, compiled_edu.grades_by_id)
        failures = classify_failures(compiled_edu, items)
        assert not any(f.category == FailureCategory.INVERSION for f in failures)

    def test_every_failure_is_self_describing(self, report):
        seen = set()
        for evaluation in report.per_case:
            for failure in evaluation.failures:
                assert isinstance(failure.category, FailureCategory)
                assert failure.detail, f"{evaluation.case_id}/{failure.category}"
                seen.add(failure.category)
        assert seen, "taxonomy never fired anywhere in the corpus"

    def test_known_defect_maps_to_a_target_failure(self, report):
        """The mortality/readmission gap must show up as a target failure.

        This keeps the known-failure list honest: if the taxonomy cannot name the
        mechanism, the golden list is only an opinion.
        """
        evaluation = next(
            e for e in report.per_case if e.case_id == "health-target-conflict"
        )
        categories = {f.category for f in evaluation.failures}
        assert categories & {
            FailureCategory.WRONG_TARGET,
            FailureCategory.CONTRADICTION_MISSED,
            FailureCategory.MISSING_TARGET_EVIDENCE,
        }, categories

    def test_case_kinds_account_for_every_case(self, report):
        # `failure_kind_counts` buckets whole cases, so it must sum to the case
        # count - that is what makes "21 of 62 cases miss" a checkable claim.
        assert sum(report.failure_kind_counts.values()) == report.case_count
        assert set(report.failure_kind_counts) <= {"none", "ranking", "retrieval"}

    def test_taxonomy_counts_match_individual_failures(self, report):
        assert sum(report.failure_taxonomy.values()) == sum(
            len(e.failures) for e in report.per_case
        )

    def test_taxonomy_only_mentions_declared_categories(self, report):
        declared = {c.value for c in FailureCategory}
        assert set(report.failure_taxonomy) <= declared

    def test_clean_cases_are_the_majority(self, report):
        assert report.failure_kind_counts.get("none", 0) > report.case_count / 2

    def test_every_declared_category_has_a_definition(self):
        assert len(list(FailureCategory)) >= 15


class TestScoreAudit:
    @pytest.fixture(scope="class")
    def audit(self, report, settings_):
        return audit_scores(report.per_case, settings_)

    def test_contributions_sum_to_score_everywhere(self, audit):
        assert audit.contributions_sum_matches_score
        assert audit.max_abs_contribution_error <= SUM_TOLERANCE

    def test_no_impossible_contributions(self, audit):
        assert audit.impossible_contributions == 0

    def test_no_duplicate_factors(self, audit):
        assert audit.duplicated_contributions == 0

    def test_every_named_factor_is_audited(self, audit):
        assert len(audit.factors) >= 10
        names = [entry.name for entry in audit.factors]
        assert len(names) == len(set(names))

    def test_weights_are_positive_and_sum_to_one_hundred(self, audit):
        assert audit.max_total_weight == pytest.approx(100.0)
        for entry in audit.factors:
            assert entry.weight > 0

    def test_factor_entries_document_their_semantics(self, audit):
        for entry in audit.factors:
            assert entry.semantic_meaning
            assert entry.raw_range
            assert isinstance(entry.higher_is_better, bool)
            assert isinstance(entry.unknown_possible, bool)
            assert isinstance(entry.mismatch_possible, bool)
            assert entry.nominal_share == pytest.approx(entry.weight / 100.0)

    def test_unknown_capable_factors_are_flagged(self, audit):
        unknown_factors = [e for e in audit.factors if e.unknown_possible]
        assert unknown_factors, "audit stopped tracking which factors can be UNKNOWN"

    def test_inflation_is_reported_not_hidden(self, audit):
        """No factor exceeds its nominal weight, and the audit says so out loud.

        The check is on ``contribution / weight``. When the denominator is pinned
        this ratio is capped at 1.0 by construction, so the audit's finding here is
        the absence of one - which is the state worth pinning, since the previous
        behaviour was an audit that reported inflation it was merely misreading.
        """
        inflated = [
            e for e in audit.factors if e.observed_max_ratio > 1.0 + INFLATION_TOLERANCE
        ]
        assert not inflated, f"unexpected inflation: {[e.name for e in inflated]}"
        for entry in audit.factors:
            assert entry.observed_max_ratio <= 1.0 + INFLATION_TOLERANCE, entry.name
            assert entry.observed_max_contribution <= entry.weight + 0.01, entry.name
            assert entry.nominal_share == pytest.approx(entry.weight / 100.0)
            assert isinstance(entry.can_dominate, bool)
            assert all(f for f in entry.findings), entry.name
        # Factors that are heavy or dominant still get flagged; a factor that is
        # neither is expected to say nothing at all.
        flagged = {e.name for e in audit.factors if e.findings}
        expected = {e.name for e in audit.factors if e.weight > 20.0 or e.can_dominate}
        assert expected <= flagged, f"missing structural findings: {sorted(expected - flagged)}"

    def test_findings_are_actionable(self, audit, report, settings_):
        """Findings render as text, and the healthy corpus produces none.

        The audit previously carried eleven warnings on this corpus; every one was
        the share-based inflation check misreading a candidate that simply scored
        below 100. So "there is at least one finding" is no longer a property of a
        healthy run - it is a property of a broken one. Both halves are checked
        here: silence when the arithmetic is sound, and a rendered finding when a
        factor is injected above its weight.
        """
        assert list(summarise_findings(audit)) == []

        victim = next(
            evaluation
            for evaluation in report.per_case
            if any(item.score > 0 for item in evaluation.ranked)
        )
        original = victim.ranked[0]
        inflated_factors = [
            f.model_copy(update={"contribution": f.weight * 1.5}) for f in original.factors
        ]
        inflated_score = sum(f.contribution for f in inflated_factors)
        poisoned = original.model_copy(
            update={
                "factors": inflated_factors,
                "score": inflated_score,
                "ranking_score": inflated_score,
            }
        )
        broken_case = victim.model_copy(update={"ranked": [poisoned]})
        broken = audit_scores([broken_case], settings_)
        rendered = list(summarise_findings(broken))
        assert rendered, "an injected 1.5x contribution produced no finding"
        for finding in rendered:
            assert isinstance(finding, str) and finding
        assert any("inflation" in f.lower() for f in rendered), rendered
        assert any(f.severity.lower() == "warning" for f in broken.findings)

    def test_no_error_severity_findings(self, audit):
        errors = [f for f in audit.findings if f.severity.lower() == "error"]
        assert errors == [], errors

    def test_dominance_table_is_sorted_and_bounded(self, report):
        table = dominance_table(report.per_case[0], limit=3)
        assert len(table) <= 3
        values = [row[2] for row in table]
        assert values == sorted(values, reverse=True)

    def test_factor_influence_is_bounded(self, report):
        influence = factor_influence_over_benchmark(report)
        assert influence
        for value in influence.values():
            assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9


class TestRendering:
    def test_case_report_names_the_case(self, report, compiled_edu):
        evaluation = next(
            e for e in report.per_case if e.case_id == compiled_edu.case.id
        )
        text = render_case_report(compiled_edu, evaluation.ranked)
        assert compiled_edu.case.id in text
        assert isinstance(text, str)

    def test_factor_table_covers_only_the_limited_candidates(self, report):
        # `limit` counts candidates, not rows: each candidate contributes one row
        # per factor, so the row count is bounded by limit * factors, not limit.
        text = render_factor_table(report.per_case[0].ranked, limit=2)
        named = [
            line
            for line in text.splitlines()
            if line.startswith("|") and "---" not in line
        ]
        data_rows = named[1:]
        candidates = {row.split("|")[1].strip() for row in data_rows}
        assert len(candidates) <= 2, text