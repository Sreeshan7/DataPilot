"""Relevance upgrade: dataset-aware research queries, the two new ranking bonuses, AI-planned queries first, execution plans,
the dataset setting-fit judge and the richer project blueprints. No test here makes a real Gemini call."""

from __future__ import annotations

import asyncio
import json

from app.schemas.dataset import SourceType
from app.schemas.research import (
    AlgorithmRecommendation,
    Level,
    Paper,
    ResearchDataset,
    ResearchProject,
    ResearchRequest,
    ResearchStatus,
)
from app.services.datasets import fit_judge
from app.services.datasets.discovery import DatasetDiscoveryResult, DatasetDiscoveryService
from app.services.research import ResearchService, execution
from app.services.research.queries import build_queries, dataset_subject, understand
from app.services.research.ranking import rank
from tests.conftest import make_candidate, make_requirements, make_settings
from tests.unit.test_research import FakeProvider, hit, settings

CHURN = ResearchProject(description="predict which telecom customers will leave", problem="churn prediction", task="binary_classification")
TELCO = ResearchDataset(name="Telco_Customer_Churn.csv", columns=["tenure", "Contract", "Churn"], target="Churn", rows=7043)


class TestDatasetAwareQueries:
    def test_a_file_label_says_nothing_about_the_subject(self):
        assert dataset_subject(ResearchDataset(name="train.csv")) == (None, [])
        assert dataset_subject(ResearchDataset(name="clean_data_v2.csv")) == (None, [])
        assert dataset_subject(None) == (None, [])

    def test_a_real_dataset_name_gives_its_subject_words(self):
        name, terms = dataset_subject(TELCO)
        assert name == "Telco Customer Churn" and terms[:2] == ["telco", "customer"]

    def test_the_dataset_name_becomes_a_query_and_no_query_repeats_a_word(self):
        plan = understand(CHURN, TELCO)
        plan.outcome, plan.task = "customer churn", "classification"
        queries, _ = build_queries(CHURN, TELCO, 5, plan=plan)
        assert any('"Telco Customer Churn" dataset' in q for q in queries)
        assert not any("churn customer churn" in q for q in queries)
        assert len(queries) <= 5 and len({q.lower() for q in queries}) == len(queries)

    def test_without_a_dataset_the_queries_are_unchanged_in_kind(self):
        queries, _ = build_queries(CHURN, None, 5)
        assert queries[0] == "churn machine learning" and not any("dataset" in q for q in queries)


def _paper(title, snippet, queries=("q1",)) -> Paper:
    return Paper(id=title[:8], title=title, url="https://arxiv.org/abs/2101.0001", source="arxiv.org", abstract_snippet=snippet,
                 publication_year=2024, year_evidence="source_metadata", matched_queries=list(queries))


class TestRankingBonuses:
    def score(self, paper, dataset=TELCO):
        plan = understand(CHURN, dataset)
        return rank([paper], CHURN, dataset, min_year=None, plan=plan, today_year=2026)[0]

    def test_a_paper_that_names_the_dataset_ranks_above_one_that_only_shares_the_topic(self):
        with_name = self.score(_paper("Churn prediction on the Telco customer dataset", "We model churn for a telco operator."))
        without = self.score(_paper("Churn prediction in banking", "We model churn for a retail bank."))
        assert with_name.score_components["dataset"] > 0 and without.score_components["dataset"] == 0
        assert with_name.relevance_score > without.relevance_score
        assert "dataset's name" in with_name.relevance_reason

    def test_a_paper_found_by_several_searches_earns_the_agreement_bonus(self):
        many = self.score(_paper("Churn prediction", "churn models", ("a", "b", "c", "d")))
        one = self.score(_paper("Churn prediction", "churn models", ("a",)))
        assert many.score_components["agreement"] == 4.0 and one.score_components["agreement"] == 0.0

    def test_the_listed_components_still_add_up_to_the_score(self):
        p = self.score(_paper("Churn prediction on the Telco dataset", "churn", ("a", "b")))
        assert abs(sum(p.score_components.values()) - p.relevance_score) < 0.05


class TestAiQueriesGoFirst:
    def test_the_scholarly_sources_that_take_two_queries_see_the_specific_ones(self):
        ai = json.dumps({"problem": "customer churn", "core_terms": [{"term": "churn", "synonyms": ["attrition"]}], "context_terms": ["telecom"],
                         "queries": ["telecom subscriber churn prediction", "customer attrition telco machine learning", "churn xgboost telecom"]})

        class Writer:
            is_configured, headroom = True, 50

            def __init__(self):
                self.replies = [ai]

            async def generate_json_text(self, system, user, max_tokens=1024):
                if self.replies:
                    return self.replies.pop(0)
                raise RuntimeError("no further replies")

        provider = FakeProvider([hit("Churn prediction in telecom", "https://arxiv.org/abs/2101.00001", "telecom churn with machine learning")])
        request = ResearchRequest(project=CHURN, dataset=TELCO)
        body = asyncio.run(ResearchService(settings(), provider, Writer()).discover(request))
        assert body.status is ResearchStatus.SUCCESS
        assert provider.calls[0] == "telecom subscriber churn prediction" and provider.calls[1] == "customer attrition telco machine learning"
        assert len(body.queries) <= 5


def _rec(method="Random Forest", rank_=1) -> AlgorithmRecommendation:
    return AlgorithmRecommendation(rank=rank_, method=method, literature_support=Level.STRONG, dataset_compatibility=Level.STRONG,
                                   evidence_paper_count=3, why="x")


class TestExecutionPlans:
    def test_every_algorithm_gets_a_built_in_plan_with_no_ai(self):
        plan = execution.deterministic_execution("XGBoost", "classification", TELCO)
        assert plan.source == "datapilot_knowledge" and "xgboost" in plan.libraries and len(plan.steps) >= 5
        assert "stratified" in plan.validate_with and plan.choose_when and plan.key_settings and plan.pitfalls

    def test_a_small_dataset_adds_a_repeat_the_validation_step(self):
        plan = execution.deterministic_execution("Random Forest", "classification", ResearchDataset(rows=300, columns=["a"]))
        assert any("300 rows" in s for s in plan.steps)

    def good(self, **kw):
        entry = {"method": "Random Forest", "choose_when": "Pick it when you want a forgiving model that needs little tuning.",
                 "libraries": ["sklearn", "pandas", "madeup-lib"],
                 "steps": ["Split with `train_test_split` using a stratified 80/20 split.", "Encode `Contract` with one-hot columns.",
                           "Fit `RandomForestClassifier` with default settings.", "Tune `n_estimators` and `max_depth` with cross-validation.",
                           "Evaluate once on the test rows."],
                 "key_settings": ["`n_estimators`: start at 300, try 100 to 800"], "validate_with": "Use a stratified split and report ROC AUC.",
                 "pitfalls": ["`tenure` and total charges move together, so importances overlap."]}
        entry.update(kw)
        return json.dumps({"algorithms": [entry]})

    def validate(self, reply):
        base = {"Random Forest": execution.deterministic_execution("Random Forest", "classification", TELCO)}
        return execution.validate_execution(reply, {"Random Forest"}, base, set(TELCO.columns)), base

    def test_a_valid_reply_replaces_the_built_in_plan_and_unknown_libraries_are_dropped(self):
        out, _ = self.validate(self.good())
        plan = out["Random Forest"]
        assert plan.source == "ai_inferred" and plan.libraries == ["scikit-learn", "pandas"] and len(plan.steps) == 5

    def test_a_made_up_column_in_backticks_is_rejected_field_by_field(self):
        out, base = self.validate(self.good(pitfalls=["`monthly_spend_score` is heavily skewed."], choose_when="Use `Phantom` here."))
        plan = out["Random Forest"]
        assert plan.pitfalls == base["Random Forest"].pitfalls and plan.choose_when == base["Random Forest"].choose_when
        assert plan.source == "ai_inferred"  # the steps were fine

    def test_promises_causes_and_links_are_rejected(self):
        for bad in ("It will reach 95% accuracy.", "`Contract` causes churn.", "See https://example.com for steps."):
            out, _ = self.validate(self.good(steps=[bad] * 6))
            assert out == {}  # no valid steps: DataPilot's own plan stands

    def test_too_few_steps_or_the_wrong_method_is_ignored(self):
        assert self.validate(self.good(steps=["Only one step."]))[0] == {}
        assert self.validate(self.good(method="Deep Learning"))[0] == {}

    def test_a_failing_or_unconfigured_writer_gives_nothing(self):
        class Boom:
            is_configured, headroom = True, 50

            async def generate_json_text(self, *a, **k):
                raise RuntimeError("boom")

        base = {"Random Forest": execution.deterministic_execution("Random Forest", None, None)}
        for writer in (Boom(), None):
            assert asyncio.run(execution.refine_execution(writer, settings(), {}, [_rec()], base, None)) == {}

    def test_the_research_response_always_carries_an_execution_plan(self):
        provider = FakeProvider([hit("Churn prediction with random forest", "https://arxiv.org/abs/2101.00001",
                                     "We predict customer churn using random forest and logistic regression on telecom data.")])
        body = asyncio.run(ResearchService(settings(), provider).discover(ResearchRequest(project=CHURN, dataset=TELCO)))
        assert body.status is ResearchStatus.SUCCESS and body.algorithm_recommendations
        assert all(r.analysis and r.analysis.execution and r.analysis.execution.steps for r in body.algorithm_recommendations)


class FitWriter:
    def __init__(self, reply, headroom=50, configured=True):
        self.reply, self.headroom, self._c, self.calls = reply, headroom, configured, 0

    @property
    def is_configured(self):
        return self._c

    async def generate_json_text(self, system, user, max_tokens=1024):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _cands():
    return [make_candidate(name="Generic taxi fares", source_id="a/generic", ranking_score=77.0, ranking_reasons=["scored"]),
            make_candidate(name="Taxi price regression", source_id="a/synthetic", ranking_score=73.0),
            make_candidate(name="NYC Yellow Taxi Trip Data", source_id="a/nyc", ranking_score=65.0),
            make_candidate(name="Other taxi data", source_id="a/other", ranking_score=60.0)]


def _scores(**by_id):
    return json.dumps({"scores": [{"id": f"kaggle:a/{k}", "fit": v, "reason": f"reason {k}"} for k, v in by_id.items()]})


class TestFitJudge:
    def test_a_close_match_moves_ahead_and_a_loose_one_behind_without_touching_any_score(self):
        fits = fit_judge.validate_fits(_scores(generic=1, synthetic=1, nyc=3, other=2), {c.id for c in _cands()})
        ordered, changed = fit_judge.reorder(_cands(), fits)
        assert changed and [c.name for c in ordered] == ["NYC Yellow Taxi Trip Data", "Other taxi data", "Generic taxi fares", "Taxi price regression"]
        assert {c.name: c.ranking_score for c in ordered} == {c.name: c.ranking_score for c in _cands()}
        assert ordered[0].ai_fit == 3 and "close match" in ordered[0].ranking_reasons[0]
        assert ordered[2].ranking_reasons[1] == "scored"  # the deterministic reasons are kept

    def test_when_nothing_changes_the_order_is_reported_unchanged(self):
        fits = fit_judge.validate_fits(_scores(generic=3, synthetic=2, nyc=2, other=2), {c.id for c in _cands()})
        assert fit_judge.reorder(_cands(), fits)[1] is False

    def test_a_bad_reply_is_validated_away(self):
        ids = {c.id for c in _cands()}
        reply = json.dumps({"scores": [{"id": "kaggle:a/nyc", "fit": 7, "reason": "x"}, {"id": "ghost", "fit": 3, "reason": "x"},
                                       {"id": "kaggle:a/other", "fit": True, "reason": "x"},
                                       {"id": "kaggle:a/generic", "fit": 2, "reason": "see https://x.io"}]})
        fits = fit_judge.validate_fits(reply, ids)
        assert list(fits) == ["kaggle:a/generic"] and fits["kaggle:a/generic"] == (2, "")

    def test_judging_skips_cleanly_without_a_key_budget_or_enough_datasets(self):
        s = make_settings()
        assert asyncio.run(fit_judge.judge_fit(FitWriter("x", configured=False), s, {}, _cands()))[0] == {}
        assert asyncio.run(fit_judge.judge_fit(FitWriter("x", headroom=1), s, {}, _cands()))[0] == {}
        assert asyncio.run(fit_judge.judge_fit(FitWriter("x"), s, {}, _cands()[:1]))[0] == {}
        assert asyncio.run(fit_judge.judge_fit(FitWriter(RuntimeError("boom")), s, {}, _cands()))[0] == {}

    def test_a_reply_covering_too_few_datasets_is_ignored(self):
        fits, why = asyncio.run(fit_judge.judge_fit(FitWriter(_scores(nyc=3)), make_settings(), {}, _cands()))
        assert fits == {} and "too few" in why

    def test_ai_suggestions_are_never_judged(self):
        suggested = make_candidate(name="Suggested", source="genai", source_id="x", source_type=SourceType.AI_SUGGESTED)
        writer = FitWriter(_scores(generic=3, synthetic=2, nyc=2, other=2))
        asyncio.run(fit_judge.judge_fit(writer, make_settings(), {}, [*_cands(), suggested]))
        assert writer.calls == 1  # judged the four registry-confirmed ones; the suggestion was not in the batch

    def run_service(self, writer, **settings_kw):
        service = DatasetDiscoveryService(settings=make_settings(**settings_kw), sources=[], fit_writer=writer)
        result = DatasetDiscoveryResult(query="taxi", candidates=_cands(), requirements=make_requirements())
        return asyncio.run(service._apply_fit_judge("taxi fares in New York", make_requirements(), result))

    def test_the_service_applies_the_judgement_and_reports_it_in_the_result(self):
        result = self.run_service(FitWriter(_scores(generic=1, synthetic=1, nyc=3, other=2)))
        assert result.candidates[0].name == "NYC Yellow Taxi Trip Data" and result.fit_judged == 4 and result.fit_reordered

    def test_the_service_keeps_the_deterministic_order_when_the_ai_is_unavailable_or_switched_off(self):
        failed = self.run_service(FitWriter(RuntimeError("boom")))
        assert [c.name for c in failed.candidates] == [c.name for c in _cands()] and failed.fit_judged == 0 and "skipped" in failed.fit_note
        writer = FitWriter("x")
        off = self.run_service(writer, dataset_fit_judge_enabled=False)
        assert writer.calls == 0 and [c.name for c in off.candidates] == [c.name for c in _cands()]
