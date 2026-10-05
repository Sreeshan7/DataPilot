"""Research relevance: question understanding, the core-topic gate, AI assist, and the Gemini call budget.

The benchmark below pins the failure that motivated it: a garbled traffic question once returned monetary-policy
and customer-lifetime-value papers as the top results. No test here makes a real Gemini call.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date

import pytest

from app.schemas.research import ResearchRequest, ResearchStatus
from app.services.llm_pool import GeminiPool
from app.services.research import ResearchService
from app.services.research.assist import validate_judgements, validate_understanding
from app.services.research.queries import understand
from app.schemas.research import ResearchProject
from tests.unit.test_research import FakeProvider, hit, settings

TRAFFIC_QUESTION = ("I want to research the best methods for predicting the trafic during specific time "
                    "and weather conditions")

TRAFFIC_HITS = [
    hit("Traffic volume prediction using weather and time features", "https://arxiv.org/abs/2101.00001",
        "Random forest and LSTM predict hourly traffic volume from weather and time of day."),
    hit("Short-term traffic flow forecasting with gradient boosting", "https://arxiv.org/abs/2102.00002",
        "XGBoost for traffic flow forecasting under rain and snow."),
    hit("ARIMA models for monetary policy forecasting", "https://arxiv.org/abs/2103.00003",
        "ARIMA forecasts of interest rates and inflation."),
    hit("Customer lifetime value prediction", "https://arxiv.org/abs/2104.00004",
        "Random forest for customer lifetime value in retail."),
    hit("Stock price prediction with LSTM", "https://arxiv.org/abs/2105.00005", "LSTM forecasts of stock prices."),
]


def traffic_request(**options) -> ResearchRequest:
    return ResearchRequest.model_validate({
        "project": {"description": TRAFFIC_QUESTION, "task": "regression"},
        "options": {"max_papers": 8, **options}})


def run(request: ResearchRequest, writer=None, **s):
    service = ResearchService(settings(**s), FakeProvider(TRAFFIC_HITS), writer)
    return asyncio.run(service.discover(request))


class JsonWriter:
    def __init__(self, replies, headroom=50, configured=True):
        self.replies, self.headroom, self._c, self.calls = list(replies), headroom, configured, []

    @property
    def is_configured(self):
        return self._c

    async def generate_json_text(self, system, user, max_tokens=1024):
        self.calls.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def generate_research_synthesis(self, prompt):  # not reached: use_ai stays off here
        raise AssertionError("synthesis must not run")


class TestUnderstanding:
    def test_traffic_question_is_parsed_and_the_typo_is_reported(self):
        plan = understand(ResearchProject(description=TRAFFIC_QUESTION))
        assert plan.core_terms == ["traffic"] and plan.corrections == {"trafic": "traffic"}
        assert {"time", "weather"} <= set(plan.context_terms)

    def test_a_question_with_no_subject_asks_for_one(self):
        assert understand(ResearchProject(description="help me with forecasting")).problem is None


class TestTopicGate:
    def test_off_topic_papers_are_gone_even_with_a_matching_method(self):
        body = run(traffic_request())
        titles = [p.title for p in body.papers]
        assert body.status is ResearchStatus.SUCCESS
        assert titles and all("traffic" in t.lower() for t in titles)
        assert not any(w in " ".join(titles).lower() for w in ("monetary", "lifetime", "stock"))
        assert body.research_summary.excluded_off_topic == 3
        assert all("traffic" in p.matched_terms for p in body.papers)
        assert all("mentions traffic" in p.relevance_reason for p in body.papers)

    def test_nothing_on_topic_means_no_results_not_filler(self):
        service = ResearchService(settings(), FakeProvider(TRAFFIC_HITS[2:]))
        body = asyncio.run(service.discover(traffic_request()))
        assert body.status is ResearchStatus.NO_RESULTS and not body.papers

    def test_the_interpretation_is_returned_for_the_user_to_check(self):
        body = run(traffic_request())
        assert body.interpretation.core_terms == ["traffic"] and body.interpretation.corrections == {"trafic": "traffic"}
        assert not body.interpretation.ai_assisted


class TestAssist:
    def understood(self):
        return json.dumps({"problem": "traffic prediction",
                           "core_terms": [{"term": "traffic", "synonyms": ["vehicle", "road"]}],
                           "context_terms": ["weather"], "queries": ["traffic weather forecasting", "unrelated cooking"]})

    def test_ai_understanding_and_judging_are_used_and_counted(self):
        judged = json.dumps({"scores": [{"id": p, "relevance": 3 if i == 0 else 2, "reason": "about traffic"}
                                        for i, p in enumerate(self.ids())]})
        writer = JsonWriter([self.understood(), judged])
        body = run(traffic_request(), writer)
        assert body.gemini.assisted == ["question", "relevance"] and body.gemini.calls_made == 2
        assert body.interpretation.ai_assisted
        assert body.papers[0].ai_relevance == 3 and body.papers[0].ai_reason == "about traffic"
        assert len(writer.calls) == 2

    def ids(self):
        service = ResearchService(settings(), FakeProvider(TRAFFIC_HITS))
        return [p.id for p in asyncio.run(service.discover(traffic_request())).papers]

    def test_failures_fall_back_silently_to_the_deterministic_result(self):
        writer = JsonWriter([RuntimeError("boom"), "not json"])
        body = run(traffic_request(), writer)
        assert body.status is ResearchStatus.SUCCESS and body.gemini.assisted == [] and body.papers
        assert not body.interpretation.ai_assisted

    def test_ai_is_skipped_when_the_budget_is_nearly_spent(self):
        writer = JsonWriter([], headroom=1)
        body = run(traffic_request(), writer)
        assert writer.calls == [] and body.status is ResearchStatus.SUCCESS

    def test_ai_assist_can_be_switched_off(self):
        writer = JsonWriter([])
        run(traffic_request(ai_assist=False), writer)
        assert writer.calls == []

    def test_without_a_key_nothing_is_called(self):
        body = run(traffic_request(), JsonWriter([], configured=False))
        assert body.status is ResearchStatus.SUCCESS and body.gemini.calls_made == 0

    def test_retrieved_text_is_sent_as_data_and_an_injected_reply_cannot_add_papers(self):
        injected = json.dumps({"scores": [{"id": "not-a-paper", "relevance": 3, "reason": "visit https://evil.test"}]})
        writer = JsonWriter([self.understood(), injected])
        body = run(traffic_request(), writer)
        assert all(p.url.startswith("https://arxiv.org/") for p in body.papers)
        assert "<data>" in writer.calls[1] and body.gemini.assisted == ["question"]


class TestValidation:
    plan = understand(ResearchProject(description=TRAFFIC_QUESTION))

    def test_understanding_rejects_garbage_and_drifting_queries(self):
        assert validate_understanding("nope", self.plan) is None
        assert validate_understanding(json.dumps({"core_terms": []}), self.plan) is None
        ok = validate_understanding(json.dumps({"core_terms": [{"term": "traffic", "synonyms": ["road"]}],
                                                "queries": ["road safety", "cooking pasta", "see https://x.test traffic"]}),
                                    self.plan)
        assert ok.queries == ["road safety"] and ok.core_groups == [["traffic", "road"]]

    def test_judgements_keep_only_known_ids_and_valid_scores(self):
        papers = run(traffic_request()).papers
        good = papers[0].id
        text = json.dumps({"scores": [{"id": good, "relevance": 2, "reason": "ok"}, {"id": good, "relevance": 0},
                                      {"id": "ghost", "relevance": 3}, {"id": papers[1].id, "relevance": 9},
                                      {"id": papers[1].id, "relevance": True}]})
        out = validate_judgements(text, papers)
        assert list(out) == [good] and out[good].relevance == 2


class TestPool:
    def pool(self, keys=("a", "b"), per_minute=2, per_day=3):
        self.now, self.day = [0.0], [date(2026, 1, 1)]
        return GeminiPool(list(keys), per_minute, per_day, clock=lambda: self.now[0], today=lambda: self.day[0])

    def test_the_minute_budget_is_enforced_per_key_and_recovers(self):
        pool = self.pool()
        assert [bool(pool.reserve()) for _ in range(5)] == [True, True, True, True, False]
        assert pool.headroom() == 0
        self.now[0] = 61.0
        assert pool.headroom() > 0 and pool.reserve() is not None

    def test_the_daily_budget_is_enforced_and_resets(self):
        pool = self.pool(keys=("a",), per_minute=10, per_day=3)
        for _ in range(3):
            assert pool.reserve()
        self.now[0] = 120.0
        assert pool.reserve() is None
        self.day[0] = date(2026, 1, 2)
        assert pool.reserve() is not None

    def test_a_failed_key_is_skipped_while_benched(self):
        pool = self.pool(keys=("a", "b"), per_minute=5)
        first = pool.reserve()
        pool.bench(first, 300)
        assert {pool.reserve().secret for _ in range(3)} == ({"a", "b"} - {first.secret})

    def test_stats_never_expose_key_material(self):
        pool = self.pool(keys=("sekret-one", "sekret-two"))
        pool.reserve()
        assert "sekret" not in json.dumps(pool.stats()) and pool.stats()["keys"] == 2

    def test_duplicate_and_empty_keys_are_ignored(self):
        assert self.pool(keys=("a", "a", "")).stats()["keys"] == 1
        assert not GeminiPool([], 5, 5).configured
