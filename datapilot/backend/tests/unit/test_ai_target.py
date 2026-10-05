"""Goal-aware target choice: an AI pick is validated, labelled as AI, and can never be invented or silent."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import random
from functools import lru_cache

from app.schemas.dataset_analysis import CandidateState, EvidenceLevel, SUPERVISED_TASKS
from app.services.dataset_analysis.ai_target import refine_with_goal
from app.services.dataset_analysis.service import AnalysisRequest, DatasetAnalysisService
from tests.unit.pi_helpers import SETTINGS, build

GOAL = "I want to predict how busy the road will be based on weather and time"


def traffic_csv() -> bytes:
    random.seed(7)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date_time", "temp_c", "rain_mm", "is_holiday", "hour", "traffic_volume"])
    for i in range(1200):
        hour = i % 24
        base = 4200 if hour in (7, 8, 16, 17, 18) else 1800
        rain = round(max(0, random.gauss(0.3, 1.0)), 2)
        w.writerow([f"2023-01-{1 + i // 24:02d} {hour:02d}:00", round(random.gauss(14, 8), 1), rain,
                    int(i % 168 > 120), hour, max(int(base - rain * 140 + random.gauss(0, 250)), 50)])
    return buf.getvalue().encode()


def analyze(target=None):
    service = DatasetAnalysisService(SETTINGS)
    return asyncio.run(service.analyze(AnalysisRequest(
        stream=io.BytesIO(traffic_csv()), file_name="traffic.csv", target_hint=target)))


class Writer:
    def __init__(self, reply, headroom=50, configured=True):
        self.reply, self.headroom, self._c, self.calls = reply, headroom, configured, []

    @property
    def is_configured(self):
        return self._c

    async def generate_json_text(self, system, user, max_tokens=1024):
        self.calls.append(user)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def refine(outcome, writer, goal=GOAL):
    return asyncio.run(refine_with_goal(outcome, goal, SETTINGS, writer))


def best(outcome):
    return outcome.analysis.target_candidates[0] if outcome.analysis.target_candidates else None


class TestAiTarget:
    def test_the_premise_names_alone_do_not_find_the_target(self):
        top = best(analyze())
        assert top is None or top.column != "traffic_volume" or top.state is not CandidateState.STRONG_CANDIDATE

    def test_a_valid_pick_becomes_a_labelled_possible_target_with_a_supervised_task(self):
        writer = Writer(json.dumps({"target": "traffic_volume", "reason": "volume measures how busy the road is"}))
        outcome, picked = refine(analyze(), writer)
        top = best(outcome)
        assert picked == "traffic_volume" and top.column == "traffic_volume"
        assert top.user_specified is False and top.state is CandidateState.POSSIBLE
        assert any(e.level is EvidenceLevel.AI_INFERRED and "AI model" in e.statement for e in top.evidence)
        assert any(t.target == "traffic_volume" and t.task in SUPERVISED_TASKS for t in outcome.analysis.task_candidates)
        assert "2023-01" not in writer.calls[0]

    def test_the_project_ideas_then_center_on_that_target_and_carry_a_research_request(self):
        outcome, _ = refine(analyze(), Writer(json.dumps({"target": "traffic_volume", "reason": ""})))
        pi = build(outcome, goal=GOAL)
        supervised = [i for i in pi.project_ideas if i.target == "traffic_volume"]
        assert supervised and supervised[0].feasibility.level in ("high", "medium", "low")
        assert supervised[0].research_request["project"]["target"] == "traffic_volume"
        assert "possible target" in " ".join(pi.understanding.limitations)  # still asks the user to confirm it

    def test_an_invented_column_or_null_changes_nothing(self):
        for reply in (json.dumps({"target": "ghost_column"}), json.dumps({"target": None}), "not json"):
            outcome = analyze()
            before = [t.column for t in outcome.analysis.target_candidates]
            outcome, picked = refine(outcome, Writer(reply))
            assert picked is None and [t.column for t in outcome.analysis.target_candidates] == before

    def test_failures_and_missing_keys_change_nothing(self):
        for writer in (Writer(RuntimeError("boom")), Writer("{}", configured=False), None):
            outcome, picked = refine(analyze(), writer)
            assert picked is None

    def test_no_call_without_a_goal_with_low_budget_or_when_the_target_is_already_known(self):
        w = Writer(json.dumps({"target": "traffic_volume"}))
        assert refine(analyze(), w, goal=None)[1] is None
        assert refine(analyze(), Writer(json.dumps({"target": "traffic_volume"}), headroom=1))[1] is None
        named = analyze(target="traffic_volume")  # the user named it: nothing to guess
        assert refine(named, w)[1] is None and w.calls == []
