"""Project blueprints: grounded deterministic plans, validated AI plans, and what the AI is never allowed to change."""

from __future__ import annotations

import asyncio
import json

from app.core.config import Settings
from app.schemas.project_intelligence import Feasibility, ProjectIdea
from app.services.project_intelligence.blueprint import (
    attach_blueprints,
    deterministic_blueprint,
    rank_ideas,
    validate_blueprint,
)
from app.services.project_intelligence.shape import shape_of
from tests.unit.pi_helpers import SETTINGS, build, intelligence_of, outcome_of


def settings(**kw):
    return Settings(gemini_api_key="", gemini_api_key_2="", app_env="testing", **kw)


class Writer:
    def __init__(self, reply=None, error=None, headroom=50, configured=True):
        self.reply, self.error, self.headroom, self._c, self.calls = reply, error, headroom, configured, []

    @property
    def is_configured(self):
        return self._c

    async def generate_json_text(self, system, user, max_tokens=1024):
        self.calls.append(user)
        if self.error:
            raise self.error
        return self.reply(json.loads(user.split("<data>\n")[1].split("\n</data>")[0])) if callable(self.reply) else self.reply


def plan_for(name, **kw):
    pi = intelligence_of(name, kw.get("target"))
    return pi, shape_of(outcome_of(name, kw.get("target")).analysis)


class TestDeterministicBlueprint:
    def test_every_idea_has_a_grounded_plan(self):
        pi, _ = plan_for("customer_churn")
        for idea in pi.project_ideas:
            bp = idea.blueprint
            assert bp and bp.source == "deterministic" and len(bp.build_steps) >= 5 and bp.effort and bp.real_world_value
            names = {c.name for c in shape_of(outcome_of("customer_churn").analysis).columns}
            assert all(k.column in names for k in bp.key_columns)

    def test_the_target_and_measured_drivers_lead_and_are_labelled_measured(self):
        pi, _ = plan_for("customer_churn")
        main = next(i for i in pi.project_ideas if i.id.startswith("predict"))
        roles = {k.column: k for k in main.blueprint.key_columns}
        assert roles[main.target].role == "target"
        drivers = [k for k in main.blueprint.key_columns if k.role == "key_driver"]
        assert drivers and all(k.evidence == "measured" for k in drivers)

    def test_a_leaky_column_is_flagged_as_excluded_never_as_a_driver(self):
        pi, _ = plan_for("student_performance_leaky")
        main = next(i for i in pi.project_ideas if i.id.startswith("predict"))
        flagged = {f.column for f in pi.feature_intelligence.features if f.role.value == "POSSIBLE_LEAKAGE"}
        assert flagged, "the fixture should contain a leaky column"
        by_col = {k.column: k.role for k in main.blueprint.key_columns}
        assert all(by_col.get(c) == "exclude" for c in flagged)

    def test_a_timestamp_is_to_be_turned_into_features_not_thrown_away(self):
        pi, _ = plan_for("sales_timeseries")
        for idea in pi.project_ideas:
            for k in idea.blueprint.key_columns:
                if "date" in k.column.lower() or "time" in k.column.lower():
                    assert k.role != "exclude" or "timestamp" not in k.why.lower()

    def test_a_descriptive_driver_idea_gets_its_own_plan_not_a_modelling_one(self):
        pi, _ = plan_for("customer_churn")
        driver = next((i for i in pi.project_ideas if i.id.startswith("drivers")), None)
        if driver:
            text = " ".join(s.title + s.detail for s in driver.blueprint.build_steps)
            assert "never 'causes'" in text and "Set a baseline" not in text


class TestAiBlueprints:
    def good_reply(self, payload, **over):
        out = []
        for idea in payload["ideas"]:
            cols = [c["name"] for c in idea["columns"]]
            out.append({
                "id": idea["id"], "headline": "A lender can use this to prioritise who to call first.",
                "real_world_value": "Supports a retention team deciding who to contact. Benefits customers who get help sooner.",
                "key_columns": [{"column": cols[0], "role": "key_driver", "why": f"`{cols[0]}` carries the main signal here."}] if cols else [],
                "build_steps": [{"title": f"Step {n}", "detail": f"Do thing {n} with `{cols[0]}`." if cols else f"Do thing {n}."} for n in range(1, 6)],
                "evaluation_plan": "Hold out a test set and report the metrics against the baseline.",
                "pitfalls": ["Watch for duplicate customers across the split."], "stretch_goals": ["Add a simple dashboard."],
                "effort": "about two weekends", **over})
        return json.dumps({"blueprints": out})

    def run(self, name, writer, **kw):
        pi = build(outcome_of(name), goal="predict churn of subscribers", writer=writer, settings=kw.get("settings", SETTINGS))
        return pi

    def test_a_valid_reply_becomes_an_ai_labelled_plan_with_the_data_roles_kept(self):
        w = Writer(self.good_reply)
        pi = self.run("customer_churn", w)
        top = pi.project_ideas[0]
        assert top.blueprint.source == "ai_inferred" and top.blueprint.effort == "about two weekends"
        assert any(k.role == "target" for k in top.blueprint.key_columns)
        assert len(w.calls) == 1  # one batched call for all top ideas
        assert "cell" not in w.calls[0].lower() and "customer_id" not in w.calls[0]  # names and types only; identifiers withheld

    def test_a_measured_role_cannot_be_overridden_and_invented_columns_are_dropped(self):
        pi0, _ = plan_for("student_performance_leaky")
        leak = next(f.column for f in pi0.feature_intelligence.features if f.role.value == "POSSIBLE_LEAKAGE")

        def reply(payload):
            idea = payload["ideas"][0]
            return json.dumps({"blueprints": [{
                "id": idea["id"], "headline": "h", "real_world_value": "v", "evaluation_plan": "e", "effort": "a weekend",
                "key_columns": [{"column": leak, "role": "key_driver", "why": "Looks very predictive."},
                                {"column": "invented_column", "role": "key_driver", "why": "Made up."}],
                "build_steps": [{"title": f"S{n}", "detail": f"Detail {n}."} for n in range(1, 6)]}]})

        pi = self.run("student_performance_leaky", Writer(reply))
        main = pi.project_ideas[0]
        roles = {k.column: k.role for k in main.blueprint.key_columns}
        assert main.blueprint.source == "ai_inferred"
        assert roles.get(leak) == "exclude" and "invented_column" not in roles

    def test_one_bad_sentence_costs_that_sentence_not_the_plan(self):
        pi = self.run("customer_churn", Writer(lambda p: self.good_reply(p, real_world_value="Low prices cause customers to stay forever.")))
        top = pi.project_ideas[0].blueprint
        assert top.source == "ai_inferred" and "cause" not in top.real_world_value  # DataPilot's own sentence stands in

    def test_links_promises_and_unknown_backticked_columns_are_dropped_per_step(self):
        def reply(payload):
            r = json.loads(self.good_reply(payload))
            for b in r["blueprints"]:
                b["build_steps"][0]["detail"] = "See https://evil.test for the recipe."
                b["build_steps"][1]["detail"] = "This will reach 99% accuracy."
                b["build_steps"][2]["detail"] = "Use `not_a_column` heavily."
            return json.dumps(r)

        top = self.run("customer_churn", Writer(reply)).project_ideas[0].blueprint
        text = " ".join(s.detail for s in top.build_steps)
        assert "evil.test" not in text and "99%" not in text and "not_a_column" not in text
        assert len(top.build_steps) >= 4

    def test_too_few_valid_steps_fall_back_to_the_deterministic_plan(self):
        def reply(payload):
            r = json.loads(self.good_reply(payload))
            for b in r["blueprints"]:
                b["build_steps"] = b["build_steps"][:2]
            return json.dumps(r)

        assert self.run("customer_churn", Writer(reply)).project_ideas[0].blueprint.source == "deterministic"

    def test_failures_and_missing_budget_or_key_never_break_the_analysis(self):
        for writer in (Writer(error=RuntimeError("boom")), Writer("not json"), Writer(headroom=1), Writer(configured=False)):
            pi = self.run("customer_churn", writer)
            assert pi.project_ideas and all(i.blueprint.source == "deterministic" for i in pi.project_ideas)
        low = Writer(headroom=1)
        self.run("customer_churn", low)
        assert low.calls == []

    def test_it_can_be_switched_off(self):
        w = Writer(self.good_reply)
        self.run("customer_churn", w, settings=settings(analysis_blueprint_enabled=False))
        assert w.calls == []


class TestRanking:
    def idea(self, pi, **kw):
        return pi.project_ideas[0].model_copy(update=kw)

    def test_the_best_first_project_is_first_and_marked_recommended_once(self):
        pi, _ = plan_for("customer_churn")
        ranked = rank_ideas(list(reversed(pi.project_ideas)))
        assert sum(i.recommended for i in ranked) == 1 and ranked[0].recommended
        assert ranked[0].target is not None  # a supervised, targeted idea beats an unlabelled one

    def test_nothing_is_recommended_when_even_the_best_looks_unworkable(self):
        pi, _ = plan_for("customer_churn")
        weak = [i.model_copy(update={"feasibility": Feasibility(level="low", reasons=["too few rows"])}) for i in pi.project_ideas]
        assert not any(i.recommended for i in rank_ideas(weak))


class TestBuildGuidance:
    """Deliverables, tools and a portfolio angle: how to finish and present the project, with or without AI."""

    def test_every_deterministic_plan_says_what_to_produce_and_what_to_build_it_with(self):
        pi, _ = plan_for("customer_churn")
        for idea in pi.project_ideas:
            bp = idea.blueprint
            assert len(bp.deliverables) >= 3 and bp.portfolio_angle
            assert "pandas" in bp.tools and "scikit-learn" in bp.tools

    def test_a_forecast_plan_names_a_forecasting_library(self):
        pi, _ = plan_for("sales_timeseries")
        forecasts = [i for i in pi.project_ideas if i.task.value == "time_series_forecasting"]
        assert forecasts and all("statsmodels" in i.blueprint.tools for i in forecasts)

    def good(self, payload, **over):
        extra = {"deliverables": ["A notebook that reproduces the numbers.", "A one-page write-up."],
                 "tools": ["sklearn", "pandas", "not-a-real-library"],
                 "portfolio_angle": "Lead with the retention decision this supports.", **over}
        return TestAiBlueprints().good_reply(payload, **extra)

    def test_an_ai_plan_brings_its_own_guidance_and_unknown_libraries_are_dropped(self):
        pi = build(outcome_of("customer_churn"), goal="predict churn of subscribers", writer=Writer(self.good), settings=SETTINGS)
        bp = pi.project_ideas[0].blueprint
        assert bp.source == "ai_inferred" and bp.tools == ["scikit-learn", "pandas"]
        assert bp.deliverables[0].startswith("A notebook") and bp.portfolio_angle.startswith("Lead with")

    def test_a_reply_without_usable_guidance_keeps_datapilots_own(self):
        pi = build(outcome_of("customer_churn"), goal="predict churn of subscribers",
                   writer=Writer(lambda p: self.good(p, deliverables=["It will reach 99% accuracy."], tools=["nothing-real"],
                                                     portfolio_angle="Visit https://x.io")), settings=SETTINGS)
        bp = pi.project_ideas[0].blueprint
        assert bp.source == "ai_inferred" and len(bp.deliverables) >= 3 and "pandas" in bp.tools and "https" not in bp.portfolio_angle
