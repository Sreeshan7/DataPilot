"""Asking for a dataset by NAME inside a sentence: extraction, routing, the AI safety net, and how it reaches analysis."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.api.v1 import assist as assist_module
from app.core.config import Settings
from app.schemas.routing import IntentRequest, UserIntent
from app.services.dataset_analysis.name_ai import extract_name, validate
from app.services.dataset_analysis.references import extract_dataset_name, is_generic_name
from app.services.routing import IntentRouter
from tests.unit.test_assist_api import client, csv_file, stub  # noqa: F401  (fixtures)

URL = "/api/v1/assist"


class TestExtraction:
    @pytest.mark.parametrize("message,expected", [
        ("could you analuze this dataset for me LIAR Dataset", "LIAR"),
        ("analyze the LIAR dataset", "LIAR"),
        ("I want to check the Fake News Detection dataset", "Fake News Detection"),
        ("tell me about the CIFAR-10 dataset", "CIFAR-10"),
        ("can you look at the iris dataset", "iris"),
    ])
    def test_the_name_is_found_even_among_filler_and_repeated_mentions(self, message, expected):
        assert extract_dataset_name(message).name == expected

    def test_a_topic_word_is_not_a_name(self):
        assert is_generic_name("student") and is_generic_name("customer sales") and is_generic_name("")
        assert not is_generic_name("LIAR") and not is_generic_name("Palmer Penguins")


class TestRouting:
    def route(self, message):
        return IntentRouter().resolve(IntentRequest(message=message, has_file=False))

    @pytest.mark.parametrize("message", [
        "could you analuze this dataset for me LIAR Dataset", "analyze the LIAR dataset",
        "can you look at the iris dataset", "check out the Titanic data for me",
    ])
    def test_a_named_dataset_is_analysed_without_asking_for_an_upload(self, message):
        r = self.route(message)
        assert r.intent is UserIntent.USER_B_DATASET_ANALYSIS and r.required_next_input is None

    def test_a_topic_alone_still_asks_for_the_dataset(self):
        r = self.route("analyze the student dataset")
        assert r.intent is UserIntent.USER_B_DATASET_ANALYSIS and r.required_next_input


class TestAiNameCheck:
    def test_the_name_must_be_the_users_own_words(self):
        message = "analyse LIAR pls"
        assert validate(json.dumps({"dataset_name": "LIAR"}), message) == "LIAR"
        assert validate(json.dumps({"dataset_name": "Titanic"}), message) is None  # remembered, not written
        assert validate(json.dumps({"dataset_name": "LIAR Plus Extras"}), message) is None

    def test_topic_words_links_nulls_and_junk_are_refused(self):
        message = "analyse student data https://x.test pls"
        for reply in (json.dumps({"dataset_name": "student"}), json.dumps({"dataset_name": "https://x.test"}),
                      json.dumps({"dataset_name": None}), "not json", json.dumps(["LIAR"])):
            assert validate(reply, message) is None

    def test_it_makes_no_call_without_a_key_budget_or_permission(self):
        class W:
            calls = 0
            headroom = 50
            is_configured = True

            async def generate_json_text(self, system, user, max_tokens=1024):
                W.calls += 1
                return json.dumps({"dataset_name": "LIAR"})

        s = lambda **kw: Settings(gemini_api_key="", gemini_api_key_2="", app_env="testing", **kw)
        run = lambda w, **kw: asyncio.run(extract_name(w, s(**kw), "analyse LIAR pls"))
        assert run(W()) == "LIAR" and W.calls == 1
        low = W()
        low.headroom = 1
        assert run(low) is None and run(W(), analysis_ai_name_enabled=False) is None and run(None) is None
        off = W()
        off.is_configured = False
        assert run(off) is None and W.calls == 1


class TestAssistFlow:
    def test_an_unparseable_request_uses_the_ai_name_and_then_analyses_that_dataset(self, client, monkeypatch):
        seen = []

        async def fake(message):
            seen.append(message)
            return "Iris"

        monkeypatch.setattr(assist_module, "_name_from_message", fake)
        body = client.post(URL, data={"message": "analyse zzq pls"}).json()
        assert seen == ["analyse zzq pls"]
        assert body["status"] == "COMPLETED" and body["journey"] == "USER_B_DATASET_ANALYSIS"
        assert body["analysis"]["analysis"]["input_resolution"]["input_type"] == "public_dataset_name"

    def test_when_the_ai_finds_no_name_the_user_is_asked(self, client, monkeypatch):
        async def none(message):
            return None

        monkeypatch.setattr(assist_module, "_name_from_message", none)
        body = client.post(URL, data={"message": "analyse zzq pls"}).json()
        assert body["status"] in ("NEEDS_INPUT", "NEEDS_CLARIFICATION") and body["analysis"] is None

    def test_no_ai_name_call_when_a_file_or_a_project_is_given(self, client, monkeypatch):
        calls = []

        async def spy(message):
            calls.append(message)
            return "Iris"

        monkeypatch.setattr(assist_module, "_name_from_message", spy)
        client.post(URL, data={"message": "analyse this"}, files=csv_file())
        client.post(URL, data={"message": "I want to build a model that predicts churn", "project_goal": "predict churn of subscribers"})
        assert calls == []

    def test_the_stage_for_reading_the_name_is_reported(self, client, monkeypatch):
        async def fake(message):
            return "Iris"

        monkeypatch.setattr(assist_module, "_name_from_message", fake)
        text = client.post(URL + "/stream", data={"message": "analyse zzq pls"}).text
        assert '"id": "name"' in text and "Working out which dataset you mean" in text
