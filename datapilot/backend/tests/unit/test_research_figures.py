"""AI figure reading: one batched call reads the accuracy/F1/... each paper states, and code checks every figure.

No test here makes a real Gemini call. The writer is a fake that recognises the figure-reading system prompt and answers
from the payload it was sent.
"""

from __future__ import annotations

import asyncio
import json

from app.schemas.research import EvidenceKind, Paper, ReportedMetric, ResearchRequest, ResearchStatus
from app.services.research import ResearchService, figures
from tests.unit.test_research import FakeProvider, hit, settings

TEXT = ("We compared several models for traffic volume. The Random Forest reached 93.4% accuracy on the PeMS dataset. "
        "XGBoost obtained an F1 of 0.912 and the LSTM an RMSE of 4.37 veh/h. Accuracy improved by 3% after tuning.")


def paper(pid="p1", abstract=TEXT) -> Paper:
    return Paper(id=pid, title="Traffic volume prediction", url="https://arxiv.org/abs/2101.00001", abstract_snippet=abstract,
                 source="arxiv.org", venue="arXiv", relevance_score=80, relevance_reason="x")


def fig(**kw):
    base = {"method": "Random Forest", "metric": "accuracy", "value": "93.4%", "setting": "PeMS dataset",
            "quote": "The Random Forest reached 93.4% accuracy on the PeMS dataset."}
    base.update(kw)
    return base


def reply(*items, pid="p1"):
    return json.dumps({"papers": [{"id": pid, "figures": list(items)}]})


def validate(r, existing=None, abstract=TEXT):
    p = paper(abstract=abstract)
    evidence = {p.id: figures.evidence_text(p, None)}
    return figures.validate_figures(r, evidence, {p.id: p}, existing or [], {p.id: "abstract"})


class TestValidation:
    def test_a_grounded_figure_is_kept_and_labelled_as_ai_read(self):
        (m,) = validate(reply(fig()))
        assert (m.metric, m.value, m.method, m.context) == ("accuracy", "93.4%", "Random Forest", "PeMS dataset")
        assert m.evidence is EvidenceKind.AI_INFERRED and m.quote.startswith("The Random Forest reached")
        assert "checked against that text" in m.statement and "not a prediction" in m.statement

    def test_several_metrics_from_one_paper(self):
        out = validate(reply(fig(), fig(method="XGBoost", metric="F1", value="0.912", setting=None,
                                        quote="XGBoost obtained an F1 of 0.912"),
                             fig(method="LSTM", metric="RMSE", value="4.37", setting=None, quote="the LSTM an RMSE of 4.37 veh/h")))
        assert [(m.method, m.metric, m.value) for m in out] == [("Random Forest", "accuracy", "93.4%"), ("XGBoost", "F1", "0.912"), ("LSTM", "RMSE", "4.37")]

    def test_a_quote_that_is_not_in_the_text_is_rejected(self):
        assert validate(reply(fig(quote="The Random Forest reached 99.9% accuracy on the PeMS dataset."))) == []

    def test_a_value_that_is_not_in_its_quote_is_rejected(self):
        assert validate(reply(fig(value="95%"))) == []

    def test_the_quote_must_name_the_metric(self):
        assert validate(reply(fig(metric="F1"))) == []

    def test_an_unknown_metric_or_a_percentage_over_100_is_rejected(self):
        assert validate(reply(fig(metric="vibes"))) == []
        text = "The model reached 250 accuracy on the data."
        assert validate(reply(fig(value="250", quote="reached 250 accuracy")), abstract=text) == []

    def test_a_figure_the_regex_already_found_is_not_repeated(self):
        known = ReportedMetric(metric="accuracy", value="93.4 %", paper_id="p1", paper_title="t", url="u", statement="s")
        assert validate(reply(fig()), existing=[known]) == []

    def test_a_method_or_setting_absent_from_the_text_is_dropped_but_the_figure_stays(self):
        (m,) = validate(reply(fig(method="QuantumBoost", setting="Mars dataset")))
        assert m.value == "93.4%" and m.method == "Random Forest"  # recovered from the quote
        assert m.context is None

    def test_a_method_is_given_its_catalogue_name(self):
        text = "The random forests classifier reached 90.1% accuracy."
        (m,) = validate(reply(fig(method="random forests classifier", value="90.1%", setting=None, quote="reached 90.1% accuracy")), abstract=text)
        assert m.method == "Random Forest"

    def test_links_and_junk_entries_are_ignored_and_nothing_raises(self):
        assert validate("not json") == [] and validate(json.dumps({"papers": "x"})) == []
        assert validate(reply("a string", None, fig(quote="see https://x.example 93.4% accuracy"))) == []
        assert validate(reply(fig(), pid="unknown")) == []

    def test_at_most_six_figures_per_paper(self):
        nums = " ".join(f"Model{i} reached {80 + i}.5% accuracy." for i in range(9))
        items = [fig(method=f"Model{i}", value=f"{80 + i}.5%", setting=None, quote=f"Model{i} reached {80 + i}.5% accuracy.") for i in range(9)]
        assert len(validate(reply(*items), abstract=nums)) == figures.MAX_PER_PAPER


class FigureWriter:
    is_configured, headroom = True, 50

    def __init__(self, reply_for=None, error=None):
        self.reply_for, self.error, self.calls = reply_for, error, []

    async def generate_json_text(self, system, user, max_tokens=1024):
        self.calls.append(system[:40])
        if self.error:
            raise self.error
        if "performance figures" in system:
            payload = json.loads(user.split("<data>\n")[1].split("\n</data>")[0])
            return self.reply_for(payload)
        raise RuntimeError("not used in this test")


class TestExtraction:
    def test_one_call_reads_every_paper_and_returns_checked_figures(self):
        papers = [paper("p1"), paper("p2")]
        w = FigureWriter(lambda payload: json.dumps({"papers": [{"id": p["id"], "figures": [fig()]} for p in payload["papers"]]}))
        out, why = asyncio.run(figures.extract_figures(w, settings(), papers, {}, []))
        assert len(w.calls) == 1 and why is None and {m.paper_id for m in out} == {"p1", "p2"}

    def test_full_text_sentences_with_numbers_are_shown_and_checked(self):
        p = paper(abstract="A short abstract with no results.")
        full = "Intro text here. In Table 2 the Random Forest reached 88.2% accuracy on the held-out set. More text."
        evidence = figures.evidence_text(p, full)
        assert "88.2% accuracy" in evidence and "Intro text" not in evidence

    def test_a_failing_writer_adds_nothing(self):
        out, why = asyncio.run(figures.extract_figures(FigureWriter(error=RuntimeError("down")), settings(), [paper()], {}, []))
        assert out == [] and "unavailable" in why

    def test_no_key_and_no_numbers_make_no_call(self):
        class Off(FigureWriter):
            is_configured = False

        w = Off(lambda p: "{}")
        assert asyncio.run(figures.extract_figures(w, settings(), [paper()], {}, [])) == ([], "no AI key is configured") and w.calls == []
        w2 = FigureWriter(lambda p: "{}")
        out, why = asyncio.run(figures.extract_figures(w2, settings(), [paper(abstract="Words only, no digits at all.")], {}, []))
        assert out == [] and w2.calls == []


class TestInService:
    HITS = [hit("Traffic volume prediction using weather and time features", "https://arxiv.org/abs/2101.00001",
                "Random forest and LSTM predict hourly traffic volume from weather. The random forest reached 93.4% accuracy on the PeMS dataset."),
            hit("Short-term traffic flow forecasting with gradient boosting", "https://arxiv.org/abs/2102.00002",
                "XGBoost for traffic flow forecasting under rain; XGBoost obtained an F1 of 0.912 in tests.")]

    def request(self):
        return ResearchRequest.model_validate({"project": {"description": "predict traffic during bad weather", "task": "classification"},
                                               "options": {"max_papers": 8, "ai_assist": True}})

    def run(self, writer, **s):
        return asyncio.run(ResearchService(settings(**s), FakeProvider(self.HITS), writer).discover(self.request()))

    def writer(self):
        def make(payload):
            out = []
            for p in payload["papers"]:
                text = p["text"]
                items = []
                if "93.4% accuracy" in text:
                    items.append(fig())
                if "F1 of 0.912" in text:
                    items.append(fig(method="XGBoost", metric="F1", value="0.912", setting=None, quote="XGBoost obtained an F1 of 0.912 in tests"))
                out.append({"id": p["id"], "figures": items})
            return json.dumps({"papers": out})
        return FigureWriter(make)

    def test_the_figures_reach_the_results_table_and_are_counted(self):
        body = self.run(self.writer())
        assert body.status is ResearchStatus.SUCCESS and "figures" in body.gemini.assisted
        ai = [m for m in body.benchmarks if m.evidence is EvidenceKind.AI_INFERRED]
        # "93.4% accuracy" is written value-first, which the regex cannot read; "F1 of 0.912" it already finds, so it is not repeated
        assert {(m.method, m.metric, m.value) for m in ai} == {("Random Forest", "accuracy", "93.4%")}
        assert all(m.quote for m in ai)
        assert [(m.metric, m.value) for m in body.benchmarks if m.metric == "F1"] == [("F1", "0.912")]

    def test_the_switch_turns_it_off_and_nothing_else_changes(self):
        w = self.writer()
        body = self.run(w, research_figures_enabled=False)
        assert "figures" not in body.gemini.assisted and not any(m.evidence is EvidenceKind.AI_INFERRED for m in body.benchmarks)
        assert body.status is ResearchStatus.SUCCESS and body.papers

    def test_a_failed_reading_leaves_the_regex_figures_untouched(self):
        with_ai = self.run(FigureWriter(error=RuntimeError("down")))
        without = self.run(None)
        assert [m.model_dump() for m in with_ai.benchmarks] == [m.model_dump() for m in without.benchmarks]
