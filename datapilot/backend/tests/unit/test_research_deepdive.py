"""In-depth analysis of papers and algorithms: grounded paper findings, algorithm profiles, and the approach."""

from __future__ import annotations

import asyncio
import json

from app.schemas.research import (
    AlgorithmRecommendation,
    EvidenceKind,
    Level,
    Paper,
    ReportedMetric,
    ResearchDataset,
    ResearchRequest,
    ResearchStatus,
)
from app.services.research import ResearchService
from app.services.research.deepdive import (
    deterministic_algorithm_analysis,
    deterministic_approach,
    evidence_text,
    supports,
    validate_algorithms,
    validate_papers,
)
from tests.unit.test_research import FakeProvider, hit, settings

ABSTRACT = ("We predict hourly traffic volume on the I-94 highway from weather and time features. "
            "Random forest demonstrated the highest accuracy among the tested models, reaching an accuracy of 93.2% on held-out days. "
            "The current models lack sufficient environmental constraints to improve performance.")


def paper(pid="p1", text=ABSTRACT, **kw) -> Paper:
    base = dict(id=pid, title="Traffic volume prediction", url="https://arxiv.org/abs/2101.00001", source="arxiv.org", abstract_snippet=text)
    base.update(kw)
    return Paper(**base)


def entry(**kw):
    base = {"id": "p1", "problem": "Predicting hourly traffic volume.", "data_used": "I-94 highway data.", "approach": "Compared several regression models.",
            "findings": [{"statement": "Random forest was the most accurate model.", "quote": "Random forest demonstrated the highest accuracy among the tested models"}],
            "limitations": [{"statement": "The models lack environmental constraints.", "quote": "The current models lack sufficient environmental constraints to improve performance"}],
            "relevance": "Shows which models to compare for your traffic project."}
    base.update(kw)
    return base


def papers_reply(*entries):
    return json.dumps({"papers": list(entries)})


class TestEvidenceAndSupport:
    def test_the_evidence_is_the_abstract_plus_the_cue_sentences_of_the_full_text(self):
        full = "Intro text that says little. " * 5 + "We collected data from four junctions over two years of operation. Results show that boosting beats the baseline clearly."
        text = evidence_text(paper(), full)
        assert ABSTRACT[:40] in text and "collected data from four junctions" in text and "Intro text" not in text

    def test_a_quote_must_back_the_statement_it_is_attached_to(self):
        assert supports("Random forest demonstrated the highest accuracy.", "RF demonstrated the highest accuracy, in these tests")
        assert not supports("The trigonometric model outperformed the others.", "Several studies were carried out evaluating different generations of models")


class TestPaperValidation:
    def run(self, reply, text=ABSTRACT, basis="abstract"):
        return validate_papers(reply, {"p1": text}, {"p1": basis})

    def test_a_verified_analysis_is_kept_with_its_quotes_and_basis(self):
        a = self.run(papers_reply(entry()))["p1"]
        assert a.problem and a.data_used and len(a.findings) == 1 and len(a.limitations) == 1 and a.basis == "abstract" and a.source == "ai_inferred"
        assert a.findings[0].quote in ABSTRACT

    def test_an_invented_quote_or_a_real_but_unrelated_one_is_dropped(self):
        fake = entry(findings=[{"statement": "The model reached great results.", "quote": "This sentence is not in the paper at all, honestly."},
                               {"statement": "The trigonometric model won.", "quote": "We predict hourly traffic volume on the I-94 highway"}])
        a = self.run(papers_reply(fake))["p1"]
        assert a.findings == []

    def test_a_number_must_appear_in_its_quote(self):
        good = entry(findings=[{"statement": "It reached an accuracy of 93.2% on held-out days.", "quote": "reaching an accuracy of 93.2% on held-out days"}])
        bad = entry(findings=[{"statement": "Random forest reached an accuracy of 97.5%.", "quote": "Random forest demonstrated the highest accuracy among the tested models"}])
        assert "93.2%" in self.run(papers_reply(good))["p1"].findings[0].statement
        assert self.run(papers_reply(bad))["p1"].findings == []

    def test_links_markdown_unknown_ids_and_empty_analyses_are_refused(self):
        linked = entry(approach="See https://evil.test for details.", problem="## A heading")
        a = self.run(papers_reply(linked))["p1"]
        assert a.approach is None and a.problem is None and a.findings  # the verifiable parts survive
        nothing = entry(problem=None, approach=None, findings=[])
        assert self.run(papers_reply(nothing)) == {} and self.run(papers_reply(entry(id="ghost"))) == {}
        assert self.run("not json") == {}

    def test_the_basis_says_when_full_text_was_read(self):
        assert self.run(papers_reply(entry()), basis="full_text")["p1"].basis == "full_text"


def rec(method="Random Forest", ids=("p1",), lit=Level.STRONG, fit=Level.STRONG):
    return AlgorithmRecommendation(rank=1, method=method, literature_support=lit, dataset_compatibility=fit,
                                   evidence_paper_count=len(ids), paper_ids=list(ids), why="w")


def dataset(rows=3000):
    return ResearchDataset(name="t", columns=["temp_c", "rain_mm", "hour", "traffic_volume"], rows=rows, target="traffic_volume")


class TestDeterministicAlgorithms:
    def test_every_algorithm_gets_a_real_profile_and_notes_on_how_the_papers_used_it(self):
        metric = ReportedMetric(metric="accuracy", value="93.2%", method="Random Forest", paper_id="p1", paper_title="t", url="https://x.org",
                                statement="s", evidence=EvidenceKind.WEB_VERIFIED)
        a = deterministic_algorithm_analysis(rec(), "regression", dataset(), {"p1": paper()}, [metric])
        assert a.source == "datapilot_knowledge" and not a.ai_enriched
        assert len(a.strengths) >= 3 and len(a.weaknesses) >= 3 and a.preprocessing and a.tuning and "`n_estimators`" in a.tuning[0]
        assert a.how_literature_used_it[0].paper_id == "p1" and "93.2%" in a.how_literature_used_it[0].note and "its own setting" in a.how_literature_used_it[0].note

    def test_the_fit_reflects_the_dataset_size_and_the_kind_of_method(self):
        small = deterministic_algorithm_analysis(rec(), "regression", dataset(300), {}, []).fit_for_your_data
        deep = deterministic_algorithm_analysis(rec("Deep Learning"), "regression", dataset(300), {}, []).fit_for_your_data
        arima = deterministic_algorithm_analysis(rec("ARIMA"), "forecasting", dataset(), {}, []).fit_for_your_data
        assert "overfit" in small and "300 rows" in small and "overfit" in deep and "time column" in arima

    def test_an_unknown_method_is_labelled_not_invented(self):
        a = deterministic_algorithm_analysis(rec("Quantum Annealer"), "regression", dataset(), {}, [])
        assert not a.strengths and "no built-in profile" in a.fit_for_your_data


class TestApproach:
    def test_baseline_primary_and_stretch_follow_task_data_and_family(self):
        recs = [rec("Deep Learning"), rec("Random Forest"), rec("XGBoost")]
        ap = deterministic_approach(recs, "regression", dataset(500))
        assert ap.baseline == "Linear Regression" and ap.primary == "Random Forest"  # a deep model is not the main one on 500 rows
        assert ap.stretch and ap.stretch != ap.primary and any("500 rows" in c for c in ap.caveats) and len(ap.steps) >= 4
        assert deterministic_approach([], "regression", None) is None

    def test_forecasting_starts_from_a_seasonal_baseline(self):
        assert deterministic_approach([rec("ARIMA")], "forecasting", dataset()).baseline == "ARIMA"


class TestAlgorithmValidation:
    def setup_method(self):
        self.recs = [rec("Random Forest", ("p1",)), rec("Linear Regression", ("p2",))]
        self.base = {r.method: deterministic_algorithm_analysis(r, "regression", dataset(), {}, []) for r in self.recs}
        self.evidence = {"p1": ABSTRACT, "p2": "We used simple linear regression as a baseline."}
        self.cols = {"temp_c", "rain_mm", "hour", "traffic_volume"}

    def reply(self, alg=None, approach=None):
        a = {"method": "Random Forest", "fit_for_your_data": "With 3000 rows it can model `hour` and `rain_mm` interactions.",
             "how_literature_used_it": [{"paper_id": "p1", "note": "Reached an accuracy of 93.2% on held-out days."}],
             "strengths": ["Handles interactions between `hour` and `temp_c`", "Needs little tuning"], "weaknesses": ["Cannot extrapolate", "Large models"],
             "preprocessing": ["Encode categorical columns", "Derive hour features"], "tuning": ["`n_estimators`", "`max_features`"]}
        a.update(alg or {})
        ap = {"summary": "Start linear, then a forest.", "baseline": "Linear Regression", "primary": "Random Forest", "stretch": None,
              "steps": ["Clean the data.", "Fit the baseline.", "Fit the forest."], "caveats": ["Paper figures do not transfer."]}
        ap.update(approach or {})
        return json.dumps({"algorithms": [a], "approach": ap})

    def run(self, reply):
        return validate_algorithms(reply, self.recs, self.base, self.evidence, self.cols)

    def test_a_valid_reply_enriches_the_profile_and_gives_an_ai_approach(self):
        out, ap = self.run(self.reply())
        a = out["Random Forest"]
        assert a.ai_enriched and a.source == "ai_inferred" and a.what_it_is == self.base["Random Forest"].what_it_is
        assert "93.2%" in a.how_literature_used_it[0].note and ap.source == "ai_inferred" and ap.primary == "Random Forest"

    def test_paper_references_numbers_and_columns_are_checked(self):
        wrong_paper = {"how_literature_used_it": [{"paper_id": "p2", "note": "Used it."}]}  # p2 does not mention Random Forest
        invented_number = {"how_literature_used_it": [{"paper_id": "p1", "note": "Reached an accuracy of 99.9% on held-out days."}]}
        unknown_column = {"fit_for_your_data": "It would use `not_a_column` heavily."}
        for change in (wrong_paper, invented_number):
            a = self.run(self.reply(change))[0]["Random Forest"]
            assert a.how_literature_used_it == self.base["Random Forest"].how_literature_used_it
        assert self.run(self.reply(unknown_column))[0]["Random Forest"].fit_for_your_data == self.base["Random Forest"].fit_for_your_data

    def test_unknown_methods_and_a_bad_approach_are_ignored(self):
        out, ap = self.run(self.reply({"method": "Made Up Method"}, {"primary": "Also Made Up"}))
        assert out == {} and ap is None
        assert self.run("not json") == ({}, None)

    def test_too_few_list_items_keep_the_knowledge_base_text(self):
        out, _ = self.run(self.reply({"strengths": ["Only one."], "weaknesses": [], "tuning": []}))
        a = out["Random Forest"]
        assert a.strengths == self.base["Random Forest"].strengths and a.tuning == self.base["Random Forest"].tuning


class Writer:
    """Answers each AI step by recognising its system prompt, building replies from the payload it is sent."""

    def __init__(self, papers=True, algorithms=True, headroom=50):
        self.papers, self.algorithms, self.headroom, self.calls = papers, algorithms, headroom, []

    is_configured = True

    async def generate_json_text(self, system, user, max_tokens=1024):
        payload = json.loads(user.split("<data>\n")[1].split("\n</data>")[0])
        self.calls.append(system[:30])
        if "analyse scientific papers" in system:
            if not self.papers:
                raise RuntimeError("down")
            out = []
            for p in payload["papers"]:
                sentence = p["text"].split(". ")[0].strip(".")
                out.append(entry(id=p["id"], findings=[{"statement": sentence[:120], "quote": sentence[:120]}], limitations=[]))
            return papers_reply(*out)
        if "advise a student choosing algorithms" in system:
            if not self.algorithms:
                raise RuntimeError("down")
            algs = [{"method": a["method"], "fit_for_your_data": "It suits this dataset well.", "how_literature_used_it": [],
                     "strengths": ["Fast to train", "Little tuning"], "weaknesses": ["Less interpretable", "Memory heavy"],
                     "preprocessing": ["Encode categories", "Handle gaps"], "tuning": ["Depth", "Learning rate"]} for a in payload["algorithms"]]
            names = [a["method"] for a in payload["algorithms"]]
            return json.dumps({"algorithms": algs, "approach": {"summary": "Baseline then the best model.", "baseline": "Linear Regression",
                                                                "primary": names[0], "stretch": None, "steps": ["One.", "Two.", "Three."], "caveats": []}})
        if "judge whether scientific papers" in system:
            return json.dumps({"scores": [{"id": p["id"], "relevance": 3, "reason": "on topic"} for p in payload["papers"]]})
        return json.dumps({"problem": "traffic prediction", "core_terms": [{"term": "traffic", "synonyms": []}], "context_terms": ["weather"],
                           "queries": [], "clarification": None, "dataset_check": None})


HITS = [
    hit("Traffic volume prediction with weather using random forest", "https://arxiv.org/abs/2101.00001",
        "Random forest predicts hourly traffic volume with weather features. It reached an accuracy of 93.2% on held-out days."),
    hit("Short-term traffic flow forecasting with XGBoost", "https://arxiv.org/abs/2102.00002",
        "XGBoost for traffic flow forecasting under rain and snow, compared with linear regression as a baseline."),
]


def discover(writer, **s):
    req = ResearchRequest.model_validate({"project": {"description": "predict traffic volume with weather", "task": "regression", "target": "traffic_volume"},
                                          "dataset": {"name": "t", "columns": ["temp_c", "rain_mm", "hour", "traffic_volume"], "rows": 3000},
                                          "options": {"max_papers": 5}})
    return asyncio.run(ResearchService(settings(**s), FakeProvider(HITS), writer).discover(req))


class TestService:
    def test_the_ai_analysis_is_attached_and_reported(self):
        body = discover(Writer())
        assert body.status is ResearchStatus.SUCCESS
        assert {"paper_analysis", "algorithm_analysis"} <= set(body.gemini.assisted)
        assert all(p.analysis and p.analysis.findings for p in body.papers)
        assert all(r.analysis and r.analysis.ai_enriched for r in body.algorithm_recommendations)
        assert body.approach.source == "ai_inferred" and body.approach.primary in {r.method for r in body.algorithm_recommendations}

    def test_without_ai_every_algorithm_still_gets_dataminer_knowledge_and_an_approach(self):
        body = discover(Writer(papers=False, algorithms=False))
        assert all(p.analysis is None for p in body.papers)
        assert all(r.analysis and r.analysis.source == "datapilot_knowledge" and r.analysis.strengths for r in body.algorithm_recommendations)
        assert body.approach and body.approach.source == "datapilot_knowledge"
        assert any("in-depth analysis was unavailable" in l for l in body.limitations)

    def test_it_can_be_switched_off_or_skipped_for_budget(self):
        off = Writer()
        body = discover(off, research_deep_analysis_enabled=False)
        assert not any("analyse scientific" in c or "advise a student" in c for c in off.calls) and body.approach
        low = Writer(headroom=1)
        discover(low)
        assert low.calls == []

    def test_a_partial_failure_keeps_what_worked(self):
        body = discover(Writer(papers=False))
        assert "algorithm_analysis" in body.gemini.assisted and "paper_analysis" not in body.gemini.assisted
        assert all(p.analysis is None for p in body.papers) and body.approach.source == "ai_inferred"
