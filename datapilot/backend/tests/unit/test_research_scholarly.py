"""Scholarly providers (OpenAlex, arXiv, Crossref) and the two-tier search.

Real structured metadata is kept with its evidence label; a missing field stays missing; web search is only a
fallback; one failing source never hides the others.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.schemas.research import EvidenceKind, ResearchStatus
from app.services.research import ResearchService
from app.services.research.methods import extract_metrics
from app.services.research.normalize import normalise
from app.services.research.providers import ProviderError, SearchHit
from app.services.research.scholarly import (
    ArxivProvider,
    CrossrefProvider,
    OpenAlexProvider,
    abstract_from_index,
    query_words,
)
from tests.unit.test_research import FakeProvider, hit, request, settings


def mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


OPENALEX = {"results": [
    {"title": "Student Academic Performance Prediction with XGBoost", "publication_year": 2024,
     "doi": "https://doi.org/10.1000/abc", "cited_by_count": 17,
     "authorships": [{"author": {"display_name": "Ada Lovelace"}}, {"author": {"display_name": "Alan Turing"}}],
     "abstract_inverted_index": {"We": [0], "compare": [1], "XGBoost": [2], "reaching": [3], "accuracy": [4],
                                 "of": [5], "92.5%": [6]},
     "primary_location": {"landing_page_url": "https://example.edu/p", "source": {"display_name": "Journal of Learning Analytics"}}},
    {"title": "", "doi": None},
]}
ARXIV = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2301.01234v1</id><published>2023-01-05T00:00:00Z</published><title> Student  Performance
 with Random Forest </title><summary>Random Forest for students.</summary><author><name>Grace Hopper</name></author>
<arxiv:doi>10.1000/xyz</arxiv:doi></entry></feed>"""
CROSSREF = {"message": {"items": [{
    "title": ["Predicting grades"], "DOI": "10.1000/q", "URL": "https://doi.org/10.1000/q",
    "issued": {"date-parts": [[2021, 3]]}, "container-title": ["Computers & Education"],
    "author": [{"given": "Katherine", "family": "Johnson"}], "abstract": "<jats:p>Uses SVM.</jats:p>",
    "is-referenced-by-count": 40}]}}


class TestProviders:
    def test_openalex_returns_real_structured_metadata(self):
        client = mock_client(lambda r: httpx.Response(200, json=OPENALEX))
        hits = asyncio.run(OpenAlexProvider(settings(), client).search('"student academic performance" classification', 5))
        assert len(hits) == 1  # the record without a title is dropped, never invented
        h = hits[0]
        assert h.authors == ("Ada Lovelace", "Alan Turing") and h.year == 2024 and h.citations == 17 and h.academic
        assert h.doi == "10.1000/abc" and h.venue == "Journal of Learning Analytics" and "92.5%" in h.snippet

    def test_arxiv_atom_is_parsed(self):
        client = mock_client(lambda r: httpx.Response(200, text=ARXIV))
        h = asyncio.run(ArxivProvider(settings(), client).search("student performance prediction", 5))[0]
        assert h.url == "https://arxiv.org/abs/2301.01234v1" and h.year == 2023 and h.authors == ("Grace Hopper",)
        assert h.title == "Student Performance with Random Forest" and h.doi == "10.1000/xyz"

    def test_crossref_is_parsed_and_markup_is_stripped(self):
        client = mock_client(lambda r: httpx.Response(200, json=CROSSREF))
        h = asyncio.run(CrossrefProvider(settings(), client).search("predicting grades", 5))[0]
        assert h.year == 2021 and h.authors == ("Katherine Johnson",) and h.snippet == "Uses SVM." and h.citations == 40

    @pytest.mark.parametrize("cls", [OpenAlexProvider, ArxivProvider, CrossrefProvider])
    def test_failures_are_typed_errors(self, cls):
        for status, code in ((429, "RATE_LIMITED"), (503, "UNAVAILABLE")):
            with pytest.raises(ProviderError) as e:
                asyncio.run(cls(settings(), mock_client(lambda r, s=status: httpx.Response(s))).search("student performance", 3))
            assert e.value.code == code

    def test_garbage_xml_is_an_invalid_response(self):
        client = mock_client(lambda r: httpx.Response(200, text="<not xml"))
        with pytest.raises(ProviderError) as e:
            asyncio.run(ArxivProvider(settings(), client).search("student performance", 3))
        assert e.value.code == "INVALID_RESPONSE"

    def test_oversized_responses_are_refused(self):
        client = mock_client(lambda r: httpx.Response(200, content=b"x" * 200))
        with pytest.raises(ProviderError):
            asyncio.run(OpenAlexProvider(settings(research_max_response_bytes=100), client).search("student performance", 3))

    def test_abstract_reconstruction_and_query_words(self):
        assert abstract_from_index({"b": [1], "a": [0]}) == "a b" and abstract_from_index(None) == ""
        assert query_words('"student performance" review OR survey') == ["student", "performance", "review", "survey"]


class TestNormalisationKeepsStructuredMetadata:
    def hit(self, **kw):
        base = dict(title="Same Paper", url="https://doi.org/10.1/a", snippet="Uses XGBoost, accuracy of 90.1%.", query="q1",
                    provider="openalex", authors=("A. One",), year=2024, doi="10.1/a", venue="Journal X", citations=9,
                    academic=True)
        base.update(kw)
        return SearchHit(**base)

    def test_metadata_is_kept_with_its_evidence(self):
        paper = normalise([self.hit()])[0][0]
        assert paper.authors == ["A. One"] and paper.publication_year == 2024
        assert paper.year_evidence is EvidenceKind.SOURCE_METADATA
        assert paper.doi == "10.1/a" and paper.venue == "Journal X" and paper.citation_count == 9

    def test_scholarly_hits_skip_the_host_allowlist_but_web_hits_do_not(self):
        scholarly = self.hit(url="https://some-journal.example/paper")
        web = SearchHit(title="X", url="https://some-journal.example/paper", snippet="", query="q", provider="web_search")
        assert len(normalise([scholarly])[0]) == 1 and normalise([web])[1] == 1

    def test_one_paper_from_two_providers_merges_and_keeps_the_richest_metadata(self):
        a = self.hit(provider="openalex")
        b = self.hit(provider="crossref", authors=None, year=None, venue=None, citations=30, snippet="", query="q2")
        papers, _, merged = normalise([a, b])
        assert len(papers) == 1 and merged == 1
        assert papers[0].authors == ["A. One"] and papers[0].citation_count == 30
        assert papers[0].matched_queries == ["q1", "q2"]

    def test_a_bare_number_is_not_a_metric(self):
        paper = normalise([self.hit(snippet="The accuracy 70 students; accuracy of 0.93 on test; F1 of 88%")])[0][0]
        values = {(m.metric, m.value) for m in extract_metrics([paper])}
        assert ("accuracy", "70") not in values and ("accuracy", "0.93") in values and ("F1", "88%") in values


class Primary(FakeProvider):
    fallback = False


def papers(n):
    return [hit(f"Student Academic Performance Study {i}", f"https://arxiv.org/abs/2401.{i:05d}",
                "Random Forest for student academic performance classification.") for i in range(n)]


class TestTieredSearch:
    def service(self, primary, web):
        web.fallback = True
        return ResearchService(settings(), providers=[primary, web])

    def test_web_search_is_not_used_when_scholarly_sources_deliver(self):
        primary, web = Primary(hits=papers(6)), FakeProvider(hits=[])
        r = asyncio.run(self.service(primary, web).discover(request()))
        assert r.status is ResearchStatus.SUCCESS and web.calls == []

    def test_web_search_fills_in_when_scholarly_sources_return_too_little(self):
        extra = papers(5)[1:] + [hit("Another Student Academic Performance Paper", "https://www.nature.com/articles/x",
                                     "student academic performance")]
        primary, web = Primary(hits=papers(1)), FakeProvider(hits=extra)
        r = asyncio.run(self.service(primary, web).discover(request()))
        assert web.calls and len(r.papers) >= 3

    def test_one_failing_source_does_not_hide_the_others(self):
        bad, good = Primary(error=ProviderError("UNAVAILABLE", "down")), Primary(hits=papers(5))
        r = asyncio.run(ResearchService(settings(), providers=[bad, good]).discover(request()))
        assert r.status is ResearchStatus.SUCCESS and any("failed" in lim for lim in r.limitations)

    def test_only_when_every_source_fails_is_search_unavailable(self):
        err = ProviderError("TIMEOUT", "timed out")
        r = asyncio.run(ResearchService(settings(), providers=[Primary(error=err), Primary(error=err)]).discover(request()))
        assert r.status is ResearchStatus.SEARCH_UNAVAILABLE and r.papers == []

    def test_default_wiring_is_scholarly_first_with_web_as_fallback(self):
        names = [(p.name, getattr(p, "fallback", False)) for p in ResearchService(settings())._providers]
        assert names == [("openalex", False), ("semanticscholar", False), ("arxiv", False), ("crossref", False), ("web_search", True)]
