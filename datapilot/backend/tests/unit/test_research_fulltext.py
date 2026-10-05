"""Open-access full text: safe fetching, bounded extraction, and honest labelling. No real network is used."""

from __future__ import annotations

import asyncio

import httpx

from app.schemas.research import EvidenceKind, ResearchRequest
from app.services.research import ResearchService
from app.services.research.fulltext import extract_text, fetch_pdf, is_public_https
from app.services.research.providers import SearchHit
from tests.unit.test_research import FakeProvider, settings

PUBLIC = "93.184.216.34"


async def resolve(host: str) -> list[str]:
    return {"evil.test": ["127.0.0.1"], "internal.test": ["10.0.0.5"], "mixed.test": [PUBLIC, "192.168.1.9"]}.get(host, [PUBLIC])


def make_pdf(lines: list[str]) -> bytes:
    """A minimal valid single-page PDF whose text is ``lines`` (enough for a text extractor)."""
    ops = "BT /F1 9 Tf 12 TL 40 760 Td " + " ".join(f"({l.replace('(', '[').replace(')', ']')}) Tj T*" for l in lines) + " ET"
    objs = [
        "<</Type/Catalog/Pages 2 0 R>>", "<</Type/Pages/Kids[3 0 R]/Count 1>>",
        "<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        f"<</Length {len(ops)}>>\nstream\n{ops}\nendstream", "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offsets = "%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<</Root 1 0 R/Size {len(objs) + 1}>>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


BODY = [f"Section line {i} about traffic volume prediction with weather features and hourly counts." for i in range(40)]
PAPER = BODY + ["We trained XGBoost and a random forest on the traffic data.",
                "The best model reached an accuracy of 93.2% on the held-out set.", "References",
                "[1] A. Author, Support Vector Machine for unrelated stuff, 2001."]


def client_for(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestExtraction:
    def test_text_comes_out_and_the_reference_list_is_cut(self):
        text = extract_text(make_pdf(PAPER))
        assert "XGBoost" in text and "93.2%" in text
        assert "unrelated stuff" not in text  # references are not evidence of what this paper did

    def test_garbage_and_encrypted_input_gives_nothing(self):
        assert extract_text(b"%PDF-1.4 not really") == "" and extract_text(b"hello") == ""

    def test_output_is_bounded(self):
        assert len(extract_text(make_pdf(BODY * 30), max_chars=300)) <= 300


class TestUrlSafety:
    def test_only_public_https_hosts_pass(self):
        ok = lambda url: asyncio.run(is_public_https(url, resolve))
        assert ok("https://example.org/paper.pdf")
        for bad in ("http://example.org/p.pdf", "https://127.0.0.1/p.pdf", "https://localhost/p.pdf",
                    "https://user:pw@example.org/p.pdf", "https://example.org:8443/p.pdf", "https://intranet/p.pdf",
                    "https://internal.test/p.pdf", "https://evil.test/p.pdf", "https://mixed.test/p.pdf",
                    "https://[::1]/p.pdf", "ftp://example.org/p.pdf", "https://example.org/a b.pdf"):
            assert not ok(bad), bad


class TestFetch:
    def fetch(self, handler, url="https://example.org/p.pdf", max_bytes=1_000_000):
        async def run():
            async with client_for(handler) as client:
                return await fetch_pdf(client, url, max_bytes, resolve)
        return asyncio.run(run())

    def test_a_real_pdf_is_returned(self):
        pdf = make_pdf(PAPER)
        assert self.fetch(lambda r: httpx.Response(200, content=pdf)) == pdf

    def test_a_redirect_to_a_private_host_is_refused(self):
        def handler(request):
            if request.url.host == "example.org":
                return httpx.Response(302, headers={"location": "https://evil.test/secret.pdf"})
            raise AssertionError("the private host must never be contacted")
        assert self.fetch(handler) is None

    def test_a_safe_redirect_is_followed_and_a_loop_is_not(self):
        pdf = make_pdf(PAPER)
        def handler(request):
            return httpx.Response(302, headers={"location": "https://cdn.example.org/p.pdf"}) if request.url.host == "example.org" \
                else httpx.Response(200, content=pdf)
        assert self.fetch(handler) == pdf
        assert self.fetch(lambda r: httpx.Response(302, headers={"location": str(r.url)})) is None

    def test_non_pdf_oversize_and_errors_give_nothing(self):
        assert self.fetch(lambda r: httpx.Response(200, content=b"<html>login</html>")) is None
        assert self.fetch(lambda r: httpx.Response(404)) is None
        assert self.fetch(lambda r: httpx.Response(200, content=b"%PDF" + b"x" * 5000), max_bytes=1000) is None
        assert self.fetch(lambda r: httpx.Response(200, headers={"content-length": "999999999"}, content=b"%PDF")) is None


def paper_hit(oa_url="https://example.org/p.pdf"):
    return SearchHit(
        title="Traffic volume prediction with weather", url="https://doi.org/10.1/tv", query="q", provider="fake",
        snippet="Traffic volume prediction with weather and hourly features. We study the problem.", academic=True,
        year=2024, oa_url=oa_url)


def request(**options):
    return ResearchRequest.model_validate({"project": {"description": "traffic volume prediction with weather", "task": "regression"},
                                           "options": {"max_papers": 5, **options}})


def discover(handler, hit=None, **s):
    service = ResearchService(settings(**s), FakeProvider([hit or paper_hit()]), fulltext_client=client_for(handler), resolver=resolve)
    return asyncio.run(service.discover(request()))


class TestServiceIntegration:
    def test_methods_and_figures_come_from_the_papers_own_text_and_are_labelled(self):
        pdf = make_pdf(PAPER)
        body = discover(lambda r: httpx.Response(200, content=pdf))
        paper = body.papers[0]
        assert paper.retrieval_status == "full_text" and EvidenceKind.PAPER_TEXT in paper.evidence and paper.full_text_chars
        assert "XGBoost" in paper.methods and paper.open_access_url == "https://example.org/p.pdf"  # not in the abstract
        figure = next(m for m in body.benchmarks if m.value == "93.2%")
        assert figure.evidence is EvidenceKind.PAPER_TEXT and "own text" in figure.statement
        assert any("open-access full text were also read" in l for l in body.limitations)
        assert any(p.evidence is EvidenceKind.PAPER_TEXT for p in body.provenance)

    def test_a_failed_fetch_leaves_the_paper_at_snippet_level(self):
        body = discover(lambda r: httpx.Response(404))
        paper = body.papers[0]
        assert paper.retrieval_status == "snippet_only" and EvidenceKind.PAPER_TEXT not in paper.evidence and not paper.full_text_chars

    def test_it_can_be_switched_off_and_papers_without_a_pdf_are_never_fetched(self):
        calls = []
        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(200, content=make_pdf(PAPER))
        discover(handler, research_fulltext_enabled=False)
        discover(handler, hit=paper_hit(oa_url=None))
        assert calls == []

    def test_the_stream_reports_the_full_text_step(self, monkeypatch):
        from fastapi.testclient import TestClient

        from app.api.v1 import research as research_api
        from app.main import app

        pdf = make_pdf(PAPER)
        service = ResearchService(settings(), FakeProvider([paper_hit()]), fulltext_client=client_for(lambda r: httpx.Response(200, content=pdf)), resolver=resolve)
        monkeypatch.setattr(research_api, "_service", service)
        text = TestClient(app).post("/api/v1/research/stream", json=request().model_dump(mode="json")).text
        assert '"id": "fulltext"' in text and "1 paper read" in text
