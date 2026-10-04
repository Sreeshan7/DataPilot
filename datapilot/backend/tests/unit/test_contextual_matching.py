"""Regression tests for contextual dictionary matching and query cleanliness.

Two reported defects are pinned here:

1. a weak single dictionary word ("learning") matched the education concept
   inside machine-learning, cybersecurity and IOT descriptions, so unrelated
   candidates inherited education vocabulary;
2. a sentence-like project description was reaching the dataset registry as a
   long, generic search query built from one request per source.

Everything is offline: no Kaggle or Gemini call is ever made.
"""

from __future__ import annotations

import pytest

from app.schemas.project import MatchType
from app.services.datasets.query_builder import build_dataset_search_query
from app.services.keyword_service import KeywordService
from tests.conftest import make_requirements


@pytest.fixture(scope="module")
def service() -> KeywordService:
    return KeywordService()


def _canonical_terms(service: KeywordService, text: str) -> set[str]:
    return {match.canonical_term for match in service.match(text)}


def _domains(service: KeywordService, text: str) -> set[str]:
    return {match.domain for match in service.match(text)}


# ---------------------------------------------------------------------------
# A weak word must not fire on its own
# ---------------------------------------------------------------------------


class TestWeakWordsNeedTheirContext:
    @pytest.mark.parametrize(
        "description",
        [
            "Intrusion detection logs for machine learning based anomaly models",
            "Deep learning models for network traffic classification",
            "IoT sensor readings used for learning a predictive maintenance model",
        ],
    )
    def test_learning_alone_does_not_mean_education(self, service, description):
        assert "education" not in _canonical_terms(service, description)

    def test_bare_learning_resolves_to_nothing(self, service):
        assert service.match("learning") == []

    def test_a_real_education_description_still_matches(self, service):
        assert "education" in _canonical_terms(service, "Learning analytics in education")

    def test_teaching_carries_an_education_context(self, service):
        terms = _canonical_terms(service, "teaching and learning in schools")
        assert "education" in terms


# ---------------------------------------------------------------------------
# Domain vocabulary must survive the same filter
# ---------------------------------------------------------------------------


class TestDomainVocabularyIsPreserved:
    @pytest.mark.parametrize(
        "description,expected",
        [
            ("IoT sensor readings for predictive maintenance", "iot"),
            ("Captured network packets and phishing campaigns", "cybersecurity"),
            ("Student enrolment and attendance records", "education"),
        ],
    )
    def test_domain_concepts_still_match(self, service, description, expected):
        assert expected in _domains(service, description)

    def test_domain_vocabulary_does_not_collide(self, service):
        domains = _domains(service, "IOT sensor telemetry")
        assert "iot" in domains
        assert "education" not in domains


# ---------------------------------------------------------------------------
# Evidence selection
# ---------------------------------------------------------------------------


class TestStrongerEvidenceWins:
    def test_the_education_evidence_is_the_multi_word_phrase(self, service):
        """The education concept must rest on "learning analytics", never on the
        weak word "learning" that appears inside it."""
        education = [m for m in service.match("learning analytics in education") if m.domain == "education"]
        assert education
        assert all(m.matched_text != "learning" for m in education)
        assert any(m.matched_text == "learning analytics" for m in education)


# ---------------------------------------------------------------------------
# Registry query cleanliness
# ---------------------------------------------------------------------------


class TestQueryIsCleanAndShort:
    @pytest.mark.parametrize(
        "description_keywords",
        [
            [
                "machine learning",
                "artificial intelligence",
                "data science",
                "success",
                "academic achievement",
            ],
            ["deep learning", "achievement", "analytics", "prediction"],
        ],
    )
    def test_generic_ml_vocabulary_is_never_queried(self, description_keywords):
        requirements = make_requirements(
            keywords=description_keywords,
            features=["attendance", "study hours"],
            target="academic performance",
        )
        tokens = build_dataset_search_query(requirements).split()
        assert tokens
        assert len(tokens) <= 6
        assert not {"machine", "artificial", "intelligence", "deep", "learning"} & set(tokens)

    def test_a_sentence_target_is_resolved_to_its_canonical_concept(self):
        requirements = make_requirements(
            target="Student performance metrics and support requirement status",
            features=[],
        )
        tokens = build_dataset_search_query(requirements).split()
        assert tokens[:2] == ["student", "performance"]
        assert "metrics" not in tokens
        assert "support" not in tokens
        assert "requirement" not in tokens

    def test_the_raw_description_is_never_returned(self):
        requirements = make_requirements(
            target="malicious network activity",
            features=["source IP", "packet length"],
            keywords=["intrusion detection"],
        )
        query = build_dataset_search_query(requirements)
        assert "i want" not in query.lower()
        assert len(query.split()) <= 6