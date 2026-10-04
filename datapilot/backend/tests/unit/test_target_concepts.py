"""Regression tests for dictionary-backed canonical target matching.

A project states its prediction target in its own words and a dataset states the
label column it actually holds. These tests pin the equivalence that literal term
matching used to miss, and pin the context guards that keep the equivalence from
becoming a false positive.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import DatasetCandidate, RankingFactorName
from app.services.datasets.ranking import DatasetRankingEngine, build_project_profile
from app.services.datasets.target_concepts import (
    get_target_concept_index,
    normalize_tokens,
    value_tokens,
)

from tests.conftest import TEST_NOW, make_candidate, make_requirements, make_settings


def _factor(candidate: DatasetCandidate, name: RankingFactorName):
    for factor in candidate.ranking_factors or []:
        if factor.name is name:
            return factor
    raise AssertionError(f"factor {name} missing from ranking output")


def _rank(target: str, description: str, domain: str | None, name: str = "Candidate") -> DatasetCandidate:
    """Rank one candidate against a single-target project in the given domain."""
    requirements = make_requirements(
        domain=domain,
        subdomain=None,
        target=target,
        features=[],
        keywords=[],
    )
    candidate = make_candidate(
        name=name,
        description=description,
        domain=domain,
        tags=[domain] if domain else [],
    )
    engine = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW)
    return engine.rank([candidate], requirements)[0]


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


class TestTargetConceptIndex:
    def test_label_spaces_come_from_the_dictionary(self):
        concepts = get_target_concept_index().concepts
        keys = {concept.label for concept in concepts.values()}
        assert "benign + malicious" in keys
        assert "attack + benign" in keys

    def test_container_words_are_not_label_values(self):
        # "binary labels normal anomaly" declares the values normal/anomaly.
        assert value_tokens("binary labels normal anomaly") == ["normal", "anomaly"]
        assert value_tokens("benign malicious labels") == ["benign", "malicious"]

    def test_plural_folding_does_not_corrupt_non_plural_words(self):
        assert value_tokens("malicious traffic attacks") == ["malicious", "traffic", "attack"]
        assert value_tokens("malicious") == ["malicious"]

    def test_label_spaces_sharing_a_value_form_one_family(self):
        index = get_target_concept_index()
        space = index.resolve("benign malicious")[0]
        family = {concept.label for concept in index.family(space)}
        assert family == {
            "attack",
            "attack + benign",
            "benign + malicious",
            "benign + sqli",
            "benign + xss",
        }

    def test_unrelated_label_values_stay_in_separate_families(self):
        index = get_target_concept_index()
        space = index.resolve("benign malicious")[0]
        labels = {concept.label for concept in index.family(space)}
        assert "family + malware" not in labels
        assert "abnormal + normal" not in labels


# ---------------------------------------------------------------------------
# The regression: a cybersecurity target must match cybersecurity label columns
# ---------------------------------------------------------------------------


class TestCybersecurityTargetMatching:
    @pytest.mark.parametrize(
        "description",
        [
            "Network intrusion detection logs with malicious vs benign traffic",
            "Flow records labelled normal vs malicious",
            "Attack vs normal classification of network flows",
            "SQL injection request logs with a benign background class",
            "Network traffic captures with labelled attacks",
        ],
    )
    def test_equivalent_label_columns_match(self, description):
        candidate = _rank("malicious network activity", description, "cybersecurity")
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.score == 1.0
        assert factor.matched_terms == ["malicious network activity"]

    def test_match_through_a_sibling_space_is_explained(self):
        candidate = _rank(
            "malicious network activity",
            "Attack vs normal classification of network flows",
            "cybersecurity",
        )
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.matched_via, "an indirect match must carry its justification"
        assert "attack + benign" in factor.matched_via[0]

    def test_a_literal_match_reports_no_concept_equivalence(self):
        candidate = _rank(
            "malicious network activity",
            "Malicious network activity labels from an intrusion detection system",
            "cybersecurity",
        )
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.score == 1.0
        assert factor.matched_via == []

    def test_the_exact_space_is_distinguishable_from_a_sibling(self):
        candidate = _rank(
            "malicious network activity",
            "Network intrusion detection logs with malicious vs benign traffic",
            "cybersecurity",
        )
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert len(factor.matched_via) == 1
        assert "via shared label values" not in factor.matched_via[0]

    @pytest.mark.parametrize(
        "description,domain",
        [
            ("Cardiology cohort with heart attack outcomes", "healthcare"),
            ("Botnet command and control traffic logs", "cybersecurity"),
            ("Penguin colony census 2020", None),
        ],
    )
    def test_unrelated_and_wrong_domain_candidates_do_not_match(self, description, domain):
        """A target nobody declares cannot be matched, and is not invented either.

        None of these candidates publish a ``target_description``, so a miss is
        unjudgeable rather than a stated absence: the factor reports UNKNOWN and
        stays out of the weighted mean instead of scoring zero.
        """
        candidate = _rank("malicious network activity", description, domain)
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.score is None
        assert factor.matched_terms == []
        assert factor.unmatched_terms == []
        assert factor.unknown_terms == ["malicious network activity"]

    def test_a_declared_unrelated_target_is_a_mismatch(self):
        """Once a target *is* declared, disagreement becomes a real absence."""
        candidate = make_candidate(
            name="Penguin Colony Census",
            description="Beak lengths and colony counts",
            target_description="body mass",
        )
        requirements = make_requirements(
            domain="cybersecurity", subdomain=None, target="malicious network activity",
            features=[], keywords=[],
        )
        engine = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW)
        ranked = engine.rank([candidate], requirements)[0]
        factor = _factor(ranked, RankingFactorName.TARGET_MATCH)
        assert factor.score == 0.0
        assert factor.unmatched_terms == ["malicious network activity"]

    def test_a_single_value_without_domain_context_is_not_enough(self):
        index = get_target_concept_index()
        assert index.match("benign malicious", frozenset({"malicious"}), None, None) is None
        # Naming two values of the space is self-evidencing.
        assert index.match(
            "benign malicious", frozenset({"benign", "malicious"}), None, None
        ) is not None

    def test_the_selected_space_does_not_depend_on_set_order(self):
        """Sibling spaces live in a frozenset, whose iteration order changes with
        the interpreter's hash seed. The best-supported space must still win."""
        index = get_target_concept_index()
        evidence = frozenset({"attack", "normal", "classification"})
        match = index.match(
            "malicious network activity",
            evidence,
            "cybersecurity",
            {"cybersecurity"},
        )
        assert match is not None
        assert match.concept.label == "attack + benign"
        assert match.equivalent is True

    def test_candidate_context_is_the_union_of_tag_and_text_domains(self):
        """A botnet dataset tagged ``iot`` still names security concepts, so the
        cybersecurity label space applies to it. A lone ``iot`` tag does not."""
        index = get_target_concept_index()
        evidence = frozenset(normalize_tokens(
            "IoT network traffic with various botnet attacks and normal traffic"
        ))
        assert index.match(
            "malicious network activity", evidence, "cybersecurity", {"iot"}
        ) is None
        assert index.match(
            "malicious network activity", evidence, "cybersecurity", {"iot", "cybersecurity"}
        ) is not None

    def test_a_conflicting_domain_still_blocks_the_match(self):
        index = get_target_concept_index()
        evidence = frozenset(normalize_tokens("Cardiology cohort with heart attack outcomes"))
        assert index.match(
            "malicious network activity", evidence, "cybersecurity", {"healthcare"}
        ) is None


# ---------------------------------------------------------------------------
# Generality: the same mechanism must work in another domain
# ---------------------------------------------------------------------------


class TestTargetMatchingGeneralises:
    def test_education_outcomes_match_an_equivalent_label_column(self):
        candidate = _rank(
            "student academic performance",
            "Course grades, attendance and pass fail outcomes per term",
            "education",
        )
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.score == 1.0

    def test_an_unrelated_domain_target_is_unjudgeable_not_zero(self):
        """Nothing declares a target, so nothing can be said about the target.

        A dataset with no published target metadata does not "lack" the required
        one - the question is simply unanswerable, and reporting it as unknown
        keeps the ranking from penalising registries that publish less.
        """
        candidate = _rank(
            "student academic performance",
            "Engine temperature readings for a combustion turbine",
            None,
        )
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.score is None
        assert factor.unknown_terms == ["student academic performance"]

    def test_no_target_leaves_the_factor_unscored(self):
        requirements = make_requirements(domain="education", subdomain=None, target=None)
        candidate = make_candidate()
        engine = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW)
        engine.rank([candidate], requirements)
        factor = _factor(candidate, RankingFactorName.TARGET_MATCH)
        assert factor.score is None
        assert factor.detail == "No target variable identified"


# ---------------------------------------------------------------------------
# Profile-level behaviour
# ---------------------------------------------------------------------------


def test_profile_records_no_concept_state_before_ranking():
    profile = build_project_profile(
        make_requirements(domain="cybersecurity", target="malicious network activity")
    )
    assert profile.target_terms == ["malicious network activity"]
    assert profile.project_domain_key == "cybersecurity"


class TestVerboseTargetsKeepTheirLabel:
    """Reducing a sentence-like target to a canonical concept can keep only the
    topic. The label the sentence also named must still count."""

    VERBOSE = "Network traffic event classification (Normal vs. Attack)"

    def test_the_reduced_term_is_the_topic_not_the_label(self):
        profile = build_project_profile(
            make_requirements(domain="cybersecurity", target=self.VERBOSE)
        )
        assert profile.raw_target == self.VERBOSE
        # Whichever topic the dictionary resolves, the original wording is kept.
        assert self.VERBOSE not in profile.target_terms

    def test_the_label_named_in_the_sentence_still_matches(self):
        requirements = make_requirements(
            domain="cybersecurity",
            target=self.VERBOSE,
            features=[],
            keywords=[],
        )
        candidate = make_candidate(
            name="IDS IoT Dataset",
            description="Captured connection records labelled 0 for normal and 1 for attack",
            domain="cybersecurity",
            tags=["cybersecurity"],
        )
        engine = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW)
        ranked = engine.rank([candidate], requirements)[0]
        factor = _factor(ranked, RankingFactorName.TARGET_MATCH)
        assert factor.score == 1.0
        # The label the sentence also named is still recovered, whichever of the
        # two mechanisms explains it: the raw target resolved against the label
        # space, or the concept it resolves to declaring that value.
        assert factor.matched_via
        assert any(
            "as written" in via or "declaring the target" in via
            for via in factor.matched_via
        )


class TestCandidateDomains:
    def test_context_is_the_union_of_the_tag_and_the_candidate_text(self):
        candidate = make_candidate(
            name="BoT-IoT Dataset",
            description="IoT network traffic with various botnet attacks and normal traffic",
            domain="iot",
            tags=["iot"],
        )
        # The tag says IoT; the candidate's own words name botnet attacks, which
        # is security. Both count, so the cybersecurity label space applies.
        assert DatasetRankingEngine._candidate_domains(candidate) == frozenset(
            {"iot", "cybersecurity"}
        )

    def test_related_hits_do_not_flood_the_domain_set(self):
        candidate = make_candidate(
            name="Intrusion Detection Logs",
            description="Packet captures from a supervised research network",
            domain="cybersecurity",
            tags=["cybersecurity"],
        )
        domains = DatasetRankingEngine._candidate_domains(candidate)
        assert "cybersecurity" in domains
        # RELATED hits reach across the whole dictionary; they must not count.
        assert len(domains) < 10

    def test_an_unrelated_candidate_does_not_gain_the_projects_domain(self):
        candidate = make_candidate(
            name="Penguin Colony Census",
            description="Beak lengths from field surveys",
            domain=None,
            tags=[],
        )
        assert DatasetRankingEngine._candidate_domains(candidate) == frozenset()


def test_tokens_helper_is_reusable_for_other_services():
    # normalize_tokens keeps every word; value_tokens additionally drops the
    # filler that carries no label value ("vs").
    assert normalize_tokens("Malicious-vs-Normal_Flows") == [
        "malicious",
        "vs",
        "normal",
        "flow",
    ]
    assert value_tokens("Malicious-vs-Normal_Flows") == ["malicious", "normal", "flow"]
