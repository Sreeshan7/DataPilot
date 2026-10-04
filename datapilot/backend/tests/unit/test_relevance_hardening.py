"""Final hardening: short-alias safety, evidence hierarchy, requirement coverage,
contradiction penalties, relevance primacy, monotonicity and explainability.

Every test here corresponds to a failure mode that a naive ranking system
actually has: a two-letter token inventing a domain, a "related" concept
outvoting an exact one, a registry that publishes less being scored as if it had
been proven wrong, popularity standing in for fit, and a low score with no
explanation.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import (
    MatchQuality,
    MatchStatus,
    RankingFactorName,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.normalizer import normalize_genai_candidate
from app.services.datasets.ranking import DatasetRankingEngine
from app.services.keyword_service import KeywordService
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_genai_candidate,
    make_requirements,
    make_settings,
)


def _rank(candidates, requirements=None, **overrides):
    engine = DatasetRankingEngine(settings=make_settings(**overrides), now=TEST_NOW)
    return engine.rank([c.model_copy(deep=True) for c in candidates], requirements or make_requirements())


def _one(candidate, requirements=None, **overrides):
    return _rank([candidate], requirements, **overrides)[0]


def _factor(candidate, name):
    return next(f for f in candidate.ranking_factors if f.name is name)


# ---------------------------------------------------------------------------
# 1. Short and generic aliases must not invent concepts
# ---------------------------------------------------------------------------


class TestShortAliasSafety:
    @pytest.fixture(scope="class")
    def ks(self) -> KeywordService:
        return KeywordService()

    @pytest.mark.parametrize(
        "text",
        [
            "patients admitted at hospital for follow up appointments",
            "students enrolled at university and sat their exams",
            "the meeting was at the school in Leeds",
        ],
    )
    def test_at_does_not_mean_algorithmic_trading(self, ks, text):
        assert not any(
            m.canonical_term == "algorithmic trading" for m in ks.match(text)
        )

    @pytest.mark.parametrize("text", ["career at the zoo", "cat", "scatter"])
    def test_short_token_is_not_matched_inside_a_longer_word(self, ks, text):
        """Word boundaries, not substrings.

        "at" occurs in "career at", "cat" and "scatter", and none of those words
        is the abbreviation.
        """
        assert not any(m.matched_text.lower() == "at" for m in ks.match(text))

    @pytest.mark.parametrize("token", ["ai", "ml", "cv", "nlp"])
    def test_a_bare_generic_abbreviation_names_nothing(self, ks, token):
        """Alone in a sentence, these carry no domain."""
        assert not ks.match(f"a project about {token} only")

    def test_a_generic_abbreviation_is_kept_when_the_context_supports_it(self, ks):
        """A broader text is judged on its own merits."""
        matches = ks.match("machine learning pipelines for predictive maintenance")
        assert any(m.domain == "artificial_intelligence" for m in matches)

    def test_a_multi_letter_technology_name_still_matches(self, ks):
        """Safety must not cost real matches: "IoT" is a concept, not a token."""
        matches = ks.match("IoT sensor readings")
        assert any(m.canonical_term == "internet of things" for m in matches)
        assert any(m.domain == "iot" for m in matches)

    def test_no_single_character_phrase_is_indexed(self, ks):
        for phrase in ks._all_phrases:
            normalized = phrase.normalized
            if len(normalized.split()) == 1:
                assert len(normalized) > 1, normalized

    def test_one_character_aliases_cannot_match(self, ks):
        """No index - canonical, synonym, alias or abbreviation - holds a 1-char key."""
        for index in (
            ks._canonical_index,
            ks._synonym_index,
            ks._alias_index,
            ks._abbreviation_index,
        ):
            assert all(len(key) > 1 for key in index)

    def test_ranking_is_not_fooled_by_a_short_token_in_prose(self):
        """End to end: "at" must not make a student dataset look financial."""
        requirements = make_requirements(
            domain="Education",
            target="final grade",
            features=["attendance"],
            keywords=["student performance"],
        )
        ranked = _one(
            make_candidate(
                name="Student records",
                description="Attendance records for students admitted at term start",
                domain="education",
            ),
            requirements,
        )
        assert _factor(ranked, RankingFactorName.COMPATIBILITY).score == 1.0


# ---------------------------------------------------------------------------
# 2. Evidence hierarchy: exact beats related, specific beats generic
# ---------------------------------------------------------------------------


class TestEvidenceHierarchy:
    def _requirements(self, **kwargs):
        return make_requirements(
            domain="Cybersecurity",
            subdomain=None,
            target="attack type",
            features=["network traffic"],
            keywords=["intrusion detection"],
            **kwargs,
        )

    def test_exact_concepts_outscore_related_only_concepts(self):
        """A candidate matching only "related" concepts must score well below one
        matching the concept the project actually named."""
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            target=None,
            features=[],
            keywords=["disease prediction"],
            related_concepts=["symptoms", "patient records", "risk scoring", "lab results"],
        )
        exact = _one(
            make_candidate(
                name="Cohort study",
                description="Records for a disease prediction study with lab results",
                domain="healthcare",
            ),
            requirements,
        )
        related = _one(
            make_candidate(
                name="Vitals archive",
                description="Symptoms and patient records only",
                domain="healthcare",
            ),
            requirements,
        )
        assert _factor(exact, RankingFactorName.KEYWORD_OVERLAP).score > _factor(
            related, RankingFactorName.KEYWORD_OVERLAP
        ).score

    def test_a_multi_word_concept_outweighs_several_bare_ones(self):
        """Coverage is a weighted mean, so specificity decides, not head count."""
        requirements = make_requirements(
            domain="Cybersecurity",
            subdomain=None,
            target=None,
            features=[],
            keywords=["intrusion detection"],
            related_concepts=["traffic", "anomaly", "alert", "risk", "score"],
        )
        specific = _one(
            make_candidate(name="IDS", description="An intrusion detection dataset", domain="cybersecurity"),
            requirements,
        )
        scattered = _one(
            make_candidate(
                name="Misc", description="Traffic, anomaly, alert, risk and score columns", domain="cybersecurity"
            ),
            requirements,
        )
        assert _factor(specific, RankingFactorName.KEYWORD_OVERLAP).score > _factor(
            scattered, RankingFactorName.KEYWORD_OVERLAP
        ).score

    def test_match_quality_is_reported_per_requirement(self):
        """A semantic match is distinguishable from a literal one."""
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            target="diabetes outcome",
            features=["bmi"],
            keywords=[],
        )
        ranked = _one(
            make_candidate(
                name="Health study",
                description="Body mass index and glucose measurements for adults",
                feature_description="bmi, blood glucose, hba1c",
                domain="healthcare",
            ),
            requirements,
        )
        feature = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        qualities = {r.term: r.quality for r in feature.requirements}
        assert qualities["bmi"] is MatchQuality.EXACT

    def test_semantically_equivalent_feature_is_not_ignored(self):
        """BMI and body mass index are the same measurement, per the dictionary."""
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            target=None,
            features=["BMI", "blood glucose"],
            keywords=[],
        )
        ranked = _one(
            make_candidate(
                name="Cohort",
                description="Adults tracked over five years",
                feature_description="body mass index, fasting blood glucose, hba1c",
                domain="healthcare",
            ),
            requirements,
        )
        feature = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        assert feature.score == 1.0
        assert "BMI" in feature.matched_terms


# ---------------------------------------------------------------------------
# 3. Requirement coverage: MATCH / UNKNOWN / MISMATCH
# ---------------------------------------------------------------------------


class TestRequirementCoverage:
    def test_each_requirement_gets_a_verdict(self):
        requirements = make_requirements(
            domain="Education",
            target="final grade",
            features=["attendance", "study hours"],
            keywords=[],
        )
        ranked = _one(
            make_candidate(
                description="Attendance records only",
                feature_description="columns: attendance",
                domain="education",
            ),
            requirements,
        )
        feature = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        verdicts = {r.term: r.status for r in feature.requirements}
        assert verdicts["attendance"] is MatchStatus.MATCH
        assert verdicts["study hours"] is MatchStatus.MISMATCH
        assert feature.coverage == pytest.approx(0.5)
        assert feature.unmatched_terms == ["study hours"]
        assert feature.unknown_terms == []

    def test_unknown_requirements_are_reported_separately_from_mismatches(self):
        requirements = make_requirements(
            domain="Education", target="final grade", features=["attendance"], keywords=[]
        )
        ranked = _one(
            make_candidate(description="A study of school enrolment", domain="education"),
            requirements,
        )
        feature = _factor(ranked, RankingFactorName.FEATURE_MATCH)
        assert feature.score is None
        assert feature.unknown_terms == ["attendance"]
        assert feature.unmatched_terms == []


    def test_coverage_is_none_when_the_dimension_has_no_requirements(self):
        requirements = make_requirements(features=[], target=None, keywords=["student performance"])
        ranked = _one(make_candidate(), requirements)
        assert _factor(ranked, RankingFactorName.FEATURE_MATCH).score is None
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).coverage is None

    def test_an_unverified_ai_candidate_is_never_called_a_mismatch(self):
        """A model's own metadata cannot contradict the requirements it was read from.

        Using an inferred target to satisfy a requirement and then penalising the
        same inferred target for conflicting with it would be incoherent, so an
        unverified candidate is never charged a contradiction.
        """
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            tasks=["classification"],
            target="disease outcome",
            features=["blood glucose"],
            data_types=["image"],
        )
        ai = normalize_genai_candidate(make_genai_candidate())
        ranked = _one(ai, requirements)
        assert _factor(ranked, RankingFactorName.COMPATIBILITY).score == 1.0
        for factor in ranked.ranking_factors:
            assert {r.status for r in factor.requirements} != {MatchStatus.MISMATCH}

    def test_an_unverified_candidate_with_no_metadata_stays_unjudgeable(self):
        """Nothing declared means nothing judged, so no zero is recorded either."""
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            target="disease outcome",
            features=["blood glucose"],
            keywords=[],
        )
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Undescribed health set",
                description="A health dataset",
                features=[],
                target_description="",
                why_relevant="",
            )
        )
        ranked = _one(ai, requirements)
        statuses = {
            r.status
            for factor in ranked.ranking_factors
            for r in factor.requirements
        }
        assert MatchStatus.MISMATCH not in statuses


# ---------------------------------------------------------------------------


class TestUnprovidedTargets:
    """A sentence that names the target while denying it is not evidence for it.

    Asked to describe a target that a dataset does not have, a model writes
    "requires feature engineering to derive churn status". That sentence contains
    every word the project asked for, so a literal match scored the dataset as a
    perfect target hit - the exact opposite of what it says.
    """

    REQUIREMENTS = dict(
        domain="Retail",
        subdomain=None,
        tasks=["classification"],
        target="customer churn status",
        features=["purchase frequency", "recency"],
        keywords=["customer churn"],
    )

    def test_a_derivable_target_is_not_a_declared_target(self):
        ranked = _one(
            make_candidate(
                name="Online Retail Dataset (UCI)",
                description="Online retail transactions",
                domain="retail",
                feature_description="InvoiceNo; CustomerID; Quantity; InvoiceDate; UnitPrice",
                target_description=(
                    "Requires feature engineering to derive churn status based on "
                    "time since last purchase."
                ),
            ),
            make_requirements(**self.REQUIREMENTS),
        )
        target = _factor(ranked, RankingFactorName.TARGET_MATCH)
        assert target.score is None
        assert target.unknown_terms
        assert "not provided" in (target.detail or "")
        assert "Requires feature engineering" in (target.detail or "")

    def test_a_dataset_with_no_label_column_is_not_a_target_hit(self):
        ranked = _one(
            make_candidate(
                name="Transaction Log Export",
                description="Raw purchase transactions",
                domain="retail",
                target_description="No churn label is provided; retention must be inferred.",
            ),
            make_requirements(**self.REQUIREMENTS),
        )
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score is None

    def test_a_dataset_that_does_declare_the_label_still_matches(self):
        ranked = _one(
            make_candidate(
                name="Retail Churn Study",
                description="Customer transactions with retention outcome",
                domain="retail",
                target_description="Binary indicator (Yes/No) for customer churn.",
            ),
            make_requirements(**self.REQUIREMENTS),
        )
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score == 1.0

    def test_a_real_churn_dataset_outranks_one_that_only_mentions_churn(self):
        requirements = make_requirements(**self.REQUIREMENTS)
        declares = make_candidate(
            name="Bank Customer Churn",
            description="Customer accounts with a churn outcome",
            domain="retail",
            target_description="Exited (1) or Retained (0).",
        )
        only_mentions = make_candidate(
            name="Online Retail Dataset (UCI)",
            description="Online retail transactions",
            domain="retail",
            feature_description="InvoiceNo; CustomerID; Quantity; InvoiceDate",
            target_description="Requires feature engineering to derive churn status.",
        )
        ranked = _rank([declares, only_mentions], requirements)
        assert ranked[0].name == "Bank Customer Churn", [
            (c.name, c.ranking_score) for c in ranked
        ]


class TestLabelHeadNouns:
    """The grammar used to name a label is not part of the label itself."""

    def test_a_target_named_with_a_head_noun_matches_a_plain_title(self):
        ranked = _one(
            make_candidate(
                name="E-Commerce Customer Churn Dataset",
                description="Customer behaviour records",
                domain="retail",
            ),
            make_requirements(
                domain="Retail",
                subdomain=None,
                tasks=[],
                target="customer churn status",
                features=[],
                keywords=[],
            ),
        )
        target = _factor(ranked, RankingFactorName.TARGET_MATCH)
        assert target.score == 1.0
        assert "customer churn" in " ".join(target.matched_via)

    def test_the_head_noun_is_not_stripped_from_a_single_word_target(self):
        ranked = _one(
            make_candidate(
                name="Sentiment Corpus",
                description="Review text with a label column",
                domain="retail",
            ),
            make_requirements(
                domain="Retail",
                subdomain=None,
                tasks=[],
                target="status",
                features=[],
                keywords=[],
            ),
        )
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score is None

def test_a_schema_listing_does_not_declare_the_ml_task():
        """Columns state what a dataset contains, not what it can be used for.

    The modelling task belongs to the project, so a dataset that enumerates its
    columns but never mentions "classification" leaves that requirement unjudged
    rather than failed.
    """
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            tasks=["classification"],
            target=None,
            features=["blood glucose"],
            keywords=[],
        )
        ranked = _one(
            make_candidate(
                description="Cohort measurements",
                feature_description="columns: blood glucose, hba1c, age",
                domain="healthcare",
            ),
            requirements,
        )
        task = _factor(ranked, RankingFactorName.TASK_MATCH)
        assert task.score is None
        assert task.unknown_terms == ["classification"]
        # The feature dimension, by contrast, can be judged from the same listing.
        assert _factor(ranked, RankingFactorName.FEATURE_MATCH).score == 1.0


# 4. Contradiction penalties need declared evidence
# ---------------------------------------------------------------------------


class TestContradictionPenalties:
    def test_a_conflicting_declared_task_is_penalised(self):
        requirements = make_requirements(domain="Education", tasks=["classification"])
        compatible = _one(
            make_candidate(name="A", task_types=["classification"], domain="education"),
            requirements,
        )
        conflicting = _one(
            make_candidate(name="B", task_types=["clustering"], domain="education"),
            requirements,
        )
        assert _factor(compatible, RankingFactorName.COMPATIBILITY).score == 1.0
        assert _factor(conflicting, RankingFactorName.COMPATIBILITY).score < 1.0
        assert _factor(conflicting, RankingFactorName.COMPATIBILITY).contradictions

    def test_a_single_foreign_domain_is_a_weaker_signal_than_a_wrong_task(self):
        requirements = make_requirements(domain="Education", tasks=["classification"])
        foreign = _one(
            make_candidate(name="A", task_types=["classification"], domain="environment", tags=[]),
            requirements,
        )
        wrong_task = _one(
            make_candidate(name="B", task_types=["clustering"], domain="education"),
            requirements,
        )
        assert (
            _factor(foreign, RankingFactorName.COMPATIBILITY).score
            > _factor(wrong_task, RankingFactorName.COMPATIBILITY).score
        )

    def test_a_file_modality_that_cannot_serve_the_request_is_penalised(self):
        """A project that needs tabular data cannot be served by images alone."""
        requirements = make_requirements(
            domain="Healthcare", tasks=["classification"], data_types=["tabular"]
        )
        images = _one(
            make_candidate(name="A", domain="healthcare", file_types=["scan1.png", "scan2.png"]),
            requirements,
        )
        tabular = _one(
            make_candidate(name="B", domain="healthcare", file_types=["patients.csv"]),
            requirements,
        )
        assert _factor(images, RankingFactorName.COMPATIBILITY).score < 1.0
        assert _factor(images, RankingFactorName.COMPATIBILITY).contradictions
        assert _factor(tabular, RankingFactorName.COMPATIBILITY).score == 1.0

    def test_silence_is_never_a_contradiction(self):
        """An AI candidate declares no tasks, files or tags, so nothing conflicts."""
        requirements = make_requirements(domain="Healthcare", tasks=["classification"], data_types=["image"])
        ai = normalize_genai_candidate(make_genai_candidate())
        assert _factor(_one(ai, requirements), RankingFactorName.COMPATIBILITY).score == 1.0

    def test_a_contradiction_is_explained_in_words(self):
        requirements = make_requirements(domain="Education", tasks=["classification"])
        ranked = _one(
            make_candidate(name="Cluster archive", task_types=["clustering"], domain="education"),
            requirements,
        )
        reasons = [r for r in ranked.ranking_reasons if "Conflicts with the request" in r]
        assert reasons
        assert "clustering" in reasons[0]


# ---------------------------------------------------------------------------
# 5. Relevance is the primary axis
# ---------------------------------------------------------------------------


class TestRelevancePrimacy:
    def test_relevance_score_ignores_publisher_signals(self):
        """Same content, different adoption: relevance must not move."""
        requirements = make_requirements()
        plain = _one(make_candidate(name="Same", download_count=10, vote_count=1, usability=None), requirements)
        famous = _one(make_candidate(name="Same", download_count=5_000_000, vote_count=9000, usability=1.0), requirements)
        assert plain.relevance_score == famous.relevance_score
        assert plain.quality_score < famous.quality_score

    def test_popularity_cannot_buy_a_better_rank_than_relevance(self):
        requirements = make_requirements(domain="Education", target="final grade", features=["attendance"], keywords=[])
        relevant = make_candidate(
            name="Attendance records",
            description="Attendance and examination marks",
            domain="education",
            download_count=50,
            vote_count=1,
            usability=None,
            updated_at=None,
        )
        popular = make_candidate(
            name="Penguin census",
            description="Beak lengths and colony counts",
            domain="environment",
            tags=["environment"],
            download_count=5_000_000,
            vote_count=9000,
            usability=1.0,
        )
        ranked = _rank([popular, relevant], requirements)
        assert ranked[0].name == "Attendance records"

    def test_quality_score_never_mixes_in_relevance_signals(self):
        ranked = _one(make_candidate())
        quality = _factor(ranked, RankingFactorName.QUALITY)
        assert quality.score is not None
        assert ranked.relevance_score is not None

    def test_a_highly_relevant_ai_suggestion_can_outrank_a_weak_verified_dataset(self):
        """Trust is a tie-breaker, not a ceiling."""
        requirements = make_requirements(domain="Education", target="final grade", features=["attendance"], keywords=[])
        verified = make_candidate(
            name="Sparse education archive",
            description="Miscellaneous school notices",
            domain="education",
            usability=None,
            download_count=1,
            vote_count=0,
            updated_at=None,
        )
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Student grade registry",
                description="Attendance, examination marks and final grade per student",
                features=["attendance", "examination marks", "final grade"],
                target_description="final grade",
            )
        )
        ranked = _rank([verified, ai], requirements)
        assert ranked[0].source_type.value != "verified_external"
        assert ranked[0].ranking_reasons[-1].startswith("AI-suggested")

    def test_verification_never_changes_relevance(self):
        """Trust is reported on its own axis, so identical content scores alike."""
        requirements = make_requirements()
        verified = make_candidate()
        ai = verified.model_copy(
            deep=True,
            update={
                "id": "genai:same-dataset",
                "source": "genai",
                "source_type": SourceType.AI_SUGGESTED,
                "verification_status": VerificationStatus.UNVERIFIED,
                "rationale": "model opinion, excluded from evidence",
            },
        )
        assert _one(verified, requirements).relevance_score == _one(ai, requirements).relevance_score


# ---------------------------------------------------------------------------
# 6. Monotonicity: more evidence never hurts, contradictions always help
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_adding_a_matching_term_never_lowers_the_score(self):
        requirements = make_requirements()
        base = _one(make_candidate(description="Attendance records"), requirements).ranking_score
        better = _one(
            make_candidate(
                description="Attendance, examination marks, assignment scores and study hours"
            ),
            requirements,
        ).ranking_score
        assert better >= base

    def test_adding_a_contradiction_never_raises_the_score(self):
        requirements = make_requirements(domain="Education", tasks=["classification"])
        clean = _one(make_candidate(name="A", task_types=["classification"]), requirements).ranking_score
        conflicted = _one(make_candidate(name="B", task_types=["clustering"]), requirements).ranking_score
        assert conflicted < clean

    def test_more_relevant_metadata_raises_the_relevance_score(self):
        requirements = make_requirements(
            domain="Education",
            target="final grade",
            features=["attendance", "study hours"],
            keywords=["student performance"],
        )
        sparse = _one(make_candidate(description="School notices"), requirements).relevance_score
        rich = _one(
            make_candidate(
                description="Student performance records with attendance and study hours",
                feature_description="columns: attendance, study hours",
                target_description="final grade",
                domain="education",
            ),
            requirements,
        ).relevance_score
        assert rich > sparse

    def test_the_same_inputs_always_rank_the_same_way(self):
        requirements = make_requirements()
        candidates = [
            make_candidate(name=f"C{i}", source_id=f"owner/{i}", download_count=i * 1000)
            for i in range(5)
        ]
        first = [c.id for c in _rank(candidates, requirements)]
        second = [c.id for c in _rank(candidates, requirements)]
        assert first == second


# ---------------------------------------------------------------------------
# 7. The four project scenarios must rank sensibly end to end
# ---------------------------------------------------------------------------


class TestScenarioMatrix:
    def test_student_grades(self):
        requirements = make_requirements(
            domain="Education",
            subdomain="Educational Analytics",
            tasks=["classification", "regression"],
            target="student grade",
            features=["attendance", "study hours"],
            keywords=["student performance"],
        )
        ranked = _one(
            make_candidate(
                name="Secondary school records",
                description="Course grades, attendance and pass fail outcomes per term",
                feature_description="columns: attendance, study hours, grade",
                target_description="final grade",
                domain="education",
            ),
            requirements,
        )
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score == 1.0
        assert _factor(ranked, RankingFactorName.FEATURE_MATCH).score == 1.0
        assert _factor(ranked, RankingFactorName.COMPATIBILITY).score == 1.0
        assert ranked.relevance_score > 70

    def test_iot_botnet_intrusion_detection(self):
        requirements = make_requirements(
            domain="Cybersecurity",
            subdomain=None,
            tasks=["classification"],
            target="attack type",
            features=["network traffic"],
            keywords=["botnet"],
        )
        ranked = _one(
            make_candidate(
                name="IoT botnet traffic",
                description="Network traffic captures from an IoT botnet, labelled attack vs normal",
                feature_description="columns: network traffic, source ip, packet count",
                target_description="attack type",
                domain="cybersecurity",
            ),
            requirements,
        )
        assert ranked.relevance_score > 60
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score == 1.0

    def test_diabetes_prediction(self):
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            tasks=["classification"],
            target="disease outcome",
            features=["bmi", "blood glucose"],
            keywords=["diabetes"],
        )
        ranked = _one(
            make_candidate(
                name="Diabetes cohort",
                description="Adult diabetes cohort with body mass index and blood glucose",
                feature_description="columns: bmi, blood glucose, hba1c, age",
                target_description="diabetes outcome",
                domain="healthcare",
            ),
            requirements,
        )
        assert _factor(ranked, RankingFactorName.FEATURE_MATCH).score == 1.0
        assert ranked.relevance_score > 60

    def test_customer_churn(self):
        requirements = make_requirements(
            domain="Retail",
            subdomain=None,
            tasks=["classification"],
            target="customer churn",
            features=["recency", "monetary value"],
            keywords=["customer churn"],
        )
        ranked = _one(
            make_candidate(
                name="Telecom churn",
                description="Subscriber churn history with retention labels",
                feature_description="columns: recency, frequency, monetary value",
                target_description="retained churn labels",
                domain="retail",
            ),
            requirements,
        )
        assert _factor(ranked, RankingFactorName.TARGET_MATCH).score == 1.0

    def test_a_wrong_domain_candidate_loses_to_a_right_one(self):
        requirements = make_requirements(
            domain="Healthcare",
            subdomain=None,
            tasks=["classification"],
            target="disease outcome",
            features=["blood glucose"],
            keywords=["diabetes"],
        )
        right = make_candidate(
            name="Diabetes cohort",
            description="Diabetes patients with blood glucose and bmi",
            domain="healthcare",
        )
        wrong = make_candidate(
            name="Penguin census",
            description="Beak lengths and colony counts",
            domain="environment",
            tags=["environment"],
        )
        ranked = _rank([wrong, right], requirements)
        assert ranked[0].name == "Diabetes cohort"


# ---------------------------------------------------------------------------
# 8. Explainability
# ---------------------------------------------------------------------------


class TestExplainability:
    def test_unknown_requirements_are_explained_to_the_user(self):
        requirements = make_requirements(
            domain="Education", target="final grade", features=["attendance"], keywords=[]
        )
        ranked = _one(make_candidate(description="School notices", domain="education"), requirements)
        notes = [r for r in ranked.ranking_reasons if "Could not confirm" in r]
        assert notes
        assert "attendance" in notes[0]

    def test_a_semantic_target_match_states_the_equivalence(self):
        requirements = make_requirements(
            domain="Cybersecurity",
            subdomain=None,
            tasks=["classification"],
            target="malicious network activity",
            features=[],
            keywords=[],
        )
        ranked = _one(
            make_candidate(
                name="Flow archive",
                description="Flow records labelled attack vs normal",
                domain="cybersecurity",
            ),
            requirements,
        )
        target = _factor(ranked, RankingFactorName.TARGET_MATCH)
        assert target.score == 1.0
        assert target.matched_via

    def test_no_reason_claims_a_match_the_factors_did_not_find(self):
        requirements = make_requirements(
            domain="Education", target="final grade", features=["attendance"], keywords=[]
        )
        ranked = _one(
            make_candidate(description="Volcanic rock spectroscopy", domain="environment", tags=["environment"]),
            requirements,
        )
        for reason in ranked.ranking_reasons:
            assert "required features" not in reason.lower()
            assert "prediction target" not in reason.lower()

    def test_reasons_stay_within_the_cap(self):
        ranked = _one(make_candidate())
        assert len(ranked.ranking_reasons) <= 7
