"""Unit tests for candidate deduplication.

Policy under test: merge only on strong evidence, and keep unrelated datasets
separate even when they look similar.
"""

from __future__ import annotations

from app.schemas.dataset import SourceType, VerificationStatus
from app.services.datasets.deduplicator import (
    DatasetDeduplicator,
    name_token_key,
    normalize_name_key,
    normalize_url_key,
)
from app.services.datasets.normalizer import normalize_genai_candidate
from tests.conftest import make_candidate, make_genai_candidate


class TestKeyHelpers:
    def test_normalize_url_key(self):
        assert normalize_url_key("https://www.kaggle.com/datasets/a/b") == (
            "kaggle.com/datasets/a/b"
        )
        assert normalize_url_key("http://kaggle.com/datasets/a/b/") == "kaggle.com/datasets/a/b"
        assert normalize_url_key("HTTPS://WWW.Kaggle.com/datasets/a/b?x=1#frag") == (
            "kaggle.com/datasets/a/b"
        )
        assert normalize_url_key(None) is None
        assert normalize_url_key("") is None

    def test_normalize_name_key(self):
        assert normalize_name_key("Student-Performance!") == "student performance"
        assert normalize_name_key(None) is None

    def test_name_token_key_drops_generic_words(self):
        assert name_token_key("Student Performance Dataset") == "performance student"
        assert name_token_key("student-performance") == "performance student"
        assert name_token_key("Data Set") is None
        assert name_token_key(None) is None


class TestMergeOnSourceIdentity:
    def test_same_source_and_source_id_merges(self):
        a = make_candidate(name="Student Data", source_id="owner/one")
        b = make_candidate(name="Completely Different Title", source_id="owner/one")
        result = DatasetDeduplicator().deduplicate([a, b])
        assert len(result) == 1
        assert result[0].merged_candidate_count == 2

    def test_different_source_ids_and_names_stay_separate(self):
        a = make_candidate(name="Student Data", source_id="owner/one")
        b = make_candidate(name="Student Grades", source_id="owner/two")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 2

    def test_same_name_and_owner_merges_even_with_different_source_ids(self):
        a = make_candidate(name="Student Data", source_id="owner/one")
        b = make_candidate(name="Student Data", source_id="owner/two")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 1

    def test_case_insensitive_source_id(self):
        a = make_candidate(name="A", source_id="Owner/One")
        b = make_candidate(name="B", source_id="owner/one")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 1


class TestMergeOnUrl:
    def test_same_url_different_source_merges(self):
        a = make_candidate(name="Student Data", url="https://x.com/data/one")
        b = make_candidate(
            name="Totally Other Name",
            source="genai",
            source_type=SourceType.AI_SUGGESTED,
            verification_status=VerificationStatus.UNVERIFIED,
            source_id="other",
            url="https://x.com/data/one/",
        )
        result = DatasetDeduplicator().deduplicate([a, b])
        assert len(result) == 1

    def test_different_urls_stay_separate(self):
        a = make_candidate(name="Student Data", url="https://x.com/a", source_id="o/a")
        b = make_candidate(name="Student Grades", url="https://y.com/b", source_id="o/b")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 2

    def test_null_urls_do_not_merge_everything(self):
        a = make_candidate(name="A", url=None, owner=None, source_id="o/a")
        b = make_candidate(name="B", url=None, owner=None, source_id="o/b")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 2


class TestMergeOnNameAndOwner:
    def test_same_name_same_owner_merges(self):
        a = make_candidate(name="Student Performance", owner="Someone")
        b = make_candidate(name="student performance", owner="someone")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 1

    def test_exact_title_merges_even_across_different_owners(self):
        """Exact title match is treated as the same dataset across sources."""
        a = make_candidate(name="Student Performance", owner="Alice", source_id="alice/d")
        b = make_candidate(name="Student Performance", owner="Bob", source_id="bob/d")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 1

    def test_similar_titles_with_different_owners_stay_separate(self):
        a = make_candidate(name="Student Performance", owner="Alice", source_id="alice/d")
        b = make_candidate(name="Student Performances", owner="Bob", source_id="bob/d")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 2

    def test_near_exact_name_across_sources_merges(self):
        a = make_candidate(name="Student Performance")
        b = normalize_genai_candidate(make_genai_candidate(name="student performance"))
        result = DatasetDeduplicator().deduplicate([a, b])
        assert len(result) == 1

    def test_generic_suffix_is_ignored(self):
        a = make_candidate(name="Student Performance")
        b = normalize_genai_candidate(make_genai_candidate(name="Student Performance Dataset"))
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 1


class TestUnrelatedDatasetsStaySeparate:
    def test_similar_but_distinct_titles_are_not_merged(self):
        a = make_candidate(name="Student Habits vs Academic Performance", source_id="o/a")
        b = make_candidate(
            name="Student Academic Performance",
            owner="different-owner",
            source_id="o/b",
            url="https://other.example/x",
        )
        result = DatasetDeduplicator().deduplicate([a, b])
        assert len(result) == 2

    def test_wholly_different_names_are_not_merged(self):
        a = make_candidate(name="Student Performance")
        b = make_candidate(name="Heart Disease Records", source_id="owner/heart")
        assert len(DatasetDeduplicator().deduplicate([a, b])) == 2

    def test_completely_different_candidates_are_all_kept(self):
        candidates = [
            make_candidate(name="Student Performance", source_id="owner/a"),
            make_candidate(name="Heart Disease", source_id="owner/b"),
            make_candidate(name="Network Intrusion Logs", source_id="owner/c"),
        ]
        assert len(DatasetDeduplicator().deduplicate(candidates)) == 3

    def test_empty_input(self):
        assert DatasetDeduplicator().deduplicate([]) == []


class TestMergePreservesVerifiedEvidence:
    def test_verified_record_wins_regardless_of_order(self):
        verified = make_candidate(name="Student Performance", url="https://x.com/a")
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a")
        )
        for order in ([verified, ai], [ai, verified]):
            dedup = DatasetDeduplicator()
            result = dedup.deduplicate([c.model_copy(deep=True) for c in order])
            assert len(result) == 1
            assert result[0].source_type is SourceType.VERIFIED_EXTERNAL
            assert result[0].verification_status is VerificationStatus.VERIFIED

    def test_verified_metadata_is_not_overwritten_by_ai_values(self):
        verified = make_candidate(
            name="Student Performance",
            url="https://x.com/a",
            license="CC0-1.0",
            usability=0.9,
        )
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Student Performance", url="https://x.com/a", license="CC BY 4.0"
            )
        )
        ai.usability = 0.1
        result = DatasetDeduplicator().deduplicate([verified, ai])
        assert result[0].license == "CC0-1.0"
        assert result[0].usability == 0.9

    def test_missing_metadata_is_filled_from_another_published_record(self):
        """Two registries describing one dataset may complete each other."""
        a = make_candidate(name="Student Performance", url="https://x.com/a")
        a.license = None
        a.owner = None
        b = make_candidate(
            name="Student Performance", url="https://x.com/a", source="hf", source_id="o/b"
        )
        b.id = "hf:o/b"
        b.owner = "UCI"
        b.license = "CC BY 4.0"
        result = DatasetDeduplicator().deduplicate([a, b])
        assert result[0].license == "CC BY 4.0"
        assert result[0].owner == "UCI"

    def test_ai_metadata_never_fills_a_verified_record(self):
        """A model's guess must not become registry-published metadata."""
        verified = make_candidate(name="Student Performance", url="https://x.com/a")
        verified.license = None
        verified.owner = None
        verified.description = None
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a")
        )
        ai.owner = "UCI"
        ai.license = "CC BY 4.0"
        result = DatasetDeduplicator().deduplicate([verified, ai])[0]
        assert result.license is None
        assert result.owner is None
        assert result.description is None
        # ...but the claim is not discarded: it is kept, labelled, and separate.
        assert result.ai_suggested_claims is not None
        assert result.ai_suggested_claims.license == "CC BY 4.0"
        assert result.ai_suggested_claims.source == "genai"

    def test_rationale_survives_a_merge_into_verified(self):
        verified = make_candidate(name="Student Performance", url="https://x.com/a")
        verified.rationale = None
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a")
        )
        result = DatasetDeduplicator().deduplicate([verified, ai])
        assert result[0].rationale is not None

    def test_tags_are_unioned_between_published_records(self):
        a = make_candidate(name="Student Performance", url="https://x.com/a")
        a.tags = ["education"]
        b = make_candidate(name="Student Performance", url="https://x.com/a", source_id="o/b")
        b.tags = ["Education", "schools"]
        result = DatasetDeduplicator().deduplicate([a, b])
        assert result[0].tags == ["education", "schools"]

    def test_ai_tags_are_not_added_to_a_verified_record(self):
        verified = make_candidate(name="Student Performance", url="https://x.com/a")
        verified.tags = ["education"]
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a")
        )
        ai.tags = ["schools"]
        result = DatasetDeduplicator().deduplicate([verified, ai])[0]
        assert result.tags == ["education"]
        assert result.ai_suggested_claims.tags == ["schools"]

    def test_both_sources_are_recorded(self):
        verified = make_candidate(name="Student Performance", url="https://x.com/a")
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a")
        )
        result = DatasetDeduplicator().deduplicate([verified, ai])
        assert "genai" in result[0].duplicate_sources
        assert "kaggle" not in result[0].duplicate_sources

    def test_three_way_merge(self):
        a = make_candidate(name="Student Performance", url="https://x.com/a")
        b = make_candidate(name="Student Performance", url="https://x.com/a", source_id="owner/b")
        c = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a")
        )
        result = DatasetDeduplicator().deduplicate([a, b, c])
        assert len(result) == 1
        assert result[0].merged_candidate_count == 3


class TestAIClaimsAreNotLaundered:
    """Regression: an AI suggestion merged into a verified record used to turn
    the model's guessed features, target and tasks into 'declared' evidence."""

    def _merged(self, **ai_overrides):
        verified = make_candidate(
            name="Student Performance",
            url="https://x.com/a",
            description="Student marks data",
            target_description=None,
            task_types=[],
        )
        ai = normalize_genai_candidate(
            make_genai_candidate(
                name="Student Performance",
                url="https://x.com/a",
                features=["attendance", "bmi"],
                target_description="final grade",
                tasks=["regression"],
                **ai_overrides,
            )
        )
        return DatasetDeduplicator().deduplicate([verified, ai])[0]

    def test_published_fields_stay_as_the_registry_published_them(self):
        merged = self._merged()
        assert merged.target_description is None
        assert merged.feature_description is None
        assert merged.task_types == []
        assert merged.description == "Student marks data"

    def test_the_models_claims_are_kept_separately(self):
        claims = self._merged().ai_suggested_claims
        assert claims is not None
        assert claims.target_description == "final grade"
        assert "bmi" in claims.feature_description
        assert claims.task_types == ["regression"]

    def test_claims_are_not_visible_to_the_evidence_text(self):
        merged = self._merged()
        assert "bmi" not in merged.evidence_text()
        assert "regression" not in merged.evidence_text()

    def test_merge_order_does_not_matter(self):
        verified = make_candidate(name="Student Performance", url="https://x.com/a",
                                  feature_description=None, task_types=[])
        ai = normalize_genai_candidate(
            make_genai_candidate(name="Student Performance", url="https://x.com/a",
                                 features=["bmi"], tasks=["regression"])
        )
        for order in ([verified, ai], [ai, verified]):
            merged = DatasetDeduplicator().deduplicate(
                [c.model_copy(deep=True) for c in order]
            )[0]
            assert merged.feature_description is None
            assert merged.task_types == []
            assert "bmi" in merged.ai_suggested_claims.feature_description

    def test_a_merged_ai_claim_cannot_create_a_feature_or_task_verdict(self):
        from app.services.datasets.intelligence import build_dataset_intelligence
        from app.services.datasets.ranking import DatasetRankingEngine
        from app.schemas.intelligence import FeatureMatchState
        from tests.conftest import TEST_NOW, make_requirements, make_settings

        merged = self._merged()
        req = make_requirements(features=["bmi"], tasks=["classification"])
        ranked = DatasetRankingEngine(settings=make_settings(), now=TEST_NOW).rank([merged], req)[0]
        intel = build_dataset_intelligence(ranked, req)
        assert {f.state for f in intel.feature_coverage.all_features} == {FeatureMatchState.UNKNOWN}
        assert intel.compatibility.contradictions == []

    def test_verified_with_verified_still_fills_gaps(self):
        a = make_candidate(name="Student Performance", url="https://x.com/a")
        a.usability = None
        b = make_candidate(name="Student Performance", url="https://x.com/a", source_id="o/b")
        b.usability = 0.7
        assert DatasetDeduplicator().deduplicate([a, b])[0].usability == 0.7


class TestOrderPreservation:
    def test_first_seen_order_is_kept(self):
        candidates = [
            make_candidate(name="First", source_id="owner/1"),
            make_candidate(name="Second", source_id="owner/2"),
            make_candidate(name="First", source_id="owner/1"),
        ]
        result = DatasetDeduplicator().deduplicate(candidates)
        assert [c.name for c in result] == ["First", "Second"]


class TestDeduplicatorIsolation:
    def test_two_instances_do_not_share_state(self):
        candidate = make_candidate(name="Student Performance", url="https://x.com/a")
        first = DatasetDeduplicator()
        first.deduplicate([candidate.model_copy(deep=True)])
        # A fresh deduplicator must not merge with a previous run's results.
        assert len(DatasetDeduplicator().deduplicate([candidate.model_copy(deep=True)])) == 1

    def test_reset_clears_indexes(self):
        dedup = DatasetDeduplicator()
        a = make_candidate(name="Student Performance", url="https://x.com/a")
        dedup.deduplicate([a])
        dedup.reset()
        assert len(dedup.deduplicate([a.model_copy(deep=True)])) == 1
