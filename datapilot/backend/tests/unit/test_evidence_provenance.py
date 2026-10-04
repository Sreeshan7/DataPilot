"""Regression tests for per-field evidence provenance.

A candidate is not verified or unverified as a whole: a registry result can be
externally confirmed and still omit most of the fields a relevance claim rests
on, while a model suggestion can be specific and still rest entirely on
generated text. These tests pin that distinction, and pin the rule that absence
stays absence.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import (
    FieldProvenance,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.evidence import (
    RELEVANCE_FIELDS,
    SOURCE_UNPUBLISHED_FIELDS,
    assess_evidence,
)
from app.services.datasets.normalizer import (
    normalize_genai_candidate,
    normalize_kaggle_record,
)
from tests.conftest import make_genai_candidate, make_kaggle_record


class TestExternalCandidate:
    def test_registry_metadata_is_external(self):
        candidate = normalize_kaggle_record(make_kaggle_record())
        assessment = candidate.evidence
        assert assessment is not None
        assert assessment.provenance is FieldProvenance.EXTERNAL
        assert "description" in assessment.external_fields
        assert "license" in assessment.external_fields
        assert "download_count" in assessment.external_fields
        assert assessment.external_fields == sorted(
            assessment.external_fields, key=RELEVANCE_FIELDS.index
        )

    def test_verified_candidate_can_still_be_missing_most_fields(self):
        """Existence is confirmed externally; the label metadata is not there.
        Collapsing the two into one flag would hide that."""
        candidate = normalize_kaggle_record(
            make_kaggle_record(subtitle="", description="", tags=[])
        )
        assessment = candidate.evidence
        assert assessment is not None
        assert assessment.verification_status is VerificationStatus.VERIFIED
        assert assessment.provenance is FieldProvenance.EXTERNAL
        assert assessment.evidence_coverage < 0.6
        assert "description" in assessment.missing_fields
        assert "tags" in assessment.missing_fields

    def test_unpublished_fields_are_explained_not_silently_dropped(self):
        assessment = normalize_kaggle_record(make_kaggle_record()).evidence
        assert assessment is not None
        unpublished = SOURCE_UNPUBLISHED_FIELDS["kaggle"]
        assert unpublished <= set(assessment.missing_fields)
        assert any("kaggle does not publish" in note for note in assessment.notes)

    def test_coverage_counts_only_supplied_fields(self):
        assessment = normalize_kaggle_record(make_kaggle_record()).evidence
        assert assessment is not None
        supplied = len(assessment.external_fields) + len(assessment.inferred_fields)
        assert assessment.evidence_coverage == pytest.approx(
            supplied / len(RELEVANCE_FIELDS)
        )


class TestDomainIsNeverClaimedAsExternal:
    def test_derived_domain_is_inferred_for_a_registry_candidate(self):
        candidate = normalize_kaggle_record(make_kaggle_record(tags=["education"]))
        assessment = candidate.evidence
        assert assessment is not None
        assert "domain" in assessment.inferred_fields
        assert "domain" not in assessment.external_fields
        assert any("derived by DataPilot" in note for note in assessment.notes)

    def test_derived_domain_is_inferred_for_a_model_candidate(self):
        candidate = normalize_genai_candidate(
            make_genai_candidate(name="Students Performance Dataset")
        )
        assessment = candidate.evidence
        assert assessment is not None
        assert "domain" in assessment.inferred_fields


class TestAICandidate:
    def test_every_present_field_is_inferred_never_external(self):
        candidate = normalize_genai_candidate(
            make_genai_candidate(
                name="Students Academic Performance",
                description="Attendance and marks",
                features=["attendance"],
                target_description="final grade",
            )
        )
        assessment = candidate.evidence
        assert assessment is not None
        assert assessment.external_fields == []
        assert assessment.inferred_fields
        assert assessment.provenance is FieldProvenance.AI_INFERRED

    def test_assessment_never_upgrades_verification_status(self):
        candidate = normalize_genai_candidate(make_genai_candidate())
        assert candidate.source_type is SourceType.AI_SUGGESTED
        assert candidate.verification_status is VerificationStatus.UNVERIFIED
        assert candidate.evidence.verification_status is VerificationStatus.UNVERIFIED

    def test_unconfirmed_existence_is_stated(self):
        candidate = normalize_genai_candidate(make_genai_candidate())
        assert any(
            "existence not externally confirmed" in note for note in candidate.evidence.notes
        )


class TestMissingStaysMissing:
    def test_a_candidate_with_no_metadata_still_reports_only_its_name(self):
        """``name`` is the one field every candidate must have; with nothing else
        supplied, coverage is minimal rather than silently optimistic."""
        candidate = normalize_kaggle_record(
            make_kaggle_record(subtitle="", description="", tags=[], license_name="")
        )
        candidate.description = None
        candidate.domain = None
        candidate.updated_at = None
        candidate.download_count = None
        assessment = assess_evidence(candidate)
        assert assessment.external_fields == ["name"]
        assert assessment.inferred_fields == []
        assert "description" in assessment.missing_fields
        assert assessment.evidence_coverage == pytest.approx(1 / len(RELEVANCE_FIELDS))

    def test_empty_collections_count_as_missing(self):
        candidate = normalize_kaggle_record(make_kaggle_record())
        candidate.tags = []
        candidate.task_types = []
        assessment = assess_evidence(candidate)
        assert "tags" in assessment.missing_fields
        assert "task_types" in assessment.missing_fields

    def test_assessment_is_idempotent(self):
        candidate = normalize_kaggle_record(make_kaggle_record())
        first = assess_evidence(candidate)
        second = assess_evidence(candidate)
        assert first.external_fields == second.external_fields
        assert first.missing_fields == second.missing_fields
        assert first.evidence_coverage == second.evidence_coverage

    def test_every_field_is_accounted_for_exactly_once(self):
        candidate = normalize_kaggle_record(make_kaggle_record())
        assessment = assess_evidence(candidate)
        buckets = [
            assessment.external_fields,
            assessment.inferred_fields,
            assessment.missing_fields,
        ]
        assert set(RELEVANCE_FIELDS) == set(buckets[0]) | set(buckets[1]) | set(buckets[2])
        assert sum(len(b) for b in buckets) == len(RELEVANCE_FIELDS)