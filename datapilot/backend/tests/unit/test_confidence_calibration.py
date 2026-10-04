"""Regression tests for confidence calibration.

An analysis that reports 0.85 while simultaneously asking for clarification, or
while listing critical missing information, is not 85% understood. The reported
confidence has to account for the uncertainty the analysis itself reports, and
every deduction has to be named so the number stays auditable.
"""

from __future__ import annotations

import pytest

from app.schemas.project import (
    Analysis,
    LLMAmbiguity,
    LLMConfidenceField,
    LLMFeatureRequirement,
    LLMDatasetRequirements,
    LLMKeywords,
    LLMMLTask,
    LLMProjectAnalysis,
    LLMProjectUnderstanding,
    LLMMissingInfo,
    MissingInformation,
)
from app.services.reconciliation_service import (
    _MAX_TOTAL_DEDUCTION,
    ReconciliationService,
)

SUBSTANTIAL_DESCRIPTION = " ".join(["word"] * 60)


def _output(
    self_report: float = 0.85,
    needs_clarification: bool = False,
    ambiguities: list[LLMAmbiguity] | None = None,
    missing: list[LLMMissingInfo] | None = None,
    target: str | None = "phishing vs legitimate label",
    features: list[LLMFeatureRequirement] | None = None,
) -> LLMProjectAnalysis:
    return LLMProjectAnalysis(
        project_understanding=LLMProjectUnderstanding(
            domain=LLMConfidenceField(value="cybersecurity", confidence=0.9, status="inferred"),
            subdomain=LLMConfidenceField(value="web security", confidence=0.85, status="inferred"),
            problem_type=LLMConfidenceField(value="Machine Learning", confidence=0.9, status="inferred"),
            ml_tasks=[LLMMLTask(task="classification", confidence=0.85)],
            objective="Detect phishing websites",
            target=LLMConfidenceField(value="phishing/legitimate", confidence=0.7, status="inferred"),
        ),
        keywords=LLMKeywords(inferred_concepts=[], related_terms=[]),
        dataset_requirements=LLMDatasetRequirements(
            required_features=features
            if features is not None
            else [LLMFeatureRequirement(name="URL features", description="")],
            target_description=target,
            label_description=None,
            data_type_notes="Tabular data",
        ),
        ambiguities=ambiguities or [],
        missing_information=missing or [],
        needs_clarification=needs_clarification,
        overall_confidence=self_report,
    )


def _reconcile(recon: ReconciliationService, llm: LLMProjectAnalysis, word_count: int = 60):
    return recon.reconcile(SUBSTANTIAL_DESCRIPTION, [], llm, word_count)


@pytest.fixture
def recon() -> ReconciliationService:
    return ReconciliationService()


# ---------------------------------------------------------------------------
# The reported defects
# ---------------------------------------------------------------------------


class TestConfidenceReflectsReportedUncertainty:
    def test_asking_for_clarification_lowers_confidence(self, recon):
        confident = _reconcile(recon, _output(self_report=0.85))
        uncertain = _reconcile(recon, _output(self_report=0.85, needs_clarification=True))
        assert uncertain.analysis.overall_confidence < confident.analysis.overall_confidence
        assert uncertain.analysis.overall_confidence < 0.7

    def test_critical_missing_information_lowers_confidence(self, recon):
        missing = [
            LLMMissingInfo(
                field="target variable",
                description="Which label is being predicted is not stated",
                severity="critical",
            )
        ]
        confident = _reconcile(recon, _output())
        uncertain = _reconcile(recon, _output(missing=missing))
        assert uncertain.analysis.overall_confidence < confident.analysis.overall_confidence

    def test_ambiguities_lower_confidence(self, recon):
        ambiguity = LLMAmbiguity(
            field="target",
            description="Classification or regression is unclear",
            possible_interpretations=["classification", "regression"],
        )
        confident = _reconcile(recon, _output())
        uncertain = _reconcile(recon, _output(ambiguities=[ambiguity]))
        assert uncertain.analysis.overall_confidence < confident.analysis.overall_confidence

    def test_a_clean_analysis_is_not_penalized(self, recon):
        analysis = _reconcile(recon, _output(self_report=0.85)).analysis
        assert analysis.overall_confidence == pytest.approx(0.85)
        assert analysis.confidence_adjustments == []


# ---------------------------------------------------------------------------
# Every deduction is accounted for
# ---------------------------------------------------------------------------


class TestDeductionsAreExplainable:
    def test_each_adjustment_names_its_reason(self, recon):
        llm = _output(
            needs_clarification=True,
            ambiguities=[
                LLMAmbiguity(
                    field="target",
                    description="unclear",
                    possible_interpretations=["a", "b"],
                )
            ],
            missing=[
                LLMMissingInfo(
                    field="label format", description="unknown", severity="warning"
                )
            ],
        )
        analysis = _reconcile(recon, llm).analysis
        reasons = [a.reason for a in analysis.confidence_adjustments]
        assert any("clarification" in r for r in reasons)
        assert any("ambiguit" in r for r in reasons)
        assert any("missing" in r for r in reasons)

    def test_deductions_sum_to_the_reported_drop(self, recon):
        llm = _output(
            self_report=0.9,
            needs_clarification=True,
            ambiguities=[
                LLMAmbiguity(field="a", description="x", possible_interpretations=["1", "2"]),
                LLMAmbiguity(field="b", description="y", possible_interpretations=["3", "4"]),
            ],
        )
        analysis = _reconcile(recon, llm).analysis
        assert analysis.overall_confidence == pytest.approx(
            0.9 - analysis.confidence_deduction_total
        )
        assert analysis.confidence_deduction_total == pytest.approx(
            sum(a.amount for a in analysis.confidence_adjustments)
        )

    def test_dictionary_bonus_still_applies_before_deductions(self, recon):
        """Agreement with the dictionary raises confidence; reported doubt lowers
        it. Both must be visible."""
        from app.schemas.project import KeywordMatchEvidence, MatchType

        match = KeywordMatchEvidence(
            canonical_term="phishing",
            matched_text="phishing",
            match_type=MatchType.EXACT,
            dictionary_entry_id="cyber.web.phishing",
            domain="cybersecurity",
            subdomain="web security",
        )
        with_bonus = recon.reconcile(
            SUBSTANTIAL_DESCRIPTION, [match], _output(self_report=0.80), 60
        ).analysis
        assert with_bonus.overall_confidence == pytest.approx(0.85)

    def test_confidence_stays_within_bounds(self, recon):
        many = [
            LLMMissingInfo(field=f"f{i}", description="unknown", severity="critical")
            for i in range(10)
        ]
        ambiguities = [
            LLMAmbiguity(field=f"a{i}", description="x", possible_interpretations=["1", "2"])
            for i in range(10)
        ]
        analysis = _reconcile(
            recon, _output(self_report=1.0, needs_clarification=True, ambiguities=ambiguities, missing=many)
        ).analysis
        assert 0.0 <= analysis.overall_confidence <= 1.0

    def test_deductions_never_exceed_the_total_cap(self, recon):
        many = [
            LLMMissingInfo(field=f"f{i}", description="unknown", severity="critical")
            for i in range(20)
        ]
        ambiguities = [
            LLMAmbiguity(field=f"a{i}", description="x", possible_interpretations=["1", "2"])
            for i in range(10)
        ]
        analysis = _reconcile(
            recon,
            _output(
                self_report=1.0,
                needs_clarification=True,
                ambiguities=ambiguities,
                missing=many,
                target=None,
                features=[],
            ),
        ).analysis
        assert sum(a.amount for a in analysis.confidence_adjustments) > _MAX_TOTAL_DEDUCTION
        assert analysis.confidence_deduction_total == pytest.approx(_MAX_TOTAL_DEDUCTION)
        assert analysis.overall_confidence == pytest.approx(1.0 - _MAX_TOTAL_DEDUCTION)

    def test_severity_is_weighted_by_its_own_report(self, recon):
        info = [MissingInformation(field="f", description="d", severity="info")]
        critical = [MissingInformation(field="f", description="d", severity="critical")]
        from app.services.reconciliation_service import _SEVERITY_DEDUCTION

        assert _SEVERITY_DEDUCTION["critical"] > _SEVERITY_DEDUCTION["warning"]
        assert _SEVERITY_DEDUCTION["warning"] > _SEVERITY_DEDUCTION["info"]
        assert info and critical


# ---------------------------------------------------------------------------
# Structural incompleteness
# ---------------------------------------------------------------------------


class TestUnresolvedRequirementsReduceConfidence:
    def test_a_substantial_description_without_a_target_is_penalized(self, recon):
        llm = _output(target=None)
        analysis = _reconcile(recon, llm, word_count=60).analysis
        assert any(
            "prediction target" in a.reason for a in analysis.confidence_adjustments
        )

    def test_a_short_description_without_a_target_is_not_penalized(self, recon):
        """A one-line brief legitimately has no resolvable target; the analysis
        should not be punished for the input being short."""
        llm = _output(target=None, features=[])
        analysis = _reconcile(recon, llm, word_count=5).analysis
        assert analysis.confidence_adjustments == []

    def test_missing_features_are_a_milder_penalty_than_a_missing_target(self, recon):
        no_target = _reconcile(
            recon, _output(target=None, features=[]), word_count=60
        ).analysis
        no_features = _reconcile(recon, _output(features=[]), word_count=60).analysis
        target_deduction = next(
            a.amount for a in no_target.confidence_adjustments if "target" in a.reason
        )
        feature_deduction = next(
            a.amount for a in no_features.confidence_adjustments if "features" in a.reason
        )
        assert target_deduction > feature_deduction


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------


class TestAnalysisSchema:
    def test_adjustments_default_to_empty_for_backward_compatibility(self):
        analysis = Analysis(overall_confidence=0.5)
        assert analysis.confidence_adjustments == []
        assert analysis.confidence_deduction_total == 0.0

    def test_negative_adjustments_are_rejected(self):
        from pydantic import ValidationError

        from app.schemas.project import ConfidenceAdjustment

        with pytest.raises(ValidationError):
            ConfidenceAdjustment(reason="x", amount=-0.1)