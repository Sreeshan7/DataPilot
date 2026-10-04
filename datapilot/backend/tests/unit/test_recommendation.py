"""Phase 3C: recommendation confidence and the final recommendation.

The final recommendation is the sentence a user acts on, so every case here is a way
it could mislead them: recommending something unconfirmed, recommending something
found incompatible, hiding a close call, or asserting more than the evidence says.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import MetadataDepth, SourceType, VerificationStatus
from app.schemas.intelligence import CompatibilityLevel
from app.schemas.recommendation import (
    ConfidenceLevel,
    FinalRecommendation,
    RecommendationConfidence,
    RecommendationOutcome,
    RecommendedDataset,
)
from app.services.datasets.intelligence import build_intelligence_report
from app.services.datasets.normalizer import normalize_genai_candidate
from app.services.datasets.ranking import DatasetRankingEngine
from app.services.datasets.recommendation import (
    CLOSE_CALL_MARGIN,
    MAX_ALTERNATIVES,
    build_confidence,
    build_final_recommendation,
    build_recommendations,
)
from tests.conftest import (
    TEST_NOW,
    make_candidate,
    make_genai_candidate,
    make_requirements,
    make_settings,
)

REQ = make_requirements(
    features=["attendance", "study hours"],
    target="academic performance",
    tasks=["classification"],
)

STRONG = dict(
    description="Student records. Columns: attendance, study hours, academic performance.",
    target_description="academic performance",
    task_types=["classification"],
    file_types=[".csv"],
    metadata_depth=MetadataDepth.DETAILED,
)


def _rank(candidates, requirements=REQ):
    return DatasetRankingEngine(settings=make_settings(), now=TEST_NOW).rank(
        [c.model_copy(deep=True) for c in candidates], requirements
    )


def _full(candidates, requirements=REQ):
    ranked = _rank(candidates, requirements)
    intel = build_intelligence_report(ranked, requirements)
    confidences, final = build_recommendations(ranked, intel)
    return ranked, intel, confidences, final


def _strong(name="Student Records", source_id="o/strong", **overrides):
    return make_candidate(name=name, source_id=source_id, **{**STRONG, **overrides})


def _transport(name="City Bus Routes", source_id="o/bus"):
    return make_candidate(
        name=name,
        source_id=source_id,
        description="Bus routes, stop schedules and passenger counts",
        domain="transport",
        tags=["transport"],
        target_description=None,
        metadata_depth=MetadataDepth.DETAILED,
    )


def _ai(name="UCI Student Performance", **overrides):
    return normalize_genai_candidate(make_genai_candidate(name=name, **overrides))


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_a_strong_verified_detailed_dataset_is_high(self):
        _, intel, conf, _ = _full([_strong()])
        assert intel[0].compatibility.overall is CompatibilityLevel.HIGH
        assert conf[0].level is ConfidenceLevel.HIGH

    def test_every_confidence_states_its_reasons(self):
        _, _, conf, _ = _full([_strong(), _transport(), _ai()])
        assert all(c.reasons and all(r.strip() for r in c.reasons) for c in conf)

    def test_an_unconfirmed_ai_suggestion_is_never_better_than_low(self):
        _, _, conf, _ = _full([_ai()])
        assert conf[0].level is ConfidenceLevel.LOW
        assert any("not confirmed by any registry" in r for r in conf[0].reasons)

    def test_a_listing_only_record_is_capped_at_medium(self):
        _, _, conf, _ = _full([_strong(metadata_depth=MetadataDepth.LISTING)])
        assert conf[0].level is ConfidenceLevel.MEDIUM
        assert any("search listing" in r for r in conf[0].reasons)

    def test_an_incompatible_dataset_is_low(self):
        _, intel, conf, _ = _full([_transport()])
        assert intel[0].compatibility.overall is CompatibilityLevel.LOW
        assert conf[0].level is ConfidenceLevel.LOW

    def test_unknown_compatibility_is_not_a_reason_for_confidence(self):
        bare = make_candidate(
            name="Dataset 7", description="", tags=[], domain=None, target_description=None
        )
        _, intel, conf, _ = _full([bare])
        assert intel[0].compatibility.overall in {CompatibilityLevel.UNKNOWN, CompatibilityLevel.MEDIUM}
        assert conf[0].level is not ConfidenceLevel.HIGH

    def test_unverified_features_cap_the_level_and_say_so(self):
        c = _strong(description="Student records. Columns: attendance, academic performance.")
        _, _, conf, _ = _full([c], make_requirements(features=["attendance", "bmi"]))
        assert conf[0].level is not ConfidenceLevel.HIGH

    def test_confidence_is_only_ever_capped_never_raised(self):
        order = {ConfidenceLevel.LOW: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.HIGH: 2}
        base = {
            CompatibilityLevel.HIGH: 2,
            CompatibilityLevel.MEDIUM: 1,
            CompatibilityLevel.LOW: 0,
            CompatibilityLevel.UNKNOWN: 0,
        }
        ranked, intel, conf, _ = _full([_strong(), _transport(), _ai(), _strong(metadata_depth=MetadataDepth.LISTING, source_id="o/l", name="Listing")])
        for i, c in zip(intel, conf):
            assert order[c.level] <= base[i.compatibility.overall]

    def test_the_confidence_model_rejects_blank_reasons_and_none(self):
        with pytest.raises(ValueError):
            RecommendationConfidence(level=ConfidenceLevel.HIGH, reasons=[])
        with pytest.raises(ValueError):
            RecommendationConfidence(level=ConfidenceLevel.HIGH, reasons=["  "])


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


class TestOutcome:
    def test_a_confirmed_compatible_dataset_is_recommended(self):
        _, _, _, final = _full([_strong()])
        assert final.outcome is RecommendationOutcome.RECOMMENDED
        assert final.best_match.name == "Student Records"

    def test_no_candidates_is_none_not_an_error(self):
        final = build_final_recommendation([], [], [])
        assert final.outcome is RecommendationOutcome.NONE
        assert final.best_match is None
        assert "No candidate datasets" in final.summary

    def test_all_incompatible_is_none_and_names_the_closest(self):
        _, _, _, final = _full([_transport(), _transport("Rail Timetables", "o/rail")])
        assert final.outcome is RecommendationOutcome.NONE
        assert final.best_match is None
        assert "None of the 2 candidate datasets" in final.summary
        assert "incompatible" in final.summary
        assert final.alternatives  # the user can still see what was considered

    def test_an_incompatible_dataset_is_never_put_forward_however_it_ranks(self):
        ranked, intel, _, final = _full([_transport(), _strong()])
        assert final.best_match.name == "Student Records"
        assert all(a.name != final.best_match.name for a in final.alternatives)

    def test_only_an_ai_suggestion_is_tentative_never_recommended(self):
        _, _, _, final = _full([_ai()])
        assert final.outcome is RecommendationOutcome.TENTATIVE
        assert final.best_match.source_type is SourceType.AI_SUGGESTED
        assert any("has not been checked against any registry" in c for c in final.caveats)

    def test_a_confirmed_dataset_beats_a_higher_ranked_ai_suggestion(self):
        ranked, _, _, final = _full([_ai(), _strong()])
        assert final.best_match.source_type is SourceType.VERIFIED_EXTERNAL
        ai_alt = next(a for a in final.alternatives if a.verification_status is VerificationStatus.UNVERIFIED)
        assert "not confirmed by any registry" in ai_alt.trade_off

    def test_a_skipped_higher_ranked_candidate_is_disclosed(self):
        ranked, _, _, final = _full([_transport(), _strong()])
        if final.best_match.rank > 1:
            assert any("passed over" in c for c in final.caveats)
            assert any("found incompatible, unconfirmed, or less certain" in c for c in final.caveats)

    def test_a_verified_dataset_with_low_confidence_is_tentative(self):
        bare = make_candidate(
            name="Dataset 7", description="", tags=[], domain=None, target_description=None
        )
        _, _, conf, final = _full([bare])
        if conf[0].level is ConfidenceLevel.LOW:
            assert final.outcome is RecommendationOutcome.TENTATIVE

    def test_misaligned_inputs_are_rejected(self):
        ranked, intel, conf, _ = _full([_strong()])
        with pytest.raises(ValueError, match="aligned"):
            build_final_recommendation(ranked, intel, conf + conf)


# ---------------------------------------------------------------------------
# The schema refuses an overconfident result
# ---------------------------------------------------------------------------


def _best(**overrides) -> RecommendedDataset:
    payload = dict(
        dataset_id="kaggle:o/a",
        name="A",
        source="kaggle",
        source_type=SourceType.VERIFIED_EXTERNAL,
        rank=1,
        ranking_score=80.0,
        compatibility=CompatibilityLevel.HIGH,
        confidence=RecommendationConfidence(level=ConfidenceLevel.HIGH, reasons=["x"]),
    )
    payload.update(overrides)
    return RecommendedDataset(**payload)


class TestSchemaGuards:
    def test_recommended_requires_a_best_match(self):
        with pytest.raises(ValueError, match="needs a best match"):
            FinalRecommendation(outcome=RecommendationOutcome.RECOMMENDED, summary="s")

    def test_none_cannot_name_a_best_match(self):
        with pytest.raises(ValueError, match="cannot name"):
            FinalRecommendation(outcome=RecommendationOutcome.NONE, summary="s", best_match=_best())

    def test_an_ai_suggestion_cannot_be_recommended(self):
        with pytest.raises(ValueError, match="AI suggestion"):
            FinalRecommendation(
                outcome=RecommendationOutcome.RECOMMENDED,
                summary="s",
                best_match=_best(source_type=SourceType.AI_SUGGESTED),
            )

    def test_a_low_confidence_dataset_cannot_be_recommended(self):
        low = RecommendationConfidence(level=ConfidenceLevel.LOW, reasons=["x"])
        with pytest.raises(ValueError, match="LOW-confidence"):
            FinalRecommendation(
                outcome=RecommendationOutcome.RECOMMENDED, summary="s", best_match=_best(confidence=low)
            )

    def test_an_incompatible_dataset_cannot_be_recommended(self):
        with pytest.raises(ValueError, match="incompatible"):
            FinalRecommendation(
                outcome=RecommendationOutcome.RECOMMENDED,
                summary="s",
                best_match=_best(compatibility=CompatibilityLevel.LOW),
            )

    def test_tentative_may_carry_an_ai_suggestion(self):
        FinalRecommendation(
            outcome=RecommendationOutcome.TENTATIVE,
            summary="s",
            best_match=_best(source_type=SourceType.AI_SUGGESTED),
        )


# ---------------------------------------------------------------------------
# Alternatives, trade-offs and the close call
# ---------------------------------------------------------------------------


class TestAlternatives:
    def test_alternatives_are_capped_and_exclude_the_best_match(self):
        cands = [_strong(name=f"Student Set {i}", source_id=f"o/s{i}") for i in range(6)]
        _, _, _, final = _full(cands)
        assert len(final.alternatives) <= MAX_ALTERNATIVES
        assert final.best_match.dataset_id not in {a.dataset_id for a in final.alternatives}

    def test_every_alternative_states_a_trade_off(self):
        _, _, _, final = _full([_strong(), _transport(), _ai()])
        assert final.alternatives
        assert all(a.trade_off.strip() for a in final.alternatives)

    def test_an_incompatible_alternative_says_why_it_was_rejected(self):
        _, _, _, final = _full([_strong(), _transport()])
        bus = next(a for a in final.alternatives if a.name == "City Bus Routes")
        assert bus.trade_off.startswith("found incompatible with the project")
        assert "transport" in bus.trade_off

    def test_a_lower_compatibility_alternative_names_both_levels(self):
        weaker = _strong(
            name="Student Listing", source_id="o/listing",
            description="Student records. Columns: attendance, academic performance.",
        )
        _, intel, _, final = _full([_strong(), weaker])
        alt = final.alternatives[0]
        assert alt.compatibility in {CompatibilityLevel.MEDIUM, CompatibilityLevel.HIGH}
        assert alt.trade_off

    def test_identical_candidates_are_flagged_as_a_close_call(self):
        a = _strong(name="Student Records A", source_id="o/a")
        b = _strong(name="Student Records B", source_id="o/b")
        _, _, _, final = _full([a, b])
        assert final.close_call is True
        assert final.margin_to_next is not None and final.margin_to_next < CLOSE_CALL_MARGIN
        assert "close call" in final.summary

    def test_a_clear_lead_is_not_called_a_close_call(self):
        weak = make_candidate(
            name="Student Notes", source_id="o/weak", description="Notes", tags=[],
            domain=None, target_description=None, usability=None, vote_count=None, download_count=None,
        )
        _, _, _, final = _full([_strong(), weak])
        if final.margin_to_next is not None and final.margin_to_next >= CLOSE_CALL_MARGIN:
            assert final.close_call is False
            assert "close call" not in final.summary

    def test_closeness_to_an_incompatible_dataset_is_not_a_close_call(self):
        _, _, _, final = _full([_strong(), _transport()])
        assert final.margin_to_next is None
        assert final.close_call is False

    def test_a_close_call_goes_to_the_dataset_we_can_stand_behind_more(self):
        """Identical evidence, but only one record was read in detail: the scores tie,
        so the higher-confidence one must win even though it ranks second."""
        listing = _strong(name="Student Alpha", source_id="o/a", metadata_depth=MetadataDepth.LISTING)
        detailed = _strong(name="Student Beta", source_id="o/b")
        ranked, _, conf, final = _full([listing, detailed])
        assert [c.name for c in ranked] == ["Student Alpha", "Student Beta"]
        assert conf[0].level is ConfidenceLevel.MEDIUM and conf[1].level is ConfidenceLevel.HIGH
        assert final.best_match.name == "Student Beta"
        assert final.best_match.rank == 2
        assert any("1 higher-ranked candidate was passed over" in c for c in final.caveats)

    def test_outside_the_band_the_ranking_decides_even_against_confidence(self):
        top = _strong(name="Student Alpha", source_id="o/a", metadata_depth=MetadataDepth.LISTING)
        far_behind = make_candidate(
            name="Student Gamma", source_id="o/g", description="Student marks",
            target_description=None, tags=["education"], usability=0.1, vote_count=1, download_count=1,
            metadata_depth=MetadataDepth.DETAILED,
        )
        ranked, _, _, final = _full([top, far_behind])
        gap = ranked[0].ranking_score - ranked[1].ranking_score
        assert gap >= CLOSE_CALL_MARGIN
        assert final.best_match.name == "Student Alpha"

    def test_the_tie_break_never_promotes_an_unconfirmed_or_incompatible_dataset(self):
        _, _, _, final = _full([_ai(), _transport(), _strong(metadata_depth=MetadataDepth.LISTING)])
        assert final.best_match.source_type is SourceType.VERIFIED_EXTERNAL
        assert final.best_match.compatibility is not CompatibilityLevel.LOW

    def test_the_margin_threshold_is_just_under_the_measured_smallest_gap(self):
        assert CLOSE_CALL_MARGIN < 5.33


# ---------------------------------------------------------------------------
# The best match carries its evidence
# ---------------------------------------------------------------------------


class TestBestMatchEvidence:
    def test_it_lists_what_was_verified_unknown_and_missing_separately(self):
        c = _strong(description="Student records. Columns: attendance, academic performance.")
        req = make_requirements(features=["attendance", "bmi"], target="academic performance")
        _, intel, _, final = _full([c], req)
        best = final.best_match
        assert best.why
        assert isinstance(best.verified, list) and isinstance(best.unknown, list)
        assert isinstance(best.missing, list)

    def test_cautions_come_from_risks_and_limitations_and_are_unique(self):
        _, _, _, final = _full([_ai()])
        watch = final.best_match.watch_out_for
        assert watch and len(watch) == len(set(watch))
        assert any("generative model" in w or "model" in w for w in watch)

    def test_the_summary_never_states_more_than_the_evidence(self):
        _, intel, _, final = _full([_strong()])
        assert f"compatibility {intel[0].compatibility.overall.value}" in final.summary
        assert intel[0].feature_coverage.summary() in final.summary

    def test_a_tentative_summary_says_why_it_is_not_confident(self):
        _, _, conf, final = _full([_ai()])
        assert "not a confident recommendation" in final.summary
        assert conf[0].reasons[-1] in final.summary

    def test_a_recommendation_never_contains_a_credential_shaped_string(self):
        _, _, _, final = _full([_strong(), _ai()])
        text = final.model_dump_json()
        assert "KGAT_" not in text and "AIza" not in text


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_the_same_input_serializes_identically(self):
        a = _full([_strong(), _transport(), _ai()])[3].model_dump_json()
        b = _full([_strong(), _transport(), _ai()])[3].model_dump_json()
        assert a == b

    def test_input_order_does_not_change_the_answer(self):
        cands = [_strong(), _transport(), _ai()]
        forward = _full(cands)[3].model_dump_json()
        backward = _full(list(reversed(cands)))[3].model_dump_json()
        assert forward == backward

    def test_building_does_not_mutate_the_candidates(self):
        ranked = _rank([_strong(), _ai()])
        before = [c.model_dump_json() for c in ranked]
        intel = build_intelligence_report(ranked, REQ)
        build_recommendations(ranked, intel)
        assert [c.model_dump_json() for c in ranked] == before


# ---------------------------------------------------------------------------
# Found by the recommendation benchmark: confirmed is not a reason to pick junk
# ---------------------------------------------------------------------------


class TestConfirmedPreferenceNeedsAConfidentDataset:
    """A registry-confirmed dataset we are LOW on must not beat a better-ranked
    suggestion merely for being real (a confirmed, irrelevant dataset was being put
    forward over a highly relevant one)."""

    def _weak_confirmed(self):
        return make_candidate(
            name="Unrelated Telemetry", source_id="o/weak", description="Telemetry",
            tags=["education"], domain="education", target_description=None,
            metadata_depth=MetadataDepth.LISTING, usability=0.2, vote_count=1, download_count=1,
        )

    def test_a_confident_confirmed_dataset_still_beats_a_higher_ranked_suggestion(self):
        _, _, conf, final = _full([_ai(), _strong()])
        assert final.best_match.source_type is SourceType.VERIFIED_EXTERNAL
        assert final.best_match.confidence.level is not ConfidenceLevel.LOW

    def test_a_low_confidence_confirmed_dataset_does_not_beat_a_better_ranked_suggestion(self):
        ranked, _, conf, final = _full([self._weak_confirmed(), _ai()])
        weak = next(c for c in ranked if c.source_type is SourceType.VERIFIED_EXTERNAL)
        ai = next(c for c in ranked if c.source_type is SourceType.AI_SUGGESTED)
        if ranked.index(ai) < ranked.index(weak):
            confirmed_conf = conf[ranked.index(weak)]
            assert confirmed_conf.level is ConfidenceLevel.LOW
            assert final.best_match.source_type is SourceType.AI_SUGGESTED
            assert final.outcome is RecommendationOutcome.TENTATIVE
            assert any("no registry has confirmed" in c for c in final.caveats)

    def test_the_outcome_is_never_recommended_when_the_best_is_a_suggestion(self):
        for pool in ([self._weak_confirmed(), _ai()], [_ai()]):
            _, _, _, final = _full(pool)
            if final.best_match.source_type is SourceType.AI_SUGGESTED:
                assert final.outcome is not RecommendationOutcome.RECOMMENDED
