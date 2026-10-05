"""Models for Phase 5: turning an analysed dataset into project intelligence.

Everything here is derived from a ``DatasetAnalysis`` and, when the file itself is available,
from its actual data. The rules these models enforce, rather than merely document:

* **No invented importance.** Association, mutual information and permutation importance exist
  only when DataPilot computed them from the user's rows. A dataset known from a registry carries
  none: its features are ``UNKNOWN`` and the response says why.
* **No invented columns.** Every project idea, feature assessment and model note refers to columns
  that exist in the analysed schema; ideas name their target and features explicitly so this can
  be checked.
* **Association is not causation.** Feature intelligence says "predictive association in this
  file", never "drives" or "causes", and carries that caveat.
* **No accuracy promises.** Model recommendations say what to try and why, never what score to
  expect.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.dataset_analysis import (
    CandidateState,
    EvidenceItem,
    EvidenceLevel,
    MLTask,
    TargetCandidate,
    TaskCandidate,
)

CAUSALITY_CAVEAT = (
    "These are predictive associations measured in this file, not causes: a feature that helps "
    "predict the target need not influence it, and may not be available when a prediction is needed."
)


# ---------------------------------------------------------------------------
# Dataset -> ML understanding
# ---------------------------------------------------------------------------


class TargetSource(str, Enum):
    USER_SPECIFIED = "user_specified"
    REGISTRY_DESIGNATED = "registry_designated"
    DETECTED = "detected"
    NONE = "none"


class LearningSetting(str, Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED_ONLY = "unsupervised_only"
    UNKNOWN = "unknown"


class DatasetMLUnderstanding(BaseModel):
    """What the dataset can be used to learn, reconciled from targets, tasks and quality."""

    learning_setting: LearningSetting
    best_target: Optional[TargetCandidate] = None
    target_source: TargetSource = TargetSource.NONE
    target_is_confident: bool = Field(
        False, description="True only for a user-named, registry-designated or STRONG target."
    )
    alternative_targets: list[TargetCandidate] = Field(default_factory=list)
    primary_task: Optional[TaskCandidate] = None
    other_tasks: list[TaskCandidate] = Field(default_factory=list)
    blockers: list[str] = Field(
        default_factory=list, description="Findings that must be dealt with before modelling."
    )
    limitations: list[str] = Field(default_factory=list)
    summary: str

    @model_validator(mode="after")
    def supervised_needs_a_target(self) -> "DatasetMLUnderstanding":
        if self.learning_setting is LearningSetting.SUPERVISED and self.best_target is None:
            raise ValueError("a supervised setting needs a target")
        if self.target_is_confident and self.best_target is None:
            raise ValueError("a confident target must exist")
        return self


# ---------------------------------------------------------------------------
# Feature intelligence
# ---------------------------------------------------------------------------


class FeatureRole(str, Enum):
    RECOMMENDED_FEATURE = "RECOMMENDED_FEATURE"
    USE_WITH_CAUTION = "USE_WITH_CAUTION"
    POSSIBLE_LEAKAGE = "POSSIBLE_LEAKAGE"
    IDENTIFIER = "IDENTIFIER"
    LOW_SIGNAL = "LOW_SIGNAL"
    UNKNOWN = "UNKNOWN"


class FeatureAssessment(BaseModel):
    column: str
    role: FeatureRole
    reasons: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel
    association: Optional[float] = Field(None, ge=0.0, le=1.0)
    association_method: Optional[str] = None
    mutual_information: Optional[float] = Field(None, ge=0.0, description="In nats.")
    permutation_importance: Optional[float] = Field(
        None, description="Mean drop in the held-out score when this column is shuffled."
    )
    permutation_importance_std: Optional[float] = Field(None, ge=0.0)
    importance_rank: Optional[int] = Field(None, ge=1)
    missing_pct: Optional[float] = Field(None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def measured_numbers_need_parsed_evidence(self) -> "FeatureAssessment":
        measured = (self.association, self.mutual_information, self.permutation_importance)
        if any(m is not None for m in measured) and self.evidence_level is not EvidenceLevel.DATAPILOT_PARSED:
            raise ValueError("a measured association or importance must be DATAPILOT_PARSED evidence")
        return self


class ImportanceMethod(BaseModel):
    """How the importance numbers were produced, so they can be judged."""

    description: str
    model: str
    scoring: str
    rows_used: int = Field(..., ge=0)
    rows_available: int = Field(..., ge=0)
    sampled: bool = False
    features_modelled: int = Field(..., ge=0)
    features_skipped: int = Field(0, ge=0)
    held_out_score: Optional[float] = None
    baseline_score: Optional[float] = Field(
        None, description="What a trivial predictor scores on the same held-out rows."
    )
    noise_floor: Optional[float] = Field(
        None, description="The highest importance any of the random yardstick columns reached."
    )
    noise_columns: int = Field(0, ge=0, description="How many random columns were added as yardsticks.")
    noise_mutual_information: Optional[float] = Field(
        None, description="Mutual information of the same random column, as a yardstick (nats)."
    )
    reliable: bool = Field(
        False, description="Whether the model beat the baseline, so its importances can be trusted."
    )
    split: str = "random"
    n_repeats: int = 0
    seed: int = 0


class FeatureIntelligence(BaseModel):
    available: bool
    unavailable_reason: Optional[str] = None
    target: Optional[str] = None
    task: Optional[MLTask] = None
    method: Optional[ImportanceMethod] = None
    features: list[FeatureAssessment] = Field(default_factory=list)
    role_counts: dict[str, int] = Field(default_factory=dict)
    caveat: str = CAUSALITY_CAVEAT

    @model_validator(mode="after")
    def unavailable_means_no_numbers(self) -> "FeatureIntelligence":
        if not self.available:
            if not self.unavailable_reason:
                raise ValueError("unavailable feature intelligence must say why")
            if self.method is not None:
                raise ValueError("unavailable feature intelligence cannot carry a method")
            for f in self.features:
                if f.evidence_level is not EvidenceLevel.DATAPILOT_PARSED and f.role not in (
                    FeatureRole.UNKNOWN, FeatureRole.IDENTIFIER
                ):
                    raise ValueError("without rows, a feature can only be UNKNOWN or a registry-marked IDENTIFIER")
                if any(v is not None for v in (f.mutual_information, f.permutation_importance)):
                    raise ValueError("importance cannot exist where it was not computed")
        return self


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ModelTier(str, Enum):
    BASELINE = "baseline"
    STRONGER = "stronger"
    SPECIALISED = "specialised"


class ModelRecommendation(BaseModel):
    model: str
    tier: ModelTier
    why: str
    preprocessing: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Project ideas
# ---------------------------------------------------------------------------


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class IdeaSource(str, Enum):
    DETERMINISTIC = "deterministic"
    GEMINI_ENHANCED = "gemini_enhanced"


class Feasibility(BaseModel):
    """How realistic the idea looks on THIS data. Computed from the profile, never promised as an outcome."""

    level: str = Field(..., description="high | medium | low")
    reasons: list[str] = Field(default_factory=list)


class ProjectIdea(BaseModel):
    id: str
    title: str = Field(..., min_length=1)
    problem_statement: str = Field(..., min_length=1)
    task: MLTask
    target: Optional[str] = None
    required_features: list[str] = Field(default_factory=list)
    optional_features: list[str] = Field(default_factory=list)
    why_it_fits: str = Field(..., min_length=1)
    challenges: list[str] = Field(default_factory=list)
    evaluation_metrics: list[str] = Field(default_factory=list)
    baseline_models: list[str] = Field(default_factory=list)
    stronger_models: list[str] = Field(default_factory=list)
    difficulty: Difficulty
    extension_potential: str = ""
    limitations: list[str] = Field(default_factory=list)
    evidence_level: EvidenceLevel
    source: IdeaSource = IdeaSource.DETERMINISTIC
    state: CandidateState = CandidateState.POSSIBLE
    feasibility: Optional[Feasibility] = None
    research_request: Optional[dict[str, Any]] = Field(
        None, description="A ready-made POST /research/stream body for this specific project (never run automatically).")

    @model_validator(mode="after")
    def features_must_not_include_the_target(self) -> "ProjectIdea":
        if self.target and (self.target in self.required_features or self.target in self.optional_features):
            raise ValueError("the target cannot be one of its own features")
        return self


# ---------------------------------------------------------------------------
# Dataset -> project compatibility
# ---------------------------------------------------------------------------


class CompatibilityStatus(str, Enum):
    MATCH = "MATCH"
    UNKNOWN = "UNKNOWN"
    MISMATCH = "MISMATCH"


class CompatibilityDimension(BaseModel):
    dimension: str
    status: CompatibilityStatus
    detail: str
    evidence: list[EvidenceItem] = Field(default_factory=list)


class ProjectCompatibility(BaseModel):
    """Whether this dataset can serve the project the user described."""

    goal_summary: str
    status: CompatibilityStatus
    dimensions: list[CompatibilityDimension] = Field(default_factory=list)
    matched_columns: list[str] = Field(default_factory=list)
    summary: str
    suggests_alternative_dataset: bool = False

    @model_validator(mode="after")
    def overall_status_follows_the_dimensions(self) -> "ProjectCompatibility":
        statuses = {d.status for d in self.dimensions}
        if CompatibilityStatus.MISMATCH in statuses and self.status is not CompatibilityStatus.MISMATCH:
            raise ValueError("a mismatching dimension makes the whole compatibility a MISMATCH")
        if self.status is CompatibilityStatus.MATCH and CompatibilityStatus.UNKNOWN in statuses \
                and CompatibilityStatus.MATCH not in statuses:
            raise ValueError("MATCH needs at least one matching dimension")
        return self


# ---------------------------------------------------------------------------
# Optional Gemini enrichment
# ---------------------------------------------------------------------------


class GeminiStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    DISABLED = "disabled"
    NOT_CONFIGURED = "not_configured"
    USED = "used"
    PARTIAL = "partial"
    FAILED = "failed"


class GeminiUsage(BaseModel):
    status: GeminiStatus = GeminiStatus.NOT_REQUESTED
    ideas_enhanced: int = 0
    ideas_rejected: int = 0
    note: Optional[str] = None


class LLMIdeaProse(BaseModel):
    """What Gemini may write for one seeded idea. Prose only: it cannot change a column."""

    id: str
    title: str
    problem_statement: str
    why_it_fits: str
    extension_potential: str


class LLMIdeaProseSet(BaseModel):
    ideas: list[LLMIdeaProse]


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class ProjectIntelligence(BaseModel):
    understanding: DatasetMLUnderstanding
    feature_intelligence: FeatureIntelligence
    project_ideas: list[ProjectIdea] = Field(default_factory=list)
    model_recommendations: list[ModelRecommendation] = Field(default_factory=list)
    compatibility: Optional[ProjectCompatibility] = None
    gemini: GeminiUsage = Field(default_factory=GeminiUsage)
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ideas_must_use_the_stated_columns(self) -> "ProjectIntelligence":
        if not 0 <= len(self.project_ideas) <= 5:
            raise ValueError("at most five project ideas are returned")
        return self
