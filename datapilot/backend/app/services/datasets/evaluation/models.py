"""Data model for the ranking evaluation framework.

Four layers, deliberately separate so a future ranker can be compared without
touching the benchmark or the metrics:

* **Benchmark input** - :class:`BenchmarkCase` and friends. Hand-authored,
  deterministic, offline. Says what the project asked for, which candidates a
  retrieval step produced, and how good each candidate actually is.
* **Evaluation output** - :class:`CaseEvaluation` / :class:`EvaluationReport`.
  Machine-readable, serialised to ``evaluation_results.json``.
* **Diagnostics** - per-candidate factor decomposition, so a wrong rank can be
  read back to the factor that caused it.
* **Analysis** - ablation, sensitivity and score-audit results.

Nothing here imports the ranking engine's private helpers, and nothing here
changes ranking behaviour. The engine is measured, not modified.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.dataset import SourceType, VerificationStatus
from app.schemas.project import ProjectAnalysisData
from app.services.datasets.evaluation.labels import RelevanceGrade


# ---------------------------------------------------------------------------
# Benchmark input
# ---------------------------------------------------------------------------


class RequirementSpec(BaseModel):
    """What the project asked for, in the analyzer's vocabulary.

    Mirrors the fields of ``ProjectAnalysisData`` that ranking actually reads.
    A benchmark case that could not be expressed here would be testing the
    analyzer, not the ranker.
    """

    domain: Optional[str] = None
    subdomain: Optional[str] = None
    tasks: list[str] = Field(default_factory=list)
    target: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    data_types: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    dictionary_keywords: list[str] = Field(default_factory=list)


class CandidateSpec(BaseModel):
    """One candidate dataset as a retrieval step would have returned it.

    ``key`` is the case-local identifier used to attach a relevance grade. The
    constructed candidate id is ``benchmark:<case_id>:<key>`` so ids stay unique
    across the whole corpus without inventing real registry identifiers.
    """

    key: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    domain: Optional[str] = None
    target_description: Optional[str] = None
    feature_description: Optional[str] = None
    task_types: list[str] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)
    source_type: SourceType = SourceType.VERIFIED_EXTERNAL
    # Left unset, verification follows ``source_type``; set explicitly to model a
    # candidate that claims to be external without being confirmed.
    verification_status: Optional[VerificationStatus] = None
    downloads: Optional[int] = None
    votes: Optional[int] = None
    usability: Optional[float] = None
    updated_at: Optional[str] = None
    # Only ever set on AI suggestions. The ranker must never match against it;
    # several benchmark cases exist to keep that true.
    rationale: Optional[str] = None
    possible_source: Optional[str] = None

    @property
    def effective_verification(self) -> VerificationStatus:
        if self.verification_status is not None:
            return self.verification_status
        if self.source_type is SourceType.VERIFIED_EXTERNAL:
            return VerificationStatus.VERIFIED
        return VerificationStatus.UNVERIFIED


class Difficulty(str, Enum):
    """How hard a case is expected to be for the current ranker."""

    STANDARD = "standard"
    #: Missing or partial metadata on the right dataset.
    SPARSE_METADATA = "sparse_metadata"
    #: Requires synonym / equivalence knowledge.
    SEMANTIC = "semantic"
    #: A popular, well-documented, irrelevant dataset is in the pool.
    DISTRACTOR = "distractor"
    #: The correct dataset is absent: a retrieval failure, not a ranking one.
    RETRIEVAL = "retrieval"


class BenchmarkCase(BaseModel):
    """One reproducible ranking scenario with human-assigned relevance grades."""

    id: str
    description: str
    category: str
    requirements: RequirementSpec
    candidates: list[CandidateSpec]
    #: candidate key -> RelevanceGrade value.
    grades: dict[str, int]
    difficulty: Difficulty = Difficulty.STANDARD
    #: False when the pool deliberately lacks the grade-3 dataset, so a missing
    #: answer must be reported as a retrieval failure rather than a ranking one.
    retrieval_complete: bool = True
    notes: str = ""

    def grade_of(self, key: str) -> int:
        if key not in self.grades:
            raise KeyError(
                f"benchmark case {self.id!r} has no grade for candidate {key!r}; "
                "every candidate must be graded"
            )
        return self.grades[key]


class CompiledCase(BaseModel):
    """A benchmark case with its real schemas built.

    ``candidates_by_id`` keeps the grade join explicit and order-independent, so
    a future ranker that reorders or rewrites candidates cannot silently
    mis-associate a grade.
    """

    model_config = {"arbitrary_types_allowed": True}

    case: BenchmarkCase
    requirements: ProjectAnalysisData
    candidates: list["object"] = Field(default_factory=list)
    grades_by_id: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


class FactorContribution(BaseModel):
    """One factor's part of a candidate's score, as shown to a user."""

    name: str
    weight: float
    score: Optional[float] = Field(
        None, description="None means the factor could not be judged at all."
    )
    contribution: float
    detail: str = ""
    matched_terms: list[str] = Field(default_factory=list)
    unmatched_terms: list[str] = Field(default_factory=list)
    unknown_terms: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    @property
    def display(self) -> str:
        """How the factor should read in the diagnostic report."""
        if self.score is None:
            if self.unknown_terms:
                return f"UNKNOWN ({', '.join(self.unknown_terms)})"
            return "UNKNOWN"
        if self.contradictions:
            return f"{self.score:.2f} (contradiction: {self.contradictions[0]})"
        return f"{self.score:.2f}"


class RankedItem(BaseModel):
    """One ranked candidate with everything needed to diagnose its position."""

    rank: int
    candidate_id: str
    name: str
    grade: int
    score: float
    relevance_score: Optional[float] = None
    quality_score: Optional[float] = None
    source_type: SourceType
    verification_status: VerificationStatus
    factors: list[FactorContribution] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    def factor(self, name: str) -> Optional[FactorContribution]:
        for item in self.factors:
            if item.name == name:
                return item
        return None


class FailureCategory(str, Enum):
    """Why a case was ranked badly.

    A single failure can carry several categories: a candidate that ranks first
    for the wrong reason usually fails on more than one axis.
    """

    WRONG_DOMAIN = "WRONG_DOMAIN"
    WRONG_TASK = "WRONG_TASK"
    WRONG_TARGET = "WRONG_TARGET"
    MISSING_TARGET_EVIDENCE = "MISSING_TARGET_EVIDENCE"
    FEATURE_MISMATCH = "FEATURE_MISMATCH"
    FEATURE_UNKNOWN = "FEATURE_UNKNOWN"
    GENERIC_TERM_DOMINATION = "GENERIC_TERM_DOMINATION"
    POPULARITY_DOMINATION = "POPULARITY_DOMINATION"
    RECENCY_DOMINATION = "RECENCY_DOMINATION"
    AI_EVIDENCE_OVERTRUST = "AI_EVIDENCE_OVERTRUST"
    VERIFICATION_CONFUSION = "VERIFICATION_CONFUSION"
    CONTRADICTION_MISSED = "CONTRADICTION_MISSED"
    SEMANTIC_EQUIVALENCE_MISSED = "SEMANTIC_EQUIVALENCE_MISSED"
    QUERY_RECALL_FAILURE = "QUERY_RECALL_FAILURE"
    METADATA_INSUFFICIENT = "METADATA_INSUFFICIENT"
    #: A usable dataset was ranked below an unusable one.
    INVERSION = "INVERSION"


class CaseFailure(BaseModel):
    category: FailureCategory
    detail: str
    candidate_ids: list[str] = Field(default_factory=list)


class FailureKind(str, Enum):
    """Retrieval problem or ranking problem - the distinction that stops us from
    "fixing" ranking when the query never returned the answer."""

    NONE = "none"
    RANKING = "ranking"
    RETRIEVAL = "retrieval"


class CaseEvaluation(BaseModel):
    """Everything measured and diagnosed about one benchmark case."""

    case_id: str
    category: str
    difficulty: Difficulty
    description: str
    query: str
    grades: dict[str, int] = Field(default_factory=dict)
    ranked: list[RankedItem] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    failures: list[CaseFailure] = Field(default_factory=list)
    failure_kind: FailureKind = FailureKind.NONE
    #: Did the pool contain a grade-3 dataset at all?
    best_available_retrieved: bool = False
    #: Rank of the best available relevant candidate, if any was retrieved.
    best_available_rank: Optional[int] = None
    #: Requirement verdicts the ranker turned into a mismatch without evidence.
    suspected_unknown_as_mismatch: int = 0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class AblationResult(BaseModel):
    """Ranking quality with one signal removed."""

    disabled: str
    metrics: dict[str, float]
    deltas: dict[str, float]
    rank_changes: int = Field(
        0, description="Cases whose top-1 candidate changed."
    )


class SensitivityScenario(BaseModel):
    """One measured perturbation of one input."""

    name: str
    description: str
    case_id: str
    candidate_id: str
    before: dict[str, float]
    after: dict[str, float]
    rank_before: int
    rank_after: int
    score_before: float
    score_after: float
    score_delta: float
    rank_delta: int


class FactorAuditEntry(BaseModel):
    name: str
    semantic_meaning: str
    raw_range: str
    weight: float
    #: weight / 100: this factor's share of a full-scale score when every factor
    #: is counted.
    nominal_share: float
    #: Largest share of a final score this factor actually reached in the run.
    observed_max_share: float = 0.0
    #: Largest point contribution this factor actually reached in the run.
    observed_max_contribution: float = 0.0
    #: Largest ``contribution / weight`` this factor reached in the run. This is
    #: the direct test of denominator inflation: 1.0 means the factor was paid
    #: exactly its nominal weight, above 1.0 means the score mean rescaled the
    #: factor set. ``observed_max_share`` cannot answer that on its own - a
    #: candidate scoring 90 rather than 100 gives every factor a larger share of
    #: the achieved total than of the full scale, with no inflation anywhere.
    observed_max_ratio: float = 0.0
    max_contribution: float
    higher_is_better: bool
    unknown_possible: bool
    mismatch_possible: bool
    can_dominate: bool
    findings: list[str] = Field(default_factory=list)


class AuditFinding(BaseModel):
    """One thing the decomposition audit noticed."""

    severity: str
    subject: str
    detail: str


class AuditReport(BaseModel):
    max_total_weight: float
    contributions_sum_matches_score: bool
    max_abs_contribution_error: float
    duplicated_contributions: int = 0
    impossible_contributions: int = 0
    #: Contributions above their pro-rata share, because other factors were
    #: dropped from the denominator. Not an arithmetic error, but a documented
    #: bias in the explanation.
    inflated_contributions: int = 0
    factors: list[FactorAuditEntry] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    """The complete, machine-readable result of one evaluation run."""

    benchmark_version: str
    ranker: str
    case_count: int
    candidate_count: int
    graded_candidate_count: int
    metrics: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, float] = Field(default_factory=dict)
    retrieval_recall: float = Field(
        0.0, description="Share of cases whose pool contained a grade-3 dataset."
    )
    ranking_quality: dict[str, float] = Field(default_factory=dict)
    failure_taxonomy: dict[str, int] = Field(default_factory=dict)
    failure_kind_counts: dict[str, int] = Field(default_factory=dict)
    generated_at: Optional[datetime] = None
    per_case: list[CaseEvaluation] = Field(default_factory=list)
    ablations: list[AblationResult] = Field(default_factory=list)
    sensitivity: list[SensitivityScenario] = Field(default_factory=list)
    audit: Optional[AuditReport] = None