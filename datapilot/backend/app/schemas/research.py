"""Models for the optional "Research this problem" workflow.

Three kinds of statement never mix, and the models keep them apart:

* **verified literature**: papers, methods and metrics taken from retrieved web evidence, each with a URL;
* **DataPilot analysis**: deterministic ranking and recommendations computed from that evidence and from
  the project's own structured context;
* **AI interpretation**: optional Gemini prose and AI-inferred methods, always labelled ``ai_inferred``.

Unknown stays unknown: a missing author, year, DOI or citation count is ``None``, never a guess.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EvidenceKind(str, Enum):
    WEB_VERIFIED = "web_verified"          # read from a retrieved page/snippet
    SOURCE_METADATA = "source_metadata"    # derived from the URL or site (venue, arXiv id, DOI in URL)
    PAPER_TEXT = "paper_text"              # reserved: extracted from full text (not produced in the MVP)
    AI_INFERRED = "ai_inferred"            # a language model's inference, never verified
    UNKNOWN = "unknown"


class ResearchStatus(str, Enum):
    SUCCESS = "success"
    NEEDS_CLARIFICATION = "needs_clarification"
    NO_RESULTS = "no_results"
    SEARCH_UNAVAILABLE = "search_unavailable"


class ResearchProject(BaseModel):
    description: Optional[str] = Field(None, max_length=2000)
    domain: Optional[str] = Field(None, max_length=100)
    problem: Optional[str] = Field(None, max_length=200)
    task: Optional[str] = Field(None, max_length=60)
    target: Optional[str] = Field(None, max_length=100)
    features: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("description", "domain", "problem", "task", "target")
    @classmethod
    def blank_is_absent(cls, v: Optional[str]) -> Optional[str]:
        cleaned = " ".join(v.split()) if v else ""
        return cleaned or None


class ResearchDataset(BaseModel):
    name: Optional[str] = Field(None, max_length=200)
    columns: list[str] = Field(default_factory=list, max_length=500)
    task: Optional[str] = Field(None, max_length=60)
    target: Optional[str] = Field(None, max_length=100)
    rows: Optional[int] = Field(None, ge=0)


class ResearchOptions(BaseModel):
    max_papers: int = Field(8, ge=1, le=15)
    min_year: Optional[int] = Field(None, ge=1950, le=2100)
    use_ai: bool = Field(False, description="Add an AI-written interpretation of the evidence (one extra call).")
    confirmed: bool = Field(False, description="The user already picked/confirmed the reading: do not ask again.")
    ai_assist: bool = Field(
        True,
        description="Let the AI understand the question and judge paper relevance (two small calls; falls back silently).",
    )


class ResearchRequest(BaseModel):
    project: ResearchProject = Field(default_factory=ResearchProject)
    dataset: Optional[ResearchDataset] = None
    options: ResearchOptions = Field(default_factory=ResearchOptions)


# ---------------------------------------------------------------------------


class Paper(BaseModel):
    id: str
    title: str
    url: str
    source: str = Field(..., description="The site the result came from, e.g. 'arxiv.org'.")
    venue: Optional[str] = None
    authors: Optional[list[str]] = Field(None, description="None unless the retrieved evidence states them.")
    publication_year: Optional[int] = None
    year_evidence: EvidenceKind = EvidenceKind.UNKNOWN
    doi: Optional[str] = None
    abstract_snippet: Optional[str] = None
    citation_count: Optional[int] = Field(None, description="Never estimated; None when not retrieved.")
    paper_type: Optional[str] = Field(None, description="survey | applied | method: read from the title/abstract wording.")
    matched_queries: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list, description="Subject words of the question found in the title/abstract.")
    ai_relevance: Optional[int] = Field(None, ge=0, le=3, description="An AI model's 0-3 judgement of topical fit (not a fact about the paper).")
    ai_reason: Optional[str] = None
    relevance_score: float = Field(0.0, ge=0.0, le=100.0)
    relevance_reason: str = ""
    score_components: dict[str, float] = Field(default_factory=dict)
    methods: list[str] = Field(default_factory=list, description="Methods named in the retrieved text.")
    retrieval_status: str = "snippet_only"
    verification_status: str = "web_verified"
    evidence: list[EvidenceKind] = Field(default_factory=lambda: [EvidenceKind.WEB_VERIFIED])

    @model_validator(mode="after")
    def a_paper_must_be_traceable(self) -> "Paper":
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("a paper needs a traceable http(s) URL")
        if self.publication_year is not None and self.year_evidence is EvidenceKind.UNKNOWN:
            raise ValueError("a publication year needs an evidence kind")
        return self


class MethodSupport(BaseModel):
    method: str
    verified_paper_count: int = Field(..., ge=0)
    paper_ids: list[str] = Field(default_factory=list)
    evidence: EvidenceKind = EvidenceKind.WEB_VERIFIED
    note: str = "Counts papers whose retrieved text mentions the method; frequency is not a ranking of quality."


class ReportedMetric(BaseModel):
    """A figure a source reported in *its own* experimental setting."""

    metric: str
    value: str
    context: Optional[str] = Field(None, description="Dataset/setting as stated by the source, else None.")
    method: Optional[str] = None
    paper_id: str
    paper_title: str
    url: str
    evidence: EvidenceKind = EvidenceKind.WEB_VERIFIED
    statement: str = Field(..., description="How to read it: reported by the source, not a prediction for the user.")


class Level(str, Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    LIMITED = "limited"
    NONE = "none"
    UNKNOWN = "unknown"


class AlgorithmRecommendation(BaseModel):
    rank: int = Field(..., ge=1)
    method: str
    literature_support: Level
    dataset_compatibility: Level
    evidence_paper_count: int = Field(..., ge=0)
    paper_ids: list[str] = Field(default_factory=list)
    why: str
    cautions: list[str] = Field(default_factory=list)


class AIInterpretation(BaseModel):
    """Optional Gemini prose. Everything here is interpretation, not verified literature."""

    landscape: str
    approaches: list[str] = Field(default_factory=list)
    comparison: str = ""
    fit_for_project: str = ""
    limitations: list[str] = Field(default_factory=list)
    ai_inferred_methods: list[str] = Field(default_factory=list)
    evidence: EvidenceKind = EvidenceKind.AI_INFERRED


class ProvenanceNote(BaseModel):
    source: str
    evidence: EvidenceKind
    note: str


class GeminiResearchUsage(BaseModel):
    requested: bool = False
    calls_made: int = Field(0, ge=0, le=5)
    assisted: list[str] = Field(default_factory=list, description="AI steps that helped before synthesis.")
    status: str = "not_requested"   # not_requested | used | rejected | failed | unavailable | disabled
    note: Optional[str] = None


class ResearchSummary(BaseModel):
    queries_used: int
    results_retrieved: int
    papers_found: int
    papers_selected: int
    results_rejected_as_non_academic: int = 0
    excluded_off_topic: int = 0
    search_provider: str


class Interpretation(BaseModel):
    """How DataPilot understood the question, shown so the user can correct it."""

    problem: Optional[str] = None
    core_terms: list[str] = Field(default_factory=list)
    context_terms: list[str] = Field(default_factory=list)
    corrections: dict[str, str] = Field(default_factory=dict)
    ai_assisted: bool = False
    outcome: Optional[str] = Field(None, description="What is being predicted or estimated, in a short phrase.")
    task: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    dataset_check: Optional["DatasetCheck"] = None
    options: list[str] = Field(default_factory=list, description="Readings to pick from when the question is ambiguous.")


class DatasetCheck(BaseModel):
    """Whether the user's wording and the dataset's columns agree. A mismatch is shown, never hidden."""

    status: str = Field(..., description="consistent | mismatch | unknown")
    note: str = ""
    source: str = "datapilot"  # datapilot (deterministic) | ai_inferred


class ResearchResponse(BaseModel):
    status: ResearchStatus
    request_id: str
    api_version: str = "v1"
    message: Optional[str] = Field(None, description="What is needed or what went wrong, in plain words.")
    queries: list[str] = Field(default_factory=list)
    interpretation: Optional[Interpretation] = None
    research_summary: ResearchSummary
    datapilot_summary: str = ""
    papers: list[Paper] = Field(default_factory=list)
    methods: list[MethodSupport] = Field(default_factory=list)
    benchmarks: list[ReportedMetric] = Field(default_factory=list)
    algorithm_recommendations: list[AlgorithmRecommendation] = Field(default_factory=list)
    ai_interpretation: Optional[AIInterpretation] = None
    gemini: GeminiResearchUsage = Field(default_factory=GeminiResearchUsage)
    limitations: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceNote] = Field(default_factory=list)


# What Gemini may return (validated before any of it is used).
class LLMResearchSynthesis(BaseModel):
    landscape: str
    approaches: list[str]
    comparison: str
    fit_for_project: str
    limitations: list[str]
    ai_inferred_methods: list[str]
