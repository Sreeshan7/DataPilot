"""Phase 4 models: what DataPilot knows about a dataset the user already has.

There are two evidence paths and they must never be confused:

``FILE_PARSED``
    The user supplied the file and DataPilot parsed it. Schema, statistics and quality
    findings are facts about *that file*.

``EXTERNAL_METADATA``
    The user named a dataset or gave a URL. DataPilot researched a dataset registry
    and holds only what that registry published. There is no row-level evidence, so
    there are no statistics, no quality findings and no feature importance - and the
    models below refuse to represent them (see ``DatasetAnalysis``).

Every claim carries an :class:`EvidenceLevel`. The levels are not a scale to be
averaged: a registry's published schema, a name-based guess and a model's suggestion
are different kinds of thing, and collapsing them is how an unverified statement
becomes a "fact".
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceLevel(str, Enum):
    """How a statement about a dataset is known, strongest first."""

    DATAPILOT_PARSED = "datapilot_parsed"
    """DataPilot read the user's file and computed this itself."""

    REGISTRY_VERIFIED = "registry_verified"
    """Published as structured metadata by a dataset registry (Kaggle, Hugging Face,
    OpenML, UCI). Not parsed by DataPilot: the registry's word for it."""

    EXTERNAL_SOURCE_VERIFIED = "external_source_verified"
    """Published by a trusted non-registry public source (an official data
    dictionary, a paper). Reserved: no code path produces it yet, and it is never
    claimed by anything that did not read such a source."""

    INFERRED = "inferred"
    """DataPilot's own deterministic reading of published text, e.g. column names
    pulled out of an owner's prose description."""

    HEURISTIC = "heuristic"
    """A rule of thumb over names or shapes (a column called ``customer_id`` looks like
    an identifier). Useful, labelled, never presented as established."""

    AI_INFERRED = "ai_inferred"
    """Suggested by a generative model."""

    UNKNOWN = "unknown"


EVIDENCE_STRENGTH: dict[EvidenceLevel, int] = {
    EvidenceLevel.DATAPILOT_PARSED: 6,
    EvidenceLevel.REGISTRY_VERIFIED: 5,
    EvidenceLevel.EXTERNAL_SOURCE_VERIFIED: 4,
    EvidenceLevel.INFERRED: 3,
    EvidenceLevel.HEURISTIC: 2,
    EvidenceLevel.AI_INFERRED: 1,
    EvidenceLevel.UNKNOWN: 0,
}

#: Levels that state a fact about the dataset rather than a guess.
VERIFIED_LEVELS: frozenset[EvidenceLevel] = frozenset(
    {
        EvidenceLevel.DATAPILOT_PARSED,
        EvidenceLevel.REGISTRY_VERIFIED,
        EvidenceLevel.EXTERNAL_SOURCE_VERIFIED,
    }
)


class EvidenceItem(BaseModel):
    """One statement and how it is known."""

    statement: str = Field(..., min_length=1)
    level: EvidenceLevel


class ProvenanceRecord(BaseModel):
    """One important claim, where it came from and how far to trust it."""

    claim: str
    source: str
    evidence_level: EvidenceLevel
    confidence: float = Field(..., ge=0.0, le=1.0)
    retrieved_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


class InputType(str, Enum):
    UPLOADED_FILE = "uploaded_file"
    PUBLIC_DATASET_NAME = "public_dataset_name"
    DATASET_URL = "dataset_url"
    AMBIGUOUS_DATASET_REFERENCE = "ambiguous_dataset_reference"
    NO_DATASET_REFERENCE = "no_dataset_reference"


class ResolutionStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    """An uploaded file needs no external resolution."""

    RESOLVED = "resolved"
    """Exactly one dataset matches."""

    PROBABLE = "probable"
    """One dataset was chosen, but others share the name and may be what was meant."""

    AMBIGUOUS = "ambiguous"
    """Several datasets plausibly match and none can be chosen; candidates returned."""

    UNRESOLVED = "unresolved"
    """Nothing matched."""

    UNSUPPORTED = "unsupported"
    """The reference points at a source DataPilot does not research."""


class DatasetAlternative(BaseModel):
    """Another dataset that shares the referenced name."""

    name: str
    source: str
    identifier: str
    url: Optional[str] = None
    column_count: Optional[int] = None
    row_count: Optional[int] = None


class DatasetInputResolution(BaseModel):
    """What the user pointed at, and how well DataPilot could pin it down."""

    input_type: InputType
    normalized_dataset_name: Optional[str] = None
    source: Optional[str] = Field(
        None, description="'upload', 'kaggle', 'huggingface', 'uci', 'openml', ..."
    )
    source_url: Optional[str] = None
    source_identifier: Optional[str] = None
    resolution_status: ResolutionStatus = ResolutionStatus.NOT_APPLICABLE
    resolution_confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    available_metadata: list[str] = Field(
        default_factory=list, description="Groups of metadata that were retrieved."
    )
    raw_file_available: bool = False
    schema_available: bool = False
    schema_evidence_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    alternatives: list[DatasetAlternative] = Field(default_factory=list)
    needs_confirmation: bool = False
    clarification: Optional[str] = None

    @model_validator(mode="after")
    def a_schema_needs_a_level(self) -> "DatasetInputResolution":
        if self.schema_available and self.schema_evidence_level is EvidenceLevel.UNKNOWN:
            raise ValueError("a schema cannot be available at evidence level UNKNOWN")
        if not self.schema_available and self.schema_evidence_level in VERIFIED_LEVELS:
            raise ValueError("no schema is available, so no verified schema level can be claimed")
        if self.raw_file_available and self.input_type is not InputType.UPLOADED_FILE:
            raise ValueError("a raw file is only ever available for an uploaded file")
        return self


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class ColumnRename(BaseModel):
    index: int
    original_name: Optional[str]
    normalized_name: str
    reason: str


class IngestionInfo(BaseModel):
    """How the uploaded file was read. Nothing here exposes a filesystem path."""

    file_type: str
    display_name: str = Field(..., description="Sanitised for display; never used as a path.")
    size_bytes: int = Field(..., ge=0)
    sheet_name: Optional[str] = None
    sheets_available: list[str] = Field(default_factory=list)
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    header_detected: bool = True
    rows_read: int = Field(..., ge=0)
    truncated: bool = False
    truncation_reason: Optional[str] = None
    malformed_rows: int = Field(0, ge=0)
    columns_renamed: list[ColumnRename] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Schema and column profiles
# ---------------------------------------------------------------------------


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class SchemaColumn(BaseModel):
    """One column as the schema source states it."""

    original_name: Optional[str] = Field(
        None, description="Exactly as written in the source; None for a synthetic name."
    )
    normalized_name: str
    data_type: Optional[str] = None
    role: Optional[str] = Field(
        None, description="The registry's own role label, e.g. 'target' or 'feature'."
    )
    description: Optional[str] = None
    evidence: EvidenceLevel


class SchemaInfo(BaseModel):
    """The dataset's column structure and exactly how it is known."""

    level: EvidenceLevel = EvidenceLevel.UNKNOWN
    source: str = "none"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    column_count: Optional[int] = Field(None, ge=0)
    row_count: Optional[int] = Field(None, ge=0)
    row_count_evidence: EvidenceLevel = EvidenceLevel.UNKNOWN
    columns: list[SchemaColumn] = Field(default_factory=list)
    note: Optional[str] = None

    @model_validator(mode="after")
    def evidence_must_agree_with_content(self) -> "SchemaInfo":
        if self.level is EvidenceLevel.UNKNOWN and self.columns:
            raise ValueError("an UNKNOWN schema cannot list columns")
        if self.columns and any(c.evidence is EvidenceLevel.UNKNOWN for c in self.columns):
            raise ValueError("a listed column cannot have UNKNOWN evidence")
        if self.row_count is not None and self.row_count_evidence is EvidenceLevel.UNKNOWN:
            raise ValueError("a row count needs an evidence level")
        if self.row_count is None and self.row_count_evidence is not EvidenceLevel.UNKNOWN:
            raise ValueError("no row count was given, so its evidence must be UNKNOWN")
        return self


class ValueCount(BaseModel):
    value: str
    count: int = Field(..., ge=0)
    share: float = Field(..., ge=0.0, le=1.0)


class NumericStats(BaseModel):
    min: float
    max: float
    mean: float
    median: float
    std: Optional[float] = Field(None, description="Sample standard deviation; None for n < 2.")
    quantiles: dict[str, float] = Field(default_factory=dict)
    zero_count: int = 0
    negative_count: int = 0
    is_integer: bool = False


class CategoricalStats(BaseModel):
    cardinality: str = Field(..., description="'low' | 'medium' | 'high'")
    top_values: list[ValueCount] = Field(default_factory=list)
    values_suppressed: bool = Field(
        False, description="True when the column looks identifying; values are withheld."
    )


class DatetimeStats(BaseModel):
    min: datetime
    max: datetime
    granularity: str = Field(..., description="e.g. 'daily', 'hourly', 'irregular'")


class TextStats(BaseModel):
    mean_length: float
    max_length: int


class ColumnFlag(str, Enum):
    CONSTANT = "constant"
    NEAR_CONSTANT = "near_constant"
    ALL_MISSING = "all_missing"
    POSSIBLE_IDENTIFIER = "possible_identifier"
    POSSIBLE_SENSITIVE_FIELD = "possible_sensitive_field"
    HIGH_CARDINALITY = "high_cardinality"
    MIXED_TYPES = "mixed_types"
    FREE_TEXT = "free_text"


class ColumnProfile(BaseModel):
    """Statistics for one column of a parsed file. Always ``DATAPILOT_PARSED``."""

    index: int
    original_name: Optional[str]
    normalized_name: str
    inferred_type: ColumnType
    physical_type: Optional[str] = None
    count: int = Field(..., ge=0, description="Non-missing values.")
    missing_count: int = Field(..., ge=0)
    missing_pct: float = Field(..., ge=0.0, le=100.0)
    unique_count: int = Field(..., ge=0)
    uniqueness_ratio: float = Field(..., ge=0.0, le=1.0)
    flags: list[ColumnFlag] = Field(default_factory=list)
    numeric: Optional[NumericStats] = None
    categorical: Optional[CategoricalStats] = None
    datetime: Optional[DatetimeStats] = None
    text: Optional[TextStats] = None
    evidence_level: EvidenceLevel = EvidenceLevel.DATAPILOT_PARSED

    @model_validator(mode="after")
    def statistics_must_fit_the_type(self) -> "ColumnProfile":
        if self.evidence_level is not EvidenceLevel.DATAPILOT_PARSED:
            raise ValueError("column statistics are only ever DATAPILOT_PARSED")
        if self.numeric is not None and self.inferred_type is not ColumnType.NUMERIC:
            raise ValueError("numeric statistics on a non-numeric column")
        if self.datetime is not None and self.inferred_type is not ColumnType.DATETIME:
            raise ValueError("datetime statistics on a non-datetime column")
        return self


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class QualityCode(str, Enum):
    MISSING_VALUES = "missing_values"
    HIGH_MISSINGNESS = "high_missingness"
    DUPLICATE_ROWS = "duplicate_rows"
    DUPLICATE_COLUMNS = "duplicate_columns"
    CONSTANT_COLUMN = "constant_column"
    NEAR_CONSTANT_COLUMN = "near_constant_column"
    POSSIBLE_IDENTIFIER = "possible_identifier"
    POSSIBLE_SENSITIVE_FIELD = "possible_sensitive_field"
    HIGH_CARDINALITY_CATEGORICAL = "high_cardinality_categorical"
    TYPE_INCONSISTENCY = "type_inconsistency"
    MALFORMED_DATES = "malformed_dates"
    INVALID_NUMERIC_VALUES = "invalid_numeric_values"
    POSSIBLE_OUTLIER = "possible_outlier"
    CLASS_IMBALANCE = "class_imbalance"
    LEAKAGE_SIGNAL = "leakage_signal"
    POTENTIAL_LEAKAGE = "potential_leakage"
    UNUSABLE_COLUMN = "unusable_column"
    MALFORMED_ROWS = "malformed_rows"
    TRUNCATED = "truncated"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class QualityFinding(BaseModel):
    """One finding, with the numbers that justify it."""

    code: QualityCode
    severity: FindingSeverity
    column: Optional[str] = None
    columns: list[str] = Field(default_factory=list)
    detail: str = Field(..., min_length=1)
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_level: EvidenceLevel


class QualityReport(BaseModel):
    findings: list[QualityFinding] = Field(default_factory=list)
    overall_missing_pct: float = Field(0.0, ge=0.0, le=100.0)
    duplicate_row_count: Optional[int] = None
    duplicate_row_pct: Optional[float] = None
    usable_column_count: int = 0
    thresholds: dict[str, float] = Field(
        default_factory=dict, description="The rule-of-thumb limits the findings used."
    )

    def by_code(self, code: QualityCode) -> list[QualityFinding]:
        return [f for f in self.findings if f.code is code]


# ---------------------------------------------------------------------------
# Targets and tasks
# ---------------------------------------------------------------------------


class CandidateState(str, Enum):
    STRONG_CANDIDATE = "strong_candidate"
    POSSIBLE = "possible"
    WEAK = "weak"
    UNKNOWN = "unknown"


class TargetKind(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    CONTINUOUS = "continuous"
    UNKNOWN = "unknown"


class TargetCandidate(BaseModel):
    """A column that may be the prediction target, with the case for and against."""

    column: str
    state: CandidateState
    confidence: float = Field(..., ge=0.0, le=1.0)
    score: float
    target_kind: TargetKind = TargetKind.UNKNOWN
    evidence: list[EvidenceItem] = Field(default_factory=list)
    against: list[str] = Field(default_factory=list)
    why_target: str = ""
    why_not: str = ""
    user_specified: bool = False


class MLTask(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RANKING = "ranking"


SUPERVISED_TASKS: frozenset[MLTask] = frozenset(
    {
        MLTask.BINARY_CLASSIFICATION,
        MLTask.MULTICLASS_CLASSIFICATION,
        MLTask.MULTILABEL_CLASSIFICATION,
        MLTask.REGRESSION,
        MLTask.TIME_SERIES_FORECASTING,
        MLTask.RANKING,
    }
)


class TaskCandidate(BaseModel):
    task: MLTask
    target: Optional[str] = None
    state: CandidateState
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def supervised_needs_a_target(self) -> "TaskCandidate":
        """Never claim supervised learning without a plausible target."""
        if self.task in SUPERVISED_TASKS and not self.target:
            raise ValueError(f"{self.task.value} is supervised and needs a target column")
        return self


# ---------------------------------------------------------------------------
# External dataset research
# ---------------------------------------------------------------------------


class WebSource(BaseModel):
    """A page a web search cited, shown so the user can check it themselves."""

    title: str = ""
    url: str


class ExternalMetadata(BaseModel):
    """What a registry published about a dataset. Never row-level facts it did not state."""

    source: str
    identifier: str
    name: str
    url: Optional[str] = None
    description: Optional[str] = None
    publisher: Optional[str] = None
    license: Optional[str] = None
    size_bytes: Optional[int] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    files: list[str] = Field(default_factory=list)
    file_types: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    registry_target_columns: list[str] = Field(
        default_factory=list, description="Columns the registry itself designates as the target."
    )
    evidence_level: EvidenceLevel = EvidenceLevel.REGISTRY_VERIFIED
    retrieved_at: Optional[datetime] = None
    row_count_evidence: EvidenceLevel = EvidenceLevel.REGISTRY_VERIFIED
    web_sources: list[WebSource] = Field(
        default_factory=list,
        description="Pages a web search cited; present only when the structure was found that way (AI_INFERRED).",
    )
    schema_columns: list[SchemaColumn] = Field(default_factory=list)
    schema_level: EvidenceLevel = EvidenceLevel.UNKNOWN
    schema_source: str = "none"
    schema_note: Optional[str] = None
    registry_statistics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Aggregate figures the registry itself published (e.g. OpenML's "
            "NumberOfMissingValues). They are the registry's computation, not DataPilot's."
        ),
    )


class ExternalConflict(BaseModel):
    """Two sources disagree about the same fact."""

    field: str
    values: dict[str, str]
    note: str


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class AnalysisMode(str, Enum):
    FILE_PARSED = "file_parsed"
    EXTERNAL_METADATA = "external_metadata"


class DatasetIdentity(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    identifier: Optional[str] = None
    resolution_confidence: float = Field(0.0, ge=0.0, le=1.0)


class AnalysisMetadata(BaseModel):
    analysis_id: str
    mode: AnalysisMode
    started_at: datetime
    duration_ms: float = 0.0
    limits: dict[str, Any] = Field(default_factory=dict)
    deterministic: bool = True


class DatasetAnalysis(BaseModel):
    """The evidence-aware result of analysing one dataset (Phase 4).

    The validator below is the structural form of the product's most important rule:
    *if DataPilot did not parse the file, it must not report anything that only
    parsing can produce.* A response that violates it cannot be constructed.
    """

    mode: AnalysisMode
    dataset_identity: DatasetIdentity
    input_resolution: DatasetInputResolution
    ingestion: Optional[IngestionInfo] = None
    external: Optional[ExternalMetadata] = None
    conflicts: list[ExternalConflict] = Field(default_factory=list)
    schema_info: SchemaInfo = Field(alias="schema")
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    quality: Optional[QualityReport] = None
    target_candidates: list[TargetCandidate] = Field(default_factory=list)
    task_candidates: list[TaskCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    metadata: AnalysisMetadata

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def raw_data_claims_require_the_raw_data(self) -> "DatasetAnalysis":
        if self.mode is AnalysisMode.FILE_PARSED:
            if self.ingestion is None:
                raise ValueError("a parsed-file analysis must describe its ingestion")
            if self.schema_info.level is not EvidenceLevel.DATAPILOT_PARSED:
                raise ValueError("a parsed file's schema must be DATAPILOT_PARSED")
            return self

        # EXTERNAL_METADATA: DataPilot saw no rows.
        if self.ingestion is not None:
            raise ValueError("an external-metadata analysis cannot have ingestion details")
        if self.column_profiles:
            raise ValueError("column statistics need the raw file; none was parsed")
        if self.quality is not None:
            raise ValueError("a quality report needs the raw file; none was parsed")
        if EvidenceLevel.DATAPILOT_PARSED in {
            self.schema_info.level,
            self.schema_info.row_count_evidence,
        }:
            raise ValueError("DataPilot did not parse a file, so nothing is DATAPILOT_PARSED")
        for candidate in self.target_candidates:
            if any(e.level is EvidenceLevel.DATAPILOT_PARSED for e in candidate.evidence):
                raise ValueError("target evidence cannot be DATAPILOT_PARSED without a file")
        return self
