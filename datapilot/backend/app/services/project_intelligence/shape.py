"""A compact, value-free view of an analysed dataset, shared by the Phase 5 generators.

Model recommendations, project ideas and goal compatibility all need the same handful of facts
(how many rows, which columns are usable, how messy, what was flagged). Reading them from the
analysis once, here, means none of those steps ever touches a raw cell, and they cannot disagree
about which columns exist: every name they emit comes from ``DataShape.columns``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.schemas.dataset_analysis import (
    AnalysisMode,
    ColumnFlag,
    ColumnType,
    DatasetAnalysis,
    EvidenceLevel,
    QualityCode,
    QualityFinding,
)

_REGISTRY_KINDS = {
    "continuous": ColumnType.NUMERIC, "numeric": ColumnType.NUMERIC, "real": ColumnType.NUMERIC,
    "float": ColumnType.NUMERIC, "double": ColumnType.NUMERIC, "integer": ColumnType.NUMERIC,
    "int": ColumnType.NUMERIC, "int64": ColumnType.NUMERIC, "float64": ColumnType.NUMERIC,
    "categorical": ColumnType.CATEGORICAL, "nominal": ColumnType.CATEGORICAL, "category": ColumnType.CATEGORICAL,
    "class_label": ColumnType.CATEGORICAL, "string": ColumnType.CATEGORICAL, "binary": ColumnType.BOOLEAN,
    "boolean": ColumnType.BOOLEAN, "date": ColumnType.DATETIME, "datetime": ColumnType.DATETIME,
    "timestamp": ColumnType.DATETIME,
}


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    kind: ColumnType
    missing_pct: Optional[float] = None
    identifier: bool = False
    sensitive: bool = False
    constant: bool = False
    high_cardinality: bool = False
    #: The registry marked it as a row identifier / to be ignored (external schemas only).
    registry_role: Optional[str] = None

    @property
    def usable_feature(self) -> bool:
        """Could this column be a model input as it stands (set aside: ids, constants, free text)?"""
        return not (self.identifier or self.constant or self.registry_role in ("row_id", "ignore")) \
            and self.kind not in (ColumnType.TEXT,)


@dataclass
class DataShape:
    parsed: bool
    n_rows: Optional[int]
    columns: list[ColumnInfo]
    schema_level: EvidenceLevel
    findings: list[QualityFinding] = field(default_factory=list)
    truncated: bool = False

    @property
    def names(self) -> set[str]:
        return {c.name for c in self.columns}

    def column(self, name: str) -> Optional[ColumnInfo]:
        return next((c for c in self.columns if c.name == name), None)

    def of_kind(self, *kinds: ColumnType) -> list[ColumnInfo]:
        return [c for c in self.columns if c.kind in kinds]

    def findings_with(self, *codes: QualityCode) -> list[QualityFinding]:
        return [f for f in self.findings if f.code in codes]

    @property
    def has_schema(self) -> bool:
        return bool(self.columns)

    @property
    def max_missing_pct(self) -> float:
        return max((c.missing_pct or 0.0 for c in self.columns), default=0.0)


def shape_of(analysis: DatasetAnalysis) -> DataShape:
    findings = list(analysis.quality.findings) if analysis.quality else []
    if analysis.mode is AnalysisMode.FILE_PARSED:
        columns = [
            ColumnInfo(
                name=p.normalized_name, kind=p.inferred_type, missing_pct=p.missing_pct,
                identifier=ColumnFlag.POSSIBLE_IDENTIFIER in p.flags,
                sensitive=ColumnFlag.POSSIBLE_SENSITIVE_FIELD in p.flags,
                constant=ColumnFlag.CONSTANT in p.flags or ColumnFlag.ALL_MISSING in p.flags,
                high_cardinality=ColumnFlag.HIGH_CARDINALITY in p.flags,
            )
            for p in analysis.column_profiles
        ]
        info = analysis.ingestion
        return DataShape(
            parsed=True, n_rows=info.rows_read if info else analysis.schema_info.row_count, columns=columns,
            schema_level=EvidenceLevel.DATAPILOT_PARSED, findings=findings,
            truncated=bool(info and info.truncated))
    columns = [
        ColumnInfo(
            name=c.normalized_name,
            kind=_REGISTRY_KINDS.get((c.data_type or "").lower(), ColumnType.UNKNOWN),
            identifier=c.role == "row_id",
            registry_role=c.role,
        )
        for c in analysis.schema_info.columns
    ]
    return DataShape(
        parsed=False, n_rows=analysis.schema_info.row_count, columns=columns,
        schema_level=analysis.schema_info.level, findings=[])
