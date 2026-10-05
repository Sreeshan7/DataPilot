"""The analysis of a parsed file: profile, then targets, then quality, then tasks.

The order is not arbitrary. Quality findings about class imbalance and leakage need to know
which columns might be the target, and the task list needs both the target candidates and
the column types, so each stage feeds the next and nothing is computed twice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    EvidenceLevel,
    QualityReport,
    SchemaColumn,
    SchemaInfo,
    TargetCandidate,
    TaskCandidate,
)
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisError
from app.services.dataset_analysis.ingestion import ParsedTable
from app.services.dataset_analysis.profiler import TypedColumn, profile_table
from app.services.dataset_analysis.quality import analyze_quality
from app.services.dataset_analysis.safety import Deadline
from app.core import progress
from app.services.dataset_analysis.targets import detect_targets
from app.services.dataset_analysis.tasks import detect_tasks

MAX_LISTED_COLUMNS = 50


@dataclass
class ParsedResult:
    table: ParsedTable
    columns: list[TypedColumn]
    n_rows: int
    targets: list[TargetCandidate]
    tasks: list[TaskCandidate]
    quality: QualityReport
    schema: SchemaInfo


def resolve_target_hint(columns: list[TypedColumn], hint: Optional[str]) -> Optional[str]:
    """The normalised column name the user meant, or a clear error if there is none.

    Matches the normalised name, the original header, and a normalised form of what the
    user typed, case-insensitively. A hint that matches nothing is an error, never silently
    ignored: acting on a target the user did not choose would be worse than stopping.
    """
    if hint is None or not hint.strip():
        return None
    wanted = hint.strip()
    folded = wanted.lower()
    squashed = re.sub(r"[^0-9a-zA-Z]+", "_", wanted).strip("_").lower()
    for column in columns:
        if column.name == wanted or column.name == squashed:
            return column.name
        if column.original is not None and column.original.strip().lower() == folded:
            return column.name
    available = [c.name for c in columns][:MAX_LISTED_COLUMNS]
    raise AnalysisError(
        ErrorCode.TARGET_COLUMN_NOT_FOUND,
        "The requested target column does not exist in this dataset.",
        details={"columns_available": available, "total_columns": len(columns)},
    )


def schema_from_parsed(columns: list[TypedColumn], table: ParsedTable) -> SchemaInfo:
    note = None
    if table.info.truncated:
        note = "row_count is the number of rows analysed, not the total in the file."
    return SchemaInfo(
        level=EvidenceLevel.DATAPILOT_PARSED,
        source="uploaded file",
        confidence=1.0,
        column_count=len(columns),
        row_count=table.info.rows_read,
        row_count_evidence=EvidenceLevel.DATAPILOT_PARSED,
        columns=[
            SchemaColumn(
                original_name=c.original,
                normalized_name=c.name,
                data_type=c.profile.inferred_type.value if c.profile else None,
                evidence=EvidenceLevel.DATAPILOT_PARSED,
            )
            for c in columns
        ],
        note=note,
    )


def analyze_table(
    table: ParsedTable,
    settings: Settings,
    deadline: Optional[Deadline] = None,
    target_hint: Optional[str] = None,
) -> ParsedResult:
    """Run every Phase 4 analysis over a parsed file."""
    n_rows = table.info.rows_read
    progress.done("read", f"{n_rows:,} rows × {len(table.frame.columns)} columns", label="Read your file")
    progress.start("profile", "Profiling every column")
    columns = profile_table(table, settings, deadline)
    progress.done("profile", f"{len(columns)} columns typed")
    hint = resolve_target_hint(columns, target_hint)
    if deadline is not None:
        deadline.check("target detection")
    progress.start("targets", "Looking for what could be predicted")
    targets = detect_targets(columns, n_rows, settings, hint)
    progress.done("targets", f"{len(targets)} candidate target(s)")
    if deadline is not None:
        deadline.check("quality analysis")
    progress.start("quality", "Checking data quality")
    quality = analyze_quality(columns, n_rows, table.info, targets, settings, deadline)
    progress.done("quality", f"{len(quality.findings)} finding(s)")
    progress.start("tasks", "Working out which tasks fit")
    tasks = detect_tasks(columns, targets, n_rows, settings)
    progress.done("tasks", f"{len(tasks)} task(s)")
    return ParsedResult(
        table=table, columns=columns, n_rows=n_rows, targets=targets, tasks=tasks,
        quality=quality, schema=schema_from_parsed(columns, table),
    )
