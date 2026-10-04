"""Evidence-backed data-quality findings for a parsed file.

Each finding states what was measured, with the numbers, and how sure we are. The wording is
chosen to match the strength of the evidence:

``LEAKAGE_SIGNAL``     a single feature predicts the target almost perfectly (measured);
``POTENTIAL_LEAKAGE``  a feature is *named* like something recorded after the outcome and is
                       also strongly associated with the target (a hint, never a verdict);
``POSSIBLE_OUTLIER``   outside Tukey's 1.5 x IQR fences, which flags unusual values, not errors.

Nothing here edits, drops or imputes anything: a finding reports and leaves the decision to
the user. All thresholds come from ``Settings`` and are echoed in the report.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import numpy as np

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    CandidateState,
    ColumnFlag,
    ColumnType,
    EvidenceLevel,
    FindingSeverity,
    IngestionInfo,
    QualityCode,
    QualityFinding,
    QualityReport,
    TargetCandidate,
)
from app.services.dataset_analysis.association import association, sample_indices
from app.services.dataset_analysis.profiler import TypedColumn
from app.services.dataset_analysis.safety import Deadline
from app.services.dataset_analysis.targets import class_counts

parsed = EvidenceLevel.DATAPILOT_PARSED
heuristic = EvidenceLevel.HEURISTIC

#: Leakage is checked against this many of the best target candidates.
LEAKAGE_TARGETS = 2
#: Features checked against a target for leakage.
LEAKAGE_MAX_FEATURES = 60
MIN_ROWS_FOR_OUTLIERS = 20
#: Association above which a *suspiciously named* feature is called potential leakage.
POTENTIAL_LEAKAGE_ASSOCIATION = 0.6

_SUSPICIOUS_NAME_TOKENS = frozenset(
    {"final", "outcome", "result", "results", "total", "after", "post", "resulting", "decision",
     "approved", "verdict", "settled"}
)

_SEVERITY_ORDER = {FindingSeverity.CRITICAL: 0, FindingSeverity.WARNING: 1, FindingSeverity.INFO: 2}


def _finding(
    code: QualityCode,
    severity: FindingSeverity,
    detail: str,
    *,
    column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    evidence: Optional[dict] = None,
    confidence: float = 1.0,
    level: EvidenceLevel = parsed,
) -> QualityFinding:
    return QualityFinding(
        code=code, severity=severity, detail=detail, column=column,
        columns=columns or ([column] if column else []), evidence=evidence or {},
        confidence=confidence, evidence_level=level,
    )


def _missing_findings(columns: list[TypedColumn], settings: Settings) -> list[QualityFinding]:
    out = []
    threshold = settings.quality_high_missing_ratio * 100
    for column in columns:
        profile = column.profile
        if profile is None or profile.missing_count == 0 or ColumnFlag.ALL_MISSING in profile.flags:
            continue
        evidence = {"missing_count": profile.missing_count, "missing_pct": round(profile.missing_pct, 2)}
        if profile.missing_pct >= threshold:
            severity = FindingSeverity.CRITICAL if profile.missing_pct >= 90 else FindingSeverity.WARNING
            out.append(_finding(
                QualityCode.HIGH_MISSINGNESS, severity,
                f"'{column.name}' is {profile.missing_pct:.1f}% missing "
                f"(the rule-of-thumb limit is {threshold:.0f}%), so imputing it is unlikely to be reliable.",
                column=column.name, evidence={**evidence, "threshold_pct": threshold}))
        else:
            severity = FindingSeverity.WARNING if profile.missing_pct >= 5 else FindingSeverity.INFO
            out.append(_finding(
                QualityCode.MISSING_VALUES, severity,
                f"'{column.name}' has {profile.missing_count} missing value(s) ({profile.missing_pct:.1f}%).",
                column=column.name, evidence=evidence))
    return out


def _duplicate_rows(columns: list[TypedColumn], n_rows: int) -> tuple[Optional[QualityFinding], int]:
    if n_rows < 2 or not columns:
        return None, 0
    import pandas as pd

    frame = pd.DataFrame(
        {c.name: [None if v is None else str(v) for v in c.values] for c in columns}, dtype=object
    )
    duplicated = int(frame.duplicated(keep="first").sum())
    if duplicated == 0:
        return None, 0
    pct = 100.0 * duplicated / n_rows
    severity = FindingSeverity.WARNING if pct >= 1.0 else FindingSeverity.INFO
    return _finding(
        QualityCode.DUPLICATE_ROWS, severity,
        f"{duplicated} row(s) ({pct:.2f}%) exactly repeat an earlier row.",
        evidence={"duplicate_rows": duplicated, "duplicate_pct": round(pct, 4)},
    ), duplicated


def _duplicate_columns(columns: list[TypedColumn]) -> list[QualityFinding]:
    groups: dict[tuple, list[str]] = {}
    for column in columns:
        if ColumnFlag.ALL_MISSING in column.flags:
            continue
        key = tuple(None if v is None else str(v) for v in column.values)
        groups.setdefault(key, []).append(column.name)
    return [
        _finding(
            QualityCode.DUPLICATE_COLUMNS, FindingSeverity.WARNING,
            f"Columns {', '.join(repr(n) for n in names)} hold identical values in every row.",
            columns=names, evidence={"columns": names})
        for names in groups.values() if len(names) > 1
    ]


def _column_flag_findings(columns: list[TypedColumn]) -> list[QualityFinding]:
    out = []
    for column in columns:
        profile = column.profile
        if profile is None:
            continue
        flags = set(profile.flags)
        if ColumnFlag.CONSTANT in flags:
            out.append(_finding(
                QualityCode.CONSTANT_COLUMN, FindingSeverity.INFO,
                f"'{column.name}' has a single value in every row, so it carries no information.",
                column=column.name, evidence={"unique_count": profile.unique_count}))
        elif ColumnFlag.NEAR_CONSTANT in flags:
            out.append(_finding(
                QualityCode.NEAR_CONSTANT_COLUMN, FindingSeverity.INFO,
                f"'{column.name}' takes one value in almost every row and carries little information.",
                column=column.name, evidence={"unique_count": profile.unique_count}))
        if column.identifier_evidence:
            confidence, why = column.identifier_evidence
            out.append(_finding(
                QualityCode.POSSIBLE_IDENTIFIER, FindingSeverity.WARNING if confidence >= 0.9 else FindingSeverity.INFO,
                f"'{column.name}' may identify rows rather than describe them: {why}. "
                "It is not removed; exclude it from model features.",
                column=column.name, confidence=confidence, level=heuristic,
                evidence={"uniqueness_ratio": profile.uniqueness_ratio}))
        if column.sensitive_evidence:
            confidence, why = column.sensitive_evidence
            out.append(_finding(
                QualityCode.POSSIBLE_SENSITIVE_FIELD, FindingSeverity.WARNING,
                f"'{column.name}' may contain personal data: {why}. Its values are withheld "
                "from this report and are never sent to an AI model.",
                column=column.name, confidence=confidence, level=heuristic))
        if (
            ColumnFlag.HIGH_CARDINALITY in flags
            and column.ctype is ColumnType.CATEGORICAL
            and not column.identifier_evidence
            and not column.sensitive_evidence
        ):
            out.append(_finding(
                QualityCode.HIGH_CARDINALITY_CATEGORICAL, FindingSeverity.INFO,
                f"'{column.name}' has {profile.unique_count} distinct values; one-hot encoding it "
                "would add that many columns.",
                column=column.name, evidence={"unique_count": profile.unique_count}))
    return out


def _type_findings(columns: list[TypedColumn]) -> list[QualityFinding]:
    out = []
    for column in columns:
        profile = column.profile
        if profile is None or ColumnFlag.MIXED_TYPES not in profile.flags:
            continue
        evidence = {"invalid_values": column.invalid_count, "non_missing": profile.count}
        if column.ctype is ColumnType.NUMERIC and column.invalid_count:
            out.append(_finding(
                QualityCode.INVALID_NUMERIC_VALUES, FindingSeverity.WARNING,
                f"'{column.name}' is numeric but {column.invalid_count} value(s) are not valid numbers.",
                column=column.name, evidence=evidence))
        elif column.ctype is ColumnType.DATETIME and column.invalid_count:
            out.append(_finding(
                QualityCode.MALFORMED_DATES, FindingSeverity.WARNING,
                f"'{column.name}' is a date column but {column.invalid_count} value(s) are not valid dates.",
                column=column.name, evidence=evidence))
        else:
            out.append(_finding(
                QualityCode.TYPE_INCONSISTENCY, FindingSeverity.WARNING,
                f"'{column.name}' mixes numbers and text, so it cannot be treated as one type.",
                column=column.name, evidence=evidence))
    return out


def _outlier_findings(columns: list[TypedColumn], settings: Settings) -> list[QualityFinding]:
    out = []
    k = settings.quality_outlier_iqr_multiplier
    for column in columns:
        if column.ctype is not ColumnType.NUMERIC or column.numeric is None or column.profile is None:
            continue
        data = column.numeric[~np.isnan(column.numeric)]
        if data.size < MIN_ROWS_FOR_OUTLIERS or column.profile.unique_count <= 10:
            continue
        q1, q3 = np.quantile(data, [0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low, high = q1 - k * iqr, q3 + k * iqr
        count = int(((data < low) | (data > high)).sum())
        if count == 0:
            continue
        share = count / data.size
        out.append(_finding(
            QualityCode.POSSIBLE_OUTLIER,
            FindingSeverity.WARNING if share > 0.05 else FindingSeverity.INFO,
            f"{count} value(s) ({share:.1%}) of '{column.name}' fall outside "
            f"[{low:.6g}, {high:.6g}] ({k} x IQR). They may be errors or genuine extremes.",
            column=column.name,
            evidence={"count": count, "share": round(share, 4), "lower_fence": float(low),
                      "upper_fence": float(high), "iqr_multiplier": k}))
    return out


def _imbalance_findings(
    columns: list[TypedColumn], targets: list[TargetCandidate], settings: Settings
) -> list[QualityFinding]:
    out = []
    by_name = {c.name: c for c in columns}
    for target in targets:
        if target.state not in (CandidateState.STRONG_CANDIDATE, CandidateState.POSSIBLE):
            continue
        column = by_name.get(target.column)
        if column is None or column.sensitive_evidence:
            continue
        counts = class_counts(column, settings)
        if not counts or len(counts) < 2:
            continue
        total = sum(counts.values())
        minority_label, minority = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
        share = minority / total
        if share >= settings.quality_imbalance_minority_ratio:
            continue
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        out.append(_finding(
            QualityCode.CLASS_IMBALANCE,
            FindingSeverity.CRITICAL if share < 0.01 else FindingSeverity.WARNING,
            f"If '{target.column}' is the target, its rarest class is {share:.2%} of rows "
            f"(limit {settings.quality_imbalance_minority_ratio:.0%}); accuracy would be misleading.",
            column=target.column,
            evidence={"minority_share": round(share, 6), "classes": {k: v for k, v in ordered},
                      "imbalance_ratio": round(max(counts.values()) / minority, 2),
                      "threshold": settings.quality_imbalance_minority_ratio}))
    return out


def _leakage_findings(
    columns: list[TypedColumn],
    targets: list[TargetCandidate],
    n_rows: int,
    settings: Settings,
    deadline: Optional[Deadline],
) -> list[QualityFinding]:
    out: list[QualityFinding] = []
    by_name = {c.name: c for c in columns}
    indices = sample_indices(n_rows, settings.analysis_sample_rows, settings.analysis_random_seed)
    chosen = [t for t in targets if t.state in (CandidateState.STRONG_CANDIDATE, CandidateState.POSSIBLE)][:LEAKAGE_TARGETS]
    for target in chosen:
        target_col = by_name.get(target.column)
        if target_col is None:
            continue
        features = [
            c for c in columns
            if c is not target_col
            and c.profile is not None
            and c.ctype not in (ColumnType.TEXT, ColumnType.UNKNOWN)
            and ColumnFlag.CONSTANT not in c.profile.flags
            and not (c.identifier_evidence and c.identifier_evidence[0] >= 0.9)
        ][:LEAKAGE_MAX_FEATURES]
        target_tokens = set(target_col.name.split("_"))
        for feature in features:
            if deadline is not None:
                deadline.check("leakage analysis")
            result = association(feature, target_col, indices, settings.task_max_classes)
            if result is None:
                continue
            accurate = (
                result.single_feature_accuracy is not None
                and result.majority_baseline is not None
                and result.single_feature_accuracy >= settings.quality_leakage_association
                and result.single_feature_accuracy - result.majority_baseline >= 0.05
            )
            # A correlation ratio is variance-based and runs above the accuracy it implies
            # (petal length separates three iris species with ratio 0.985 but a 96.7% rule), so for
            # it the one-column rule must also be near-perfect before this is called leakage.
            strong_value = result.value >= settings.quality_leakage_association and not (
                result.method == "correlation_ratio"
                and result.single_feature_accuracy is not None
                and result.single_feature_accuracy < settings.quality_leakage_association
            )
            if strong_value or accurate:
                detail = (
                    f"'{feature.name}' alone predicts '{target.column}' almost perfectly "
                    f"({result.method} {result.value:.3f}"
                    + (f", one-column rule right {result.single_feature_accuracy:.1%} of the time"
                       if result.single_feature_accuracy is not None else "")
                    + "). Check whether it is recorded after, or derived from, the outcome before using it."
                )
                out.append(_finding(
                    QualityCode.LEAKAGE_SIGNAL, FindingSeverity.CRITICAL, detail,
                    columns=[feature.name, target.column], column=feature.name,
                    confidence=0.9,
                    evidence={"target": target.column, "method": result.method,
                              "association": result.value, "pairs": result.n,
                              "single_feature_accuracy": result.single_feature_accuracy,
                              "majority_baseline": result.majority_baseline,
                              "threshold": settings.quality_leakage_association}))
                continue
            name_parts = set(feature.name.split("_"))
            suspicious = bool(name_parts & _SUSPICIOUS_NAME_TOKENS) or (
                len(target_tokens) > 0 and target_col.name in feature.name and feature.name != target_col.name
            )
            if suspicious and result.value >= POTENTIAL_LEAKAGE_ASSOCIATION:
                out.append(_finding(
                    QualityCode.POTENTIAL_LEAKAGE, FindingSeverity.WARNING,
                    f"'{feature.name}' is named like something recorded with or after the outcome and is "
                    f"strongly associated with '{target.column}' ({result.method} {result.value:.2f}). "
                    "That is a hint, not a finding of leakage.",
                    columns=[feature.name, target.column], column=feature.name,
                    confidence=0.5, level=heuristic,
                    evidence={"target": target.column, "method": result.method, "association": result.value}))
    return out


def _unusable(columns: list[TypedColumn]) -> Optional[QualityFinding]:
    reasons: dict[str, str] = {}
    for column in columns:
        flags = set(column.flags)
        if ColumnFlag.ALL_MISSING in flags:
            reasons[column.name] = "every value is missing"
        elif ColumnFlag.CONSTANT in flags:
            reasons[column.name] = "a single value in every row"
        elif ColumnFlag.FREE_TEXT in flags:
            reasons[column.name] = "free text, which needs text processing before modelling"
    if not reasons:
        return None
    return _finding(
        QualityCode.UNUSABLE_COLUMN, FindingSeverity.INFO,
        f"{len(reasons)} column(s) cannot be used as ordinary model features without further work: "
        + ", ".join(f"'{k}' ({v})" for k, v in reasons.items()) + ".",
        columns=list(reasons), evidence={"reasons": reasons})


def _ingestion_findings(info: IngestionInfo) -> list[QualityFinding]:
    out = []
    if info.malformed_rows:
        out.append(_finding(
            QualityCode.MALFORMED_ROWS, FindingSeverity.WARNING,
            f"{info.malformed_rows} row(s) had the wrong number of fields and were repaired "
            "(short rows padded with missing values, extra fields dropped).",
            evidence={"malformed_rows": info.malformed_rows}))
    if info.truncated:
        out.append(_finding(
            QualityCode.TRUNCATED, FindingSeverity.WARNING,
            f"Only the first {info.rows_read} rows were analysed ({info.truncation_reason}); "
            "every figure describes those rows, not the whole file.",
            evidence={"rows_analysed": info.rows_read}))
    return out


def analyze_quality(
    columns: list[TypedColumn],
    n_rows: int,
    ingestion: IngestionInfo,
    targets: list[TargetCandidate],
    settings: Settings,
    deadline: Optional[Deadline] = None,
) -> QualityReport:
    """All quality findings for a parsed file, in a deterministic order."""
    findings: list[QualityFinding] = []
    findings += _ingestion_findings(ingestion)
    findings += _missing_findings(columns, settings)
    duplicate_finding, duplicate_count = _duplicate_rows(columns, n_rows)
    if duplicate_finding:
        findings.append(duplicate_finding)
    findings += _duplicate_columns(columns)
    findings += _column_flag_findings(columns)
    findings += _type_findings(columns)
    findings += _outlier_findings(columns, settings)
    findings += _imbalance_findings(columns, targets, settings)
    findings += _leakage_findings(columns, targets, n_rows, settings, deadline)
    unusable = _unusable(columns)
    if unusable:
        findings.append(unusable)

    findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.code.value, f.column or "", tuple(f.columns)))

    total_cells = n_rows * len(columns)
    missing_cells = sum(c.profile.missing_count for c in columns if c.profile)
    unusable_names = set(unusable.columns) if unusable else set()
    identifiers = {c.name for c in columns if c.identifier_evidence and c.identifier_evidence[0] >= 0.9}
    return QualityReport(
        findings=findings,
        overall_missing_pct=round(100.0 * missing_cells / total_cells, 4) if total_cells else 0.0,
        duplicate_row_count=duplicate_count,
        duplicate_row_pct=round(100.0 * duplicate_count / n_rows, 4) if n_rows else 0.0,
        usable_column_count=len([c for c in columns if c.name not in unusable_names | identifiers]),
        thresholds={
            "high_missing_ratio": settings.quality_high_missing_ratio,
            "near_constant_ratio": settings.quality_near_constant_ratio,
            "identifier_unique_ratio": settings.quality_identifier_unique_ratio,
            "outlier_iqr_multiplier": settings.quality_outlier_iqr_multiplier,
            "imbalance_minority_ratio": settings.quality_imbalance_minority_ratio,
            "leakage_association": settings.quality_leakage_association,
        },
    )
