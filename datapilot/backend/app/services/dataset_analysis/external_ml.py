"""Targets and tasks for a dataset known only from a registry's schema.

With no rows to look at, the only honest evidence is what the registry published (a designated
target column, declared task types, declared column types) and what a column's *name* suggests.
So this module differs from the parsed-file path in three ways, all deliberate:

* evidence is ``REGISTRY_VERIFIED`` (the registry said so) or ``HEURISTIC`` (a name looks like a
  target); it is never ``DATAPILOT_PARSED``;
* a candidate can be ``STRONG`` only when the registry itself designates it; name cues alone
  cap at ``POSSIBLE``;
* the number of classes, balance and sample size are unknown, so tasks say so instead of
  guessing them.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.dataset_analysis import (
    CandidateState,
    EvidenceItem,
    EvidenceLevel,
    ExternalMetadata,
    MLTask,
    SchemaColumn,
    TargetCandidate,
    TargetKind,
    TaskCandidate,
)
from app.services.dataset_analysis.targets import (
    MAX_CANDIDATES,
    POSSIBLE_SCORE,
    WEAK_SCORE,
    _logistic,
    semantic_name_score,
)

registry = EvidenceLevel.REGISTRY_VERIFIED
heuristic = EvidenceLevel.HEURISTIC

#: Words that, in a column name, usually mean "the thing being measured or decided" rather than an input.
_OUTCOME_WORDS = frozenset({
    "score", "scores", "grade", "grades", "mark", "marks", "price", "cost", "salary", "rating", "revenue", "sales",
    "result", "outcome", "label", "class", "target", "status", "churn", "survived", "diagnosis", "default",
    "fraud", "income", "demand", "yield", "attrition",
})
_NUMERIC_TYPES = {"continuous", "numeric", "real", "float", "double"}
_CATEGORICAL_TYPES = {"categorical", "nominal", "string", "binary", "boolean", "class_label", "category"}


def _kind(column: SchemaColumn, meta: ExternalMetadata) -> TargetKind:
    declared = (column.data_type or "").lower()
    classes = meta.registry_statistics.get("NumberOfClasses")
    if classes is None and declared == "class_label":
        # Hugging Face publishes the number of classes of a ClassLabel feature.
        found = re.match(r"(\d+) classes", column.description or "")
        classes = float(found.group(1)) if found else None
    if declared == "binary" or declared == "boolean" or (classes is not None and classes == 2):
        return TargetKind.BINARY
    if declared in _CATEGORICAL_TYPES:
        if classes is not None and classes >= 3:
            return TargetKind.MULTICLASS
        return TargetKind.UNKNOWN  # categorical, but the number of classes was not published
    if declared in _NUMERIC_TYPES:
        return TargetKind.CONTINUOUS
    return TargetKind.UNKNOWN


def _is_categorical(column: SchemaColumn) -> bool:
    return (column.data_type or "").lower() in _CATEGORICAL_TYPES


def detect_external_targets(
    meta: ExternalMetadata, target_hint: Optional[str] = None
) -> list[TargetCandidate]:
    """Target candidates from a registry schema, strongest first."""
    columns = meta.schema_columns
    if not columns:
        return []
    designated = {t.strip().lower() for t in meta.registry_target_columns}
    hint = (target_hint or "").strip().lower() or None
    candidates: list[TargetCandidate] = []

    for position, column in enumerate(columns):
        names = {(column.original_name or "").lower(), column.normalized_name.lower()}
        is_designated = bool(designated & names)
        is_hinted = bool(hint and hint in names)
        evidence: list[EvidenceItem] = []
        against: list[str] = []
        score = 0.0

        if is_designated:
            score += 5.0
            evidence.append(EvidenceItem(
                statement=f"the registry ({meta.source}) designates '{column.original_name or column.normalized_name}' as the target",
                level=registry))
        if column.role in ("row_id", "ignore"):
            against.append(f"the registry marks it as '{column.role}'")
            score -= 3.0
        name_score, name_evidence = semantic_name_score(column.normalized_name, position, len(columns))
        score += name_score
        evidence.extend(name_evidence)
        if column.data_type:
            from_ai = column.evidence is EvidenceLevel.AI_INFERRED
            evidence.append(EvidenceItem(
                statement=(f"web sources (read by AI, not verified) list its type as {column.data_type}" if from_ai
                           else f"the source lists its type as {column.data_type}"),
                level=EvidenceLevel.AI_INFERRED if from_ai else registry))
            if position == len(columns) - 1 and column.data_type.lower() not in ("text", "string"):
                score += 1.0
                evidence.append(EvidenceItem(
                    statement=f"is the last column and a {column.data_type.lower()} one, where datasets usually put the outcome",
                    level=heuristic))
        if _OUTCOME_WORDS & set(re.split(r"[^a-z0-9]+", column.normalized_name.lower())):
            score += 1.0
            evidence.append(EvidenceItem(statement="its name describes a measured outcome (score, price, status...)", level=heuristic))
        against.append("no rows were read, so its values, balance and missingness are unknown")

        if is_designated or is_hinted:
            state = CandidateState.STRONG_CANDIDATE
        elif score >= POSSIBLE_SCORE:
            state = CandidateState.POSSIBLE  # a name alone never makes a target strong
        elif score >= WEAK_SCORE:
            state = CandidateState.WEAK
        else:
            continue
        candidates.append(TargetCandidate(
            column=column.normalized_name,
            state=state,
            confidence=1.0 if is_hinted else min(_logistic(score), 0.95 if is_designated else 0.6),
            score=round(score, 4),
            target_kind=_kind(column, meta),
            evidence=evidence,
            against=against,
            why_target=("specified by the user; " if is_hinted else "") + "; ".join(e.statement for e in evidence[:3]),
            why_not="; ".join(against[:3]),
            user_specified=is_hinted,
        ))
    candidates.sort(key=lambda c: (not c.user_specified, -c.score, c.column))
    return candidates[:MAX_CANDIDATES]


def detect_external_tasks(
    meta: ExternalMetadata, targets: list[TargetCandidate]
) -> list[TaskCandidate]:
    """Tasks the registry's own information supports. Never claims what needs rows."""
    tasks: list[TaskCandidate] = []
    declared = {t.lower() for t in meta.tasks}
    unknowns = ["the number of rows and the class balance were not verified"] if meta.row_count is None else []
    by_name = {c.normalized_name: c for c in meta.schema_columns}

    for target in targets[:2]:
        if target.state is CandidateState.WEAK:
            continue
        column = by_name.get(target.column)
        limits = list(unknowns) + ["no rows were read; upload the file to confirm the target's values"]
        evidence = [EvidenceItem(statement=f"'{target.column}' is a {target.state.value.replace('_', ' ')} target", level=registry if target.state is CandidateState.STRONG_CANDIDATE else heuristic)]
        confidence = round(target.confidence * 0.85, 4)
        state = CandidateState.POSSIBLE if target.state is CandidateState.STRONG_CANDIDATE else CandidateState.WEAK
        kind = target.target_kind
        if kind is TargetKind.BINARY:
            tasks.append(TaskCandidate(task=MLTask.BINARY_CLASSIFICATION, target=target.column, state=state,
                                       confidence=confidence, evidence=evidence, limitations=limits))
        elif kind is TargetKind.MULTICLASS:
            tasks.append(TaskCandidate(task=MLTask.MULTICLASS_CLASSIFICATION, target=target.column, state=state,
                                       confidence=confidence, evidence=evidence, limitations=limits))
        elif kind is TargetKind.CONTINUOUS:
            tasks.append(TaskCandidate(task=MLTask.REGRESSION, target=target.column, state=state,
                                       confidence=confidence, evidence=evidence, limitations=limits))
            if "time series" in " ".join(declared):
                tasks.append(TaskCandidate(
                    task=MLTask.TIME_SERIES_FORECASTING, target=target.column, state=state, confidence=confidence,
                    evidence=evidence + [EvidenceItem(statement="the registry lists 'Time Series' among the dataset's tasks", level=registry)],
                    limitations=limits))
        elif any("classif" in t for t in declared) or (column is not None and _is_categorical(column)):
            # The type alone does not say how many classes there are (UCI lists a coded
            # diagnosis as "Integer"), but the registry's own declared task or a categorical
            # type does say it is classification.
            why = ("the registry declares classification as a task" if any("classif" in t for t in declared)
                   else f"the registry lists the column's type as {column.data_type}")
            tasks.append(TaskCandidate(
                task=MLTask.MULTICLASS_CLASSIFICATION, target=target.column, state=state, confidence=confidence,
                evidence=evidence + [EvidenceItem(statement=why, level=registry)],
                limitations=limits + [
                    "the number of classes was not published: this is classification, and may be binary"]))
        elif any("regress" in t for t in declared):
            tasks.append(TaskCandidate(
                task=MLTask.REGRESSION, target=target.column, state=state, confidence=confidence,
                evidence=evidence + [EvidenceItem(statement="the registry declares regression as a task", level=registry)],
                limitations=limits))

    if any("cluster" in t for t in declared):
        tasks.append(TaskCandidate(
            task=MLTask.CLUSTERING, state=CandidateState.POSSIBLE, confidence=0.5,
            evidence=[EvidenceItem(statement="the registry lists clustering among the dataset's tasks", level=registry)],
            limitations=unknowns))

    unique: dict[tuple[MLTask, Optional[str]], TaskCandidate] = {}
    for task in tasks:
        unique.setdefault((task.task, task.target), task)
    order = {CandidateState.STRONG_CANDIDATE: 0, CandidateState.POSSIBLE: 1, CandidateState.WEAK: 2}
    return sorted(unique.values(), key=lambda t: (order[t.state], -t.confidence, t.task.value, t.target or ""))
