"""Which machine-learning tasks does this dataset realistically support?

Rules that keep the answer honest:

* **No target, no supervised task.** A supervised task is only ever proposed for a column
  that is a plausible target (state POSSIBLE or better, or named by the user). The task
  model itself refuses to be built otherwise.
* **Unsupervised tasks never carry a target** they do not have.
* **Size and balance limit the claim.** Too few rows, a class with almost no members, or an
  irregular time axis downgrade or annotate a task rather than being left out of sight.

Each task states the evidence it rests on (read from the data, or a name-based heuristic)
and what would limit it.
"""

from __future__ import annotations

import re
from typing import Optional

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    CandidateState,
    ColumnFlag,
    ColumnType,
    EvidenceItem,
    EvidenceLevel,
    MLTask,
    TargetCandidate,
    TargetKind,
    TaskCandidate,
)
from app.services.dataset_analysis.profiler import TypedColumn
from app.services.dataset_analysis.targets import _usable_as_feature, class_counts

#: Minority share below which a binary label is framed as anomaly detection.
ANOMALY_MINORITY_SHARE = 0.05
#: A class this small cannot be split across training and validation folds.
MIN_CLASS_MEMBERS = 5
MAX_TASK_TARGETS = 3

_GROUP_NAMES = frozenset({"query_id", "qid", "group", "group_id", "session_id", "search_id"})
_RANK_NAMES = frozenset({"rank", "ranking", "relevance", "relevance_score", "position"})
_STATE_ORDER = {
    CandidateState.STRONG_CANDIDATE: 0,
    CandidateState.POSSIBLE: 1,
    CandidateState.WEAK: 2,
    CandidateState.UNKNOWN: 3,
}
_TASK_ORDER = {task: i for i, task in enumerate(MLTask)}

parsed = EvidenceLevel.DATAPILOT_PARSED
heuristic = EvidenceLevel.HEURISTIC


def _numeric_features(columns: list[TypedColumn]) -> list[TypedColumn]:
    return [
        c for c in columns
        if c.ctype is ColumnType.NUMERIC
        and _usable_as_feature(c)
        and ColumnFlag.POSSIBLE_IDENTIFIER not in c.flags
    ]


def _datetime_columns(columns: list[TypedColumn]) -> list[TypedColumn]:
    return [
        c for c in columns
        if c.ctype is ColumnType.DATETIME and c.profile is not None and c.profile.unique_count >= 10
    ]


def _downgrade(state: CandidateState) -> CandidateState:
    return {
        CandidateState.STRONG_CANDIDATE: CandidateState.POSSIBLE,
        CandidateState.POSSIBLE: CandidateState.WEAK,
    }.get(state, state)


def _size_limits(n_rows: int, settings: Settings) -> list[str]:
    if n_rows < settings.task_min_rows_supervised:
        return [
            f"only {n_rows} rows: too few to train and validate a model reliably "
            f"(at least {settings.task_min_rows_supervised} are needed)"
        ]
    return []


def _supervised_for_target(
    target: TargetCandidate,
    column: TypedColumn,
    columns: list[TypedColumn],
    n_rows: int,
    settings: Settings,
) -> list[TaskCandidate]:
    out: list[TaskCandidate] = []
    base_state = target.state
    limits = _size_limits(n_rows, settings)
    if limits:
        base_state = _downgrade(base_state)

    counts = class_counts(column, settings)
    class_limits: list[str] = []
    minority_share = None
    if counts:
        total = sum(counts.values())
        smallest_label, smallest = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
        minority_share = smallest / total if total else None
        if smallest < MIN_CLASS_MEMBERS:
            class_limits.append(
                f"the smallest class has only {smallest} member(s): it cannot be split across "
                "validation folds"
            )
        elif minority_share is not None and minority_share < settings.quality_imbalance_minority_ratio:
            class_limits.append(
                f"the minority class is {minority_share:.1%} of rows: accuracy would be misleading; "
                "use precision/recall-style metrics"
            )

    kind = target.target_kind
    if kind is TargetKind.BINARY:
        out.append(TaskCandidate(
            task=MLTask.BINARY_CLASSIFICATION, target=target.column, state=base_state,
            confidence=target.confidence,
            evidence=[EvidenceItem(statement=f"'{target.column}' has two classes", level=parsed)],
            limitations=limits + class_limits))
    elif kind is TargetKind.MULTICLASS:
        classes = column.profile.unique_count if column.profile else 0
        out.append(TaskCandidate(
            task=MLTask.MULTICLASS_CLASSIFICATION, target=target.column, state=base_state,
            confidence=target.confidence,
            evidence=[EvidenceItem(statement=f"'{target.column}' has {classes} classes", level=parsed)],
            limitations=limits + class_limits))
    elif kind is TargetKind.CONTINUOUS:
        out.append(TaskCandidate(
            task=MLTask.REGRESSION, target=target.column, state=base_state,
            confidence=target.confidence,
            evidence=[EvidenceItem(statement=f"'{target.column}' is a continuous numeric quantity", level=parsed)],
            limitations=limits))
        # Time series: a regular time axis plus a numeric target.
        for date_col in _datetime_columns(columns):
            granularity = date_col.profile.datetime.granularity if date_col.profile and date_col.profile.datetime else "insufficient"
            if granularity == "insufficient":
                continue
            regular = granularity != "irregular"
            ts_state = base_state if regular else CandidateState.WEAK
            notes = list(limits)
            if not regular:
                notes.append(f"the timestamps in '{date_col.name}' are irregular, which complicates forecasting")
            out.append(TaskCandidate(
                task=MLTask.TIME_SERIES_FORECASTING, target=target.column, state=ts_state,
                confidence=round(target.confidence * (0.9 if regular else 0.5), 4),
                evidence=[EvidenceItem(
                    statement=f"'{date_col.name}' is a date/time column with {granularity} spacing",
                    level=parsed)],
                limitations=notes))
            break

    # A rare positive class is also an anomaly-detection problem.
    if (
        kind is TargetKind.BINARY
        and minority_share is not None
        and minority_share < ANOMALY_MINORITY_SHARE
        and counts and min(counts.values()) >= 2
    ):
        out.append(TaskCandidate(
            task=MLTask.ANOMALY_DETECTION, target=target.column, state=_downgrade(base_state),
            confidence=round(target.confidence * 0.7, 4),
            evidence=[EvidenceItem(
                statement=f"the rarer value of '{target.column}' is {minority_share:.1%} of rows",
                level=parsed)],
            limitations=limits + ["labelled anomalies are rare; treat as imbalanced classification or unsupervised detection"]))
    return out


def _multilabel(columns: list[TypedColumn]) -> Optional[TaskCandidate]:
    groups: dict[str, list[TypedColumn]] = {}
    for column in columns:
        if column.ctype is ColumnType.BOOLEAN and "_" in column.name:
            groups.setdefault(column.name.split("_")[0], []).append(column)
    for prefix, members in sorted(groups.items()):
        if len(members) >= 3:
            return TaskCandidate(
                task=MLTask.MULTILABEL_CLASSIFICATION, target=members[0].name, state=CandidateState.POSSIBLE,
                confidence=0.5,
                evidence=[EvidenceItem(
                    statement=f"{len(members)} boolean columns share the prefix '{prefix}_' "
                              f"({', '.join(m.name for m in members[:4])}...)",
                    level=heuristic)],
                limitations=["treat the prefixed columns together as one multi-label target"])
    for column in columns:
        if column.ctype is ColumnType.CATEGORICAL and column.labels:
            present = [x for x in column.labels if x]
            if len(present) >= 20:
                multi = sum(1 for x in present if re.search(r"[;|]", x)) / len(present)
                if multi >= 0.3:
                    return TaskCandidate(
                        task=MLTask.MULTILABEL_CLASSIFICATION, target=column.name,
                        state=CandidateState.POSSIBLE, confidence=0.5,
                        evidence=[EvidenceItem(
                            statement=f"{multi:.0%} of '{column.name}' values hold several labels separated by ; or |",
                            level=parsed)],
                        limitations=["the multi-label column must be split into one indicator per label"])
    return None


def _ranking(columns: list[TypedColumn]) -> Optional[TaskCandidate]:
    names = {c.name: c for c in columns}
    rank = next((names[n] for n in names if n in _RANK_NAMES and names[n].ctype is ColumnType.NUMERIC), None)
    group = next((names[n] for n in names if n in _GROUP_NAMES), None)
    if rank is not None and group is not None:
        return TaskCandidate(
            task=MLTask.RANKING, target=rank.name, state=CandidateState.WEAK, confidence=0.4,
            evidence=[EvidenceItem(
                statement=f"'{rank.name}' looks like a ranking label and '{group.name}' like its query group",
                level=heuristic)],
            limitations=["ranking needs the rows grouped by query; confirm the meaning of both columns"])
    return None


def detect_tasks(
    columns: list[TypedColumn],
    targets: list[TargetCandidate],
    n_rows: int,
    settings: Settings,
) -> list[TaskCandidate]:
    """Every task the dataset supports, strongest first. Deterministic."""
    by_name = {c.name: c for c in columns}
    tasks: list[TaskCandidate] = []

    plausible = [
        t for t in targets
        if t.user_specified or t.state in (CandidateState.STRONG_CANDIDATE, CandidateState.POSSIBLE)
    ][:MAX_TASK_TARGETS]
    for target in plausible:
        column = by_name.get(target.column)
        if column is not None:
            tasks.extend(_supervised_for_target(target, column, columns, n_rows, settings))

    numeric = _numeric_features(columns)
    limits = _size_limits(n_rows, settings)
    if len(numeric) >= 2 and n_rows >= 20:
        tasks.append(TaskCandidate(
            task=MLTask.CLUSTERING, state=CandidateState.POSSIBLE if not limits else CandidateState.WEAK,
            confidence=0.5,
            evidence=[EvidenceItem(statement=f"{len(numeric)} numeric columns can describe each row", level=parsed)],
            limitations=limits + ["clusters have no ground truth here; judging them needs domain knowledge"]))

    has_anomaly_with_target = any(t.task is MLTask.ANOMALY_DETECTION for t in tasks)
    has_binary_target = any(t.task is MLTask.BINARY_CLASSIFICATION for t in tasks)
    if len(numeric) >= 3 and n_rows >= 50 and not has_anomaly_with_target and not has_binary_target:
        tasks.append(TaskCandidate(
            task=MLTask.ANOMALY_DETECTION, state=CandidateState.WEAK, confidence=0.35,
            evidence=[EvidenceItem(
                statement=f"{len(numeric)} numeric columns allow unsupervised outlier detection", level=parsed)],
            limitations=["there is no label to validate against; flagged rows are only unusual, not known-bad"]))

    for extra in (_multilabel(columns), _ranking(columns)):
        if extra is not None:
            tasks.append(extra)

    unique: dict[tuple[MLTask, Optional[str]], TaskCandidate] = {}
    for task in tasks:
        unique.setdefault((task.task, task.target), task)
    return sorted(
        unique.values(),
        key=lambda t: (_STATE_ORDER[t.state], -t.confidence, _TASK_ORDER[t.task], t.target or ""),
    )
