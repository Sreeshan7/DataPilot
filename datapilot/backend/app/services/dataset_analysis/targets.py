"""Which column might be the prediction target?

There is no single reliable signal, so several weak ones are combined and every point is
shown with its reason, for *and* against. A candidate is never reported without the case
against it.

Signals, with the weight each contributes to ``score``:

========================================  ======  ===========================
signal                                    weight  kind of evidence
========================================  ======  ===========================
named 'target', 'label', 'class', 'y'...   +3.0   heuristic (name)
named like a typical outcome ('churn')     +2.0   heuristic (name)
name token listed by the concept dictionary +1.0  heuristic (name)
last column                                +0.75  heuristic (position)
exactly two distinct values                +1.5   parsed (structure)
3..N classes                               +1.0   parsed (structure)
continuous numeric                         +0.5   parsed (structure)
predictable from other columns             +0.75  parsed (association)
associated with nothing                    -0.75  parsed (association)
too many distinct values for a label       -2.0   parsed (structure)
mostly missing                             -1 / -2 parsed (structure)
looks like an identifier                   -2.0   heuristic
========================================  ======  ===========================

``score`` maps to a state (STRONG >= 4.5, POSSIBLE >= 2.5, WEAK >= 1.0) and, through a
monotone logistic, to a confidence. That confidence orders candidates; it is not a
calibrated probability and must not be read as one. A strong state needs a name cue: a
binary last column with no other support tops out at POSSIBLE.

The dictionary signal reuses the project's own concept dictionary (the label vocabulary of
``target_requirements``), so it grows with the dictionary rather than with a word list here.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    CandidateState,
    ColumnFlag,
    ColumnType,
    EvidenceItem,
    EvidenceLevel,
    TargetCandidate,
    TargetKind,
)
from app.services.dataset_analysis.association import (
    association,
    is_categorical_like,
    sample_indices,
)
from app.services.dataset_analysis.profiler import TypedColumn

STRONG_SCORE = 4.5
POSSIBLE_SCORE = 2.5
WEAK_SCORE = 1.0
MAX_CANDIDATES = 8
#: Candidates examined for a relationship with the other columns (the second pass costs
#: one association per other column, so it is limited to the front-runners).
RELATIONSHIP_CANDIDATES = 5
RELATIONSHIP_MAX_COLUMNS = 40
STRONG_ASSOCIATION = 0.3
NO_ASSOCIATION = 0.1

CONVENTIONAL_EXACT = frozenset(
    {"target", "label", "labels", "class", "y", "output", "outcome", "response", "dependent",
     "dependent_variable", "ground_truth", "gt", "target_variable"}
)
CONVENTIONAL_OUTCOMES = frozenset(
    {"survived", "survival", "churn", "churned", "default", "defaulted", "fraud", "is_fraud",
     "isfraud", "diagnosis", "species", "price", "sale_price", "saleprice", "median_house_value",
     "medv", "quality", "grade", "final_grade", "g3", "passed", "readmitted", "result", "status",
     "income_class", "attrition", "converted", "clicked", "approved", "malignant", "disease",
     "has_disease", "heart_disease", "diabetes", "outcome_class", "performance"}
)

_VOCABULARY: Optional[dict[str, str]] = None


def dictionary_target_vocabulary() -> dict[str, str]:
    """Label-value tokens the knowledge base lists as targets, each with one concept.

    Built from the same index the ranker uses for label spaces, so nothing here is
    maintained separately. Deterministic: concepts are visited in key order.
    """
    global _VOCABULARY
    if _VOCABULARY is None:
        from app.services.datasets.target_concepts import get_target_concept_index

        vocabulary: dict[str, str] = {}
        index = get_target_concept_index()
        for key in sorted(index.concepts):
            concept = index.concepts[key]
            owner = concept.concept_ids[0] if concept.concept_ids else key
            for member in sorted(concept.members):
                vocabulary.setdefault(member, owner)
        _VOCABULARY = vocabulary
    return _VOCABULARY


def _logistic(score: float) -> float:
    """Monotone map from score to (0, 1). It orders candidates; it is not a probability."""
    return round(1.0 / (1.0 + math.exp(-(score - 3.0) / 1.2)), 4)


def target_kind(column: TypedColumn, settings: Settings) -> TargetKind:
    if column.profile is None:
        return TargetKind.UNKNOWN
    unique = column.profile.unique_count
    if column.ctype is ColumnType.BOOLEAN or unique == 2:
        return TargetKind.BINARY
    if column.ctype is ColumnType.CATEGORICAL and 3 <= unique <= settings.task_max_classes:
        return TargetKind.MULTICLASS
    if column.ctype is ColumnType.NUMERIC:
        if column.is_integer and 3 <= unique <= settings.task_max_classes:
            return TargetKind.MULTICLASS
        return TargetKind.CONTINUOUS
    return TargetKind.UNKNOWN


def _eligible(column: TypedColumn) -> bool:
    if column.ctype in (ColumnType.UNKNOWN, ColumnType.DATETIME, ColumnType.TEXT):
        return False
    flags = set(column.flags)
    if ColumnFlag.CONSTANT in flags or ColumnFlag.ALL_MISSING in flags:
        return False
    if column.identifier_evidence and column.identifier_evidence[0] >= 0.9:
        return False
    if column.sensitive_evidence and column.sensitive_evidence[0] >= 0.85:
        return False
    return True


def regular_time_column(columns: list[TypedColumn]) -> Optional[TypedColumn]:
    """A date/time column with enough distinct values and a regular spacing, if any."""
    for column in columns:
        profile = column.profile
        if (
            column.ctype is ColumnType.DATETIME
            and profile is not None
            and profile.unique_count >= 10
            and profile.datetime is not None
            and profile.datetime.granularity not in ("insufficient", "irregular")
        ):
            return column
    return None


def _structural(
    column: TypedColumn, settings: Settings, time_column: Optional[TypedColumn] = None
) -> tuple[float, list[EvidenceItem], list[str]]:
    profile = column.profile
    assert profile is not None
    score = 0.0
    evidence: list[EvidenceItem] = []
    against: list[str] = []
    unique = profile.unique_count
    parsed = EvidenceLevel.DATAPILOT_PARSED

    if column.ctype is ColumnType.BOOLEAN or unique == 2:
        score += 1.5
        evidence.append(EvidenceItem(statement="has exactly two distinct values, like a binary outcome", level=parsed))
    elif column.ctype is ColumnType.CATEGORICAL:
        if 3 <= unique <= settings.task_max_classes:
            score += 1.0
            evidence.append(EvidenceItem(statement=f"has {unique} distinct values, like a class label", level=parsed))
        elif unique > settings.task_max_classes:
            score -= 2.0
            against.append(f"has {unique} distinct values, too many for a class label")
    elif column.ctype is ColumnType.NUMERIC:
        if column.is_integer and 3 <= unique <= settings.task_max_classes:
            score += 1.0
            evidence.append(EvidenceItem(
                statement=f"is an integer with {unique} distinct values, class-like or ordinal", level=parsed))
        else:
            score += 0.5
            evidence.append(EvidenceItem(statement="is a continuous numeric quantity", level=parsed))
            if column.is_integer and profile.uniqueness_ratio >= 0.9 and column.identifier_evidence:
                score -= 1.0
                against.append("almost every value is a distinct integer, which looks like a counter or id")
            if time_column is not None and not column.identifier_evidence:
                score += 1.0
                evidence.append(EvidenceItem(
                    statement=f"is a numeric quantity indexed by the regular date/time column '{time_column.name}'",
                    level=parsed))

    if profile.missing_pct > 50:
        score -= 2.0
        against.append(f"{profile.missing_pct:.0f}% of its values are missing")
    elif profile.missing_pct > 30:
        score -= 1.0
        against.append(f"{profile.missing_pct:.0f}% of its values are missing")
    if ColumnFlag.NEAR_CONSTANT in profile.flags:
        score -= 1.5
        against.append("it is almost constant")
    if column.identifier_evidence:
        score -= 2.0
        against.append(f"it may be an identifier ({column.identifier_evidence[1]})")
    if column.sensitive_evidence:
        score -= 1.0
        against.append(f"it may be a sensitive field ({column.sensitive_evidence[1]})")
    return score, evidence, against


def _semantic(column: TypedColumn, position: int, n_columns: int) -> tuple[float, list[EvidenceItem]]:
    return semantic_name_score(column.name, position, n_columns)


def semantic_name_score(name: str, position: int, n_columns: int) -> tuple[float, list[EvidenceItem]]:
    """What a column's *name and position* say about it being a target.

    Needs no data, so it is also what a schema-only dataset (known from a registry, never
    parsed) can be judged on. Every point is a ``HEURISTIC``.
    """
    score = 0.0
    evidence: list[EvidenceItem] = []
    heuristic = EvidenceLevel.HEURISTIC
    if name in CONVENTIONAL_EXACT:
        score += 3.0
        evidence.append(EvidenceItem(statement=f"is named '{name}', a conventional name for a target", level=heuristic))
    elif name in CONVENTIONAL_OUTCOMES or name.removeprefix("is_") in CONVENTIONAL_OUTCOMES:
        score += 2.0
        evidence.append(EvidenceItem(statement=f"is named '{name}', typically an outcome", level=heuristic))

    from app.services.datasets.target_concepts import value_tokens

    vocabulary = dictionary_target_vocabulary()
    hits = [t for t in value_tokens(name.replace("_", " ")) if t in vocabulary]
    if hits:
        score += 1.0
        token = hits[0]
        evidence.append(EvidenceItem(
            statement=f"'{token}' is a target value the knowledge base lists for {vocabulary[token]}",
            level=heuristic))
    if n_columns > 1 and position == n_columns - 1:
        score += 0.75
        evidence.append(EvidenceItem(statement="is the last column, where datasets commonly put the label", level=heuristic))
    return score, evidence


def _state(score: float) -> Optional[CandidateState]:
    if score >= STRONG_SCORE:
        return CandidateState.STRONG_CANDIDATE
    if score >= POSSIBLE_SCORE:
        return CandidateState.POSSIBLE
    if score >= WEAK_SCORE:
        return CandidateState.WEAK
    return None


def _summarise(items: list[str]) -> str:
    return "; ".join(items[:3])


def detect_targets(
    columns: list[TypedColumn],
    n_rows: int,
    settings: Settings,
    target_hint: Optional[str] = None,
) -> list[TargetCandidate]:
    """Rank the columns that could be the target, best first, each with its case for and against."""
    n_columns = len(columns)
    time_column = regular_time_column(columns)
    scored: dict[str, dict] = {}
    for position, column in enumerate(columns):
        if column.profile is None:
            continue
        hinted = target_hint is not None and column.name == target_hint
        if not _eligible(column) and not hinted:
            continue
        s_score, s_evidence, against = _structural(column, settings, time_column)
        n_score, n_evidence = _semantic(column, position, n_columns)
        scored[column.name] = {
            "column": column, "position": position, "score": s_score + n_score,
            "evidence": s_evidence + n_evidence, "against": against, "hinted": hinted,
        }

    # Second pass: does the rest of the table actually relate to this column?
    indices = sample_indices(n_rows, settings.analysis_sample_rows, settings.analysis_random_seed)
    front = sorted(scored.values(), key=lambda c: (-c["score"], c["position"]))[:RELATIONSHIP_CANDIDATES]
    others_pool = [
        c for c in columns
        if c.profile is not None and _usable_as_feature(c)
    ][:RELATIONSHIP_MAX_COLUMNS]
    for entry in front:
        target = entry["column"]
        best: Optional[tuple[float, str, str]] = None
        measured = 0
        for other in others_pool:
            if other is target:
                continue
            result = association(other, target, indices, settings.task_max_classes)
            if result is None:
                continue
            measured += 1
            if best is None or result.value > best[0]:
                best = (result.value, other.name, result.method)
        if best is None or measured < 2:
            continue
        if best[0] >= STRONG_ASSOCIATION:
            entry["score"] += 0.75
            entry["evidence"].append(EvidenceItem(
                statement=f"is predictable from other columns (strongest association {best[0]:.2f} "
                          f"with '{best[1]}', {best[2]})",
                level=EvidenceLevel.DATAPILOT_PARSED))
        elif best[0] < NO_ASSOCIATION:
            entry["score"] -= 0.75
            entry["against"].append(
                f"no other column is meaningfully associated with it (strongest {best[0]:.2f})")

    candidates: list[TargetCandidate] = []
    for entry in scored.values():
        column: TypedColumn = entry["column"]
        score = round(entry["score"], 4)
        state = _state(score)
        hinted = entry["hinted"]
        if hinted:
            state = CandidateState.STRONG_CANDIDATE
        if state is None:
            continue
        candidates.append(
            TargetCandidate(
                column=column.name,
                state=state,
                confidence=1.0 if hinted else _logistic(score),
                score=score,
                target_kind=target_kind(column, settings),
                evidence=entry["evidence"],
                against=entry["against"],
                why_target=("specified by the user; " if hinted else "")
                + _summarise([e.statement for e in entry["evidence"]]),
                why_not=_summarise(entry["against"]),
                user_specified=hinted,
            )
        )
    candidates.sort(key=lambda c: (not c.user_specified, -c.score, _position(columns, c.column)))
    return candidates[:MAX_CANDIDATES]


def _position(columns: list[TypedColumn], name: str) -> int:
    return next((i for i, c in enumerate(columns) if c.name == name), len(columns))


def _usable_as_feature(column: TypedColumn) -> bool:
    flags = set(column.flags)
    if column.ctype in (ColumnType.UNKNOWN, ColumnType.TEXT):
        return False
    if ColumnFlag.CONSTANT in flags or ColumnFlag.ALL_MISSING in flags:
        return False
    if column.identifier_evidence and column.identifier_evidence[0] >= 0.9:
        return False
    return True


def class_counts(column: TypedColumn, settings: Settings) -> Optional[Counter]:
    """Counts per class when the column is class-like, else ``None``."""
    if not is_categorical_like(column, settings.task_max_classes):
        return None
    if column.labels is not None:
        return Counter(x for x in column.labels if x is not None)
    if column.numeric is not None:
        data = column.numeric[~np.isnan(column.numeric)]
        return Counter(str(v) for v in data)
    return None
