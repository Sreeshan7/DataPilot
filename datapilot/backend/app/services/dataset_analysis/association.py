"""Statistical association between two columns of a parsed file.

Used for two different jobs that must agree on the numbers: spotting a feature that is
*too* predictive of a target (a leakage signal) and, in Phase 5, ranking features. It is
association, never causation: a high value means "knowing this column tells you about that
one under this measure", nothing more.

Measures, chosen by the kinds of the two columns so each is valid for what it is applied to:

==================  ==================  =============================================
feature             target              measure
==================  ==================  =============================================
numeric             numeric             |Spearman| rank correlation
categorical         categorical         Cramer's V, bias-corrected (Bergsma 2013)
numeric             categorical         correlation ratio eta (between / total variance)
categorical         numeric             correlation ratio eta
==================  ==================  =============================================

Every value lies in [0, 1]. A measure that cannot be computed validly (a constant column,
fewer than three paired values, a feature with so many distinct values that grouping by it
is meaningless) returns ``None`` instead of a number, so "no signal" and "could not be
measured" are never confused.

All sampling uses a fixed seed, so the same file always gives the same result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.schemas.dataset_analysis import ColumnType
from app.services.dataset_analysis.profiler import TypedColumn

#: Fewer paired observations than this and no measure is reported.
MIN_PAIRS = 3
#: A categorical feature with more distinct values than this share of the rows is not
#: grouped: with almost one row per group every association is spuriously perfect.
MAX_GROUP_FRACTION = 0.5
MAX_BINS = 20


@dataclass(frozen=True)
class Association:
    value: float
    method: str
    n: int
    #: Share of rows a one-feature rule ("predict the most common target value for each
    #: feature value") gets right; only for a categorical target.
    single_feature_accuracy: Optional[float] = None
    majority_baseline: Optional[float] = None


def sample_indices(n_rows: int, max_rows: int, seed: int) -> np.ndarray:
    """Row positions to analyse: all rows, or a seeded sample in original order."""
    if n_rows <= max_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_rows, replace=False))


def is_categorical_like(column: TypedColumn, max_classes: int) -> bool:
    """Whether a column should be treated as classes rather than a quantity."""
    if column.ctype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN):
        return True
    if column.ctype is ColumnType.NUMERIC and column.is_integer and column.numeric is not None:
        data = column.numeric[~np.isnan(column.numeric)]
        return 2 <= np.unique(data).size <= max_classes
    return False


def _numeric_values(column: TypedColumn) -> Optional[np.ndarray]:
    if column.ctype in (ColumnType.NUMERIC, ColumnType.BOOLEAN) and column.numeric is not None:
        return column.numeric
    if column.ctype is ColumnType.DATETIME and column.datetimes is not None:
        return np.array(
            [np.nan if d is None else d.timestamp() for d in column.datetimes], dtype=float
        )
    return None


def _codes(column: TypedColumn) -> Optional[tuple[np.ndarray, int]]:
    """Integer class codes (-1 where missing) and the number of classes."""
    if column.ctype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN) and column.labels is not None:
        labels = column.labels
    elif column.ctype is ColumnType.NUMERIC and column.numeric is not None:
        labels = [None if np.isnan(v) else repr(float(v)) for v in column.numeric]
    else:
        return None
    mapping: dict[str, int] = {}
    out = np.full(len(labels), -1, dtype=np.int64)
    for i, label in enumerate(labels):
        if label is None:
            continue
        out[i] = mapping.setdefault(label, len(mapping))
    return out, len(mapping)


def _rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(values, method="average")


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < MIN_PAIRS or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    value = abs(float(np.corrcoef(rx, ry)[0, 1]))
    return None if np.isnan(value) else min(1.0, value)


def _correlation_ratio(codes: np.ndarray, k: int, y: np.ndarray) -> Optional[float]:
    if y.size < MIN_PAIRS or k < 2 or np.ptp(y) == 0:
        return None
    counts = np.bincount(codes, minlength=k).astype(float)
    sums = np.bincount(codes, weights=y, minlength=k)
    present = counts > 0
    means = np.zeros(k)
    means[present] = sums[present] / counts[present]
    grand = y.mean()
    between = float((counts[present] * (means[present] - grand) ** 2).sum())
    total = float(((y - grand) ** 2).sum())
    if total <= 0:
        return None
    return min(1.0, float(np.sqrt(between / total)))


def _cramers_v(a: np.ndarray, ka: int, b: np.ndarray, kb: int) -> Optional[float]:
    n = a.size
    if n < MIN_PAIRS or ka < 2 or kb < 2:
        return None
    table = np.zeros((ka, kb))
    np.add.at(table, (a, b), 1)
    # drop classes with no observations among the paired rows
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    r, k = table.shape
    if r < 2 or k < 2:
        return None
    expected = table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    phi2 = chi2 / n
    # Bergsma's bias correction: raw Cramer's V is inflated when tables are large
    # relative to the number of observations.
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    if denom <= 0:
        return None
    return min(1.0, float(np.sqrt(phi2c / denom)))


def _bin_numeric(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Quantile-bin a numeric feature so a one-feature rule cannot memorise each value."""
    distinct = np.unique(values)
    if distinct.size <= MAX_BINS:
        codes = np.searchsorted(distinct, values)
        return codes.astype(np.int64), int(distinct.size)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, MAX_BINS + 1)[1:-1]))
    codes = np.searchsorted(edges, values, side="right")
    return codes.astype(np.int64), int(edges.size + 1)


def _single_feature_accuracy(
    feature_codes: np.ndarray, kf: int, target_codes: np.ndarray, kt: int
) -> tuple[float, float]:
    """(accuracy of 'majority target per feature value', accuracy of 'always the majority')."""
    table = np.zeros((kf, kt))
    np.add.at(table, (feature_codes, target_codes), 1)
    n = float(table.sum())
    return float(table.max(axis=1).sum() / n), float(table.sum(axis=0).max() / n)


def association(
    feature: TypedColumn,
    target: TypedColumn,
    indices: np.ndarray,
    max_classes: int = 20,
) -> Optional[Association]:
    """Association between one feature column and the target over the given rows.

    Returns ``None`` when the pair cannot be measured validly. Only rows where both
    columns have a value are used.
    """
    if feature is target or feature.ctype in (ColumnType.TEXT, ColumnType.UNKNOWN):
        return None
    if target.ctype in (ColumnType.TEXT, ColumnType.UNKNOWN, ColumnType.DATETIME):
        return None

    target_as_classes = is_categorical_like(target, max_classes)
    feature_as_classes = feature.ctype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)

    f_num = _numeric_values(feature)
    t_num = _numeric_values(target)
    f_codes = _codes(feature) if feature_as_classes else None
    t_codes = _codes(target) if target_as_classes else None

    # Valid paired rows.
    mask = np.ones(len(indices), dtype=bool)
    if f_codes is not None:
        mask &= f_codes[0][indices] >= 0
    elif f_num is not None:
        mask &= ~np.isnan(f_num[indices])
    else:
        return None
    if t_codes is not None:
        mask &= t_codes[0][indices] >= 0
    elif t_num is not None:
        mask &= ~np.isnan(t_num[indices])
    else:
        return None
    rows = indices[mask]
    n = int(rows.size)
    if n < MIN_PAIRS:
        return None

    if f_codes is not None and f_codes[1] > max(MAX_BINS * 2.5, MAX_GROUP_FRACTION * n):
        return None

    value: Optional[float]
    method: str
    accuracy = baseline = None

    if t_codes is not None:  # classification-style target
        tcodes, kt = _recode(t_codes[0][rows])
        if kt < 2:
            return None
        if f_codes is not None:
            fcodes, kf = _recode(f_codes[0][rows])
            value, method = _cramers_v(fcodes, kf, tcodes, kt), "cramers_v"
        else:
            fvals = f_num[rows]
            if np.ptp(fvals) == 0:
                return None
            value, method = _correlation_ratio(tcodes, kt, fvals), "correlation_ratio"
            fcodes, kf = _bin_numeric(fvals)
        if value is None:
            return None
        accuracy, baseline = _single_feature_accuracy(fcodes, kf, tcodes, kt)
        if kf >= max(2.0, 0.5 * n):
            # about one row per group: "accuracy" is memorisation, not predictive power
            accuracy = baseline = None
        return Association(round(value, 6), method, n, accuracy, baseline)

    y = t_num[rows]
    if f_codes is not None:
        fcodes, kf = _recode(f_codes[0][rows])
        value, method = _correlation_ratio(fcodes, kf, y), "correlation_ratio"
    else:
        value, method = _spearman(f_num[rows], y), "spearman"
    if value is None:
        return None
    return Association(round(value, 6), method, n)


def _recode(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Compact class codes to 0..k-1 over the rows actually present."""
    unique, inverse = np.unique(codes, return_inverse=True)
    return inverse.astype(np.int64), int(unique.size)


def looks_like_leakage(result: Association, threshold: float) -> bool:
    """Whether one feature predicts the target so well that it is probably not a legitimate input.

    A correlation ratio is variance-based and runs above the accuracy it implies (petal length
    separates three iris species with ratio 0.985 but a 96.7% rule), so for it the one-column
    rule must also be near-perfect. Used by both the quality findings and Phase 5 feature roles so
    the two can never disagree about what counts as leakage.
    """
    accurate = (
        result.single_feature_accuracy is not None
        and result.majority_baseline is not None
        and result.single_feature_accuracy >= threshold
        and result.single_feature_accuracy - result.majority_baseline >= 0.05
    )
    strong_value = result.value >= threshold and not (
        result.method == "correlation_ratio"
        and result.single_feature_accuracy is not None
        and result.single_feature_accuracy < threshold
    )
    return strong_value or accurate


def class_codes(column: TypedColumn) -> Optional[tuple[np.ndarray, int]]:
    """Integer class codes (-1 where missing) and the class count; ``None`` if not class-like."""
    return _codes(column)
