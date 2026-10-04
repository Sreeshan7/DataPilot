"""Column profiling for a parsed file: types, statistics, and per-column flags.

Everything here is computed from the actual values, so every :class:`ColumnProfile`
is ``DATAPILOT_PARSED``. The interpretation of a column from its *name* (it looks like
an identifier, it looks sensitive) is a different kind of statement and is reported as a
flag with its own evidence in the quality stage, never as a measured fact.

Type inference is deliberately strict and stated:

* A column is numeric / datetime when at least ``TYPE_DOMINANCE`` (90%) of its non-missing
  values parse as such. The remainder are counted as invalid values rather than hidden, so
  "mostly numeric with a few bad entries" is visible instead of being coerced away.
* Anything else is categorical or free text; a column that is a mixture is flagged
  ``MIXED_TYPES``.
* A column holding only {0, 1}, {true, false} or {yes, no} is boolean.

Statistics are never invented for a column that cannot support them: a standard deviation
needs two values, a quantile needs a number, and an empty column has none of either.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    CategoricalStats,
    ColumnFlag,
    ColumnProfile,
    ColumnType,
    DatetimeStats,
    NumericStats,
    TextStats,
    ValueCount,
)
from app.services.dataset_analysis.ingestion import ParsedTable
from app.services.dataset_analysis.safety import Deadline

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

#: Share of non-missing values that must parse for a column to take that type.
TYPE_DOMINANCE = 0.9
#: Share of non-missing values that must actually parse as dates for a date-looking
#: column to be treated as datetime.
MIN_PARSED_DATE_SHARE = 0.5
#: A text column whose mean length exceeds this is free text, not a category.
TEXT_MEAN_LENGTH = 40
#: Unique-value counts that separate low / medium / high cardinality.
LOW_CARDINALITY_MAX = 10
MEDIUM_CARDINALITY_MAX = 50
#: Values shown per categorical column.
TOP_VALUES = 5
#: Values examined when looking for patterns such as e-mail addresses.
PATTERN_SAMPLE = 200

QUANTILES = {"p05": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}

_BOOL_TRUE = frozenset({"true", "t", "yes", "y"})
_BOOL_FALSE = frozenset({"false", "f", "no", "n"})
_BOOL_WORDS = _BOOL_TRUE | _BOOL_FALSE

# Dotted dates need a four-digit year: "1.2.34" is far more often a version number than a
# date, and a library will cheerfully parse it as one.
_DATE_RE = re.compile(
    r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$"
    r"|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
    r"|^\d{1,2}\.\d{1,2}\.\d{4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$"
)

# ---- identifier / sensitive patterns --------------------------------------

_IDENTIFIER_NAME = re.compile(
    r"(^|_)(id|uid|uuid|guid|pk|index|idx|key)$|^(id|uuid|guid|index|idx|rownum|row_?number)$",
    re.IGNORECASE,
)
_SENSITIVE_NAME_TOKENS = {
    "email": "an e-mail address", "e_mail": "an e-mail address", "phone": "a phone number",
    "mobile": "a phone number", "telephone": "a phone number", "fax": "a phone number",
    "address": "an address", "street": "an address", "postcode": "a postal code",
    "postal": "a postal code", "zip": "a postal code", "zipcode": "a postal code",
    "ssn": "a national identifier", "passport": "a national identifier",
    "social_security": "a national identifier", "dob": "a date of birth",
    "birthdate": "a date of birth", "birth_date": "a date of birth",
    "first_name": "a person's name", "last_name": "a person's name",
    "full_name": "a person's name", "surname": "a person's name", "firstname": "a person's name",
    "lastname": "a person's name", "ip_address": "an IP address", "credit_card": "a payment card",
    "card_number": "a payment card", "iban": "a bank account", "password": "a credential",
}
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?[\d][\d\s\-().]{6,18}\d$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_ADDRESS_RE = re.compile(
    r"\b\d+\s+\w+(\s+\w+)?\s+(street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?)\b",
    re.IGNORECASE,
)


@dataclass
class TypedColumn:
    """A column after type inference, kept for every later stage to reuse."""

    name: str
    original: Optional[str]
    ctype: ColumnType
    values: list[Any]
    n_rows: int
    #: float array aligned with ``values`` (NaN where missing/invalid); numeric and
    #: 0/1-coded boolean columns only.
    numeric: Optional["np.ndarray"] = None
    #: strings aligned with ``values`` for categorical / boolean / text columns.
    labels: Optional[list[Optional[str]]] = None
    datetimes: Optional[list[Optional[datetime]]] = None
    invalid_count: int = 0
    is_integer: bool = False
    flags: list[ColumnFlag] = field(default_factory=list)
    identifier_evidence: Optional[tuple[float, str]] = None
    sensitive_evidence: Optional[tuple[float, str]] = None
    profile: Optional[ColumnProfile] = None

    @property
    def non_missing(self) -> int:
        return sum(1 for v in self.values if not _is_missing(v))


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _parse_float(text: str) -> Optional[float]:
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _classify_values(values: list[Any]) -> dict[str, Any]:
    """Counts of what the non-missing values are, plus the parsed numbers and dates."""
    import numpy as np
    import pandas as pd

    n = len(values)
    numeric = np.full(n, np.nan)
    kinds: Counter[str] = Counter()
    bool_words: list[Optional[str]] = [None] * n
    present = [(i, v) for i, v in enumerate(values) if not _is_missing(v)]

    str_positions = [i for i, v in present if isinstance(v, str)]
    # Strings: parse numbers in bulk, then dates.
    if str_positions:
        series = pd.Series([values[i] for i in str_positions], dtype=object)
        parsed = pd.to_numeric(series, errors="coerce")
        arr = parsed.to_numpy(dtype=float, na_value=np.nan)
        finite = np.isfinite(arr)
        for pos, ok, number in zip(str_positions, finite, arr):
            if ok:
                numeric[pos] = number
                kinds["num"] += 1
        lowered = {pos: values[pos].lower() for pos in str_positions}
        for pos in str_positions:
            if lowered[pos] in _BOOL_WORDS:
                kinds["boolstr"] += 1
                bool_words[pos] = lowered[pos]

    for i, v in present:
        if isinstance(v, str):
            continue
        if isinstance(v, bool):
            kinds["bool"] += 1
            numeric[i] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) or hasattr(v, "dtype"):
            number = float(v)
            if math.isfinite(number):
                numeric[i] = number
                kinds["num"] += 1
        elif isinstance(v, (datetime, date)):
            kinds["date"] += 1
    return {"numeric": numeric, "kinds": kinds, "bool_words": bool_words, "present": present}


def _parse_dates(values: list[Any], present: list[tuple[int, Any]]) -> tuple[list[Optional[datetime]], int, int]:
    """(parsed datetimes aligned to values, number parsed, number of date-looking values)."""
    import pandas as pd

    out: list[Optional[datetime]] = [None] * len(values)
    parsed_count = 0
    looking = 0
    string_positions = []
    for i, v in present:
        if isinstance(v, datetime):
            out[i] = v.replace(tzinfo=None)
            parsed_count += 1
            looking += 1
        elif isinstance(v, date):
            out[i] = datetime(v.year, v.month, v.day)
            parsed_count += 1
            looking += 1
        elif isinstance(v, str) and _DATE_RE.match(v.strip()):
            string_positions.append(i)
            looking += 1
    if string_positions:
        texts = [str(values[i]).strip() for i in string_positions]
        dayfirst = _infer_dayfirst(texts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(
                pd.Series(texts), errors="coerce", dayfirst=dayfirst, format="mixed", utc=False
            )
        for pos, stamp in zip(string_positions, parsed):
            if not pd.isna(stamp):
                out[pos] = stamp.to_pydatetime().replace(tzinfo=None)
                parsed_count += 1
    return out, parsed_count, looking


def _infer_dayfirst(texts: list[str]) -> bool:
    """dd/mm vs mm/dd: decided by any component that can only be a day (> 12)."""
    first_big = second_big = False
    for text in texts[:2000]:
        m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}", text)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            first_big |= a > 12
            second_big |= b > 12
    return first_big and not second_big


def infer_column(name: str, original: Optional[str], values: list[Any]) -> TypedColumn:
    """Decide a column's type from its values and attach what later stages need."""
    import numpy as np

    n = len(values)
    info = _classify_values(values)
    kinds: Counter[str] = info["kinds"]
    present: list[tuple[int, Any]] = info["present"]
    count = len(present)
    column = TypedColumn(name=name, original=original, ctype=ColumnType.UNKNOWN, values=values, n_rows=n)
    if count == 0:
        column.flags.append(ColumnFlag.ALL_MISSING)
        return column

    numeric = info["numeric"]
    num_count = kinds["num"] + kinds["bool"]
    plain_numeric = kinds["num"]

    # Boolean: real booleans, yes/no style words, or 0/1 numbers.
    distinct_text = {str(v).strip().lower() if isinstance(v, str) else v for _, v in present}
    bool_like_words = distinct_text <= (_BOOL_WORDS | {"0", "1", 0, 1, 0.0, 1.0, True, False})
    if len(distinct_text) <= 2 and bool_like_words and count > 0 and (
        kinds["bool"] + kinds["boolstr"] > 0 or distinct_text <= {"0", "1", 0, 1, 0.0, 1.0}
    ):
        labels: list[Optional[str]] = [None] * n
        coded = np.full(n, np.nan)
        for i, v in present:
            word = str(v).strip().lower() if not isinstance(v, bool) else ("true" if v else "false")
            labels[i] = word
            coded[i] = 1.0 if word in _BOOL_TRUE | {"1", "1.0"} else 0.0
        column.ctype = ColumnType.BOOLEAN
        column.labels = labels
        column.numeric = coded
        return column

    # Numeric.
    if plain_numeric / count >= TYPE_DOMINANCE and plain_numeric >= 1:
        finite = numeric[~np.isnan(numeric)]
        column.ctype = ColumnType.NUMERIC
        column.numeric = numeric
        column.invalid_count = count - plain_numeric
        column.is_integer = bool(finite.size and np.all(finite == np.floor(finite)))
        if column.invalid_count:
            column.flags.append(ColumnFlag.MIXED_TYPES)
        return column

    # Datetime.
    datetimes, parsed_dates, looking = _parse_dates(values, present)
    # Date-looking is not enough ("1.2.34" is a version): at least half must really parse.
    if looking / count >= TYPE_DOMINANCE and parsed_dates / count >= MIN_PARSED_DATE_SHARE:
        column.ctype = ColumnType.DATETIME
        column.datetimes = datetimes
        column.invalid_count = count - parsed_dates
        if column.invalid_count:
            column.flags.append(ColumnFlag.MIXED_TYPES)
        return column

    # Categorical or free text.
    labels = [None if _is_missing(v) else (v if isinstance(v, str) else str(v)) for v in values]
    non_missing_labels = [x for x in labels if x is not None]
    mean_len = sum(len(x) for x in non_missing_labels) / max(1, len(non_missing_labels))
    column.labels = labels
    column.ctype = ColumnType.TEXT if mean_len > TEXT_MEAN_LENGTH else ColumnType.CATEGORICAL
    if plain_numeric and plain_numeric / count > 0.1:
        column.flags.append(ColumnFlag.MIXED_TYPES)
        column.invalid_count = count - plain_numeric if plain_numeric / count >= 0.5 else 0
    return column


# ---------------------------------------------------------------------------
# Identifier / sensitive evidence
# ---------------------------------------------------------------------------


def _name_tokens(normalized: str) -> set[str]:
    parts = normalized.lower().split("_")
    tokens = set(parts)
    tokens.update("_".join(parts[i : i + 2]) for i in range(len(parts) - 1))
    tokens.add(normalized.lower())
    return tokens


def _is_sequential(numeric: Optional["np.ndarray"]) -> bool:
    """Whether integer values form a run or an arithmetic progression (1, 2, 3 ... or 10, 20, 30 ...).

    Unique integer-valued columns are not all identifiers: prices, populations and counts
    are too. What marks a row identifier is that the values are *assigned in order*.
    """
    import numpy as np

    if numeric is None:
        return False
    data = np.sort(numeric[~np.isnan(numeric)])
    if data.size < 3:
        return False
    steps = np.diff(data)
    if steps.size and np.all(steps == steps[0]) and steps[0] > 0:
        return True
    return bool(data[-1] - data[0] + 1 <= 1.05 * data.size)


def assess_identifier(column: TypedColumn, unique_ratio: float, count: int, thr: float) -> Optional[tuple[float, str]]:
    """Whether a column looks like a row/entity identifier, with confidence and why."""
    name_hit = bool(_IDENTIFIER_NAME.search(column.name))
    unique_hit = unique_ratio >= thr and count >= 10
    shape_hit = False
    if unique_hit:
        if column.ctype is ColumnType.NUMERIC:
            shape_hit = column.is_integer and _is_sequential(column.numeric)
        elif column.ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
            shape_hit = column.ctype is ColumnType.CATEGORICAL
    if name_hit and unique_hit:
        return 0.95, f"named like an identifier and {unique_ratio:.0%} of values are distinct"
    if name_hit:
        return 0.6, "named like an identifier (values repeat, so it may be a foreign key)"
    if unique_hit and shape_hit:
        return 0.6, f"{unique_ratio:.0%} of values are distinct" + (
            " and run in sequence" if column.ctype is ColumnType.NUMERIC else ", as in a row identifier"
        )
    return None


def assess_sensitive(column: TypedColumn, count: int) -> Optional[tuple[float, str]]:
    """Whether a column looks personally identifying, by its name and by value patterns."""
    name_kind = None
    for token in _name_tokens(column.name):
        if token in _SENSITIVE_NAME_TOKENS:
            name_kind = _SENSITIVE_NAME_TOKENS[token]
            break

    value_kind = None
    if column.ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT) and column.labels:
        sample = [x for x in column.labels if x is not None][:PATTERN_SAMPLE]
        if sample:
            def share(pattern) -> float:
                return sum(1 for x in sample if pattern.match(x.strip())) / len(sample)

            if share(_EMAIL_RE) >= 0.6:
                value_kind = "e-mail addresses"
            elif share(_SSN_RE) >= 0.6:
                value_kind = "national-identifier-shaped values"
            elif any(ch in x for x in sample[:20] for ch in "+-() ") and share(_PHONE_RE) >= 0.7:
                value_kind = "phone-number-shaped values"
            elif sum(1 for x in sample if _ADDRESS_RE.search(x)) / len(sample) >= 0.5:
                value_kind = "street addresses"
    if name_kind and value_kind:
        return 0.95, f"named like {name_kind} and the values look like {value_kind}"
    if value_kind:
        return 0.85, f"the values look like {value_kind}"
    if name_kind:
        return 0.6, f"named like {name_kind}"
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Ten significant digits: stable across platforms without flattening tiny values.

    Rounding to a fixed number of decimals would turn 1e-9 into 0.0, silently erasing
    every small-scale feature; significant digits keep the scale.
    """
    return float(f"{float(value):.10g}")


def _numeric_stats(column: TypedColumn) -> Optional[NumericStats]:
    import numpy as np

    arr = column.numeric
    if arr is None:
        return None
    data = arr[~np.isnan(arr)]
    if data.size == 0:
        return None
    return NumericStats(
        min=_round(data.min()),
        max=_round(data.max()),
        mean=_round(data.mean()),
        median=_round(np.median(data)),
        std=_round(data.std(ddof=1)) if data.size >= 2 else None,
        quantiles={k: _round(np.quantile(data, q)) for k, q in QUANTILES.items()},
        zero_count=int((data == 0).sum()),
        negative_count=int((data < 0).sum()),
        is_integer=column.is_integer,
    )


def _granularity(stamps: list[datetime]) -> str:
    unique = sorted(set(stamps))
    if len(unique) < 3:
        return "insufficient"
    diffs = [(b - a).total_seconds() for a, b in zip(unique, unique[1:])]
    common, freq = Counter(diffs).most_common(1)[0]
    if freq / len(diffs) < 0.6:
        return "irregular"
    for limit, label in ((1, "secondly"), (60, "minutely"), (3600, "hourly"), (86400, "daily"), (604800, "weekly")):
        if abs(common - limit) < 0.5:
            return label
    if 28 * 86400 <= common <= 31 * 86400:
        return "monthly"
    if 365 * 86400 <= common <= 366 * 86400:
        return "yearly"
    return "irregular"


def _datetime_stats(column: TypedColumn) -> Optional[DatetimeStats]:
    stamps = [d for d in (column.datetimes or []) if d is not None]
    if not stamps:
        return None
    return DatetimeStats(min=min(stamps), max=max(stamps), granularity=_granularity(stamps))


def _cardinality(unique_count: int) -> str:
    if unique_count <= LOW_CARDINALITY_MAX:
        return "low"
    if unique_count <= MEDIUM_CARDINALITY_MAX:
        return "medium"
    return "high"


def profile_column(column: TypedColumn, settings: Settings) -> ColumnProfile:
    """Compute the profile of one typed column and record its flags."""
    count = column.non_missing
    missing = column.n_rows - count
    labels_or_values: list[Any]
    if column.ctype is ColumnType.NUMERIC and column.numeric is not None:
        import numpy as np

        labels_or_values = [float(x) for x in column.numeric[~np.isnan(column.numeric)]]
    elif column.ctype is ColumnType.DATETIME:
        labels_or_values = [d for d in (column.datetimes or []) if d is not None]
    elif column.labels is not None:
        labels_or_values = [x for x in column.labels if x is not None]
    else:
        labels_or_values = [v for v in column.values if not _is_missing(v)]

    counter = Counter(labels_or_values)
    unique_count = len(counter)
    ratio = round(unique_count / count, 6) if count else 0.0

    flags = list(column.flags)
    if count and unique_count == 1:
        flags.append(ColumnFlag.CONSTANT)
    elif count >= 20 and counter:
        top_share = counter.most_common(1)[0][1] / count
        if top_share >= settings.quality_near_constant_ratio:
            flags.append(ColumnFlag.NEAR_CONSTANT)
    if column.ctype is ColumnType.TEXT:
        flags.append(ColumnFlag.FREE_TEXT)
    if column.ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT) and unique_count > max(
        MEDIUM_CARDINALITY_MAX, 0.5 * count
    ):
        flags.append(ColumnFlag.HIGH_CARDINALITY)

    column.identifier_evidence = assess_identifier(
        column, ratio, count, settings.quality_identifier_unique_ratio
    )
    column.sensitive_evidence = assess_sensitive(column, count)
    if column.identifier_evidence:
        flags.append(ColumnFlag.POSSIBLE_IDENTIFIER)
    if column.sensitive_evidence:
        flags.append(ColumnFlag.POSSIBLE_SENSITIVE_FIELD)
    column.flags = list(dict.fromkeys(flags))

    numeric = _numeric_stats(column) if column.ctype is ColumnType.NUMERIC else None
    dt = _datetime_stats(column) if column.ctype is ColumnType.DATETIME else None
    categorical = None
    text = None
    if column.ctype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN, ColumnType.TEXT):
        suppress = bool(column.sensitive_evidence or column.identifier_evidence)
        top = []
        if not suppress:
            ordered = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:TOP_VALUES]
            top = [ValueCount(value=str(v), count=c, share=round(c / count, 6)) for v, c in ordered]
        categorical = CategoricalStats(
            cardinality=_cardinality(unique_count), top_values=top, values_suppressed=suppress
        )
    if column.ctype is ColumnType.TEXT and column.labels:
        lengths = [len(x) for x in column.labels if x is not None]
        text = TextStats(mean_length=round(sum(lengths) / len(lengths), 4), max_length=max(lengths))

    physical = None
    if column.ctype is ColumnType.NUMERIC:
        physical = "integer" if column.is_integer else "float"
    elif column.ctype is ColumnType.BOOLEAN:
        physical = "boolean"
    elif column.ctype is ColumnType.DATETIME:
        physical = "datetime"
    elif column.ctype in (ColumnType.CATEGORICAL, ColumnType.TEXT):
        physical = "string"

    return ColumnProfile(
        index=0,
        original_name=column.original,
        normalized_name=column.name,
        inferred_type=column.ctype,
        physical_type=physical,
        count=count,
        missing_count=missing,
        missing_pct=round(100.0 * missing / column.n_rows, 4) if column.n_rows else 0.0,
        unique_count=unique_count,
        uniqueness_ratio=ratio,
        flags=column.flags,
        numeric=numeric,
        categorical=categorical,
        datetime=dt,
        text=text,
    )


def profile_table(
    table: ParsedTable, settings: Settings, deadline: Optional[Deadline] = None
) -> list[TypedColumn]:
    """Infer types and profile every column of a parsed file, in column order."""
    columns: list[TypedColumn] = []
    frame = table.frame
    for index, name in enumerate(frame.columns):
        if deadline is not None:
            deadline.check("profiling")
        values = list(frame[name])
        typed = infer_column(name, table.original_names[index], values)
        profile = profile_column(typed, settings)
        profile = profile.model_copy(update={"index": index})
        typed.profile = profile
        columns.append(typed)
    return columns
