"""Read a dataset's real column header (and column types) from the first bytes of its file.

Registries such as Kaggle publish no structured schema, but the file itself has one: its first line. Reading only the
first few KB gives the *actual* column names and, from a handful of rows, whether each column is numeric, yes/no or
categorical. Nothing is guessed and no cell value leaves this module: callers receive names, kinds and counts only.

This is not an analysis of the dataset: a few KB says nothing reliable about missingness, balance or importance, and
the response keeps saying so.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Optional

PEEK_BYTES = 64 * 1024
MAX_COLUMNS = 500
MAX_NAME = 80
MAX_SAMPLE_ROWS = 300
QUANTITY_DISTINCT = 15
_NAME_OK = re.compile(r"^[\w .\-()/%#:+]+$", re.UNICODE)
_BOOL = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}
_MISSING = {"", "na", "n/a", "nan", "null", "none", "?", "-"}
_PEEKABLE = (".csv", ".tsv", ".txt")


@dataclass(frozen=True)
class SampleColumn:
    name: str
    kind: str  # Continuous | Integer | Binary | Categorical | Text
    role: Optional[str] = None  # "row_id" when the sampled values are unique per row


@dataclass(frozen=True)
class SampleSchema:
    columns: list[SampleColumn]
    delimiter: str
    sample_rows: int
    bytes_read: int
    total_bytes: Optional[int]
    avg_row_bytes: Optional[float]

    @property
    def estimated_rows(self) -> Optional[int]:
        """About how many rows the whole file has: file size over the average sampled row. An estimate only."""
        if self.total_bytes and self.avg_row_bytes and self.avg_row_bytes > 0 and self.sample_rows >= 20:
            return int(self.total_bytes / self.avg_row_bytes)
        return None


def choose_file(names: list[str]) -> Optional[str]:
    """The file most likely to be the main table: a csv/tsv that is not obviously a test split or a sample submission."""
    candidates = [n for n in names if n.lower().endswith(_PEEKABLE)]
    if not candidates:
        return None

    def rank(name: str) -> tuple[int, int, str]:
        low = name.lower()
        penalty = sum(w in low for w in ("test", "submission", "sample", "readme", "metadata", "dictionary"))
        boost = 0 if any(w in low for w in ("train", "data", "main")) else 1
        return penalty, boost, low

    return sorted(candidates, key=rank)[0]


def _decode(raw: bytes) -> Optional[str]:
    if b"\x00" in raw[:4096]:
        return None  # binary: not a delimited text file
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def _kind(values: list[str]) -> str:
    present = [v.strip() for v in values if v.strip().lower() not in _MISSING]
    if not present:
        return "Categorical"
    lowered = {v.lower() for v in present}
    if len(lowered) == 2 or (len(lowered) == 1 and lowered <= _BOOL):
        return "Binary"  # two values: two classes (yes/no, M/F, 0/1)
    numeric = 0
    integer = 0
    for v in present:
        try:
            float(v.replace(",", ""))
        except ValueError:
            continue
        numeric += 1
        integer += bool(re.fullmatch(r"[+-]?\d+", v))
    if numeric / len(present) >= 0.9:
        # Many distinct values in the sample means a quantity (whole-number scores and counts included); a few distinct
        # whole numbers could be class codes, so those stay "Integer" and the task is not guessed from them.
        if len(set(present)) > QUANTITY_DISTINCT:
            return "Continuous"
        return "Integer" if integer == numeric else "Continuous"
    avg = sum(len(v) for v in present) / len(present)
    return "Text" if avg > 40 else "Categorical"


_ID_NAME = re.compile(r"(^|[_\W])(id|uuid|guid|index|idx|key|number|no)($|[_\W])", re.I)


def _role(name: str, values: list[str]) -> Optional[str]:
    """"row_id" when every sampled value is different and the name (or the sheer uniqueness) says identifier."""
    present = [v.strip() for v in values if v.strip().lower() not in _MISSING]
    if len(present) < 30 or len(set(present)) / len(present) < 0.98:
        return None
    if _ID_NAME.search(name):
        return "row_id"
    return "row_id" if len(present) >= 100 and _kind(present) in ("Text", "Categorical") else None


def sample_schema(raw: bytes, total_bytes: Optional[int] = None) -> Optional[SampleSchema]:
    """Column names and kinds from the head of a delimited file, or ``None`` if it does not look like one."""
    text = _decode(raw)
    if not text or "\n" not in text:
        return None
    complete = text[: text.rfind("\n")] if len(raw) >= PEEK_BYTES - 1 else text  # a cut-off last row is dropped
    first_line = complete.split("\n", 1)[0]
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = max(",;\t|", key=first_line.count)
        if first_line.count(delimiter) == 0:
            return None
    rows = []
    for row in csv.reader(io.StringIO(complete), delimiter=delimiter):
        rows.append(row)
        if len(rows) > MAX_SAMPLE_ROWS:
            break
    if len(rows) < 1:
        return None
    header = [h.strip() for h in rows[0]]
    if not 1 < len(header) <= MAX_COLUMNS:
        return None
    # A header is names, not data: reject lines whose "names" are mostly numbers or look like prose.
    if sum(bool(re.fullmatch(r"[+-]?\d+(\.\d+)?", h)) for h in header) > len(header) / 2:
        return None
    names = []
    for i, h in enumerate(header):
        clean = " ".join(h.split())[:MAX_NAME]
        if not clean or not _NAME_OK.match(clean):
            clean = f"column_{i + 1}"
        names.append(clean)
    body = [r for r in rows[1:] if len(r) == len(header)]
    columns = [SampleColumn(name, _kind([r[i] for r in body]) if body else "Categorical", _role(name, [r[i] for r in body]))
               for i, name in enumerate(names)]
    consumed = sum(len(delimiter.join(r)) + 1 for r in body)
    avg = (consumed / len(body)) if body else None
    return SampleSchema(columns, delimiter, len(body), len(raw), total_bytes, avg)
