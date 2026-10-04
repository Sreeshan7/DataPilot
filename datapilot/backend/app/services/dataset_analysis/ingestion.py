"""Reading an uploaded CSV or Excel file into a table, safely and without guessing quietly.

Everything that can go wrong with a user's file is handled here and reported, never
swallowed: an unusual encoding, a ragged row, a duplicate column name, a header that may
not be a header, a file larger than the limits. The result describes how the file was read
(:class:`IngestionInfo`) so no later stage has to trust that the parse was clean.

Design choices:

* **No dtype guessing by the parser.** CSV cells arrive as strings and Excel cells with
  the types the workbook stored; column types are inferred later, once, by the profiler.
  A library that decides ``007`` is the number 7 would lose information we cannot get back.
* **Hard caps, stated truncation.** Rows, columns and total cells are bounded by settings.
  A file beyond them is truncated and says so; it is never treated as complete.
* **Columns are never silently renamed.** The original header is kept next to the
  normalised name, and every rename is listed with its reason.
* **Nothing is executed.** Excel is opened read-only with cached values; formulas are not
  evaluated and macros are not loaded.
"""

from __future__ import annotations

import csv
import io
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from itertools import chain, islice
from typing import TYPE_CHECKING, Any, Optional

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.schemas.dataset_analysis import ColumnRename, IngestionInfo
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import IngestionError
from app.services.dataset_analysis.safety import Deadline, validate_extension

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = get_logger(__name__)

#: Cells treated as "no value". Compared case-insensitively after trimming. Only tokens
#: that are placeholders and nothing else: "?" (UCI's convention) and "-" are included
#: because a literal one is far rarer than a placeholder.
#:
#: "none", "nil" and "missing" are deliberately NOT here. "None" is a real category in
#: many datasets ("no refund", "no previous defaults"), and treating it as missing would
#: silently erase that value (it once turned a two-valued column into a constant one).
#: Every placeholder actually converted is reported in the ingestion warnings.
MISSING_TOKENS = frozenset(
    {"", "na", "n/a", "nan", "null", "?", "-", "--", "#n/a", "n.a."}
)

_EXCEL_ERRORS = frozenset({"#DIV/0!", "#N/A", "#NAME?", "#NULL!", "#NUM!", "#REF!", "#VALUE!"})

_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
_SNIFF_BYTES = 64 * 1024
_HEADER_LOOKAHEAD = 30
_DEADLINE_EVERY_ROWS = 5000

_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_DATE_RE = re.compile(
    r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})$"
)

#: Magic numbers of files that are certainly not delimited text.
_BINARY_SIGNATURES = (
    b"%PDF", b"MZ", b"\x7fELF", b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"\xd0\xcf\x11\xe0",
    b"Rar!", b"7z\xbc\xaf",
)


@dataclass
class ParsedTable:
    """An uploaded file as a table, plus a description of how it was read."""

    frame: "pd.DataFrame"
    original_names: list[Optional[str]]
    info: IngestionInfo


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def looks_numeric(value: str) -> bool:
    return bool(_NUMERIC_RE.match(value.strip()))


def _normalize_one(raw: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", raw.strip()).strip("_").lower()
    return cleaned


def normalize_column_names(
    raw: list[Optional[str]],
) -> tuple[list[str], list[ColumnRename]]:
    """Unique, filesystem-and-key-safe names, with every change recorded.

    The originals are untouched elsewhere; this only produces the ``normalized_name`` used
    as a key. A name that is empty becomes ``unnamed_<n>``, a duplicate gets ``_2``, ``_3``
    and so on, and a name changed by normalisation alone is recorded too, so nothing is ever
    renamed without a trace.
    """
    normalized: list[str] = []
    renames: list[ColumnRename] = []
    seen: Counter[str] = Counter()
    for index, original in enumerate(raw):
        text = (original or "").strip()
        base = _normalize_one(text) if text else ""
        reason: Optional[str] = None
        if not base:
            base = f"unnamed_{index + 1}" if not text else f"column_{index + 1}"
            reason = "the column header was empty" if not text else "the header had no letters or digits"
        name = base
        seen[base] += 1
        if seen[base] > 1:
            name = f"{base}_{seen[base]}"
            while name in seen:
                seen[base] += 1
                name = f"{base}_{seen[base]}"
            reason = reason or "the column name duplicates an earlier column"
        seen[name] += 0 if name == base else 1
        if reason is None and name != text:
            reason = "normalised to lower-case letters, digits and underscores"
        normalized.append(name)
        if reason is not None:
            renames.append(
                ColumnRename(
                    index=index,
                    original_name=original if text else None,
                    normalized_name=name,
                    reason=reason,
                )
            )
    return normalized, renames


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _decode(data: bytes) -> tuple[str, str, bool]:
    """Text, encoding name, and whether a lossy fallback was needed."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig", False
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16", False
        except UnicodeDecodeError as exc:
            raise IngestionError(
                ErrorCode.FILE_MALFORMED, "The file looks UTF-16 encoded but could not be decoded."
            ) from exc
    try:
        return data.decode("utf-8"), "utf-8", False
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252", True
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1", True


def _check_text_content(sample: bytes) -> None:
    if sample.startswith(b"PK\x03\x04"):
        raise IngestionError(
            ErrorCode.FILE_MALFORMED,
            "This file is a zip/Excel archive, not delimited text. If it is a workbook, "
            "give it the .xlsx extension.",
        )
    if any(sample.startswith(sig) for sig in _BINARY_SIGNATURES):
        raise IngestionError(ErrorCode.FILE_MALFORMED, "This is not a text file.")
    utf16 = sample.startswith((b"\xff\xfe", b"\xfe\xff"))
    if b"\x00" in sample and not utf16:
        raise IngestionError(ErrorCode.FILE_MALFORMED, "The file contains binary data, not text.")


def _sniff_delimiter(text: str, forced: Optional[str] = None) -> str:
    if forced:
        return forced
    sample = text[:_SNIFF_BYTES]
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    if not lines:
        return ","
    best, best_score = ",", (-1, -1)
    for delimiter in _CANDIDATE_DELIMITERS:
        try:
            counts = [len(row) for row in csv.reader(lines, delimiter=delimiter)]
        except csv.Error:
            continue
        if not counts or max(counts) < 2:
            continue
        mode, freq = Counter(counts).most_common(1)[0]
        # Prefer the delimiter giving the most rows of one consistent width, then width.
        score = (freq, mode)
        if mode >= 2 and score > best_score:
            best, best_score = delimiter, score
    return best


def _is_data_like(value: str) -> bool:
    text = value.strip()
    return bool(text) and (looks_numeric(text) or bool(_DATE_RE.match(text)))


def detect_header(rows: list[list[str]]) -> tuple[bool, Optional[str]]:
    """Whether the first row is a header, and a warning when that is only an assumption.

    A first row in which half or more of the cells look like numbers or dates is data. A
    first row of words is a header, and is confirmed when some later column holds numbers
    that the first row does not (the usual signature of a header above numeric data).
    """
    if not rows:
        return True, None
    first = rows[0]
    non_empty = [c for c in first if c.strip()]
    if not non_empty:
        return False, None
    data_like = sum(1 for c in non_empty if _is_data_like(c))
    if data_like / len(non_empty) >= 0.5:
        return False, None
    later = rows[1:]
    confirmed = any(
        index < len(first)
        and not _is_data_like(first[index])
        and sum(1 for r in later if index < len(r) and _is_data_like(r[index]))
        > len(later) / 2
        for index in range(len(first))
    )
    unique_short = len(set(non_empty)) == len(non_empty) and all(len(c) <= 64 for c in non_empty)
    if confirmed or (unique_short and len(later) == 0):
        return True, None
    if unique_short:
        return True, (
            "The first row was assumed to be the header; no numeric column confirms it."
        )
    return True, "The first row was assumed to be the header; some of its values repeat."


def _clean_cell(value: str, seen: Optional[set] = None) -> Optional[str]:
    stripped = value.strip()
    if stripped.lower() in MISSING_TOKENS:
        if seen is not None and stripped:
            seen.add(stripped)
        return None
    return stripped


def _placeholder_warning(seen: set) -> Optional[str]:
    """Name the non-empty placeholders that were read as missing, so nothing is silent."""
    if not seen:
        return None
    listed = ", ".join(repr(t) for t in sorted(seen, key=str.lower)[:10])
    return f"These placeholder values were read as missing: {listed}."


def parse_csv(
    path: str,
    display_name: str,
    size_bytes: int,
    settings: Settings,
    deadline: Optional[Deadline] = None,
    delimiter: Optional[str] = None,
) -> ParsedTable:
    import pandas as pd

    started = time.monotonic()
    with open(path, "rb") as handle:
        data = handle.read()
    if not data.strip():
        raise IngestionError(ErrorCode.FILE_EMPTY, "The file is empty.")
    _check_text_content(data[:8192])

    text, encoding, fallback = _decode(data)
    del data
    chosen = _sniff_delimiter(text, delimiter)

    warnings: list[str] = []
    if fallback:
        warnings.append(
            f"The file is not valid UTF-8; it was read as {encoding}. Accented characters may be wrong."
        )

    previous_limit = csv.field_size_limit()
    csv.field_size_limit(settings.analysis_max_field_chars)
    try:
        return _read_rows(
            text, chosen, encoding, display_name, size_bytes, settings, deadline, warnings, started
        )
    except csv.Error as exc:
        raise IngestionError(
            ErrorCode.FILE_MALFORMED,
            "The file could not be parsed as delimited text (a quoted field is unterminated "
            "or a single cell is larger than the allowed length).",
        ) from exc
    finally:
        csv.field_size_limit(previous_limit)


def _read_rows(
    text: str,
    delimiter: str,
    encoding: str,
    display_name: str,
    size_bytes: int,
    settings: Settings,
    deadline: Optional[Deadline],
    warnings: list[str],
    started: float,
) -> ParsedTable:
    import pandas as pd

    reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=False)
    raw_head = list(islice(reader, _HEADER_LOOKAHEAD))
    head = [row for row in raw_head if any(cell.strip() for cell in row)]
    blank_in_head = len(raw_head) - len(head)
    if not head:
        raise IngestionError(ErrorCode.FILE_EMPTY, "The file contains no data.")

    has_header, header_warning = detect_header(head)
    if header_warning:
        warnings.append(header_warning)

    if has_header:
        raw_header: list[Optional[str]] = [cell.strip() for cell in head[0]]
        ncols = len(raw_header)
        pending = head[1:]
    else:
        widths = Counter(len(row) for row in head)
        ncols = widths.most_common(1)[0][0]
        raw_header = [None] * ncols
        pending = head
        warnings.append("No header row was found; columns are named column_1, column_2, ...")

    if ncols > settings.analysis_max_columns:
        raise IngestionError(
            ErrorCode.FILE_TOO_WIDE,
            f"The file has {ncols} columns; the limit is {settings.analysis_max_columns}.",
            details={"max_columns": settings.analysis_max_columns},
        )
    if ncols == 0:
        raise IngestionError(ErrorCode.FILE_MALFORMED, "No columns could be found.")

    row_limit = min(settings.analysis_max_rows, settings.analysis_max_cells // ncols)
    if row_limit < 1:
        raise IngestionError(
            ErrorCode.FILE_TOO_WIDE,
            "The file has too many columns for the cell limit.",
            details={"max_cells": settings.analysis_max_cells},
        )
    limit_reason = (
        f"the row limit of {settings.analysis_max_rows}"
        if row_limit == settings.analysis_max_rows
        else f"the cell limit of {settings.analysis_max_cells} ({row_limit} rows of {ncols} columns)"
    )

    columns: list[list[Optional[str]]] = [[] for _ in range(ncols)]
    rows_read = malformed = 0
    blank = blank_in_head
    truncated = False
    placeholders: set[str] = set()

    def consume(row: list[str]) -> bool:
        nonlocal rows_read, malformed, blank
        if not any(cell.strip() for cell in row):
            blank += 1
            return True
        if len(row) != ncols:
            extra = row[ncols:]
            if len(row) < ncols or any(cell.strip() for cell in extra):
                malformed += 1
            row = (row + [""] * ncols)[:ncols]
        for index in range(ncols):
            columns[index].append(_clean_cell(row[index], placeholders))
        rows_read += 1
        return rows_read < row_limit

    stream = chain(pending, reader)
    for position, row in enumerate(stream):
        if deadline is not None and position % _DEADLINE_EVERY_ROWS == 0:
            deadline.check("parsing")
        if not consume(row):
            break
    if rows_read >= row_limit:
        # The limit was reached: is there anything left, or did the file end exactly here?
        for row in stream:
            if any(cell.strip() for cell in row):
                truncated = True
                break

    if rows_read == 0:
        raise IngestionError(
            ErrorCode.FILE_NO_DATA_ROWS, "The file has a header but no data rows."
        )

    if blank:
        warnings.append(f"{blank} blank row(s) were skipped.")
    if note := _placeholder_warning(placeholders):
        warnings.append(note)
    if malformed:
        share = malformed / max(1, rows_read + malformed)
        warnings.append(
            f"{malformed} row(s) had the wrong number of fields: short rows were padded with "
            "missing values and extra fields were dropped."
        )
        if share > 0.2:
            warnings.append("More than 20% of rows were malformed; the delimiter may be wrong.")
    if truncated:
        warnings.append(f"The file was truncated at {limit_reason}; statistics describe the rows read.")

    names, renames = normalize_column_names(raw_header)
    frame = pd.DataFrame({name: values for name, values in zip(names, columns)}, dtype=object)
    info = IngestionInfo(
        file_type="csv",
        display_name=display_name,
        size_bytes=size_bytes,
        encoding=encoding,
        delimiter=delimiter,
        header_detected=has_header,
        rows_read=rows_read,
        truncated=truncated,
        truncation_reason=limit_reason if truncated else None,
        malformed_rows=malformed,
        columns_renamed=renames,
        warnings=warnings,
        duration_ms=round((time.monotonic() - started) * 1000, 2),
    )
    return ParsedTable(frame=frame, original_names=[n if n else None for n in raw_header], info=info)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------


def check_xlsx_container(path: str, size_bytes: int, settings: Settings) -> None:
    """Refuse a malformed, encrypted or decompression-bomb workbook before opening it.

    An .xlsx is a zip. A few kilobytes can expand to gigabytes, so the declared
    uncompressed size and ratio are checked from the zip directory, which costs nothing and
    runs before a single byte is inflated.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"PK\x03\x04":
                raise IngestionError(
                    ErrorCode.FILE_MALFORMED, "The file is not a valid .xlsx workbook."
                )
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = {info.filename for info in infos}
            if len(infos) > 10_000:
                raise IngestionError(ErrorCode.FILE_UNSAFE, "The workbook contains too many parts.")
            if any(info.flag_bits & 0x1 for info in infos):
                raise IngestionError(
                    ErrorCode.FILE_MALFORMED,
                    "The workbook is password-protected; remove the password and upload again.",
                )
            if "xl/workbook.xml" not in names:
                raise IngestionError(
                    ErrorCode.FILE_MALFORMED, "The file is not a valid .xlsx workbook."
                )
            total = sum(info.file_size for info in infos)
            if total > settings.analysis_max_excel_uncompressed_bytes:
                raise IngestionError(
                    ErrorCode.FILE_UNSAFE,
                    "The workbook expands to more data than the allowed limit.",
                )
            if total / max(1, size_bytes) > settings.analysis_max_excel_compression_ratio:
                raise IngestionError(
                    ErrorCode.FILE_UNSAFE,
                    "The workbook is compressed far more than a real spreadsheet and was refused.",
                )
    except zipfile.BadZipFile as exc:
        raise IngestionError(
            ErrorCode.FILE_MALFORMED, "The file is not a valid .xlsx workbook."
        ) from exc


def _excel_cell(value: Any, seen: Optional[set] = None) -> tuple[Any, bool]:
    """(cleaned value, was_error_cell)."""
    if value is None:
        return None, False
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in _EXCEL_ERRORS:
            return None, True
        if stripped.lower() in MISSING_TOKENS:
            if seen is not None and stripped:
                seen.add(stripped)
            return None, False
        return stripped, False
    return value, False


def _excel_header_decision(row: tuple[Any, ...]) -> bool:
    cells = [c for c in row if c is not None and not (isinstance(c, str) and not c.strip())]
    if not cells:
        return False
    texts = [c for c in cells if isinstance(c, str)]
    return len(texts) / len(cells) > 0.5 and sum(
        1 for c in texts if _is_data_like(c)
    ) / max(1, len(texts)) < 0.5


def parse_xlsx(
    path: str,
    display_name: str,
    size_bytes: int,
    settings: Settings,
    deadline: Optional[Deadline] = None,
    sheet_name: Optional[str] = None,
) -> ParsedTable:
    import pandas as pd

    started = time.monotonic()
    check_xlsx_container(path, size_bytes, settings)

    from openpyxl import load_workbook

    try:
        workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    except Exception as exc:  # openpyxl raises many types for corrupt input
        logger.warning("Workbook could not be opened (%s)", type(exc).__name__)
        raise IngestionError(
            ErrorCode.FILE_MALFORMED, "The workbook could not be read; it may be corrupt."
        ) from exc

    try:
        names_available = list(workbook.sheetnames)
        if len(names_available) > settings.analysis_max_excel_sheets:
            raise IngestionError(
                ErrorCode.EXCEL_TOO_MANY_SHEETS,
                f"The workbook has {len(names_available)} sheets; the limit is "
                f"{settings.analysis_max_excel_sheets}.",
                details={"max_sheets": settings.analysis_max_excel_sheets},
            )
        if not names_available:
            raise IngestionError(ErrorCode.FILE_EMPTY, "The workbook has no sheets.")

        chosen = _choose_sheet(workbook, names_available, sheet_name)
        return _read_sheet(
            workbook[chosen], chosen, names_available, display_name, size_bytes, settings,
            deadline, started,
        )
    except IngestionError:
        raise
    except Exception as exc:
        logger.warning("Sheet could not be read (%s)", type(exc).__name__)
        raise IngestionError(
            ErrorCode.FILE_MALFORMED, "The worksheet could not be read; it may be corrupt."
        ) from exc
    finally:
        try:
            workbook.close()
        except Exception:  # pragma: no cover - closing a read-only book is best effort
            pass


def _choose_sheet(workbook, names: list[str], requested: Optional[str]) -> str:
    if requested is not None:
        if requested in names:
            return requested
        folded = [n for n in names if n.lower() == requested.lower()]
        if folded:
            return folded[0]
        raise IngestionError(
            ErrorCode.EXCEL_SHEET_NOT_FOUND,
            "The requested sheet does not exist in the workbook.",
            details={"sheets_available": names},
        )
    for name in names:
        sheet = workbook[name]
        if getattr(sheet, "sheet_state", "visible") != "visible":
            continue
        for row in sheet.iter_rows(values_only=True, max_row=50):
            if any(c is not None and str(c).strip() for c in row):
                return name
    raise IngestionError(ErrorCode.FILE_EMPTY, "Every sheet in the workbook is empty.")


def _read_sheet(
    sheet,
    sheet_name: str,
    sheets_available: list[str],
    display_name: str,
    size_bytes: int,
    settings: Settings,
    deadline: Optional[Deadline],
    started: float,
) -> ParsedTable:
    import pandas as pd

    rows = sheet.iter_rows(values_only=True)
    first: Optional[tuple[Any, ...]] = None
    for row in rows:
        if any(c is not None and str(c).strip() for c in row):
            first = row
            break
    if first is None:
        raise IngestionError(ErrorCode.FILE_EMPTY, "The selected sheet is empty.")

    warnings: list[str] = []
    has_header = _excel_header_decision(first)
    width = len(first)
    while width and (first[width - 1] is None or (isinstance(first[width - 1], str) and not first[width - 1].strip())) and has_header:
        width -= 1
    ncols = width
    if ncols == 0:
        raise IngestionError(ErrorCode.FILE_EMPTY, "The selected sheet has no columns.")
    if ncols > settings.analysis_max_columns:
        raise IngestionError(
            ErrorCode.FILE_TOO_WIDE,
            f"The sheet has {ncols} columns; the limit is {settings.analysis_max_columns}.",
            details={"max_columns": settings.analysis_max_columns},
        )

    row_limit = min(settings.analysis_max_rows, settings.analysis_max_cells // ncols)
    if row_limit < 1:
        raise IngestionError(ErrorCode.FILE_TOO_WIDE, "The sheet has too many columns for the cell limit.")
    limit_reason = (
        f"the row limit of {settings.analysis_max_rows}"
        if row_limit == settings.analysis_max_rows
        else f"the cell limit of {settings.analysis_max_cells}"
    )

    if has_header:
        raw_header: list[Optional[str]] = [
            ("" if c is None else str(c).strip()) for c in first[:ncols]
        ]
    else:
        raw_header = [None] * ncols
        warnings.append("No header row was found; columns are named column_1, column_2, ...")

    columns: list[list[Any]] = [[] for _ in range(ncols)]
    rows_read = blank = error_cells = 0
    truncated = False
    placeholders: set[str] = set()

    def consume(row: tuple[Any, ...]) -> bool:
        nonlocal rows_read, blank, error_cells
        cells = list(row[:ncols]) + [None] * max(0, ncols - len(row))
        # A row is blank only if it was empty as written. A row whose cells are all
        # placeholders ("NA") or errors is data with missing values, not a blank row.
        if all(c is None or (isinstance(c, str) and not c.strip()) for c in cells):
            blank += 1
            return True
        cleaned = []
        for value in cells:
            cell, was_error = _excel_cell(value, placeholders)
            error_cells += was_error
            cleaned.append(cell)
        for index, value in enumerate(cleaned):
            columns[index].append(value)
        rows_read += 1
        return rows_read < row_limit

    keep_going = True
    if not has_header:
        keep_going = consume(first)
    position = 0
    for row in rows:
        if not keep_going:
            if any(c is not None and str(c).strip() for c in row):
                truncated = True
                break
            continue
        if deadline is not None and position % _DEADLINE_EVERY_ROWS == 0:
            deadline.check("parsing")
        position += 1
        keep_going = consume(row)

    if rows_read == 0:
        raise IngestionError(ErrorCode.FILE_NO_DATA_ROWS, "The sheet has a header but no data rows.")
    if blank:
        warnings.append(f"{blank} blank row(s) were skipped.")
    if note := _placeholder_warning(placeholders):
        warnings.append(note)
    if error_cells:
        warnings.append(f"{error_cells} cell(s) contained spreadsheet errors and were treated as missing.")
    if truncated:
        warnings.append(f"The sheet was truncated at {limit_reason}; statistics describe the rows read.")
    if len(sheets_available) > 1:
        warnings.append(
            f"The workbook has {len(sheets_available)} sheets; only '{sheet_name}' was analysed."
        )

    names, renames = normalize_column_names(raw_header)
    frame = pd.DataFrame({name: values for name, values in zip(names, columns)}, dtype=object)
    info = IngestionInfo(
        file_type="xlsx",
        display_name=display_name,
        size_bytes=size_bytes,
        sheet_name=sheet_name,
        sheets_available=sheets_available,
        header_detected=has_header,
        rows_read=rows_read,
        truncated=truncated,
        truncation_reason=limit_reason if truncated else None,
        columns_renamed=renames,
        warnings=warnings,
        duration_ms=round((time.monotonic() - started) * 1000, 2),
    )
    return ParsedTable(frame=frame, original_names=[n if n else None for n in raw_header], info=info)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def parse_file(
    path: str,
    display_name: str,
    size_bytes: int,
    settings: Settings,
    deadline: Optional[Deadline] = None,
    sheet_name: Optional[str] = None,
) -> ParsedTable:
    """Parse a stored upload according to its (validated) extension."""
    kind = validate_extension(display_name)
    if kind == "xlsx":
        return parse_xlsx(path, display_name, size_bytes, settings, deadline, sheet_name)
    delimiter = "\t" if display_name.lower().endswith(".tsv") else None
    return parse_csv(path, display_name, size_bytes, settings, deadline, delimiter)
