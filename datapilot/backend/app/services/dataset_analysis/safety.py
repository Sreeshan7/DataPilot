"""Safe handling of uploaded files: names, temporary storage, and the time budget.

Rules this module exists to enforce:

* The user's filename is only ever *displayed*. It is sanitised for that purpose and is
  never used to build a path, so path traversal has nothing to act on.
* An upload is copied to a randomly named temporary file in chunks, with the size cap
  enforced as bytes arrive. Content-Length is never trusted, and an oversized upload is
  stopped as soon as it crosses the cap instead of being read in full.
* The temporary file is always removed, and a failure to remove it is reported without
  ever revealing where it was.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from app.core.logging_config import get_logger
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisTimeout, IngestionError

logger = get_logger(__name__)

CHUNK_BYTES = 64 * 1024
MAX_DISPLAY_NAME_CHARS = 100

#: Extensions DataPilot parses. ``.xls`` (legacy binary Excel) and ``.xlsm`` (macro
#: workbooks) are refused on purpose and say how to proceed.
SUPPORTED_EXTENSIONS = {".csv": "csv", ".tsv": "csv", ".xlsx": "xlsx"}

_UNSAFE_NAME_CHARS = re.compile(r"[^\w.\- ()]+")


def sanitize_display_name(name: Optional[str]) -> str:
    """A safe label for the uploaded file. Never a path and never used as one."""
    if not name:
        return "upload"
    base = re.split(r"[\\/]", name.replace("\x00", ""))[-1]
    base = "".join(ch for ch in base if ch.isprintable())
    base = _UNSAFE_NAME_CHARS.sub("_", " ".join(base.split())).strip(" .")
    if not base:
        return "upload"
    if len(base) > MAX_DISPLAY_NAME_CHARS:
        stem, dot, ext = base.rpartition(".")
        if dot and len(ext) <= 8:
            base = stem[: MAX_DISPLAY_NAME_CHARS - len(ext) - 1] + "." + ext
        else:
            base = base[:MAX_DISPLAY_NAME_CHARS]
    return base


def file_extension(display_name: str) -> str:
    return Path(display_name).suffix.lower()


def validate_extension(display_name: str) -> str:
    """Return the parser key ('csv' or 'xlsx') for a supported file, else raise."""
    ext = file_extension(display_name)
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    if ext == ".xls":
        raise IngestionError(
            ErrorCode.FILE_TYPE_UNSUPPORTED,
            "Legacy .xls workbooks are not supported. Save the file as .xlsx or .csv and upload again.",
            status_code=415,
        )
    if ext == ".xlsm":
        raise IngestionError(
            ErrorCode.FILE_TYPE_UNSUPPORTED,
            "Macro-enabled workbooks (.xlsm) are not supported. Save a copy as .xlsx or .csv.",
            status_code=415,
        )
    raise IngestionError(
        ErrorCode.FILE_TYPE_UNSUPPORTED,
        "Only .csv, .tsv and .xlsx files can be analysed.",
        status_code=415,
        details={"supported_extensions": sorted(SUPPORTED_EXTENSIONS)},
    )


class StoredUpload:
    """An upload copied to private temporary storage."""

    def __init__(self, path: str, size_bytes: int) -> None:
        self._path = path
        self.size_bytes = size_bytes
        self.cleanup_failed = False

    @property
    def path(self) -> str:
        return self._path

    def remove(self) -> None:
        """Delete the file. A failure is recorded and logged without the path."""
        try:
            os.remove(self._path)
        except FileNotFoundError:
            return
        except OSError as exc:
            self.cleanup_failed = True
            logger.warning("A temporary upload could not be removed (%s)", type(exc).__name__)


@contextmanager
def store_upload(stream: BinaryIO, max_bytes: int) -> Iterator[StoredUpload]:
    """Copy ``stream`` to a randomly named temporary file, enforcing ``max_bytes``.

    The cap is checked per chunk as data arrives, so a hostile upload is cut off at the
    limit and never held in memory. The file is deleted when the block exits, whether
    it succeeds or raises.
    """
    fd, path = tempfile.mkstemp(prefix="datapilot_", suffix=".upload")
    stored = StoredUpload(path, 0)
    try:
        total = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = stream.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise IngestionError(
                        ErrorCode.FILE_TOO_LARGE,
                        f"The file is larger than the {max_bytes // (1024 * 1024)} MiB limit.",
                        status_code=413,
                        details={"max_bytes": max_bytes},
                    )
                out.write(chunk)
        stored.size_bytes = total
        yield stored
    finally:
        stored.remove()


class Deadline:
    """A wall-clock budget checked between stages and inside long loops.

    Python cannot interrupt a running computation, so work calls :meth:`check` at stage
    boundaries and every few thousand rows; once the budget is spent the next check raises
    :class:`AnalysisTimeout`.
    """

    def __init__(self, seconds: float, clock=time.monotonic) -> None:
        self._clock = clock
        self._limit = clock() + seconds

    def check(self, stage: str) -> None:
        if self._clock() > self._limit:
            raise AnalysisTimeout(stage)

    def remaining(self) -> float:
        return max(0.0, self._limit - self._clock())
