"""Errors raised by dataset analysis, each carrying a stable code and an HTTP status.

Messages are written for the user and are always safe to return: they never contain a
filesystem path, an upstream response body, a cell value, or an exception string.
"""

from __future__ import annotations

from typing import Any, Optional

from app.schemas.project import ErrorCode


class AnalysisError(Exception):
    """A request that cannot be analysed, for a reason the user can act on."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        status_code: int = 422,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class IngestionError(AnalysisError):
    """The uploaded file could not be read safely."""


class AnalysisTimeout(AnalysisError):
    def __init__(self, stage: str) -> None:
        super().__init__(
            ErrorCode.ANALYSIS_TIMEOUT,
            "The analysis took longer than the allowed time and was stopped. Try a smaller file.",
            status_code=504,
            details={"stage": stage},
        )
