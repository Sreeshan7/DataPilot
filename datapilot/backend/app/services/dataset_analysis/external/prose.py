"""Pulling a column list out of a registry's prose description.

Kaggle publishes no structured schema, but many owners list their columns in the
description. That text is the owner's statement, yet reading it with a pattern is
DataPilot's own interpretation and may be partial or wrong, so anything extracted here is
returned with a method name and is *always* reported at ``INFERRED`` evidence, never as a
registry-verified schema.

Three layouts are recognised, strongest first: a markdown table with a column-name header,
a bullet/numbered list under a "columns"-style heading, and an inline "Columns: a, b, c".
A candidate list is accepted only if it is plausible (at least three names, short, mostly
identifier-like); anything that does not look like a schema yields nothing rather than a guess.
"""

from __future__ import annotations

import re
from typing import Optional

MIN_COLUMNS = 3
MAX_COLUMNS = 300
MAX_NAME_CHARS = 60

_HEADER_WORDS = frozenset(
    {"column", "columns", "field", "fields", "name", "variable", "variables", "feature",
     "features", "attribute", "attributes", "column name", "feature name", "variable name",
     "field name", "attribute name"}
)
_HEADING_RE = re.compile(r"(?i)\b(columns?|features?|attributes?|variables?|fields?)\b")
_BULLET_RE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)])\s*[`*_\"']*([A-Za-z0-9_][\w \-./()%#]{0,58}?)[`*_\"']*\s*(?:[:–—\-(]|$)"
)
_INLINE_RE = re.compile(
    r"(?i)\b(?:columns?|features?|attributes?|variables?|fields?)\s*(?:are|include|includes|contains?|:|-)\s*([^\n.]{10,600})"
)
_NAME_OK = re.compile(r"^[A-Za-z0-9_][\w \-./()%#]*$")


def _clean(name: str) -> str:
    """Trim surrounding markdown/quotes only; underscores inside a name are part of it."""
    return name.strip().strip("`*\"' ").strip()


def _plausible(names: list[str]) -> bool:
    if not (MIN_COLUMNS <= len(names) <= MAX_COLUMNS):
        return False
    if len(set(n.lower() for n in names)) != len(names):
        return False
    valid = [n for n in names if 0 < len(n) <= MAX_NAME_CHARS and _NAME_OK.match(n)]
    return len(valid) / len(names) >= 0.9


def _table(lines: list[str]) -> Optional[list[str]]:
    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip().strip("*`").lower() for c in line.strip().strip("|").split("|")]
        index = next((k for k, c in enumerate(cells) if c in _HEADER_WORDS), None)
        if index is None:
            continue
        names: list[str] = []
        for row in lines[i + 1 :]:
            if not row.strip().startswith("|"):
                break
            parts = [c.strip() for c in row.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", p) for p in parts if p):
                continue
            if index < len(parts) and parts[index]:
                names.append(_clean(parts[index]))
        if _plausible(names):
            return names
    return None


def _bullets(lines: list[str]) -> Optional[list[str]]:
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not _HEADING_RE.search(stripped):
            continue
        if not (stripped.endswith(":") or stripped.startswith("#") or len(stripped) < 40):
            continue
        names: list[str] = []
        for row in lines[i + 1 :]:
            if not row.strip():
                if names:
                    break
                continue
            match = _BULLET_RE.match(row)
            if not match:
                break
            names.append(_clean(match.group(1)))
        if _plausible(names):
            return names
    return None


def _inline(text: str) -> Optional[list[str]]:
    for match in _INLINE_RE.finditer(text):
        parts = re.split(r",|;|\band\b", match.group(1))
        names = [_clean(p) for p in parts if p.strip()]
        names = [n for n in names if len(n.split()) <= 4]
        if _plausible(names):
            return names
    return None


def extract_columns(description: Optional[str]) -> tuple[list[str], str]:
    """(column names, method) pulled from prose, or ``([], "")`` when nothing credible is found."""
    if not description or len(description) < 20:
        return [], ""
    lines = description.replace("\r\n", "\n").split("\n")
    for method, finder in (("markdown table", lambda: _table(lines)),
                           ("bullet list", lambda: _bullets(lines)),
                           ("inline list", lambda: _inline(description))):
        names = finder()
        if names:
            return names, method
    return [], ""
