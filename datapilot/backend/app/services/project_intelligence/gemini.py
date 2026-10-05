"""Optional Gemini pass that rewrites the prose of project ideas, and nothing else.

One structured call, only when the caller asks for it, the feature is enabled and a key is
configured. Software has already fixed every idea's target, task and features; the model may only
rephrase ``title``, ``problem_statement``, ``why_it_fits`` and ``extension_potential``.

What this module guarantees, whatever the model returns:

* the request carries no cell value and no identifier or sensitive column name, and the user's
  goal and the column names sit in a ``<data>`` block labelled as untrusted;
* the reply is validated per idea. It is rejected (and the deterministic prose kept) if it names a
  backticked column that is not allowed, promises an accuracy, claims causation, contains a link or
  markdown structure, is empty, or is over length;
* a failed, timed-out or unconfigured call never fails the analysis: the deterministic ideas are
  returned with a status and a warning;
* the API key is never read here, logged or placed in any message.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional, Protocol

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.prompts.project_ideas import build_project_idea_prose_prompt
from app.schemas.project_intelligence import (
    GeminiStatus,
    GeminiUsage,
    IdeaSource,
    LLMIdeaProseSet,
    ProjectIdea,
)
from app.services.llm_service import LLMServiceError
from app.services.project_intelligence.shape import DataShape

logger = get_logger(__name__)

MAX_TITLE = 100
MAX_PROSE = 600
MAX_GOAL_CHARS = 500
_SAFE_NAME = re.compile(r"^[a-z0-9_]{1,60}$")
_BACKTICKED = re.compile(r"`([^`\n]*)`")
_PROMISE = re.compile(
    r"\b\d{1,3}(?:\.\d+)?\s*%|\bguarantee\w*|\bwill\s+(?:achieve|reach|attain|deliver)\b|"
    r"\b(?:achiev|reach|attain)\w*\s+(?:an?\s+)?(?:accuracy|f1|auc|r2|r-squared|score)\b|\baccuracy\s+of\s+\d",
    re.I)
_CAUSAL = re.compile(r"\bcaus(?:e|es|ed|ing|al|ation)\b|\bdrives?\b|\bdriven\s+by\b", re.I)
_STRUCTURE = re.compile(r"https?://|www\.|^\s*#{1,6}\s|```|\[[^\]]*\]\([^)]*\)|<[a-z/][^>]*>", re.I | re.M)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class IdeaWriter(Protocol):
    @property
    def is_configured(self) -> bool: ...

    async def generate_project_idea_prose(self, user_prompt: str) -> LLMIdeaProseSet: ...


def _clean_goal(goal: Optional[str]) -> Optional[str]:
    if not goal:
        return None
    text = _CONTROL.sub(" ", " ".join(goal.split()))
    return text[:MAX_GOAL_CHARS] or None


def build_payload(ideas: list[ProjectIdea], shape: DataShape, goal: Optional[str], settings: Settings) -> tuple[dict, set[str]]:
    """The document sent to the model, and the set of column names its reply may mention."""
    withheld = {c.name for c in shape.columns if c.identifier or c.sensitive}
    referenced: list[str] = []
    for idea in ideas:
        for name in ([idea.target] if idea.target else []) + idea.required_features + idea.optional_features:
            if name not in referenced:
                referenced.append(name)
    allowed = [n for n in referenced if n not in withheld and _SAFE_NAME.match(n)]
    cap = settings.analysis_gemini_max_columns
    allowed = allowed[:cap]
    allowed_set = set(allowed)
    types = {c.name: c.kind.value for c in shape.columns}
    payload = {
        "project_goal": _clean_goal(goal),
        "dataset": {
            "rows": shape.n_rows,
            "source": "an uploaded file" if shape.parsed else "a public registry's published schema",
        },
        "allowed_columns": [{"name": n, "type": types.get(n, "unknown")} for n in allowed],
        "ideas": [
            {
                "id": idea.id,
                "task": idea.task.value,
                "target": idea.target if idea.target in allowed_set else None,
                "required_features": [f for f in idea.required_features if f in allowed_set],
                "optional_features": [f for f in idea.optional_features if f in allowed_set],
            }
            for idea in ideas
        ],
    }
    return payload, allowed_set


def _reject_reason(text: str, limit: int, allowed: set[str]) -> Optional[str]:
    if not text or not text.strip():
        return "empty"
    if len(text) > limit:
        return "too long"
    if _CONTROL.search(text) or _STRUCTURE.search(text):
        return "contains a link, markdown structure or control characters"
    if _PROMISE.search(text):
        return "promises or states an accuracy"
    if _CAUSAL.search(text):
        return "claims cause and effect"
    for span in _BACKTICKED.findall(text):
        if span not in allowed:
            return "names a column that is not in the schema"
    return None


async def enhance_ideas(
    ideas: list[ProjectIdea],
    shape: DataShape,
    goal: Optional[str],
    writer: Optional[IdeaWriter],
    settings: Settings,
    *,
    requested: bool,
) -> tuple[list[ProjectIdea], GeminiUsage, list[str]]:
    """Return the ideas (possibly with better prose), what happened, and any warnings."""
    if not requested:
        return ideas, GeminiUsage(status=GeminiStatus.NOT_REQUESTED), []
    if not settings.analysis_gemini_enabled:
        return ideas, GeminiUsage(status=GeminiStatus.DISABLED, note="AI-written prose is turned off on this server."), []
    if writer is None or not writer.is_configured:
        return ideas, GeminiUsage(
            status=GeminiStatus.NOT_CONFIGURED, note="No Gemini key is configured, so the ideas use DataPilot's own wording."), []
    if not ideas:
        return ideas, GeminiUsage(status=GeminiStatus.NOT_REQUESTED, note="There were no ideas to rephrase."), []

    payload, allowed = build_payload(ideas, shape, goal, settings)
    try:
        reply = await asyncio.wait_for(
            writer.generate_project_idea_prose(build_project_idea_prose_prompt(payload)),
            timeout=settings.llm_request_timeout)
    except asyncio.TimeoutError:
        code = "LLM_TIMEOUT"
    except LLMServiceError as exc:
        code = exc.code
    except Exception as exc:  # never let the optional layer break the analysis
        code = type(exc).__name__
    else:
        return _merge(ideas, reply, allowed)
    logger.info("Gemini idea prose unavailable: %s", code)  # the code only: never the prompt or reply
    return ideas, GeminiUsage(
        status=GeminiStatus.FAILED, note=f"The AI service did not respond usefully ({code}); the ideas use DataPilot's own wording."), [
        "AI-written prose was requested but could not be obtained; the deterministic ideas are shown."]


def _merge(ideas: list[ProjectIdea], reply: LLMIdeaProseSet, allowed: set[str]) -> tuple[list[ProjectIdea], GeminiUsage, list[str]]:
    by_id = {}
    for entry in reply.ideas:
        by_id.setdefault(entry.id, entry)  # a repeated id: the first wins
    merged: list[ProjectIdea] = []
    enhanced = rejected = 0
    warnings: list[str] = []
    for idea in ideas:
        entry = by_id.get(idea.id)
        if entry is None:
            merged.append(idea)
            rejected += 1
            warnings.append(f"The AI reply had no text for idea '{idea.id}'; its own wording was kept.")
            continue
        problems = {
            "title": _reject_reason(entry.title, MAX_TITLE, allowed),
            "problem_statement": _reject_reason(entry.problem_statement, MAX_PROSE, allowed),
            "why_it_fits": _reject_reason(entry.why_it_fits, MAX_PROSE, allowed),
            "extension_potential": _reject_reason(entry.extension_potential, MAX_PROSE, allowed),
        }
        bad = {k: v for k, v in problems.items() if v}
        if bad:
            merged.append(idea)
            rejected += 1
            first = next(iter(bad.items()))
            warnings.append(f"AI text for idea '{idea.id}' was discarded ({first[0]} {first[1]}); its own wording was kept.")
            continue
        merged.append(idea.model_copy(update={
            "title": entry.title.strip(), "problem_statement": entry.problem_statement.strip(),
            "why_it_fits": entry.why_it_fits.strip(), "extension_potential": entry.extension_potential.strip(),
            "source": IdeaSource.GEMINI_ENHANCED,
        }))
        enhanced += 1
    status = GeminiStatus.USED if enhanced and not rejected else GeminiStatus.PARTIAL if enhanced else GeminiStatus.FAILED
    note = None if status is GeminiStatus.USED else (
        f"{enhanced} of {len(ideas)} ideas were rephrased; the rest kept DataPilot's own wording after validation.")
    return merged, GeminiUsage(status=status, ideas_enhanced=enhanced, ideas_rejected=rejected, note=note), warnings
