"""Live progress for long requests, reported from wherever the work really happens.

``bind`` attaches a sink for the current request (a context variable, so worker threads started with
``asyncio.to_thread`` report to the same request). Anywhere in the pipeline, ``start`` / ``done`` / ``skip`` announce a
stage; with no sink bound every call is a no-op, so the plain JSON endpoints behave exactly as before.

Events carry only a stage id, a fixed label and counts: never user text, file contents or retrieved content. A failing
sink is dropped silently: progress is cosmetic and must never break the work it describes.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

Sink = Callable[[dict], None]
_sink: ContextVar[Optional[Sink]] = ContextVar("datapilot_progress_sink", default=None)


@contextmanager
def bind(sink: Optional[Sink]) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def _emit(event: dict) -> None:
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        _sink.set(None)


def plan(steps: list[tuple[str, str]]) -> None:
    _emit({"type": "plan", "steps": [{"id": i, "label": label} for i, label in steps]})


def _step(step_id: str, state: str, label: Optional[str], detail: Optional[str]) -> None:
    event = {"type": "step", "id": step_id, "state": state, "detail": detail}
    if label:
        event["label"] = label
    _emit(event)


def start(step_id: str, label: Optional[str] = None, detail: Optional[str] = None) -> None:
    _step(step_id, "running", label, detail)


def done(step_id: str, detail: Optional[str] = None, label: Optional[str] = None) -> None:
    _step(step_id, "done", label, detail)


def skip(step_id: str, detail: Optional[str] = None, label: Optional[str] = None) -> None:
    _step(step_id, "skipped", label, detail)
