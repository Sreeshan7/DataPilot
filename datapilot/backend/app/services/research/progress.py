"""Live progress for a research run. Events describe what the pipeline is really doing, as it does it.

The service reports each stage through a ``Progress``; with no listener every call is a no-op, so the plain
``/research/discover`` endpoint behaves exactly as before. ``/research/stream`` forwards the events as
Server-Sent Events. Events carry only stage ids, fixed labels and counts: never project text or retrieved content.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Optional

Listener = Callable[[dict], Awaitable[None]]


class Progress:
    def __init__(self, listener: Optional[Listener] = None) -> None:
        self._listener = listener

    async def _send(self, event: dict) -> None:
        if self._listener is not None:
            try:
                await self._listener(event)
            except Exception:  # progress is cosmetic: it must never break the research itself
                self._listener = None

    async def plan(self, steps: list[tuple[str, str]]) -> None:
        await self._send({"type": "plan", "steps": [{"id": i, "label": label} for i, label in steps]})

    async def start(self, step_id: str, detail: Optional[str] = None) -> None:
        await self._send({"type": "step", "id": step_id, "state": "running", "detail": detail})

    async def done(self, step_id: str, detail: Optional[str] = None) -> None:
        await self._send({"type": "step", "id": step_id, "state": "done", "detail": detail})

    async def skip(self, step_id: str, detail: Optional[str] = None) -> None:
        await self._send({"type": "step", "id": step_id, "state": "skipped", "detail": detail})
