"""A shared, budgeted pool of Gemini API keys.

Every Gemini call in DataPilot goes through here, so the free-tier limits are enforced in one place:

* a sliding one-minute window and a per-day counter *per key*, both set below the published limits
  (``gemini_requests_per_minute`` / ``gemini_requests_per_day``), so DataPilot stops itself before Google stops it;
* the key with the most headroom is used next; a 429 or 503 benches that key briefly and the call is retried
  once on the other key (never in a loop: the free tier is small);
* when every key is spent the call fails fast with ``LLM_RATE_LIMITED`` and callers fall back to their
  deterministic path. Nothing waits and nothing retries in the background;
* the budget is one process-wide object, whichever service created the client;
* key material is never logged and never leaves this module: statistics identify keys as "key 1", "key 2".
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

from google import genai

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

BENCH_ON_LIMIT_SECONDS = 60.0
BENCH_ON_OVERLOAD_SECONDS = 15.0


@dataclass
class _Key:
    secret: str
    label: str
    window: deque = field(default_factory=deque)  # timestamps of calls in the last minute
    day: date = field(default_factory=date.today)
    day_calls: int = 0
    benched_until: float = 0.0


class GeminiPool:
    def __init__(self, keys: list[str], per_minute: int, per_day: int, clock: Callable[[], float] = time.monotonic,
                 today: Callable[[], date] = date.today) -> None:
        unique = list(dict.fromkeys(k for k in keys if k))
        self._keys = [_Key(secret=k, label=f"key {i + 1}") for i, k in enumerate(unique)]
        self._per_minute, self._per_day = max(1, per_minute), max(1, per_day)
        self._clock, self._today = clock, today
        self._lock = threading.Lock()
        self._clients: dict[str, genai.Client] = {}

    @property
    def configured(self) -> bool:
        return bool(self._keys)

    def _roll(self, key: _Key) -> None:
        now = self._clock()
        while key.window and now - key.window[0] >= 60.0:
            key.window.popleft()
        if key.day != self._today():
            key.day, key.day_calls = self._today(), 0

    def reserve(self, exclude: frozenset[str] = frozenset()) -> Optional[_Key]:
        """Take one call from the key with the most headroom, or ``None`` if no key can afford one."""
        with self._lock:
            now = self._clock()
            best: Optional[_Key] = None
            for key in self._keys:
                if key.secret in exclude or key.benched_until > now:
                    continue
                self._roll(key)
                if len(key.window) >= self._per_minute or key.day_calls >= self._per_day:
                    continue
                if best is None or len(key.window) < len(best.window):
                    best = key
            if best is not None:
                best.window.append(now)
                best.day_calls += 1
            return best

    def bench(self, key: _Key, seconds: float) -> None:
        with self._lock:
            key.benched_until = self._clock() + seconds

    def client(self, key: _Key) -> genai.Client:
        with self._lock:
            if key.secret not in self._clients:
                self._clients[key.secret] = genai.Client(api_key=key.secret)
            return self._clients[key.secret]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            for key in self._keys:
                self._roll(key)
            return {
                "keys": len(self._keys),
                "calls_last_minute": sum(len(k.window) for k in self._keys),
                "calls_today": sum(k.day_calls for k in self._keys),
                "per_minute_limit": self._per_minute * len(self._keys),
                "per_day_limit": self._per_day * len(self._keys),
            }

    def headroom(self) -> int:
        """Calls that could be made right now across all keys (used to skip optional calls when nearly spent)."""
        with self._lock:
            now = self._clock()
            total = 0
            for key in self._keys:
                if key.benched_until > now:
                    continue
                self._roll(key)
                total += max(0, min(self._per_minute - len(key.window), self._per_day - key.day_calls))
            return total


class _Models:
    def __init__(self, pool: GeminiPool) -> None:
        self._pool = pool

    def generate_content(self, **kwargs: Any) -> Any:
        tried: set[str] = set()
        last: Optional[Exception] = None
        for _ in range(len(self._pool._keys)):
            key = self._pool.reserve(frozenset(tried))
            if key is None:
                break
            tried.add(key.secret)
            try:
                return self._pool.client(key).models.generate_content(**kwargs)
            except genai.errors.ClientError as exc:
                text = str(exc).lower()
                if "429" in text or "quota" in text or "rate" in text:
                    self._pool.bench(key, BENCH_ON_LIMIT_SECONDS)
                    logger.info("Gemini %s is rate limited; benched", key.label)
                    last = exc
                    continue
                raise
            except genai.errors.ServerError as exc:
                self._pool.bench(key, BENCH_ON_OVERLOAD_SECONDS)
                logger.info("Gemini %s is overloaded; trying the other key if any", key.label)
                last = exc
                continue
        if last is not None:
            raise last
        from app.services.llm_service import LLMServiceError  # late import: llm_service imports this module

        raise LLMServiceError("LLM_RATE_LIMITED", "DataPilot's AI request budget is used up for the moment.")


class PooledClient:
    """Looks like ``genai.Client`` to the callers (``client.models.generate_content``), but spends the shared budget."""

    def __init__(self, pool: GeminiPool) -> None:
        self.models = _Models(pool)
        self.pool = pool


_POOLS: dict[tuple, GeminiPool] = {}
_POOL_LOCK = threading.Lock()


def get_pool(settings: Settings) -> GeminiPool:
    """One pool per distinct (keys, limits): the budget is shared by every service in the process."""
    keys = tuple(k for k in (settings.gemini_api_key, settings.gemini_api_key_2) if k)
    signature = (keys, settings.gemini_requests_per_minute, settings.gemini_requests_per_day)
    with _POOL_LOCK:
        if signature not in _POOLS:
            _POOLS[signature] = GeminiPool(list(keys), settings.gemini_requests_per_minute, settings.gemini_requests_per_day)
        return _POOLS[signature]
