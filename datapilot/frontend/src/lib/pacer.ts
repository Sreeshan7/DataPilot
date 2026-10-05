/**
 * Shows real progress events at a readable pace. Every event is a fact the server reported; the pacer only makes
 * sure each one stays on screen for at least `gapMs`, so stages that finish in a few milliseconds are still seen.
 * Events are applied in order and none is dropped. `flush` resolves once everything queued has been shown.
 */
export function createPacer(gapMs: number, maxBacklogMs = 3000) {
  const queue: (() => void)[] = [];
  let timer: ReturnType<typeof setTimeout> | undefined;
  let waiting: (() => void)[] = [];

  const settle = () => { const done = waiting; waiting = []; done.forEach((r) => r()); };

  const tick = () => {
    timer = undefined;
    const next = queue.shift();
    if (!next) { settle(); return; }
    next();
    // A long backlog speeds up so the answer is never held back for more than about maxBacklogMs.
    const gap = queue.length ? Math.min(gapMs, maxBacklogMs / queue.length) : gapMs;
    timer = setTimeout(tick, gap); // the cooldown also spaces out events that arrive just after this one
  };

  return {
    push(apply: () => void) {
      if (gapMs <= 0) { apply(); return; }
      queue.push(apply);
      if (!timer) tick();
    },
    flush(): Promise<void> {
      if (!timer && queue.length === 0) return Promise.resolve();
      return new Promise((resolve) => waiting.push(resolve));
    },
    cancel() { queue.length = 0; if (timer) clearTimeout(timer); timer = undefined; settle(); },
  };
}
