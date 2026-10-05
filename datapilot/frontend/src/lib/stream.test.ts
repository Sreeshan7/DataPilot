import { afterEach, describe, expect, it, vi } from "vitest";
import { parseSse } from "@/api/stream";
import { createPacer } from "./pacer";

describe("parseSse", () => {
  it("splits complete events, ignores keep-alives and keeps a partial tail", () => {
    const text = 'event: plan\ndata: {"steps":[]}\n\n: keep-alive\n\nevent: step\ndata: {"id":"a","state":"done"}\n\nevent: res';
    const { events, rest } = parseSse(text);
    expect(events).toEqual([{ event: "plan", data: { steps: [] } }, { event: "step", data: { id: "a", state: "done" } }]);
    expect(rest).toBe("event: res");
  });
  it("skips a malformed event without failing the stream", () => {
    const { events } = parseSse('event: step\ndata: {oops\n\nevent: result\ndata: {"status":"success"}\n\n');
    expect(events).toEqual([{ event: "result", data: { status: "success" } }]);
  });
});

describe("createPacer", () => {
  afterEach(() => vi.useRealTimers());
  it("shows every event in order, spaced out, and flush waits for the last one", async () => {
    vi.useFakeTimers();
    const pacer = createPacer(100);
    const seen: number[] = [];
    [1, 2, 3].forEach((n) => pacer.push(() => seen.push(n)));
    expect(seen).toEqual([1]);
    let flushed = false;
    void pacer.flush().then(() => { flushed = true; });
    await vi.advanceTimersByTimeAsync(100);
    expect(seen).toEqual([1, 2]);
    await vi.advanceTimersByTimeAsync(300);
    expect(seen).toEqual([1, 2, 3]);
    expect(flushed).toBe(true);
  });
  it("applies immediately when pacing is off (reduced motion)", () => {
    const pacer = createPacer(0);
    const seen: number[] = [];
    pacer.push(() => seen.push(1)); pacer.push(() => seen.push(2));
    expect(seen).toEqual([1, 2]);
  });
});
