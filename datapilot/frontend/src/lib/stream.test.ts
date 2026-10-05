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

import { applyProgress } from "./live";

describe("applyProgress", () => {
  const plan = { type: "plan" as const, steps: [{ id: "a", label: "A" }, { id: "b", label: "B" }, { id: "c", label: "C" }] };
  it("plans waiting steps and updates them as the server reports", () => {
    let s = applyProgress([], plan);
    expect(s.map((x) => x.state)).toEqual(["pending", "pending", "pending"]);
    s = applyProgress(s, { type: "step", id: "a", state: "running" });
    s = applyProgress(s, { type: "step", id: "a", state: "done", detail: "5 rows" });
    expect(s[0]).toMatchObject({ id: "a", state: "done", detail: "5 rows" });
  });
  it("inserts a stage the plan could not know before the first waiting one", () => {
    let s = applyProgress([], plan);
    s = applyProgress(s, { type: "step", id: "a", state: "done" });
    s = applyProgress(s, { type: "step", id: "source-kaggle", state: "running", label: "Searching kaggle" });
    expect(s.map((x) => x.id)).toEqual(["a", "source-kaggle", "b", "c"]);
    expect(s[1].label).toBe("Searching kaggle");
  });
  it("keeps already-reported steps when the plan arrives late", () => {
    let s = applyProgress([], { type: "step", id: "route", state: "done", label: "Understanding your request" });
    s = applyProgress(s, plan);
    expect(s.map((x) => x.id)).toEqual(["route", "a", "b", "c"]);
  });
});
