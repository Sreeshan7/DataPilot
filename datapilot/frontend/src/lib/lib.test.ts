import { describe, expect, it } from "vitest";
import { describeHttpError } from "@/api/client";
import { traceFromAssist, traceFromResearch, EXPECTED_STEPS } from "@/lib/activity";
import { bytes, compactNumber, hostOf, humanize, percent, safeUrl } from "@/lib/format";
import { validateFile, validateLink } from "@/components/composer/Composer";
import { reducer, restore, titleFrom } from "@/features/conversations/store";
import { goalOf } from "@/features/assistant/useAssistant";
import type { AssistResponse, ResearchResponse } from "@/types";
import type { Conversation } from "@/features/conversations/types";

describe("error copy", () => {
  it("maps documented statuses to human messages", () => {
    expect(describeHttpError(413).message).toMatch(/larger than DataPilot can analyse/);
    expect(describeHttpError(415).message).toMatch(/\.csv, \.tsv and \.xlsx/);
    expect(describeHttpError(504).retryable).toBe(true);
    expect(describeHttpError(404).message).toMatch(/older version/);
    expect(describeHttpError(500).message).not.toMatch(/exception|traceback/i);
  });
  it("uses the backend's own message only for 422, where it is written to be shown", () => {
    expect(describeHttpError(422, "X", "Pick a .csv").message).toBe("Pick a .csv");
    expect(describeHttpError(502, "X", "Traceback in /srv/app.py").message).not.toMatch(/Traceback|srv/);
  });
});

describe("format helpers never invent values", () => {
  it("keeps unknown as a dash", () => {
    expect(percent(null)).toBe("—");
    expect(compactNumber(undefined)).toBe("—");
    expect(bytes(null)).toBe("—");
  });
  it("formats known values", () => {
    expect(percent(0.914, 1)).toBe("91.4%");
    expect(bytes(2048)).toBe("2.0 KB");
    expect(humanize("binary_classification")).toBe("Binary classification");
  });
  it("only ever links http(s)", () => {
    expect(safeUrl("https://arxiv.org/abs/1")).toBe("https://arxiv.org/abs/1");
    expect(safeUrl("javascript:alert(1)")).toBeNull();
    expect(safeUrl("data:text/html,x")).toBeNull();
    expect(safeUrl(null)).toBeNull();
    expect(hostOf("https://www.nature.com/x")).toBe("nature.com");
  });
});

describe("composer validation", () => {
  const file = (name: string, size = 10) => new File([new Uint8Array(size)], name);
  it("accepts csv/tsv/xlsx and rejects others", () => {
    expect(validateFile(file("a.csv"))).toBeNull();
    expect(validateFile(file("a.XLSX"))).toBeNull();
    expect(validateFile(file("a.exe"))).toMatch(/\.csv/);
  });
  it("rejects empty and oversized files", () => {
    expect(validateFile(file("a.csv", 0))).toMatch(/empty/);
    const big = new File([], "b.csv");
    Object.defineProperty(big, "size", { value: 26 * 1024 * 1024 });
    expect(validateFile(big)).toMatch(/limit/);
  });
  it("accepts only http(s) links", () => {
    expect(validateLink("https://www.kaggle.com/datasets/a/b")).toBeNull();
    expect(validateLink("ftp://x")).toMatch(/https/);
    expect(validateLink("not a link")).toMatch(/link/);
  });
});

const baseRouting = { intent: "USER_A_DATASET_DISCOVERY", confidence: 0.95, evidence: [], reason: "", required_next_input: null, clarification: null, follow_up_intent: null, dataset_reference: null };
function discovery(over: Partial<AssistResponse["discovery"]> = {}): AssistResponse {
  return {
    success: true, request_id: "r", api_version: "v1", status: "COMPLETED", journey: "USER_A_DATASET_DISCOVERY", routing: baseRouting,
    headline: "h", prompt_to_user: null, analysis: null, next_actions: [],
    discovery: {
      success: true, request_id: "r", api_version: "v1", search_query: "q", count: 3, ranking_weights: {}, final_recommendation: null, recommendations: [],
      project_requirements: { project_understanding: { domain: { value: "education", confidence: 1 }, ml_tasks: [{ task: "classification", confidence: 1 }] } },
      sources: [{ source: "kaggle", source_type: "verified_external", status: "ok", candidate_count: 20, duration_ms: 1, error_code: null, message: null },
                { source: "genai", source_type: "ai_suggested", status: "failed", candidate_count: 0, duration_ms: 1, error_code: "X", message: null }],
      meta: { raw_candidates: 24 }, ...over,
    },
  } as unknown as AssistResponse;
}

describe("activity trace is built only from the response", () => {
  it("reports real source counts and marks a failed source as skipped, not done", () => {
    const steps = traceFromAssist(discovery());
    const kaggle = steps.find((s) => s.id === "source-kaggle")!;
    const genai = steps.find((s) => s.id === "source-genai")!;
    expect(kaggle).toMatchObject({ status: "done", detail: "20 candidates" });
    expect(genai.status).toBe("skipped");
    expect(steps.find((s) => s.id === "rank")!.detail).toBe("24 found, 3 ranked");
    expect(steps.find((s) => s.id === "requirements")!.detail).toBe("domain: education · task: classification");
  });
  it("does not invent a recommendation step when none was made", () => {
    expect(traceFromAssist(discovery()).some((s) => s.id === "recommend")).toBe(false);
  });
  it("shows a clarification as a skipped step instead of progress", () => {
    const r = { ...discovery(), status: "NEEDS_CLARIFICATION", discovery: null, routing: { ...baseRouting, intent: "AMBIGUOUS" } } as unknown as AssistResponse;
    const steps = traceFromAssist(r);
    expect(steps.every((s) => s.status === "skipped")).toBe(true);
    expect(steps.map((s) => s.label).join(" ")).toMatch(/Asked a question instead of guessing/);
  });
  it("describes research from the summary counts, and AI only if it was actually used", () => {
    const base = { status: "success", queries: ["a"], methods: [{}, {}], gemini: { requested: false, status: "not_requested", note: null, calls_made: 0 },
      research_summary: { queries_used: 4, results_retrieved: 13, papers_found: 8, papers_selected: 7, results_rejected_as_non_academic: 5, search_provider: "web_search" } };
    const steps = traceFromResearch(base as unknown as ResearchResponse);
    expect(steps.find((s) => s.id === "search")!.detail).toBe("13 results retrieved");
    expect(steps.find((s) => s.id === "filter")!.detail).toMatch(/8 papers, 5 non-academic dropped/);
    expect(steps.some((s) => s.id === "ai")).toBe(false);
    const failed = traceFromResearch({ ...base, gemini: { requested: true, status: "failed", note: "quota", calls_made: 1 } } as unknown as ResearchResponse);
    expect(failed.find((s) => s.id === "ai")).toMatchObject({ status: "skipped", detail: "quota" });
  });
  it("lists expected steps only as an outline per workflow", () => {
    expect(EXPECTED_STEPS.discovery).toHaveLength(5);
    expect(EXPECTED_STEPS.research).toContain("Searching academic literature");
  });
});

describe("conversation store", () => {
  const empty = { conversations: [], activeId: null };
  it("creates one blank conversation and reuses it", () => {
    const a = reducer(empty, { type: "new" });
    const b = reducer(a, { type: "new" });
    expect(b.conversations).toHaveLength(1);
  });
  it("titles a conversation from its first message", () => {
    let s = reducer(empty, { type: "new" });
    const id = s.activeId!;
    s = reducer(s, { type: "append", id, turns: [{ id: "u", role: "user", text: "Predict churn for my telecom customers using a long description that goes on", at: 1 }] });
    expect(s.conversations[0].title.endsWith("…")).toBe(true);
    expect(titleFrom("   ")).toBe("New conversation");
  });
  it("turns a pending request left over from a closed page into a retryable error", () => {
    const state = { conversations: [{ id: "c", title: "t", createdAt: 1, updatedAt: 1, turns: [{ id: "a", role: "assistant", workflow: "assist", status: "pending", at: 1 }] }], activeId: "c" };
    const restored = restore(JSON.stringify(state));
    const turn = restored.conversations[0].turns[0] as { status: string; error: { retryable: boolean } };
    expect(turn.status).toBe("error");
    expect(turn.error.retryable).toBe(true);
  });
  it("survives corrupt storage", () => {
    expect(restore("{not json").conversations).toEqual([]);
    expect(restore(null).activeId).toBeNull();
  });
  it("deleting the active conversation selects another", () => {
    let s = reducer(empty, { type: "new" });
    s = reducer(s, { type: "append", id: s.activeId!, turns: [{ id: "u", role: "user", text: "a", at: 1 }] });
    s = reducer(s, { type: "new" });
    const [first, second] = s.conversations;
    const next = reducer(s, { type: "delete", id: first.id });
    expect(next.activeId).toBe(second.id);
  });
});

describe("goalOf", () => {
  it("carries the project description from an earlier discovery turn", () => {
    const conv: Conversation = {
      id: "c", title: "t", createdAt: 1, updatedAt: 1,
      turns: [
        { id: "u1", role: "user", text: "predict student performance", at: 1 },
        { id: "a1", role: "assistant", workflow: "assist", status: "done", at: 2, result: { kind: "assist", data: { journey: "USER_A_DATASET_DISCOVERY" } as AssistResponse } },
      ],
    };
    expect(goalOf(conv)).toBe("predict student performance");
    expect(goalOf(null)).toBeUndefined();
    expect(goalOf({ ...conv, turns: [] })).toBeUndefined();
  });
});
