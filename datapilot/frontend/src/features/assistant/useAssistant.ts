import { useCallback, useRef } from "react";
import { assist, resolveIntent } from "@/api/assist";
import { ApiError } from "@/api/client";
import { discoverResearch } from "@/api/research";
import type { ConversationStore } from "@/features/conversations/store";
import type { Attachment, Conversation, RetryInfo, Understood } from "@/features/conversations/types";
import { journeyLabel } from "@/lib/activity";
import { uid } from "@/lib/utils";
import type { IntentResolution, NextAction, ResearchRequest } from "@/types";

export interface Submission {
  text: string;
  file?: File;
  link?: string;
  datasetName?: string;
}

export type ActionOutcome = "started" | "needs_file" | "unsupported";

function understoodFrom(r: IntentResolution): Understood {
  const journey = r.intent === "USER_A_DATASET_DISCOVERY" ? "discovery" : r.intent === "USER_B_DATASET_ANALYSIS" ? "analysis" : "ambiguous";
  return { journey, label: journeyLabel(r.intent), confidence: r.confidence };
}

function attachmentOf(s: Submission): Attachment | undefined {
  if (s.file) return { kind: "file", label: s.file.name };
  if (s.link) return { kind: "link", label: s.link };
  if (s.datasetName) return { kind: "name", label: s.datasetName };
  return undefined;
}

/** The project the user described earlier in this conversation, so a dataset can later be judged against it. */
export function goalOf(conversation: Conversation | null): string | undefined {
  if (!conversation) return undefined;
  const turns = conversation.turns;
  for (let i = turns.length - 1; i > 0; i--) {
    const t = turns[i];
    const before = turns[i - 1];
    if (t.role === "assistant" && t.result?.kind === "assist" && t.result.data.journey === "USER_A_DATASET_DISCOVERY" && before.role === "user") {
      return before.text.slice(0, 1500);
    }
  }
  return undefined;
}

export function useAssistant(store: ConversationStore) {
  const files = useRef(new Map<string, File>());
  const aborts = useRef(new Map<string, AbortController>());
  const activeRef = useRef<Conversation | null>(null);
  activeRef.current = store.active;

  const finishWithError = useCallback((convId: string, turnId: string, error: unknown, retry: RetryInfo) => {
    const e = error instanceof ApiError ? error : new ApiError("network", "Something unexpected happened. Please try again.");
    store.patch(convId, turnId, {
      status: "error", progress: undefined, retry,
      error: { message: e.kind === "aborted" ? "Stopped. Nothing was changed." : e.message, retryable: e.kind === "aborted" ? true : e.retryable },
    });
  }, [store]);

  const runAssist = useCallback(async (convId: string, turnId: string, s: Submission, projectGoal: string | undefined) => {
    const controller = new AbortController();
    aborts.current.set(turnId, controller);
    const retry: RetryInfo = { type: "assist", message: s.text || undefined, datasetUrl: s.link, datasetName: s.datasetName, projectGoal, fileName: s.file?.name };
    try {
      if (!s.file) {
        // Free and deterministic: tells the user, truthfully, what was understood before the long call runs.
        try {
          const resolved = await resolveIntent({ message: s.text, datasetUrl: s.link, datasetName: s.datasetName }, { signal: controller.signal });
          store.patch(convId, turnId, { understood: understoodFrom(resolved) });
        } catch (e) {
          if (e instanceof ApiError && e.kind === "aborted") throw e;
        }
      } else {
        store.patch(convId, turnId, {
          understood: { journey: "analysis", label: "analysing a dataset", confidence: 1 },
          progress: { phase: "uploading", fraction: 0 },
        });
      }
      const data = await assist(
        { message: s.text || undefined, file: s.file, datasetUrl: s.link, datasetName: s.datasetName, projectGoal },
        {
          signal: controller.signal,
          onUploadProgress: (fraction) => store.patch(convId, turnId, { progress: { phase: "uploading", fraction } }),
          onUploadComplete: () => store.patch(convId, turnId, { progress: { phase: "processing" } }),
        },
      );
      store.patch(convId, turnId, { status: "done", progress: undefined, result: { kind: "assist", data }, retry });
    } catch (error) {
      finishWithError(convId, turnId, error, retry);
    } finally {
      aborts.current.delete(turnId);
    }
  }, [store, finishWithError]);

  const send = useCallback((s: Submission) => {
    const convId = store.ensure(activeRef.current?.id ?? null);
    const goal = goalOf(activeRef.current) ?? (s.file && s.text.trim().split(/\s+/).length >= 4 ? s.text.slice(0, 1500) : undefined);
    const turnId = uid("turn");
    const text = s.text.trim() || (s.file ? "Analyze this dataset" : s.link ? "Analyze this dataset" : s.datasetName ? `Analyze the ${s.datasetName} dataset` : "");
    store.append(convId, [
      { id: uid("turn"), role: "user", text, attachment: attachmentOf(s), at: Date.now() },
      { id: turnId, role: "assistant", workflow: "assist", status: "pending", at: Date.now() },
    ]);
    if (s.file) files.current.set(turnId, s.file);
    void runAssist(convId, turnId, { ...s, text }, goal);
  }, [store, runAssist]);

  const runResearch = useCallback(async (convId: string, turnId: string, request: ResearchRequest) => {
    const controller = new AbortController();
    aborts.current.set(turnId, controller);
    const retry: RetryInfo = { type: "research", request };
    try {
      const data = await discoverResearch(request, { signal: controller.signal });
      store.patch(convId, turnId, { status: "done", result: { kind: "research", data }, retry });
    } catch (error) {
      finishWithError(convId, turnId, error, retry);
    } finally {
      aborts.current.delete(turnId);
    }
  }, [store, finishWithError]);

  const startResearch = useCallback((payload: ResearchRequest, useAi: boolean) => {
    const convId = store.ensure(activeRef.current?.id ?? null);
    const turnId = uid("turn");
    const request: ResearchRequest = { ...payload, options: { ...(payload.options ?? {}), use_ai: useAi } as ResearchRequest["options"] };
    store.append(convId, [
      { id: uid("turn"), role: "user", text: "Research this problem", at: Date.now() },
      { id: turnId, role: "assistant", workflow: "research", status: "pending", at: Date.now() },
    ]);
    void runResearch(convId, turnId, request);
  }, [store, runResearch]);

  /** Maps a backend next action to a real call. Anything the UI cannot perform is reported, not faked. */
  const runAction = useCallback((action: NextAction, options?: { useAi?: boolean }): ActionOutcome => {
    const payload = (action.payload ?? {}) as Record<string, unknown>;
    switch (action.action) {
      case "RESEARCH_PROBLEM":
        if (action.available === false || !payload.project) return "unsupported";
        startResearch(payload as unknown as ResearchRequest, Boolean(options?.useAi));
        return "started";
      case "UPLOAD_FILE":
        return "needs_file";
      case "ANALYZE_ALTERNATIVE":
      case "ANALYZE_RECOMMENDED_DATASET":
        if (typeof payload.dataset_url !== "string") return "unsupported";
        send({ text: action.description, link: payload.dataset_url });
        return "started";
      case "FIND_ALTERNATIVE_DATASET":
        if (typeof payload.description !== "string") return "unsupported";
        send({ text: `Find me a dataset for this project: ${payload.description}` });
        return "started";
      default:
        return "unsupported";
    }
  }, [send, startResearch]);

  const retry = useCallback((convId: string, turnId: string) => {
    const conv = store.conversations.find((c) => c.id === convId);
    const turn = conv?.turns.find((t) => t.id === turnId);
    if (!turn || turn.role !== "assistant" || !turn.retry) return;
    store.patch(convId, turnId, { status: "pending", error: undefined, result: undefined, progress: undefined });
    if (turn.retry.type === "research") { void runResearch(convId, turnId, turn.retry.request); return; }
    const r = turn.retry;
    const file = files.current.get(turnId);
    if (r.fileName && !file) {
      store.patch(convId, turnId, { status: "error", error: { message: "Attach the file again to retry: DataPilot doesn't keep uploads.", retryable: false } });
      return;
    }
    void runAssist(convId, turnId, { text: r.message ?? "", file, link: r.datasetUrl, datasetName: r.datasetName }, r.projectGoal);
  }, [store, runAssist, runResearch]);

  const cancel = useCallback((turnId: string) => aborts.current.get(turnId)?.abort(), []);
  const fileFor = useCallback((turnId: string) => {
    const f = files.current.get(turnId);
    return f ? { name: f.name, size: f.size } : undefined;
  }, []);

  return { send, runAction, retry, cancel, fileFor };
}
