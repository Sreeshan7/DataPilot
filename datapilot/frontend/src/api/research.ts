import type { ResearchRequest, ResearchResponse } from "@/types";
import { ApiError, postJson, type RequestOptions } from "./client";
import { postSse } from "./stream";

/**
 * POST /api/v1/research/discover. Only ever called after the user explicitly chooses "Research this
 * problem"; the backend makes a few searches and, if the AI options are on, a few small Gemini calls.
 */
export function discoverResearch(request: ResearchRequest, options?: RequestOptions): Promise<ResearchResponse> {
  return postJson<ResearchResponse>("/api/v1/research/discover", request, { timeoutMs: 90_000, ...options });
}

/** One live stage reported by the server while it works. */
export type ProgressEvent =
  | { type: "plan"; steps: { id: string; label: string }[] }
  | { type: "step"; id: string; state: "running" | "done" | "skipped"; detail?: string | null };

/**
 * POST /api/v1/research/stream: the same work and the same final answer as `discoverResearch`, with each stage
 * reported as it really happens. If the server is older and has no stream endpoint, it quietly falls back.
 */
export async function streamResearch(
  request: ResearchRequest, { signal, onProgress }: { signal?: AbortSignal; onProgress: (e: ProgressEvent) => void },
): Promise<ResearchResponse> {
  let result: ResearchResponse | undefined;
  let failure: string | undefined;
  try {
    await postSse("/api/v1/research/stream", request, {
      signal,
      onEvent: ({ event, data }) => {
        if (event === "plan" || event === "step") onProgress(data as ProgressEvent);
        else if (event === "result") result = data as ResearchResponse;
        else if (event === "error") failure = (data as { message?: string }).message ?? "Research couldn't be completed.";
      },
    });
  } catch (error) {
    if (error instanceof ApiError && error.kind === "http" && (error.status === 404 || error.status === 405)) {
      return discoverResearch(request, { signal });
    }
    throw error;
  }
  if (result) return result;
  throw new ApiError("http", failure ?? "The connection closed before the research finished. Please try again.", 500);
}
