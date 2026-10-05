import type { AssistResponse, IntentResolution } from "@/types";
import { ApiError, postForm, postJson, toApiError, type RequestOptions } from "./client";
import type { ProgressEvent } from "@/lib/live";
import { postFormSse } from "./stream";

export interface AssistInput {
  message?: string;
  file?: File;
  datasetUrl?: string;
  datasetName?: string;
  /** What the user is building, carried across turns so a dataset can be judged against it. */
  projectGoal?: string;
  useAi?: boolean;
}

function formOf(input: AssistInput): FormData {
  const form = new FormData();
  if (input.message) form.set("message", input.message);
  if (input.file) form.set("file", input.file, input.file.name);
  if (input.datasetUrl) form.set("dataset_url", input.datasetUrl);
  if (input.datasetName) form.set("dataset_name", input.datasetName);
  if (input.projectGoal) form.set("project_goal", input.projectGoal);
  if (input.useAi) form.set("use_ai", "true");
  return form;
}

/** POST /api/v1/assist: the one entry point; the backend decides which journey to run. */
export function assist(input: AssistInput, options?: RequestOptions): Promise<AssistResponse> {
  return postForm<AssistResponse>("/api/v1/assist", formOf(input), options);
}

/**
 * POST /api/v1/assist/stream: the same work and the same final answer as `assist`, with each stage reported as it
 * really happens (and real upload progress). Against an older server without the stream endpoint it quietly falls back.
 */
export async function streamAssist(
  input: AssistInput,
  { signal, onProgress, onUploadProgress, onUploadComplete }: {
    signal?: AbortSignal; onProgress: (e: ProgressEvent) => void;
    onUploadProgress?: (fraction: number) => void; onUploadComplete?: () => void;
  },
): Promise<AssistResponse> {
  let result: AssistResponse | undefined;
  let failure: ApiError | undefined;
  try {
    await postFormSse("/api/v1/assist/stream", formOf(input), {
      signal, onUploadProgress, onUploadComplete,
      onEvent: ({ event, data }) => {
        if (event === "plan" || event === "step") onProgress(data as ProgressEvent);
        else if (event === "result") result = data as AssistResponse;
        else if (event === "error") {
          const e = data as { status?: number; detail?: unknown };
          failure = toApiError(e.status ?? 500, { detail: e.detail });
        }
      },
    });
  } catch (error) {
    if (error instanceof ApiError && error.kind === "http" && (error.status === 404 || error.status === 405)) {
      return assist(input, { signal, onUploadProgress, onUploadComplete });
    }
    throw error;
  }
  if (failure) throw failure;
  if (result) return result;
  throw new ApiError("http", "The connection closed before DataPilot finished. Please try again.", 500);
}

/** POST /api/v1/intent/resolve: the free, deterministic router. Used to tell the user, truthfully, what was understood. */
export function resolveIntent(
  input: { message?: string; hasFile?: boolean; datasetUrl?: string; datasetName?: string },
  options?: RequestOptions,
): Promise<IntentResolution> {
  return postJson<IntentResolution>(
    "/api/v1/intent/resolve",
    { message: input.message, has_file: Boolean(input.hasFile), dataset_url: input.datasetUrl, dataset_name: input.datasetName },
    { timeoutMs: 8_000, ...options },
  );
}
