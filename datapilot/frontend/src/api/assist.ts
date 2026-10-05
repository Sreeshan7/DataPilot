import type { AssistResponse, IntentResolution } from "@/types";
import { postForm, postJson, type RequestOptions } from "./client";

export interface AssistInput {
  message?: string;
  file?: File;
  datasetUrl?: string;
  datasetName?: string;
  /** What the user is building, carried across turns so a dataset can be judged against it. */
  projectGoal?: string;
  useAi?: boolean;
}

/** POST /api/v1/assist: the one entry point; the backend decides which journey to run. */
export function assist(input: AssistInput, options?: RequestOptions): Promise<AssistResponse> {
  const form = new FormData();
  if (input.message) form.set("message", input.message);
  if (input.file) form.set("file", input.file, input.file.name);
  if (input.datasetUrl) form.set("dataset_url", input.datasetUrl);
  if (input.datasetName) form.set("dataset_name", input.datasetName);
  if (input.projectGoal) form.set("project_goal", input.projectGoal);
  if (input.useAi) form.set("use_ai", "true");
  return postForm<AssistResponse>("/api/v1/assist", form, options);
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
