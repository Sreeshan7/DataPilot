import type { ReadyResponse } from "@/types";
import { getJson, type RequestOptions } from "./client";

/** GET /ready: whether the backend is up and which optional capabilities are configured. */
export function getReadiness(options?: RequestOptions): Promise<ReadyResponse> {
  return getJson<ReadyResponse>("/ready", { timeoutMs: 6_000, ...options });
}
