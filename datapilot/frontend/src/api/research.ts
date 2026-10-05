import type { ResearchRequest, ResearchResponse } from "@/types";
import { postJson, type RequestOptions } from "./client";

/**
 * POST /api/v1/research/discover. Only ever called after the user explicitly chooses "Research this
 * problem"; the backend makes a few web searches and, if `use_ai` is true, one Gemini call.
 */
export function discoverResearch(request: ResearchRequest, options?: RequestOptions): Promise<ResearchResponse> {
  return postJson<ResearchResponse>("/api/v1/research/discover", request, { timeoutMs: 90_000, ...options });
}
