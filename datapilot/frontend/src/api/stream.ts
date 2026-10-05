import { ApiError, toApiError } from "./client";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export interface SseEvent { event: string; data: unknown }

/** Splits an SSE text buffer into complete events; returns the unconsumed tail. Comment lines (": ...") are ignored. */
export function parseSse(buffer: string): { events: SseEvent[]; rest: string } {
  const events: SseEvent[] = [];
  const blocks = buffer.split(/\r?\n\r?\n/);
  const rest = blocks.pop() ?? "";
  for (const block of blocks) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith(":")) continue;
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length === 0) continue;
    try { events.push({ event, data: JSON.parse(data.join("\n")) }); } catch { /* a malformed event is skipped */ }
  }
  return { events, rest };
}

/**
 * POST a JSON body and read a text/event-stream response, calling `onEvent` as each event arrives.
 * `idleTimeoutMs` is the longest the stream may stay silent (the server sends keep-alives).
 */
export async function postSse(
  path: string, body: unknown,
  { signal, onEvent, idleTimeoutMs = 60_000 }: { signal?: AbortSignal; onEvent: (e: SseEvent) => void; idleTimeoutMs?: number },
): Promise<void> {
  const controller = new AbortController();
  if (signal) {
    if (signal.aborted) controller.abort();
    signal.addEventListener("abort", () => controller.abort(), { once: true });
  }
  let idle = false;
  let timer = setTimeout(() => { idle = true; controller.abort(); }, idleTimeoutMs);
  const bump = () => { clearTimeout(timer); timer = setTimeout(() => { idle = true; controller.abort(); }, idleTimeoutMs); };
  try {
    const response = await fetch(`${BASE}${path}`, {
      method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body), signal: controller.signal,
    });
    if (!response.ok) throw toApiError(response.status, await response.json().catch(() => undefined));
    if (!response.body) throw new ApiError("network", "DataPilot can't be reached right now. Check that the backend is running, then try again.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      bump();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parsed = parseSse(buffer);
      buffer = parsed.rest;
      parsed.events.forEach(onEvent);
    }
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (idle) throw new ApiError("timeout", "That took too long to answer. Please try again.");
    if (controller.signal.aborted) throw new ApiError("aborted", "Cancelled.", undefined, undefined, false);
    throw new ApiError("network", "DataPilot can't be reached right now. Check that the backend is running, then try again.");
  } finally {
    clearTimeout(timer);
  }
}
