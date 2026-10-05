/**
 * The one place that talks HTTP. Components never call fetch.
 *
 * - typed `ApiError` with a human message: stack traces, raw bodies and internals are never surfaced;
 * - cancellation through AbortSignal and a per-request timeout;
 * - multipart uploads go through XMLHttpRequest so upload progress is *real* (fetch cannot report it).
 */

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export type ApiErrorKind = "network" | "timeout" | "aborted" | "http";

export class ApiError extends Error {
  constructor(
    public readonly kind: ApiErrorKind,
    message: string,
    public readonly status?: number,
    public readonly code?: string,
    public readonly retryable = true,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
  onUploadProgress?: (fraction: number) => void;
  onUploadComplete?: () => void;
}

/** Messages for the statuses the backend documents. The backend's own error text is used only when it is safe. */
export function describeHttpError(status: number, _code?: string, backendMessage?: string): { message: string; retryable: boolean } {
  switch (status) {
    case 413: return { message: "That file is larger than DataPilot can analyse. Try a smaller file or a sample of it.", retryable: false };
    case 415: return { message: "DataPilot can read .csv, .tsv and .xlsx files. This file type isn't supported.", retryable: false };
    case 422: return { message: backendMessage || "DataPilot couldn't use that input. Check it and try again.", retryable: false };
    case 404:
    case 405:
      return { message: "DataPilot's server doesn't offer this feature. It may be running an older version; update it and try again.", retryable: false };
    case 429: return { message: "A data source is busy right now. Give it a moment and try again.", retryable: true };
    case 502: return { message: "A dataset source is temporarily unavailable. Please try again shortly.", retryable: true };
    case 504: return { message: "That took longer than allowed and was stopped. Try again, or use a smaller file.", retryable: true };
    default:
      return status >= 500
        ? { message: "Something went wrong on DataPilot's side. Please try again.", retryable: true }
        : { message: backendMessage || "The request couldn't be completed.", retryable: false };
  }
}

function backendMessage(body: unknown): { code?: string; message?: string } {
  const detail = (body as { detail?: unknown })?.detail;
  const error = (detail as { error?: { code?: string; message?: string } } | undefined)?.error;
  return { code: error?.code, message: typeof error?.message === "string" ? error.message : undefined };
}

export function toApiError(status: number, body: unknown): ApiError {
  const { code, message } = backendMessage(body);
  const described = describeHttpError(status, code, message);
  return new ApiError("http", described.message, status, code, described.retryable);
}

function link(signal: AbortSignal | undefined, controller: AbortController): () => void {
  if (!signal) return () => undefined;
  if (signal.aborted) controller.abort();
  const onAbort = () => controller.abort();
  signal.addEventListener("abort", onAbort, { once: true });
  return () => signal.removeEventListener("abort", onAbort);
}

export async function getJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  return sendJson<T>("GET", path, undefined, options);
}

export async function postJson<T>(path: string, body: unknown, options: RequestOptions = {}): Promise<T> {
  return sendJson<T>("POST", path, body, options);
}

async function sendJson<T>(method: string, path: string, body: unknown, { signal, timeoutMs = 60_000 }: RequestOptions): Promise<T> {
  const controller = new AbortController();
  const unlink = link(signal, controller);
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(`${BASE}${path}`, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: controller.signal,
    });
    const data: unknown = await response.json().catch(() => undefined);
    if (!response.ok) throw toApiError(response.status, data);
    return data as T;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (timedOut) throw new ApiError("timeout", "That took too long to answer. Please try again.");
    if (controller.signal.aborted) throw new ApiError("aborted", "Cancelled.", undefined, undefined, false);
    throw new ApiError("network", "DataPilot can't be reached right now. Check that the backend is running, then try again.");
  } finally {
    clearTimeout(timer);
    unlink();
  }
}

export function postForm<T>(path: string, form: FormData, { signal, timeoutMs = 120_000, onUploadProgress, onUploadComplete }: RequestOptions = {}): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}${path}`);
    xhr.timeout = timeoutMs;
    xhr.responseType = "text";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onUploadProgress?.(event.loaded / event.total);
    };
    xhr.upload.onload = () => onUploadComplete?.();
    xhr.onload = () => {
      let data: unknown;
      try { data = JSON.parse(xhr.responseText); } catch { data = undefined; }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data as T);
      else reject(toApiError(xhr.status, data));
    };
    xhr.onerror = () => reject(new ApiError("network", "DataPilot can't be reached right now. Check that the backend is running, then try again."));
    xhr.ontimeout = () => reject(new ApiError("timeout", "That took too long to answer. Please try again."));
    xhr.onabort = () => reject(new ApiError("aborted", "Cancelled.", undefined, undefined, false));
    if (signal) {
      if (signal.aborted) { xhr.abort(); return; }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(form);
  });
}
