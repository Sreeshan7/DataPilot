import type { AssistResponse, ResearchRequest, ResearchResponse } from "@/types";

export interface Attachment {
  kind: "file" | "link" | "name";
  label: string;
}

export interface UserTurn {
  id: string;
  role: "user";
  text: string;
  attachment?: Attachment;
  at: number;
}

export type Result =
  | { kind: "assist"; data: AssistResponse }
  | { kind: "research"; data: ResearchResponse };

/** What to run again on retry. A file cannot be persisted, so file retries only work within the session. */
export type RetryInfo =
  | { type: "assist"; message?: string; datasetUrl?: string; datasetName?: string; projectGoal?: string; fileName?: string }
  | { type: "research"; request: ResearchRequest };

export interface Understood {
  journey: "discovery" | "analysis" | "ambiguous";
  label: string;
  confidence: number;
}

/** A stage the server reported while working. `pending` stages were announced but have not started. */
export interface LiveStep { id: string; label: string; state: "pending" | "running" | "done" | "skipped"; detail?: string }

export interface AssistantTurn {
  id: string;
  role: "assistant";
  at: number;
  workflow: "assist" | "research";
  status: "pending" | "done" | "error";
  /** What the free, deterministic router said before the long call started (a real, completed step). */
  understood?: Understood;
  progress?: { phase: "uploading" | "processing"; fraction?: number; steps?: LiveStep[] };
  result?: Result;
  error?: { message: string; retryable: boolean };
  retry?: RetryInfo;
}

export type Turn = UserTurn | AssistantTurn;

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  turns: Turn[];
}
