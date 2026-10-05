import { useCallback, useEffect, useMemo, useReducer } from "react";
import { uid } from "@/lib/utils";
import type { AssistantTurn, Conversation, Turn } from "./types";

/**
 * Conversations live in this browser only (localStorage). Nothing is sent to or stored by the backend, which
 * has no persistence yet; the shape below is what a server-side history would later replace.
 */
const KEY = "datapilot.conversations.v1";
const MAX_CONVERSATIONS = 30;

interface State {
  conversations: Conversation[];
  activeId: string | null;
}

type Action =
  | { type: "new" }
  | { type: "select"; id: string }
  | { type: "delete"; id: string }
  | { type: "append"; id: string; turns: Turn[] }
  | { type: "patch"; id: string; turnId: string; patch: Partial<AssistantTurn> }
  | { type: "ensure"; id: string };

const blank = (id: string): Conversation => ({ id, title: "New conversation", createdAt: Date.now(), updatedAt: Date.now(), turns: [] });

export function titleFrom(text: string): string {
  const clean = text.replace(/\s+/g, " ").trim();
  return clean.length > 48 ? `${clean.slice(0, 47).trimEnd()}…` : clean || "New conversation";
}

export function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "new": {
      const existing = state.conversations.find((c) => c.turns.length === 0);
      if (existing) return { ...state, activeId: existing.id };
      const c = blank(uid("conv"));
      return { conversations: [c, ...state.conversations].slice(0, MAX_CONVERSATIONS), activeId: c.id };
    }
    case "ensure": {
      if (state.conversations.some((c) => c.id === action.id)) return state;
      return { conversations: [blank(action.id), ...state.conversations], activeId: action.id };
    }
    case "select":
      return state.conversations.some((c) => c.id === action.id) ? { ...state, activeId: action.id } : state;
    case "delete": {
      const rest = state.conversations.filter((c) => c.id !== action.id);
      return { conversations: rest, activeId: state.activeId === action.id ? rest[0]?.id ?? null : state.activeId };
    }
    case "append":
      return {
        ...state,
        conversations: state.conversations.map((c) => {
          if (c.id !== action.id) return c;
          const turns = [...c.turns, ...action.turns];
          const firstUser = turns.find((t) => t.role === "user");
          return { ...c, turns, updatedAt: Date.now(), title: c.turns.length === 0 && firstUser?.role === "user" ? titleFrom(firstUser.text) : c.title };
        }),
      };
    case "patch":
      return {
        ...state,
        conversations: state.conversations.map((c) =>
          c.id !== action.id ? c : {
            ...c, updatedAt: Date.now(),
            turns: c.turns.map((t) => (t.id === action.turnId && t.role === "assistant" ? { ...t, ...action.patch } : t)),
          }),
      };
  }
}

/** A turn that was still running when the page closed can never finish: say so instead of spinning forever. */
export function restore(raw: string | null): State {
  try {
    const parsed = JSON.parse(raw ?? "null") as State | null;
    if (!parsed || !Array.isArray(parsed.conversations)) return { conversations: [], activeId: null };
    const conversations = parsed.conversations.map((c) => ({
      ...c,
      turns: c.turns.map((t) =>
        t.role === "assistant" && t.status === "pending"
          ? { ...t, status: "error" as const, progress: undefined, error: { message: "This request was interrupted when the page was closed.", retryable: true } }
          : t),
    }));
    return { conversations, activeId: conversations.some((c) => c.id === parsed.activeId) ? parsed.activeId : conversations[0]?.id ?? null };
  } catch {
    return { conversations: [], activeId: null };
  }
}

function load(): State {
  try { return restore(localStorage.getItem(KEY)); } catch { return { conversations: [], activeId: null }; }
}

export function useConversationStore() {
  const [state, dispatch] = useReducer(reducer, undefined, load);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      try { localStorage.setItem(KEY, JSON.stringify(state)); }
      catch { /* storage full or blocked: the session still works, it just won't be remembered */ }
    }, 250);
    return () => window.clearTimeout(handle);
  }, [state]);

  const active = useMemo(() => state.conversations.find((c) => c.id === state.activeId) ?? null, [state]);

  const newConversation = useCallback(() => dispatch({ type: "new" }), []);
  const select = useCallback((id: string) => dispatch({ type: "select", id }), []);
  const remove = useCallback((id: string) => dispatch({ type: "delete", id }), []);
  const append = useCallback((id: string, turns: Turn[]) => dispatch({ type: "append", id, turns }), []);
  const patch = useCallback((id: string, turnId: string, p: Partial<AssistantTurn>) => dispatch({ type: "patch", id, turnId, patch: p }), []);
  /** Returns the id to write into, creating a conversation when none is open. */
  const ensure = useCallback((current: string | null): string => {
    if (current) return current;
    const id = uid("conv");
    dispatch({ type: "ensure", id });
    return id;
  }, []);

  return { conversations: state.conversations, active, newConversation, select, remove, append, patch, ensure };
}

export type ConversationStore = ReturnType<typeof useConversationStore>;
