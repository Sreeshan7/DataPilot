import type { LiveStep } from "@/features/conversations/types";

/** A progress event as the server sends it (see POST /assist/stream and /research/stream). */
export type ProgressEvent =
  | { type: "plan"; steps: { id: string; label: string }[] }
  | { type: "step"; id: string; state: "running" | "done" | "skipped"; label?: string; detail?: string | null };

/**
 * Folds one server event into the list of steps. Pure: the same events always give the same list.
 * - `plan` announces the stages that will run (shown as waiting);
 * - `step` updates a known stage, or, for a stage the plan could not know (e.g. each source searched), inserts it
 *   before the first stage that is still waiting, so the list reads in the order things happened.
 */
export function applyProgress(steps: LiveStep[], event: ProgressEvent): LiveStep[] {
  if (event.type === "plan") {
    const known = new Map(steps.map((s) => [s.id, s]));
    const planned = event.steps.map((s) => known.get(s.id) ?? { ...s, state: "pending" as const });
    const started = steps.filter((s) => !event.steps.some((p) => p.id === s.id));
    return [...started, ...planned];
  }
  const detail = event.detail ?? undefined;
  if (steps.some((s) => s.id === event.id)) {
    return steps.map((s) => (s.id === event.id ? { ...s, state: event.state, detail, label: event.label ?? s.label } : s));
  }
  const fresh: LiveStep = { id: event.id, label: event.label ?? event.id, state: event.state, detail };
  const firstWaiting = steps.findIndex((s) => s.state === "pending");
  return firstWaiting < 0 ? [...steps, fresh] : [...steps.slice(0, firstWaiting), fresh, ...steps.slice(firstWaiting)];
}
