import { BookOpenText } from "lucide-react";
import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import type { NextAction } from "@/types";

/**
 * The explicit "Research this problem" step. Research never runs by itself: it costs web searches and, only if the
 * box below is ticked, one Gemini call. The tick is off by default and disabled when no key is configured.
 */
export function ResearchOffer({ action, onRun, aiAvailable, busy }: {
  action: NextAction; onRun: (useAi: boolean, question?: string) => void; aiAvailable: boolean; busy: boolean;
}) {
  const payload = (action.payload ?? {}) as { project?: { problem?: string | null; description?: string | null } };
  const suggested = payload.project?.problem ?? payload.project?.description ?? "";
  const fullPrompt = (payload.project?.description ?? "").trim();
  const [useAi, setUseAi] = useState(false);
  const [question, setQuestion] = useState(suggested);
  const id = useId();
  const qid = useId();
  const available = action.available !== false;
  return (
    <section className="animate-fade-up rounded-2xl border border-border bg-gradient-to-br from-accent-soft/50 via-surface to-surface p-5 sm:p-6" aria-label="Research this problem">
      <div className="flex items-start gap-4">
        <span className="grid size-10 shrink-0 place-items-center rounded-xl border border-accent/30 bg-accent-soft text-accent"><BookOpenText className="size-5" aria-hidden /></span>
        <div className="min-w-0 flex-1">
          <h3 className="text-[15px] font-semibold">{action.label ?? "Research this problem"}</h3>
          <p className="mt-1 text-[13.5px] leading-6 text-muted-foreground">
            {available
              ? "Find relevant papers and the methods they use, with research-backed algorithm recommendations. Every paper links to its source."
              : action.reason ?? "There isn't enough detail yet to research."}
          </p>
          {available && suggested && (
            <div className="mt-4">
              <label htmlFor={qid} className="text-xs font-medium text-subtle">What should the papers be about?</label>
              <input id={qid} value={question} maxLength={200} onChange={(e) => setQuestion(e.target.value)}
                className="mt-1.5 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none transition-colors focus:border-accent/60" />
              {fullPrompt && (
                <div className="mt-3 rounded-lg border border-border bg-raised/60 px-3.5 py-3" aria-label="Your project description">
                  <h4 className="text-[11px] font-medium uppercase tracking-wider text-subtle">Your project description</h4>
                  <p className="mt-1.5 whitespace-pre-wrap text-[13.5px] leading-6 text-muted-foreground [overflow-wrap:anywhere]">{fullPrompt}</p>
                </div>
              )}
            </div>
          )}
          {available && (
            <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-3">
              <Button variant="primary" disabled={busy} onClick={() => (question.trim() ? onRun(useAi && aiAvailable, question.trim()) : onRun(useAi && aiAvailable))}>Research this problem</Button>
              <label htmlFor={id} className={`flex items-center gap-2 text-[13px] ${aiAvailable ? "cursor-pointer text-muted-foreground" : "cursor-not-allowed text-subtle"}`}>
                <input id={id} type="checkbox" checked={useAi && aiAvailable} disabled={!aiAvailable} onChange={(e) => setUseAi(e.target.checked)} className="size-4 rounded border-border accent-[hsl(var(--accent))]" />
                Add an AI summary <span className="text-subtle">{aiAvailable ? "(one extra Gemini call)" : "(not configured)"}</span>
              </label>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
