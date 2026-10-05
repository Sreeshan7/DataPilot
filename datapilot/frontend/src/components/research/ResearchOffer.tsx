import { BookOpenText } from "lucide-react";
import { useId, useState } from "react";
import { Button } from "@/components/ui/button";
import type { NextAction } from "@/types";

/**
 * The explicit "Research this problem" step. Research never runs by itself: it costs web searches and, only if the
 * box below is ticked, one Gemini call. The tick is off by default and disabled when no key is configured.
 */
export function ResearchOffer({ action, onRun, aiAvailable, busy }: {
  action: NextAction; onRun: (useAi: boolean) => void; aiAvailable: boolean; busy: boolean;
}) {
  const [useAi, setUseAi] = useState(false);
  const id = useId();
  const available = action.available !== false;
  return (
    <section className="animate-fade-up rounded-2xl border border-border bg-gradient-to-br from-accent-soft/50 via-surface to-surface p-4 sm:p-5" aria-label="Research this problem">
      <div className="flex items-start gap-3.5">
        <span className="grid size-10 shrink-0 place-items-center rounded-xl border border-accent/30 bg-accent-soft text-accent"><BookOpenText className="size-5" aria-hidden /></span>
        <div className="min-w-0 flex-1">
          <h3 className="text-[15px] font-semibold">{action.label ?? "Research this problem"}</h3>
          <p className="mt-1 text-[13.5px] leading-6 text-muted-foreground">
            {available
              ? "Find relevant papers and the methods they use, with research-backed algorithm recommendations. Every paper links to its source."
              : action.reason ?? "There isn't enough detail yet to research."}
          </p>
          {available && (
            <div className="mt-3.5 flex flex-wrap items-center gap-x-4 gap-y-2.5">
              <Button variant="primary" disabled={busy} onClick={() => onRun(useAi && aiAvailable)}>Research this problem</Button>
              <label htmlFor={id} className={`flex items-center gap-2 text-[13px] ${aiAvailable ? "cursor-pointer text-muted-foreground" : "cursor-not-allowed text-subtle"}`}>
                <input id={id} type="checkbox" checked={useAi && aiAvailable} disabled={!aiAvailable} onChange={(e) => setUseAi(e.target.checked)} className="size-4 rounded border-border accent-[hsl(var(--accent))]" />
                Add an AI summary <span className="text-subtle">{aiAvailable ? "(uses one Gemini call)" : "(not configured)"}</span>
              </label>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
