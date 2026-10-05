import { Check, Circle, Minus } from "lucide-react";
import { useEffect, useState } from "react";
import { Disclosure } from "@/components/ui/disclosure";
import { Skeleton } from "@/components/ui/skeleton";
import { EXPECTED_STEPS, WORKING_TITLE, type TraceStep, type Workflow } from "@/lib/activity";
import { bytes } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { LiveStep, Understood } from "@/features/conversations/types";

interface PendingProps {
  workflow: Workflow;
  understood?: Understood;
  progress?: { phase: "uploading" | "processing"; fraction?: number; steps?: LiveStep[] };
  fileName?: string;
  fileSize?: number;
  onCancel?: () => void;
}

function useElapsed(): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const start = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, []);
  return seconds;
}

function StepIcon({ state }: { state: LiveStep["state"] }) {
  if (state === "done") return <Check className="mt-0.5 size-4 shrink-0 animate-check-pop text-success" aria-label="done" />;
  if (state === "skipped") return <Minus className="mt-0.5 size-4 shrink-0 animate-fade-in text-subtle" aria-label="skipped" />;
  if (state === "running") {
    return (
      <span className="mt-0.5 grid size-4 shrink-0 place-items-center" aria-label="working">
        <span className="size-3.5 animate-spin-ring rounded-full border-2 border-accent/25 border-t-accent motion-reduce:animate-none" />
      </span>
    );
  }
  return <Circle className="mt-0.5 size-4 shrink-0 text-border" aria-label="waiting" />;
}

/** The stages the server reported, live. Each tick is a stage that really finished; nothing here is scripted. */
function LiveSteps({ steps }: { steps: LiveStep[] }) {
  const finished = steps.filter((s) => s.state === "done" || s.state === "skipped").length;
  const pct = steps.length ? Math.round((finished / steps.length) * 100) : 0;
  return (
    <>
      <div className="mt-3 h-0.5 overflow-hidden rounded-full bg-muted" aria-hidden>
        <div className="relative h-full rounded-full bg-accent/70 transition-[width] duration-500 ease-out" style={{ width: `${Math.max(pct, 6)}%` }}>
          <span className="absolute inset-y-0 w-1/3 animate-sweep bg-gradient-to-r from-transparent via-white/40 to-transparent motion-reduce:hidden" />
        </div>
      </div>
      <ol className="mt-4 space-y-2.5 text-[13.5px]">
        {steps.map((s) => (
          <li key={s.id} className={cn("flex animate-step-in items-start gap-3 transition-opacity duration-300", s.state === "pending" && "opacity-45")}>
            <StepIcon state={s.state} />
            <span className="min-w-0 flex-1">
              <span className={cn(s.state === "running" ? "text-working font-medium" : s.state === "pending" ? "text-muted-foreground" : s.state === "skipped" ? "text-muted-foreground" : "text-foreground")}>
                {s.label}{s.state === "running" ? "…" : ""}
              </span>
              {s.detail && s.state !== "pending" ? <span className="animate-fade-in text-subtle"> · {s.detail}</span> : null}
            </span>
          </li>
        ))}
      </ol>
    </>
  );
}

/**
 * Shown while a request is open. It is deliberately honest: the backend returns one final answer, so there is no
 * live per-stage feed. What is *known* is ticked (the request was understood, bytes were uploaded); what the server
 * is working through is listed as an outline with no ticks. The ticked record of what really happened appears
 * with the result (see ActivityTrace).
 */
export function ActivityPending({ workflow, understood, progress, fileName, fileSize, onCancel }: PendingProps) {
  const uploading = progress?.phase === "uploading";
  const live = progress?.steps && progress.steps.length > 0 ? progress.steps : undefined;
  const seconds = useElapsed();
  const pct = Math.round((progress?.fraction ?? 0) * 100);
  const title = uploading ? `Uploading ${fileName ?? "your file"}` : WORKING_TITLE[workflow];
  return (
    <section aria-label="DataPilot is working" className="animate-fade-up rounded-xl border border-border bg-surface/60 p-5">
      <div className="flex items-center justify-between gap-3">
        <div role="status" aria-live="polite" className="flex min-w-0 items-center gap-2.5">
          <span className="relative grid size-4 place-items-center" aria-hidden>
            <span className="absolute size-4 animate-ping rounded-full bg-accent/30 motion-reduce:hidden" />
            <span className="size-2 rounded-full bg-accent" />
          </span>
          <span className="truncate text-sm font-medium">{title}…</span>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {live && seconds >= 2 && <span className="text-xs tabular-nums text-subtle" aria-hidden>{seconds}s</span>}
          {onCancel && <button type="button" onClick={onCancel} className="rounded-md px-2 py-1 text-xs text-subtle transition-colors hover:bg-muted hover:text-foreground">Stop</button>}
        </div>
      </div>

      {live && <LiveSteps steps={live} />}

      <ul className={cn("mt-3 space-y-2 text-[13.5px]", live && "hidden")}>
        {understood && (
          <li className="flex items-start gap-2.5 text-foreground">
            <Check className="mt-0.5 size-4 shrink-0 text-success" aria-hidden />
            <span>Understood your request as <strong className="font-medium">{understood.label}</strong></span>
          </li>
        )}
        {fileName && (
          <li className={cn("flex items-start gap-2.5", uploading ? "text-foreground" : "text-foreground")}>
            {uploading ? <Circle className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden /> : <Check className="mt-0.5 size-4 shrink-0 text-success" aria-hidden />}
            <span className="min-w-0 flex-1">
              {uploading ? "Uploading" : "Uploaded"} <span className="font-medium">{fileName}</span>{fileSize ? <span className="text-subtle"> · {bytes(fileSize)}</span> : null}
              {uploading && (
                <span className="mt-1.5 block h-1 overflow-hidden rounded-full bg-muted" role="progressbar" aria-label="Upload progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
                  <span className="block h-full rounded-full bg-accent transition-[width] duration-150" style={{ width: `${pct}%` }} />
                </span>
              )}
            </span>
          </li>
        )}
        {!uploading && EXPECTED_STEPS[workflow].map((label, i) => (
          <li key={label} className="flex items-start gap-2.5 text-muted-foreground" style={{ animationDelay: `${i * 40}ms` }}>
            <Circle className="mt-0.5 size-4 shrink-0 text-border" aria-hidden />
            <span>{label}</span>
          </li>
        ))}
      </ul>
      {!uploading && (
        <p className="mt-4 text-xs leading-5 text-subtle">
          {live
            ? "Live from DataPilot: each tick is a stage that has really finished."
            : "DataPilot returns one result when it's done; the steps above are what it works through, and you'll see exactly what it did."}
        </p>
      )}
    </section>
  );
}

/** What really happened, from the response's own fields. Collapsed by default: the answer comes first. */
export function ActivityTrace({ steps }: { steps: TraceStep[] }) {
  if (steps.length === 0) return null;
  const done = steps.filter((s) => s.status === "done").length;
  return (
    <Disclosure label="What DataPilot did" meta={`${done} of ${steps.length} steps`} className="rounded-xl border border-border bg-surface/40 px-3.5 py-2.5">
      <ol className="space-y-2 pb-1 text-[13.5px]">
        {steps.map((s) => (
          <li key={s.id} className="flex items-start gap-2.5">
            {s.status === "done"
              ? <Check className="mt-0.5 size-4 shrink-0 text-success" aria-label="done" />
              : <Minus className="mt-0.5 size-4 shrink-0 text-subtle" aria-label="skipped or unavailable" />}
            <span className={cn("min-w-0", s.status === "skipped" && "text-muted-foreground")}>
              {s.label}
              {s.detail ? <span className="text-subtle"> · {s.detail}</span> : null}
            </span>
          </li>
        ))}
      </ol>
    </Disclosure>
  );
}

export function ResultSkeleton({ workflow }: { workflow: Workflow }) {
  if (workflow === "analysis") {
    return (
      <div aria-hidden className="mt-4 space-y-3">
        <Skeleton className="h-5 w-2/5" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-16 rounded-xl" />)}</div>
        <Skeleton className="h-28 rounded-xl" />
      </div>
    );
  }
  return (
    <div aria-hidden className="mt-4 space-y-3">
      <Skeleton className="h-5 w-1/3" />
      {[0, 1, 2].map((i) => <Skeleton key={i} className="h-28 rounded-xl" />)}
    </div>
  );
}
