import { Rich } from "@/components/chat/Rich";
import { AlertOctagon, Info } from "lucide-react";
import { Badge, type Tone } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import { humanize } from "@/lib/format";
import type { FeatureIntelligence } from "@/types";

const ROLE: Record<string, { tone: Tone; label: string }> = {
  RECOMMENDED_FEATURE: { tone: "success", label: "Recommended" },
  USE_WITH_CAUTION: { tone: "warning", label: "Use with caution" },
  POSSIBLE_LEAKAGE: { tone: "danger", label: "Possible leakage" },
  IDENTIFIER: { tone: "neutral", label: "Identifier" },
  LOW_SIGNAL: { tone: "neutral", label: "Low signal" },
  UNKNOWN: { tone: "neutral", label: "Not assessed" },
};

export function FeaturePanel({ fi }: { fi: FeatureIntelligence }) {
  if (!fi.available) {
    return (
      <div className="flex items-start gap-2.5 rounded-xl border border-border bg-surface/60 p-4 text-[13.5px] leading-6 text-muted-foreground">
        <Info className="mt-1 size-4 shrink-0 text-subtle" aria-hidden />
        <div>
          <p className="font-medium text-foreground">Feature importance isn't available</p>
          <p>{fi.unavailable_reason}</p>
        </div>
      </div>
    );
  }
  const leaks = fi.features.filter((f) => f.role === "POSSIBLE_LEAKAGE");
  const maxImp = Math.max(0.0001, ...fi.features.map((f) => f.permutation_importance ?? 0));
  const m = fi.method;
  return (
    <div className="space-y-3">
      {leaks.length > 0 && (
        <div role="alert" className="flex items-start gap-2.5 rounded-xl border border-danger/35 bg-danger/10 p-3.5 text-[13.5px] leading-6">
          <AlertOctagon className="mt-0.5 size-4 shrink-0 text-danger" aria-hidden />
          <div>
            <p className="font-medium text-danger">Possible leakage: {leaks.map((l) => l.column).join(", ")}</p>
            <p className="text-foreground/80">{leaks[0].reasons[0]}. These columns were kept out of the importance model so they can't hide the genuine signals. Confirm they're known before the outcome, and aren't derived from it.</p>
          </div>
        </div>
      )}
      {m && (
        <p className="text-[13px] leading-5 text-muted-foreground">
          {m.reliable
            ? <>A quick held-out model scored <strong className="font-medium text-foreground">{m.held_out_score}</strong> on {m.scoring.replace(/_/g, " ")} against a trivial baseline of {m.baseline_score}, so the ranking below is meaningful.</>
            : <>A quick held-out model did <strong className="font-medium text-foreground">not clearly beat</strong> a trivial predictor, so per-feature importance is withheld rather than shown as a ranking.</>}
          {" "}Measured on {m.rows_used.toLocaleString()}{m.sampled ? ` of ${m.rows_available.toLocaleString()}` : ""} rows.
        </p>
      )}
      <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
        {fi.features.map((f) => {
          const role = ROLE[f.role] ?? ROLE.UNKNOWN;
          const imp = f.permutation_importance;
          return (
            <li key={f.column} className="px-3.5 py-3">
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
                <span className="min-w-0 flex-1 truncate font-mono text-[13px]" title={f.column}>{f.column}</span>
                <Badge tone={role.tone}>{role.label}</Badge>
              </div>
              {imp != null && imp > 0 ? (
                <div className="mt-2 flex items-center gap-2.5" aria-label={`Importance ${imp.toFixed(3)}`}>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted"><span className="block h-full rounded-full bg-accent" style={{ width: `${Math.max(3, (imp / maxImp) * 100)}%` }} /></span>
                  <span className="w-12 text-right text-xs tabular-nums text-subtle">{imp.toFixed(3)}</span>
                </div>
              ) : null}
              {(f.reasons.length > 0 || f.cautions.length > 0) && (
                <Disclosure label="Why" className="mt-1.5" meta={f.association != null ? `${humanize(f.association_method ?? "association")} ${f.association.toFixed(2)}` : undefined}>
                  <ul className="list-disc space-y-1 pl-5 text-[13px] leading-5 text-muted-foreground marker:text-subtle">
                    {f.reasons.map((r, i) => <li key={`r${i}`}><Rich>{r}</Rich></li>)}
                    {f.cautions.map((c, i) => <li key={`c${i}`} className="text-warning"><Rich>{c}</Rich></li>)}
                  </ul>
                </Disclosure>
              )}
            </li>
          );
        })}
      </ul>
      <p className="text-xs leading-5 text-subtle">{fi.caveat}</p>
    </div>
  );
}
