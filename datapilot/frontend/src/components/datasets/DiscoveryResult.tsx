import { AlertTriangle, CheckCircle2, CircleHelp } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { plural } from "@/lib/format";
import type { DatasetRecommendationResponse } from "@/types";
import { DatasetCard } from "./DatasetCard";

const INITIAL = 4;

const OUTCOME = {
  recommended: { icon: CheckCircle2, tone: "text-success", title: "Best match" },
  tentative: { icon: CircleHelp, tone: "text-warning", title: "A tentative match" },
  none: { icon: AlertTriangle, tone: "text-danger", title: "No dependable match" },
} as const;

export function DiscoveryResult({ data, onAnalyze, busy }: { data: DatasetRecommendationResponse; onAnalyze: (url: string, name: string) => void; busy: boolean }) {
  const [all, setAll] = useState(false);
  const final = data.final_recommendation;
  const shown = all ? data.recommendations : data.recommendations.slice(0, INITIAL);
  const down = data.sources.filter((s) => s.status === "failed");
  const outcome = final ? OUTCOME[final.outcome] : null;
  const Icon = outcome?.icon;

  return (
    <div className="space-y-4">
      {final && outcome && Icon && (
        <section className="rounded-2xl border border-border bg-gradient-to-b from-raised to-surface p-4 sm:p-5" aria-label="Recommendation summary">
          <div className="flex items-start gap-3">
            <Icon className={`mt-0.5 size-5 shrink-0 ${outcome.tone}`} aria-hidden />
            <div className="min-w-0">
              <h3 className="text-[15px] font-semibold">{outcome.title}{final.best_match ? `: ${final.best_match.name}` : ""}</h3>
              <p className="mt-1 text-[14px] leading-6 text-muted-foreground">{final.summary}</p>
              {final.close_call && <Badge tone="warning" className="mt-2">Close call: the top results are within a few points</Badge>}
              {final.caveats.length > 0 && (
                <ul className="mt-2.5 list-disc space-y-1 pl-4 text-[13px] leading-5 text-muted-foreground marker:text-subtle">
                  {final.caveats.slice(0, 3).map((c, i) => <li key={i}>{c}</li>)}
                </ul>
              )}
            </div>
          </div>
        </section>
      )}

      {down.length > 0 && (
        <p className="flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-[13px] leading-5 text-warning" role="note">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span>{down.map((s) => s.source).join(", ")} {down.length === 1 ? "was" : "were"} unavailable, so these results may be incomplete.</span>
        </p>
      )}

      {data.recommendations.length === 0 ? (
        <p className="rounded-xl border border-border bg-surface p-4 text-[14px] text-muted-foreground">No datasets matched. Try describing what you want to predict and from what kind of data.</p>
      ) : (
        <>
          <h3 className="text-sm font-medium text-muted-foreground">{plural(data.count, "dataset")} ranked</h3>
          <div className="space-y-3">
            {shown.map((item, i) => <DatasetCard key={item.dataset.id} item={item} index={i} busy={busy} onAnalyze={onAnalyze} recommended={final?.outcome === "recommended" && final.best_match?.dataset_id === item.dataset.id} />)}
          </div>
          {data.recommendations.length > INITIAL && (
            <Button variant="ghost" size="sm" onClick={() => setAll((v) => !v)}>
              {all ? "Show fewer" : `Show ${data.recommendations.length - INITIAL} more`}
            </Button>
          )}
        </>
      )}
    </div>
  );
}
