import { Check, ExternalLink, HelpCircle, ScanSearch, ShieldAlert, Sparkles, X } from "lucide-react";
import { Badge, type Tone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Disclosure } from "@/components/ui/disclosure";
import { humanize, safeUrl, score } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { RankedDataset } from "@/types";

const COMPAT: Record<string, { tone: Tone; label: string }> = {
  high: { tone: "success", label: "High compatibility" },
  medium: { tone: "info", label: "Medium compatibility" },
  low: { tone: "danger", label: "Low compatibility" },
  unknown: { tone: "neutral", label: "Compatibility unknown" },
};
const SEVERITY: Record<string, Tone> = { info: "neutral", warning: "warning", critical: "danger" };
const MAX_CHIPS = 6;

function FeatureChips({ item }: { item: RankedDataset }) {
  const cov = item.intelligence?.compatibility.feature_coverage;
  if (!cov) return null;
  const all = [
    ...cov.matched.map((f) => ({ f, kind: "matched" as const })),
    ...cov.unknown.map((f) => ({ f, kind: "unknown" as const })),
    ...cov.missing.map((f) => ({ f, kind: "missing" as const })),
  ];
  if (all.length === 0) return null;
  const shown = all.slice(0, MAX_CHIPS);
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Requested features">
      {shown.map(({ f, kind }) => (
        <li key={`${kind}-${f.requirement}`} title={f.detail}
          className={cn("inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs",
            kind === "matched" && "border-success/30 bg-success/10 text-success",
            kind === "unknown" && "border-border bg-muted/50 text-muted-foreground",
            kind === "missing" && "border-danger/30 bg-danger/10 text-danger")}>
          {kind === "matched" ? <Check className="size-3" aria-label="matched" /> : kind === "unknown" ? <HelpCircle className="size-3" aria-label="unknown" /> : <X className="size-3" aria-label="missing" />}
          {humanize(f.requirement)}
        </li>
      ))}
      {all.length > MAX_CHIPS && <li className="px-1 py-0.5 text-xs text-subtle">+{all.length - MAX_CHIPS} more</li>}
    </ul>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h4 className="mb-1 text-xs font-medium uppercase tracking-wider text-subtle">{title}</h4>
      <ul className="list-disc space-y-1 pl-4 text-[13.5px] leading-5 text-muted-foreground marker:text-subtle">
        {items.map((t, i) => <li key={i}>{t}</li>)}
      </ul>
    </div>
  );
}

export function DatasetCard({ item, onAnalyze, busy, index = 0, recommended = false }: { item: RankedDataset; onAnalyze: (url: string, name: string) => void; busy: boolean; index?: number; recommended?: boolean }) {
  const d = item.dataset;
  const intel = item.intelligence;
  const compat = COMPAT[intel?.compatibility.overall ?? "unknown"] ?? COMPAT.unknown;
  const unverified = item.source_type === "ai_suggested";
  const url = safeUrl(d.url);
  const why = intel?.explanation.why_recommended.slice(0, 2) ?? item.reasons.slice(0, 2);
  const risks = (intel?.risks ?? []).filter((r) => r.severity !== "info").slice(0, 3);
  const ex = intel?.explanation;
  const factors = item.ranking_factors.filter((f) => f.contribution > 0).sort((a, b) => b.contribution - a.contribution).slice(0, 5);

  return (
    <article className={cn("animate-fade-up rounded-2xl border bg-surface p-4 sm:p-5", recommended ? "border-accent/50 shadow-[0_0_0_1px_hsl(var(--accent)/0.15)]" : "border-border")} style={{ animationDelay: `${Math.min(index, 5) * 55}ms` }} aria-label={`Dataset ${item.rank}: ${d.name}`}>
      <header className="flex items-start gap-3">
        <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-lg border border-border bg-raised text-[13px] font-semibold text-muted-foreground" aria-label={`Rank ${item.rank}`}>{item.rank}</span>
        <div className="min-w-0 flex-1">
          <h3 className="text-[16px] font-semibold leading-snug tracking-tight [overflow-wrap:anywhere]">{d.name}</h3>
          <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-muted-foreground">
            {recommended && <Badge tone="accent"><Check className="size-3" aria-hidden />Recommended</Badge>}
            <span className="capitalize">{d.source}</span>
            {d.owner ? <><span aria-hidden>·</span><span className="[overflow-wrap:anywhere]">{d.owner}</span></> : null}
            {unverified && <Badge tone="warning"><ShieldAlert className="size-3" aria-hidden />AI suggestion · not verified</Badge>}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <div className="text-xl font-semibold tabular-nums leading-none">{score(item.ranking_score)}</div>
          <div className="mt-1 text-[11px] uppercase tracking-wider text-subtle">Score</div>
        </div>
      </header>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Badge tone={compat.tone}>{compat.label}</Badge>
        {item.confidence && <Badge tone="neutral">Confidence: {item.confidence.level}</Badge>}
        {d.task_types.slice(0, 2).map((t) => <Badge key={t}>{humanize(t)}</Badge>)}
        {d.ai_fit != null && (
          <Badge tone={d.ai_fit >= 3 ? "success" : d.ai_fit === 2 ? "info" : "warning"}
            title="An AI model's judgement of how well this dataset matches the setting you described. It can change the order, never the score.">
            <Sparkles className="size-3" aria-hidden />Setting match {d.ai_fit}/3
          </Badge>
        )}
      </div>
      {d.ai_fit != null && d.ai_fit_reason && <p className="mt-2 text-[13px] leading-5 text-subtle">AI note: {d.ai_fit_reason}</p>}

      <div className="mt-3.5"><FeatureChips item={item} /></div>

      {why.length > 0 && (
        <div className="mt-4">
          <h4 className="mb-1 text-xs font-medium uppercase tracking-wider text-subtle">Why this dataset</h4>
          <ul className="list-disc space-y-1 pl-4 text-[14px] leading-6 text-foreground/85 marker:text-subtle">
            {why.map((w, i) => <li key={i}>{w.replace(/\s+/g, " ").trim().replace(/^./, (c) => c.toUpperCase())}</li>)}
          </ul>
        </div>
      )}

      {risks.length > 0 && (
        <ul className="mt-3 space-y-1.5" aria-label="Risks">
          {risks.map((r, i) => (
            <li key={i} className="flex items-start gap-2 text-[13px] leading-5 text-muted-foreground">
              <Badge tone={SEVERITY[r.severity] ?? "neutral"} className="mt-0.5 shrink-0">{r.severity}</Badge>
              <span>{r.detail}</span>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        {!unverified && url ? (
          <Button size="sm" variant="primary" disabled={busy} onClick={() => onAnalyze(url, d.name)}><ScanSearch aria-hidden />Analyze dataset</Button>
        ) : null}
        {url && (
          <a href={url} target="_blank" rel="noopener noreferrer" className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
            Open source <ExternalLink className="size-3.5" aria-hidden />
          </a>
        )}
      </div>

      {(ex || factors.length > 0) && (
        <Disclosure label="Evidence and details" className="mt-3 border-t border-border pt-3">
          <div className="space-y-4">
            {ex && <List title="Verified" items={ex.what_is_verified} />}
            {ex && <List title="Inferred (not verified)" items={ex.what_is_inferred} />}
            {ex && <List title="Unknown" items={ex.what_is_unknown} />}
            {ex && <List title="Missing" items={ex.what_is_missing} />}
            {ex && <List title="Limitations" items={ex.limitations} />}
            {factors.length > 0 && (
              <div>
                <h4 className="mb-1.5 text-xs font-medium uppercase tracking-wider text-subtle">What drove the score</h4>
                <ul className="space-y-1.5">
                  {factors.map((f) => (
                    <li key={f.name} className="grid grid-cols-[8.5rem_1fr_2.5rem] items-center gap-2 text-[13px] text-muted-foreground">
                      <span className="truncate">{humanize(f.name)}</span>
                      <span className="h-1.5 overflow-hidden rounded-full bg-muted"><span className="block h-full rounded-full bg-accent/70" style={{ width: `${Math.min(100, (f.contribution / (factors[0].contribution || 1)) * 100)}%` }} /></span>
                      <span className="text-right tabular-nums">{f.contribution.toFixed(1)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </Disclosure>
      )}
    </article>
  );
}
