import { Rich } from "@/components/chat/Rich";
import { Database, ExternalLink, Target } from "lucide-react";
import type { ReactNode } from "react";
import { Badge, type Tone } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import { bytes, compactNumber, hostOf, humanize, percent, plural, safeUrl } from "@/lib/format";
import type { DatasetAnalysisResponse, QualityFinding, ResearchRequest } from "@/types";
import { FeaturePanel } from "./FeaturePanel";
import { CompatibilityPanel, IdeasPanel, ModelsPanel } from "./IdeasPanel";

const SEVERITY: Record<string, { tone: Tone; rank: number }> = { critical: { tone: "danger", rank: 0 }, warning: { tone: "warning", rank: 1 }, info: { tone: "neutral", rank: 2 } };

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="space-y-2.5">
      <div>
        <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>
        {hint && <p className="text-[13px] leading-5 text-subtle">{hint}</p>}
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface px-3.5 py-3">
      <div className="text-[11px] font-medium uppercase tracking-wider text-subtle">{label}</div>
      <div className="mt-1 text-xl font-semibold tabular-nums leading-none">{value}</div>
      {sub && <div className="mt-1.5 truncate text-xs text-subtle">{sub}</div>}
    </div>
  );
}

function Findings({ findings }: { findings: QualityFinding[] }) {
  const sorted = [...findings].sort((a, b) => (SEVERITY[a.severity]?.rank ?? 3) - (SEVERITY[b.severity]?.rank ?? 3));
  if (sorted.length === 0) return <p className="rounded-xl border border-success/25 bg-success/10 px-3.5 py-3 text-[13.5px] text-success">No data-quality problems were found.</p>;
  const row = (f: QualityFinding, i: number) => (
    <li key={`${f.code}-${f.column}-${i}`} className="flex items-start gap-2.5 px-3.5 py-2.5 text-[13.5px] leading-5">
      <Badge tone={SEVERITY[f.severity]?.tone ?? "neutral"} className="mt-0.5 shrink-0">{f.severity}</Badge>
      <span className="min-w-0"><span className="font-medium">{humanize(f.code)}</span><span className="text-muted-foreground"> · <Rich>{f.detail}</Rich></span></span>
    </li>
  );
  return (
    <div className="overflow-hidden rounded-xl border border-border bg-surface">
      <ul className="divide-y divide-border">{sorted.slice(0, 5).map(row)}</ul>
      {sorted.length > 5 && <div className="border-t border-border px-3.5 py-2"><Disclosure label={`Show ${sorted.length - 5} more`}><ul className="-mx-3.5 divide-y divide-border">{sorted.slice(5).map(row)}</ul></Disclosure></div>}
    </div>
  );
}

function rowsNote(a: DatasetAnalysisResponse["analysis"], parsed: boolean, rows: number | null): string {
  if (parsed) return a.ingestion?.truncated ? "truncated at the limit" : "read from your file";
  if (rows == null) return "not published";
  if (a.schema.row_count_evidence === "inferred") return "estimated from the file's size";
  if (a.schema.row_count_evidence === "ai_inferred") return "from web sources, unverified";
  return "reported by the registry";
}

function columnsNote(a: DatasetAnalysisResponse["analysis"], parsed: boolean): string {
  if (parsed) return plural(a.column_profiles.length, "profiled column");
  if (a.schema.level === "ai_inferred") return "from web sources, unverified";
  if (a.schema.source.startsWith("file header")) return "read from the file's header";
  return a.schema.level.replace(/_/g, " ");
}

function WebSources({ a }: { a: DatasetAnalysisResponse["analysis"] }) {
  const sources = a.external?.web_sources ?? [];
  if (sources.length === 0) return null;
  return (
    <section aria-label="Web sources" className="rounded-xl border border-warning/30 bg-warning/10 p-3.5">
      <p className="text-[13.5px] leading-6 text-foreground/90">
        <strong className="font-medium">Found through a web search.</strong> No dataset registry published this structure, so an AI model
        read it from the pages below. Treat it as a lead and confirm it before relying on it.
      </p>
      <ul className="mt-2 space-y-1">
        {sources.map((s) => (
          <li key={s.url} className="text-[13px]">
            <a href={safeUrl(s.url) ?? undefined} target="_blank" rel="noopener noreferrer nofollow" className="inline-flex items-center gap-1.5 text-accent hover:underline">
              {s.title || hostOf(s.url) || "source"} <ExternalLink className="size-3" aria-hidden />
            </a>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function AnalysisView({ data, onResearch, busy }: { data: DatasetAnalysisResponse; onResearch?: (request: ResearchRequest, label: string) => void; busy?: boolean }) {
  const a = data.analysis;
  const pi = data.project_intelligence;
  const parsed = a.mode === "file_parsed";
  const rows = parsed ? a.ingestion?.rows_read ?? null : a.schema.row_count;
  const cols = a.schema.column_count ?? (a.column_profiles.length || null);
  const findings = a.quality?.findings ?? [];
  const u = pi?.understanding;

  return (
    <div className="space-y-7">
      <p className="flex items-start gap-2.5 rounded-xl border border-border bg-surface/60 px-3.5 py-3 text-[13.5px] leading-6 text-muted-foreground">
        <Database className="mt-1 size-4 shrink-0 text-accent" aria-hidden />
        <span>{data.evidence_statement}</span>
      </p>

      <WebSources a={a} />

      <Section title="Dataset profile">
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          <Stat label="Rows" value={rows != null && !parsed && a.schema.row_count_evidence !== "registry_verified" ? `~${compactNumber(rows)}` : compactNumber(rows)} sub={rowsNote(a, parsed, rows)} />
          <Stat label="Columns" value={compactNumber(cols)} sub={columnsNote(a, parsed)} />
          <Stat label="Missing" value={parsed && a.quality ? `${a.quality.overall_missing_pct.toFixed(1)}%` : "—"} sub={parsed ? "of all cells" : "needs the file"} />
          <Stat label="Duplicates" value={parsed && a.quality?.duplicate_row_count != null ? compactNumber(a.quality.duplicate_row_count) : "—"} sub={parsed ? "repeated rows" : "needs the file"} />
        </div>
        {parsed && a.ingestion && <p className="text-xs text-subtle">{a.ingestion.display_name} · {a.ingestion.file_type} · {bytes(a.ingestion.size_bytes)}</p>}
      </Section>

      {u && (
        <Section title="Target and task" hint="What the data can be used to predict.">
          <div className="rounded-xl border border-border bg-surface p-4">
            {u.best_target ? (
              <>
                <div className="flex flex-wrap items-center gap-2.5">
                  <Target className="size-4 text-accent" aria-hidden />
                  <span className="font-mono text-[14px] font-medium">{u.best_target.column}</span>
                  <Badge tone={u.target_is_confident ? "success" : "warning"}>{u.target_is_confident ? "Confident" : "Unconfirmed"}</Badge>
                  <Badge>{humanize(u.target_source)}</Badge>
                  {u.primary_task && <Badge tone="accent">{humanize(u.primary_task.task)}</Badge>}
                </div>
                <p className="mt-2.5 text-[14px] leading-6 text-muted-foreground"><Rich>{u.summary}</Rich></p>
                {u.alternative_targets.length > 0 && (
                  <p className="mt-2 text-[13px] text-subtle">Other candidates: {u.alternative_targets.map((t) => t.column).join(", ")}. Name the target yourself in your next message to change it.</p>
                )}
              </>
            ) : <p className="text-[14px] leading-6 text-muted-foreground">{u.summary}</p>}
            {u.blockers.length > 0 && (
              <ul className="mt-3 space-y-1 border-t border-border pt-3 text-[13px] leading-5 text-danger">{u.blockers.slice(0, 3).map((b, i) => <li key={i}>• {b}</li>)}</ul>
            )}
          </div>
        </Section>
      )}

      {parsed && (
        <Section title="Data quality" hint={`${plural(findings.length, "finding")}${findings.some((f) => f.severity === "critical") ? " (including critical ones)" : ""}`}>
          <Findings findings={findings} />
        </Section>
      )}

      {pi && (
        <Section title="Feature intelligence" hint="Which columns help predict the target, measured on your rows.">
          <FeaturePanel fi={pi.feature_intelligence} />
        </Section>
      )}

      {pi?.compatibility && <CompatibilityPanel c={pi.compatibility} />}

      {pi && (
        <Section title="Project ideas" hint="Grounded in your actual columns.">
          <IdeasPanel ideas={pi.project_ideas} onResearch={onResearch} busy={busy} />
        </Section>
      )}

      {pi && pi.model_recommendations.length > 0 && (
        <Section title="Techniques to try" hint="No score is implied for any of these.">
          <ModelsPanel models={pi.model_recommendations} />
        </Section>
      )}

      {parsed && a.column_profiles.length > 0 && (
        <Disclosure label="All columns" meta={`${a.column_profiles.length}`} className="rounded-xl border border-border bg-surface/40 px-3.5 py-2.5">
          <div className="-mx-1 overflow-x-auto pb-1">
            <table className="w-full min-w-[34rem] text-left text-[13px]">
              <thead className="text-xs text-subtle"><tr><th className="py-1.5 pr-3 font-medium">Column</th><th className="pr-3 font-medium">Type</th><th className="pr-3 text-right font-medium">Missing</th><th className="pr-3 text-right font-medium">Distinct</th><th className="font-medium">Flags</th></tr></thead>
              <tbody className="divide-y divide-border">
                {a.column_profiles.map((c) => (
                  <tr key={c.normalized_name}>
                    <td className="max-w-[12rem] truncate py-1.5 pr-3 font-mono" title={c.normalized_name}>{c.normalized_name}</td>
                    <td className="pr-3 text-muted-foreground">{c.inferred_type}</td>
                    <td className="pr-3 text-right tabular-nums text-muted-foreground">{percent(c.missing_pct / 100, 1)}</td>
                    <td className="pr-3 text-right tabular-nums text-muted-foreground">{c.unique_count.toLocaleString()}</td>
                    <td className="text-xs text-subtle">{c.flags.map(humanize).join(", ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Disclosure>
      )}

      {(a.limitations.length > 0 || (pi?.limitations.length ?? 0) > 0 || data.notices.length > 0) && (
        <Disclosure label="Limitations and notes" className="rounded-xl border border-border bg-surface/40 px-3.5 py-2.5">
          <ul className="list-disc space-y-1 pl-5 text-[13px] leading-5 text-muted-foreground marker:text-subtle">
            {[...a.limitations, ...(pi?.limitations ?? []), ...data.notices].map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </Disclosure>
      )}
    </div>
  );
}
