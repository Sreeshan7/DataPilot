import { AlertTriangle, ArrowRight, ExternalLink, Info, ListChecks, Quote, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import { Markdown } from "@/components/chat/Markdown";
import { Rich } from "@/components/chat/Rich";
import { Badge, type Tone } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import { hostOf, humanize, plural, safeUrl } from "@/lib/format";
import type { AlgorithmRecommendation, Paper, RecommendedApproach, ReportedMetric, ResearchResponse } from "@/types";

const LEVEL: Record<string, Tone> = { strong: "success", moderate: "info", limited: "warning", none: "neutral", unknown: "neutral" };

function Section({ title, hint, children }: { title: string; hint?: string; children: ReactNode }) {
  return (
    <section className="space-y-4">
      <div>
        <h3 className="text-base font-semibold tracking-tight">{title}</h3>
        {hint && <p className="mt-0.5 text-[13px] leading-5 text-subtle">{hint}</p>}
      </div>
      {children}
    </section>
  );
}

function Labelled({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <h6 className="text-[11px] font-medium uppercase tracking-wider text-subtle">{label}</h6>
      <div className="mt-1 text-[13.5px] leading-6 text-foreground/90">{children}</div>
    </div>
  );
}

function Quoted({ items }: { items: { statement: string; quote: string }[] }) {
  return (
    <ul className="space-y-3">
      {items.map((f, i) => (
        <li key={i}>
          <p>{f.statement}</p>
          <blockquote className="mt-1 flex gap-2 border-l-2 border-border pl-3 text-[12.5px] italic leading-5 text-subtle [overflow-wrap:anywhere]">
            <Quote className="mt-0.5 size-3 shrink-0" aria-hidden /><span>“{f.quote}”</span>
          </blockquote>
        </li>
      ))}
    </ul>
  );
}

/** What the paper did, from its own text. Every finding is shown with the exact words it rests on. */
function PaperAnalysisBlock({ a }: { a: NonNullable<Paper["analysis"]> }) {
  return (
    <Disclosure
      defaultOpen label={<span className="flex items-center gap-2 text-[13.5px] font-semibold text-foreground"><Sparkles className="size-3.5 text-accent" aria-hidden />Analysis of this paper</span>}
      meta={a.basis === "full_text" ? "read from its text" : "read from its abstract"} className="mt-3 rounded-lg border border-accent/20 bg-accent-soft/10 px-3.5 py-2.5">
      <div className="space-y-3.5 pb-1.5">
        {a.problem && <Labelled label="The problem">{a.problem}</Labelled>}
        {a.data_used && <Labelled label="Data it used">{a.data_used}</Labelled>}
        {a.approach && <Labelled label="Approach">{a.approach}</Labelled>}
        {a.findings.length > 0 && <Labelled label="What it found"><Quoted items={a.findings} /></Labelled>}
        {a.limitations.length > 0 && <Labelled label="Limits it states"><Quoted items={a.limitations} /></Labelled>}
        {a.relevance && (
          <Labelled label="For your project"><span className="text-muted-foreground">{a.relevance}</span> <Badge tone="accent" title="An AI model's view, not a fact from the paper">AI view</Badge></Labelled>
        )}
        <p className="text-xs text-subtle">Findings and limits are quoted from the {a.basis === "full_text" ? "first pages of the paper" : "abstract"}; each quote was checked against that text.</p>
      </div>
    </Disclosure>
  );
}

export function PaperCard({ paper, index }: { paper: Paper; index: number }) {
  const url = safeUrl(paper.url);
  return (
    <article className="animate-fade-up rounded-xl border border-border bg-surface p-5" style={{ animationDelay: `${Math.min(index, 6) * 45}ms` }}>
      <h4 className="text-[15px] font-semibold leading-snug [overflow-wrap:anywhere]">
        {url ? <a href={url} target="_blank" rel="noopener noreferrer nofollow" className="hover:text-accent hover:underline hover:underline-offset-4">{paper.title}</a> : paper.title}
      </h4>
      <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-muted-foreground">
        <span>{paper.venue ?? paper.source}</span>
        {paper.publication_year != null && (
          <><span aria-hidden>·</span><span title={paper.year_evidence === "source_metadata" ? "Year taken from the paper's identifier" : "Year found in the retrieved text"}>{paper.publication_year}</span></>
        )}
        {paper.authors && paper.authors.length > 0 && <><span aria-hidden>·</span><span className="[overflow-wrap:anywhere]">{paper.authors.join(", ")}</span></>}
      </p>
      <div className="mt-3 flex flex-wrap items-center gap-1.5">
        <Badge tone="accent">Relevance {Math.round(paper.relevance_score)}</Badge>
        {paper.retrieval_status === "full_text" && <Badge tone="success" title="Methods and figures for this paper were matched in the first pages of its own open-access text">Full text read</Badge>}
        {paper.paper_type && <Badge tone={paper.paper_type === "survey" ? "info" : "neutral"} title="Read from the title and abstract wording">{paper.paper_type === "survey" ? "Survey" : paper.paper_type === "applied" ? "Applied study" : "Method paper"}</Badge>}
        {paper.citation_count ? <Badge title="Citation count reported by the index">{paper.citation_count.toLocaleString()} citations</Badge> : null}
        {paper.ai_relevance != null && <Badge tone={paper.ai_relevance >= 3 ? "success" : "info"} title="An AI model's judgement of topical fit, not a fact about the paper"><Sparkles className="size-3" aria-hidden />AI fit {paper.ai_relevance}/3</Badge>}
        {(paper.matched_terms ?? []).map((t) => <Badge key={`t-${t}`} tone="neutral">mentions {t}</Badge>)}
        {paper.methods.map((m) => <Badge key={m}>{m}</Badge>)}
      </div>
      <p className="mt-3 text-[13.5px] leading-6 text-muted-foreground">{paper.relevance_reason}</p>
      {paper.ai_reason && <p className="mt-1 text-[13px] leading-5 text-subtle">AI note: {paper.ai_reason}</p>}
      {paper.analysis && <PaperAnalysisBlock a={paper.analysis} />}
      {paper.abstract_snippet && (
        <Disclosure label="Retrieved text" className="mt-2" meta="from the source">
          <blockquote className="border-l-2 border-border pl-3 text-[13px] leading-6 text-muted-foreground [overflow-wrap:anywhere]">{paper.abstract_snippet}</blockquote>
          <p className="mt-2 text-xs text-subtle">Only this snippet was retrieved; the full paper was not opened.</p>
        </Disclosure>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5">
        {url && (
          <a href={url} target="_blank" rel="noopener noreferrer nofollow" className="inline-flex items-center gap-1.5 text-[13px] text-muted-foreground hover:text-foreground">
            {hostOf(url)} <ExternalLink className="size-3.5" aria-hidden />
          </a>
        )}
        {safeUrl(paper.open_access_url ?? "") && (
          <a href={safeUrl(paper.open_access_url ?? "") ?? undefined} target="_blank" rel="noopener noreferrer nofollow" className="inline-flex items-center gap-1.5 text-[13px] text-accent hover:underline">
            Open PDF <ExternalLink className="size-3.5" aria-hidden />
          </a>
        )}
      </div>
    </article>
  );
}

function Bullets({ title, items, tone }: { title: string; items: string[]; tone?: "good" | "bad" }) {
  if (!items.length) return null;
  const marker = tone === "good" ? "marker:text-success" : tone === "bad" ? "marker:text-warning" : "marker:text-subtle";
  return (
    <div>
      <h6 className="text-[11px] font-medium uppercase tracking-wider text-subtle">{title}</h6>
      <ul className={`mt-1.5 list-disc space-y-1 pl-4 text-[13.5px] leading-6 text-muted-foreground ${marker}`}>{items.map((t, i) => <li key={i}><Rich>{t}</Rich></li>)}</ul>
    </div>
  );
}

/** How to actually run one algorithm: when to pick it, what to use, the steps, the settings and the pitfalls. */
function ExecutionPlan({ e }: { e: NonNullable<AlgorithmRecommendation["analysis"]>["execution"] }) {
  if (!e || (e.steps.length === 0 && !e.choose_when)) return null;
  return (
    <Disclosure defaultOpen label={<span className="flex items-center gap-2 text-[13.5px] font-semibold text-foreground"><ListChecks className="size-3.5 text-accent" aria-hidden />How to run it</span>}
      meta={e.source === "ai_inferred" ? "written for your data" : "general guidance"} className="mt-2.5 rounded-lg border border-accent/20 bg-accent-soft/10 px-3.5 py-2.5">
      <div className="space-y-4 pb-1.5">
        {e.libraries.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wider text-subtle">Use</span>
            {e.libraries.map((l) => <span key={l} className="rounded-md border border-border bg-muted/50 px-2 py-0.5 font-mono text-xs text-muted-foreground">{l}</span>)}
          </div>
        )}
        {e.steps.length > 0 && (
          <ol className="space-y-2.5">
            {e.steps.map((step, i) => (
              <li key={i} className="flex gap-3 text-[13.5px] leading-6">
                <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border border-accent/30 bg-accent-soft text-[11px] font-semibold text-accent" aria-hidden>{i + 1}</span>
                <span className="text-muted-foreground"><Rich>{step}</Rich></span>
              </li>
            ))}
          </ol>
        )}
        <div className="grid gap-4 sm:grid-cols-2">
          <Bullets title="Settings to tune" items={e.key_settings} />
          <Bullets title="Pitfalls" items={e.pitfalls} tone="bad" />
        </div>
        {e.validate_with && <Labelled label="How to validate it"><Rich>{e.validate_with}</Rich></Labelled>}
      </div>
    </Disclosure>
  );
}

function Recommendation({ r, titles, first }: { r: AlgorithmRecommendation; titles: Record<string, string>; first: boolean }) {
  const a = r.analysis;
  return (
    <li className="px-4 py-4 sm:px-5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="grid size-6 place-items-center rounded-md border border-border bg-raised text-xs font-semibold text-muted-foreground">{r.rank}</span>
        <span className="text-[15px] font-semibold">{r.method}</span>
        <Badge tone={LEVEL[r.literature_support]}>Literature: {r.literature_support}</Badge>
        <Badge tone={LEVEL[r.dataset_compatibility]}>Fit for your task: {r.dataset_compatibility}</Badge>
        <span className="text-xs text-subtle">{plural(r.evidence_paper_count, "paper")}</span>
        {a?.ai_enriched && <Badge tone="accent" title="Refined by an AI model using the papers and your dataset"><Sparkles className="size-3" aria-hidden />AI-refined</Badge>}
      </div>
      {a?.execution?.choose_when && <p className="mt-2.5 text-[14px] leading-6 text-foreground/90"><span className="font-medium">Pick it when: </span><Rich>{a.execution.choose_when}</Rich></p>}
      {a?.fit_for_your_data && <p className="mt-2 text-[14px] leading-6 text-foreground/90"><span className="font-medium">For your data: </span><Rich>{a.fit_for_your_data}</Rich></p>}
      {r.cautions.length > 0 && <ul className="mt-1.5 list-disc space-y-0.5 pl-5 text-[13px] leading-5 text-warning marker:text-warning/60">{r.cautions.map((c, i) => <li key={i}>{c}</li>)}</ul>}
      {a?.execution && <ExecutionPlan e={a.execution} />}
      {a && (
        <Disclosure defaultOpen={first} label="Full analysis" meta="strengths, weaknesses, preparation, tuning" className="mt-2.5">
          <div className="space-y-4 pb-1">
            <p className="text-[13.5px] leading-6 text-muted-foreground">{a.what_it_is}</p>
            <div className="grid gap-4 sm:grid-cols-2">
              <Bullets title="Strengths" items={a.strengths} tone="good" />
              <Bullets title="Weaknesses" items={a.weaknesses} tone="bad" />
              <Bullets title="Prepare the data" items={a.preprocessing} />
              <Bullets title="Tune these" items={a.tuning} />
            </div>
            {a.how_literature_used_it.length > 0 && (
              <div>
                <h6 className="text-[11px] font-medium uppercase tracking-wider text-subtle">How the papers used it</h6>
                <ul className="mt-1.5 space-y-2">
                  {a.how_literature_used_it.map((u) => (
                    <li key={u.paper_id} className="text-[13.5px] leading-6 text-muted-foreground"><span className="font-medium text-foreground/90">{titles[u.paper_id] ?? "A retrieved paper"}</span>: {u.note}</li>
                  ))}
                </ul>
              </div>
            )}
            <p className="text-xs text-subtle">{a.ai_enriched ? "Profile from DataPilot's built-in knowledge, refined by an AI model against the papers and your dataset." : "Profile from DataPilot's built-in knowledge of the method; paper notes come from the retrieved text."}</p>
          </div>
        </Disclosure>
      )}
    </li>
  );
}

function Comparison({ recs, benchmarks }: { recs: AlgorithmRecommendation[]; benchmarks: ReportedMetric[] }) {
  if (recs.length < 2) return null;
  const best = (method: string) => benchmarks.find((b) => b.method === method);
  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-surface">
      <table className="w-full min-w-[34rem] text-left text-[13px]">
        <thead className="text-xs text-subtle"><tr><th className="px-3.5 py-2 font-medium">Algorithm</th><th className="font-medium">Literature</th><th className="font-medium">Fit for your task</th><th className="font-medium">Papers</th><th className="px-3.5 font-medium">A reported figure</th></tr></thead>
        <tbody className="divide-y divide-border">
          {recs.map((r) => {
            const m = best(r.method);
            return (
              <tr key={r.method} className="align-top">
                <td className="px-3.5 py-2 font-medium">{r.rank}. {r.method}</td>
                <td className="py-2"><Badge tone={LEVEL[r.literature_support]}>{r.literature_support}</Badge></td>
                <td className="py-2"><Badge tone={LEVEL[r.dataset_compatibility]}>{r.dataset_compatibility}</Badge></td>
                <td className="py-2 tabular-nums text-muted-foreground">{r.evidence_paper_count}</td>
                <td className="px-3.5 py-2 text-muted-foreground">{m ? <>{m.metric} {m.value} <span className="text-xs text-subtle">in its own setting</span></> : "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Approach({ a }: { a: RecommendedApproach }) {
  const pill = (label: string, name: string, tone: Tone) => (
    <span className="inline-flex items-center gap-2 rounded-lg border border-border bg-raised px-3 py-1.5 text-[13px]"><span className="text-xs text-subtle">{label}</span><Badge tone={tone}>{name}</Badge></span>
  );
  return (
    <div className="rounded-2xl border border-accent/30 bg-gradient-to-b from-accent-soft/25 to-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="text-[15px] font-semibold">Recommended approach</h4>
        <Badge tone={a.source === "ai_inferred" ? "accent" : "neutral"}>{a.source === "ai_inferred" ? "AI-written, from the papers and your data" : "DataPilot's guidance"}</Badge>
      </div>
      <p className="mt-2.5 text-[14.5px] leading-6 text-foreground/90">{a.summary}</p>
      <div className="mt-4 flex flex-wrap items-center gap-2.5">
        {pill("Baseline", a.baseline, "neutral")}<ArrowRight className="size-4 text-subtle" aria-hidden />
        {pill("Main model", a.primary, "success")}
        {a.stretch && <><ArrowRight className="size-4 text-subtle" aria-hidden />{pill("Contrast with", a.stretch, "info")}</>}
      </div>
      <ol className="mt-5 space-y-2.5">
        {a.steps.map((step, i) => (
          <li key={i} className="flex gap-3 text-[13.5px] leading-6">
            <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border border-accent/30 bg-accent-soft text-[11px] font-semibold text-accent" aria-hidden>{i + 1}</span>
            <span className="text-muted-foreground"><Rich>{step}</Rich></span>
          </li>
        ))}
      </ol>
      {a.caveats.length > 0 && (
        <ul className="mt-4 list-disc space-y-1 border-t border-border pt-3 pl-5 text-[13px] leading-5 text-subtle">{a.caveats.map((c, i) => <li key={i}>{c}</li>)}</ul>
      )}
    </div>
  );
}

function Benchmarks({ items }: { items: ReportedMetric[] }) {
  return (
    <div className="space-y-2">
      <p className="flex items-start gap-2 text-[13px] leading-5 text-muted-foreground"><Info className="mt-0.5 size-4 shrink-0 text-subtle" aria-hidden />Each figure was reported by its source in its own experiment and dataset. It is not a prediction for yours.</p>
      <div className="overflow-x-auto rounded-xl border border-border bg-surface">
        <table className="w-full min-w-[30rem] text-left text-[13px]">
          <thead className="text-xs text-subtle"><tr><th className="px-3.5 py-2 font-medium">Metric</th><th className="font-medium">Value</th><th className="font-medium">Method</th><th className="px-3.5 font-medium">Source</th></tr></thead>
          <tbody className="divide-y divide-border">
            {items.map((m, i) => (
              <tr key={`${m.paper_id}-${m.metric}-${i}`} className="align-top">
                <td className="px-3.5 py-2">{m.metric}{m.context ? <div className="text-xs text-subtle">{m.context}</div> : null}</td>
                <td className="py-2 font-medium tabular-nums">{m.value}</td>
                <td className="py-2 text-muted-foreground">{m.method ?? "—"}{m.evidence === "paper_text" ? <div className="text-xs text-subtle">from the paper's text</div> : null}</td>
                <td className="px-3.5 py-2"><a href={safeUrl(m.url) ?? undefined} target="_blank" rel="noopener noreferrer nofollow" className="line-clamp-2 text-accent hover:underline [overflow-wrap:anywhere]">{m.paper_title}</a></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Every model the research touched, side by side, with the one DataPilot suggests for this project marked. */
function ModelSummary({ data }: { data: ResearchResponse }) {
  const norm = (s: string) => s.trim().toLowerCase();
  const recs = data.algorithm_recommendations;
  const recNames = new Set(recs.map((r) => norm(r.method)));
  const extra = data.methods.filter((m) => !recNames.has(norm(m.method)));
  if (recs.length === 0 && extra.length === 0) return null;
  const ap = data.approach;
  const roleOf = (method: string): { label: string; tone: Tone } | null => {
    if (!ap) return null;
    if (norm(ap.primary) === norm(method)) return { label: "Main model", tone: "success" };
    if (norm(ap.baseline) === norm(method)) return { label: "Baseline", tone: "neutral" };
    if (ap.stretch && norm(ap.stretch) === norm(method)) return { label: "Contrast with", tone: "info" };
    return null;
  };
  const bestName = ap?.primary ?? recs[0]?.method;
  const best = recs.find((r) => norm(r.method) === norm(bestName ?? "")) ?? recs[0];
  const same = (a: string, b: string) => { const x = norm(a), y = norm(b); return !!x && !!y && (x === y || x.includes(y) || y.includes(x)); };
  const figures = (method: string) => data.benchmarks.filter((b) => b.method && same(b.method, method));
  const rowNames = [...recs.map((r) => r.method), ...extra.map((m) => m.method)];
  const benchOnly = [...new Set(data.benchmarks.filter((b) => b.method && !rowNames.some((n) => same(n, b.method as string))).map((b) => b.method as string))];
  const unattributed = data.benchmarks.filter((b) => !b.method).length;
  const Figures = ({ items }: { items: ReportedMetric[] }) => items.length === 0 ? <span className="text-subtle">—</span> : (
    <ul className="space-y-1.5">
      {items.map((m, i) => (
        <li key={`${m.paper_id}-${m.metric}-${i}`} className="leading-5">
          <span className="text-muted-foreground">{m.metric}</span>{" "}
          <a href={safeUrl(m.url) ?? undefined} target="_blank" rel="noopener noreferrer nofollow" title={`${m.paper_title}${m.context ? " · " + m.context : ""}`} className="font-semibold tabular-nums text-foreground hover:text-accent hover:underline">{m.value}</a>
        </li>
      ))}
    </ul>
  );
  const th = "px-3.5 py-2.5 font-medium whitespace-nowrap";
  return (
    <Section title="Models compared and the best one for your project" hint="Every model found in the research, side by side. The highlighted row is the one DataPilot recommends for your project.">
      {best && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border border-success/30 bg-success/10 px-4 py-3 text-[14px]">
          <Badge tone="success">Best for your project</Badge>
          <strong className="font-semibold">{best.method}</strong>
          <span className="text-muted-foreground">literature: {best.literature_support} · fit for your task: {best.dataset_compatibility} · {plural(best.evidence_paper_count, "paper")}</span>
        </div>
      )}
      <div className="overflow-x-auto rounded-xl border border-border bg-surface">
        <table className="w-full min-w-[52rem] text-left text-[13px]">
          <thead className="bg-raised text-xs uppercase tracking-wider text-subtle">
            <tr><th className={th}>#</th><th className={th}>Model</th><th className={th}>Role</th><th className={th}>Literature</th><th className={th}>Fit for your task</th><th className={th}>Papers</th><th className={th}>Main strength</th><th className={th}>Main weakness</th><th className={th}>Reported results (accuracy &amp; other metrics)</th></tr>
          </thead>
          <tbody className="divide-y divide-border">
            {recs.map((r) => {
              const isBest = best != null && norm(r.method) === norm(best.method);
              const role = roleOf(r.method);
              return (
                <tr key={r.method} className={`align-top ${isBest ? "bg-success/10" : ""}`}>
                  <td className="px-3.5 py-3 tabular-nums text-muted-foreground">{r.rank}</td>
                  <td className="px-3.5 py-3 font-semibold">{r.method}{isBest && <span className="ml-2 align-middle"><Badge tone="success">Best</Badge></span>}</td>
                  <td className="px-3.5 py-3">{role ? <Badge tone={role.tone}>{role.label}</Badge> : <span className="text-subtle">—</span>}</td>
                  <td className="px-3.5 py-3"><Badge tone={LEVEL[r.literature_support]}>{r.literature_support}</Badge></td>
                  <td className="px-3.5 py-3"><Badge tone={LEVEL[r.dataset_compatibility]}>{r.dataset_compatibility}</Badge></td>
                  <td className="px-3.5 py-3 tabular-nums text-muted-foreground">{r.evidence_paper_count}</td>
                  <td className="min-w-[11rem] px-3.5 py-3 text-muted-foreground">{r.analysis?.strengths?.[0] ? `✓ ${r.analysis.strengths[0]}` : "—"}</td>
                  <td className="min-w-[11rem] px-3.5 py-3 text-muted-foreground">{(r.analysis?.weaknesses?.[0] ?? r.cautions[0]) ? `⚠ ${r.analysis?.weaknesses?.[0] ?? r.cautions[0]}` : "—"}</td>
                  <td className="min-w-[12rem] px-3.5 py-3"><Figures items={figures(r.method)} /></td>
                </tr>
              );
            })}
            {extra.map((m, i) => (
              <tr key={`x-${m.method}`} className="align-top">
                <td className="px-3.5 py-3 tabular-nums text-muted-foreground">{recs.length + i + 1}</td>
                <td className="px-3.5 py-3 font-semibold">{m.method}</td>
                <td className="px-3.5 py-3"><span className="text-subtle">—</span></td>
                <td className="px-3.5 py-3"><Badge tone="neutral">mentioned</Badge></td>
                <td className="px-3.5 py-3"><span className="text-subtle">—</span></td>
                <td className="px-3.5 py-3 tabular-nums text-muted-foreground">{m.verified_paper_count}</td>
                <td className="px-3.5 py-3 text-subtle">—</td><td className="px-3.5 py-3 text-subtle">—</td><td className="min-w-[12rem] px-3.5 py-3"><Figures items={figures(m.method)} /></td>
              </tr>
            ))}
            {benchOnly.map((name, i) => (
              <tr key={`b-${name}`} className="align-top">
                <td className="px-3.5 py-3 tabular-nums text-muted-foreground">{recs.length + extra.length + i + 1}</td>
                <td className="px-3.5 py-3 font-semibold">{name}</td>
                <td className="px-3.5 py-3"><span className="text-subtle">—</span></td>
                <td className="px-3.5 py-3"><Badge tone="neutral">mentioned</Badge></td>
                <td className="px-3.5 py-3"><span className="text-subtle">—</span></td>
                <td className="px-3.5 py-3 text-subtle">—</td><td className="px-3.5 py-3 text-subtle">—</td><td className="px-3.5 py-3 text-subtle">—</td>
                <td className="min-w-[12rem] px-3.5 py-3"><Figures items={figures(name)} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-subtle">Reported results come from each paper's own experiment and dataset, not from yours; hover a number to see its paper, click to open it.{unattributed > 0 ? ` ${unattributed} more figure${unattributed === 1 ? "" : "s"} could not be tied to one method and ${unattributed === 1 ? "is" : "are"} listed under Reported results.` : ""} Paper counts show how often a method appears, which is evidence of use, not proof it is best.</p>
    </Section>
  );
}

function Understood({ data }: { data: ResearchResponse }) {
  const i = data.interpretation;
  if (!i || i.core_terms.length === 0) return null;
  const fixes = Object.entries(i.corrections);
  return (
    <div className="mt-4 flex flex-wrap items-center gap-x-2 gap-y-2 border-t border-border pt-4 text-[13px] text-muted-foreground">
      <span>Searched for</span>
      {i.core_terms.map((t) => <Badge key={t} tone="accent">{t}</Badge>)}
      {i.context_terms.length > 0 && <><span>in the context of</span>{i.context_terms.map((t) => <Badge key={t}>{t}</Badge>)}</>}
      {fixes.length > 0 && <span className="text-subtle">· read “{fixes[0][0]}” as “{fixes[0][1]}”{fixes.length > 1 ? ` (+${fixes.length - 1} more)` : ""}</span>}
      {i.ai_assisted && <Badge tone="accent"><Sparkles className="size-3" aria-hidden />AI helped interpret</Badge>}
      {(i.outcome || i.task) && (
        <span className="basis-full text-subtle">
          {i.outcome ? <>Predicting <strong className="font-medium text-foreground/90">{i.outcome}</strong></> : null}
          {i.outcome && i.task ? " · " : null}{i.task ? <>task: {i.task}</> : null}
        </span>
      )}
      {i.dataset_check && i.dataset_check.status !== "unknown" && (
        <span className={`basis-full rounded-lg border px-3 py-2 ${i.dataset_check.status === "mismatch" ? "border-warning/30 bg-warning/10 text-warning" : "border-success/30 bg-success/10 text-success"}`}>
          {i.dataset_check.status === "mismatch" ? "Check this: " : "Matches your data: "}{i.dataset_check.note}{i.dataset_check.source === "ai_inferred" ? " (AI-read)" : ""}
        </span>
      )}
    </div>
  );
}

export function ResearchResult({ data, onRetry, onRefine, busy }: { data: ResearchResponse; onRetry?: () => void; onRefine?: (problem: string) => void; busy?: boolean }) {
  if (data.status !== "success") {
    const copy: Record<string, string> = {
      needs_clarification: "I need a bit more about the problem before I can research it.",
      no_results: "I searched, but found no academic sources that matched this problem well enough to show.",
      search_unavailable: "Literature search is temporarily unavailable, so I haven't shown anything I couldn't verify.",
    };
    return (
      <div className="space-y-4">
        <div className="rounded-xl border border-border bg-surface p-5">
          <p className="text-[14.5px] font-medium">{copy[data.status] ?? "Research couldn't be completed."}</p>
          {data.message && <p className="mt-1.5 text-[13.5px] leading-6 text-muted-foreground">{data.message}</p>}
          {data.status === "needs_clarification" && (data.interpretation?.options ?? []).length > 0 && onRefine && (
            <div className="mt-4 flex flex-wrap gap-2.5" role="group" aria-label="Pick what you mean">
              {(data.interpretation?.options ?? []).map((o) => (
                <button key={o} type="button" disabled={busy} onClick={() => onRefine(o)}
                  className="rounded-full border border-border bg-raised px-4 py-2 text-[13.5px] transition-colors hover:border-accent/50 hover:bg-muted disabled:opacity-50">{o}</button>
              ))}
            </div>
          )}
          {onRetry && data.status !== "needs_clarification" && <button type="button" onClick={onRetry} className="mt-3 rounded-lg border border-border bg-raised px-3 py-1.5 text-sm hover:bg-muted">Try again</button>}
        </div>
        {data.limitations.length > 0 && <p className="text-xs leading-5 text-subtle">{data.limitations[0]}</p>}
      </div>
    );
  }
  const s = data.research_summary;
  const titles = Object.fromEntries(data.papers.map((p) => [p.id, p.title]));
  const ai = data.ai_interpretation;
  const aiNote = data.gemini.requested && data.gemini.status !== "used";
  return (
    <div className="space-y-10">
      <section className="rounded-2xl border border-border bg-gradient-to-b from-raised to-surface p-5 sm:p-6" aria-label="Research summary">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[13px] text-muted-foreground">
          <span><strong className="text-lg font-semibold tabular-nums text-foreground">{s.papers_selected}</strong> relevant papers</span>
          <span><strong className="text-lg font-semibold tabular-nums text-foreground">{data.methods.length}</strong> methods named</span>
          <span className="text-subtle">from {s.results_retrieved} results · {plural(s.queries_used, "search", "searches")}{(s.excluded_off_topic ?? 0) > 0 ? ` · ${s.excluded_off_topic} off-topic left out` : ""}</span>
        </div>
        <p className="mt-3 text-[14px] leading-6 text-foreground/85">{data.datapilot_summary}</p>
        <Understood data={data} />
      </section>

      {aiNote && (
        <p className="flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/10 px-3 py-2 text-[13px] leading-5 text-warning" role="note">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
          <span>AI enhancement is temporarily unavailable. DataPilot can still show the verified literature below.</span>
        </p>
      )}

      {data.approach && <Approach a={data.approach} />}

      {data.algorithm_recommendations.length > 0 && (
        <Section title="Algorithm analysis" hint="What the papers use, how each method suits your data, and how to prepare and tune it. Frequency is evidence of use, not proof of the best method.">
          <Comparison recs={data.algorithm_recommendations} benchmarks={data.benchmarks} />
          <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
            {data.algorithm_recommendations.map((r, i) => <Recommendation key={r.method} r={r} titles={titles} first={i === 0} />)}
          </ul>
        </Section>
      )}

      {ai && (
        <Section title="AI interpretation" hint="Written by an AI model from the evidence above. Not verified literature.">
          <div className="rounded-xl border border-accent/25 bg-accent-soft/20 p-4">
            <Badge tone="accent" className="mb-2"><Sparkles className="size-3" aria-hidden />AI interpretation · checked against the retrieved evidence</Badge>
            <Markdown>{ai.landscape}</Markdown>
            {ai.approaches.length > 0 && <Markdown className="mt-2">{ai.approaches.map((a) => `- ${a}`).join("\n")}</Markdown>}
            {ai.comparison && <Markdown className="mt-2">{ai.comparison}</Markdown>}
            {ai.fit_for_project && <Markdown className="mt-2">{ai.fit_for_project}</Markdown>}
            {ai.ai_inferred_methods.length > 0 && <p className="mt-3 text-[13px] text-muted-foreground">AI also suggests (not found in the retrieved papers): {ai.ai_inferred_methods.join(", ")}.</p>}
          </div>
        </Section>
      )}

      <Section title="Papers" hint="Ranked by relevance to your problem. Every one links to its source.">
        <div className="space-y-4">{data.papers.map((p, i) => <PaperCard key={p.id} paper={p} index={i} />)}</div>
      </Section>

      {data.methods.length > 0 && (
        <Section title="Methods in the literature" hint="Counted from each paper's retrieved title and snippet.">
          <ul className="flex flex-wrap gap-2">
            {data.methods.map((m) => <li key={m.method} className="rounded-lg border border-border bg-surface px-3 py-1.5 text-[13px]">{m.method} <span className="text-subtle">· {plural(m.verified_paper_count, "paper")}</span></li>)}
          </ul>
        </Section>
      )}

      {data.benchmarks.length > 0 && <Section title="Reported results"><Benchmarks items={data.benchmarks} /></Section>}

      <ModelSummary data={data} />

      <Disclosure label="Limitations and provenance" className="rounded-xl border border-border bg-surface/40 px-3.5 py-2.5">
        <div className="space-y-3 text-[13px] leading-5 text-muted-foreground">
          <ul className="list-disc space-y-1 pl-5 marker:text-subtle">{data.limitations.map((l, i) => <li key={i}>{l}</li>)}</ul>
          <ul className="space-y-1.5">{data.provenance.map((p, i) => <li key={i}><Badge className="mr-2">{humanize(p.evidence)}</Badge>{p.note}</li>)}</ul>
          <details className="text-xs text-subtle"><summary className="cursor-pointer">Searches used</summary><ul className="mt-1.5 space-y-1">{data.queries.map((q) => <li key={q} className="font-mono">{q}</li>)}</ul></details>
        </div>
      </Disclosure>
    </div>
  );
}
