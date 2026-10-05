import { AlertTriangle, ExternalLink, Info, Sparkles } from "lucide-react";
import type { ReactNode } from "react";
import { Markdown } from "@/components/chat/Markdown";
import { Badge, type Tone } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import { hostOf, humanize, plural, safeUrl } from "@/lib/format";
import type { AlgorithmRecommendation, Paper, ReportedMetric, ResearchResponse } from "@/types";

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
        {paper.ai_relevance != null && <Badge tone={paper.ai_relevance >= 3 ? "success" : "info"} title="An AI model's judgement of topical fit, not a fact about the paper"><Sparkles className="size-3" aria-hidden />AI fit {paper.ai_relevance}/3</Badge>}
        {(paper.matched_terms ?? []).map((t) => <Badge key={`t-${t}`} tone="neutral">mentions {t}</Badge>)}
        {paper.methods.map((m) => <Badge key={m}>{m}</Badge>)}
      </div>
      <p className="mt-3 text-[13.5px] leading-6 text-muted-foreground">{paper.relevance_reason}</p>
      {paper.ai_reason && <p className="mt-1 text-[13px] leading-5 text-subtle">AI note: {paper.ai_reason}</p>}
      {paper.abstract_snippet && (
        <Disclosure label="Retrieved text" className="mt-2" meta="from the source">
          <blockquote className="border-l-2 border-border pl-3 text-[13px] leading-6 text-muted-foreground [overflow-wrap:anywhere]">{paper.abstract_snippet}</blockquote>
          <p className="mt-2 text-xs text-subtle">Only this snippet was retrieved; the full paper was not opened.</p>
        </Disclosure>
      )}
      {url && (
        <a href={url} target="_blank" rel="noopener noreferrer nofollow" className="mt-2.5 inline-flex items-center gap-1.5 text-[13px] text-muted-foreground hover:text-foreground">
          {hostOf(url)} <ExternalLink className="size-3.5" aria-hidden />
        </a>
      )}
    </article>
  );
}

function Recommendation({ r }: { r: AlgorithmRecommendation }) {
  return (
    <li className="px-4 py-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="grid size-6 place-items-center rounded-md border border-border bg-raised text-xs font-semibold text-muted-foreground">{r.rank}</span>
        <span className="text-[14.5px] font-medium">{r.method}</span>
        <Badge tone={LEVEL[r.literature_support]}>Literature: {r.literature_support}</Badge>
        <Badge tone={LEVEL[r.dataset_compatibility]}>Fit for your task: {r.dataset_compatibility}</Badge>
        <span className="text-xs text-subtle">{plural(r.evidence_paper_count, "paper")}</span>
      </div>
      <p className="mt-2 text-[13.5px] leading-6 text-muted-foreground">{r.why}</p>
      {r.cautions.length > 0 && <ul className="mt-1 list-disc space-y-0.5 pl-5 text-[13px] leading-5 text-warning marker:text-warning/60">{r.cautions.map((c, i) => <li key={i}>{c}</li>)}</ul>}
    </li>
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
                <td className="py-2 text-muted-foreground">{m.method ?? "—"}</td>
                <td className="px-3.5 py-2"><a href={safeUrl(m.url) ?? undefined} target="_blank" rel="noopener noreferrer nofollow" className="line-clamp-2 text-accent hover:underline [overflow-wrap:anywhere]">{m.paper_title}</a></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
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
    </div>
  );
}

export function ResearchResult({ data, onRetry }: { data: ResearchResponse; onRetry?: () => void }) {
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
          {onRetry && data.status !== "needs_clarification" && <button type="button" onClick={onRetry} className="mt-3 rounded-lg border border-border bg-raised px-3 py-1.5 text-sm hover:bg-muted">Try again</button>}
        </div>
        {data.limitations.length > 0 && <p className="text-xs leading-5 text-subtle">{data.limitations[0]}</p>}
      </div>
    );
  }
  const s = data.research_summary;
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

      {data.algorithm_recommendations.length > 0 && (
        <Section title="Algorithm recommendations" hint="Literature support combined with fit for your task. Frequency is evidence of use, not proof of the best method.">
          <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">{data.algorithm_recommendations.map((r) => <Recommendation key={r.method} r={r} />)}</ul>
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
