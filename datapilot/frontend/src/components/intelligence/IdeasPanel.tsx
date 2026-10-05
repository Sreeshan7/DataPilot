import { Rich } from "@/components/chat/Rich";
import { Lightbulb, Sparkles } from "lucide-react";
import { Badge, type Tone } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import { humanize } from "@/lib/format";
import type { ModelRecommendation, ProjectCompatibility, ProjectIdea } from "@/types";

const DIFFICULTY: Record<string, Tone> = { beginner: "success", intermediate: "info", advanced: "warning" };
const COMPAT: Record<string, { tone: Tone; label: string }> = {
  MATCH: { tone: "success", label: "Fits your project" },
  UNKNOWN: { tone: "neutral", label: "Can't confirm a fit" },
  MISMATCH: { tone: "danger", label: "Doesn't fit your project" },
};
const TIER: Record<string, string> = { baseline: "Start here", stronger: "Then try", specialised: "Specialised" };

function Bullets({ title, items }: { title: string; items: string[] }) {
  if (!items.length) return null;
  return (
    <div>
      <h5 className="mb-1 text-xs font-medium uppercase tracking-wider text-subtle">{title}</h5>
      <ul className="list-disc space-y-1 pl-4 text-[13px] leading-5 text-muted-foreground marker:text-subtle">{items.map((t, i) => <li key={i}><Rich>{t}</Rich></li>)}</ul>
    </div>
  );
}

export function IdeaCard({ idea }: { idea: ProjectIdea }) {
  return (
    <article className="rounded-xl border border-border bg-surface p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={DIFFICULTY[idea.difficulty] ?? "neutral"}>{idea.difficulty}</Badge>
        <Badge>{humanize(idea.task)}</Badge>
        {idea.source === "gemini_enhanced" && <Badge tone="accent"><Sparkles className="size-3" aria-hidden />AI-phrased</Badge>}
      </div>
      <h4 className="mt-2.5 text-[15px] font-semibold leading-snug [overflow-wrap:anywhere]"><Rich>{idea.title}</Rich></h4>
      <p className="mt-1.5 text-[14px] leading-6 text-muted-foreground"><Rich>{idea.problem_statement}</Rich></p>
      {idea.required_features.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="Features this idea uses">
          {idea.required_features.map((f) => <li key={f} className="rounded-md border border-border bg-muted/50 px-2 py-0.5 font-mono text-xs text-muted-foreground">{f}</li>)}
        </ul>
      )}
      <Disclosure label="Details" className="mt-3 border-t border-border pt-2.5">
        <div className="space-y-3.5">
          <p className="text-[13.5px] leading-6 text-muted-foreground"><Rich>{idea.why_it_fits}</Rich></p>
          <Bullets title="Challenges" items={idea.challenges} />
          <Bullets title="How to evaluate" items={idea.evaluation_metrics} />
          <Bullets title="Baselines" items={idea.baseline_models} />
          <Bullets title="Stronger models" items={idea.stronger_models} />
          <Bullets title="Limitations" items={idea.limitations} />
        </div>
      </Disclosure>
    </article>
  );
}

export function IdeasPanel({ ideas }: { ideas: ProjectIdea[] }) {
  if (ideas.length === 0) return <p className="rounded-xl border border-border bg-surface/60 p-4 text-[13.5px] leading-6 text-muted-foreground">There isn't enough grounded information to suggest projects for this dataset yet. A file with a confirmed target gives the most.</p>;
  return <div className="grid gap-3 lg:grid-cols-2">{ideas.map((i) => <IdeaCard key={i.id} idea={i} />)}</div>;
}

export function ModelsPanel({ models }: { models: ModelRecommendation[] }) {
  if (!models.length) return null;
  return (
    <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-surface">
      {models.map((m, i) => (
        <li key={`${m.model}-${i}`} className="px-3.5 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[14px] font-medium">{m.model}</span>
            <Badge tone={m.tier === "baseline" ? "accent" : "neutral"}>{TIER[m.tier] ?? m.tier}</Badge>
          </div>
          <p className="mt-1 text-[13.5px] leading-6 text-muted-foreground">{m.why}</p>
          {(m.cautions.length > 0 || m.preprocessing.length > 0) && (
            <Disclosure label="Preparation and cautions" className="mt-1">
              <div className="space-y-2.5">
                <Bullets title="Preparation" items={m.preprocessing} />
                <Bullets title="Watch out for" items={m.cautions} />
              </div>
            </Disclosure>
          )}
        </li>
      ))}
    </ul>
  );
}

export function CompatibilityPanel({ c }: { c: ProjectCompatibility }) {
  const style = COMPAT[c.status] ?? COMPAT.UNKNOWN;
  return (
    <section className="rounded-xl border border-border bg-surface p-4" aria-label="Goal compatibility">
      <div className="flex items-center gap-2.5">
        <Lightbulb className="size-4 text-accent" aria-hidden />
        <Badge tone={style.tone}>{style.label}</Badge>
      </div>
      <p className="mt-2 text-[14px] leading-6 text-foreground/90"><Rich>{c.summary}</Rich></p>
      <ul className="mt-3 space-y-2">
        {c.dimensions.map((d) => (
          <li key={d.dimension} className="flex items-start gap-2.5 text-[13.5px] leading-5">
            <Badge tone={d.status === "MATCH" ? "success" : d.status === "MISMATCH" ? "danger" : "neutral"} className="mt-0.5 w-[4.6rem] justify-center capitalize">{d.status.toLowerCase()}</Badge>
            <span className="text-muted-foreground"><span className="font-medium capitalize text-foreground">{d.dimension}</span>: {d.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
