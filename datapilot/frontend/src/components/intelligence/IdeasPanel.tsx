import { Rich } from "@/components/chat/Rich";
import { BookOpenText, Lightbulb, Sparkles } from "lucide-react";
import { Badge, type Tone } from "@/components/ui/badge";
import { Disclosure } from "@/components/ui/disclosure";
import { humanize } from "@/lib/format";
import { Button } from "@/components/ui/button";
import type { ModelRecommendation, ProjectCompatibility, ProjectIdea, ResearchRequest } from "@/types";

const DIFFICULTY: Record<string, Tone> = { beginner: "success", intermediate: "info", advanced: "warning" };
const COMPAT: Record<string, { tone: Tone; label: string }> = {
  MATCH: { tone: "success", label: "Fits your project" },
  UNKNOWN: { tone: "neutral", label: "Can't confirm a fit" },
  MISMATCH: { tone: "danger", label: "Doesn't fit your project" },
};
const FEASIBILITY: Record<string, { tone: Tone; label: string }> = {
  high: { tone: "success", label: "Looks very workable" },
  medium: { tone: "info", label: "Workable with care" },
  low: { tone: "warning", label: "Needs more data" },
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

const ROLE: Record<string, { tone: Tone; label: string }> = {
  target: { tone: "accent", label: "Target" },
  key_driver: { tone: "success", label: "Key driver" },
  supporting: { tone: "info", label: "Supporting" },
  watch_out: { tone: "warning", label: "Watch out" },
  exclude: { tone: "danger", label: "Leave out" },
};

function KeyColumns({ columns }: { columns: NonNullable<ProjectIdea["blueprint"]>["key_columns"] }) {
  if (!columns.length) return null;
  return (
    <div>
      <h5 className="text-xs font-medium uppercase tracking-wider text-subtle">What matters in your data</h5>
      <ul className="mt-2.5 divide-y divide-border overflow-hidden rounded-lg border border-border bg-background/40">
        {columns.map((c) => {
          const role = ROLE[c.role] ?? ROLE.supporting;
          return (
            <li key={c.column} className="flex flex-wrap items-start gap-x-3 gap-y-1.5 px-3.5 py-3">
              <span className="rounded-md border border-border bg-muted/50 px-2 py-0.5 font-mono text-xs">{c.column}</span>
              <Badge tone={role.tone}>{role.label}</Badge>
              {c.evidence === "measured" && <Badge title="Measured on your rows, not guessed">Measured</Badge>}
              {c.evidence === "ai_inferred" && <Badge title="An AI model's reading of the column name; not measured"><Sparkles className="size-3" aria-hidden />AI view</Badge>}
              <p className="basis-full text-[13.5px] leading-6 text-muted-foreground"><Rich>{c.why}</Rich></p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function BuildSteps({ steps }: { steps: NonNullable<ProjectIdea["blueprint"]>["build_steps"] }) {
  return (
    <ol className="space-y-3.5">
      {steps.map((st, i) => (
        <li key={`${i}-${st.title}`} className="flex gap-3.5">
          <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full border border-accent/30 bg-accent-soft text-xs font-semibold text-accent" aria-hidden>{i + 1}</span>
          <div className="min-w-0">
            <p className="text-[14px] font-medium leading-6">{st.title}</p>
            <p className="text-[13.5px] leading-6 text-muted-foreground"><Rich>{st.detail}</Rich></p>
          </div>
        </li>
      ))}
    </ol>
  );
}

export function IdeaCard({ idea, onResearch, busy }: { idea: ProjectIdea; onResearch?: (request: ResearchRequest, label: string) => void; busy?: boolean }) {
  const feasibility = idea.feasibility ? (FEASIBILITY[idea.feasibility.level] ?? FEASIBILITY.medium) : null;
  const bp = idea.blueprint;
  return (
    <article className={`rounded-2xl border bg-surface p-5 sm:p-6 ${idea.recommended ? "border-accent/40 shadow-[0_0_0_1px_hsl(var(--accent)/0.12)]" : "border-border"}`}>
      <div className="flex flex-wrap items-center gap-2">
        {idea.recommended && <Badge tone="accent"><Sparkles className="size-3" aria-hidden />Best first project</Badge>}
        {feasibility && <Badge tone={feasibility.tone}>{feasibility.label}</Badge>}
        <Badge tone={DIFFICULTY[idea.difficulty] ?? "neutral"}>{idea.difficulty}</Badge>
        <Badge>{humanize(idea.task)}</Badge>
        {bp?.effort && <Badge title="A realistic estimate for a motivated learner">{bp.effort}</Badge>}
        {bp?.source === "ai_inferred" && <Badge tone="accent" title="Written by an AI model from your column names and types; check it against your data"><Sparkles className="size-3" aria-hidden />AI-written plan</Badge>}
      </div>

      <h4 className="mt-3 text-[17px] font-semibold leading-snug tracking-tight [overflow-wrap:anywhere]"><Rich>{idea.title}</Rich></h4>
      <p className="mt-1.5 text-[14.5px] leading-6 text-foreground/90"><Rich>{bp?.headline && bp.headline !== idea.title.replace(/`/g, "") ? bp.headline : idea.problem_statement}</Rich></p>

      {bp && (
        <div className="mt-5 space-y-6">
          <div>
            <h5 className="text-xs font-medium uppercase tracking-wider text-subtle">Why it matters</h5>
            <p className="mt-1.5 text-[14px] leading-6 text-muted-foreground"><Rich>{bp.real_world_value}</Rich></p>
          </div>

          <KeyColumns columns={bp.key_columns} />

          <Disclosure defaultOpen={idea.recommended} label={<span className="text-[14px] font-semibold text-foreground">How to build it</span>} meta={`${bp.build_steps.length} steps`} className="rounded-lg border border-border px-3.5 py-2.5">
            <div className="space-y-5 pb-1.5 pt-1">
              <BuildSteps steps={bp.build_steps} />
              {bp.evaluation_plan && (
                <div className="border-t border-border pt-4">
                  <h5 className="text-xs font-medium uppercase tracking-wider text-subtle">How to evaluate</h5>
                  <p className="mt-1.5 text-[13.5px] leading-6 text-muted-foreground"><Rich>{bp.evaluation_plan}</Rich></p>
                </div>
              )}
              <Bullets title="Watch out for" items={bp.pitfalls} />
              <Bullets title="Stretch goals" items={bp.stretch_goals} />
            </div>
          </Disclosure>
        </div>
      )}

      {!bp && idea.required_features.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="Features this idea uses">
          {idea.required_features.map((f) => <li key={f} className="rounded-md border border-border bg-muted/50 px-2 py-0.5 font-mono text-xs text-muted-foreground">{f}</li>)}
        </ul>
      )}
      {idea.feasibility && idea.feasibility.reasons.length > 0 && (
        <p className="mt-4 text-[13px] leading-5 text-subtle">Data check: {idea.feasibility.reasons.join(" · ")}</p>
      )}

      {onResearch && idea.research_request && (
        <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-border pt-4">
          <Button variant="primary" disabled={busy}
            onClick={() => onResearch(idea.research_request as unknown as ResearchRequest, idea.title.replace(/`/g, ""))}>
            <BookOpenText aria-hidden />Research this project
          </Button>
          <span className="text-[13px] text-subtle">Finds real papers and the methods they use for this exact project.</span>
        </div>
      )}

      <Disclosure label="More detail" className="mt-4">
        <div className="space-y-3.5">
          <p className="text-[13.5px] leading-6 text-muted-foreground"><Rich>{idea.why_it_fits}</Rich></p>
          <Bullets title="Challenges" items={idea.challenges} />
          <Bullets title="Metrics" items={idea.evaluation_metrics} />
          <Bullets title="Baselines" items={idea.baseline_models} />
          <Bullets title="Stronger models" items={idea.stronger_models} />
          <Bullets title="Limitations" items={idea.limitations} />
        </div>
      </Disclosure>
    </article>
  );
}

export function IdeasPanel({ ideas, onResearch, busy }: { ideas: ProjectIdea[]; onResearch?: (request: ResearchRequest, label: string) => void; busy?: boolean }) {
  if (ideas.length === 0) return <p className="rounded-xl border border-border bg-surface/60 p-4 text-[13.5px] leading-6 text-muted-foreground">There isn't enough grounded information to suggest projects for this dataset yet. A file with a confirmed target gives the most.</p>;
  return <div className="space-y-5">{ideas.map((i) => <IdeaCard key={i.id} idea={i} onResearch={onResearch} busy={busy} />)}</div>;
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
