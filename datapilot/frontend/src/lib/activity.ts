/**
 * Product-level activity, built only from facts the backend returned.
 *
 * The backend answers each request with one final response, so there is no live stream of its internal
 * stages. This module therefore never invents progress. It provides:
 *  - `EXPECTED_STEPS`: the stages a workflow performs, shown as an outline (no ticks) while the call is open;
 *  - `traceFrom*`: after the response, a record of what really happened, with the real counts.
 * (A truly live progress feed would need a streaming endpoint; see DEVELOPMENT_STATE.md.)
 */
import type { AssistResponse, DatasetAnalysisResponse, DatasetRecommendationResponse, ResearchResponse } from "@/types";
import { plural } from "./format";

export type StepStatus = "done" | "skipped";

export interface TraceStep {
  id: string;
  label: string;
  detail?: string;
  status: StepStatus;
}

export type Workflow = "discovery" | "analysis" | "research";

export const EXPECTED_STEPS: Record<Workflow, string[]> = {
  discovery: ["Extracting requirements", "Searching datasets", "Comparing candidates", "Checking compatibility", "Preparing recommendations"],
  analysis: ["Reading dataset", "Profiling columns", "Checking data quality", "Detecting target and task", "Checking leakage", "Generating project ideas"],
  research: ["Building search queries", "Searching academic literature", "Filtering relevant papers", "Comparing methods", "Preparing research findings"],
};

export const WORKING_TITLE: Record<Workflow, string> = {
  discovery: "Finding datasets for your project",
  analysis: "Analysing your dataset",
  research: "Researching your problem",
};

export function journeyLabel(journey: string | null | undefined): string {
  if (journey === "USER_A_DATASET_DISCOVERY") return "finding a dataset";
  if (journey === "USER_B_DATASET_ANALYSIS") return "analysing a dataset";
  return "your request";
}

const step = (id: string, label: string, detail?: string, status: StepStatus = "done"): TraceStep => ({ id, label, detail, status });

function meta(discovery: DatasetRecommendationResponse): Record<string, unknown> {
  return (discovery.meta ?? {}) as Record<string, unknown>;
}

export function traceFromDiscovery(d: DatasetRecommendationResponse): TraceStep[] {
  const out: TraceStep[] = [];
  const u = d.project_requirements?.project_understanding;
  const bits = [u?.domain?.value && `domain: ${u.domain.value}`, u?.ml_tasks?.[0]?.task && `task: ${u.ml_tasks[0].task}`].filter(Boolean);
  out.push(step("requirements", "Extracted project requirements", bits.join(" · ") || undefined));
  for (const s of d.sources ?? []) {
    const ok = s.status === "ok";
    out.push(step(`source-${s.source}`, `Searched ${s.source}`,
      ok ? plural(s.candidate_count, "candidate") : s.status === "skipped" ? "skipped" : "unavailable right now", ok ? "done" : "skipped"));
  }
  const raw = meta(d).raw_candidates;
  out.push(step("rank", "Compared and ranked candidates",
    typeof raw === "number" ? `${raw} found, ${d.count} ranked` : `${d.count} ranked`));
  const levels: Record<string, number> = {};
  for (const r of d.recommendations ?? []) {
    const k = r.intelligence?.compatibility?.overall;
    if (k) levels[k] = (levels[k] ?? 0) + 1;
  }
  const summary = Object.entries(levels).map(([k, n]) => `${n} ${k}`).join(", ");
  out.push(step("compat", "Checked compatibility", summary || undefined, summary ? "done" : "skipped"));
  if (d.final_recommendation) out.push(step("recommend", "Prepared a recommendation", d.final_recommendation.outcome));
  return out;
}

export function traceFromAnalysis(r: DatasetAnalysisResponse): TraceStep[] {
  const a = r.analysis;
  const pi = r.project_intelligence;
  const out: TraceStep[] = [];
  if (a.mode === "file_parsed") {
    out.push(step("read", "Read your file", a.ingestion
      ? `${a.ingestion.rows_read.toLocaleString()} rows × ${a.schema.column_count ?? a.column_profiles.length} columns${a.ingestion.truncated ? " (truncated at the limit)" : ""}`
      : undefined));
    out.push(step("profile", "Profiled columns", plural(a.column_profiles.length, "column")));
    const findings = a.quality?.findings ?? [];
    const critical = findings.filter((f) => f.severity === "critical").length;
    out.push(step("quality", "Checked data quality", `${plural(findings.length, "finding")}${critical ? `, ${critical} critical` : ""}`));
  } else {
    const src = a.external?.source;
    out.push(step("lookup", src ? `Looked up ${src}` : "Looked up the dataset", a.external ? a.external.name : "no registry record found", a.external ? "done" : "skipped"));
    const cols = a.schema.column_count;
    const label = a.schema.level === "ai_inferred" ? "Searched the web for the structure (AI-read, unverified)"
      : a.schema.source.startsWith("file header") ? "Read the column header from the dataset's file"
      : "Read the published schema";
    out.push(step("schema", label, cols ? `${cols} columns` : "no schema is published", cols ? "done" : "skipped"));
    out.push(step("norows", "Rows were not read", "statistics and quality checks need the file", "skipped"));
  }
  if (pi) {
    const u = pi.understanding;
    out.push(step("target", "Detected target and task",
      u.best_target ? `${u.best_target.column}${u.primary_task ? ` · ${u.primary_task.task.replace(/_/g, " ")}` : ""}` : "no confident target",
      u.best_target ? "done" : "skipped"));
    const fi = pi.feature_intelligence;
    if (fi.available) {
      const leaks = fi.features.filter((f) => f.role === "POSSIBLE_LEAKAGE").length;
      out.push(step("leakage", "Checked for leakage", leaks ? plural(leaks, "possible leak") : "none found"));
      out.push(step("signal", "Measured feature signal", fi.method ? `${fi.method.rows_used.toLocaleString()} rows, ${fi.method.reliable ? "model beat the baseline" : "weak signal"}` : undefined));
    } else {
      out.push(step("signal", "Measured feature signal", fi.unavailable_reason ?? "not available", "skipped"));
    }
    out.push(step("ideas", "Generated project ideas", plural(pi.project_ideas.length, "idea"), pi.project_ideas.length ? "done" : "skipped"));
    if (pi.gemini.status === "used" || pi.gemini.status === "partial") out.push(step("ai", "AI rephrased the ideas", "text only; columns unchanged"));
  }
  return out;
}

export function traceFromAssist(r: AssistResponse): TraceStep[] {
  const route = r.routing;
  const out: TraceStep[] = [
    step("route", `Understood your request as ${route.intent === "AMBIGUOUS" ? "unclear" : journeyLabel(route.intent)}`,
      `${Math.round(route.confidence * 100)}% sure`, route.intent === "AMBIGUOUS" ? "skipped" : "done"),
  ];
  if (r.status !== "COMPLETED") out.push(step("ask", r.status === "NEEDS_CLARIFICATION" ? "Asked a question instead of guessing" : "Needs more input", undefined, "skipped"));
  if (r.discovery) out.push(...traceFromDiscovery(r.discovery));
  if (r.analysis) out.push(...traceFromAnalysis(r.analysis));
  return out;
}

export function traceFromResearch(r: ResearchResponse): TraceStep[] {
  const s = r.research_summary;
  const out: TraceStep[] = [step("queries", "Built search queries", plural(s.queries_used, "query", "queries"), s.queries_used ? "done" : "skipped")];
  if (r.status === "needs_clarification") return out;
  if (r.status === "search_unavailable") return [...out, step("search", "Searched the web", "the search service is unavailable", "skipped")];
  out.push(step("search", "Searched the web", `${plural(s.results_retrieved, "result")} retrieved`));
  out.push(step("filter", "Kept academic sources", `${plural(s.papers_found, "paper")}${s.results_rejected_as_non_academic ? `, ${s.results_rejected_as_non_academic} non-academic dropped` : ""}`));
  out.push(step("select", "Selected the most relevant", plural(s.papers_selected, "paper"), s.papers_selected ? "done" : "skipped"));
  out.push(step("methods", "Compared methods", `${plural(r.methods.length, "method")} named in the retrieved text`, r.methods.length ? "done" : "skipped"));
  if (r.gemini.requested) {
    out.push(step("ai", "AI interpretation", r.gemini.status === "used" ? "1 call, checked against the evidence" : r.gemini.note ?? r.gemini.status, r.gemini.status === "used" ? "done" : "skipped"));
  }
  return out;
}
