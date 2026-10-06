import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DatasetCard } from "@/components/datasets/DatasetCard";
import { IdeaCard } from "@/components/intelligence/IdeasPanel";
import { ResearchOffer } from "@/components/research/ResearchOffer";
import { ResearchResult } from "@/components/research/ResearchResult";
import type { NextAction } from "@/types";

const offer = (project: Record<string, unknown>) =>
  ({ action: "RESEARCH_PROBLEM", id: "research_problem", label: "Research this problem", available: true, payload: { project } }) as unknown as NextAction;

describe("Research offer shows the whole prompt", () => {
  it("shows the full project description under the question box, read-only", () => {
    render(<ResearchOffer action={offer({ problem: "churn prediction", description: "I want to predict which telecom customers will leave next month." })} onRun={() => undefined} aiAvailable busy={false} />);
    const panel = screen.getByLabelText("Your project description");
    expect(within(panel).getByText("I want to predict which telecom customers will leave next month.")).toBeInTheDocument();
    expect((screen.getByLabelText("What should the papers be about?") as HTMLInputElement).value).toBe("churn prediction");
  });

  it("shows no description panel when there is no description", () => {
    render(<ResearchOffer action={offer({ problem: "churn prediction" })} onRun={() => undefined} aiAvailable busy={false} />);
    expect(screen.queryByLabelText("Your project description")).toBeNull();
  });
});

const metric = (m: string, v: string, method: string | null, id = "p1") =>
  ({ metric: m, value: v, method, paper_id: id, paper_title: "A churn paper", url: "https://x.org/p", statement: "s", evidence: "web_verified", context: "its own data" });
const execution = {
  choose_when: "You want the highest accuracy on a table and can spend time tuning.", libraries: ["xgboost", "pandas"],
  steps: ["Split the data with a stratified 80/20 split.", "Fit with default settings.", "Tune the learning rate.", "Evaluate once on the test rows."],
  key_settings: ["learning_rate: start at 0.1"], validate_with: "Use stratified cross-validation and report ROC AUC.", pitfalls: ["Defaults can overfit small data."], source: "ai_inferred",
};
const rec = (rank: number, method: string, extra: Record<string, unknown> = {}) => ({
  rank, method, literature_support: "strong", dataset_compatibility: "strong", evidence_paper_count: 2, paper_ids: [], why: "w", cautions: [],
  analysis: { what_it_is: "x", fit_for_your_data: "ff", strengths: ["Accurate on tables"], weaknesses: ["Needs tuning"], preprocessing: [], tuning: [], how_literature_used_it: [], source: "datapilot_knowledge", ai_enriched: false, ...extra },
});
const research = {
  status: "success", queries: [], limitations: [], provenance: [], papers: [],
  methods: [{ method: "Random Forest", verified_paper_count: 2 }, { method: "SVM Ensemble", verified_paper_count: 1 }],
  benchmarks: [metric("Accuracy", "96.2%", "XGBoost"), metric("F1", "0.91", "XGBoost"), metric("AUC", "0.97", "xgboost"), metric("Accuracy", "88.0%", "Random Forest"), metric("Accuracy", "70.1%", "k-Nearest Neighbors")],
  research_summary: { queries_used: 1, results_retrieved: 1, papers_found: 1, papers_selected: 1, results_rejected_as_non_academic: 0, excluded_off_topic: 0, search_provider: "x" },
  datapilot_summary: "Found sources.", gemini: { requested: false, calls_made: 0, assisted: [], status: "not_requested" },
  algorithm_recommendations: [rec(1, "XGBoost", { execution }), rec(2, "Random Forest", { execution: { ...execution, source: "datapilot_knowledge", choose_when: "You want a forgiving model." } })],
  approach: { summary: "s", baseline: "Logistic Regression", primary: "XGBoost", stretch: "Random Forest", steps: ["one"], caveats: [], source: "datapilot_knowledge" },
} as never;

describe("How to run each algorithm", () => {
  it("shows when to pick each algorithm, what to use, and the ordered steps", () => {
    render(<ResearchResult data={research} />);
    expect(screen.getByText(/You want the highest accuracy on a table/)).toBeInTheDocument();
    expect(screen.getByText(/You want a forgiving model/)).toBeInTheDocument();     // a second option, not only one best
    expect(screen.getAllByText("How to run it")).toHaveLength(2);
    expect(screen.getAllByText("xgboost").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Split the data with a stratified 80/20 split.")).toHaveLength(2);
    expect(screen.getByText("written for your data")).toBeInTheDocument();
    expect(screen.getByText("general guidance")).toBeInTheDocument();
  });

  it("still renders when an older server sends no execution plan", () => {
    const bare = JSON.parse(JSON.stringify(research));
    bare.algorithm_recommendations.forEach((r: { analysis: { execution?: unknown } }) => { delete r.analysis.execution; });
    render(<ResearchResult data={bare} />);
    expect(screen.queryByText("How to run it")).toBeNull();
    expect(screen.getByText("Algorithm analysis")).toBeInTheDocument();
  });
});

describe("Models compared table", () => {
  it("lists every reported metric for each model, marks the best one, and keeps models that only the papers mention", () => {
    render(<ResearchResult data={research} />);
    const heading = screen.getByText("Models compared and the best one for your project");
    const section = heading.closest("section") as HTMLElement;
    const table = within(section).getByRole("table");
    const xgb = within(table).getByText("XGBoost").closest("tr") as HTMLElement;
    for (const v of ["96.2%", "0.91", "0.97"]) expect(within(xgb).getByText(v)).toBeInTheDocument();   // accuracy, F1 and AUC, not just one figure
    expect(within(xgb).getByText("Best")).toBeInTheDocument();
    expect(within(xgb).getByText("Main model")).toBeInTheDocument();
    expect(within(table).getByText("88.0%")).toBeInTheDocument();
    expect(within(table).getByText("SVM Ensemble")).toBeInTheDocument();                        // named in the papers, not recommended
    expect(within(table).getByText("k-Nearest Neighbors")).toBeInTheDocument();                  // has a figure but no recommendation row
    expect(within(section).getByText("Best for your project")).toBeInTheDocument();
  });

  it("links each figure to the paper it came from", () => {
    render(<ResearchResult data={research} />);
    const link = screen.getAllByRole("link", { name: "96.2%" })[0];
    expect(link).toHaveAttribute("href", "https://x.org/p");
    expect(link).toHaveAttribute("title", expect.stringContaining("A churn paper"));
  });
});

describe("Project ideas: how to finish and present", () => {
  const idea = {
    id: "i1", title: "Predict `Churn`", problem_statement: "Build a model.", task: "binary_classification", target: "Churn", required_features: ["tenure"],
    optional_features: [], why_it_fits: "ok", challenges: [], evaluation_metrics: [], baseline_models: [], stronger_models: [], difficulty: "beginner",
    extension_potential: "", limitations: [], evidence_level: "datapilot_parsed", state: "POSSIBLE", recommended: true,
    blueprint: { headline: "Retention teams decide who to call first.", real_world_value: "Helps retention.", key_columns: [], build_steps: [{ title: "Clean", detail: "Clean it." }],
      evaluation_plan: "", pitfalls: [], stretch_goals: [], effort: "a weekend", source: "deterministic",
      deliverables: ["A notebook that reproduces every step"], tools: ["pandas", "scikit-learn"], portfolio_angle: "Lead with the decision this supports." },
  } as never;

  it("shows deliverables, tools and how to present the project", () => {
    render(<IdeaCard idea={idea} />);
    expect(screen.getByText("What to deliver")).toBeInTheDocument();
    expect(screen.getByText("A notebook that reproduces every step")).toBeInTheDocument();
    expect(screen.getByText("scikit-learn")).toBeInTheDocument();
    expect(screen.getByText("Lead with the decision this supports.")).toBeInTheDocument();
  });

  it("hides that block for a plan from an older server", () => {
    const old = JSON.parse(JSON.stringify(idea));
    delete old.blueprint.deliverables; delete old.blueprint.tools; delete old.blueprint.portfolio_angle;
    render(<IdeaCard idea={old} />);
    expect(screen.queryByText("What to deliver")).toBeNull();
    expect(screen.queryByText("How to present it")).toBeNull();
  });
});

describe("Dataset setting match", () => {
  const item = (fit: number | null, reason: string | null) => ({
    rank: 1, ranking_score: 65.2, verification_status: "verified", source_type: "verified_external", reasons: ["matches the target"], ranking_factors: [],
    dataset: { name: "NYC Yellow Taxi Trip Data", source: "kaggle", owner: "x", url: "https://kaggle.com/d/x", task_types: ["regression"], ai_fit: fit, ai_fit_reason: reason },
  }) as never;

  it("shows the AI setting match and its reason", () => {
    render(<DatasetCard item={item(3, "Covers New York taxi trips with fares.")} onAnalyze={() => undefined} busy={false} />);
    expect(screen.getByText("Setting match 3/3")).toBeInTheDocument();
    expect(screen.getByText(/Covers New York taxi trips with fares\./)).toBeInTheDocument();
  });

  it("shows nothing when the AI did not judge the dataset", () => {
    render(<DatasetCard item={item(null, null)} onAnalyze={() => undefined} busy={false} />);
    expect(screen.queryByText(/Setting match/)).toBeNull();
  });
});
