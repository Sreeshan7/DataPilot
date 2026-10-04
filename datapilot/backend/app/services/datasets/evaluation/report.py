"""Report rendering: the human-readable side of an evaluation run.

Produces ``RANKING_EVALUATION.md`` and the machine-readable JSON next to it. The
Markdown is written for someone deciding whether to trust the ranker, so it leads
with the numbers, then the failures, then the analysis - and it never states a
conclusion the numbers do not support. Where a metric is weak because the pool
was weak, that is said on the same page.

Reproducibility: the report records the benchmark version, the ranker class, and
the full weight configuration used for the run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from app.core.config import Settings
from app.services.datasets.evaluation.ablation import render_ablation_table
from app.services.datasets.evaluation.audit import factor_influence_over_benchmark
from app.services.datasets.evaluation.diagnostics import render_case_report
from app.services.datasets.evaluation.fixtures import BENCHMARK_NOW
from app.services.datasets.evaluation.models import (
    AblationResult,
    EvaluationReport,
    SensitivityScenario,
)
from app.services.datasets.evaluation.sensitivity import (
    render_sensitivity_table,
    summarise_sensitivity,
)

REPORT_JSON_NAME = "evaluation_results.json"
REPORT_MARKDOWN_NAME = "RANKING_EVALUATION.md"


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _grade_summary(report: EvaluationReport) -> dict[str, int]:
    counts = {"3": 0, "2": 0, "1": 0, "0": 0, "-1": 0}
    for case in report.per_case:
        for grade in case.grades.values():
            counts[str(grade)] = counts.get(str(grade), 0) + 1
    return counts


def render_report(
    report: EvaluationReport,
    settings: Settings,
    ablations: Sequence[AblationResult],
    sensitivity: Sequence[SensitivityScenario],
) -> str:
    """Build the full Markdown report."""
    lines: list[str] = []
    add = lines.append

    add("# Dataset Ranking Evaluation")
    add("")
    add(
        f"Offline evaluation of `{report.ranker}` against the deterministic ranking "
        f"benchmark (version {report.benchmark_version}). No network access, no model "
        "calls, no live APIs: every case is a fixture and every number below is "
        "reproducible from the committed corpus."
    )
    add("")

    # -- run configuration ------------------------------------------------
    add("## Run configuration")
    add("")
    add(f"- benchmark cases: **{report.case_count}**")
    add(f"- graded candidates: **{report.graded_candidate_count}**")
    add(f"- evaluation clock: `{BENCHMARK_NOW.isoformat()}` (fixed, so recency is stable)")
    add(f"- ranker: `{report.ranker}`")
    add("- weights:")
    for name in sorted(dir(settings)):
        if name.startswith("ranking_weight_"):
            add(f"  - `{name}` = {getattr(settings, name):g}")
    add("")

    # -- headline metrics --------------------------------------------------
    add("## 1. Ranking quality")
    add("")
    add("Macro-averaged over cases. Relevance cut is grade >= 2; NDCG uses the full grade scale.")
    add("")
    add("| metric | value |")
    add("| --- | ---: |")
    for name in sorted(report.metrics):
        if name.startswith("pool_") or name == "relevant_in_pool":
            continue
        add(f"| `{name}` | {_fmt(report.metrics[name])} |")
    add("")

    grade_counts = _grade_summary(report)
    add("Graded candidate distribution: " + ", ".join(
        f"grade {g}: {n}" for g, n in sorted(grade_counts.items(), reverse=True)
    ))
    add("")

    add("### Retrieval recall vs ranking quality")
    add("")
    add(
        f"- **retrieval recall** (pool contained a grade-3 dataset): "
        f"{_fmt(report.retrieval_recall)}"
    )
    add(
        f"- **ranking quality on answered cases** (NDCG@5): "
        f"{_fmt(report.ranking_quality.get('ndcg@5'))}"
    )
    add(
        f"- **ranking quality across all cases** (NDCG@5): {_fmt(report.metrics.get('ndcg@5'))}"
    )
    add("")
    add(
        "The gap between these two numbers is the part of the score that query "
        "construction, not ranking, is responsible for."
    )
    add("")

    # -- diagnostics ------------------------------------------------------
    add("## 2. Diagnostic rates")
    add("")
    add("| diagnostic | rate |")
    add("| --- | ---: |")
    labels = {
        "incompatible_in_top3": "clearly incompatible datasets in the top 3 (lower is better)",
        "unknown_treated_as_mismatch": "requirement verdicts read as MISMATCH without evidence",
        "no_grade3_at_rank1": "cases whose rank 1 is not a grade-3 dataset",
        "no_grade3_in_top3": "cases with no grade-3 dataset in the top 3",
    }
    for name in sorted(report.diagnostics):
        add(f"| {labels.get(name, name)} | {_fmt(report.diagnostics[name])} |")
    add("")

    # -- per-category breakdown -------------------------------------------
    add("## 3. Quality by category")
    add("")
    add("| category | cases | P@1 | NDCG@5 | hit@3 |")
    add("| --- | ---: | ---: | ---: | ---: |")
    categories = sorted({case.category for case in report.per_case})
    for category in categories:
        subset = [c for c in report.per_case if c.category == category]
        n = len(subset)
        p1 = sum(c.metrics.get("precision@1", 0.0) for c in subset) / n
        n5 = sum(c.metrics.get("ndcg@5", 0.0) for c in subset) / n
        h3 = sum(c.metrics.get("hit_rate@3", 0.0) for c in subset) / n
        add(f"| `{category}` | {n} | {_fmt(p1)} | {_fmt(n5)} | {_fmt(h3)} |")
    add("")

    # -- failures ---------------------------------------------------------
    add("## 4. Failure taxonomy")
    add("")
    if report.failure_taxonomy:
        add("| category | cases |")
        add("| --- | ---: |")
        for name, count in sorted(report.failure_taxonomy.items(), key=lambda kv: -kv[1]):
            add(f"| `{name}` | {count} |")
    else:
        add("No failure category fired on this run.")
    add("")
    add("Retrieval vs ranking: " + ", ".join(
        f"`{k}` {v}" for k, v in sorted(report.failure_kind_counts.items())
    ))
    add("")

    add("### Failing cases")
    add("")
    failing = [c for c in report.per_case if c.failures]
    if not failing:
        add("No case produced a failure finding.")
    for case in sorted(failing, key=lambda c: c.case_id):
        categories = ", ".join(sorted({f.category.value for f in case.failures}))
        add(f"- `{case.case_id}` [{case.category}] - {categories}")
    add("")

    add("## 5. Factor influence")
    add("")
    influence = factor_influence_over_benchmark(report)
    add("Average share of a final score, per factor, across every ranked candidate.")
    add("")
    add("| factor | average share of score |")
    add("| --- | ---: |")
    for name, share in sorted(influence.items(), key=lambda kv: -kv[1]):
        add(f"| `{name}` | {_fmt(share)} |")
    add("")
    if report.audit is not None:
        add("### Score decomposition audit")
        add("")
        audit = report.audit
        add(
            f"- contributions sum to the displayed score: "
            f"**{audit.contributions_sum_matches_score}** "
            f"(largest error {audit.max_abs_contribution_error})"
        )
        add(f"- arithmetic errors: {audit.impossible_contributions}")
        add(f"- duplicate factors: {audit.duplicated_contributions}")
        add(f"- factors exceeding their nominal share: {audit.inflated_contributions}")
        add("")
        for finding in audit.findings:
            add(f"- **{finding.severity}** `{finding.subject}`: {finding.detail}")
        add("")
        add("| factor | meaning | weight | nominal share | observed max share | unknown? | mismatch? |")
        add("| --- | --- | ---: | ---: | ---: | :---: | :---: |")
        for entry in audit.factors:
            add(
                f"| `{entry.name}` | {entry.semantic_meaning} | {entry.weight:g} | "
                f"{entry.nominal_share:.2f} | {entry.observed_max_share:.2f} | "
                f"{'yes' if entry.unknown_possible else 'no'} | "
                f"{'yes' if entry.mismatch_possible else 'no'} |"
            )
        add("")

    # -- ablation ---------------------------------------------------------
    add("## 6. Ablation")
    add("")
    add(
        "Each signal is removed by setting its weight to zero and re-running the whole "
        "benchmark. Deltas are variant minus baseline; negative means the baseline was "
        "better, i.e. the removed signal was earning its place."
    )
    add("")
    add(render_ablation_table(report, ablations))
    add("")

    # -- sensitivity ------------------------------------------------------
    add("## 7. Sensitivity")
    add("")
    add(
        "One input is perturbed at a time on a fixed sample of cases and the engine is "
        "re-run end to end. Measures what each piece of evidence is actually worth in "
        "ranking terms."
    )
    add("")
    add(render_sensitivity_table(summarise_sensitivity(sensitivity)))
    add("")

    # -- per-case detail --------------------------------------------------
    add("## 8. Per-case diagnostics")
    add("")
    add(
        "Every case, its ranking, and the factors that produced each position. "
        "`evidence/unknown/publisher` splits each candidate's score into the share "
        "driven by declared evidence, by unjudgeable factors, and by publisher signals."
    )
    add("")
    for case in report.per_case:
        add(render_case_report(_compiled_for(case, report), case.ranked))

    add("## 9. Limitations")
    add("")
    for limitation in _limitations(report):
        add(f"- {limitation}")
    add("")

    return "\n".join(lines)


def _compiled_for(case, report: EvaluationReport):
    """Rebuild just enough of a compiled case for the diagnostic renderer."""
    from app.services.datasets.evaluation.benchmark import load_benchmark
    from app.services.datasets.evaluation.fixtures import compile_case
    from app.services.datasets.evaluation.models import BenchmarkCase

    for candidate in load_benchmark():
        if candidate.id == case.case_id:
            return compile_case(BenchmarkCase(**candidate.model_dump()))
    raise KeyError(case.case_id)


def _limitations(report: EvaluationReport) -> list[str]:
    """Stated limits of this benchmark, generated from what it actually contains."""
    limits = [
        "The candidate pools are fixtures, not live registry results. They are built "
        "to exercise the ranker, so absolute scores describe this corpus and not "
        "Kaggle at large.",
        "Retrieval quality is measured only as 'did the pool contain the right "
        "answer'. Real query performance against a live registry is not represented.",
        "Only "
        f"{sum(1 for c in report.per_case for i in c.ranked if i.verification_status.value != 'verified')}"
        " candidates are unverified, so the verification ablation is weakly "
        "identified: a null result there means 'not enough unverified candidates to "
        "flip a ranking', not 'verification does not matter'.",
    ]
    if report.audit and report.audit.inflated_contributions:
        limits.append(
            "Factor contributions are point shares of a weighted mean, so a factor "
            "whose peers were dropped from the denominator exceeds its nominal "
            "weight. Sparse-metadata candidates are flattered on the dimensions "
            "they do publish; the audit table quantifies it per factor."
        )
    limits.append(
        "Grades are human judgements made by one author against written criteria. "
        "They are defensible, not objective, and the criteria are in "
        "`labels.py` so a disagreement can be resolved rather than argued about."
    )
    return limits


def write_reports(
    report: EvaluationReport,
    settings: Settings,
    ablations: Sequence[AblationResult],
    sensitivity: Sequence[SensitivityScenario],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write both report artefacts. Returns ``(json_path, markdown_path)``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_NAME
    markdown_path = output_dir / REPORT_MARKDOWN_NAME

    payload = report.model_dump(mode="json")
    payload["ablations"] = [a.model_dump(mode="json") for a in ablations]
    payload["sensitivity"] = [s.model_dump(mode="json") for s in sensitivity]
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_report(report, settings, ablations, sensitivity), encoding="utf-8"
    )
    return json_path, markdown_path