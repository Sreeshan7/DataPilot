"""Run the offline ranking evaluation and write both report artefacts.

    python -m scripts.run_evaluation            # full run, writes into backend/evaluation
    python -m scripts.run_evaluation --quick    # skip ablation and sensitivity

No network access, no API keys, no live registry calls. The run is deterministic:
the only timestamps come from a fixed clock inside the benchmark, so two runs of
this script on the same code produce the same metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.datasets.evaluation.ablation import run_ablation  # noqa: E402
from app.services.datasets.evaluation.audit import audit_scores  # noqa: E402
from app.services.datasets.evaluation.benchmark import load_benchmark  # noqa: E402
from app.services.datasets.evaluation.golden_set import run_golden_set  # noqa: E402
from app.services.datasets.evaluation.report import write_reports  # noqa: E402
from app.services.datasets.evaluation.runner import evaluate_ranker  # noqa: E402
from app.services.datasets.evaluation.sensitivity import run_sensitivity  # noqa: E402
from app.services.datasets.ranking import DatasetRankingEngine  # noqa: E402

DEFAULT_OUTPUT = BACKEND_ROOT / "evaluation"


def _offline_settings() -> Settings:
    """Settings with every credential blanked, so a stray key cannot cause traffic."""
    return Settings(
        gemini_api_key="",
        kaggle_api_token="",
        kaggle_enabled=False,
        dataset_discovery_genai_enabled=False,
        app_env="testing",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory for evaluation_results.json and RANKING_EVALUATION.md",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip ablation and sensitivity (report is still complete in metrics)",
    )
    args = parser.parse_args(argv)

    settings = _offline_settings()
    cases = load_benchmark()

    def factory(candidate_settings: Settings) -> DatasetRankingEngine:
        return DatasetRankingEngine(candidate_settings)

    print(f"evaluating {len(cases)} benchmark cases...")
    report = evaluate_ranker(factory(settings), cases)
    report.audit = audit_scores(report.per_case, settings)

    ablations = []
    sensitivity = []
    if not args.quick:
        print("running ablation...")
        ablations = run_ablation(factory, settings, cases, report)
        print("running sensitivity...")
        sensitivity = run_sensitivity(factory, settings, cases, max_cases=8)

    print("checking golden set...")
    golden_problems = run_golden_set(factory(settings).rank)
    if golden_problems:
        print("golden set violations:")
        for problem in golden_problems:
            print(f"  - {problem}")

    json_path, markdown_path = write_reports(
        report, settings, ablations, sensitivity, args.output
    )

    print()
    print(f"cases            {report.case_count}")
    print(f"graded candidates{report.graded_candidate_count}")
    print()
    # Two different populations, and conflating them is the easiest way to
    # misreport this benchmark. `metrics` covers every case, including the
    # deliberate edge probes (all-incompatible, retrieval-miss, all-UNKNOWN), so
    # it is the honest headline but is dragged down by cases no ranker can win.
    # `ranking_quality` covers only cases with a grade-3 candidate, so it answers
    # "given the right dataset was retrieved, is it ranked first?".
    print("all cases (incl. edge probes)")
    print(f"  ndcg@5         {report.metrics['ndcg@5']:.4f}")
    print(f"  ndcg@10        {report.metrics['ndcg@10']:.4f}")
    print(f"  precision@1    {report.metrics['precision@1']:.4f}")
    print(f"  mrr            {report.metrics['mrr']:.4f}")
    print(f"  hit_rate@3     {report.metrics['hit_rate@3']:.4f}")
    print()
    print("graded cases only (a correct answer exists)")
    print(f"  ndcg@5         {report.ranking_quality['ndcg@5']:.4f}")
    print(f"  precision@1    {report.ranking_quality['precision@1']:.4f}")
    print(f"  mrr            {report.ranking_quality['mrr']:.4f}")
    print(f"  mrr@grade3     {report.ranking_quality['mrr_highly_relevant']:.4f}")
    print()
    print(f"retrieval recall {report.retrieval_recall:.4f}")
    print(
        f"incompatible@3  {report.diagnostics['incompatible_in_top3']:.4f}"
    )
    print(
        f"unknown=>mismatch {report.diagnostics['unknown_treated_as_mismatch']:.4f}"
    )
    print(f"audit errors    {report.audit.impossible_contributions}")
    print(f"audit warnings  {len(report.audit.findings)}")
    print()
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 1 if golden_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())