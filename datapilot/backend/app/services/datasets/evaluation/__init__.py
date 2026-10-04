"""Offline ranking evaluation framework.

A measurement instrument for the dataset ranker, not a test of it. Nothing in this
package modifies ranking behaviour, adjusts a weight, or reaches the network.

Typical use::

    from app.core.config import Settings
    from app.services.datasets.ranking import DatasetRankingEngine
    from app.services.datasets.evaluation import evaluate_ranker

    report = evaluate_ranker(DatasetRankingEngine(Settings()), load_benchmark())
    print(report.metrics["ndcg@5"])

For a future ranker that is not the current engine, pass anything with a
``rank(candidates, requirements)`` method; the benchmark, metrics, diagnostics and
failure taxonomy are unchanged.
"""

from app.services.datasets.evaluation.ablation import (
    ABLATION_TARGETS,
    run_ablation,
)
from app.services.datasets.evaluation.audit import audit_scores
from app.services.datasets.evaluation.benchmark import load_benchmark
from app.services.datasets.evaluation.diagnostics import (
    evidence_quality,
    render_case_report,
)
from app.services.datasets.evaluation.fixtures import BENCHMARK_NOW, BENCHMARK_VERSION, compile_case
from app.services.datasets.evaluation.golden_set import (
    GOLDEN_CASES,
    check_golden_case,
    load_golden_cases,
    run_golden_set,
)
from app.services.datasets.evaluation.labels import GRADE_DEFINITIONS, RelevanceGrade
from app.services.datasets.evaluation.metrics import (
    aggregate_metrics,
    case_metrics,
    incompatible_top_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.datasets.evaluation.models import (
    AblationResult,
    AuditReport,
    BenchmarkCase,
    CandidateSpec,
    CaseEvaluation,
    CompiledCase,
    Difficulty,
    EvaluationReport,
    FactorContribution,
    FailureCategory,
    FailureKind,
    RankedItem,
    RequirementSpec,
    SensitivityScenario,
)
from app.services.datasets.evaluation.properties import (
    property_contributions_sum_to_score,
    property_determinism,
    property_incompatible_not_above_relevant,
    property_input_order_independence,
    property_no_factor_duplication,
    property_publisher_factors_absent_when_unpublished,
    property_requirement_factors_within_bounds,
    property_unknown_is_not_mismatch,
    property_unverified_not_penalised_for_missing_evidence,
    property_within_weight_bounds,
)
from app.services.datasets.evaluation.report import write_reports
from app.services.datasets.evaluation.runner import evaluate_case, evaluate_ranker
from app.services.datasets.evaluation.sensitivity import run_sensitivity, summarise_sensitivity
from app.services.datasets.evaluation.taxonomy import classify_failures

__all__ = [
    "ABLATION_TARGETS",
    "BENCHMARK_NOW",
    "BENCHMARK_VERSION",
    "GRADE_DEFINITIONS",
    "GOLDEN_CASES",
    "AblationResult",
    "AuditReport",
    "BenchmarkCase",
    "CandidateSpec",
    "CaseEvaluation",
    "CompiledCase",
    "Difficulty",
    "EvaluationReport",
    "FactorContribution",
    "FailureCategory",
    "FailureKind",
    "RankedItem",
    "RelevanceGrade",
    "RequirementSpec",
    "SensitivityScenario",
    "aggregate_metrics",
    "audit_scores",
    "case_metrics",
    "check_golden_case",
    "classify_failures",
    "compile_case",
    "evaluate_case",
    "evaluate_ranker",
    "evidence_quality",
    "incompatible_top_rate",
    "load_benchmark",
    "load_golden_cases",
    "ndcg_at_k",
    "precision_at_k",
    "property_contributions_sum_to_score",
    "property_determinism",
    "property_incompatible_not_above_relevant",
    "property_input_order_independence",
    "property_no_factor_duplication",
    "property_publisher_factors_absent_when_unpublished",
    "property_requirement_factors_within_bounds",
    "property_unknown_is_not_mismatch",
    "property_unverified_not_penalised_for_missing_evidence",
    "property_within_weight_bounds",
    "recall_at_k",
    "reciprocal_rank",
    "render_case_report",
    "run_ablation",
    "run_golden_set",
    "run_sensitivity",
    "summarise_sensitivity",
    "write_reports",
]