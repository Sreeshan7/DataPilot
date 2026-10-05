"""Phase 5 benchmark: project intelligence against what the generators actually planted.

Ground truth comes from how each synthetic dataset was built (which columns feed the target, which are
noise, which leak it), written down independently of the analyzer's output. Measured with the
production importance settings. Deterministic and offline; the numbers describe this project's own
synthetic data, not real-world accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.schemas.project_intelligence import CompatibilityStatus, FeatureRole
from app.services.dataset_analysis.evaluation.benchmark import analyze_bytes
from app.services.dataset_analysis.evaluation.datasets import GENERATORS, HELDOUT_GENERATORS, wide_numeric
from app.services.dataset_analysis.service import AnalysisOutcome
from app.services.project_intelligence import ProjectIntelligenceService
from app.services.project_intelligence.shape import shape_of

import asyncio

_WIDE = {f"f{j:03d}" for j in range(120)}


@dataclass(frozen=True)
class FeatureTruth:
    drivers: frozenset[str]          # genuinely feed the target
    noise: frozenset[str]            # planted as unrelated
    must_find: frozenset[str] = frozenset()   # strong enough that missing it is a failure
    ambiguous: frozenset[str] = frozenset()   # derived from a driver: either label is defensible
    leaks: frozenset[str] = frozenset()
    identifiers: frozenset[str] = frozenset()


def _f(*names: str) -> frozenset[str]:
    return frozenset(names)


TRUTH: dict[str, FeatureTruth] = {
    "student_performance": FeatureTruth(_f("previous_marks", "study_hours", "attendance", "assignment_scores", "participation"),
                                        _f("gender"), _f("previous_marks", "study_hours"), identifiers=_f("student_id")),
    "student_performance_leaky": FeatureTruth(_f("previous_marks", "study_hours", "attendance", "assignment_scores", "participation"),
                                              _f("gender"), _f("previous_marks"), leaks=_f("final_exam_score"),
                                              identifiers=_f("student_id")),
    "customer_churn": FeatureTruth(_f("tenure_months", "monthly_charges", "contract", "support_calls"), _f(), _f("contract"),
                                   identifiers=_f("customer_id", "email")),
    "house_prices": FeatureTruth(_f("sqft", "bedrooms", "bathrooms", "year_built", "neighborhood"), _f(), _f("sqft"),
                                 identifiers=_f("house_id")),
    "diabetes": FeatureTruth(_f("age", "bmi", "glucose"), _f("blood_pressure", "insulin"), _f("glucose"),
                             identifiers=_f("patient_id")),
    "bank_marketing": FeatureTruth(_f("balance", "job", "housing"), _f("age", "marital", "education", "loan")),
    "fraud_transactions": FeatureTruth(_f("amount"), _f("hour", "distance_km"), _f("amount"), identifiers=_f("transaction_id")),
    "iris_like": FeatureTruth(_f("sepal_length", "sepal_width", "petal_length", "petal_width"), _f(), _f("petal_length")),
    "employee_attrition": FeatureTruth(_f("overtime", "years_at_company", "job_satisfaction"),
                                       _f("department", "job_role", "marital_status"),
                                       ambiguous=_f("monthly_income"), identifiers=_f("employee_number", "ssn", "phone_number")),
}


@dataclass(frozen=True)
class GoalCase:
    dataset: str
    goal: str
    expected: CompatibilityStatus


M, U, X = CompatibilityStatus.MATCH, CompatibilityStatus.UNKNOWN, CompatibilityStatus.MISMATCH
GOAL_CASES: tuple[GoalCase, ...] = (
    GoalCase("customer_churn", "Predict customer churn using tenure, contract and support calls", M),
    GoalCase("student_performance", "Predict student academic performance using attendance, previous marks and study hours", M),
    GoalCase("diabetes", "Predict whether a patient has diabetes from BMI, glucose and age", M),
    GoalCase("sales_timeseries", "Forecast daily units sold using promotions", M),
    GoalCase("house_prices", "Estimate the price of a house using square footage and bedrooms", M),
    GoalCase("fraud_transactions", "Detect fraud using transaction amount and hour", M),
    GoalCase("house_prices", "Classify whether a patient has diabetes using BMI and glucose", X),
    GoalCase("customer_churn", "Predict hospital readmission using age and diagnosis", X),
    GoalCase("customer_churn", "Forecast monthly revenue over time", X),
    GoalCase("diabetes", "Predict customer churn using tenure and contract", X),
    GoalCase("tiny_table", "Predict label using a and b", X),
    GoalCase("customer_churn", "I want to learn machine learning", U),
    GoalCase("customer_churn", "Predict customer churn using tenure, contract and favourite colour", U),
    GoalCase("student_performance", "Explore the data", U),
)


class DatasetFeatureResult(BaseModel):
    name: str
    reliable: Optional[bool] = None
    recommended: list[str] = Field(default_factory=list)
    noise_recommended: list[str] = Field(default_factory=list)
    drivers_missed: list[str] = Field(default_factory=list)
    leaks_missed: list[str] = Field(default_factory=list)
    leaks_recommended: list[str] = Field(default_factory=list)
    identifiers_missed: list[str] = Field(default_factory=list)
    ideas: int = 0
    invented_columns: list[str] = Field(default_factory=list)


class ProjectBenchmarkReport(BaseModel):
    metrics: dict[str, float] = Field(default_factory=dict)
    datasets: list[DatasetFeatureResult] = Field(default_factory=list)
    goal_misses: list[str] = Field(default_factory=list)
    safety_violations: list[str] = Field(default_factory=list)


def _ratio(n: int, d: int) -> float:
    return round(n / d, 4) if d else 1.0


def benchmark_settings() -> Settings:
    return Settings(gemini_api_key="", kaggle_api_token="", app_env="testing", external_enabled=False)


def run_project_benchmark() -> ProjectBenchmarkReport:
    settings = benchmark_settings()
    service = ProjectIntelligenceService(settings)
    report = ProjectBenchmarkReport()
    outcomes: dict[str, AnalysisOutcome] = {}
    sources = {**GENERATORS, **HELDOUT_GENERATORS, "wide_numeric": wide_numeric}

    def outcome(name: str) -> AnalysisOutcome:
        if name not in outcomes:
            dataset = sources[name]()
            from app.services.dataset_analysis.service import AnalysisRequest, DatasetAnalysisService
            import io
            outcomes[name] = asyncio.run(DatasetAnalysisService(settings).analyze(
                AnalysisRequest(stream=io.BytesIO(dataset.csv_bytes()), file_name=f"{name}.csv")))
        return outcomes[name]

    truths = dict(TRUTH)
    truths["wide_numeric"] = FeatureTruth(_f("f003", "f017"), frozenset(_WIDE - {"f003", "f017"}), _f("f003"))

    for name, truth in truths.items():
        out = outcome(name)
        pi = asyncio.run(service.build(out))
        roles = {f.column: f for f in pi.feature_intelligence.features}
        rec = sorted(c for c, f in roles.items() if f.role is FeatureRole.RECOMMENDED_FEATURE)
        columns = shape_of(out.analysis).names
        invented: set[str] = set()
        for idea in pi.project_ideas:
            for column in set(idea.required_features) | set(idea.optional_features) | ({idea.target} - {None}):
                if column not in columns:
                    invented.add(column)
            text = " ".join([idea.title, idea.problem_statement, idea.why_it_fits, *idea.challenges])
            invented |= {s for s in re.findall(r"`([^`]*)`", text) if s not in columns}
        result = DatasetFeatureResult(
            name=name, reliable=pi.feature_intelligence.method.reliable if pi.feature_intelligence.method else None,
            recommended=rec,
            noise_recommended=[c for c in rec if c in truth.noise],
            drivers_missed=sorted(c for c in truth.must_find
                                  if roles.get(c) is None or roles[c].role is not FeatureRole.RECOMMENDED_FEATURE),
            leaks_missed=sorted(c for c in truth.leaks if roles.get(c) is None or roles[c].role is not FeatureRole.POSSIBLE_LEAKAGE),
            leaks_recommended=[c for c in rec if c in truth.leaks],
            identifiers_missed=sorted(c for c in truth.identifiers if roles.get(c) is None or roles[c].role is not FeatureRole.IDENTIFIER),
            ideas=len(pi.project_ideas), invented_columns=sorted(invented))
        report.datasets.append(result)

    # every other dataset also gets its ideas checked for invented columns
    extra_invented = 0
    for name in sorted(set(sources) - set(truths)):
        out = outcome(name)
        pi = asyncio.run(service.build(out))
        columns = shape_of(out.analysis).names
        for idea in pi.project_ideas:
            extra_invented += len((set(idea.required_features) | set(idea.optional_features) | ({idea.target} - {None})) - columns)

    results = report.datasets
    recommended_total = sum(len(r.recommended) for r in results)
    noise_rec = sum(len(r.noise_recommended) for r in results)
    must_total = sum(len(t.must_find) for n, t in truths.items())
    leak_total = sum(len(t.leaks) for t in truths.values())
    id_total = sum(len(t.identifiers) for t in truths.values())
    report.metrics = {
        "datasets": float(len(results)),
        "recommended_precision": _ratio(recommended_total - noise_rec - sum(
            1 for n, r in zip(truths, results) for c in r.recommended
            if c not in truths[n].drivers and c not in truths[n].ambiguous and c not in truths[n].noise), recommended_total),
        "noise_recommended": float(noise_rec),
        "strong_driver_recall": _ratio(must_total - sum(len(r.drivers_missed) for r in results), must_total),
        "leak_detection_recall": _ratio(leak_total - sum(len(r.leaks_missed) for r in results), leak_total),
        "leaks_recommended": float(sum(len(r.leaks_recommended) for r in results)),
        "identifier_recall": _ratio(id_total - sum(len(r.identifiers_missed) for r in results), id_total),
        "invented_columns": float(sum(len(r.invented_columns) for r in results) + extra_invented),
    }

    # ---- goal compatibility ------------------------------------------------------------
    right = 0
    wrong_match = wrong_mismatch = 0
    for case in GOAL_CASES:
        pi = asyncio.run(service.build(outcome(case.dataset), goal=case.goal))
        got = pi.compatibility.status
        if got is case.expected:
            right += 1
        else:
            report.goal_misses.append(f"{case.dataset}: '{case.goal}' expected {case.expected.value} got {got.value}")
            # the two errors that matter: calling a bad fit good, and rejecting a good fit
            wrong_match += got is M and case.expected is not M
            wrong_mismatch += got is X and case.expected is not X
    report.metrics["goal_cases"] = float(len(GOAL_CASES))
    report.metrics["goal_accuracy"] = _ratio(right, len(GOAL_CASES))
    report.metrics["goal_false_match"] = float(wrong_match)
    report.metrics["goal_false_mismatch"] = float(wrong_mismatch)

    for r in results:
        if r.invented_columns:
            report.safety_violations.append(f"{r.name}: invented columns {r.invented_columns}")
        if r.leaks_recommended:
            report.safety_violations.append(f"{r.name}: a leaking column was recommended {r.leaks_recommended}")
    if extra_invented:
        report.safety_violations.append("invented columns in the remaining datasets")
    if wrong_match:
        report.safety_violations.append("a goal was matched to a dataset that contradicts it")
    return report
