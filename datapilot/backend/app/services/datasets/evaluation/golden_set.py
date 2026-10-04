"""The permanent golden set: 10 cases whose expected top result is pinned.

The benchmark measures quality; the golden set prevents *regressions*. It is
deliberately small, deliberately boring, and deliberately about behaviours that
must never break:

* the correct answer wins;
* an incompatible dataset never reaches the top;
* sparse metadata does not become a mismatch;
* an AI rationale is never treated as evidence;
* declared contradictions are penalised;
* rankings are deterministic and order-independent.

Each expectation states the intended behaviour rather than a frozen score, so a
legitimate re-weighting does not require rewriting the set, while a behavioural
regression cannot pass by nudging a number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from app.schemas.dataset import DatasetCandidate
from app.services.datasets.evaluation.benchmark import load_benchmark
from app.services.datasets.evaluation.fixtures import compile_case
from app.services.datasets.evaluation.labels import RelevanceGrade


@dataclass(frozen=True)
class GoldenCase:
    """One pinned expectation."""

    case_id: str
    #: Candidate key that must be first. ``None`` means "no particular candidate
    #: must be first", used when the point of the case is a property of the list
    #: rather than its head.
    expect_top_key: Optional[str]
    #: Candidate keys that must not appear in the top ``forbidden_top_n``.
    forbidden_keys: tuple[str, ...] = ()
    forbidden_top_n: int = 3
    why: str = ""


GOLDEN_CASES: tuple[GoldenCase, ...] = (
    GoldenCase(
        case_id="edu-student-performance",
        expect_top_key="full-cohort",
        why="the dataset that publishes every required feature and target wins",
    ),
    GoldenCase(
        case_id="finance-churn",
        expect_top_key=None,
        forbidden_keys=("generic-customer", "churn-retail"),
        forbidden_top_n=2,
        why="only the two datasets with a real churn label may occupy the top 2",
    ),
    GoldenCase(
        case_id="health-diabetes-bmi",
        expect_top_key="pima-diabetes",
        why="clinical diabetes data outranks insurance charges and review sentiment",
    ),
    GoldenCase(
        case_id="health-target-conflict",
        expect_top_key=None,
        forbidden_keys=("declared-readmission",),
        forbidden_top_n=1,
        why="a declared mortality/readmission conflict keeps the wrong label out of first place",
    ),
    GoldenCase(
        case_id="cyber-iot-intrusion",
        expect_top_key="bot-iot",
        forbidden_keys=("fifa-gameplay",),
        why="real IoT intrusion data beats energy traces and unrelated game traffic",
    ),
    GoldenCase(
        case_id="health-readmission-sparse",
        expect_top_key=None,
        forbidden_keys=("generic-readmission-hints",),
        forbidden_top_n=1,
        why="a generic notes collection must not displace the actual encounter extract",
    ),
    GoldenCase(
        case_id="finance-ai-inferred-target",
        expect_top_key="profit-margin",
        forbidden_keys=("ai-revenue",),
        forbidden_top_n=1,
        why="an AI suggestion claiming the wrong target in prose is not evidence",
    ),
    GoldenCase(
        case_id="edge-ai-rationale-trap",
        expect_top_key="declared-pm25",
        forbidden_keys=("ai-pm25-rationale",),
        forbidden_top_n=1,
        why="declared metadata beats AI rationale text on the same case",
    ),
    GoldenCase(
        case_id="nlp-multilingual-sentiment",
        expect_top_key="multilingual-reviews",
        why="the multilingual sentiment corpus outranks the larger single-language sets",
    ),
    GoldenCase(
        case_id="retail-demand-forecast",
        expect_top_key="walmart-sales",
        forbidden_keys=("fashion-ecommerce",),
        forbidden_top_n=2,
        why="per-SKU weekly demand with price and promotion is the right answer",
    ),
)


@dataclass(frozen=True)
class KnownFailure:
    """A golden expectation that is correct but not yet met.

    Kept as an explicit, named list rather than by loosening the expectation,
    for three reasons: the suite stays green so a *new* regression is loud; the gap
    stays permanently visible instead of being quietly accepted; and whoever fixes
    it has to delete a line that says what the defect was, which is the moment to
    decide whether the fix was right.

    An entry is a debt with an owner, not a passing test.
    """

    case_id: str
    description: str
    #: Why the current behaviour is wrong, in mechanism terms.
    mechanism: str
    #: What would have to change for this to pass.
    fix_requires: str
    #: Benchmark case that measures the same defect, so progress is visible.
    tracked_by: str


KNOWN_FAILURES: tuple[KnownFailure, ...] = (
    KnownFailure(
        case_id="health-target-conflict",
        description=(
            "a dataset declaring a readmission outcome ranks first for a project "
            "that needs mortality"
        ),
        mechanism=(
            "the target-concept dictionary has no label space for 'mortality', so "
            "the project side resolves to nothing. contradiction detection is "
            "deliberately silent when either side is unknown, so the declared "
            "readmission label is never compared against the project's target"
        ),
        fix_requires=(
            "dictionary coverage: a mortality / in-hospital death label space, or a "
            "documented decision that an unresolved project target disables target "
            "evidence on both sides rather than scoring it as a match"
        ),
        tracked_by="health-target-conflict case; taxonomy category WRONG_TARGET",
    ),
    KnownFailure(
        case_id="health-readmission-sparse",
        description=(
            "a generic care-planning notes collection can outrank the actual "
            "encounter extract"
        ),
        mechanism=(
            "the right dataset publishes only prose, so feature and target evidence "
            "is UNKNOWN and earns no credit, while the wrong dataset declares a "
            "task type and is credited for it. keyword_overlap cannot separate them "
            "because both are healthcare notes"
        ),
        fix_requires=(
            "a decision on whether declaring a task type without declaring any "
            "evidence about the actual data should earn task_match credit"
        ),
        tracked_by="health-readmission-sparse case; taxonomy category FEATURE_UNKNOWN",
    ),
)


def known_failure_for(case_id: str) -> Optional[KnownFailure]:
    return next((f for f in KNOWN_FAILURES if f.case_id == case_id), None)


def golden_case_ids() -> tuple[str, ...]:
    return tuple(g.case_id for g in GOLDEN_CASES)


def load_golden_cases() -> list:
    """The benchmark cases referenced by the golden set, in golden order."""
    index = {case.id: case for case in load_benchmark()}
    missing = [case_id for case_id in golden_case_ids() if case_id not in index]
    if missing:
        raise KeyError(f"golden set references missing benchmark cases: {missing}")
    return [index[case_id] for case_id in golden_case_ids()]


def check_golden_case(
    case,
    ranked: Sequence[DatasetCandidate],
    *,
    enforce_known_failures: bool = False,
) -> list[str]:
    """Return a list of violated expectations; empty means the case passed.

    Expectations listed in :data:`KNOWN_FAILURES` are reported but not counted as
    violations unless ``enforce_known_failures`` is set. That flag exists for the
    benchmark run, which wants the full picture, and for whoever fixes a known
    failure: setting it turns a listed failure back into a hard failure.
    """
    compiled = compile_case(case)
    by_key = {
        spec.key: candidate
        for spec, candidate in zip(case.candidates, compiled.candidates)
    }
    expectations = next(g for g in GOLDEN_CASES if g.case_id == case.id)
    known = known_failure_for(case.id)
    problems: list[str] = []

    if not ranked:
        return [f"{case.id}: ranking produced no candidates"]

    if expectations.expect_top_key is not None:
        top = ranked[0]
        expected_id = by_key[expectations.expect_top_key].id
        if top.id != expected_id:
            expected_grade = RelevanceGrade(case.grades[expectations.expect_top_key])
            actual_grade = next(
                case.grades[spec.key]
                for spec in case.candidates
                if by_key[spec.key].id == top.id
            )
            problems.append(
                f"{case.id}: expected '{expectations.expect_top_key}' "
                f"(grade {int(expected_grade)}) first, got '{top.name}' "
                f"(grade {actual_grade}) - {expectations.why}"
            )

    for key in expectations.forbidden_keys:
        forbidden_id = by_key[key].id
        for position, candidate in enumerate(ranked[: expectations.forbidden_top_n], start=1):
            if candidate.id == forbidden_id:
                problems.append(
                    f"{case.id}: '{key}' must not be in the top "
                    f"{expectations.forbidden_top_n}, found at rank {position} - "
                    f"{expectations.why}"
                )
                break

    if problems and known is not None and not enforce_known_failures:
        return []
    return problems


__all__ = [
    "GOLDEN_CASES",
    "KNOWN_FAILURES",
    "GoldenCase",
    "KnownFailure",
    "check_golden_case",
    "golden_case_ids",
    "known_failure_for",
    "load_golden_cases",
    "run_golden_set",
]


def run_golden_set(rank: Callable, enforce_known_failures: bool = False) -> list[str]:
    """Run ``rank`` over every golden case and collect all violated expectations.

    :param rank: any callable taking ``(candidates, requirements)``.
    """
    problems: list[str] = []
    for case in load_golden_cases():
        compiled = compile_case(case)
        ranked = rank(list(compiled.candidates), compiled.requirements)
        problems.extend(
            check_golden_case(
                case, ranked, enforce_known_failures=enforce_known_failures
            )
        )
    return problems