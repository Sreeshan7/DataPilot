"""Builders that turn benchmark specs into the schemas the ranker consumes.

The benchmark is authored as data, but ranking only ever sees
``ProjectAnalysisData`` and ``DatasetCandidate``. Building those two schemas here
- rather than in the tests' conftest - is what makes the benchmark runnable from
a plain script and from tests with identical inputs, so a metric measured in CI
is the same metric a report was generated from.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.schemas.dataset import DatasetCandidate, LLMDiscoveredCandidate
from app.schemas.project import (
    Analysis,
    DataTypeRequirement,
    DatasetRequirements,
    DomainInfo,
    FeatureRequirement,
    InputInfo,
    KeywordMatchEvidence,
    Keywords,
    MLTask,
    MatchType,
    ProblemTypeInfo,
    ProjectAnalysisData,
    ProjectUnderstanding,
    SubdomainInfo,
    TargetInfo,
    TargetRequirement,
)
from app.services.datasets.evaluation.models import (
    BenchmarkCase,
    CandidateSpec,
    CompiledCase,
    RequirementSpec,
)

#: Version stamped into every report so a stale number is never mistaken for a
#: current one.
BENCHMARK_VERSION = "1.0.0"

#: Fixed clock. Recency is a ranking factor, so an evaluation that used the wall
#: clock would not be reproducible; every timestamp in the corpus is expressed
#: relative to this instant.
BENCHMARK_NOW = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse a benchmark timestamp, rejecting anything not UTC-anchored."""
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_requirements(
    spec: RequirementSpec,
    *,
    objective: str = "Offline ranking benchmark scenario",
) -> ProjectAnalysisData:
    """Build the analyzer output a benchmark case describes.

    Mirrors what ``make_requirements`` does in the test suite: canonical keywords
    go to ``Keywords.canonical`` and dictionary-verified terms carry their match
    evidence, because the ranker uses dictionary evidence to justify declaring a
    required concept absent rather than unknown.
    """
    return ProjectAnalysisData(
        input=InputInfo(word_count=32),
        project_understanding=ProjectUnderstanding(
            domain=DomainInfo(value=spec.domain, confidence=0.9) if spec.domain else None,
            subdomain=(
                SubdomainInfo(value=spec.subdomain, confidence=0.8)
                if spec.subdomain
                else None
            ),
            problem_type=ProblemTypeInfo(value="Machine Learning", confidence=0.9),
            ml_tasks=[MLTask(task=t, confidence=0.8) for t in spec.tasks],
            objective=objective,
            target=(
                TargetInfo(value=spec.target, confidence=0.7, status="inferred")
                if spec.target
                else None
            ),
        ),
        keywords=Keywords(
            canonical=list(spec.keywords),
            dictionary_matches=[
                KeywordMatchEvidence(
                    canonical_term=term,
                    matched_text=term,
                    match_type=MatchType.EXACT,
                    dictionary_entry_id=f"benchmark/{term}",
                    domain=spec.domain or "",
                    subdomain=spec.subdomain or "",
                )
                for term in spec.dictionary_keywords
            ],
            llm_inferred=[],
            related_concepts=list(spec.related_concepts),
            provenance=[],
            conflicts=[],
        ),
        dataset_requirements=DatasetRequirements(
            feature_requirements=[
                FeatureRequirement(name=f, description="", importance="required")
                for f in spec.features
            ],
            target_requirements=(
                [TargetRequirement(name=spec.target, description="Outcome to predict")]
                if spec.target
                else []
            ),
            data_type_requirements=[
                DataTypeRequirement(type=d, description="") for d in spec.data_types
            ],
        ),
        analysis=Analysis(overall_confidence=0.85),
    )


def build_candidate(spec: CandidateSpec, *, case_id: str) -> DatasetCandidate:
    """Build the candidate a retrieval step would have returned.

    Ids are ``benchmark:<case_id>/<key>``. They are deliberately not shaped like
    real registry references: a benchmark must never look like it is asserting a
    dataset exists when it does not.
    """
    source_id = f"{case_id}/{spec.key}"
    payload = {
        "id": f"benchmark:{source_id}",
        "name": spec.name,
        "source": "kaggle-fixture",
        "source_type": spec.source_type,
        "source_id": source_id,
        "verification_status": spec.effective_verification,
        "url": f"https://example.invalid/benchmark/{source_id}",
        "description": spec.description,
        "owner": "benchmark-fixture",
        "tags": list(spec.tags),
        "domain": spec.domain,
        "license": "CC0-1.0",
        "download_count": spec.downloads,
        "vote_count": spec.votes,
        "usability": spec.usability,
        "updated_at": parse_timestamp(spec.updated_at),
        "target_description": spec.target_description,
        "feature_description": spec.feature_description,
        "task_types": list(spec.task_types),
        "file_types": list(spec.file_types),
        "possible_source": spec.possible_source,
    }
    if spec.source_type.value == "ai_suggested":
        payload["rationale"] = spec.rationale or spec.description
    return DatasetCandidate(**payload)


def build_ai_suggestion(spec: CandidateSpec, *, case_id: str) -> LLMDiscoveredCandidate:
    """Build the pre-normalisation AI suggestion a candidate spec describes."""
    return LLMDiscoveredCandidate(
        name=spec.name,
        possible_source=spec.possible_source or "unknown",
        url=f"https://example.invalid/benchmark/{case_id}/{spec.key}",
        description=spec.description,
        why_relevant=spec.rationale or spec.description,
        confidence=0.8,
        is_well_known=False,
        suggested_features=[],
        target_description=spec.target_description,
        ml_tasks=list(spec.task_types),
        license="unknown",
    )


def compile_case(case: BenchmarkCase) -> CompiledCase:
    """Build the real schemas for one case and pre-join grades to candidate ids."""
    candidates = [build_candidate(spec, case_id=case.id) for spec in case.candidates]
    grades_by_id: dict[str, int] = {}
    for spec, candidate in zip(case.candidates, candidates):
        grades_by_id[candidate.id] = case.grade_of(spec.key)
    return CompiledCase(
        case=case,
        requirements=build_requirements(case.requirements, objective=case.description),
        candidates=candidates,
        grades_by_id=grades_by_id,
    )