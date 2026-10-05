"""The *offer* of research: a next action a client can show, never a research run.

``/assist`` and ``/datasets/analyze`` call these helpers to describe whether "Research this problem" is
available, with a pre-filled request for ``POST /api/v1/research/discover``. Building the offer performs no
search and no Gemini call; the user's explicit selection is what triggers the research endpoint.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.analysis_response import NextAction
from app.schemas.dataset_analysis import AnalysisMode, DatasetAnalysis
from app.schemas.project import ProjectAnalysisData
from app.schemas.project_intelligence import FeatureRole, ProjectIntelligence
from app.schemas.research import ResearchProject
from app.services.research.queries import problem_phrase

ENDPOINT = "/api/v1/research/discover"
_ID = "research_problem"
_LABEL = "Research this problem"


def _action(available: bool, reason: str, payload: Optional[dict] = None) -> NextAction:
    return NextAction(
        action="RESEARCH_PROBLEM", id=_ID, label=_LABEL, available=available, reason=reason,
        description="Search recent papers and methods related to this problem (uses web search; optional AI summary).",
        endpoint=ENDPOINT, payload=payload or {})


def offer_from_analysis(
    analysis: DatasetAnalysis, intelligence: Optional[ProjectIntelligence], goal: Optional[str]
) -> NextAction:
    understanding = intelligence.understanding if intelligence else None
    target = understanding.best_target if understanding else None
    confident = bool(understanding and understanding.target_is_confident and target)
    task = understanding.primary_task.task.value if understanding and understanding.primary_task else None
    problem = problem_phrase(ResearchProject(description=goal)) if goal else None
    if problem is None and confident and target is not None:
        problem = f"{target.column.replace('_', ' ')} prediction"
    if problem is None:
        return _action(False, "Describe the problem (or confirm the target column) so research has something specific to search for.")
    features = [f.column for f in intelligence.feature_intelligence.features
                if f.role is FeatureRole.RECOMMENDED_FEATURE][:12] if intelligence else []
    columns = [c.normalized_name for c in analysis.schema_info.columns
               if c.role not in ("row_id", "ignore")][:50]
    name = analysis.dataset_identity.name
    payload = {
        "project": {"description": goal, "problem": problem, "task": task,
                    "target": target.column if confident and target else None, "features": features},
        "dataset": {"name": name, "columns": columns, "task": task,
                    "target": target.column if confident and target else None,
                    "rows": analysis.schema_info.row_count},
        "options": {"max_papers": 8, "use_ai": False},
    }
    why = "The problem is specific enough to search." + (
        "" if analysis.mode is AnalysisMode.FILE_PARSED else " (Dataset known from its registry schema only.)")
    return _action(True, why, payload)


def offer_from_requirements(description: Optional[str], requirements: Optional[ProjectAnalysisData]) -> NextAction:
    understanding = requirements.project_understanding if requirements else None
    target = understanding.target.value if understanding and understanding.target else None
    task = understanding.ml_tasks[0].task if understanding and understanding.ml_tasks else None
    domain = understanding.domain.value if understanding and understanding.domain else None
    objective = understanding.objective if understanding else None
    project = ResearchProject(description=description or objective, problem=None, task=task, domain=domain, target=target)
    problem = problem_phrase(project) or (problem_phrase(ResearchProject(description=objective)) if objective else None)
    if problem is None:
        return _action(False, "The project description is too general to research; say what outcome you want to predict.")
    features = [f.name for f in requirements.dataset_requirements.feature_requirements][:12] if requirements else []
    payload = {
        "project": {"description": description, "problem": problem, "task": task, "domain": domain,
                    "target": target, "features": features},
        "options": {"max_papers": 8, "use_ai": False},
    }
    return _action(True, "The project is well-defined enough to search for related literature.", payload)
