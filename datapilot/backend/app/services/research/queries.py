"""Deterministic research query builder: a few focused queries, never a keyword soup."""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.research import ResearchDataset, ResearchProject

MAX_QUERIES = 5
_STOP = {
    "i", "we", "want", "to", "need", "would", "like", "build", "create", "make", "develop", "a", "an", "the",
    "using", "use", "from", "with", "based", "on", "of", "for", "and", "or", "that", "which", "my", "our",
    "model", "system", "project", "predict", "predicts", "prediction", "using", "by", "can", "will", "how",
    "machine", "learning", "ml", "ai", "dataset", "data", "find", "me", "please", "help", "is", "are", "in",
}
_CUT = re.compile(r"\b(?:using|from|based on|with|by analy[sz]ing|given|on the basis of)\b", re.I)
_TASK_WORDS = {
    "classification": "classification", "binary_classification": "classification",
    "multiclass_classification": "classification", "regression": "regression",
    "time_series_forecasting": "forecasting", "forecasting": "forecasting", "clustering": "clustering",
    "anomaly_detection": "anomaly detection", "ranking": "ranking", "multilabel_classification": "classification",
}
_TABULAR_TASKS = {"classification", "regression"}


def _clean(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s\-/]", " ", text).split())


def task_word(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    key = task.strip().lower().replace(" ", "_").replace("-", "_")
    return _TASK_WORDS.get(key)


def problem_phrase(project: ResearchProject) -> Optional[str]:
    """The problem as a short noun phrase: the stated one, or the leading content words of the description."""
    if project.problem:
        phrase = _clean(project.problem)
        return phrase if len(phrase.split()) >= 2 else None
    if project.description:
        head = _CUT.split(project.description, maxsplit=1)[0]  # the inputs after "using ..." are not the problem
        content = [w for w in _clean(head).split() if w.lower() not in _STOP and len(w) > 2]
        return " ".join(content[:5]) if len(content) >= 2 else None
    return None


def build_queries(
    project: ResearchProject, dataset: Optional[ResearchDataset] = None, limit: int = MAX_QUERIES
) -> tuple[list[str], Optional[str]]:
    """(queries, why_not). An empty list comes with the reason, so the caller can ask the user."""
    problem = problem_phrase(project)
    task = task_word(project.task) or (task_word(dataset.task) if dataset else None)
    domain = _clean(project.domain) if project.domain else None
    if problem is None:
        return [], ("Research needs to know the problem. Give a problem statement such as "
                    "'student academic performance prediction' (a description naming the outcome also works).")
    quoted = f'"{problem}"'
    queries = [f"{quoted} machine learning"]
    if task:
        queries.append(f"{quoted} {task}")
    if domain and domain.lower() not in problem.lower():
        queries.append(f"{quoted} {domain}")
    queries.append(f"{quoted} review OR survey")
    if task in _TABULAR_TASKS:
        queries.append(f"{quoted} random forest gradient boosting")
    elif task == "forecasting":
        queries.append(f"{quoted} LSTM ARIMA")
    seen: set[str] = set()
    unique = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]
    return unique[: max(1, min(limit, MAX_QUERIES))], None
