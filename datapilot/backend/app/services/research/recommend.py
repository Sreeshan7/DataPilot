"""Algorithm recommendations: literature support combined with compatibility to the user's own data.

Literature support counts papers whose retrieved text mentions a method (strong >= 3, moderate 2, limited 1);
frequency is one input, not a verdict. Dataset compatibility comes from the method's standard use against the
user's task and, when known, the dataset's size and column types. Methods that are only compatible, never
found in the literature, are not listed: this section reports what the retrieved evidence supports.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.research import AlgorithmRecommendation, Level, MethodSupport, ResearchDataset, ResearchProject
from app.services.research.methods import CATALOG
from app.services.research.queries import task_word

_INFO = {m.name: m for m in CATALOG}
_LEVEL_ORDER = {Level.STRONG: 3, Level.MODERATE: 2, Level.LIMITED: 1, Level.UNKNOWN: 0, Level.NONE: -1}
SMALL_ROWS = 1000
_DEEP = {"Deep Learning", "LSTM", "Transformer", "Convolutional Neural Network", "Neural Network"}


def support_level(count: int) -> Level:
    return Level.STRONG if count >= 3 else Level.MODERATE if count == 2 else Level.LIMITED if count == 1 else Level.NONE


def compatibility(
    method: str, task: Optional[str], rows: Optional[int], has_text_or_images: bool
) -> tuple[Level, list[str]]:
    info = _INFO.get(method)
    cautions: list[str] = []
    if info is None or task is None:
        return Level.UNKNOWN, ["your task is not known, so compatibility cannot be judged"] if task is None else []
    if task not in info.tasks:
        return Level.NONE, [f"{method} is not a standard choice for {task}"]
    level = Level.STRONG
    if not info.tabular and not has_text_or_images:
        level = Level.MODERATE
        cautions.append(f"{method} is usually applied to sequences, text or images rather than plain tables")
    if method in _DEEP and rows is not None and rows < SMALL_ROWS:
        level = Level.MODERATE if level is Level.STRONG else level
        cautions.append(f"with {rows:,} rows a deep model is likely to overfit; compare it with tree ensembles")
    if rows is None and method in _DEEP:
        cautions.append("the dataset size is unknown; deep models need a lot of data")
    return level, cautions


def recommend(
    methods: list[MethodSupport],
    project: ResearchProject,
    dataset: Optional[ResearchDataset],
    max_items: int = 6,
) -> list[AlgorithmRecommendation]:
    task = task_word(project.task) or task_word(dataset.task if dataset else None)
    rows = dataset.rows if dataset else None
    rich = False
    scored = []
    for support in methods:
        comp, cautions = compatibility(support.method, task, rows, rich)
        if comp is Level.NONE:
            continue
        lit = support_level(support.verified_paper_count)
        scored.append((support, lit, comp, cautions))
    # Both kinds of evidence count: literature support and fit with the user's task.
    scored.sort(key=lambda s: (-(_LEVEL_ORDER[s[1]] + _LEVEL_ORDER[s[2]]), -s[0].verified_paper_count, s[0].method))
    out = []
    for rank, (support, lit, comp, cautions) in enumerate(scored[:max_items], start=1):
        n = support.verified_paper_count
        why = (f"{n} retrieved paper{'s' if n != 1 else ''} mention {support.method}; "
               + (f"it is a standard choice for {task} on tabular data. " if comp is Level.STRONG and task
                  else "compatibility with your task is only partly established. ")
               + "Literature frequency is evidence of use, not proof that it is best for your data: validate it "
                 "against a simple baseline on a held-out set.")
        out.append(AlgorithmRecommendation(
            rank=rank, method=support.method, literature_support=lit, dataset_compatibility=comp,
            evidence_paper_count=n, paper_ids=support.paper_ids, why=why, cautions=cautions))
    return out
