"""Understand the research question, then build a few focused queries from it. Deterministic and offline.

Raw user text is a poor search query: "I want to research the best methods for predicting the trafic during specific
time and weather conditions" is mostly meta-words ("research", "best", "methods"), a typo ("trafic") and a context
clause ("during ... weather"). ``understand`` separates that into

* **core topics**: what the papers must be about ("traffic"); each is a group of interchangeable words, which an
  optional LLM pass may widen with synonyms ("vehicle", "road");
* **context terms**: conditions that sharpen the search but do not define it ("weather", "time");
* **corrections**: typos repaired, reported to the user so nothing changes silently.

The core topics also drive the relevance gate in ``ranking.py``: a paper that never mentions them is dropped.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from app.schemas.research import ResearchDataset, ResearchProject

MAX_QUERIES = 5
MAX_CORE = 4

#: Words about the act of researching or predicting, not about the subject.
META_WORDS = frozenset("""
i we want to need would like build create make develop a an the using use from with based on of for and or that which my
our model models system project predict predicts predicting prediction predictions forecast forecasting forecasts
research researching best good better top methods method techniques technique algorithms algorithm approach approaches
papers paper literature review study studies state art find me please help is are in by can will how what which
machine learning ml ai dataset data specific certain particular different various some any many more most
analysis analyse analyze understand explore improve improving accurate accuracy performance conditions condition
company companies business organization organisation
""".split()) - {"performance"}  # "student performance" is a subject; it is only meta when alone

#: Words that, after a connector, describe the situation rather than the subject.
_CONNECTOR = re.compile(r"\b(?:during|under|in|at|given|when|across|depending on|according to|based on|using|from|with)\b", re.I)
_TASK_WORDS = {
    "classification": "classification", "binary_classification": "classification",
    "multiclass_classification": "classification", "regression": "regression",
    "time_series_forecasting": "forecasting", "forecasting": "forecasting", "clustering": "clustering",
    "anomaly_detection": "anomaly detection", "ranking": "ranking", "multilabel_classification": "classification",
}
_TABULAR_TASKS = {"classification", "regression"}
#: Common subjects of ML projects, so a typo of one of them is recognised even if the dictionary lacks it.
_EXTRA_VOCAB = frozenset("""
traffic weather student customer churn fraud diabetes price energy crop medical health disease cancer credit loan
stock sales demand electricity temperature climate rainfall pollution air quality transport vehicle accident patient
employee attrition house housing salary sentiment spam image speech text network intrusion malware
""".split())


def _clean(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s\-/]", " ", text).split())


def task_word(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    key = task.strip().lower().replace(" ", "_").replace("-", "_")
    return _TASK_WORDS.get(key)


@lru_cache(maxsize=1)
def _vocabulary() -> frozenset[str]:
    words = set(_EXTRA_VOCAB)
    try:  # the project's own concept dictionary knows hundreds of subjects
        from app.services.keyword_service import KeywordService

        for term in KeywordService().get_all_canonical_terms():
            words.update(w for w in re.findall(r"[a-z]{5,}", term.lower()))
    except Exception:  # an optional aid: without it only the built-in list is used
        pass
    return frozenset(words)


def fix_typos(words: list[str]) -> tuple[list[str], dict[str, str]]:
    """Repair a word that is one slip away from a known subject ("trafic" -> "traffic"). Reported, never silent."""
    vocab = _vocabulary()
    fixed, corrections = [], {}
    for word in words:
        low = word.lower()
        if len(low) >= 5 and low not in vocab and low not in META_WORDS:
            match = difflib.get_close_matches(low, vocab, n=1, cutoff=0.86)
            if match and match[0][0] == low[0] and abs(len(match[0]) - len(low)) <= 1:
                corrections[word] = match[0]
                fixed.append(match[0])
                continue
        fixed.append(low)
    return fixed, corrections


def _content(text: str) -> list[str]:
    return [w for w in _clean(text).split() if w.lower() not in META_WORDS and len(w) > 2 and not w.isdigit()]


@dataclass
class QueryPlan:
    problem: Optional[str]
    core_groups: list[list[str]]                 # each group: interchangeable words, any one counts
    context_terms: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    why_not: Optional[str] = None
    ai_assisted: bool = False
    outcome: Optional[str] = None                # what is predicted, e.g. "traffic volume"
    task: Optional[str] = None                   # classification | regression | forecasting | ...
    features: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)   # readings to pick from when the question is ambiguous
    clarification: Optional[str] = None          # the one question to ask instead of guessing
    dataset_check: Optional[tuple[str, str, str]] = None   # (status, note, source)

    @property
    def core_terms(self) -> list[str]:
        return [g[0] for g in self.core_groups]

    @property
    def all_core_words(self) -> list[str]:
        return [w for g in self.core_groups for w in g]


def understand(project: ResearchProject, dataset: Optional[ResearchDataset] = None) -> QueryPlan:
    """Core topics, context terms and corrections from whatever the project provides."""
    text = project.problem or project.description or ""
    parts = _CONNECTOR.split(text, maxsplit=1)
    head, tail = parts[0], parts[1] if len(parts) > 1 else ""
    # An explicit short problem statement is already a subject: do not cut it at a connector.
    if project.problem and len(_clean(project.problem).split()) <= 6:
        head, tail = project.problem, ""
    head_words, corrections = fix_typos(_content(head))
    tail_words, tail_fix = fix_typos(_content(tail))
    corrections.update(tail_fix)
    core = list(dict.fromkeys(head_words))[:MAX_CORE]
    if not core and tail_words:  # "predicting during winter storms": the context is all there is
        core, tail_words = tail_words[:MAX_CORE], []
    context = [w for w in dict.fromkeys(tail_words) if w not in core][:3]
    for column in ((project.features or []) + (dataset.columns if dataset else []))[:20]:
        for word in re.findall(r"[a-z]{4,}", column.lower().replace("_", " ")):
            if word in ("weather", "temperature", "holiday", "season", "rainfall", "humidity") and word not in context + core:
                context.append(word)
    context = context[:3]
    if not core:
        return QueryPlan(problem=None, core_groups=[], why_not=(
            "Research needs to know the problem. Give a problem statement such as 'student academic performance "
            "prediction' (a description naming the outcome also works)."))
    return QueryPlan(problem=" ".join(core), core_groups=[[w] for w in core], context_terms=context, corrections=corrections)


def check_dataset(project: ResearchProject, dataset: Optional[ResearchDataset]) -> Optional[tuple[str, str, str]]:
    """Deterministic: does the stated target exist among the dataset's columns? (status, note, source) or None."""
    if not dataset or not dataset.columns:
        return None
    wanted = project.target or None
    if not wanted:
        return None
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    if any(norm(wanted) == norm(c) for c in dataset.columns):
        return ("consistent", f"The target '{wanted}' is a column in the dataset.", "datapilot")
    return ("mismatch", f"The stated target '{wanted}' is not among the dataset's {len(dataset.columns)} columns.", "datapilot")


def problem_phrase(project: ResearchProject) -> Optional[str]:
    return understand(project).problem


def build_queries(
    project: ResearchProject, dataset: Optional[ResearchDataset] = None, limit: int = MAX_QUERIES,
    plan: Optional[QueryPlan] = None,
) -> tuple[list[str], Optional[str]]:
    """(queries, why_not). An empty list comes with the reason, so the caller can ask the user."""
    plan = plan or understand(project, dataset)
    if plan.problem is None:
        return [], plan.why_not
    task = plan.task or task_word(project.task) or (task_word(dataset.task) if dataset else None)
    domain = _clean(project.domain) if project.domain else None
    topic = f'"{plan.problem}"' if len(plan.core_terms) > 1 else plan.problem
    queries = [f"{topic} machine learning"]
    if task:
        queries.append(f"{topic} {task}")
    if plan.outcome and plan.outcome.lower() not in plan.problem.lower():
        queries.append(f"{topic} {plan.outcome}")
    if plan.context_terms:
        queries.append(f"{plan.problem} {' '.join(plan.context_terms[:2])} prediction")
    if domain and domain.lower() not in plan.problem.lower():
        queries.append(f"{topic} {domain}")
    queries.append(f"{topic} review OR survey")
    if task in _TABULAR_TASKS:
        queries.append(f"{topic} random forest gradient boosting")
    elif task == "forecasting":
        queries.append(f"{topic} LSTM ARIMA")
    seen: set[str] = set()
    unique = [q for q in queries if not (q.lower() in seen or seen.add(q.lower()))]
    plan.queries = unique[: max(1, min(limit, MAX_QUERIES))]
    return plan.queries, None
