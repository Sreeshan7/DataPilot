"""Explainable relevance ranking of retrieved papers (0-100). Not an academic quality score.

Components and weights (sum 100):

* problem 40: how much of the problem phrase appears in the title/snippet (all words in the title counts most)
* task 10, domain 10, target 10: whether those terms appear
* methods 10: names a recognised ML method
* recency 10: newer is better, scaled over the last ten years (unknown year earns nothing, never a guess)
* source 10: a known publisher or repository scores higher than a university page

Citation counts are never retrieved, so they play no part. Ties break on URL, so the order is deterministic.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from app.schemas.research import Paper, ResearchDataset, ResearchProject
from app.services.research.normalize import host_of, venue_for
from app.services.research.queries import problem_phrase, task_word

_WORD = re.compile(r"[a-z0-9]+")
_FILLER = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "using", "with", "by"}


def _words(text: str) -> set[str]:
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in _WORD.findall(text.lower()) if w not in _FILLER}


def _overlap(terms: set[str], text: set[str]) -> float:
    return len(terms & text) / len(terms) if terms else 0.0


def rank(
    papers: list[Paper],
    project: ResearchProject,
    dataset: Optional[ResearchDataset],
    *,
    min_year: Optional[int],
    today_year: Optional[int] = None,
) -> list[Paper]:
    year_now = today_year or date.today().year
    problem = _words(problem_phrase(project) or "")
    task = _words(task_word(project.task) or task_word(dataset.task if dataset else None) or "")
    domain = _words(project.domain or "")
    target_text = project.target or (dataset.target if dataset else None) or ""
    target = _words(target_text)
    scored: list[Paper] = []
    for paper in papers:
        if min_year and paper.publication_year is not None and paper.publication_year < min_year:
            continue  # an unknown year is kept (never guessed), a known older year is excluded
        title, whole = _words(paper.title), _words(f"{paper.title} {paper.abstract_snippet or ''}")
        p = _overlap(problem, whole) * 30 + _overlap(problem, title) * 10
        t = 10.0 * _overlap(task, whole)
        d = 10.0 * _overlap(domain, whole)
        g = 10.0 * _overlap(target, whole)
        m = 10.0 if paper.methods else 0.0
        r = 0.0
        if paper.publication_year is not None:
            r = 10.0 * max(0.0, min(1.0, 1.0 - (year_now - paper.publication_year) / 10.0))
        s = 10.0 if venue_for(host_of(paper.url)) and paper.venue not in ("ResearchGate", "Google Scholar") else 5.0
        comps = {"problem": p, "task": t, "domain": d, "target": g, "methods": m, "recency": r, "source": s}
        score = round(sum(comps.values()), 2)
        reasons = []
        if problem and _overlap(problem, title) >= 0.99:
            reasons.append("its title addresses the problem directly")
        elif problem and _overlap(problem, whole) >= 0.5:
            reasons.append("its text discusses the problem")
        elif p > 0:
            reasons.append("it only partly matches the problem")
        if t:
            reasons.append(f"it covers {task_word(project.task) or 'the task'}")
        if paper.methods:
            reasons.append("it names " + ", ".join(paper.methods[:3]))
        if paper.publication_year:
            reasons.append(f"published {paper.publication_year}")
        scored.append(paper.model_copy(update={
            "relevance_score": min(score, 100.0), "score_components": {k: round(v, 2) for k, v in comps.items()},
            "relevance_reason": ("Relevant because " + "; ".join(reasons) + ".") if reasons
            else "Weak match: little of the problem appears in the retrieved text."}))
    return sorted(scored, key=lambda x: (-x.relevance_score, x.url))
