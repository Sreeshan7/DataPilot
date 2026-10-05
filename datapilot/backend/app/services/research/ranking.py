"""Explainable relevance ranking of retrieved papers (0-100). Not an academic quality score.

Components and weights (sum 100):

* problem 40: share of the question's core topics found in the title/snippet (title counts most)
* context 10: the situation words (weather, season...) found
* task 10, domain 5, target 5: whether those terms appear
* methods 10: names a recognised ML method
* recency 10: newer is better, scaled over the last ten years (unknown year earns nothing, never a guess)
* source 10: a known publisher or repository scores higher than a university page

TOPIC GATE: a paper must mention at least half of the core topics (any synonym counts) or it is dropped, whatever
else it scores. A method paper that never mentions the subject is not about the subject.

Citation counts are never retrieved, so they play no part. Ties break on URL, so the order is deterministic.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from app.schemas.research import Paper, ResearchDataset, ResearchProject
from app.services.research.normalize import host_of, venue_for
from app.services.research.queries import QueryPlan, task_word, understand

_WORD = re.compile(r"[a-z0-9]+")
_FILLER = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "using", "with", "by"}


def _words(text: str) -> set[str]:
    return {w[:-1] if len(w) > 4 and w.endswith("s") else w for w in _WORD.findall(text.lower()) if w not in _FILLER}


def _overlap(terms: set[str], text: set[str]) -> float:
    return len(terms & text) / len(terms) if terms else 0.0


def _stems(text: str) -> set[str]:
    return _words(text)


def _group_hit(group: list[str], whole: set[str], lowered: str) -> Optional[str]:
    for word in group:
        if " " in word or "-" in word:
            if word in lowered:
                return word
        elif next(iter(_stems(word)), word) in whole:
            return word
    return None


def rank(
    papers: list[Paper],
    project: ResearchProject,
    dataset: Optional[ResearchDataset],
    *,
    min_year: Optional[int],
    today_year: Optional[int] = None,
    plan: Optional[QueryPlan] = None,
) -> list[Paper]:
    year_now = today_year or date.today().year
    plan = plan or understand(project, dataset)
    groups = plan.core_groups
    need = -(-len(groups) // 2)  # ceil(n / 2)
    context = _words(" ".join(plan.context_terms))
    task = _words(task_word(project.task) or task_word(dataset.task if dataset else None) or "")
    domain = _words(project.domain or "")
    target_text = project.target or (dataset.target if dataset else None) or ""
    target = _words(target_text)
    scored: list[Paper] = []
    for paper in papers:
        if min_year and paper.publication_year is not None and paper.publication_year < min_year:
            continue  # an unknown year is kept (never guessed), a known older year is excluded
        title, whole = _words(paper.title), _words(f"{paper.title} {paper.abstract_snippet or ''}")
        lowered = f"{paper.title} {paper.abstract_snippet or ''}".lower()
        in_title = _words(paper.title)
        hits = [(g, _group_hit(g, whole, lowered)) for g in groups]
        matched = [h for _, h in hits if h]
        if groups and len(matched) < need:
            continue  # the topic gate
        title_hits = sum(1 for g in groups if _group_hit(g, in_title, paper.title.lower()))
        share = len(matched) / len(groups) if groups else 0.0
        p = share * 30 + (title_hits / len(groups) if groups else 0.0) * 10
        c = 10.0 * _overlap(context, whole)
        t = 10.0 * _overlap(task, whole)
        d = 5.0 * _overlap(domain, whole)
        g = 5.0 * _overlap(target, whole)
        m = 10.0 if paper.methods else 0.0
        r = 0.0
        if paper.publication_year is not None:
            r = 10.0 * max(0.0, min(1.0, 1.0 - (year_now - paper.publication_year) / 10.0))
        s = 10.0 if venue_for(host_of(paper.url)) and paper.venue not in ("ResearchGate", "Google Scholar") else 5.0
        comps = {"problem": p, "context": c, "task": t, "domain": d, "target": g, "methods": m, "recency": r, "source": s}
        score = round(sum(comps.values()), 2)
        reasons = []
        if matched:
            where = "its title" if title_hits == len(groups) else "its title or abstract"
            reasons.append(f"{where} mentions {', '.join(matched)}")
        if c:
            reasons.append("it also covers " + ", ".join(w for w in plan.context_terms if next(iter(_stems(w)), w) in whole))
        if t:
            reasons.append(f"it covers {task_word(project.task) or 'the task'}")
        if paper.methods:
            reasons.append("it names " + ", ".join(paper.methods[:3]))
        if paper.publication_year:
            reasons.append(f"published {paper.publication_year}")
        scored.append(paper.model_copy(update={
            "relevance_score": min(score, 100.0), "score_components": {k: round(v, 2) for k, v in comps.items()},
            "matched_terms": matched,
            "relevance_reason": ("Selected because " + "; ".join(reasons) + ".") if reasons
            else "Weak match: little of the problem appears in the retrieved text."}))
    return sorted(scored, key=lambda x: (-x.relevance_score, x.url))
