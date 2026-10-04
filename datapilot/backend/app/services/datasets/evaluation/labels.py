"""Graded relevance labels for the ranking benchmark.

Binary relevant/irrelevant cannot express the thing this project keeps running
into: a dataset that publishes almost nothing may still be exactly the right
dataset, while a dataset with a perfect metadata record can be completely
unusable. The grade therefore records *how good the fit is*, and the label set
includes a negative grade so that "clearly incompatible" is not silently
recorded as merely "irrelevant".

Grades are deliberately independent of how much metadata a candidate publishes.
A dataset whose metadata is silent is graded on what it *is*, not on how loud it
is; missing metadata is an evaluation concern (METADATA_INSUFFICIENT), not a
relevance grade.
"""

from __future__ import annotations

from enum import IntEnum


class RelevanceGrade(IntEnum):
    """Graded relevance of one candidate for one benchmark case."""

    #: Directly satisfies domain + task + target + the major required features.
    HIGHLY_RELEVANT = 3
    #: Genuinely useful, but some metadata or requirements are missing.
    RELEVANT = 2
    #: Possibly usable after substantial transformation or feature engineering.
    WEAK = 1
    #: Weak relationship, but not genuinely suitable for this project.
    IRRELEVANT = 0
    #: Clearly incompatible: wrong subject, wrong modality, unusable label.
    INCOMPATIBLE = -1


#: The conventional binary cut. Used by precision/recall so the headline numbers
#: mean what a reader expects, while NDCG keeps the full graded signal.
RELEVANT_THRESHOLD = RelevanceGrade.RELEVANT

#: At least this good counts as a "correct answer" for the retrieval question.
HIGHLY_RELEVANT_THRESHOLD = RelevanceGrade.HIGHLY_RELEVANT


def is_relevant(grade: int, threshold: int = RELEVANT_THRESHOLD) -> bool:
    """Whether ``grade`` clears the relevance threshold used by P@k / R@k."""
    return grade >= threshold


def is_highly_relevant(grade: int) -> bool:
    """Whether ``grade`` is the best answer available for the case."""
    return grade >= HIGHLY_RELEVANT_THRESHOLD


def is_incompatible(grade: int) -> bool:
    """Whether the candidate is explicitly unusable, not merely unrelated."""
    return grade <= RelevanceGrade.INCOMPATIBLE


GRADE_DEFINITIONS: dict[RelevanceGrade, str] = {
    RelevanceGrade.HIGHLY_RELEVANT: (
        "Directly satisfies domain + task + target + the major required features."
    ),
    RelevanceGrade.RELEVANT: (
        "Genuinely useful, but some metadata or requirements are missing. Still a "
        "dataset a practitioner would open."
    ),
    RelevanceGrade.WEAK: (
        "May be usable after substantial transformation or feature engineering - the "
        "signal is there but the dataset does not answer the question as shipped."
    ),
    RelevanceGrade.IRRELEVANT: (
        "Weak relationship but not genuinely suitable. Present in the result set "
        "because the vocabulary overlaps, not because the dataset serves the task."
    ),
    RelevanceGrade.INCOMPATIBLE: (
        "Clearly incompatible: a different subject, a different modality, or a label "
        "the project cannot use. Must never outrank a usable dataset."
    ),
}