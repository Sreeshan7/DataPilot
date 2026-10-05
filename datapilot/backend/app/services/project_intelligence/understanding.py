"""Reconcile an analysis's target and task candidates into one account of what the data can learn.

Phase 4 produces ranked target candidates and the tasks each supports. This step does not
re-detect anything; it settles which of them to put forward, how confident that choice is, and
what stands in the way, using only what the analysis already established.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.dataset_analysis import (
    SUPERVISED_TASKS,
    AnalysisMode,
    CandidateState,
    DatasetAnalysis,
    EvidenceLevel,
    FindingSeverity,
    MLTask,
    TargetCandidate,
    TaskCandidate,
)
from app.schemas.project_intelligence import (
    DatasetMLUnderstanding,
    LearningSetting,
    TargetSource,
)

MAX_ALTERNATIVES = 3
_STATE_ORDER = {
    CandidateState.STRONG_CANDIDATE: 0,
    CandidateState.POSSIBLE: 1,
    CandidateState.WEAK: 2,
    CandidateState.UNKNOWN: 3,
}


def _target_source(analysis: DatasetAnalysis, target: TargetCandidate) -> TargetSource:
    if target.user_specified:
        return TargetSource.USER_SPECIFIED
    if analysis.mode is AnalysisMode.EXTERNAL_METADATA and any(
        e.level is EvidenceLevel.REGISTRY_VERIFIED and "designates" in e.statement for e in target.evidence
    ):
        return TargetSource.REGISTRY_DESIGNATED
    return TargetSource.DETECTED


def _primary_task(
    tasks: list[TaskCandidate], target: Optional[TargetCandidate]
) -> Optional[TaskCandidate]:
    """The best task for the chosen target; failing that, the best task overall."""
    # Among equally supported tasks the more specific wins: a regular time series is forecast,
    # not merely regressed.
    ordered = sorted(
        tasks,
        key=lambda t: (_STATE_ORDER[t.state], t.task is not MLTask.TIME_SERIES_FORECASTING, -t.confidence, t.task.value),
    )
    if target is not None:
        for task in ordered:
            if task.target == target.column and task.state is not CandidateState.WEAK:
                return task
        for task in ordered:
            if task.target == target.column:
                return task
    for task in ordered:
        if task.task not in SUPERVISED_TASKS and task.state is not CandidateState.WEAK:
            return task
    return ordered[0] if ordered else None


def understand(analysis: DatasetAnalysis) -> DatasetMLUnderstanding:
    targets = sorted(
        analysis.target_candidates,
        key=lambda t: (not t.user_specified, _STATE_ORDER[t.state], -t.score, t.column),
    )
    best = targets[0] if targets else None
    # A WEAK candidate is a hint, not a recommendation: keep it visible but not as "the" target.
    if best is not None and best.state in (CandidateState.WEAK, CandidateState.UNKNOWN) and not best.user_specified:
        confident, source = False, TargetSource.DETECTED
    elif best is not None:
        source = _target_source(analysis, best)
        confident = source in (TargetSource.USER_SPECIFIED, TargetSource.REGISTRY_DESIGNATED) \
            or best.state is CandidateState.STRONG_CANDIDATE
    else:
        confident, source = False, TargetSource.NONE

    primary = _primary_task(analysis.task_candidates, best)
    other = [t for t in analysis.task_candidates if t is not primary]

    if primary is None:
        setting = LearningSetting.UNKNOWN  # nothing the data supports could be established
    elif primary.task in SUPERVISED_TASKS and best is not None:
        setting = LearningSetting.SUPERVISED
    elif primary.task in SUPERVISED_TASKS:
        setting = LearningSetting.UNKNOWN  # a supervised task with no target cannot occur; stay honest
    else:
        setting = LearningSetting.UNSUPERVISED_ONLY

    blockers: list[str] = []
    if analysis.quality is not None:
        blockers = [
            f.detail for f in analysis.quality.findings if f.severity is FindingSeverity.CRITICAL
        ][:5]

    limitations = list(analysis.limitations)
    if best is not None and not confident:
        limitations.append(
            f"'{best.column}' is only a {best.state.value.replace('_', ' ')} target; confirm it "
            "(or name the target yourself) before building on it."
        )
    if primary is not None and primary.limitations:
        limitations.extend(l for l in primary.limitations if l not in limitations)

    return DatasetMLUnderstanding(
        learning_setting=setting,
        best_target=best,
        target_source=source,
        target_is_confident=confident,
        alternative_targets=targets[1:1 + MAX_ALTERNATIVES],
        primary_task=primary,
        other_tasks=other,
        blockers=blockers,
        limitations=limitations,
        summary=_summary(analysis, setting, best, source, confident, primary),
    )


def _summary(
    analysis: DatasetAnalysis,
    setting: LearningSetting,
    best: Optional[TargetCandidate],
    source: TargetSource,
    confident: bool,
    primary: Optional[TaskCandidate],
) -> str:
    where = "your file" if analysis.mode is AnalysisMode.FILE_PARSED else "the registry's published schema"
    if setting is LearningSetting.SUPERVISED and best is not None and primary is not None:
        how = {
            TargetSource.USER_SPECIFIED: "the target you named",
            TargetSource.REGISTRY_DESIGNATED: "the target the registry designates",
        }.get(source, "the leading target candidate" if confident else "a possible target")
        return (
            f"From {where}, '{best.column}' is {how}, which supports "
            f"{primary.task.value.replace('_', ' ')}."
        )
    if setting is LearningSetting.UNSUPERVISED_ONLY:
        return (
            f"From {where}, no column looks like a prediction target, so only unsupervised work "
            "(clustering, anomaly detection) is supported unless you name a target."
        )
    return (
        f"From {where}, DataPilot cannot tell what this dataset could be used to predict; "
        "the information needed is not available."
    )
