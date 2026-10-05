"""Can this dataset serve the project the user described? MATCH, UNKNOWN or MISMATCH, with reasons.

The goal is read deterministically (no model call): a task cue ("forecast", "classify", "segment"),
the thing to predict ("predict *student performance* using ..."), and the inputs it names
("... using *attendance, study hours*"). Each is compared with what the dataset actually offers:

* **target**: does a column (or the dictionary concept behind it) name what the goal wants predicted?
* **task**: does the data support that kind of task?
* **features**: do the inputs the goal mentions appear as columns?
* **volume**: are there enough rows for a supervised goal?

The rules mirror User A's: an unknown stays UNKNOWN, a MISMATCH needs positive evidence of a
contradiction (a forecasting goal and no time column; a classification goal and no categorical
column at all; two targets the dictionary places in different label spaces), and absence of a
match is never promoted to a mismatch. One MISMATCH makes the whole answer a MISMATCH and suggests
looking for another dataset.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.dataset_analysis import (
    AnalysisMode,
    CandidateState,
    ColumnType,
    DatasetAnalysis,
    EvidenceItem,
    EvidenceLevel,
    MLTask,
    TargetKind,
)
from app.schemas.project_intelligence import (
    CompatibilityDimension,
    CompatibilityStatus,
    DatasetMLUnderstanding,
    ProjectCompatibility,
)
from app.services.datasets.target_concepts import get_target_concept_index, singularize, value_tokens
from app.services.project_intelligence.shape import DataShape

MATCH, UNKNOWN, MISMATCH = CompatibilityStatus.MATCH, CompatibilityStatus.UNKNOWN, CompatibilityStatus.MISMATCH
MAX_GOAL_CHARS = 2000
MIN_ROWS_SUPERVISED = 30
COMFORTABLE_ROWS = 100

_TASK_CUES: list[tuple[MLTask, re.Pattern[str]]] = [
    (MLTask.TIME_SERIES_FORECASTING, re.compile(
        r"\bforecast\w*|\btime[ -]series\b|\bpredict\w*\s+(?:the\s+)?(?:future|next)\b|\bover\s+time\b", re.I)),
    (MLTask.CLUSTERING, re.compile(r"\bcluster\w*|\bsegment\w*|\bgroup\s+(?:similar|customers|users|records)\b", re.I)),
    (MLTask.ANOMALY_DETECTION, re.compile(r"\banomal\w*|\boutlier\w*|\bunusual\b|\bintrusion\b|\bfraud\b|\bmalicious\b", re.I)),
    (MLTask.BINARY_CLASSIFICATION, re.compile(
        r"\bclassif\w*|\bwhether\b|\bchurn\w*|\bdetect\w*|\bdiagnos\w*|\bspam\b|\bwill\s+\w+\s+(?:or\s+not)?|"
        r"\bdefault\w*|\bpass\s+or\s+fail\b|\bwhich\s+(?:category|class)\b", re.I)),
    (MLTask.REGRESSION, re.compile(
        r"\bregress\w*|\bhow\s+(?:much|many)\b|\bestimat\w*|\bpredict\w*\s+(?:the\s+)?(?:price|amount|cost|salary|"
        r"score|grade|value|sales|revenue|temperature|count|quantity|yield|rating|demand|consumption|income)\b", re.I)),
]
_TARGET = re.compile(
    r"\b(?:predict(?:s|ing)?|forecast(?:s|ing)?|estimat(?:e|es|ing)|classif(?:y|ies|ying)|detect(?:s|ing)?|"
    r"identify(?:ing)?|determin(?:e|ing)|diagnos(?:e|ing)|measur(?:e|ing))\s+"
    r"(?:(?:whether|if|the|a|an|which|how)\s+)*(?P<t>[^.,;:!?]+?)"
    r"(?=\s+(?:using|from|based\s+on|with|by|via|given|for|in|of|when|before|after|over)\b|[.,;:!?]|$)",
    re.I,
)
_FEATURES = re.compile(
    r"\b(?:using|from|based\s+on|with|by\s+analy[sz]ing|by\s+using|given|on\s+the\s+basis\s+of)\s+(?P<f>[^.;:!?]+)", re.I)
_SPLIT = re.compile(r",|\band\b|\bplus\b|&|/")
_FILLER = {"the", "their", "its", "each", "every", "some", "any", "of", "own", "his", "her", "a", "an", "various",
           "several", "different", "other", "such", "as", "like", "data", "information", "details", "levels", "level"}


#: Common abbreviations in column names and the words a project would use for them.
_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "sqft": ("square", "foot", "footage", "feet", "area"), "bmi": ("body", "mass", "index"),
    "qty": ("quantity",), "amt": ("amount",), "num": ("number",), "avg": ("average",), "yr": ("year",),
    "dob": ("birth", "date"), "pct": ("percent", "percentage"), "temp": ("temperature",),
    "cust": ("customer",), "bp": ("blood", "pressure"), "gpa": ("grade", "point", "average"),
    "tenure": ("duration",),
}
#: Words that name a kind of quantity rather than a particular one.
_GENERIC = {"amount", "count", "number", "rate", "ratio", "score", "total", "value", "level", "type", "status"}


class GoalReading:
    """What the project text asks for, extracted without any model."""

    def __init__(self, goal: str) -> None:
        self.text = " ".join(goal.split())[:MAX_GOAL_CHARS]
        self.tasks: list[MLTask] = [task for task, pattern in _TASK_CUES if pattern.search(self.text)]
        match = _TARGET.search(self.text)
        self.predictive = match is not None  # it asks for something to be predicted, named or not
        phrase = match.group("t").strip() if match else None
        self.target_phrase = phrase if phrase and value_tokens(phrase) else None
        features: list[str] = []
        feature_match = _FEATURES.search(self.text)
        if feature_match:
            for part in _SPLIT.split(feature_match.group("f")):
                words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9']*", part) if w.lower() not in _FILLER]
                if 0 < len(words) <= 4:
                    features.append(" ".join(words))
        self.feature_phrases = features[:12]

    @property
    def summary(self) -> str:
        bits = []
        if self.tasks:
            bits.append("task: " + ", ".join(t.value.replace("_", " ") for t in self.tasks))
        if self.target_phrase:
            bits.append(f"predicts: {self.target_phrase}")
        if self.feature_phrases:
            bits.append("inputs: " + ", ".join(self.feature_phrases))
        return "; ".join(bits) if bits else "the goal does not state a task, a target or inputs"


def _tokens(text: str) -> set[str]:
    return {singularize(t) for t in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if len(t) > 1}


def _same_word(a: str, b: str) -> bool:
    """Equal, or one is a clear prefix of the other ('promo' / 'promotion')."""
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def _match_column(phrase: str, names: set[str]) -> Optional[str]:
    """The column whose name best matches the phrase, or None.

    A column matches when words of the phrase and of its name coincide and those words make up at
    least half of the phrase or half of the name: one shared generic word ("patient" in "patient has
    diabetes" and "patient_id") is not a match.
    """
    wanted = _tokens(phrase) - _FILLER
    if not wanted:
        return None
    best: tuple[float, str] | None = None
    for name in sorted(names):
        base = _tokens(name)
        have = base | {w for t in base for w in _ABBREVIATIONS.get(t, ())}
        if not have:
            continue
        shared_wanted = {w for w in wanted if any(_same_word(w, h) for h in have)}
        shared_have = {h for h in have if any(_same_word(w, h) for w in wanted)}
        if not shared_wanted:
            continue
        # Words like "rate" or "amount" alone say nothing about which quantity is meant: they only
        # count when the column itself is made of nothing else ("amount").
        generic = set(value_tokens(" ".join(shared_wanted))) == set()
        if generic and any(t not in _GENERIC for t in base):
            continue
        strength = max(len(shared_wanted) / len(wanted), len(shared_have) / len(have))
        if strength >= 0.5 and (best is None or strength > best[0]):
            best = (strength, name)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------


def _target_dimension(
    goal: GoalReading, understanding: DatasetMLUnderstanding, shape: DataShape
) -> CompatibilityDimension:
    if not goal.target_phrase:
        return CompatibilityDimension(
            dimension="target", status=UNKNOWN, detail="The goal does not say what is to be predicted.")
    candidates = [t for t in ([understanding.best_target] if understanding.best_target else []) + understanding.alternative_targets]
    # An identifier, constant or free-text column is never what a project predicts.
    eligible = {c.name for c in shape.columns if c.usable_feature}
    direct = _match_column(goal.target_phrase, eligible)
    if direct is not None:
        is_detected = any(t.column == direct for t in candidates)
        note = "" if is_detected else f" It is not the detected target; name it with target='{direct}' to use it."
        return CompatibilityDimension(
            dimension="target", status=MATCH,
            detail=f"The goal wants '{goal.target_phrase}', and column '{direct}' carries that name.{note}",
            evidence=[EvidenceItem(statement=f"'{goal.target_phrase}' shares its wording with column '{direct}'",
                                   level=EvidenceLevel.HEURISTIC)])
    index = get_target_concept_index()
    wanted = index.expectation(goal.target_phrase)
    if wanted:
        for column in [t.column for t in candidates]:
            have = index.expectation(column)
            if have and (wanted.values & have.values or any(
                index.family(a) & index.family(b) for a in wanted.spaces for b in have.spaces)):
                return CompatibilityDimension(
                    dimension="target", status=MATCH,
                    detail=f"The goal's target '{goal.target_phrase}' and column '{column}' name the same outcome "
                           "in the project dictionary.",
                    evidence=[EvidenceItem(statement="matched through the project's concept dictionary",
                                           level=EvidenceLevel.INFERRED)])
        resolvable = [t.column for t in candidates if index.expectation(t.column)]
        # A contradiction needs a schema we can trust: parsed rows or a registry's structured schema. A column
        # list only read from a description is often partial, so it can never prove "no column is named for it".
        trusted = shape.parsed or shape.schema_level is EvidenceLevel.REGISTRY_VERIFIED
        if trusted and candidates and len(resolvable) == len(candidates):
            names = ", ".join(f"'{c}'" for c in resolvable)
            return CompatibilityDimension(
                dimension="target", status=MISMATCH,
                detail=f"The goal wants '{goal.target_phrase}', but every candidate target here ({names}) is a "
                       "different outcome in the project dictionary, and no column is named for it.",
                evidence=[EvidenceItem(statement="the dictionary places the goal's target and the dataset's targets "
                                                 "in different label spaces", level=EvidenceLevel.INFERRED)])
    return CompatibilityDimension(
        dimension="target", status=UNKNOWN,
        detail=f"No column is clearly named for '{goal.target_phrase}', but one may hold it under another name.")


def _task_dimension(
    goal: GoalReading, analysis: DatasetAnalysis, shape: DataShape
) -> CompatibilityDimension:
    if not goal.tasks:
        return CompatibilityDimension(
            dimension="task", status=UNKNOWN, detail="The goal does not state what kind of model is wanted.")
    offered = {t.task for t in analysis.task_candidates if t.state is not CandidateState.WEAK}
    offered_any = {t.task for t in analysis.task_candidates}
    kinds = {c.kind for c in shape.columns if c.usable_feature}
    class_like_target = any(t.target_kind in (TargetKind.BINARY, TargetKind.MULTICLASS) for t in analysis.target_candidates)
    verdicts: list[tuple[CompatibilityStatus, str]] = []
    for task in goal.tasks:
        label = task.value.replace("_", " ")
        acceptable = {task}
        if task is MLTask.BINARY_CLASSIFICATION:
            acceptable = {MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION}
        if task is MLTask.ANOMALY_DETECTION:
            acceptable = {MLTask.ANOMALY_DETECTION, MLTask.BINARY_CLASSIFICATION}
        if acceptable & offered:
            verdicts.append((MATCH, f"the data supports {label}"))
        elif shape.parsed and task is MLTask.TIME_SERIES_FORECASTING and ColumnType.DATETIME not in {c.kind for c in shape.columns}:
            verdicts.append((MISMATCH, "a forecasting goal needs a date or time column and this file has none"))
        elif shape.parsed and task in (MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION) \
                and not (kinds & {ColumnType.CATEGORICAL, ColumnType.BOOLEAN}) and not class_like_target:
            verdicts.append((MISMATCH, "a classification goal needs a categorical or yes/no column and this file has none"))
        elif shape.parsed and task is MLTask.REGRESSION and ColumnType.NUMERIC not in kinds:
            verdicts.append((MISMATCH, "a regression goal needs a numeric column and this file has none"))
        elif acceptable & offered_any:
            verdicts.append((UNKNOWN, f"{label} is only weakly supported by this data"))
        else:
            verdicts.append((UNKNOWN, f"{label} was not established for this data"))
    statuses = {s for s, _ in verdicts}
    status = MATCH if MATCH in statuses else MISMATCH if statuses == {MISMATCH} else UNKNOWN
    return CompatibilityDimension(dimension="task", status=status, detail="; ".join(d for _, d in verdicts))


def _feature_dimension(goal: GoalReading, shape: DataShape, understanding: DatasetMLUnderstanding) -> tuple[CompatibilityDimension, list[str]]:
    if not goal.feature_phrases:
        return CompatibilityDimension(
            dimension="features", status=UNKNOWN, detail="The goal does not list the inputs it would use."), []
    target = understanding.best_target.column if understanding.best_target else None
    names = {c.name for c in shape.columns if c.name != target and c.usable_feature}
    found: dict[str, str] = {}
    missing: list[str] = []
    for phrase in goal.feature_phrases:
        column = _match_column(phrase, names)
        if column is not None:
            found[phrase] = column
        else:
            missing.append(phrase)
    matched = sorted(set(found.values()))
    if not missing:
        status = MATCH
        count = "The input" if len(found) == 1 else f"All {len(found)} inputs"
        verb = "has" if len(found) == 1 else "have"
        detail = f"{count} the goal mentions {verb} a matching column ({', '.join(f'{p} -> {c}' for p, c in found.items())})."
    elif found:
        status = UNKNOWN
        detail = (f"{len(found)} of {len(goal.feature_phrases)} mentioned inputs match a column; no column is named "
                  f"for: {', '.join(missing)} (they may exist under other names).")
    elif len(goal.feature_phrases) >= 2 and shape.has_schema and (shape.parsed or shape.schema_level is EvidenceLevel.REGISTRY_VERIFIED):
        status = MISMATCH
        detail = f"None of the {len(goal.feature_phrases)} inputs the goal mentions ({', '.join(missing)}) matches any column."
    else:
        status = UNKNOWN
        detail = f"No column is clearly named for: {', '.join(missing)}."
    return CompatibilityDimension(dimension="features", status=status, detail=detail), matched


def _volume_dimension(goal: GoalReading, shape: DataShape) -> CompatibilityDimension:
    supervised = (bool(goal.tasks) and any(
        t not in (MLTask.CLUSTERING, MLTask.ANOMALY_DETECTION) for t in goal.tasks)) or goal.predictive
    if shape.n_rows is None:
        return CompatibilityDimension(dimension="volume", status=UNKNOWN, detail="The number of rows is not known.")
    if shape.n_rows < MIN_ROWS_SUPERVISED and supervised:
        return CompatibilityDimension(
            dimension="volume", status=MISMATCH,
            detail=f"Only {shape.n_rows} rows: too few to train and validate a supervised model.")
    if shape.n_rows >= COMFORTABLE_ROWS:
        return CompatibilityDimension(dimension="volume", status=MATCH, detail=f"{shape.n_rows:,} rows is a workable size.")
    return CompatibilityDimension(
        dimension="volume", status=UNKNOWN, detail=f"{shape.n_rows} rows is small: expect noisy estimates.")


def assess_compatibility(
    goal_text: str, analysis: DatasetAnalysis, understanding: DatasetMLUnderstanding, shape: DataShape
) -> ProjectCompatibility:
    goal = GoalReading(goal_text)
    if not shape.has_schema:
        dimension = CompatibilityDimension(
            dimension="schema", status=UNKNOWN,
            detail="The dataset's columns are not known, so nothing about the goal can be checked against them.")
        return ProjectCompatibility(
            goal_summary=goal.summary, status=UNKNOWN, dimensions=[dimension],
            summary="This dataset's structure is unknown, so its fit for the project cannot be judged. "
                    "Upload the file to find out.")
    features, matched = _feature_dimension(goal, shape, understanding)
    dimensions = [_target_dimension(goal, understanding, shape), _task_dimension(goal, analysis, shape),
                  features, _volume_dimension(goal, shape)]
    core = dimensions[:3]
    if any(d.status is MISMATCH for d in dimensions):
        status = MISMATCH
    elif sum(d.status is MATCH for d in core) >= 2 and not (features.status is UNKNOWN and goal.feature_phrases):
        # An input the project names but no column carries keeps the answer short of a MATCH,
        # as an absent requested feature keeps a dataset out of HIGH confidence in User A.
        status = MATCH
    else:
        status = UNKNOWN
    mismatches = [d for d in dimensions if d.status is MISMATCH]
    where = "your file" if analysis.mode is AnalysisMode.FILE_PARSED else "the registry's published schema"
    if status is MISMATCH:
        summary = f"From {where}, this dataset does not fit the project: " + " ".join(d.detail for d in mismatches)
    elif status is MATCH:
        summary = f"From {where}, this dataset fits the project on {sum(d.status is MATCH for d in core)} of 3 checks, " \
                  "with nothing contradicting it."
    else:
        summary = f"From {where}, nothing contradicts the project, but not enough could be confirmed to call it a match."
    return ProjectCompatibility(
        goal_summary=goal.summary, status=status, dimensions=dimensions, matched_columns=matched,
        summary=summary, suggests_alternative_dataset=status is MISMATCH)
