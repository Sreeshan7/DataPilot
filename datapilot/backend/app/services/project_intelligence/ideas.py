"""Project ideas grounded in the dataset, generated without a language model.

Every idea is built from facts the analysis established: a target that exists, a task the data
supports, features that exist and have been judged (when rows were available). The model never
invents a column, because nothing here is free text: names come from ``DataShape.columns`` and are
checked again by ``validate_ideas`` before anything is returned.

Between three and five ideas are returned when the data can support that many. An idea that the
data cannot support (a forecast without a time axis, a text model without text) is simply not
generated, rather than generated and caveated.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.dataset_analysis import (
    SUPERVISED_TASKS,
    AnalysisMode,
    CandidateState,
    ColumnType,
    DatasetAnalysis,
    EvidenceLevel,
    MLTask,
    QualityCode,
    TargetCandidate,
)
from app.schemas.project_intelligence import (
    DatasetMLUnderstanding,
    Difficulty,
    FeatureIntelligence,
    FeatureRole,
    IdeaSource,
    ProjectIdea,
)
from app.services.project_intelligence.shape import DataShape

MAX_IDEAS = 5
MAX_REQUIRED = 8
MAX_OPTIONAL = 6
MIN_SEGMENT_FEATURES = 3

_METRICS = {
    MLTask.BINARY_CLASSIFICATION: [
        "ROC AUC", "precision and recall (PR AUC when positives are rare)", "balanced accuracy",
        "confusion matrix at a chosen threshold"],
    MLTask.MULTICLASS_CLASSIFICATION: ["macro F1", "balanced accuracy", "per-class recall", "confusion matrix"],
    MLTask.REGRESSION: ["MAE", "RMSE", "R-squared", "residual plots"],
    MLTask.TIME_SERIES_FORECASTING: [
        "MAE and MASE against a seasonal-naive baseline", "walk-forward (rolling-origin) validation"],
    MLTask.CLUSTERING: ["silhouette score", "stability across random seeds", "interpretability of the segments"],
    MLTask.ANOMALY_DETECTION: ["precision among reviewed alerts", "alert rate", "agreement between detectors"],
}
_BASELINES = {
    MLTask.BINARY_CLASSIFICATION: ["majority-class predictor", "logistic regression"],
    MLTask.MULTICLASS_CLASSIFICATION: ["majority-class predictor", "multinomial logistic regression"],
    MLTask.REGRESSION: ["mean predictor", "ridge regression"],
    MLTask.TIME_SERIES_FORECASTING: ["seasonal-naive / last value", "exponential smoothing"],
    MLTask.CLUSTERING: ["k-means"],
    MLTask.ANOMALY_DETECTION: ["robust z-score / IQR rules"],
}
_STRONGER = {
    MLTask.BINARY_CLASSIFICATION: ["gradient-boosted trees", "random forest"],
    MLTask.MULTICLASS_CLASSIFICATION: ["gradient-boosted trees", "random forest"],
    MLTask.REGRESSION: ["gradient-boosted trees", "random forest"],
    MLTask.TIME_SERIES_FORECASTING: ["gradient-boosted trees on lag and calendar features", "ARIMA-family models"],
    MLTask.CLUSTERING: ["Gaussian mixture model", "hierarchical clustering"],
    MLTask.ANOMALY_DETECTION: ["isolation forest", "local outlier factor"],
}
_VERB = {
    MLTask.BINARY_CLASSIFICATION: "Classify", MLTask.MULTICLASS_CLASSIFICATION: "Classify",
    MLTask.REGRESSION: "Estimate", MLTask.TIME_SERIES_FORECASTING: "Forecast",
}


def _slug(*parts: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", "-", "-".join(p for p in parts if p).lower()).strip("-")[:60]


def _pretty(name: str) -> str:
    return name.replace("_", " ")


def _names(items: list[str], limit: int = 4) -> str:
    shown = [f"`{n}`" for n in items[:limit]]
    more = len(items) - limit
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


# ---------------------------------------------------------------------------
# Evidence basis
# ---------------------------------------------------------------------------


def _basis(analysis: DatasetAnalysis) -> EvidenceLevel:
    if analysis.mode is AnalysisMode.FILE_PARSED:
        return EvidenceLevel.DATAPILOT_PARSED
    level = analysis.schema_info.level
    known = (EvidenceLevel.REGISTRY_VERIFIED, EvidenceLevel.INFERRED, EvidenceLevel.AI_INFERRED)
    return level if level in known else EvidenceLevel.INFERRED


def _feature_lists(
    shape: DataShape, intel: FeatureIntelligence, target: Optional[str]
) -> tuple[list[str], list[str]]:
    """(required, optional) features for a supervised idea about ``target``."""
    if intel.available and intel.target == target:
        required = [f.column for f in intel.features if f.role is FeatureRole.RECOMMENDED_FEATURE]
        optional = [f.column for f in intel.features if f.role is FeatureRole.USE_WITH_CAUTION]
        return required[:MAX_REQUIRED], optional[:MAX_OPTIONAL]
    usable = [c.name for c in shape.columns if c.usable_feature and c.name != target]
    return usable[:MAX_REQUIRED], usable[MAX_REQUIRED:MAX_REQUIRED + MAX_OPTIONAL]


def _idea_state(target: TargetCandidate) -> CandidateState:
    """An idea is as firm as its target: STRONG for a strong target, otherwise POSSIBLE."""
    return CandidateState.STRONG_CANDIDATE if target.state is CandidateState.STRONG_CANDIDATE else CandidateState.POSSIBLE


def _difficulty(
    shape: DataShape, task: MLTask, *, imbalanced: bool, n_features: int, has_text: bool, many_classes: bool
) -> Difficulty:
    points = 0
    points += imbalanced
    points += shape.parsed and shape.max_missing_pct > 10.0
    points += n_features > 30
    points += shape.n_rows is not None and shape.n_rows < 200
    points += task is MLTask.TIME_SERIES_FORECASTING
    points += has_text
    points += many_classes
    return Difficulty.BEGINNER if points <= 1 else Difficulty.INTERMEDIATE if points <= 3 else Difficulty.ADVANCED


def _challenges(shape: DataShape, target: Optional[str], intel: FeatureIntelligence) -> list[str]:
    out: list[str] = []
    for f in shape.findings_with(QualityCode.CLASS_IMBALANCE):
        out.append(f"class imbalance in `{f.column}`: use precision/recall-style metrics and consider class weights"
                   if f.column else "class imbalance: use precision/recall-style metrics")
        break
    miss = [c.name for c in shape.columns if (c.missing_pct or 0) > 20]
    if miss:
        out.append(f"substantial missing values in {_names(miss)}: decide how to impute or drop them")
    leaky = [f.column for f in intel.features if f.role is FeatureRole.POSSIBLE_LEAKAGE] if intel.available else []
    if leaky:
        out.append(f"{_names(leaky)} may leak the outcome and must be excluded or justified")
    if shape.n_rows is not None and shape.n_rows < 200:
        out.append(f"only {shape.n_rows} rows: use cross-validation and keep models simple")
    if shape.findings_with(QualityCode.DUPLICATE_ROWS):
        out.append("duplicate rows can leak between training and test splits: deduplicate before splitting")
    if any(c.high_cardinality for c in shape.columns):
        out.append("high-cardinality categorical columns need grouping or target/frequency encoding")
    if any(c.sensitive for c in shape.columns):
        out.append("sensitive personal columns are present: consider privacy and fairness before using them")
    return out


def _limitations(
    analysis: DatasetAnalysis, intel: Optional[FeatureIntelligence], target: Optional[TargetCandidate],
) -> list[str]:
    """Caveats for an idea; ``intel`` is None for ideas that do not use feature importance."""
    out: list[str] = []
    if analysis.mode is AnalysisMode.EXTERNAL_METADATA:
        out.append("based on the registry's published schema only: no rows were read, so feature "
                   "usefulness, class balance and data quality are not known")
    if target is not None and target.state is not CandidateState.STRONG_CANDIDATE and not target.user_specified:
        out.append(f"`{target.column}` is only a {target.state.value.replace('_', ' ')} target: confirm it first")
    if intel is not None:
        if intel.available and intel.method is not None and not intel.method.reliable:
            out.append("a quick model on this data did not beat a trivial predictor, so signal may be weak")
        out.append("feature roles describe predictive association in this file, not cause and effect")
    return out


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_ideas(
    analysis: DatasetAnalysis,
    understanding: DatasetMLUnderstanding,
    intel: FeatureIntelligence,
    shape: DataShape,
) -> list[ProjectIdea]:
    if not shape.has_schema:
        return []
    basis = _basis(analysis)
    imbalanced = bool(shape.findings_with(QualityCode.CLASS_IMBALANCE))
    has_text = any(c.kind is ColumnType.TEXT for c in shape.columns)
    candidates: list[ProjectIdea] = []

    primary = understanding.primary_task
    best = understanding.best_target
    if primary is not None and primary.task in SUPERVISED_TASKS and best is not None:
        idea = _supervised_idea(analysis, understanding, intel, shape, basis, best, primary.task, primary=True,
                                imbalanced=imbalanced, has_text=has_text)
        text = _text_idea(analysis, understanding, intel, shape, basis, best, primary.task)
        if idea is None and text is not None:
            candidates.append(text)
        if idea:
            candidates.append(idea)
            driver = _driver_idea(idea, intel, shape, basis, understanding)
            if text:
                candidates.append(text)
            for alternative in understanding.alternative_targets:
                alt_idea = _alternative_target_idea(
                    analysis, understanding, intel, shape, basis, alternative, imbalanced, has_text)
                if alt_idea:
                    candidates.append(alt_idea)
                    break
            if driver:
                candidates.append(driver)
            ts = _forecast_idea(analysis, understanding, intel, shape, basis, idea)
            if ts:
                candidates.append(ts)
    elif primary is not None and primary.task is MLTask.TIME_SERIES_FORECASTING and best is not None:
        idea = _supervised_idea(analysis, understanding, intel, shape, basis, best, primary.task, primary=True,
                                imbalanced=imbalanced, has_text=has_text)
        if idea:
            candidates.append(idea)

    anomaly = _anomaly_idea(analysis, understanding, shape, basis)
    if anomaly:
        candidates.append(anomaly)
    segment = _segment_idea(analysis, understanding, shape, basis)
    if segment:
        candidates.append(segment)
    topics = _topic_idea(analysis, shape, basis)
    if topics:
        candidates.append(topics)

    seen: set[str] = set()
    unique = []
    for idea in candidates:
        if idea.id not in seen:
            seen.add(idea.id)
            unique.append(idea)
    return unique[:MAX_IDEAS]


def _supervised_idea(
    analysis, understanding, intel, shape, basis, target: TargetCandidate, task: MLTask, *, primary: bool,
    imbalanced: bool, has_text: bool,
) -> Optional[ProjectIdea]:
    if shape.column(target.column) is None:
        return None
    required, optional = _feature_lists(shape, intel, target.column)
    if not required and not optional:
        return None
    verb = _VERB.get(task, "Predict")
    classes = shape.column(target.column)
    many_classes = bool(task is MLTask.MULTICLASS_CLASSIFICATION and classes and classes.kind is ColumnType.CATEGORICAL
                        and any(f.code is QualityCode.HIGH_CARDINALITY_CATEGORICAL and f.column == target.column
                                for f in shape.findings))
    why = [understanding.summary]
    if intel.available and intel.target == target.column and required:
        why.append(f"On this file {_names(required, 3)} showed the clearest signal.")
    elif analysis.mode is AnalysisMode.EXTERNAL_METADATA:
        why.append("The registry publishes the schema, so the idea is grounded in its columns.")
    if shape.n_rows:
        why.append(f"{shape.n_rows:,} rows are {'available' if shape.parsed else 'reported by the registry'}.")
    return ProjectIdea(
        id=_slug("predict" if primary else "alt", target.column, task.value),
        title=f"{verb} `{target.column}`" if task is not MLTask.MULTICLASS_CLASSIFICATION else f"Classify `{target.column}` into its categories",
        problem_statement=(
            f"Build a {task.value.replace('_', ' ')} model that predicts `{target.column}` for a record from the "
            f"other columns ({_names(required or optional)})."),
        task=task, target=target.column, required_features=required, optional_features=optional,
        why_it_fits=" ".join(why),
        challenges=_challenges(shape, target.column, intel),
        evaluation_metrics=_METRICS.get(task, []),
        baseline_models=_BASELINES.get(task, []),
        stronger_models=_STRONGER.get(task, []),
        difficulty=_difficulty(shape, task, imbalanced=imbalanced, n_features=len(required) + len(optional),
                               has_text=has_text, many_classes=many_classes),
        extension_potential="Compare against a baseline, add feature engineering, calibrate probabilities, "
                            "and report which columns the model leans on.",
        limitations=_limitations(analysis, intel, target),
        evidence_level=basis,
        state=_idea_state(target),
    )


def _driver_idea(
    main: ProjectIdea, intel: FeatureIntelligence, shape: DataShape, basis, understanding
) -> Optional[ProjectIdea]:
    """'What is associated with X?': needs measured features, so only for parsed files."""
    if not intel.available or intel.target != main.target:
        return None
    strong = [f.column for f in intel.features if f.role is FeatureRole.RECOMMENDED_FEATURE]
    cautioned = [f.column for f in intel.features if f.role is FeatureRole.USE_WITH_CAUTION]
    measured = strong or cautioned
    if not measured or main.target is None:
        return None
    return ProjectIdea(
        id=_slug("drivers", main.target),
        title=f"Which columns are most associated with `{main.target}`?",
        problem_statement=(
            f"Explain `{main.target}` rather than only predict it: rank the columns "
            f"({_names(measured)}) by how much each adds, and show how the outcome changes across their values."),
        task=main.task, target=main.target, required_features=strong[:MAX_REQUIRED],
        optional_features=cautioned[:MAX_OPTIONAL],
        why_it_fits="Feature intelligence measured these columns on your file against random-column yardsticks, "
                    "so an explanatory analysis starts from evidence instead of guesses.",
        challenges=["association is not causation: say 'associated with', never 'causes'",
                    "correlated columns share credit, so individual importances understate each of them"],
        evaluation_metrics=["stability of the ranking across resamples", "agreement between permutation importance and mutual information"],
        baseline_models=["univariate association table"], stronger_models=["permutation importance on gradient-boosted trees", "partial-dependence plots"],
        difficulty=Difficulty.BEGINNER,
        extension_potential="Add SHAP values, subgroup analysis, or a causal-inference design if intervention is the goal.",
        limitations=["descriptive analysis: it does not show what would happen if a column were changed"],
        evidence_level=basis, state=CandidateState.POSSIBLE)


def _alternative_target_idea(
    analysis, understanding, intel, shape, basis, target: TargetCandidate, imbalanced: bool, has_text: bool
) -> Optional[ProjectIdea]:
    if target.state in (CandidateState.WEAK, CandidateState.UNKNOWN) and not target.user_specified:
        return None
    task = next((t.task for t in analysis.task_candidates
                 if t.target == target.column and t.task in SUPERVISED_TASKS and t.state is not CandidateState.WEAK), None)
    if task is None:
        return None
    return _supervised_idea(analysis, understanding, intel, shape, basis, target, task, primary=False,
                            imbalanced=imbalanced, has_text=has_text)


def _text_idea(analysis, understanding, intel, shape, basis, target: TargetCandidate, task: MLTask) -> Optional[ProjectIdea]:
    texts = [c.name for c in shape.columns if c.kind is ColumnType.TEXT]
    if not texts or task not in (MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION, MLTask.REGRESSION):
        return None
    return ProjectIdea(
        id=_slug("text", target.column, task.value),
        title=f"Predict `{target.column}` from the text in `{texts[0]}`",
        problem_statement=(f"Turn the free text in {_names(texts)} into features and predict `{target.column}` from it."),
        task=task, target=target.column, required_features=texts[:2],
        optional_features=[c.name for c in shape.columns if c.usable_feature and c.name not in texts and c.name != target.column][:MAX_OPTIONAL],
        why_it_fits=f"`{texts[0]}` is free text, which tabular models ignore; it is the part of this dataset that needs NLP.",
        challenges=["short or repetitive texts carry little signal", "vocabulary leakage between train and test if split carelessly"],
        evaluation_metrics=_METRICS.get(task, []), baseline_models=["TF-IDF with logistic regression"],
        stronger_models=["pretrained sentence embeddings with a linear head", "fine-tuned transformer (if there is enough data)"],
        difficulty=Difficulty.INTERMEDIATE,
        extension_potential="Compare bag-of-words with embeddings and inspect the words the model relies on.",
        limitations=_limitations(analysis, intel, target), evidence_level=basis,
        state=CandidateState.POSSIBLE)


def _forecast_idea(analysis, understanding, intel, shape, basis, main: ProjectIdea) -> Optional[ProjectIdea]:
    """A forecasting variant when the primary task is plain regression on a time-indexed table."""
    if main.task is MLTask.TIME_SERIES_FORECASTING or main.target is None:
        return None
    task = next((t for t in analysis.task_candidates
                 if t.task is MLTask.TIME_SERIES_FORECASTING and t.target == main.target
                 and t.state is not CandidateState.WEAK), None)
    if task is None:
        return None
    required = [c for c in main.required_features if (shape.column(c) and shape.column(c).kind is not ColumnType.DATETIME)]
    return ProjectIdea(
        id=_slug("forecast", main.target),
        title=f"Forecast `{main.target}` over time",
        problem_statement=f"Forecast future values of `{main.target}` using its history and the other columns.",
        task=MLTask.TIME_SERIES_FORECASTING, target=main.target, required_features=required,
        optional_features=main.optional_features,
        why_it_fits="The table has a regular time axis, so the order of rows carries information a random split would waste.",
        challenges=["validate chronologically, never with a random split", "trends and seasonality need explicit handling"],
        evaluation_metrics=_METRICS[MLTask.TIME_SERIES_FORECASTING], baseline_models=_BASELINES[MLTask.TIME_SERIES_FORECASTING],
        stronger_models=_STRONGER[MLTask.TIME_SERIES_FORECASTING], difficulty=Difficulty.INTERMEDIATE,
        extension_potential="Add holiday/promotion effects and prediction intervals.",
        limitations=_limitations(analysis, intel, understanding.best_target), evidence_level=basis,
        state=CandidateState.POSSIBLE)


def _unsupervised_features(shape: DataShape) -> list[str]:
    """Columns worth clustering on: usable, not mostly empty, and not a copy of another column."""
    duplicates: set[str] = {
        f.column for f in shape.findings_with(QualityCode.LEAKAGE_SIGNAL, QualityCode.POTENTIAL_LEAKAGE) if f.column
    }  # a column derived from an outcome would dominate any grouping
    for finding in shape.findings_with(QualityCode.DUPLICATE_COLUMNS):
        duplicates.update(finding.columns[1:])  # keep the first of each group
    return [
        c.name for c in shape.columns
        if c.usable_feature and c.name not in duplicates and (c.missing_pct or 0.0) <= 50.0
        and c.kind in (ColumnType.NUMERIC, ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
    ]


def _anomaly_idea(analysis, understanding, shape, basis) -> Optional[ProjectIdea]:
    offered = any(t.task is MLTask.ANOMALY_DETECTION and t.state is not CandidateState.WEAK for t in analysis.task_candidates)
    outliers = shape.findings_with(QualityCode.POSSIBLE_OUTLIER)
    usable = set(_unsupervised_features(shape))
    numeric = [c.name for c in shape.columns if c.name in usable and c.kind is ColumnType.NUMERIC]
    if not numeric or not (offered or outliers):
        return None
    flagged = [f.column for f in outliers if f.column][:3]
    return ProjectIdea(
        id="anomaly-detection", title="Detect unusual records",
        problem_statement=f"Flag rows that look unusual across {_names(numeric)} without needing labels.",
        task=MLTask.ANOMALY_DETECTION, target=None, required_features=numeric[:MAX_REQUIRED],
        optional_features=[c for c in _unsupervised_features(shape) if c not in numeric][:MAX_OPTIONAL],
        why_it_fits=("Outlying values were found in " + _names(flagged) + "." if flagged
                     else "The dataset has a rare class, which makes unusual-record detection a natural framing."),
        challenges=["no labels to validate against: review a sample of flagged rows with a domain expert",
                    "unusual is not the same as wrong"],
        evaluation_metrics=_METRICS[MLTask.ANOMALY_DETECTION], baseline_models=_BASELINES[MLTask.ANOMALY_DETECTION],
        stronger_models=_STRONGER[MLTask.ANOMALY_DETECTION], difficulty=Difficulty.INTERMEDIATE,
        extension_potential="Combine detectors, add a human-review loop and track the alert rate over time.",
        limitations=_limitations(analysis, None, None)
        + ["flagged rows are unusual under the model, not confirmed errors"],
        evidence_level=basis, state=CandidateState.POSSIBLE)


def _segment_idea(analysis, understanding, shape, basis) -> Optional[ProjectIdea]:
    feats = _unsupervised_features(shape)
    if len(feats) < MIN_SEGMENT_FEATURES or (shape.n_rows is not None and shape.n_rows < 50):
        return None
    return ProjectIdea(
        id="segmentation", title="Segment the records into groups",
        problem_statement=f"Group similar records using {_names(feats)} and describe what distinguishes each group.",
        task=MLTask.CLUSTERING, target=None, required_features=feats[:MAX_REQUIRED],
        optional_features=feats[MAX_REQUIRED:MAX_REQUIRED + MAX_OPTIONAL],
        why_it_fits=f"There are {len(feats)} usable columns and no need for a label, so clustering can reveal structure.",
        challenges=["choosing the number of groups", "mixed numeric and categorical columns need care when scaling"],
        evaluation_metrics=_METRICS[MLTask.CLUSTERING], baseline_models=_BASELINES[MLTask.CLUSTERING],
        stronger_models=_STRONGER[MLTask.CLUSTERING], difficulty=Difficulty.BEGINNER,
        extension_potential="Profile each segment and test whether segments differ on the outcome you care about.",
        limitations=_limitations(analysis, None, None)
        + ["clusters are a description of the data, not a ground truth"],
        evidence_level=basis, state=CandidateState.POSSIBLE)


def _topic_idea(analysis, shape, basis) -> Optional[ProjectIdea]:
    texts = [c.name for c in shape.columns if c.kind is ColumnType.TEXT]
    if not texts:
        return None
    return ProjectIdea(
        id="text-topics", title=f"Discover themes in `{texts[0]}`",
        problem_statement=f"Group the free text in {_names(texts)} into themes and describe each.",
        task=MLTask.CLUSTERING, target=None, required_features=texts[:2], optional_features=[],
        why_it_fits=f"`{texts[0]}` is free text, which can be grouped by meaning without any labels.",
        challenges=["themes are subjective: validate by reading examples"],
        evaluation_metrics=["coherence of the themes when read by a person", "stability across seeds"],
        baseline_models=["TF-IDF with k-means"], stronger_models=["sentence embeddings with clustering"],
        difficulty=Difficulty.INTERMEDIATE, extension_potential="Track themes over time or by group.",
        limitations=_limitations(analysis, None, None) + ["themes are exploratory and not verified"],
        evidence_level=basis, state=CandidateState.WEAK)


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------


def validate_ideas(ideas: list[ProjectIdea], columns: set[str]) -> list[ProjectIdea]:
    """Keep only ideas whose target and features all exist; the guard against invented columns."""
    kept = []
    for idea in ideas:
        referenced = set(idea.required_features) | set(idea.optional_features)
        if idea.target:
            referenced.add(idea.target)
        if referenced <= columns:
            kept.append(idea)
    return kept
