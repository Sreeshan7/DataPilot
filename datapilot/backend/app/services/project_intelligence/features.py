"""Feature intelligence: which columns help predict the target, measured on the user's own rows.

Three independent measurements, because each fails differently:

* **association** (``association.py``): a model-free statistic per feature;
* **mutual information** (scikit-learn): any dependence, linear or not, again model-free;
* **permutation importance** of a held-out random forest: how much a model actually loses when the
  column is shuffled, judged against a random "noise" column and against a trivial baseline.

Importance exists only when DataPilot has the rows. A dataset known only from a registry gets
``external_feature_intelligence``: no numbers, every feature ``UNKNOWN`` (or ``IDENTIFIER`` where the
registry itself says so), and a reason. Nothing here ever fabricates a ranking.

Roles are leakage-aware. A column that predicts the target almost perfectly is set apart as
``POSSIBLE_LEAKAGE`` and kept *out* of the model so it cannot hide the legitimate features. When the
model does not beat a trivial predictor on held-out rows, no feature is recommended: the ranking
would be noise, and the response says so.

All randomness uses ``Settings.analysis_random_seed``, so one file always gives one answer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from app.core.config import Settings
from app.schemas.dataset_analysis import (
    ColumnFlag,
    ColumnType,
    DatasetAnalysis,
    EvidenceLevel,
    MLTask,
    QualityCode,
)
from app.schemas.project_intelligence import (
    DatasetMLUnderstanding,
    FeatureAssessment,
    FeatureIntelligence,
    FeatureRole,
    ImportanceMethod,
)
from app.services.dataset_analysis.association import (
    association,
    class_codes,
    looks_like_leakage,
    sample_indices,
)
from app.services.dataset_analysis.pipeline import ParsedResult
from app.services.dataset_analysis.profiler import TypedColumn
from app.services.dataset_analysis.safety import Deadline

#: An association at or above this counts as evidence that the column carries signal.
ASSOCIATION_STRONG = 0.15
#: Mutual information (nats) below this is treated as nothing, whatever the random column scores.
MI_FLOOR = 0.02
#: Share of all positive importance above which one feature is called out as dominant.
DOMINANCE_SHARE = 0.80
#: A held-out score this high with one dominant feature is worth a caution.
DOMINANCE_MIN_SCORE = 0.90
#: The model must beat the trivial predictor by this much for its importances to be trusted.
SKILL_MARGIN = 0.05
MI_NOISE_MARGIN = 1.5
#: An importance must exceed this many standard deviations to count as measured.
IMPORTANCE_SIGMAS = 2.0
MIN_NOISE_COLUMNS = 3
MAX_NOISE_COLUMNS = 10
HIGH_MISSING_PCT = 40.0
MIN_TEST_ROWS = 8

_IMPORTANCE_TASKS = frozenset({
    MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION,
    MLTask.REGRESSION, MLTask.TIME_SERIES_FORECASTING,
})


def unavailable(reason: str, target: Optional[str] = None, task: Optional[MLTask] = None) -> FeatureIntelligence:
    return FeatureIntelligence(available=False, unavailable_reason=reason, target=target, task=task)


# ---------------------------------------------------------------------------
# External-only datasets
# ---------------------------------------------------------------------------


def external_feature_intelligence(analysis: DatasetAnalysis, target: Optional[str]) -> FeatureIntelligence:
    """No rows, so no importance: list the registry's columns and say what is not known."""
    columns = analysis.schema_info.columns
    reason = (
        "DataPilot has not read this dataset's rows, so it cannot measure which features predict "
        "the target. Upload the file for association, mutual-information and permutation-importance analysis."
    )
    if not columns:
        return unavailable(reason + " The registry also publishes no column list.", target)
    features: list[FeatureAssessment] = []
    for column in columns:
        if column.normalized_name == target:
            continue
        registry_id = column.role == "row_id"
        features.append(FeatureAssessment(
            column=column.normalized_name,
            role=FeatureRole.IDENTIFIER if registry_id else FeatureRole.UNKNOWN,
            reasons=["the registry marks it as a row identifier"] if registry_id
            else ["not assessed: no rows were read"],
            evidence_level=column.evidence if registry_id else EvidenceLevel.UNKNOWN,
        ))
    return FeatureIntelligence(
        available=False, unavailable_reason=reason, target=target, features=features,
        role_counts=_counts(features),
    )


def _counts(features: list[FeatureAssessment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in features:
        counts[f.role.value] = counts.get(f.role.value, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Parsed files
# ---------------------------------------------------------------------------


def _feature_matrix(columns: list[TypedColumn], rows: np.ndarray) -> np.ndarray:
    """Float matrix over the given rows: numbers as they are, dates as epoch seconds, categories
    as integer codes. Missing values stay NaN (the forest handles them natively)."""
    out = np.full((rows.size, len(columns)), np.nan)
    for j, column in enumerate(columns):
        if column.ctype in (ColumnType.NUMERIC, ColumnType.BOOLEAN) and column.numeric is not None:
            out[:, j] = column.numeric[rows]
        elif column.ctype is ColumnType.DATETIME and column.datetimes is not None:
            out[:, j] = [np.nan if column.datetimes[i] is None else column.datetimes[i].timestamp() for i in rows]
        elif column.ctype is ColumnType.CATEGORICAL:
            coded = class_codes(column)
            if coded is not None:
                values = coded[0][rows].astype(float)
                values[values < 0] = np.nan
                out[:, j] = values
    return out


def _discrete_mask(columns: list[TypedColumn]) -> np.ndarray:
    return np.array([c.ctype in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN) for c in columns])


def _target_vector(target: TypedColumn, rows: np.ndarray, classification: bool):
    if classification:
        coded = class_codes(target)
        return None if coded is None else coded[0][rows]
    return None if target.numeric is None else target.numeric[rows]


def _usable_rows(y: np.ndarray, classification: bool) -> np.ndarray:
    return (y >= 0) if classification else ~np.isnan(y)


def _model_and_score(classification: bool, binary: bool, settings: Settings):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    common = dict(
        n_estimators=settings.importance_n_estimators,
        random_state=settings.analysis_random_seed,
        n_jobs=1,
        min_samples_leaf=3,
    )
    if classification:
        # ROC AUC for two classes (insensitive to imbalance and to the 0.5 cut-off);
        # balanced accuracy when there are more.
        scoring = "roc_auc" if binary else "balanced_accuracy"
        return RandomForestClassifier(class_weight="balanced_subsample", **common), scoring
    return RandomForestRegressor(**common), "r2"


def _split(
    rows_y: np.ndarray, classification: bool, chronological_order: Optional[np.ndarray], settings: Settings
):
    """Positions (within the usable rows) for training and held-out testing."""
    n = rows_y.size
    n_test = max(MIN_TEST_ROWS, int(round(n * settings.importance_test_fraction)))
    n_test = min(n_test, n - MIN_TEST_ROWS)
    if chronological_order is not None:
        order = chronological_order
        return order[: n - n_test], order[n - n_test:], "chronological"
    from sklearn.model_selection import train_test_split

    positions = np.arange(n)
    stratify = None
    if classification:
        _, counts = np.unique(rows_y, return_counts=True)
        if counts.min() >= 2 and n_test >= counts.size:
            stratify = rows_y
    train, test = train_test_split(
        positions, test_size=n_test, random_state=settings.analysis_random_seed, stratify=stratify)
    return np.sort(train), np.sort(test), "random" + (" (stratified)" if stratify is not None else "")


def _time_order(columns: list[TypedColumn], rows: np.ndarray) -> Optional[np.ndarray]:
    for column in columns:
        if column.ctype is ColumnType.DATETIME and column.datetimes is not None:
            stamps = np.array(
                [np.nan if column.datetimes[i] is None else column.datetimes[i].timestamp() for i in rows])
            if not np.isnan(stamps).all():
                return np.argsort(np.where(np.isnan(stamps), np.inf, stamps), kind="stable")
    return None


def build_feature_intelligence(
    parsed: ParsedResult,
    understanding: DatasetMLUnderstanding,
    settings: Settings,
    deadline: Optional[Deadline] = None,
) -> FeatureIntelligence:
    target = understanding.best_target
    task = understanding.primary_task
    if target is None or task is None:
        return unavailable(
            "There is no target to relate the features to. Name the target column yourself, or "
            "choose an unsupervised task.", None, task.task if task else None)
    if task.task not in _IMPORTANCE_TASKS:
        return unavailable(
            f"Feature importance is measured for classification, regression and forecasting, not "
            f"for {task.task.value.replace('_', ' ')}.", target.column, task.task)

    by_name = {c.name: c for c in parsed.columns}
    target_col = by_name.get(target.column)
    if target_col is None:
        return unavailable("The target column is not in the parsed file.", target.column, task.task)

    classification = task.task in (MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION)
    indices = sample_indices(parsed.n_rows, settings.importance_max_rows, settings.analysis_random_seed)
    sampled = parsed.n_rows > settings.importance_max_rows

    y_all = _target_vector(target_col, indices, classification)
    if y_all is None:
        return unavailable("The target column could not be read as a usable target.", target.column, task.task)
    valid = _usable_rows(y_all, classification)
    rows = indices[valid]
    y = y_all[valid]
    if rows.size < settings.task_min_rows_supervised:
        return unavailable(
            f"Only {rows.size} rows have a value for '{target.column}': too few to measure importance "
            f"reliably (at least {settings.task_min_rows_supervised} are needed).", target.column, task.task)

    if classification:
        classes, counts = np.unique(y, return_counts=True)
        if classes.size < 2:
            return unavailable("The target has a single value, so there is nothing to predict.", target.column, task.task)
        # A class with one member can be neither split nor scored; set those rows aside.
        keep = np.isin(y, classes[counts >= 2])
        rows, y = rows[keep], y[keep]
        if np.unique(y).size < 2 or rows.size < settings.task_min_rows_supervised:
            return unavailable(
                "Too few rows per class to measure importance reliably.", target.column, task.task)

    leaky = _leaky_columns(parsed)
    assessments: dict[str, FeatureAssessment] = {}
    modelable: list[TypedColumn] = []
    associations: dict[str, object] = {}

    for column in parsed.columns:
        if column is target_col:
            continue
        if deadline is not None:
            deadline.check("feature analysis")
        profile = column.profile
        missing = profile.missing_pct if profile else None
        flags = set(column.flags)
        result = association(column, target_col, rows, settings.task_max_classes)
        associations[column.name] = result
        base = dict(column=column.name, missing_pct=missing)
        measured = dict(
            association=None if result is None else result.value,
            association_method=None if result is None else result.method,
        )

        if task.task is MLTask.TIME_SERIES_FORECASTING and column.ctype is ColumnType.DATETIME:
            assessments[column.name] = FeatureAssessment(
                role=FeatureRole.UNKNOWN, evidence_level=EvidenceLevel.DATAPILOT_PARSED,
                reasons=["it is the time axis: it orders the data for forecasting and is not scored as a predictor"],
                **base)
            continue
        if ColumnFlag.POSSIBLE_IDENTIFIER in flags:
            assessments[column.name] = FeatureAssessment(
                role=FeatureRole.IDENTIFIER, evidence_level=EvidenceLevel.DATAPILOT_PARSED,
                reasons=[(column.identifier_evidence[1] if column.identifier_evidence
                          else "its values identify rows rather than describe them")],
                **base)
            continue
        leak_reason = leaky.get(column.name)
        if leak_reason is None and result is not None and looks_like_leakage(result, settings.quality_leakage_association):
            leak_reason = (f"alone it predicts '{target.column}' almost perfectly "
                           f"({result.method} {result.value:.3f})")
        if leak_reason is not None:
            assessments[column.name] = FeatureAssessment(
                role=FeatureRole.POSSIBLE_LEAKAGE, evidence_level=EvidenceLevel.DATAPILOT_PARSED,
                reasons=[leak_reason],
                cautions=["kept out of the importance model so it cannot hide the other features; "
                          "check it is known before the outcome, and not derived from it"],
                **base, **measured)
            continue
        if ColumnFlag.CONSTANT in flags or ColumnFlag.ALL_MISSING in flags:
            assessments[column.name] = FeatureAssessment(
                role=FeatureRole.LOW_SIGNAL, evidence_level=EvidenceLevel.DATAPILOT_PARSED,
                reasons=["every value is missing" if ColumnFlag.ALL_MISSING in flags else "it has a single value"],
                **base, **measured)
            continue
        if column.ctype in (ColumnType.TEXT, ColumnType.UNKNOWN):
            assessments[column.name] = FeatureAssessment(
                role=FeatureRole.UNKNOWN, evidence_level=EvidenceLevel.DATAPILOT_PARSED,
                reasons=["free text (or an unrecognised type): not assessed here; it would need text features"],
                **base, **measured)
            continue
        modelable.append(column)

    # --- choose what goes into the model -----------------------------------------------
    cap = settings.importance_max_features
    ranked = sorted(
        modelable,
        key=lambda c: (-(associations[c.name].value if associations[c.name] is not None else -1.0), c.name))
    modelled = ranked[:cap]
    skipped = ranked[cap:]

    if not modelable:
        features = sorted(assessments.values(), key=lambda f: (-(f.association or 0.0), f.column))
        return FeatureIntelligence(
            available=False, target=target.column, task=task.task, features=features,
            role_counts=_counts(features),
            unavailable_reason="No column is usable as a numeric or categorical feature (identifiers, "
                               "constants, free text and leaking columns are set aside), so there is "
                               "nothing to measure.")

    method: Optional[ImportanceMethod] = None
    importance: dict[str, tuple[float, float]] = {}
    mutual: dict[str, float] = {}
    if modelled:
        method, importance, mutual = _measure(
            modelled, target_col, rows, y, classification, task.task, parsed, settings, deadline,
            skipped=len(skipped), available=parsed.n_rows, sampled=sampled, confirmed=understanding.target_is_confident,
            target_name=target.column)

    if method is not None and not method.reliable:
        importance = {}  # a model that does not beat a trivial predictor gives no ranking worth showing
    reliable = bool(method and method.reliable)
    noise = max(method.noise_floor, 0.0) if method and method.noise_floor is not None else 0.0
    noise_mi = method.noise_mutual_information if method and method.noise_mutual_information is not None else 0.0
    mi_bar = max(MI_NOISE_MARGIN * noise_mi, MI_FLOOR)
    positive_total = sum(max(m, 0.0) for m, _ in importance.values())

    for column in modelable:
        result = associations[column.name]
        profile = column.profile
        flags = set(column.flags)
        mi = mutual.get(column.name)
        measured = dict(
            association=None if result is None else result.value,
            association_method=None if result is None else result.method,
            mutual_information=mi,
            missing_pct=profile.missing_pct if profile else None,
        )
        mean, std = importance.get(column.name, (None, None))

        # Concerns that stop a column being recommended outright.
        blocking: list[str] = []
        # Notes that are worth saying but do not change the role.
        notes: list[str] = []
        if column.sensitive_evidence or ColumnFlag.POSSIBLE_SENSITIVE_FIELD in flags:
            blocking.append("looks like a sensitive personal attribute: using it may raise privacy or fairness concerns")
        if profile is not None and profile.missing_pct > HIGH_MISSING_PCT:
            blocking.append(f"{profile.missing_pct:.0f}% of its values are missing")
        if ColumnFlag.HIGH_CARDINALITY in flags:
            blocking.append("many distinct categories: risk of overfitting; consider grouping rare values")
        if ColumnFlag.MIXED_TYPES in flags:
            blocking.append("mixed value types: clean it before use")
        if column in skipped:
            blocking.append(f"not included in the importance model (only the {cap} most associated columns are)")

        # Three independent measures; a column is recommended when at least two agree.
        evidence_for: list[str] = []
        evidence_against: list[str] = []
        if mean is not None and reliable:
            if mean > noise and mean > IMPORTANCE_SIGMAS * (std or 0.0):
                evidence_for.append(f"shuffling it costs the held-out model {mean:.3f} ({method.scoring}), "
                                    f"above the random-column yardstick of {noise:.3f}")
            else:
                evidence_against.append("shuffling it hurts the held-out model no more than shuffling a random column")
        if result is not None:
            text = f"{result.method.replace('_', ' ')} with '{target.column}' is {result.value:.2f}"
            (evidence_for if result.value >= ASSOCIATION_STRONG else evidence_against).append(text)
        if mi is not None:
            if mi > mi_bar:
                evidence_for.append(f"mutual information {mi:.3f} nats, above the random-column bar of {mi_bar:.3f}")
            else:
                evidence_against.append(f"mutual information {mi:.3f} nats is not above the random-column bar of {mi_bar:.3f}")

        votes = len(evidence_for)
        if result is None and mi is None:
            role = FeatureRole.UNKNOWN
            reasons = ["its relationship with the target could not be measured validly"]
        elif reliable and mean is not None:
            if votes >= 2:
                role, reasons = FeatureRole.RECOMMENDED_FEATURE, evidence_for
            elif votes == 1:
                role = FeatureRole.USE_WITH_CAUTION
                reasons = evidence_for + ["but the other measures do not confirm it"]
            else:
                role, reasons = FeatureRole.LOW_SIGNAL, evidence_against
        else:
            # No trustworthy model ranking: association and mutual information alone cannot recommend.
            if votes >= 1:
                role = FeatureRole.USE_WITH_CAUTION
                reasons = evidence_for + (
                    ["the model on this sample did not beat a trivial predictor, so it cannot confirm the ranking"]
                    if method is not None and not reliable else [])
            else:
                role, reasons = FeatureRole.LOW_SIGNAL, evidence_against

        if role is FeatureRole.RECOMMENDED_FEATURE and blocking:
            role = FeatureRole.USE_WITH_CAUTION
        if (role is FeatureRole.RECOMMENDED_FEATURE and mean is not None and positive_total > 0
                and mean / positive_total > DOMINANCE_SHARE and (method.held_out_score or 0) > DOMINANCE_MIN_SCORE):
            notes.append("it carries almost all of the model's signal: confirm it is known when a prediction is needed")
        assessments[column.name] = FeatureAssessment(
            column=column.name, role=role, reasons=reasons, cautions=blocking + notes,
            evidence_level=EvidenceLevel.DATAPILOT_PARSED,
            permutation_importance=None if mean is None else round(mean, 4),
            permutation_importance_std=None if std is None else round(std, 4),
            **measured)

    ordered_names = [
        name for name, _ in sorted(importance.items(), key=lambda kv: (-kv[1][0], kv[0]))
    ]
    for rank, name in enumerate(ordered_names, start=1):
        assessments[name] = assessments[name].model_copy(update={"importance_rank": rank})

    features = sorted(
        assessments.values(),
        key=lambda f: (f.importance_rank is None, f.importance_rank or 0,
                       -(f.association or 0.0), f.column))
    return FeatureIntelligence(
        available=True, target=target.column, task=task.task, method=method,
        features=features, role_counts=_counts(features))


def _leaky_columns(parsed: ParsedResult) -> dict[str, str]:
    out: dict[str, str] = {}
    for finding in parsed.quality.findings:
        if finding.code in (QualityCode.LEAKAGE_SIGNAL, QualityCode.POTENTIAL_LEAKAGE) and finding.column:
            out.setdefault(finding.column, finding.detail.split(". Check")[0].split(" That is a hint")[0].strip())
    return out


def _measure(
    modelled: list[TypedColumn],
    target_col: TypedColumn,
    rows: np.ndarray,
    y: np.ndarray,
    classification: bool,
    task: MLTask,
    parsed: ParsedResult,
    settings: Settings,
    deadline: Optional[Deadline],
    *,
    skipped: int,
    available: int,
    sampled: bool,
    confirmed: bool,
    target_name: str,
):
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import get_scorer

    seed = settings.analysis_random_seed
    X = _feature_matrix(modelled, rows)
    rng = np.random.default_rng(seed)
    # Several random columns, not one: with dozens of features, a single lucky noise column
    # would make the yardstick too easy to clear and let noise features through.
    n_noise = min(MAX_NOISE_COLUMNS, max(MIN_NOISE_COLUMNS, len(modelled) // 4))
    noise_columns = rng.normal(size=(rows.size, n_noise))
    X_all = np.hstack([X, noise_columns])
    noise_names = [f"__noise_{i}__" for i in range(n_noise)]
    names = [c.name for c in modelled] + noise_names

    # ---- mutual information (descriptive, over every usable row) ----------------------
    if deadline is not None:
        deadline.check("mutual information")
    medians = np.array([np.nanmedian(X[:, j]) if not np.isnan(X[:, j]).all() else 0.0 for j in range(X.shape[1])])
    filled = np.where(np.isnan(X), medians, X)
    discrete = _discrete_mask(modelled)
    if classification:
        mi = mutual_info_classif(filled, y, discrete_features=discrete, random_state=seed)
        noise_mi = float(np.max(mutual_info_classif(noise_columns, y, discrete_features=False, random_state=seed)))
    else:
        mi = mutual_info_regression(filled, y, discrete_features=discrete, random_state=seed)
        noise_mi = float(np.max(mutual_info_regression(noise_columns, y, discrete_features=False, random_state=seed)))
    mutual = {c.name: round(float(max(v, 0.0)), 4) for c, v in zip(modelled, mi)}

    # ---- held-out model ---------------------------------------------------------------
    order = _time_order(parsed.columns, rows) if task is MLTask.TIME_SERIES_FORECASTING else None
    train, test, split = _split(y, classification, order, settings)
    n_classes = int(np.unique(y).size) if classification else 0
    model, scoring = _model_and_score(classification, n_classes == 2, settings)
    if deadline is not None:
        deadline.check("importance model")
    model.fit(X_all[train], y[train])
    classes_in_test = int(np.unique(y[test]).size) if classification else 0
    one_class_test = classification and classes_in_test < 2  # no score can be computed on it
    if one_class_test:
        held_out, reliable, importance, noise_floor = float("nan"), False, {}, 0.0
    else:
        held_out = float(get_scorer(scoring)(model, X_all[test], y[test]))
        reliable = held_out >= (1.0 / n_classes if classification else 0.0) + SKILL_MARGIN
        if deadline is not None:
            deadline.check("permutation importance")
        result = permutation_importance(
            model, X_all[test], y[test], scoring=scoring,
            n_repeats=settings.importance_n_repeats, random_state=seed, n_jobs=1)
        importance = {
            name: (float(mean), float(std))
            for name, mean, std in zip(names, result.importances_mean, result.importances_std)
        }
        noise_floor = max(importance.pop(name)[0] for name in noise_names)
    baseline = (0.5 if n_classes == 2 else 1.0 / n_classes) if classification else 0.0
    description = (
        f"Random forest ({settings.importance_n_estimators} trees, min leaf 3) fitted on {train.size} rows; "
        f"importance is the mean drop in held-out {scoring} over {settings.importance_n_repeats} shuffles of each "
        f"column, on {test.size} held-out rows. {n_noise} random columns are included as yardsticks."
    )
    if not confirmed:
        description += f" '{target_name}' is a possible target, not a confirmed one."
    if not reliable:
        description += (" The model did not beat a trivial predictor on the held-out rows, so per-feature "
                        "importances are withheld rather than shown as a ranking.")
    method = ImportanceMethod(
        description=description,
        model="RandomForestClassifier" if classification else "RandomForestRegressor",
        scoring=scoring, rows_used=int(rows.size), rows_available=int(available), sampled=sampled,
        features_modelled=len(modelled), features_skipped=skipped,
        held_out_score=None if one_class_test else round(held_out, 4), baseline_score=round(baseline, 4),
        noise_floor=round(noise_floor, 4), noise_columns=n_noise,
        noise_mutual_information=round(float(max(noise_mi, 0.0)), 4),
        reliable=bool(reliable), split=split,
        n_repeats=settings.importance_n_repeats, seed=seed)
    return method, importance, mutual

