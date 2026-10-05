"""Which models to try for the task the data supports, and why, in terms of this dataset.

The advice is a function of the task and of facts measured about the data (rows, column types,
missingness, class balance), and every recommendation says which fact it rests on. It never
predicts a score: a model is *worth trying*, not *expected to reach* anything, and the response
carries a statement to that effect.
"""

from __future__ import annotations

from typing import Optional

from app.schemas.dataset_analysis import ColumnType, MLTask, QualityCode
from app.schemas.project_intelligence import DatasetMLUnderstanding, ModelRecommendation, ModelTier
from app.services.project_intelligence.shape import DataShape

NO_PROMISE = (
    "No score is implied for any model here: what a model achieves depends on the data, and only a "
    "held-out evaluation on your own data can show it."
)

SMALL_DATA_ROWS = 200
MISSING_NOTABLE_PCT = 5.0


def _rows_phrase(shape: DataShape) -> str:
    return f"{shape.n_rows:,} rows" if shape.n_rows is not None else "an unknown number of rows"


def _imbalanced(shape: DataShape) -> bool:
    return bool(shape.findings_with(QualityCode.CLASS_IMBALANCE))


def recommend_models(understanding: DatasetMLUnderstanding, shape: DataShape) -> list[ModelRecommendation]:
    task = understanding.primary_task.task if understanding.primary_task else None
    if task is None:
        return []
    small = shape.n_rows is not None and shape.n_rows < SMALL_DATA_ROWS
    missing = shape.parsed and shape.max_missing_pct > MISSING_NOTABLE_PCT
    categorical = [c for c in shape.columns if c.kind in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN) and c.usable_feature]
    high_card = [c.name for c in shape.columns if c.high_cardinality and c.usable_feature]
    has_text = any(c.kind is ColumnType.TEXT for c in shape.columns)

    out: list[ModelRecommendation] = []
    if task in (MLTask.BINARY_CLASSIFICATION, MLTask.MULTICLASS_CLASSIFICATION):
        binary = task is MLTask.BINARY_CLASSIFICATION
        imbalanced = _imbalanced(shape)
        out.append(ModelRecommendation(
            model="Majority-class baseline (dummy classifier)", tier=ModelTier.BASELINE,
            why="Every model must beat predicting the most common class; without it a score means nothing."
                + (" With imbalanced classes this baseline can look deceptively good on accuracy." if imbalanced else ""),
        ))
        logistic_pre = ["scale numeric columns", "one-hot encode categorical columns"]
        if missing:
            logistic_pre.append("impute missing values (the dataset has columns with missing values)")
        out.append(ModelRecommendation(
            model="Logistic regression" if binary else "Multinomial logistic regression", tier=ModelTier.BASELINE,
            why="A simple, interpretable first model; its coefficients show which columns matter and give "
                f"a bar for stronger models to clear. Suitable for {_rows_phrase(shape)}.",
            preprocessing=logistic_pre,
            cautions=(["use class weights: the classes are imbalanced"] if imbalanced else []),
        ))
        if small:
            out.append(ModelRecommendation(
                model="Regularised linear model or a shallow decision tree", tier=ModelTier.STRONGER,
                why=f"With {_rows_phrase(shape)}, flexible models overfit easily; keep capacity low and "
                    "judge by cross-validation rather than a single split.",
                cautions=["a single train/test split of so few rows gives a very noisy estimate"]))
        else:
            out.append(ModelRecommendation(
                model="Gradient-boosted trees (e.g. scikit-learn HistGradientBoosting, XGBoost, LightGBM)",
                tier=ModelTier.STRONGER,
                why="Usually strong on tabular data, handles mixed column types and non-linear effects"
                    + ("; HistGradientBoosting accepts missing values natively" if missing else "") + ".",
                preprocessing=(["encode categorical columns (ordinal or native categorical support)"] if categorical else []),
                cautions=(["tune class weights or the decision threshold: the classes are imbalanced"] if imbalanced else [])))
            out.append(ModelRecommendation(
                model="Random forest", tier=ModelTier.STRONGER,
                why="A robust, low-tuning ensemble that makes a good second opinion and gives permutation importances.",
            ))
        if has_text:
            out.append(ModelRecommendation(
                model="TF-IDF features with logistic regression", tier=ModelTier.SPECIALISED,
                why="The file has free-text column(s) that tabular models ignore; TF-IDF plus a linear model "
                    "is the standard first text baseline."))
        if imbalanced:
            out.append(ModelRecommendation(
                model="Cost-sensitive learning with threshold tuning", tier=ModelTier.SPECIALISED,
                why="The minority class is rare; choose the decision threshold on precision/recall, not accuracy.",
                cautions=["report precision, recall and PR-AUC rather than accuracy"]))
    elif task is MLTask.REGRESSION:
        out.append(ModelRecommendation(
            model="Mean-prediction baseline (dummy regressor)", tier=ModelTier.BASELINE,
            why="The score of always predicting the average; any useful model must beat it."))
        out.append(ModelRecommendation(
            model="Linear or ridge regression", tier=ModelTier.BASELINE,
            why=f"An interpretable baseline that shows linear effects; ridge keeps it stable with {_rows_phrase(shape)}.",
            preprocessing=["scale numeric columns", "one-hot encode categorical columns"]
            + (["impute missing values"] if missing else []),
            cautions=["check residual plots: a skewed target often needs a log transform"]))
        if small:
            out.append(ModelRecommendation(
                model="Regularised regression or a shallow tree", tier=ModelTier.STRONGER,
                why=f"With {_rows_phrase(shape)}, prefer simple models and cross-validation."))
        else:
            out.append(ModelRecommendation(
                model="Gradient-boosted trees (e.g. HistGradientBoosting, XGBoost, LightGBM)", tier=ModelTier.STRONGER,
                why="Captures non-linear effects and interactions on tabular data"
                    + ("; HistGradientBoosting accepts missing values natively" if missing else "") + "."))
            out.append(ModelRecommendation(
                model="Random forest regressor", tier=ModelTier.STRONGER,
                why="A robust ensemble with little tuning; cannot extrapolate beyond the range of the training target."))
    elif task is MLTask.TIME_SERIES_FORECASTING:
        out.append(ModelRecommendation(
            model="Seasonal-naive / last-value baseline", tier=ModelTier.BASELINE,
            why="Repeating the last value (or the same period last cycle) is the bar every forecast must clear."))
        out.append(ModelRecommendation(
            model="Exponential smoothing or ARIMA-family model", tier=ModelTier.BASELINE,
            why="Classical models for a single series with trend and seasonality; fast and interpretable.",
            cautions=["needs a regular time index; irregular timestamps must be resampled first"]))
        out.append(ModelRecommendation(
            model="Gradient-boosted trees on lag and calendar features", tier=ModelTier.STRONGER,
            why="Lets other columns (such as promotions) inform the forecast alongside past values.",
            preprocessing=["build lag and rolling-window features", "add calendar features (weekday, month)"],
            cautions=["validate chronologically (walk-forward), never with a random split",
                      "tree models cannot extrapolate a trend: de-trend or difference the target"]))
    elif task is MLTask.CLUSTERING:
        out.append(ModelRecommendation(
            model="k-means", tier=ModelTier.BASELINE,
            why="The standard first clustering; choose k with silhouette scores and domain sense.",
            preprocessing=["scale numeric columns", "encode or drop categorical columns"],
            cautions=["no labels exist to validate against: judge clusters by stability and interpretability"]))
        out.append(ModelRecommendation(
            model="Gaussian mixture model or hierarchical clustering", tier=ModelTier.STRONGER,
            why="Handles elongated clusters and gives soft assignments or a dendrogram to inspect."))
    elif task is MLTask.ANOMALY_DETECTION:
        out.append(ModelRecommendation(
            model="Robust z-score / IQR rules", tier=ModelTier.BASELINE,
            why="Transparent per-column outlier rules; also the baseline other detectors should beat."))
        out.append(ModelRecommendation(
            model="Isolation forest", tier=ModelTier.STRONGER,
            why="A fast, scalable detector that needs no labels and copes with many columns.",
            cautions=["flagged rows are unusual, not necessarily wrong: review a sample before acting"]))
        out.append(ModelRecommendation(
            model="Local outlier factor", tier=ModelTier.STRONGER,
            why="Finds points that are unusual relative to their neighbourhood rather than globally."))
    elif task is MLTask.MULTILABEL_CLASSIFICATION:
        out.append(ModelRecommendation(
            model="One-vs-rest logistic regression", tier=ModelTier.BASELINE,
            why="Fits one binary classifier per label: a transparent multi-label baseline."))
        out.append(ModelRecommendation(
            model="Random forest (native multi-output)", tier=ModelTier.STRONGER,
            why="Handles several labels at once without one model per label."))
    elif task is MLTask.RANKING:
        out.append(ModelRecommendation(
            model="Gradient-boosted ranking (e.g. LightGBM or XGBoost LambdaRank)", tier=ModelTier.SPECIALISED,
            why="Learning-to-rank models optimise the ordering within each group directly.",
            cautions=["rows must be grouped by query; confirm which column defines the group"]))

    if high_card:
        out.append(ModelRecommendation(
            model="Target or frequency encoding for high-cardinality columns", tier=ModelTier.SPECIALISED,
            why=f"{', '.join(high_card[:3])} {'has' if len(high_card) == 1 else 'have'} many distinct values, "
                "which one-hot encoding turns into hundreds of columns.",
            cautions=["fit the encoder inside cross-validation to avoid leaking the target"]))
    return out[:7]
