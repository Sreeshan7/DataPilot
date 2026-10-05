"""What is generally known about each method DataPilot can recognise in the literature.

This is textbook-level guidance written by DataPilot, not something read from a retrieved paper and not AI output. It
gives every algorithm in the analysis a real profile (what it is, where it shines, where it struggles, what to prepare,
what to tune) even when no AI is available, and it grounds the AI's own analysis: the AI is asked to relate this
knowledge to the user's dataset and to the papers, not to replace it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    family: str                 # boosting | forest | tree | linear | svm | distance | bayes | neural | sequence | timeseries | cluster | anomaly | ensemble
    what: str
    strengths: tuple[str, ...]
    weaknesses: tuple[str, ...]
    preprocessing: tuple[str, ...]
    tuning: tuple[str, ...]


PROFILES: dict[str, Profile] = {
    "XGBoost": Profile(
        "boosting", "Gradient-boosted decision trees with built-in regularisation, engineered for speed and accuracy on tabular data.",
        ("Often the strongest default on structured data", "Handles missing values and mixed feature scales natively", "Built-in regularisation helps limit overfitting"),
        ("Many hyperparameters, and defaults can overfit small datasets", "Harder to interpret than a single tree or a linear model", "Categorical columns must be encoded first"),
        ("Encode categorical columns (one-hot or target encoding)", "No feature scaling is needed", "Remove identifiers and leaking columns before training"),
        ("`learning_rate` together with `n_estimators` (use early stopping)", "`max_depth` and `min_child_weight`", "`subsample` and `colsample_bytree`", "`reg_lambda` / `reg_alpha`")),
    "LightGBM": Profile(
        "boosting", "A fast gradient-boosting library that grows trees leaf-wise and bins features for speed on large tables.",
        ("Very fast on large datasets and many features", "Native handling of categorical columns", "Competitive accuracy with low memory use"),
        ("Leaf-wise growth overfits small datasets easily", "Sensitive to `num_leaves` and `min_data_in_leaf`", "Less interpretable than simpler models"),
        ("Declare categorical columns instead of one-hot encoding them", "No scaling needed", "Use a time-aware split if rows are ordered in time"),
        ("`num_leaves` and `max_depth`", "`min_data_in_leaf`", "`learning_rate` with early stopping", "`feature_fraction` and `bagging_fraction`")),
    "CatBoost": Profile(
        "boosting", "Gradient boosting with ordered target statistics, designed to handle categorical columns well with little tuning.",
        ("Strong out-of-the-box results", "Best-in-class handling of high-cardinality categorical columns", "Ordered boosting reduces target leakage in encodings"),
        ("Slower to train than LightGBM", "Larger models and heavier memory use", "Fewer community recipes than XGBoost"),
        ("Pass categorical columns by name", "No scaling needed", "Keep a clean validation set for early stopping"),
        ("`depth`", "`learning_rate` with early stopping", "`l2_leaf_reg`", "`iterations`")),
    "Gradient Boosting": Profile(
        "boosting", "Builds an additive model of shallow trees, each one correcting the errors of the previous ones.",
        ("Accurate on many tabular problems", "Works with mixed feature types after encoding", "Flexible loss functions"),
        ("Sequential training is slower than bagging", "Sensitive to hyperparameters and noisy labels", "Can overfit without early stopping"),
        ("Encode categorical columns", "Impute missing values (the classic implementation does not handle them)", "No scaling needed"),
        ("`n_estimators` and `learning_rate`", "`max_depth`", "`subsample`", "`min_samples_leaf`")),
    "Random Forest": Profile(
        "forest", "An ensemble of decorrelated decision trees trained on bootstrap samples; predictions are averaged or voted.",
        ("Robust with little tuning", "Resists overfitting better than a single tree", "Gives permutation importances and out-of-bag estimates"),
        ("Often slightly behind boosting on tabular accuracy", "Cannot extrapolate beyond the range of the training targets (regression)", "Large models are slow to predict and hard to explain"),
        ("Encode categorical columns", "No scaling needed", "Impute missing values"),
        ("`n_estimators`", "`max_features`", "`min_samples_leaf`", "`max_depth`")),
    "Decision Tree": Profile(
        "tree", "Splits the data with a sequence of simple rules, giving a model a person can read.",
        ("Fully interpretable", "No scaling needed and handles non-linear relationships", "Fast to train and predict"),
        ("Overfits easily when grown deep", "Unstable: small data changes can change the tree", "Usually beaten by ensembles on accuracy"),
        ("Encode categorical columns", "Impute missing values", "No scaling needed"),
        ("`max_depth`", "`min_samples_leaf`", "`ccp_alpha` (pruning)")),
    "Logistic Regression": Profile(
        "linear", "A linear model for classification that outputs calibrated class probabilities.",
        ("Fast, stable and easy to explain through coefficients", "Good probability estimates", "A strong baseline that is hard to overfit"),
        ("Assumes a linear decision boundary unless features are engineered", "Sensitive to correlated inputs", "Needs scaling when regularised"),
        ("Scale numeric columns", "One-hot encode categorical columns", "Check for multicollinearity"),
        ("`C` (inverse regularisation strength)", "`penalty` (l1 / l2 / elastic net)", "`class_weight` for imbalance")),
    "Linear Regression": Profile(
        "linear", "Fits a weighted sum of the inputs to predict a number; ridge and lasso add regularisation.",
        ("Simple, fast and interpretable", "An honest baseline every regression project needs", "Ridge/lasso cope with many or correlated inputs"),
        ("Assumes a linear relationship", "Sensitive to outliers and strongly correlated inputs", "Cannot capture interactions unless added by hand"),
        ("Scale numeric columns (needed for ridge/lasso)", "One-hot encode categorical columns", "Inspect and treat outliers"),
        ("`alpha` for ridge or lasso", "Interaction and polynomial terms", "Log-transform a skewed target")),
    "Support Vector Machine": Profile(
        "svm", "Finds the boundary (or regression tube) with the widest margin, optionally in a kernel-induced space.",
        ("Effective in high-dimensional spaces", "Works well on small to medium clean datasets", "Kernels capture non-linear patterns"),
        ("Training slows sharply beyond tens of thousands of rows", "Needs scaling and careful tuning", "No direct probabilities and hard to interpret"),
        ("Scale all numeric columns", "Encode categorical columns", "Consider a subsample for tuning on large data"),
        ("`C`", "`kernel` and `gamma`", "`epsilon` (for regression)")),
    "k-Nearest Neighbors": Profile(
        "distance", "Predicts from the labels or values of the most similar training rows.",
        ("No training step and trivial to explain", "Naturally non-linear", "A useful sanity-check baseline"),
        ("Prediction is slow on large data", "Very sensitive to feature scale and irrelevant columns", "Degrades in high dimensions"),
        ("Scale numeric columns", "Drop or down-weight irrelevant columns", "Encode categorical columns carefully"),
        ("`n_neighbors`", "Distance metric", "`weights` (uniform or distance)")),
    "Naive Bayes": Profile(
        "bayes", "Applies Bayes' rule assuming the inputs are independent given the class.",
        ("Extremely fast and works with very little data", "A strong baseline for text", "Handles many features easily"),
        ("The independence assumption is rarely true", "Probabilities are often poorly calibrated", "Usually beaten by tuned discriminative models"),
        ("Choose the variant to match the data (Gaussian, multinomial, Bernoulli)", "For text, use counts or TF-IDF", "Discretise or scale only if the variant needs it"),
        ("Smoothing parameter `alpha`", "Variant choice", "Feature selection")),
    "Neural Network": Profile(
        "neural", "A multi-layer perceptron: stacked layers of weighted sums and non-linearities trained by gradient descent.",
        ("Can model complex non-linear interactions", "Flexible and extendable", "Benefits from large datasets"),
        ("Needs more data and tuning than tree ensembles on tables", "Sensitive to feature scaling and initialisation", "Hard to interpret"),
        ("Scale numeric columns", "Embed or one-hot encode categoricals", "Hold out a validation set for early stopping"),
        ("Layer count and width", "`learning_rate` and weight decay", "Dropout and early stopping")),
    "Deep Learning": Profile(
        "neural", "Deep neural networks that learn their own features from raw sequences, images or text.",
        ("Learns features directly from raw data", "State of the art on images, audio and text", "Transfer learning reduces data needs"),
        ("Needs a lot of data and compute", "Hard to interpret and to debug", "Often not better than gradient boosting on plain tables"),
        ("Normalise inputs", "Augment data where possible", "Use pretrained models for images and text"),
        ("Architecture and depth", "`learning_rate`, batch size and schedule", "Regularisation (dropout, weight decay, augmentation)")),
    "LSTM": Profile(
        "sequence", "A recurrent network with gates that can remember information across long stretches of a sequence.",
        ("Captures long-range temporal dependencies", "Handles several input series at once", "Flexible for forecasting and sequence labelling"),
        ("Needs enough data and slow to train", "Easy to overfit on short series", "Often not better than a tuned boosting model with lag features"),
        ("Scale each series", "Cut the data into fixed-length windows", "Split by time, never at random"),
        ("Window length", "Hidden size and layer count", "Dropout and `learning_rate`", "Early stopping")),
    "Transformer": Profile(
        "neural", "An attention-based network that relates every position of an input to every other one.",
        ("Captures long-range context", "Pretrained models give strong results with modest data", "Parallelises well on GPUs"),
        ("Heavy compute and memory", "Overfits small datasets when trained from scratch", "Hard to interpret"),
        ("Use the model's own tokeniser for text", "Truncate or window long inputs", "Fine-tune a pretrained checkpoint where one exists"),
        ("Pretrained checkpoint choice", "Small `learning_rate` with warm-up", "Epochs and maximum sequence length")),
    "Convolutional Neural Network": Profile(
        "neural", "A network that slides learned filters over images, signals or sequences to detect local patterns.",
        ("Excellent for images and other grid-like data", "Parameter sharing keeps models compact", "Transfer learning works well"),
        ("Needs substantial labelled data or a pretrained base", "Compute-hungry", "Hard to interpret"),
        ("Normalise and resize inputs", "Augment images (flips, crops)", "Start from a pretrained backbone"),
        ("Architecture or backbone", "`learning_rate` and schedule", "Augmentation strength")),
    "ARIMA": Profile(
        "timeseries", "A classical statistical model that predicts a series from its own past values and past errors.",
        ("Interpretable and works with modest amounts of data", "Provides prediction intervals", "A strong, hard-to-beat baseline for single series"),
        ("Assumes the series can be made stationary", "Univariate unless regressors are added (ARIMAX)", "Struggles with multiple seasonalities and sudden shifts"),
        ("Check stationarity and difference the series", "Fix gaps and irregular timestamps", "Split by time"),
        ("Orders `(p, d, q)` and seasonal `(P, D, Q, s)` (use AIC or an automatic search)", "Whether to add external regressors")),
    "Prophet": Profile(
        "timeseries", "An additive forecasting model with trend, seasonality and holiday effects, built to be easy to use.",
        ("Handles missing data and outliers gracefully", "Holidays and several seasonalities are built in", "Quick to fit and easy to explain"),
        ("Less accurate than tuned models on complex dynamics", "Extra regressors must be known for the future", "Limited to a single series at a time"),
        ("Provide `ds` and `y` columns", "Mark holidays and known events", "Keep the time order for validation"),
        ("`changepoint_prior_scale`", "`seasonality_mode` (additive or multiplicative)", "`seasonality_prior_scale` and holidays")),
    "K-Means": Profile(
        "cluster", "Groups rows into k clusters by minimising the distance to each cluster's centre.",
        ("Simple, fast and scalable", "Easy to explain with cluster profiles", "A sensible first clustering method"),
        ("Assumes round, similarly sized clusters", "You must choose k in advance", "Sensitive to scale, outliers and initialisation"),
        ("Scale numeric columns", "Encode or leave out categorical columns", "Log-transform skewed columns"),
        ("`n_clusters` (compare silhouette and elbow)", "`n_init`", "The scaling choice")),
    "Isolation Forest": Profile(
        "anomaly", "Isolates unusual rows by how quickly random splits separate them from the rest.",
        ("Fast, scales well and needs few assumptions", "Works without labels", "Handles many features"),
        ("You must pick a contamination level or threshold", "Scores are hard to explain", "Weaker on local or clustered anomalies"),
        ("Remove identifiers", "Encode categoricals if they matter", "No scaling needed"),
        ("`n_estimators`", "`max_samples`", "`contamination` (or a score threshold)")),
    "AdaBoost": Profile(
        "boosting", "Builds a strong classifier by re-weighting rows so each new weak learner focuses on earlier mistakes.",
        ("Simple and needs little tuning", "Works well with shallow trees as base learners", "Can rank features by usage"),
        ("Sensitive to noisy labels and outliers", "Usually beaten by modern gradient boosting", "Sequential and not parallelisable"),
        ("Clean outliers and label noise", "Encode categoricals", "No scaling needed"),
        ("`n_estimators`", "`learning_rate`", "Depth of the base learner")),
    "Ensemble Methods": Profile(
        "ensemble", "Combines several different models (voting, blending or stacking) so their errors partly cancel.",
        ("Usually gives a small, reliable accuracy gain", "Reduces variance across models", "Lets different model families complement each other"),
        ("More complex to build and maintain", "Stacking leaks information if done without out-of-fold predictions", "Harder to explain"),
        ("Produce out-of-fold predictions for stacking", "Keep one untouched test set", "Make the base models diverse"),
        ("Choice of base models", "Regularisation of the meta-learner", "Blend weights")),
}

#: The simplest sensible baseline for each kind of task.
BASELINES = {
    "classification": ("Logistic Regression", "Predict the majority class, then a regularised logistic regression."),
    "regression": ("Linear Regression", "Predict the mean, then a ridge regression."),
    "forecasting": ("ARIMA", "A seasonal-naive forecast (repeat the previous period), then ARIMA."),
    "clustering": ("K-Means", "K-Means with a few values of k."),
    "anomaly detection": ("Isolation Forest", "Robust z-scores, then Isolation Forest."),
}
