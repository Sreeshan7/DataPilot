"""Methods and reported metrics, read from retrieved text only.

A method is "mentioned" when its name appears in a paper's retrieved title or snippet; that is all this
claims. A metric is kept only when the text states a figure next to a metric name, and it is reported as
what the *source* said in *its own* setting, never as a prediction for the user's data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.research import EvidenceKind, MethodSupport, Paper, ReportedMetric


@dataclass(frozen=True)
class MethodInfo:
    name: str
    pattern: re.Pattern[str]
    tasks: frozenset[str]            # tasks it is a standard choice for
    tabular: bool = True
    note: str = ""


def _m(name: str, regex: str, tasks: str, tabular: bool = True, note: str = "") -> MethodInfo:
    return MethodInfo(name, re.compile(regex, re.I), frozenset(tasks.split()), tabular, note)


CATALOG: tuple[MethodInfo, ...] = (
    _m("XGBoost", r"\bxg\s?boost\b|\bextreme gradient boosting\b", "classification regression forecasting"),
    _m("LightGBM", r"\blight\s?gbm\b|\blight gradient boosting\b", "classification regression forecasting"),
    _m("CatBoost", r"\bcat\s?boost\b", "classification regression"),
    _m("Gradient Boosting", r"\bgradient[- ]boost(?:ed|ing)?\b(?!\s*\(?xgb)", "classification regression forecasting"),
    _m("Random Forest", r"\brandom forests?\b", "classification regression"),
    _m("Decision Tree", r"\bdecision trees?\b|\bCART\b", "classification regression"),
    _m("Logistic Regression", r"\blogistic regression\b", "classification"),
    _m("Linear Regression", r"\blinear regression\b|\bridge regression\b|\blasso\b", "regression"),
    _m("Support Vector Machine", r"\bsupport vector (?:machines?|regress\w+)\b|\bSVM\b|\bSVR\b", "classification regression"),
    _m("k-Nearest Neighbors", r"\bk[- ]nearest neighbou?rs?\b|\bk-?NN\b", "classification regression"),
    _m("Naive Bayes", r"\bna[iï]ve bayes\b", "classification"),
    _m("Neural Network", r"\bneural networks?\b|\bmulti-?layer perceptron\b|\bMLP\b|\bANN\b", "classification regression forecasting"),
    _m("Deep Learning", r"\bdeep learning\b|\bdeep neural\b", "classification regression forecasting", False),
    _m("LSTM", r"\bLSTM\b|\blong short[- ]term memory\b", "forecasting classification", False),
    _m("Transformer", r"\btransformers?\b|\bBERT\b", "classification forecasting", False),
    _m("Convolutional Neural Network", r"\bCNN\b|\bconvolutional neural\b", "classification", False),
    _m("ARIMA", r"\bARIMA\b|\bSARIMA\b", "forecasting", False),
    _m("Prophet", r"\bProphet\b", "forecasting", False),
    _m("K-Means", r"\bk-?means\b", "clustering"),
    _m("Isolation Forest", r"\bisolation forests?\b", "anomaly detection"),
    _m("AdaBoost", r"\bada\s?boost\b", "classification regression"),
    _m("Ensemble Methods", r"\bensemble (?:learning|methods?|model)s?\b|\bstacking\b", "classification regression"),
)

_METRIC_NAMES = {
    "accuracy": "accuracy", "f1": "F1", "f1-score": "F1", "f1 score": "F1", "auc": "AUC", "roc-auc": "AUC",
    "rmse": "RMSE", "mae": "MAE", "precision": "precision", "recall": "recall", "r2": "R2", "r-squared": "R2",
}
_METRIC = re.compile(
    r"\b(?P<name>accuracy|f1[- ]?score|f1|roc-?auc|auc|rmse|mae|precision|recall|r2|r-squared)\b"
    r"(?:\s+(?:of|was|is|reached|achieved|=|:|at|up to))?\s*[:=]?\s*(?P<value>\d{1,3}(?:\.\d+)?\s?%|0\.\d+)",
    re.I)
_CONTEXT = re.compile(r"\bon (?:the |a |an )?(?P<c>[A-Z][\w\- ]{2,50}? (?:dataset|data set|data))\b")


def methods_in(text: str) -> list[str]:
    found = []
    for info in CATALOG:
        if info.pattern.search(text) and info.name not in found:
            found.append(info.name)
    # "Gradient Boosting" is the generic family: drop it when a specific booster is named.
    if "Gradient Boosting" in found and any(n in found for n in ("XGBoost", "LightGBM", "CatBoost")):
        found.remove("Gradient Boosting")
    return found


def annotate_methods(papers: list[Paper]) -> None:
    for paper in papers:
        paper.methods = methods_in(f"{paper.title}. {paper.abstract_snippet or ''}")


def aggregate_methods(papers: list[Paper]) -> list[MethodSupport]:
    by_method: dict[str, list[str]] = {}
    for paper in papers:
        for method in paper.methods:
            by_method.setdefault(method, []).append(paper.id)
    return [
        MethodSupport(method=m, verified_paper_count=len(ids), paper_ids=ids)
        for m, ids in sorted(by_method.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]


def extract_metrics(papers: list[Paper]) -> list[ReportedMetric]:
    out: list[ReportedMetric] = []
    for paper in papers:
        text = paper.abstract_snippet or ""
        if not text:
            continue
        for match in _METRIC.finditer(text):
            value = match.group("value").replace(" ", "")
            if not re.search(r"[%.]", value) and float(value) > 100:
                continue
            window = text[max(0, match.start() - 120): match.end() + 120]
            methods = methods_in(window)
            context = _CONTEXT.search(text)
            out.append(ReportedMetric(
                metric=_METRIC_NAMES.get(match.group("name").lower(), match.group("name")), value=value,
                context=context.group("c") if context else None, method=methods[0] if methods else None,
                paper_id=paper.id, paper_title=paper.title, url=paper.url,
                statement=f"Reported by the source in its own experimental setting; not a prediction for your dataset."))
    seen: set[tuple] = set()
    unique = []
    for metric in out:
        key = (metric.paper_id, metric.metric, metric.value)
        if key not in seen:
            seen.add(key)
            unique.append(metric)
    return unique
