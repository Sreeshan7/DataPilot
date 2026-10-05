"""Relevance benchmark: labelled candidate sets across different problems, scored for precision and recall.

The candidates are synthetic fixtures written for this test (they stand in for what search returns), each labelled
relevant or not by hand. They pin the behaviour of the question parser + topic gate + ranking, so a change that
lets junk through, or drops real matches, fails here. ``run_benchmark`` is importable for ad-hoc reports.
"""

from __future__ import annotations

import asyncio

from app.schemas.research import ResearchRequest
from app.services.research import ResearchService
from tests.unit.test_research import FakeProvider, hit, settings

# (question, task, [(title, snippet, relevant)])
CASES = [
    ("I want to research the best methods for predicting the trafic during specific time and weather conditions", "regression", [
        ("Traffic volume prediction with weather and time features", "Random forest for hourly traffic volume.", True),
        ("Short-term traffic flow forecasting using gradient boosting", "XGBoost for traffic flow under rain.", True),
        ("ARIMA forecasting of interest rates", "Monetary policy and inflation.", False),
        ("Customer lifetime value prediction", "Retail customers and revenue.", False)]),
    ("predict diabetis risk from patient data", "classification", [
        ("Diabetes risk prediction with machine learning", "Random forest on patient records for diabetes risk.", True),
        ("Early detection of diabetes using ensemble classifiers", "Patient data and diabetes screening.", True),
        ("Heart failure survival analysis", "Cardiology cohort.", False),
        ("Image segmentation with U-Net", "Medical images.", False)]),
    ("customer churn prediction for a telecom company", "classification", [
        ("Customer churn prediction in telecom using XGBoost", "Telecom churn classification.", True),
        ("A survey of churn modelling in telecommunications", "Review of churn methods for telecom.", True),
        ("Stock price prediction with LSTM", "Equity markets.", False),
        ("Telecom network fault detection", "Base station alarms.", False)]),
    ("student academic performance prediction using attendance", "classification", [
        ("Predicting student academic performance with random forest", "Attendance and marks.", True),
        ("Student dropout prediction", "Academic performance of students and attendance.", True),
        ("Attendance tracking with face recognition", "A camera system for classrooms.", False),
        ("Crop yield prediction", "Agriculture.", False)]),
    ("house price prediction with regression", "regression", [
        ("House price prediction using gradient boosting regression", "Housing market features.", True),
        ("Hedonic regression of housing prices", "House price drivers.", True),
        ("Predicting electricity demand", "Power grid load.", False),
        ("Sentiment analysis of tweets", "Social media text.", False)]),
    ("credit card fraud detection", "classification", [
        ("Credit card fraud detection with imbalanced learning", "Fraud classification on card transactions.", True),
        ("Deep learning for credit card fraud", "Neural networks detect fraudulent card payments.", True),
        ("Credit scoring with logistic regression", "Loan default.", False),
        ("Network intrusion detection", "Cyber security.", False)]),
    ("electricity demand forecasting", "forecasting", [
        ("Short-term electricity demand forecasting with LSTM", "Hourly electricity demand.", True),
        ("Forecasting electricity demand using temperature", "Demand and weather.", True),
        ("Solar panel defect detection", "Images of panels.", False),
        ("Bitcoin price forecasting", "Cryptocurrency.", False)]),
    ("employee attrition prediction", "classification", [
        ("Employee attrition prediction using machine learning", "HR data and attrition.", True),
        ("Predicting employee turnover with ensemble methods", "Employee attrition risk.", True),
        ("Customer retention strategy", "Marketing.", False),
        ("Resume parsing with NLP", "Hiring text.", False)]),
    ("air quality prediction", "regression", [
        ("Air quality prediction with deep learning", "PM2.5 air quality forecasting.", True),
        ("Machine learning for urban air quality estimation", "Air quality sensors.", True),
        ("Water quality classification", "River samples.", False),
        ("Weather radar nowcasting", "Rain.", False)]),
    ("crop yield prediction", "regression", [
        ("Crop yield prediction using random forest", "Satellite and climate data for crop yield.", True),
        ("Deep learning for crop yield estimation", "Crop yield from remote sensing.", True),
        ("Plant disease detection from leaf images", "Leaf images.", False),
        ("Food price forecasting", "Markets.", False)]),
]


def run_benchmark() -> dict:
    """Aggregate precision/recall of the shown papers over all cases (per-case detail under ``cases``)."""
    cases, tp, shown, relevant = [], 0, 0, 0
    for index, (question, task, papers) in enumerate(CASES):
        hits = [hit(title, f"https://arxiv.org/abs/2{index:03d}.{i:05d}", snippet) for i, (title, snippet, _) in enumerate(papers)]
        labels = {h.title: ok for h, (_, _, ok) in zip(hits, papers)}
        request = ResearchRequest.model_validate({"project": {"description": question, "task": task}, "options": {"max_papers": 15}})
        body = asyncio.run(ResearchService(settings(), FakeProvider(hits)).discover(request))
        titles = [p.title for p in body.papers]
        good = sum(1 for t in titles if labels.get(t))
        total_relevant = sum(labels.values())
        tp, shown, relevant = tp + good, shown + len(titles), relevant + total_relevant
        cases.append({"question": question, "shown": len(titles), "relevant_shown": good, "relevant_total": total_relevant})
    return {"precision": tp / shown if shown else 0.0, "recall": tp / relevant if relevant else 0.0, "cases": cases}


def test_the_pipeline_shows_only_relevant_papers_and_keeps_the_real_matches():
    result = run_benchmark()
    for case in result["cases"]:
        assert case["relevant_shown"] == case["shown"], case  # nothing off-topic is shown
        assert case["relevant_shown"] >= case["relevant_total"] - 0, case  # and no real match is lost
    assert result["precision"] == 1.0 and result["recall"] == 1.0
