"""Deterministic synthetic datasets for testing and benchmarking dataset analysis.

Every dataset is generated from a fixed seed, so a run always sees identical data and its
expected properties can be stated exactly. They are written as CSV bytes (the form a user
uploads) so the benchmark exercises ingestion as well as analysis.

The datasets are *shaped* like the real ones people bring - a student-performance table, a
churn table, house prices, a diabetes table, a daily sales series, unlabelled sensor
readings, a bank-marketing style table - but they are synthetic: they stand for the
structure, not for any real data, and nothing measured on them is a claim about real data.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np


@dataclass
class SyntheticDataset:
    name: str
    columns: dict[str, list[Optional[str]]]

    @property
    def n_rows(self) -> int:
        return len(next(iter(self.columns.values())))

    def csv_bytes(self, delimiter: str = ",") -> bytes:
        out = io.StringIO()
        writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
        writer.writerow(self.columns.keys())
        for row in zip(*self.columns.values()):
            writer.writerow(["" if v is None else v for v in row])
        return out.getvalue().encode("utf-8")


def _s(values) -> list[Optional[str]]:
    return [None if v is None else str(v) for v in values]


def _f(values, digits: int = 2) -> list[str]:
    return [f"{float(v):.{digits}f}" for v in values]


def _with_missing(values: list[str], rng, share: float) -> list[Optional[str]]:
    mask = rng.random(len(values)) < share
    return [None if m else v for v, m in zip(values, mask)]


# ---------------------------------------------------------------------------


def student_performance(n: int = 300, seed: int = 11, leaky: bool = False) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(50, 100, n)
    previous = rng.uniform(30, 100, n)
    assignments = np.clip(previous * 0.5 + rng.normal(35, 8, n), 0, 100)
    study_hours = rng.uniform(0, 10, n)
    participation = rng.choice(["low", "medium", "high"], n, p=[0.3, 0.5, 0.2])
    bonus = np.select([participation == "high", participation == "medium"], [5.0, 2.0], 0.0)
    final = np.clip(
        0.25 * attendance + 0.35 * previous + 0.15 * assignments + 2.0 * study_hours + bonus
        + rng.normal(0, 4, n) - 10, 0, 100,
    )
    columns: dict[str, list] = {
        "student_id": [f"S{1000 + i}" for i in range(n)],
        "attendance": _f(attendance),
        "previous_marks": _f(previous),
        "assignment_scores": _f(assignments),
        "study_hours": _f(study_hours),
        "participation": _s(participation),
        "gender": _s(rng.choice(["F", "M"], n)),
    }
    if leaky:
        columns["final_exam_score"] = _f(final * 0.99 + rng.normal(0, 0.05, n))
    columns["final_grade"] = _f(final)
    return SyntheticDataset("student_performance_leaky" if leaky else "student_performance", columns)


def customer_churn(n: int = 600, seed: int = 21) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, n)
    monthly = rng.uniform(20, 120, n)
    contract = rng.choice(["month-to-month", "one year", "two year"], n, p=[0.55, 0.25, 0.2])
    calls = rng.poisson(2.0, n)
    logit = -1.0 - 0.04 * tenure + 0.015 * monthly + 0.3 * calls + np.where(contract == "month-to-month", 1.2, -0.6)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    return SyntheticDataset("customer_churn", {
        "customer_id": [f"C{20000 + i}" for i in range(n)],
        "email": [f"user{i}@example.com" for i in range(n)],
        "tenure_months": _s(tenure),
        "monthly_charges": _f(monthly),
        "contract": _s(contract),
        "support_calls": _s(calls),
        "churn": ["Yes" if c else "No" for c in churn],
    })


def house_prices(n: int = 400, seed: int = 31) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(600, 4000, n)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    year = rng.integers(1950, 2023, n)
    hood = rng.choice(["north", "south", "east", "west", "centre"], n)
    premium = {"north": 20000, "south": -10000, "east": 0, "west": 5000, "centre": 40000}
    price = 80 * sqft + 9000 * bedrooms + 12000 * bathrooms + 300 * (year - 1950) + \
        np.array([premium[h] for h in hood]) + rng.normal(0, 15000, n)
    return SyntheticDataset("house_prices", {
        "house_id": _s(range(1, n + 1)),
        "sqft": _f(sqft, 0),
        "bedrooms": _s(bedrooms),
        "bathrooms": _s(bathrooms),
        "year_built": _s(year),
        "neighborhood": _s(hood),
        "price": _f(price, 0),
    })


def diabetes(n: int = 400, seed: int = 41) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 80, n)
    bmi = rng.normal(31, 6, n).clip(18, 55)
    glucose = rng.normal(120, 30, n).clip(60, 220)
    pressure = rng.normal(72, 12, n).clip(40, 120)
    insulin = rng.gamma(2.0, 60.0, n)
    logit = -9.0 + 0.04 * age + 0.07 * bmi + 0.035 * glucose
    outcome = rng.random(n) < 1 / (1 + np.exp(-logit))
    return SyntheticDataset("diabetes", {
        "patient_id": _s(range(1, n + 1)),
        "age": _s(age),
        "bmi": _f(bmi, 1),
        "glucose": _f(glucose, 0),
        "blood_pressure": _f(pressure, 0),
        "insulin": _with_missing(_f(insulin, 0), rng, 0.35),
        "outcome": _s(outcome.astype(int)),
    })


def sales_timeseries(n: int = 365, seed: int = 51) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    start = datetime(2023, 1, 1)
    days = np.arange(n)
    units = 200 + 0.3 * days + 25 * np.sin(2 * np.pi * days / 7) + rng.normal(0, 6, n)
    promo = (rng.random(n) < 0.15).astype(int)
    units = units + 40 * promo
    return SyntheticDataset("sales_timeseries", {
        "date": [(start + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in days],
        "store": ["A"] * n,
        "promo": _s(promo),
        "units_sold": _f(units, 1),
    })


def sensor_readings(n: int = 800, seed: int = 61) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(20, 2, n), rng.normal(1000, 15, n), rng.normal(0.5, 0.1, n)
    spikes = rng.choice(n, size=8, replace=False)
    a[spikes] += 15
    return SyntheticDataset("sensor_readings", {
        "sensor_a": _f(a), "sensor_b": _f(b), "sensor_c": _f(c, 3),
    })


def bank_marketing(n: int = 500, seed: int = 71) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    job = rng.choice(["admin", "technician", "services", "management", "retired", "student",
                      "blue-collar", "unemployed", "entrepreneur", "housemaid", "self-employed",
                      "unknown"], n)
    education = rng.choice(["primary", "secondary", "tertiary", "unknown"], n)
    marital = rng.choice(["married", "single", "divorced"], n)
    age = rng.integers(18, 85, n)
    balance = rng.normal(1200, 2500, n)
    housing = rng.choice(["yes", "no"], n)
    loan = rng.choice(["yes", "no"], n, p=[0.2, 0.8])
    logit = -2.0 + 0.0004 * balance + np.where(job == "student", 1.0, 0) + np.where(housing == "no", 0.6, 0)
    subscribed = rng.random(n) < 1 / (1 + np.exp(-logit))
    return SyntheticDataset("bank_marketing", {
        "age": _s(age), "job": _s(job), "marital": _s(marital), "education": _s(education),
        "balance": _f(balance, 0), "housing": _s(housing), "loan": _s(loan),
        "y": ["yes" if s else "no" for s in subscribed],
    })


def fraud_transactions(n: int = 2000, seed: int = 81) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.5, 1.0, n)
    hour = rng.integers(0, 24, n)
    distance = rng.exponential(10, n)
    fraud = rng.random(n) < 0.02
    amount[fraud] *= 4
    return SyntheticDataset("fraud_transactions", {
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "amount": _f(amount),
        "hour": _s(hour),
        "distance_km": _f(distance),
        "is_fraud": _s(fraud.astype(int)),
    })


def review_text(n: int = 120, seed: int = 91) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    words = ["great", "poor", "service", "product", "arrived", "late", "quality", "friendly",
             "broken", "excellent", "would", "recommend", "never", "again", "price", "value"]
    reviews = [" ".join(rng.choice(words, size=int(rng.integers(12, 25)))) + f" ref{i}" for i in range(n)]
    return SyntheticDataset("review_text", {
        "review_id": _s(range(1, n + 1)),
        "review_text": reviews,
        "rating": _s(rng.integers(1, 6, n)),
    })


def tiny_table() -> SyntheticDataset:
    return SyntheticDataset("tiny_table", {
        "a": _s([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        "b": _s([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]),
        "label": ["x", "y"] * 6,
    })


def messy_table(n: int = 200, seed: int = 101) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    base = rng.normal(50, 10, n)
    rows = {
        "id": _s(range(n)),
        "value": _f(base),
        "value_copy": _f(base),
        "constant": ["same"] * n,
        "mostly_missing": _with_missing(_f(rng.normal(0, 1, n)), rng, 0.8),
        "mixed": [str(i) if i % 25 else "oops" for i in range(n)],
        "category": _s(rng.choice(["a", "b", "c"], n)),
    }
    for key in rows:  # duplicate the first 10 rows
        rows[key] = rows[key] + rows[key][:10]
    return SyntheticDataset("messy_table", rows)


GENERATORS: dict[str, Callable[[], SyntheticDataset]] = {
    "student_performance": student_performance,
    "student_performance_leaky": lambda: student_performance(leaky=True),
    "customer_churn": customer_churn,
    "house_prices": house_prices,
    "diabetes": diabetes,
    "sales_timeseries": sales_timeseries,
    "sensor_readings": sensor_readings,
    "bank_marketing": bank_marketing,
    "fraud_transactions": fraud_transactions,
    "review_text": review_text,
    "tiny_table": tiny_table,
    "messy_table": messy_table,
}
