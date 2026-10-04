"""Live end-to-end check of ``POST /api/v1/datasets/recommend``.

Unlike the unit suite this talks to the real services (Kaggle, Hugging Face and
Gemini), so it needs the credentials in ``.env`` and network access, and it spends
a few Gemini calls. Run it from ``datapilot/backend``::

    python -m scripts.run_e2e

It sends five project descriptions through the real API app and asserts the trust
invariants on every response, rather than judging relevance by eye. A scenario
that cannot be evaluated because an upstream service failed is reported as an
*infrastructure* failure, not a code failure, so a Gemini 503 does not read as a
regression. Exit code is 1 only for a violated invariant.

Nothing credential-shaped is printed or written; the saved report holds metadata
only.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fastapi.testclient import TestClient

from app.main import app

REPORT_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "e2e_report.json"

SCENARIOS: dict[str, str] = {
    "student": (
        "I want to build a machine learning system that predicts student academic "
        "performance using attendance, previous examination marks, assignment scores, "
        "study hours, and participation in class."
    ),
    "iot_botnet": (
        "I want to develop a machine learning system that detects malicious network "
        "activity in IoT devices by analyzing network traffic and identifying abnormal "
        "behavior such as botnet attacks."
    ),
    "diabetes": (
        "I want to predict whether a patient has diabetes from BMI, blood glucose, age "
        "and insulin levels using a classification model."
    ),
    "retail_churn": (
        "I want to predict which retail customers will churn using purchase history, "
        "customer tenure, and support interactions."
    ),
    "energy_forecast": (
        "I want to forecast hourly household electricity consumption using historical "
        "energy usage and temperature data."
    ),
}

_SECRET = re.compile(r"KGAT_[A-Za-z0-9]|AIza[A-Za-z0-9_-]{10}|Bearer\s+\S{12,}")
_INFRA_CODES = {
    "DATASET_SOURCE_UNAVAILABLE",
    "DATASET_SOURCE_TIMEOUT",
    "DATASET_RATE_LIMITED",
    "DATASET_INVALID_RESPONSE",
}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class ScenarioResult:
    key: str
    status: str  # "ok" | "invariant_failed" | "infrastructure"
    checks: list[Check] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def _check(checks: list[Check], name: str, ok: bool, detail: str = "") -> None:
    checks.append(Check(name, bool(ok), "" if ok else detail))


def evaluate(body: dict[str, Any], raw_text: str) -> list[Check]:
    """Every trust invariant that must hold on one recommend response."""
    checks: list[Check] = []
    recs = body["recommendations"]
    final = body.get("final_recommendation") or {}

    _check(checks, "response carries a final recommendation", bool(final), "missing")
    _check(checks, "no credential-shaped text in the response", not _SECRET.search(raw_text), "secret-like string found")
    _check(checks, "ranks are 1..n in order", [r["rank"] for r in recs] == list(range(1, len(recs) + 1)), "rank gap")
    _check(
        checks,
        "recommendations are sorted by score",
        all(a["ranking_score"] >= b["ranking_score"] - 1e-9 for a, b in zip(recs, recs[1:])),
        "not best-first",
    )

    for rec in recs:
        ds, intel, conf = rec["dataset"], rec.get("intelligence"), rec.get("confidence")
        tag = f"#{rec['rank']} {ds['name'][:32]}"
        ai = ds["source_type"] == "ai_suggested"

        _check(checks, f"{tag}: has intelligence and confidence", bool(intel and conf), "missing")
        if not (intel and conf):
            continue

        total = sum(f.get("contribution", 0.0) for f in rec["ranking_factors"])
        _check(checks, f"{tag}: ranking_score equals the sum of contributions",
               abs(total - rec["ranking_score"]) < 0.05, f"{total:.2f} vs {rec['ranking_score']:.2f}")

        cov = intel["feature_coverage"]
        _check(checks, f"{tag}: feature buckets partition the requirements",
               len(cov["matched"]) + len(cov["unknown"]) + len(cov["missing"]) == cov["required_count"], "bucket mismatch")

        if ai:
            _check(checks, f"{tag}: AI suggestion stays unverified", ds["verification_status"] == "unverified", ds["verification_status"])
            _check(checks, f"{tag}: AI suggestion has no verified schema", intel["schema_state"] == "unknown", intel["schema_state"])
            _check(checks, f"{tag}: AI suggestion has no SCHEMA_VERIFIED claim",
                   all(c["layer"] != "schema_verified" for c in intel["evidence"]), "forged verified schema")
            _check(checks, f"{tag}: AI suggestion confidence is LOW", conf["level"] == "low", conf["level"])
            _check(checks, f"{tag}: AI suggestion reports no missing features", not cov["missing"], "absence claimed from a model")
            _check(checks, f"{tag}: AI suggestion has no columns", not ds.get("columns"), "columns on an AI record")
        else:
            _check(checks, f"{tag}: registry record is verified", ds["verification_status"] == "verified", ds["verification_status"])

        if intel["schema_state"] == "verified":
            _check(checks, f"{tag}: verified schema is backed by columns the registry published",
                   bool(ds.get("columns")) and not ai, "verified without published columns")
        if cov["missing"]:
            _check(checks, f"{tag}: missing features are only claimed on a schema",
                   intel["schema_state"] in {"verified", "partial"}, intel["schema_state"])
        if intel["schema_state"] == "unknown":
            _check(checks, f"{tag}: unknown schema never yields missing features", not cov["missing"], "absence on unknown schema")

        _check(checks, f"{tag}: UNKNOWN target is not reported as incompatible without evidence",
               not (intel["target_analysis"]["state"] == "incompatible" and intel["target_analysis"]["layer"] not in {"source_declared", "schema_verified"}),
               "incompatible on weak evidence")

    outcome = final.get("outcome")
    best = final.get("best_match")
    if outcome == "recommended":
        _check(checks, "RECOMMENDED best match is registry-confirmed", best and best["source_type"] == "verified_external", "unconfirmed recommended")
        _check(checks, "RECOMMENDED best match is not low confidence", best and best["confidence"]["level"] != "low", "low confidence")
        _check(checks, "RECOMMENDED best match is not incompatible", best and best["compatibility"] != "low", "incompatible")
    if outcome == "none":
        _check(checks, "NONE names no best match", best is None, "best match present")
    if best:
        _check(checks, "best match is one of the returned datasets", best["dataset_id"] in {r["dataset"]["id"] for r in recs}, "unknown id")
        _check(checks, "best match carries at least one reason", bool(best["why"]), "no reasons")

    meta = body.get("meta", {})
    enrichment = meta.get("enrichment", {})
    _check(checks, "enrichment stays within its bound", enrichment.get("attempted", 0) <= 5, str(enrichment))
    return checks


def run_scenario(client: TestClient, key: str, description: str) -> tuple[ScenarioResult, dict[str, Any] | None]:
    response = client.post("/api/v1/datasets/recommend", json={"description": description, "limit": 10})
    if response.status_code != 200:
        detail = response.json().get("detail", {})
        code = (detail.get("error") or {}).get("code") if isinstance(detail, dict) else None
        status = "infrastructure" if (response.status_code in {429, 502, 504} or code in _INFRA_CODES) else "invariant_failed"
        result = ScenarioResult(key, status, [Check("HTTP 200", False, f"{response.status_code} {code}")])
        return result, None

    body = response.json()
    result = ScenarioResult(key, "ok", evaluate(body, response.text))
    if result.failed:
        result.status = "invariant_failed"

    recs = body["recommendations"]
    final = body["final_recommendation"]
    result.summary = {
        "query": body["search_query"],
        "sources": {s["source"]: f'{s["status"]}:{s["candidate_count"]}' for s in body["sources"]},
        "meta": {k: body["meta"][k] for k in ("raw_candidates", "verified_count", "ai_suggested_count", "enrichment", "confirmed_ai_suggestions")},
        "outcome": final["outcome"],
        "best_match": (final["best_match"] or {}).get("name"),
        "close_call": final["close_call"],
        "top": [
            {
                "name": r["dataset"]["name"],
                "source": r["dataset"]["source"],
                "score": round(r["ranking_score"], 2),
                "compat": r["intelligence"]["compatibility"]["overall"],
                "confidence": r["confidence"]["level"],
                "schema": r["intelligence"]["schema_state"],
                "target": r["intelligence"]["target_analysis"]["state"],
                "features": r["intelligence"]["feature_coverage"]["required_count"],
                "depth": r["dataset"]["metadata_depth"],
            }
            for r in recs[:5]
        ],
    }
    return result, body


def main(argv: list[str]) -> int:
    only = set(argv[1:])
    client = TestClient(app)
    results: list[ScenarioResult] = []
    for key, description in SCENARIOS.items():
        if only and key not in only:
            continue
        print(f"\n=== {key} ===")
        result, _ = run_scenario(client, key, description)
        results.append(result)
        if result.summary:
            s = result.summary
            print(f"query     : {s['query']}")
            print(f"sources   : {s['sources']}")
            print(f"meta      : {s['meta']}")
            print(f"outcome   : {s['outcome']}  best={s['best_match']!r}  close_call={s['close_call']}")
            for t in s["top"]:
                print(f"  {t['score']:6.2f} {t['compat']:7} conf={t['confidence']:6} schema={t['schema']:8} target={t['target']:12} {t['source']:11} [{t['depth']}] {t['name'][:46]}")
        passed = len(result.checks) - len(result.failed)
        print(f"checks    : {passed}/{len(result.checks)} passed  -> {result.status}")
        for c in result.failed:
            print(f"   FAIL: {c.name}: {c.detail}")

    total = sum(len(r.checks) for r in results)
    failed = sum(len(r.failed) for r in results)
    infra = [r.key for r in results if r.status == "infrastructure"]
    broken = [r.key for r in results if r.status == "invariant_failed"]
    print(f"\nTOTAL: {total - failed}/{total} checks passed across {len(results)} scenarios")
    if infra:
        print(f"INFRASTRUCTURE (not a code failure): {', '.join(infra)}")
    if broken:
        print(f"INVARIANT FAILURES: {', '.join(broken)}")

    REPORT_PATH.write_text(
        json.dumps(
            [
                {
                    "scenario": r.key,
                    "status": r.status,
                    "checks_passed": len(r.checks) - len(r.failed),
                    "checks_total": len(r.checks),
                    "failures": [{"check": c.name, "detail": c.detail} for c in r.failed],
                    "summary": r.summary,
                }
                for r in results
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {REPORT_PATH}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
