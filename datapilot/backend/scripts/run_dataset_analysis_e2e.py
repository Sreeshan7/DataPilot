"""Live end-to-end check of User B (dataset analysis) and the unified ``/assist`` workflow.

Unlike the unit suite this talks to the real registries (UCI, OpenML, Hugging Face, Kaggle), so it
needs network access (a Kaggle token in ``.env`` is optional; without it Kaggle is simply skipped
by the registry layer). It needs no Gemini key and makes no Gemini call. Run from ``datapilot/backend``::

    python -m scripts.run_dataset_analysis_e2e

It drives the real FastAPI app and asserts the trust invariants on every response instead of judging
by eye: no statistics for a dataset that was only researched, no parsed-evidence claims, a STRONG
target only when the registry designates it, "unknown" when a schema is not published. A scenario
that cannot be evaluated because a registry failed is reported as an *infrastructure* failure, not a
code failure. Exit code is 1 only for a violated invariant.

Nothing credential-shaped is printed or written; the saved report holds metadata only, never a
dataset's contents.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.main import app
from app.services.dataset_analysis.evaluation.datasets import GENERATORS

REPORT_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "dataset_analysis_e2e_report.json"
ANALYZE = "/api/v1/datasets/analyze"
ASSIST = "/api/v1/assist"

_SECRET = re.compile(r"KGAT_[A-Za-z0-9]|AIza[A-Za-z0-9_-]{10}|Bearer\s+\S{12,}")
_INFRA_STATUS = {429, 502, 504}


@dataclass
class Scenario:
    name: str
    form: dict[str, str]
    endpoint: str = ANALYZE
    expect: str = "external"  # external | unknown | ambiguous | unresolved | file
    file: str | None = None
    notes: str = ""


SCENARIOS = [
    Scenario("uci_name_iris", {"dataset_name": "Iris"}),
    Scenario("openml_name_titanic", {"dataset_name": "Titanic"}),
    Scenario("huggingface_name_mnist", {"dataset_name": "MNIST"}),
    Scenario("uci_link_heart_disease", {"dataset_url": "https://archive.ics.uci.edu/dataset/45/heart+disease"}),
    Scenario("openml_link_iris", {"dataset_url": "https://www.openml.org/d/61"}),
    Scenario("huggingface_link_mnist", {"dataset_url": "https://huggingface.co/datasets/mnist"}),
    Scenario("kaggle_link_titanic", {"dataset_url": "https://www.kaggle.com/datasets/yasserh/titanic-dataset"},
             expect="any_external", notes="Kaggle may publish no schema: UNKNOWN is the correct answer then."),
    Scenario("message_names_dataset", {"message": "Analyze the Kaggle Titanic dataset"}, endpoint=ASSIST),
    Scenario("ambiguous_name", {"dataset_name": "student"}, expect="ambiguous"),
    Scenario("fake_name", {"dataset_name": "Zzyzx Totally Fake Dataset 9921"}, expect="unresolved"),
    Scenario("uploaded_file_churn", {}, expect="file", file="customer_churn"),
    Scenario("assist_upload", {"message": "here is my data"}, endpoint=ASSIST, expect="file", file="customer_churn"),
]


@dataclass
class Outcome:
    scenario: str
    status: str  # passed | failed | infrastructure
    http: int = 0
    summary: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)


def _analysis(body: dict[str, Any]) -> dict[str, Any] | None:
    if "analysis" in body and isinstance(body["analysis"], dict) and "analysis" in body["analysis"]:
        return body["analysis"]["analysis"]  # /assist wraps the /analyze response
    if "mode" in body.get("analysis", {}):
        return body["analysis"]
    return None


def check(scenario: Scenario, status: int, body: dict[str, Any], text: str) -> Outcome:
    out = Outcome(scenario.name, "passed", http=status)
    if _SECRET.search(text):
        out.violations.append("a credential-shaped string appears in the response")
    if status in _INFRA_STATUS:
        out.status = "infrastructure"
        return out
    if status != 200:
        out.violations.append(f"unexpected HTTP {status}")
        out.status = "failed"
        return out

    analysis = _analysis(body)
    if analysis is None:
        out.violations.append("no analysis in the response")
        out.status = "failed"
        return out

    schema, resolution = analysis["schema"], analysis["input_resolution"]
    targets, tasks = analysis["target_candidates"], analysis["task_candidates"]
    out.summary = {
        "mode": analysis["mode"], "resolution": resolution["resolution_status"],
        "schema_level": schema["level"], "columns": schema["column_count"], "rows": schema["row_count"],
        "best_target": targets[0]["column"] if targets else None,
        "tasks": [t["task"] for t in tasks[:3]],
        "alternatives": len(resolution["alternatives"]),
    }

    if scenario.expect == "file":
        if analysis["mode"] != "file_parsed":
            out.violations.append("an uploaded file was not parsed")
        if not analysis["column_profiles"] or analysis["quality"] is None:
            out.violations.append("a parsed file has no statistics or quality report")
    else:
        if analysis["mode"] != "external_metadata":
            out.violations.append("a name or link was reported as a parsed file")
        if analysis["column_profiles"] or analysis["quality"] is not None or analysis["ingestion"] is not None:
            out.violations.append("statistics, quality findings or ingestion were claimed without reading rows")
        if "datapilot_parsed" in text:
            out.violations.append("parsed-file evidence appears for a dataset that was not parsed")
        for t in targets:
            if t["state"] == "strong_candidate" and not any(
                "designates" in e["statement"] and e["level"] == "registry_verified" for e in t["evidence"]
            ):
                out.violations.append(f"target '{t['column']}' is STRONG without the registry designating it")
        for t in tasks:
            if not any("no rows were read" in limit for limit in t["limitations"]):
                out.violations.append(f"task {t['task']} does not say that no rows were read")
        if schema["level"] == "unknown" and (targets or tasks):
            out.violations.append("targets or tasks were produced without any known schema")
        if schema["level"] == "datapilot_parsed":
            out.violations.append("a schema read from a registry is labelled as parsed")

    if scenario.expect == "ambiguous" and resolution["resolution_status"] not in ("ambiguous", "not_applicable"):
        out.violations.append("an ambiguous name was resolved to a dataset")
    if scenario.expect == "ambiguous" and analysis["external"] is not None:
        out.violations.append("an ambiguous name produced a dataset identity")
    if scenario.expect == "unresolved" and (analysis["external"] is not None or schema["level"] != "unknown"):
        out.violations.append("a made-up dataset was given an identity or a schema")
    if scenario.expect == "external" and analysis["external"] is None:
        out.status = "infrastructure"  # nothing found: a registry was likely unreachable
        out.summary["note"] = "no registry returned a record"
        return out

    out.status = "failed" if out.violations else "passed"
    return out


def run_scenario(client: TestClient, scenario: Scenario) -> Outcome:
    files = None
    if scenario.file:
        files = {"file": (f"{scenario.file}.csv", GENERATORS[scenario.file]().csv_bytes(), "text/csv")}
    try:
        response = client.post(scenario.endpoint, data=scenario.form, files=files)
    except Exception as exc:  # network or app failure: not a trust violation
        return Outcome(scenario.name, "infrastructure", summary={"error": type(exc).__name__})
    try:
        body = response.json()
    except ValueError:
        body = {}
    return check(scenario, response.status_code, body, response.text)


def main() -> int:
    client = TestClient(app, raise_server_exceptions=False)
    outcomes = [run_scenario(client, s) for s in SCENARIOS]
    REPORT_PATH.write_text(
        json.dumps([o.__dict__ for o in outcomes], indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    for o in outcomes:
        print(f"[{o.status.upper():14s}] {o.scenario:28s} http={o.http} {json.dumps(o.summary, default=str)[:170]}")
        for violation in o.violations:
            print(f"      ! {violation}")
    failed = [o for o in outcomes if o.status == "failed"]
    infra = [o for o in outcomes if o.status == "infrastructure"]
    print(f"\n{len(outcomes) - len(failed) - len(infra)} passed, {len(infra)} infrastructure, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
