"""A controllable fake of the four dataset registries, for offline tests.

Routes by host and path, records every request, and lets a test set exactly what each
registry says. Response shapes mirror what the live registries returned when probed
(UCI answers HTTP 200 with the status in the body, OpenML uses 412 for "no results",
Hugging Face redirects with a relative Location and answers 401 for a missing dataset).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import httpx

from app.core.config import Settings
from app.services.dataset_analysis.external.fetcher import SafeFetcher


def uci_record(uci_id=53, name="Iris", rows=150, variables=None, tasks=("Classification",), **extra):
    variables = variables if variables is not None else [
        {"name": "sepal length", "role": "Feature", "type": "Continuous", "description": None},
        {"name": "sepal width", "role": "Feature", "type": "Continuous", "description": None},
        {"name": "petal length", "role": "Feature", "type": "Continuous", "description": None},
        {"name": "petal width", "role": "Feature", "type": "Continuous", "description": None},
        {"name": "class", "role": "Target", "type": "Categorical", "description": None},
    ]
    return {"status": 200, "data": {
        "uci_id": uci_id, "name": name, "repository_url": f"https://archive.ics.uci.edu/dataset/{uci_id}/{name.lower()}",
        "data_url": f"https://archive.ics.uci.edu/static/public/{uci_id}/data.csv",
        "abstract": f"The {name} dataset.", "area": "Biology", "tasks": list(tasks),
        "characteristics": ["Tabular"], "num_instances": rows, "num_features": len(variables) - 1,
        "has_missing_values": "no", "last_updated": "Fri Nov 03 2023", "variables": variables,
        "target_col": [v["name"] for v in variables if v["role"] == "Target"], **extra}}


def openml_list(*versions):
    return {"data": {"dataset": [
        {"did": did, "name": name, "version": version, "status": "active", "format": "ARFF",
         "quality": [{"name": "NumberOfInstances", "value": str(rows)}, {"name": "NumberOfFeatures", "value": str(cols)}]}
        for did, name, version, rows, cols in versions]}}


def openml_desc(did=61, name="iris", version=1, licence="Public", target="class"):
    return {"data_set_description": {
        "id": str(did), "name": name, "version": str(version), "description": f"{name} data",
        "default_target_attribute": target, "licence": licence, "format": "ARFF",
        "upload_date": "2017-04-06T12:38:28", "tag": ["Biology", "study_1"], "creator": "R. Fisher"}}


def openml_features(*cols):
    feats = [{"index": str(i), "name": n, "data_type": t, "is_target": str(tg).lower(),
              "is_ignore": "false", "is_row_identifier": "false", "number_of_missing_values": "0"}
             for i, (n, t, tg) in enumerate(cols)]
    return {"data_features": {"feature": feats if len(feats) != 1 else feats[0]}}


def openml_qualities(**q):
    return {"data_qualities": {"quality": [{"name": k, "value": str(v)} for k, v in q.items()]}}


def hf_record(dataset_id="owner/titanic", features=None, rows=891, description="A passenger table.", **extra):
    features = features if features is not None else [
        {"name": "survived", "dtype": "int64"}, {"name": "age", "dtype": "float64"},
        {"name": "sex", "dtype": "string"}, {"name": "fare", "dtype": "float64"},
        {"name": "label", "dtype": {"class_label": {"names": {"0": "no", "1": "yes"}}}},
    ]
    record = {"id": dataset_id, "author": dataset_id.split("/")[0], "downloads": 100, "likes": 3,
              "private": False, "disabled": False, "description": description,
              "lastModified": "2024-04-23T10:00:00.000Z", "createdAt": "2024-04-20T10:00:00.000Z",
              "tags": ["task_categories:tabular-classification", "license:cc-by-4.0", "format:csv", "survival"],
              "siblings": [{"rfilename": "train.csv"}, {"rfilename": "README.md"}],
              "cardData": {"pretty_name": dataset_id.split("/")[-1].title(), "license": "cc-by-4.0"}}
    if features is not False:
        record["cardData"]["dataset_info"] = {"features": features, "splits": [{"name": "train", "num_examples": rows}],
                                              "dataset_size": 12345}
    record.update(extra)
    return record


def kaggle_item(ref="owner/titanic", title="Titanic", downloads=5000):
    return {"refNullable": ref, "titleNullable": title, "downloadCountNullable": downloads}


def kaggle_view(title="Titanic", description="Columns: PassengerId, Survived, Pclass, Name, Sex, Age, Fare.",
                subtitle="Who survived?", license_name="CC0"):
    return {"titleNullable": title, "descriptionNullable": description, "subtitleNullable": subtitle,
            "licenseNameNullable": license_name, "totalBytesNullable": 61194, "ownerNameNullable": "Owner",
            "lastUpdatedNullable": "2024-01-02T00:00:00Z", "tags": [{"nameNullable": "classification"}]}


class MockRegistries:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, Callable[[httpx.Request], Optional[httpx.Response]]]] = []
        self.default_status = 500

    # -- configuration helpers -------------------------------------------------
    def route(self, pattern: str, responder) -> "MockRegistries":
        """``pattern`` is matched as a substring of 'host/path?query'."""
        if not callable(responder):
            body, status = (responder if isinstance(responder, tuple) else (responder, 200))
            responder = lambda request, b=body, s=status: self._json(b, s)  # noqa: E731
        self.routes.append((pattern, responder))
        return self

    @staticmethod
    def _json(body: Any, status: int = 200, headers: Optional[dict] = None, content_type="application/json"):
        return httpx.Response(
            status, content=json.dumps(body).encode(), headers={"content-type": content_type, **(headers or {})}
        )

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        target = f"{request.url.host}{request.url.path}" + (f"?{request.url.query.decode()}" if request.url.query else "")
        for pattern, responder in reversed(self.routes):
            if pattern in target:
                response = responder(request)
                if response is not None:
                    return response
        return httpx.Response(self.default_status)

    # -- plumbing --------------------------------------------------------------
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def fetcher(self, settings: Settings) -> SafeFetcher:
        return SafeFetcher(settings, self.client())

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    def urls(self) -> list[str]:
        return [str(r.url) for r in self.requests]
