"""Registry adapters: Kaggle, Hugging Face, UCI and OpenML.

Each adapter does two things - find datasets by name, and fetch one dataset's metadata -
and maps what its registry published into :class:`ExternalMetadata`. All network access goes
through :class:`SafeFetcher`.

Evidence rules, applied identically by every adapter:

* A schema the registry publishes *as structured data* (UCI ``variables``, OpenML
  ``features``, Hugging Face ``dataset_info.features``) is ``REGISTRY_VERIFIED``.
* Column names read out of a prose description (Kaggle has no structured schema) are
  ``INFERRED``, with the extraction method named.
* No schema at all is ``UNKNOWN``. A schema is never filled in from a name or a guess.
* Numbers (rows, files, sizes) are only reported when the registry stated them.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote

from app.core.config import Settings
from app.schemas.dataset_analysis import EvidenceLevel, ExternalMetadata, SchemaColumn
from app.services.dataset_analysis.external.fetcher import FetchError, SafeFetcher
from app.services.dataset_analysis.external.prose import extract_columns
from app.services.dataset_analysis.ingestion import normalize_column_names

MAX_FILES_LISTED = 20
MAX_TAGS = 20
SEARCH_LIMIT = 20


@dataclass
class ExternalHit:
    """One dataset a registry returned for a name search."""

    source: str
    identifier: str
    name: str
    url: Optional[str] = None
    popularity: float = 0.0
    column_count: Optional[int] = None
    row_count: Optional[int] = None


def _schema_columns(
    pairs: list[tuple[str, Optional[str], Optional[str], Optional[str]]], evidence: EvidenceLevel
) -> list[SchemaColumn]:
    """(name, data_type, role, description) tuples -> schema columns with normalised names."""
    names = [p[0] for p in pairs]
    normalized, _ = normalize_column_names(names)
    return [
        SchemaColumn(
            original_name=name, normalized_name=norm, data_type=dtype, role=role,
            description=(description or None) and description[:300], evidence=evidence,
        )
        for (name, dtype, role, description), norm in zip(pairs, normalized)
    ]


def _int(value: Any) -> Optional[int]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == number and number >= 0 else None


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _when(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    for parse in (datetime.fromisoformat, lambda t: datetime.strptime(t, "%a %b %d %Y")):
        try:
            stamp = parse(text)
        except ValueError:
            continue
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return None


class ExternalAdapter:
    name = "abstract"
    #: Registries whose structured schema outranks prose, listed first when a name matches
    #: several. Authority is about where a dataset *originates*, not about popularity.
    authority = 99

    def __init__(self, settings: Settings, fetcher: SafeFetcher) -> None:
        self._settings = settings
        self._fetcher = fetcher

    @property
    def enabled(self) -> bool:
        return bool(self._settings.external_enabled)

    async def search(self, name: str) -> list[ExternalHit]:  # pragma: no cover - interface
        raise NotImplementedError

    async def fetch(self, identifier: str) -> ExternalMetadata:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------------------------------------------------------------
# UCI
# ---------------------------------------------------------------------------


class UCIAdapter(ExternalAdapter):
    name = "uci"
    authority = 0

    @property
    def _base(self) -> str:
        return self._settings.uci_api_base_url.rstrip("/")

    async def search(self, name: str) -> list[ExternalHit]:
        body = await self._fetcher.get_json(
            f"{self._base}/datasets/list", {"filter": "python", "search": name}
        )
        if not isinstance(body, dict) or body.get("status") != 200:
            return []
        hits = []
        for item in _as_list(body.get("data")):
            if not isinstance(item, dict) or item.get("id") is None or not item.get("name"):
                continue
            hits.append(ExternalHit(
                source="uci", identifier=str(item["id"]), name=str(item["name"]),
                url=f"https://archive.ics.uci.edu/dataset/{item['id']}"))
        return hits[:SEARCH_LIMIT]

    async def fetch(self, identifier: str) -> ExternalMetadata:
        params = {"id": int(identifier)} if identifier.isdigit() else {"name": identifier}
        body = await self._fetcher.get_json(f"{self._base}/dataset", params)
        # UCI answers HTTP 200 with the real status inside the body.
        if not isinstance(body, dict) or body.get("status") != 200 or not isinstance(body.get("data"), dict):
            raise FetchError(FetchError.NOT_FOUND, "The registry has no such dataset.")
        data = body["data"]

        variables = [v for v in _as_list(data.get("variables")) if isinstance(v, dict) and v.get("name")]
        pairs = [
            (str(v["name"]), v.get("type"), str(v.get("role") or "").lower() or None, v.get("description"))
            for v in variables
        ]
        targets = [str(v["name"]) for v in variables if str(v.get("role") or "").lower() == "target"]
        if not targets:
            targets = [str(t) for t in _as_list(data.get("target_col")) if t]
        columns = _schema_columns(pairs, EvidenceLevel.REGISTRY_VERIFIED)
        data_url = data.get("data_url")

        return ExternalMetadata(
            source="uci",
            identifier=str(data.get("uci_id", identifier)),
            name=str(data.get("name") or identifier),
            url=data.get("repository_url"),
            description=(data.get("abstract") or "").strip() or None,
            publisher="UCI Machine Learning Repository",
            row_count=_int(data.get("num_instances")),
            column_count=len(columns) or None,
            files=[data_url.rsplit("/", 1)[-1]] if isinstance(data_url, str) and data_url else [],
            file_types=[".csv"] if isinstance(data_url, str) and data_url.endswith(".csv") else [],
            tasks=[str(t) for t in _as_list(data.get("tasks"))],
            tags=[str(t) for t in (_as_list(data.get("characteristics")) + _as_list(data.get("area")))][:MAX_TAGS],
            updated_at=_when(data.get("last_updated")),
            registry_target_columns=targets,
            retrieved_at=datetime.now(timezone.utc),
            schema_columns=columns,
            schema_level=EvidenceLevel.REGISTRY_VERIFIED if columns else EvidenceLevel.UNKNOWN,
            schema_source="UCI variables" if columns else "none",
            schema_note=None if columns else "UCI published no variable list for this dataset.",
            registry_statistics={
                "has_missing_values": 1.0 if str(data.get("has_missing_values", "")).lower() == "yes" else 0.0
            } if data.get("has_missing_values") is not None else {},
        )


# ---------------------------------------------------------------------------
# OpenML
# ---------------------------------------------------------------------------

_OPENML_STATS = (
    "NumberOfMissingValues", "NumberOfInstancesWithMissingValues", "NumberOfClasses",
    "MajorityClassSize", "MinorityClassSize", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures",
)


class OpenMLAdapter(ExternalAdapter):
    name = "openml"
    authority = 1

    @property
    def _base(self) -> str:
        return self._settings.openml_api_base_url.rstrip("/")

    async def search(self, name: str) -> list[ExternalHit]:
        try:
            body = await self._fetcher.get_json(
                f"{self._base}/data/list/data_name/{quote(name.lower(), safe='')}/limit/{SEARCH_LIMIT}",
                not_found_statuses=frozenset({404, 412}),
            )
        except FetchError as exc:
            if exc.code == FetchError.NOT_FOUND:
                return []  # OpenML says "No results" with HTTP 412
            raise
        rows = _as_list(((body or {}).get("data") or {}).get("dataset") if isinstance(body, dict) else None)
        hits = []
        for item in rows:
            if not isinstance(item, dict) or item.get("status") != "active" or item.get("did") is None:
                continue
            quality = {q.get("name"): q.get("value") for q in _as_list(item.get("quality")) if isinstance(q, dict)}
            version = item.get("version")
            hits.append(ExternalHit(
                source="openml", identifier=str(item["did"]),
                name=f"{item.get('name')} (v{version})" if version is not None else str(item.get("name")),
                url=f"https://www.openml.org/d/{item['did']}",
                column_count=_int(quality.get("NumberOfFeatures")),
                row_count=_int(quality.get("NumberOfInstances")),
                popularity=float(version or 0),
            ))
        return hits

    async def fetch(self, identifier: str) -> ExternalMetadata:
        nf = frozenset({404, 412})
        description, features, qualities = await asyncio.gather(
            self._fetcher.get_json(f"{self._base}/data/{identifier}", not_found_statuses=nf),
            self._fetcher.get_json(f"{self._base}/data/features/{identifier}", not_found_statuses=nf),
            self._fetcher.get_json(f"{self._base}/data/qualities/{identifier}", not_found_statuses=nf),
            return_exceptions=True,
        )
        if isinstance(description, BaseException):
            raise description
        info = (description or {}).get("data_set_description") if isinstance(description, dict) else None
        if not isinstance(info, dict):
            raise FetchError(FetchError.NOT_FOUND, "The registry has no such dataset.")

        pairs: list[tuple[str, Optional[str], Optional[str], Optional[str]]] = []
        targets: list[str] = []
        feature_list = []
        if isinstance(features, dict):
            feature_list = _as_list(((features.get("data_features") or {}).get("feature")))
        for f in feature_list:
            if not isinstance(f, dict) or not f.get("name"):
                continue
            role = "feature"
            if str(f.get("is_target")).lower() == "true":
                role = "target"
                targets.append(str(f["name"]))
            elif str(f.get("is_row_identifier")).lower() == "true":
                role = "row_id"
            elif str(f.get("is_ignore")).lower() == "true":
                role = "ignore"
            pairs.append((str(f["name"]), f.get("data_type"), role, None))
        if not targets:
            targets = [str(t) for t in _as_list(info.get("default_target_attribute")) if t]

        stats: dict[str, float] = {}
        if isinstance(qualities, dict):
            for q in _as_list(((qualities.get("data_qualities") or {}).get("quality"))):
                if isinstance(q, dict) and q.get("name") in _OPENML_STATS + ("NumberOfInstances",):
                    try:
                        stats[q["name"]] = float(q["value"])
                    except (TypeError, ValueError):
                        pass
        columns = _schema_columns(pairs, EvidenceLevel.REGISTRY_VERIFIED)
        version = info.get("version")
        return ExternalMetadata(
            source="openml",
            identifier=str(info.get("id", identifier)),
            name=f"{info.get('name')} (v{version})" if version is not None else str(info.get("name") or identifier),
            url=f"https://www.openml.org/d/{info.get('id', identifier)}",
            description=(info.get("description") or "").strip() or None,
            publisher=info.get("creator") if isinstance(info.get("creator"), str) else None,
            license=info.get("licence") if isinstance(info.get("licence"), str) else None,
            row_count=_int(stats.get("NumberOfInstances")),
            column_count=len(columns) or None,
            files=[],
            file_types=[f".{str(info['format']).lower()}"] if info.get("format") else [],
            tags=[str(t) for t in _as_list(info.get("tag"))][:MAX_TAGS],
            created_at=_when(info.get("upload_date")),
            registry_target_columns=targets,
            retrieved_at=datetime.now(timezone.utc),
            schema_columns=columns,
            schema_level=EvidenceLevel.REGISTRY_VERIFIED if columns else EvidenceLevel.UNKNOWN,
            schema_source="OpenML features" if columns else "none",
            schema_note=None if columns else "OpenML published no feature list for this dataset.",
            registry_statistics={k: v for k, v in stats.items() if k in _OPENML_STATS},
        )


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------


def _hf_dtype(dtype: Any) -> Optional[str]:
    if isinstance(dtype, str):
        return dtype
    if isinstance(dtype, dict) and dtype:
        return str(next(iter(dtype)))
    if isinstance(dtype, list):
        return "list"
    return None


def _hf_configs(card: dict) -> list[dict]:
    info = card.get("dataset_info")
    return [c for c in _as_list(info) if isinstance(c, dict)]


class HuggingFaceAdapter(ExternalAdapter):
    name = "huggingface"
    authority = 2

    @property
    def enabled(self) -> bool:
        return bool(self._settings.external_enabled and self._settings.huggingface_enabled)

    @property
    def _base(self) -> str:
        return self._settings.huggingface_api_base_url.rstrip("/")

    async def search(self, name: str) -> list[ExternalHit]:
        body = await self._fetcher.get_json(
            f"{self._base}/datasets", {"search": name, "limit": SEARCH_LIMIT}
        )
        hits = []
        for item in _as_list(body):
            if not isinstance(item, dict) or not item.get("id") or item.get("private") or item.get("disabled"):
                continue
            dataset_id = str(item["id"])
            hits.append(ExternalHit(
                source="huggingface", identifier=dataset_id,
                name=dataset_id.split("/")[-1].replace("_", " ").replace("-", " "),
                url=f"https://huggingface.co/datasets/{dataset_id}",
                popularity=float(item.get("downloads") or 0),
            ))
        return hits

    async def fetch(self, identifier: str) -> ExternalMetadata:
        # A missing dataset and a private one are indistinguishable here (both answer 401).
        record = await self._fetcher.get_json(
            f"{self._base}/datasets/{identifier}", not_found_statuses=frozenset({401, 403, 404})
        )
        if not isinstance(record, dict) or not record.get("id"):
            raise FetchError(FetchError.NOT_FOUND, "The registry has no such dataset.")
        if record.get("private") or record.get("disabled"):
            raise FetchError(FetchError.NOT_FOUND, "The registry has no such public dataset.")
        card = record.get("cardData") if isinstance(record.get("cardData"), dict) else {}
        tags = [t for t in _as_list(record.get("tags")) if isinstance(t, str)]
        configs = _hf_configs(card)

        pairs: list[tuple[str, Optional[str], Optional[str], Optional[str]]] = []
        seen: set[str] = set()
        for config in configs[:1]:  # the first (default) configuration
            for feature in _as_list(config.get("features")):
                if isinstance(feature, dict) and feature.get("name") and feature["name"] not in seen:
                    seen.add(feature["name"])
                    pairs.append((str(feature["name"]), _hf_dtype(feature.get("dtype")), None, None))
        columns = _schema_columns(pairs, EvidenceLevel.REGISTRY_VERIFIED)

        rows = None
        if configs:
            counts = [_int(s.get("num_examples")) for s in _as_list(configs[0].get("splits")) if isinstance(s, dict)]
            counts = [c for c in counts if c is not None]
            rows = sum(counts) if counts else None
        size = _int(configs[0].get("dataset_size")) if configs else None

        level, source, note = EvidenceLevel.UNKNOWN, "none", "Hugging Face published no structured column list for this dataset."
        description = (record.get("description") or "").strip() or None
        if columns:
            level, source, note = EvidenceLevel.REGISTRY_VERIFIED, "Hugging Face dataset_info.features", None
            if len(configs) > 1:
                note = "The dataset has several configurations; the first one's columns are listed."
        else:
            prose, method = extract_columns(description)
            if prose:
                columns = _schema_columns([(n, None, None, None) for n in prose], EvidenceLevel.INFERRED)
                level, source = EvidenceLevel.INFERRED, f"description ({method})"
                note = "Column names were read from the description's text and may be incomplete."

        tasks = [t.split(":", 1)[1].replace("-", " ") for t in tags if t.startswith("task_categories:")]
        tasks += [str(t).replace("-", " ") for t in _as_list(card.get("task_categories"))]
        license_text = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None) or (
            _as_list(card.get("license")) or [None])[0]
        return ExternalMetadata(
            source="huggingface",
            identifier=str(record["id"]),
            name=str(card.get("pretty_name") or str(record["id"]).split("/")[-1]),
            url=f"https://huggingface.co/datasets/{record['id']}",
            description=description[:4000] if description else None,
            publisher=record.get("author"),
            license=str(license_text) if license_text else None,
            size_bytes=size,
            row_count=rows,
            column_count=len(columns) or None,
            files=[str(s.get("rfilename")) for s in _as_list(record.get("siblings")) if isinstance(s, dict)][:MAX_FILES_LISTED],
            file_types=sorted({f".{t.split(':', 1)[1]}" for t in tags if t.startswith("format:")}),
            tasks=list(dict.fromkeys(tasks)),
            tags=[t for t in tags if ":" not in t][:MAX_TAGS],
            updated_at=_when(record.get("lastModified")),
            created_at=_when(record.get("createdAt")),
            retrieved_at=datetime.now(timezone.utc),
            schema_columns=columns,
            schema_level=level,
            schema_source=source,
            schema_note=note,
        )


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------


def _kf(record: dict, key: str) -> Any:
    """Kaggle sends every field twice: ``<key>`` (False when absent) and ``<key>Nullable``."""
    value = record.get(f"{key}Nullable")
    return value if value is not None else (None if record.get(key) is False else record.get(key))


class KaggleAdapter(ExternalAdapter):
    name = "kaggle"
    authority = 3

    @property
    def enabled(self) -> bool:
        return bool(self._settings.external_enabled and self._settings.kaggle_enabled)

    @property
    def _base(self) -> str:
        return self._settings.kaggle_api_base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        token = self._settings.kaggle_api_token
        return {"Authorization": f"Bearer {token}"} if token else {}

    async def search(self, name: str) -> list[ExternalHit]:
        body = await self._fetcher.get_json(
            f"{self._base}/datasets/list", {"search": name}, self._headers()
        )
        hits = []
        for item in _as_list(body):
            if not isinstance(item, dict):
                continue
            ref, title = _kf(item, "ref"), _kf(item, "title")
            if not ref or not title:
                continue
            hits.append(ExternalHit(
                source="kaggle", identifier=str(ref), name=str(title),
                url=f"https://www.kaggle.com/datasets/{ref}",
                popularity=float(_kf(item, "downloadCount") or 0),
            ))
        return hits[:SEARCH_LIMIT]

    async def fetch(self, identifier: str) -> ExternalMetadata:
        view, files = await asyncio.gather(
            self._fetcher.get_json(f"{self._base}/datasets/view/{identifier}", headers=self._headers()),
            self._fetcher.get_json(f"{self._base}/datasets/list/{identifier}", headers=self._headers()),
            return_exceptions=True,
        )
        if isinstance(view, BaseException):
            raise view
        if not isinstance(view, dict):
            raise FetchError(FetchError.INVALID_RESPONSE, "The registry returned an unexpected response.")

        names = []
        if isinstance(files, dict):
            names = [str(f.get("name")) for f in _as_list(files.get("datasetFiles")) if isinstance(f, dict) and f.get("name")]
        subtitle = _kf(view, "subtitle")
        body = _kf(view, "description")
        description = "\n\n".join(p for p in (subtitle, body) if isinstance(p, str) and p.strip()) or None

        prose, method = extract_columns(body if isinstance(body, str) else None)
        columns = _schema_columns([(n, None, None, None) for n in prose], EvidenceLevel.INFERRED) if prose else []
        extensions = sorted({"." + n.rsplit(".", 1)[-1].lower() for n in names if "." in n})
        return ExternalMetadata(
            source="kaggle",
            identifier=identifier,
            name=str(_kf(view, "title") or identifier.split("/")[-1]),
            url=f"https://www.kaggle.com/datasets/{identifier}",
            description=description[:6000] if description else None,
            publisher=_kf(view, "ownerName") or _kf(view, "creatorName"),
            license=_kf(view, "licenseName"),
            size_bytes=_int(_kf(view, "totalBytes")),
            files=names[:MAX_FILES_LISTED],
            file_types=extensions,
            tags=[str(t.get("name") or t.get("nameNullable")) for t in _as_list(view.get("tags")) if isinstance(t, dict)][:MAX_TAGS],
            updated_at=_when(_kf(view, "lastUpdated")),
            retrieved_at=datetime.now(timezone.utc),
            schema_columns=columns,
            schema_level=EvidenceLevel.INFERRED if columns else EvidenceLevel.UNKNOWN,
            schema_source=f"description ({method})" if columns else "none",
            schema_note=(
                "Kaggle publishes no structured schema; these column names were read from the "
                "owner's description and may be incomplete or inexact."
                if columns else
                "Kaggle publishes no structured schema and the description does not list the columns."
            ),
        )


def build_adapters(settings: Settings, fetcher: SafeFetcher) -> list[ExternalAdapter]:
    """The enabled adapters, most authoritative first."""
    adapters: list[ExternalAdapter] = [
        UCIAdapter(settings, fetcher), OpenMLAdapter(settings, fetcher),
        HuggingFaceAdapter(settings, fetcher), KaggleAdapter(settings, fetcher),
    ]
    return sorted((a for a in adapters if a.enabled), key=lambda a: a.authority)
