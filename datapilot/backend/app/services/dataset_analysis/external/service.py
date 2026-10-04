"""Researching a dataset the user only named or linked, in trusted registries.

This service answers one question honestly: *what do the registries say about this
dataset?* It never downloads the dataset and never reports a row-level fact the registry
did not state.

Resolving a **link** is exact: the link names the registry and the dataset.

Resolving a **name** searches every enabled registry once and keeps only datasets whose
identifying words are *identical* to the name (a near-match is not a match). Then:

* one exact match anywhere           -> RESOLVED
* several (other versions, other
  owners, other registries)          -> PROBABLE: the most authoritative is analysed, the
                                        rest are listed, and the answer says it needs
                                        confirming. Registries disagree about what a famous
                                        name means (OpenML has five "Titanic"s), and pretending
                                        otherwise would be a guess presented as a fact.
* only partial matches               -> AMBIGUOUS: candidates, no analysis
* nothing                            -> UNRESOLVED

Authority order for choosing among exact matches is where a dataset *originates* (UCI,
OpenML, then the user-uploaded hubs Hugging Face and Kaggle); popularity only breaks ties
within a registry. When the top two disagree about the schema, that is reported as a conflict.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.config import Settings
from app.core.logging_config import get_logger
from app.schemas.dataset_analysis import (
    DatasetAlternative,
    DatasetInputResolution,
    EvidenceLevel,
    ExternalConflict,
    ExternalMetadata,
    InputType,
    ResolutionStatus,
)
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisError
from app.services.dataset_analysis.external.adapters import (
    ExternalAdapter,
    ExternalHit,
    build_adapters,
)
from app.services.dataset_analysis.external.fetcher import FetchError, SafeFetcher
from app.services.dataset_analysis.references import normalize_dataset_name
from app.services.datasets.deduplicator import name_token_key

logger = get_logger(__name__)

MAX_ALTERNATIVES = 5
#: How many exact matches are tried in order if the first cannot be fetched.
MAX_FETCH_ATTEMPTS = 2
CONFIDENCE_SINGLE_MATCH = 0.92
CONFIDENCE_AUTHORITATIVE_AMONG_MANY = 0.75
CONFIDENCE_USER_UPLOADED_AMONG_MANY = 0.55
_VERSION_SUFFIX = re.compile(r"\s*\(v\d+\)\s*$")

_ERROR_FOR = {
    FetchError.TIMEOUT: (ErrorCode.DATASET_SOURCE_TIMEOUT, 504),
    FetchError.RATE_LIMITED: (ErrorCode.DATASET_RATE_LIMITED, 429),
    FetchError.UNAVAILABLE: (ErrorCode.DATASET_SOURCE_UNAVAILABLE, 502),
    FetchError.INVALID_RESPONSE: (ErrorCode.DATASET_INVALID_RESPONSE, 502),
    FetchError.UNSAFE: (ErrorCode.DATASET_URL_UNSUPPORTED, 422),
}


def _analysis_error(error: FetchError, source: str) -> AnalysisError:
    code, status = _ERROR_FOR.get(error.code, (ErrorCode.DATASET_SOURCE_UNAVAILABLE, 502))
    return AnalysisError(code, f"{source}: {error.message}", status_code=status, details={"source": source})


@dataclass
class ExternalResearch:
    resolution: DatasetInputResolution
    metadata: Optional[ExternalMetadata] = None
    conflicts: list[ExternalConflict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _base_name(name: str) -> str:
    return _VERSION_SUFFIX.sub("", name)


def _column_key(names: list[str]) -> frozenset[str]:
    return frozenset(re.sub(r"[^a-z0-9]+", "", n.lower()) for n in names if n)


class ExternalDatasetMetadataService:
    def __init__(
        self,
        settings: Settings,
        fetcher: Optional[SafeFetcher] = None,
        adapters: Optional[list[ExternalAdapter]] = None,
    ) -> None:
        self._settings = settings
        self._fetcher = fetcher or SafeFetcher(settings)
        self._adapters = adapters if adapters is not None else build_adapters(settings, self._fetcher)

    # ------------------------------------------------------------------

    async def research(self, resolution: DatasetInputResolution) -> ExternalResearch:
        if not self._settings.external_enabled:
            return ExternalResearch(
                resolution=resolution,
                limitations=["External dataset research is disabled, so no registry was consulted."],
            )
        if resolution.input_type is InputType.DATASET_URL and resolution.source:
            return await self._by_url(resolution)
        if resolution.input_type is InputType.PUBLIC_DATASET_NAME:
            return await self._by_name(resolution)
        return ExternalResearch(resolution=resolution)

    def _adapter(self, source: str) -> Optional[ExternalAdapter]:
        return next((a for a in self._adapters if a.name == source), None)

    # ------------------------------------------------------------------ URL

    async def _by_url(self, resolution: DatasetInputResolution) -> ExternalResearch:
        adapter = self._adapter(resolution.source or "")
        if adapter is None:
            raise AnalysisError(
                ErrorCode.DATASET_URL_UNSUPPORTED,
                f"Researching {resolution.source} datasets is not enabled.",
            )
        try:
            metadata = await adapter.fetch(resolution.source_identifier or "")
        except FetchError as exc:
            if exc.code == FetchError.NOT_FOUND:
                return ExternalResearch(
                    resolution=resolution.model_copy(update={
                        "resolution_status": ResolutionStatus.UNRESOLVED,
                        "resolution_confidence": 0.0,
                        "evidence": resolution.evidence + [f"{adapter.name} has no public dataset at that link"],
                        "needs_confirmation": True,
                        "clarification": "Check the link, or upload the file.",
                    }),
                    limitations=[
                        f"No public {adapter.name} dataset was found at that link, so nothing about its "
                        "structure could be verified."
                    ],
                )
            raise _analysis_error(exc, adapter.name) from exc
        return ExternalResearch(resolution=self._resolved(resolution, metadata, ResolutionStatus.RESOLVED, 0.97), metadata=metadata)

    # ----------------------------------------------------------------- name

    async def _by_name(self, resolution: DatasetInputResolution) -> ExternalResearch:
        name = resolution.normalized_dataset_name or ""
        wanted_key = name_token_key(name)
        adapters = [a for a in self._adapters if not resolution.source or a.name == resolution.source]
        if not adapters:
            raise AnalysisError(
                ErrorCode.DATASET_URL_UNSUPPORTED,
                f"Researching {resolution.source} datasets is not enabled.",
            )

        results = await asyncio.gather(*(a.search(name) for a in adapters), return_exceptions=True)
        hits: list[tuple[ExternalAdapter, ExternalHit]] = []
        failures: list[tuple[str, FetchError]] = []
        for adapter, result in zip(adapters, results):
            if isinstance(result, FetchError):
                failures.append((adapter.name, result))
            elif isinstance(result, BaseException):
                logger.error("Registry search raised %s", type(result).__name__)
                failures.append((adapter.name, FetchError(FetchError.UNAVAILABLE, "The registry failed unexpectedly.")))
            else:
                hits.extend((adapter, h) for h in result)

        limitations: list[str] = []
        if failures:
            limitations.append(
                "These registries could not be searched: "
                + ", ".join(f"{s} ({e.code.lower().replace('_', ' ')})" for s, e in failures) + "."
            )
        if failures and len(failures) == len(adapters):
            raise _analysis_error(failures[0][1], failures[0][0])

        exact = [(a, h) for a, h in hits if wanted_key and name_token_key(_base_name(h.name)) == wanted_key]
        if not exact:
            return self._no_exact(resolution, hits, wanted_key, limitations)

        exact.sort(key=lambda ah: (ah[0].authority, -ah[1].popularity, ah[1].identifier))
        metadata, chosen_index = await self._fetch_first(exact, failures)
        if metadata is None:
            limitations.append(
                "A registry listed the dataset but its record could not be read, so nothing about its "
                "structure could be verified."
            )
            return ExternalResearch(
                resolution=resolution.model_copy(update={
                    "resolution_status": ResolutionStatus.UNRESOLVED, "resolution_confidence": 0.0,
                    "evidence": resolution.evidence + ["a registry listed the name but its record was unavailable"],
                    "needs_confirmation": True,
                }),
                limitations=limitations,
            )

        primary_adapter, primary_hit = exact[chosen_index]
        others = [pair for i, pair in enumerate(exact) if i != chosen_index]
        if not others:
            status, confidence = ResolutionStatus.RESOLVED, CONFIDENCE_SINGLE_MATCH
        else:
            status = ResolutionStatus.PROBABLE
            confidence = (
                CONFIDENCE_AUTHORITATIVE_AMONG_MANY if primary_adapter.authority <= 1
                else CONFIDENCE_USER_UPLOADED_AMONG_MANY
            )
        alternatives = [
            DatasetAlternative(
                name=h.name, source=h.source, identifier=h.identifier, url=h.url,
                column_count=h.column_count, row_count=h.row_count,
            )
            for _, h in others[:MAX_ALTERNATIVES]
        ]
        conflicts = await self._conflicts(metadata, others)
        out = self._resolved(resolution, metadata, status, confidence)
        out = out.model_copy(update={
            "alternatives": alternatives,
            "needs_confirmation": status is ResolutionStatus.PROBABLE,
            "clarification": (
                f"{len(others) + 1} datasets share the name '{name}'. DataPilot analysed the "
                f"{primary_hit.source} one ({metadata.name}); give a link if you meant another."
                if others else None
            ),
            "evidence": out.evidence + (
                [f"{len(others) + 1} registries/versions share this name; the most authoritative was chosen"]
                if others else ["exactly one registry dataset has this name"]
            ),
        })
        if others:
            limitations.append(
                "Several datasets share this name and their structure can differ; confirm which one you mean."
            )
        return ExternalResearch(resolution=out, metadata=metadata, conflicts=conflicts, limitations=limitations)

    def _no_exact(
        self,
        resolution: DatasetInputResolution,
        hits: list[tuple[ExternalAdapter, ExternalHit]],
        wanted_key: Optional[str],
        limitations: list[str],
    ) -> ExternalResearch:
        wanted = set((wanted_key or "").split())
        near = [(a, h) for a, h in hits if wanted and wanted <= set((name_token_key(_base_name(h.name)) or "").split())]
        near.sort(key=lambda ah: (ah[0].authority, -ah[1].popularity, ah[1].identifier))
        if not near:
            return ExternalResearch(
                resolution=resolution.model_copy(update={
                    "resolution_status": ResolutionStatus.UNRESOLVED, "resolution_confidence": 0.0,
                    "evidence": resolution.evidence + ["no registry has a dataset with this name"],
                    "needs_confirmation": True,
                    "clarification": "No public dataset by that name was found. Check the spelling, give a link, or upload the file.",
                }),
                limitations=limitations + ["No dataset by that name was found, so nothing about its structure could be verified."],
            )
        return ExternalResearch(
            resolution=resolution.model_copy(update={
                "input_type": InputType.AMBIGUOUS_DATASET_REFERENCE,
                "resolution_status": ResolutionStatus.AMBIGUOUS, "resolution_confidence": 0.3,
                "alternatives": [
                    DatasetAlternative(name=h.name, source=h.source, identifier=h.identifier, url=h.url,
                                       column_count=h.column_count, row_count=h.row_count)
                    for _, h in near[:MAX_ALTERNATIVES]
                ],
                "evidence": resolution.evidence + [f"{len(near)} datasets partly match the name but none match it exactly"],
                "needs_confirmation": True,
                "clarification": "Several datasets resemble that name. Choose one by its link, or upload the file.",
            }),
            limitations=limitations + ["The name matched no dataset exactly, so none was analysed."],
        )

    async def _fetch_first(
        self,
        exact: list[tuple[ExternalAdapter, ExternalHit]],
        failures: list[tuple[str, FetchError]],
    ) -> tuple[Optional[ExternalMetadata], int]:
        last_error: Optional[tuple[str, FetchError]] = None
        for index, (adapter, hit) in enumerate(exact[:MAX_FETCH_ATTEMPTS]):
            try:
                return await adapter.fetch(hit.identifier), index
            except FetchError as exc:
                last_error = (adapter.name, exc)
                if exc.code != FetchError.NOT_FOUND:
                    failures.append((adapter.name, exc))
        if last_error is not None and last_error[1].code != FetchError.NOT_FOUND:
            raise _analysis_error(last_error[1], last_error[0])
        return None, 0

    async def _conflicts(
        self, primary: ExternalMetadata, others: list[tuple[ExternalAdapter, ExternalHit]]
    ) -> list[ExternalConflict]:
        """Compare the primary with the strongest alternative from another registry."""
        out: list[ExternalConflict] = []
        for adapter, hit in others:
            if hit.row_count is not None and primary.row_count is not None and hit.row_count != primary.row_count:
                out.append(ExternalConflict(
                    field="row_count",
                    values={f"{primary.source}:{primary.identifier}": str(primary.row_count),
                            f"{hit.source}:{hit.identifier}": str(hit.row_count)},
                    note="The registries report different row counts for datasets with this name.",
                ))
                break
        candidate = next(((a, h) for a, h in others if h.source != primary.source and a.authority <= 2), None)
        if candidate is None or not primary.schema_columns:
            return out
        adapter, hit = candidate
        try:
            other = await adapter.fetch(hit.identifier)
        except FetchError:
            return out
        if not other.schema_columns:
            return out
        a = [c.original_name or c.normalized_name for c in primary.schema_columns]
        b = [c.original_name or c.normalized_name for c in other.schema_columns]
        if _column_key(a) != _column_key(b):
            only_a, only_b = sorted(_column_key(a) - _column_key(b)), sorted(_column_key(b) - _column_key(a))
            out.append(ExternalConflict(
                field="columns",
                values={
                    f"{primary.source}:{primary.identifier}": f"{len(a)} columns: {', '.join(a[:8])}{'...' if len(a) > 8 else ''}",
                    f"{other.source}:{other.identifier}": f"{len(b)} columns: {', '.join(b[:8])}{'...' if len(b) > 8 else ''}",
                },
                note=(
                    "The two registries list different columns for datasets with this name"
                    + (f" (only in the first: {', '.join(only_a[:5])})" if only_a else "")
                    + (f" (only in the second: {', '.join(only_b[:5])})" if only_b else "")
                    + "."
                ),
            ))
        return out

    # ------------------------------------------------------------------

    def _resolved(
        self,
        resolution: DatasetInputResolution,
        metadata: ExternalMetadata,
        status: ResolutionStatus,
        confidence: float,
    ) -> DatasetInputResolution:
        available = [
            label for label, present in (
                ("description", metadata.description), ("license", metadata.license),
                ("size", metadata.size_bytes), ("row_count", metadata.row_count),
                ("files", metadata.files), ("tasks", metadata.tasks),
                ("schema", metadata.schema_columns),
            ) if present
        ]
        schema_known = bool(metadata.schema_columns)
        return resolution.model_copy(update={
            "normalized_dataset_name": normalize_dataset_name(_base_name(metadata.name)) or resolution.normalized_dataset_name,
            "source": metadata.source,
            "source_url": metadata.url,
            "source_identifier": metadata.identifier,
            "resolution_status": status,
            "resolution_confidence": confidence,
            "available_metadata": available,
            "schema_available": schema_known,
            "schema_evidence_level": metadata.schema_level if schema_known else EvidenceLevel.UNKNOWN,
        })
