"""Analysing a dataset the user already has (User B, Phase 4).

Two paths with one rule: *if DataPilot has the data, it analyses the data; if it only has a
name or a link, it reports what trusted registries say and is explicit about everything it could
not verify.*

    uploaded file ----> secure ingestion -> profile -> targets -> quality -> tasks   (DATAPILOT_PARSED)
    name or link  ----> registry research -> schema/metadata -> targets/tasks from the schema
                                                                (REGISTRY_VERIFIED / INFERRED / UNKNOWN)

The file path is CPU- and disk-bound, so it runs in a worker thread under a wall-clock budget
and the event loop stays responsive. The budget is cooperative (Python cannot kill a thread), so
on expiry the request is answered with a timeout while the worker stops at its next check.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Optional

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.dataset_analysis import (
    AnalysisMetadata,
    AnalysisMode,
    DatasetAnalysis,
    DatasetIdentity,
    DatasetInputResolution,
    EvidenceLevel,
    InputType,
    ProvenanceRecord,
    ResolutionStatus,
    SchemaColumn,
    SchemaInfo,
)
from app.schemas.project import ErrorCode
from app.services.dataset_analysis.errors import AnalysisError, AnalysisTimeout
from app.services.dataset_analysis.external.service import ExternalDatasetMetadataService, ExternalResearch
from app.services.dataset_analysis.external_ml import detect_external_targets, detect_external_tasks
from app.services.dataset_analysis.ingestion import parse_file
from app.services.dataset_analysis.input_resolver import DatasetInputResolver
from app.services.dataset_analysis.pipeline import ParsedResult, analyze_table
from app.services.dataset_analysis.safety import (
    Deadline,
    file_extension,
    sanitize_display_name,
    store_upload,
    validate_extension,
)

logger = get_logger(__name__)

#: Slack on top of the analysis budget before the request itself is abandoned, so the worker's
#: own cooperative deadline normally fires first and reports the stage it was in.
TIMEOUT_GRACE_SECONDS = 5.0

NOT_PARSED_LIMITATION = (
    "DataPilot has not parsed the raw file, so row-level statistics, data-quality checks and "
    "feature importance cannot be verified. Upload the file for full analysis."
)
NO_SCHEMA_LIMITATION = (
    "The public source does not expose enough schema information to verify the dataset's "
    "structure. Upload the file for full analysis."
)


@dataclass
class AnalysisRequest:
    stream: Optional[BinaryIO] = None
    file_name: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_url: Optional[str] = None
    message: Optional[str] = None
    source_hint: Optional[str] = None
    target_hint: Optional[str] = None
    sheet_name: Optional[str] = None


@dataclass
class AnalysisOutcome:
    analysis: DatasetAnalysis
    parsed: Optional[ParsedResult] = None
    research: Optional[ExternalResearch] = None


def _provenance_for_file(result: ParsedResult, name: str) -> list[ProvenanceRecord]:
    parsed = EvidenceLevel.DATAPILOT_PARSED
    records = [
        ProvenanceRecord(claim=f"the schema of {name} ({len(result.columns)} columns)", source="uploaded file",
                         evidence_level=parsed, confidence=1.0),
        ProvenanceRecord(claim=f"the file has {result.n_rows} rows analysed", source="uploaded file",
                         evidence_level=parsed, confidence=1.0),
        ProvenanceRecord(claim="column types and statistics", source="uploaded file",
                         evidence_level=parsed, confidence=1.0),
        ProvenanceRecord(claim=f"{len(result.quality.findings)} data-quality finding(s)", source="uploaded file",
                         evidence_level=parsed, confidence=1.0),
    ]
    if result.targets:
        best = result.targets[0]
        records.append(ProvenanceRecord(
            claim=f"'{best.column}' is the leading target candidate", source="name, structure and association",
            evidence_level=EvidenceLevel.HEURISTIC, confidence=best.confidence))
    return records


def _provenance_for_external(research: ExternalResearch) -> list[ProvenanceRecord]:
    meta = research.metadata
    resolution = research.resolution
    if meta is None:
        return []
    registry = EvidenceLevel.REGISTRY_VERIFIED
    when = meta.retrieved_at
    by_link = resolution.input_type is InputType.DATASET_URL
    records = [ProvenanceRecord(
        claim=f"the dataset is {meta.name} ({meta.source}:{meta.identifier})", source=meta.source,
        evidence_level=registry if by_link else EvidenceLevel.INFERRED,
        confidence=resolution.resolution_confidence, retrieved_at=when)]
    if meta.schema_columns:
        records.append(ProvenanceRecord(
            claim=f"the schema lists {len(meta.schema_columns)} columns", source=meta.schema_source,
            evidence_level=meta.schema_level, confidence=0.95 if meta.schema_level is registry else 0.5,
            retrieved_at=when))
    for label, value in (("row count", meta.row_count), ("license", meta.license), ("description", meta.description)):
        if value:
            records.append(ProvenanceRecord(
                claim=f"the {label} is published", source=meta.source, evidence_level=registry,
                confidence=0.95, retrieved_at=when))
    if meta.registry_target_columns:
        records.append(ProvenanceRecord(
            claim=f"the registry designates {', '.join(meta.registry_target_columns)} as the target",
            source=meta.source, evidence_level=registry, confidence=0.95, retrieved_at=when))
    return records


class DatasetAnalysisService:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        resolver: Optional[DatasetInputResolver] = None,
        external: Optional[ExternalDatasetMetadataService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._resolver = resolver or DatasetInputResolver()
        self._external = external or ExternalDatasetMetadataService(self._settings)

    # ------------------------------------------------------------------

    async def analyze(self, request: AnalysisRequest) -> AnalysisOutcome:
        started = datetime.now(timezone.utc)
        clock = time.monotonic()
        resolution = self._resolver.resolve(
            has_file=request.stream is not None,
            file_name=request.file_name,
            dataset_name=request.dataset_name,
            dataset_url=request.dataset_url,
            message=request.message,
            source_hint=request.source_hint,
        )
        logger.info("Dataset analysis: input=%s source=%s", resolution.input_type.value, resolution.source)

        if resolution.input_type is InputType.UPLOADED_FILE:
            outcome = await self._analyze_file(request, resolution, started, clock)
        else:
            outcome = await self._analyze_reference(request, resolution, started, clock)
        logger.info(
            "Dataset analysis finished: mode=%s ms=%.0f",
            outcome.analysis.mode.value, outcome.analysis.metadata.duration_ms,
        )
        return outcome

    # ------------------------------------------------------------------ file

    def _process_file(self, request: AnalysisRequest, display: str) -> tuple[ParsedResult, bool]:
        deadline = Deadline(self._settings.analysis_time_limit_seconds)
        suffix = file_extension(display)  # validated against the allowlist before this point
        with store_upload(request.stream, self._settings.analysis_max_upload_bytes, suffix) as stored:  # type: ignore[arg-type]
            table = parse_file(
                stored.path, display, stored.size_bytes, self._settings, deadline, request.sheet_name
            )
            result = analyze_table(table, self._settings, deadline, request.target_hint)
        return result, stored.cleanup_failed

    async def _analyze_file(
        self, request: AnalysisRequest, resolution: DatasetInputResolution, started: datetime, clock: float
    ) -> AnalysisOutcome:
        display = sanitize_display_name(request.file_name)
        validate_extension(display)  # refuse an unsupported type before reading a byte
        try:
            result, cleanup_failed = await asyncio.wait_for(
                asyncio.to_thread(self._process_file, request, display),
                timeout=self._settings.analysis_time_limit_seconds + TIMEOUT_GRACE_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise AnalysisTimeout("analysis") from exc

        warnings = list(result.table.info.warnings)
        if cleanup_failed:
            warnings.append("A temporary copy of the upload could not be removed from the server.")
        limitations = self._file_limitations(result)
        name = resolution.normalized_dataset_name or display
        analysis = DatasetAnalysis(
            mode=AnalysisMode.FILE_PARSED,
            dataset_identity=DatasetIdentity(name=name, source="upload", resolution_confidence=1.0),
            input_resolution=resolution.model_copy(update={
                "schema_available": True, "schema_evidence_level": EvidenceLevel.DATAPILOT_PARSED}),
            ingestion=result.table.info,
            schema=result.schema,
            column_profiles=[c.profile for c in result.columns if c.profile is not None],
            quality=result.quality,
            target_candidates=result.targets,
            task_candidates=result.tasks,
            warnings=warnings,
            limitations=limitations,
            provenance=_provenance_for_file(result, name),
            metadata=self._metadata(AnalysisMode.FILE_PARSED, started, clock),
        )
        return AnalysisOutcome(analysis=analysis, parsed=result)

    @staticmethod
    def _file_limitations(result: ParsedResult) -> list[str]:
        out = []
        if result.table.info.truncated:
            out.append("The file was larger than the analysis limits; every figure describes the rows read.")
        if not result.targets:
            out.append("No column looked like a prediction target, so only unsupervised tasks are listed.")
        if result.n_rows < 30:
            out.append(f"Only {result.n_rows} rows were read: too few to draw reliable conclusions.")
        return out

    # ------------------------------------------------------------- reference

    async def _analyze_reference(
        self, request: AnalysisRequest, resolution: DatasetInputResolution, started: datetime, clock: float
    ) -> AnalysisOutcome:
        if resolution.input_type is InputType.NO_DATASET_REFERENCE:
            raise AnalysisError(
                ErrorCode.DATASET_INPUT_REQUIRED, resolution.clarification or "Provide a file, dataset name or link.",
                details={"input_type": resolution.input_type.value},
            )
        if resolution.resolution_status is ResolutionStatus.UNSUPPORTED:
            raise AnalysisError(
                ErrorCode.DATASET_URL_UNSUPPORTED,
                resolution.clarification or "That link is not one DataPilot can research.",
                details={"reason": "; ".join(resolution.evidence)},
            )

        research = await self._external.research(resolution)
        meta = research.metadata
        resolution = research.resolution
        warnings = list(research.warnings)

        schema = self._schema_from_metadata(meta)
        targets = detect_external_targets(meta, request.target_hint) if meta else []
        if request.target_hint and meta and meta.schema_columns and not any(t.user_specified for t in targets):
            raise AnalysisError(
                ErrorCode.TARGET_COLUMN_NOT_FOUND,
                "The requested target column is not in the dataset's published schema.",
                details={"columns_available": [c.normalized_name for c in meta.schema_columns][:50]},
            )
        if request.target_hint and (meta is None or not meta.schema_columns):
            warnings.append(
                "The requested target column could not be checked because no schema is available for this dataset."
            )
        tasks = detect_external_tasks(meta, targets) if meta else []

        limitations = list(research.limitations)
        if meta is not None:
            limitations.append(NOT_PARSED_LIMITATION)
            if not meta.schema_columns:
                limitations.append(NO_SCHEMA_LIMITATION)
            elif meta.schema_level is not EvidenceLevel.REGISTRY_VERIFIED:
                limitations.append(
                    "The column list was read from the description's text, not published as structured "
                    "data, so it may be incomplete or inexact."
                )
            if meta.schema_note:
                limitations.append(meta.schema_note)
            if not targets:
                limitations.append("No column could be identified as a likely target from the published schema.")
        if resolution.input_type is InputType.AMBIGUOUS_DATASET_REFERENCE and resolution.clarification:
            limitations.append(resolution.clarification)

        analysis = DatasetAnalysis(
            mode=AnalysisMode.EXTERNAL_METADATA,
            dataset_identity=DatasetIdentity(
                name=(meta.name if meta else resolution.normalized_dataset_name),
                source=resolution.source, url=resolution.source_url, identifier=resolution.source_identifier,
                resolution_confidence=resolution.resolution_confidence),
            input_resolution=resolution,
            external=meta,
            conflicts=research.conflicts,
            schema=schema,
            target_candidates=targets,
            task_candidates=tasks,
            warnings=warnings,
            limitations=list(dict.fromkeys(limitations)),
            provenance=_provenance_for_external(research),
            metadata=self._metadata(AnalysisMode.EXTERNAL_METADATA, started, clock),
        )
        return AnalysisOutcome(analysis=analysis, research=research)

    @staticmethod
    def _schema_from_metadata(meta) -> SchemaInfo:
        if meta is None or not meta.schema_columns:
            return SchemaInfo(
                level=EvidenceLevel.UNKNOWN, source="none", confidence=0.0,
                column_count=meta.column_count if meta else None,
                row_count=meta.row_count if meta else None,
                row_count_evidence=EvidenceLevel.REGISTRY_VERIFIED if meta and meta.row_count is not None else EvidenceLevel.UNKNOWN,
                note=(meta.schema_note if meta else "No dataset was identified, so no schema is known."),
            )
        return SchemaInfo(
            level=meta.schema_level,
            source=meta.schema_source,
            confidence=0.95 if meta.schema_level is EvidenceLevel.REGISTRY_VERIFIED else 0.5,
            column_count=len(meta.schema_columns),
            row_count=meta.row_count,
            row_count_evidence=EvidenceLevel.REGISTRY_VERIFIED if meta.row_count is not None else EvidenceLevel.UNKNOWN,
            columns=list(meta.schema_columns),
            note=meta.schema_note,
        )

    # ------------------------------------------------------------------

    def _metadata(self, mode: AnalysisMode, started: datetime, clock: float) -> AnalysisMetadata:
        s = self._settings
        return AnalysisMetadata(
            analysis_id=str(uuid.uuid4()),
            mode=mode,
            started_at=started,
            duration_ms=round((time.monotonic() - clock) * 1000, 2),
            limits={
                "max_upload_bytes": s.analysis_max_upload_bytes, "max_rows": s.analysis_max_rows,
                "max_columns": s.analysis_max_columns, "max_cells": s.analysis_max_cells,
                "sample_rows": s.analysis_sample_rows, "random_seed": s.analysis_random_seed,
                "time_limit_seconds": s.analysis_time_limit_seconds,
            },
        )
