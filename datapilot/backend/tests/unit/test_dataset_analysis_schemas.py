"""The structural guarantees of the Phase 4 models.

The rule these tests protect: *if DataPilot did not parse the file, it must not report
anything that only parsing can produce.* It is enforced by the models, so a response
that breaks it cannot even be constructed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.schemas.dataset_analysis import (
    AnalysisMetadata,
    AnalysisMode,
    CandidateState,
    ColumnProfile,
    ColumnType,
    DatasetAnalysis,
    DatasetIdentity,
    DatasetInputResolution,
    EvidenceItem,
    EvidenceLevel,
    IngestionInfo,
    InputType,
    MLTask,
    NumericStats,
    QualityReport,
    ResolutionStatus,
    SchemaColumn,
    SchemaInfo,
    TargetCandidate,
    TaskCandidate,
)

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)


def _resolution(**kw):
    base = dict(input_type=InputType.PUBLIC_DATASET_NAME)
    base.update(kw)
    return DatasetInputResolution(**base)


def _meta(mode):
    return AnalysisMetadata(analysis_id="a1", mode=mode, started_at=NOW)


def _profile(**kw):
    base = dict(
        index=0, original_name="age", normalized_name="age", inferred_type=ColumnType.NUMERIC,
        count=10, missing_count=0, missing_pct=0.0, unique_count=10, uniqueness_ratio=1.0,
    )
    base.update(kw)
    return ColumnProfile(**base)


def _external(**kw):
    base = dict(
        mode=AnalysisMode.EXTERNAL_METADATA,
        dataset_identity=DatasetIdentity(name="Iris"),
        input_resolution=_resolution(),
        schema=SchemaInfo(
            level=EvidenceLevel.REGISTRY_VERIFIED, source="uci",
            columns=[SchemaColumn(original_name="x", normalized_name="x", evidence=EvidenceLevel.REGISTRY_VERIFIED)],
        ),
        metadata=_meta(AnalysisMode.EXTERNAL_METADATA),
    )
    base.update(kw)
    return DatasetAnalysis(**base)


def _ingestion():
    return IngestionInfo(file_type="csv", display_name="a.csv", size_bytes=10, rows_read=3)


class TestExternalAnalysisCannotClaimRawData:
    def test_a_registry_schema_is_a_valid_external_analysis(self):
        assert _external().mode is AnalysisMode.EXTERNAL_METADATA

    def test_it_cannot_carry_column_statistics(self):
        with pytest.raises(ValueError, match="column statistics need the raw file"):
            _external(column_profiles=[_profile()])

    def test_it_cannot_carry_a_quality_report(self):
        with pytest.raises(ValueError, match="quality report needs the raw file"):
            _external(quality=QualityReport())

    def test_it_cannot_describe_an_ingestion(self):
        with pytest.raises(ValueError, match="cannot have ingestion"):
            _external(ingestion=_ingestion())

    def test_its_schema_cannot_be_datapilot_parsed(self):
        with pytest.raises(ValueError, match="nothing is DATAPILOT_PARSED"):
            _external(
                schema=SchemaInfo(
                    level=EvidenceLevel.DATAPILOT_PARSED, source="x",
                    columns=[SchemaColumn(original_name="x", normalized_name="x", evidence=EvidenceLevel.DATAPILOT_PARSED)],
                )
            )

    def test_its_row_count_cannot_be_datapilot_parsed(self):
        with pytest.raises(ValueError, match="nothing is DATAPILOT_PARSED"):
            _external(
                schema=SchemaInfo(
                    level=EvidenceLevel.REGISTRY_VERIFIED, source="uci", row_count=150,
                    row_count_evidence=EvidenceLevel.DATAPILOT_PARSED,
                )
            )

    def test_its_target_evidence_cannot_claim_a_file_was_read(self):
        candidate = TargetCandidate(
            column="y", state=CandidateState.POSSIBLE, confidence=0.5, score=1.0,
            evidence=[EvidenceItem(statement="2 distinct values", level=EvidenceLevel.DATAPILOT_PARSED)],
        )
        with pytest.raises(ValueError, match="without a file"):
            _external(target_candidates=[candidate])

    def test_a_registry_row_count_is_allowed_at_registry_level(self):
        analysis = _external(
            schema=SchemaInfo(level=EvidenceLevel.REGISTRY_VERIFIED, source="uci", row_count=150,
                              row_count_evidence=EvidenceLevel.REGISTRY_VERIFIED)
        )
        assert analysis.schema_info.row_count == 150


class TestParsedAnalysis:
    def _parsed(self, **kw):
        base = dict(
            mode=AnalysisMode.FILE_PARSED,
            dataset_identity=DatasetIdentity(name="a"),
            input_resolution=_resolution(input_type=InputType.UPLOADED_FILE, raw_file_available=True,
                                         schema_available=True, schema_evidence_level=EvidenceLevel.DATAPILOT_PARSED),
            ingestion=_ingestion(),
            schema=SchemaInfo(level=EvidenceLevel.DATAPILOT_PARSED, source="upload",
                              columns=[SchemaColumn(original_name="a", normalized_name="a", evidence=EvidenceLevel.DATAPILOT_PARSED)]),
            metadata=_meta(AnalysisMode.FILE_PARSED),
        )
        base.update(kw)
        return DatasetAnalysis(**base)

    def test_a_parsed_analysis_with_ingestion_is_valid(self):
        assert self._parsed().mode is AnalysisMode.FILE_PARSED

    def test_it_must_describe_its_ingestion(self):
        with pytest.raises(ValueError, match="must describe its ingestion"):
            self._parsed(ingestion=None)

    def test_its_schema_must_be_datapilot_parsed(self):
        with pytest.raises(ValueError, match="must be DATAPILOT_PARSED"):
            self._parsed(schema=SchemaInfo(level=EvidenceLevel.REGISTRY_VERIFIED, source="x"))


class TestSchemaInfo:
    def test_an_unknown_schema_cannot_list_columns(self):
        with pytest.raises(ValueError, match="UNKNOWN schema cannot list columns"):
            SchemaInfo(level=EvidenceLevel.UNKNOWN, columns=[
                SchemaColumn(original_name="x", normalized_name="x", evidence=EvidenceLevel.REGISTRY_VERIFIED)])

    def test_a_listed_column_cannot_have_unknown_evidence(self):
        with pytest.raises(ValueError, match="UNKNOWN evidence"):
            SchemaInfo(level=EvidenceLevel.REGISTRY_VERIFIED, source="x", columns=[
                SchemaColumn(original_name="x", normalized_name="x", evidence=EvidenceLevel.UNKNOWN)])

    def test_a_row_count_needs_evidence_and_evidence_needs_a_row_count(self):
        with pytest.raises(ValueError, match="needs an evidence level"):
            SchemaInfo(level=EvidenceLevel.REGISTRY_VERIFIED, source="x", row_count=5)
        with pytest.raises(ValueError, match="evidence must be UNKNOWN"):
            SchemaInfo(level=EvidenceLevel.REGISTRY_VERIFIED, source="x", row_count_evidence=EvidenceLevel.REGISTRY_VERIFIED)

    def test_a_synthetic_column_name_has_no_original(self):
        column = SchemaColumn(original_name=None, normalized_name="column_1", evidence=EvidenceLevel.DATAPILOT_PARSED)
        assert column.original_name is None


class TestInputResolution:
    def test_an_available_schema_needs_a_level(self):
        with pytest.raises(ValueError, match="UNKNOWN"):
            _resolution(schema_available=True, schema_evidence_level=EvidenceLevel.UNKNOWN)

    def test_no_schema_cannot_claim_a_verified_level(self):
        with pytest.raises(ValueError, match="no schema is available"):
            _resolution(schema_available=False, schema_evidence_level=EvidenceLevel.REGISTRY_VERIFIED)

    def test_a_raw_file_is_only_ever_an_uploaded_file(self):
        with pytest.raises(ValueError, match="only ever available for an uploaded file"):
            _resolution(raw_file_available=True)

    def test_defaults_are_conservative(self):
        r = _resolution()
        assert r.raw_file_available is False and r.schema_available is False
        assert r.resolution_status is ResolutionStatus.NOT_APPLICABLE


class TestColumnProfile:
    def test_statistics_are_always_datapilot_parsed(self):
        with pytest.raises(ValueError, match="only ever DATAPILOT_PARSED"):
            _profile(evidence_level=EvidenceLevel.REGISTRY_VERIFIED)

    def test_numeric_stats_need_a_numeric_column(self):
        stats = NumericStats(min=0, max=1, mean=0.5, median=0.5, std=0.1)
        with pytest.raises(ValueError, match="non-numeric"):
            _profile(inferred_type=ColumnType.TEXT, numeric=stats)

    def test_percentages_are_bounded(self):
        with pytest.raises(ValueError):
            _profile(missing_pct=120.0)
        with pytest.raises(ValueError):
            _profile(uniqueness_ratio=1.5)


class TestTasksAndTargets:
    def test_supervised_learning_needs_a_target(self):
        with pytest.raises(ValueError, match="needs a target"):
            TaskCandidate(task=MLTask.BINARY_CLASSIFICATION, state=CandidateState.POSSIBLE, confidence=0.5)

    def test_unsupervised_tasks_need_none(self):
        assert TaskCandidate(task=MLTask.CLUSTERING, state=CandidateState.POSSIBLE, confidence=0.5).target is None

    def test_an_evidence_item_cannot_be_blank(self):
        with pytest.raises(ValueError):
            EvidenceItem(statement="", level=EvidenceLevel.HEURISTIC)
