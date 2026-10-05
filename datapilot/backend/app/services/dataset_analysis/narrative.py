"""Plain-language statements derived from an analysis, never from a model.

The most important sentences in the product live here: the ones that say whether DataPilot
analysed the user's actual file or only read public metadata. They are generated from the
analysis's own evidence fields, so they cannot say more than the data does.
"""

from __future__ import annotations

from app.schemas.analysis_response import NextAction
from app.schemas.dataset_analysis import (
    AnalysisMode,
    DatasetAnalysis,
    EvidenceLevel,
    FindingSeverity,
    InputType,
    ResolutionStatus,
)

ANALYZE_ENDPOINT = "/api/v1/datasets/analyze"


def headline(analysis: DatasetAnalysis) -> str:
    if analysis.mode is AnalysisMode.FILE_PARSED:
        return "DataPilot analyzed your actual file."
    resolution = analysis.input_resolution
    if resolution.input_type is InputType.AMBIGUOUS_DATASET_REFERENCE or (
        resolution.resolution_status is ResolutionStatus.AMBIGUOUS
    ):
        return "That reference matches several datasets, so none was analyzed."
    if analysis.external is None:
        return (
            "DataPilot could not identify that dataset in any registry it researches, so nothing "
            "about it has been verified."
        )
    level = analysis.schema_info.level
    if level is EvidenceLevel.AI_INFERRED:
        return (
            "DataPilot found this dataset through a web search. Its columns were read by AI from the cited pages, "
            "so treat them as a lead to confirm; the raw file has not been parsed."
        )
    if level is EvidenceLevel.REGISTRY_VERIFIED and analysis.schema_info.source.startswith("file header"):
        return "DataPilot read the real column header from the dataset's file, but has not analysed the data itself."
    if level is EvidenceLevel.UNKNOWN:
        return (
            "The public source does not expose enough schema information to verify the dataset "
            "structure. Upload the file for full analysis."
        )
    if level is EvidenceLevel.INFERRED:
        return (
            "DataPilot found public metadata for this dataset, but its column list is only read "
            "from the description's text, and the raw file has not been parsed."
        )
    return "DataPilot found public metadata for this dataset, but has not parsed the raw file."


def evidence_statement(analysis: DatasetAnalysis) -> str:
    if analysis.mode is AnalysisMode.FILE_PARSED:
        ing, schema = analysis.ingestion, analysis.schema_info
        scope = f"{ing.rows_read} rows and {schema.column_count} columns"
        partial = " (the file was truncated, so these describe the rows read)" if ing and ing.truncated else ""
        return (
            f"The schema, statistics, quality findings and target candidates were computed from "
            f"your file: {scope}{partial}. Name-based judgments (identifiers, sensitive fields, likely "
            "targets) are labelled as heuristics."
        )
    meta = analysis.external
    if meta is None:
        return "No registry record was found, so there is no verified information about this dataset."
    schema = analysis.schema_info
    pieces = [f"Everything here comes from {meta.source}'s published metadata"]
    if schema.level is EvidenceLevel.AI_INFERRED:
        pieces.append("the column list was read by an AI model from web pages cited below, and is not verified")
    elif schema.level is EvidenceLevel.REGISTRY_VERIFIED and schema.source.startswith("file header"):
        pieces.append("the column names and types were read from the first few KB of the dataset's own file")
    elif schema.level is EvidenceLevel.REGISTRY_VERIFIED:
        pieces.append("the column list is the registry's own structured schema")
    elif schema.level is EvidenceLevel.INFERRED:
        pieces.append("the column list was inferred from the owner's description and may be incomplete")
    else:
        pieces.append("no column list could be verified")
    pieces.append(
        "no rows were read, so there are no statistics, data-quality checks or feature importance"
    )
    return "; ".join(pieces) + "."


def next_step(analysis: DatasetAnalysis) -> str:
    resolution = analysis.input_resolution
    if analysis.mode is AnalysisMode.FILE_PARSED:
        quality = analysis.quality
        critical = [f for f in (quality.findings if quality else []) if f.severity is FindingSeverity.CRITICAL]
        if critical:
            return (
                f"Review the {len(critical)} critical finding(s) first (for example possible leakage or "
                "a near-empty column) before modelling."
            )
        if analysis.target_candidates:
            best = analysis.target_candidates[0]
            return (
                f"Confirm the target (leading candidate: '{best.column}'), then pick one of the supported "
                "tasks. You can name the target yourself with the 'target' field."
            )
        return "No column looks like a target; consider clustering or anomaly detection, or name the target yourself."
    if resolution.needs_confirmation and resolution.clarification:
        return resolution.clarification
    if analysis.external is None:
        return "Check the dataset's name or link, or upload the file."
    if analysis.schema_info.level is EvidenceLevel.UNKNOWN:
        return "Upload the dataset file: the public source does not publish its columns, so nothing else can be verified."
    if analysis.schema_info.level is EvidenceLevel.AI_INFERRED:
        return "Check the cited sources to confirm this is the dataset you mean, then upload the file for verified analysis."
    return (
        "Upload the actual file for row-level statistics, data-quality checks and real feature "
        "importance; until then, treat everything here as the registry's description of the dataset."
    )


def next_actions(analysis: DatasetAnalysis) -> list[NextAction]:
    actions: list[NextAction] = []
    if analysis.mode is AnalysisMode.EXTERNAL_METADATA:
        actions.append(NextAction(
            action="UPLOAD_FILE",
            description="Upload the dataset file for row-level analysis.",
            endpoint=ANALYZE_ENDPOINT,
            payload={"file": "<the dataset file>"},
        ))
    resolution = analysis.input_resolution
    if resolution.alternatives:
        for alt in resolution.alternatives[:3]:
            if alt.url:
                actions.append(NextAction(
                    action="ANALYZE_ALTERNATIVE",
                    description=f"Analyze {alt.name} ({alt.source}) instead.",
                    endpoint=ANALYZE_ENDPOINT,
                    payload={"dataset_url": alt.url},
                ))
    return actions
