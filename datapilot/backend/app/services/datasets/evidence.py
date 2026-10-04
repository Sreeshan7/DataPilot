"""Per-field evidence provenance for a dataset candidate.

A candidate is not "verified" or "unverified" as a whole: a registry result can
be externally verified and still be missing the three fields the target factor
depends on, while a model suggestion can be rich and specific yet rest entirely
on generated text. Collapsing that into one flag hides exactly the distinction a
user needs, so this module records it per field.

The classification adds no new trust concept. It only partitions the fields a
candidate already carries:

* ``external`` - published by the registry the candidate came from;
* ``ai_inferred`` - produced by a generative model, or derived by DataPilot
  itself and therefore not published by anyone;
* ``missing`` - the source supplied nothing, so nothing can be claimed.

Absence stays absence: a field the payload never contained is reported as
missing, never guessed.
"""

from __future__ import annotations

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    EvidenceAssessment,
    FieldProvenance,
    SourceType,
)

logger = get_logger(__name__)

# The metadata the ranker reads. These are the fields a relevance claim actually
# rests on, so they are the ones whose provenance is worth reporting.
RELEVANCE_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "domain",
    "tags",
    "target_description",
    "feature_description",
    "task_types",
    "license",
    "updated_at",
    "download_count",
)

# Fields DataPilot derives rather than reads from the source. ``domain`` is
# always our own mapping of the candidate's tags (or of an AI candidate's text)
# onto a dictionary domain; no registry publishes it. Reporting it as external
# would claim verification we never performed.
DERIVED_FIELDS: frozenset[str] = frozenset({"domain"})

# Fields a source structurally cannot supply. Listed so the assessment can say
# "this registry does not publish that" instead of implying a gap in the record.
SOURCE_UNPUBLISHED_FIELDS: dict[str, frozenset[str]] = {
    "kaggle": frozenset({"target_description", "feature_description", "task_types"}),
}


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (str, list, tuple, set, dict)):
        return len(value) > 0
    return True


def assess_evidence(candidate: DatasetCandidate) -> EvidenceAssessment:
    """Classify each relevance-bearing field of one candidate by provenance."""
    external: list[str] = []
    inferred: list[str] = []
    missing: list[str] = []

    unpublishable = SOURCE_UNPUBLISHED_FIELDS.get(candidate.source, frozenset())
    from_registry = candidate.source_type is SourceType.VERIFIED_EXTERNAL

    for field in RELEVANCE_FIELDS:
        if not _is_present(getattr(candidate, field, None)):
            missing.append(field)
            continue
        if field in DERIVED_FIELDS or not from_registry:
            inferred.append(field)
        else:
            external.append(field)

    coverage = round(
        (len(external) + len(inferred)) / len(RELEVANCE_FIELDS), 4
    ) if RELEVANCE_FIELDS else 0.0

    if external:
        provenance = FieldProvenance.EXTERNAL
    elif inferred:
        provenance = FieldProvenance.AI_INFERRED
    else:
        provenance = FieldProvenance.MISSING

    notes: list[str] = []
    if "domain" in inferred:
        notes.append(
            "domain is derived by DataPilot from the candidate's own metadata, "
            "not published by the source"
        )
    if unpublishable & set(missing):
        listed = ", ".join(sorted(unpublishable & set(missing)))
        notes.append(f"{candidate.source} does not publish: {listed}")
    if missing:
        notes.append(
            f"{len(missing)} of {len(RELEVANCE_FIELDS)} relevance fields were not "
            "supplied by the source"
        )
    if not from_registry:
        notes.append(
            "existence not externally confirmed; every present field comes from "
            "a generative model"
        )

    logger.debug(
        "Assessed evidence for %s: %d external, %d inferred, %d missing",
        candidate.id,
        len(external),
        len(inferred),
        len(missing),
    )
    return EvidenceAssessment(
        provenance=provenance,
        verification_status=candidate.verification_status,
        external_fields=external,
        inferred_fields=inferred,
        missing_fields=missing,
        evidence_coverage=coverage,
        notes=notes,
    )