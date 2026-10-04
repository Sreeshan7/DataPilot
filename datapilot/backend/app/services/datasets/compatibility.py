"""Contradiction detection between project requirements and candidate metadata.

The ranker's other factors all answer "how much of what I asked for can I see?".
This module answers the opposite question: "does this dataset positively say it is
something else?". Those are different situations and must not be confused:

    no evidence            -> UNKNOWN, no penalty (see evidence.py)
    evidence of a different subject -> MISMATCH, penalty

Only *declared* metadata is treated as positive evidence, never silence. The
signals below are all things a registry publishes on purpose - task types, the
topic of the record, the file types of the dataset - so a contradiction means
somebody stated it. That is why an AI suggestion's free-text description is not
used here: a model asserting "image captioning" is not a fact about the dataset.

Every rule is skipped when either side is empty. Silence on either side yields no
contradiction, by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from app.core.logging_config import get_logger
from app.schemas.dataset import DatasetCandidate, VerificationStatus
from app.services.datasets.target_concepts import TargetConceptIndex, value_tokens

logger = get_logger(__name__)

_PUNCT = re.compile(r"[^a-z0-9]+")

# Penalties are shares of the compatibility factor, so they express "how much
# evidence of incompatibility did we find" on one scale rather than being
# arbitrary offsets. They are deliberately smaller for weak signals: a candidate
# that merely sits in another domain is usually still usable, while one that
# declares only a different ML task or only a different label space is not.
TASK_PENALTY = 0.6
TARGET_PENALTY = 0.5
MODALITY_PENALTY = 0.4
DOMAIN_PENALTY = 0.3

# File extensions grouped into modality. A project that needs tabular data and a
# dataset published only as images cannot satisfy it, and both halves of that
# claim are published facts.
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"})
_TEXT_EXTENSIONS = frozenset({".txt", ".pdf", ".doc", ".docx", ".rtf", ".epub"})
_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".flac", ".m4a", ".ogg"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
_TABULAR_EXTENSIONS = frozenset({".csv", ".tsv", ".parquet", ".xls", ".xlsx", ".json"})

_MODALITY_BY_EXTENSION = {
    **{ext: "image" for ext in _IMAGE_EXTENSIONS},
    **{ext: "text" for ext in _TEXT_EXTENSIONS},
    **{ext: "audio" for ext in _AUDIO_EXTENSIONS},
    **{ext: "video" for ext in _VIDEO_EXTENSIONS},
    **{ext: "tabular" for ext in _TABULAR_EXTENSIONS},
}

# Words that name a modality in a project's data-type requirement. Kept as plain
# modality vocabulary rather than concept ids: this is a statement about file
# formats, not about the ML knowledge base.
_MODALITY_WORDS: dict[str, frozenset[str]] = {
    "image": frozenset({"image", "images", "image data", "pixel", "pixels", "radiology"}),
    "text": frozenset({"text", "textual", "document", "documents", "corpus"}),
    "audio": frozenset({"audio", "sound", "speech", "voice"}),
    "video": frozenset({"video", "videos"}),
    "tabular": frozenset({"tabular", "table", "tables", "structured", "spreadsheet", "rows"}),
}


@dataclass(frozen=True)
class Contradiction:
    """One positively-evidenced conflict between project and candidate."""

    dimension: str
    penalty: float
    message: str


def _tokens(text: Optional[str]) -> frozenset[str]:
    if not text:
        return frozenset()
    return frozenset(t for t in _PUNCT.sub(" ", text.lower()).split() if t)


def _task_overlap(project_tasks: Sequence[str], candidate_tasks: Sequence[str]) -> bool:
    """Whether any required task appears in the declared task list.

    Compared token-wise on purpose: Kaggle writes ``binary_classification`` where
    a project says ``classification``, and substring matching would keep missing
    that. Two tasks that share no word at all are genuinely different tasks
    ("classification" vs "forecasting"), which is the case the rule is for.
    """
    required: set[str] = set()
    for task in project_tasks:
        required |= _tokens(task)
    declared: set[str] = set()
    for task in candidate_tasks:
        declared |= _tokens(task)
    return bool(required & declared)


def _project_modalities(data_type_notes: Sequence[str]) -> frozenset[str]:
    out: set[str] = set()
    for note in data_type_notes:
        tokens = _tokens(note)
        for modality, words in _MODALITY_WORDS.items():
            if any(word in note.lower() for word in words) or (
                tokens & _tokens(" ".join(sorted(words)))
            ):
                out.add(modality)
    return frozenset(out)


def _candidate_modalities(candidate: DatasetCandidate) -> frozenset[str]:
    """Modalities the candidate's published metadata actually proves.

    A modality the registry states outright (Hugging Face's ``modality:image``)
    outranks one guessed from a file extension: image datasets are routinely stored
    as ``.parquet``, a container format that would otherwise read as "tabular".
    Extensions are only consulted when the registry labelled no modality.
    """
    stated = {
        label.strip().lower()
        for label in candidate.modalities
        if label.strip().lower() in _MODALITY_WORDS
    }
    if stated:
        return frozenset(stated)
    out: set[str] = set()
    for name in candidate.file_types:
        ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
        modality = _MODALITY_BY_EXTENSION.get(ext)
        if modality:
            out.add(modality)
    return frozenset(out)


def _target_families(index: TargetConceptIndex, target: Optional[str]) -> frozenset[str]:
    """Every label space key compatible with ``target``, or empty if unknown."""
    expectation = index.expectation(target)
    keys: set[str] = set()
    for space in expectation.spaces:
        for sibling in index.family(space):
            keys.add(sibling.key)
    return frozenset(keys)


def find_contradictions(
    candidate: DatasetCandidate,
    *,
    project_domain: Optional[str],
    project_tasks: Sequence[str],
    project_targets: Sequence[str],
    project_data_types: Sequence[str],
    candidate_domains: frozenset[str],
    target_index: TargetConceptIndex,
) -> list[Contradiction]:
    """Every conflict the candidate's published metadata positively supports.

    Returns nothing for a candidate whose metadata was inferred by a model. A
    model's guess that "final grade" is the target is not a registry statement,
    and treating it as one would let the system contradict a requirement using
    the same unverified authority it used to satisfy it. Those candidates are
    scored on their relevance and trust instead.

    Ordered by descending penalty so the strongest contradiction explains the
    score, and deterministic in every tie-break (all iteration is over sorted
    sequences).
    """
    found: list[Contradiction] = []

    if candidate.verification_status is not VerificationStatus.VERIFIED:
        logger.debug(
            "Skipping contradiction detection for unverified candidate %s: its "
            "metadata was inferred, not declared",
            candidate.id,
        )
        return found

    # 1. Declared ML task. Kaggle publishes ``taskTypes``, so a record listing only
    #    "forecasting" when the project needs "classification" is a stated fact.
    if candidate.task_types and project_tasks and not _task_overlap(
        project_tasks, candidate.task_types
    ):
        required = ", ".join(sorted({t for t in project_tasks if t}))
        declared = ", ".join(sorted({t for t in candidate.task_types if t}))
        found.append(
            Contradiction(
                dimension="ml_task",
                penalty=TASK_PENALTY,
                message=(
                    f"dataset supports only {declared} while the project requires {required}"
                ),
            )
        )

    # 2. Declared target. Only compared when both sides name a label space:
    #    silence on either side is not a disagreement.
    if candidate.target_description and project_targets:
        candidate_spaces = _target_families(target_index, candidate.target_description)
        if candidate_spaces:
            for target in project_targets:
                project_spaces = _target_families(target_index, target)
                if project_spaces and not (project_spaces & candidate_spaces):
                    found.append(
                        Contradiction(
                            dimension="target",
                            penalty=TARGET_PENALTY,
                            message=(
                                f"dataset target {candidate.target_description!r} declares "
                                f"different label values than the project's {target!r}"
                            ),
                        )
                    )
                    break

    # 3. Modality. Published file extensions against a stated data type.
    modalities = _candidate_modalities(candidate)
    if modalities:
        required_modalities = _project_modalities(project_data_types)
        if required_modalities and not (modalities & required_modalities):
            found.append(
                Contradiction(
                    dimension="modality",
                    penalty=MODALITY_PENALTY,
                    message=(
                        f"dataset publishes only {', '.join(sorted(modalities))} files while "
                        f"the project requires {', '.join(sorted(required_modalities))} data"
                    ),
                )
            )

    # 4. Subject matter. Only when the candidate resolves to exactly one domain
    #    and it is not the project's: a dataset that is demonstrably about one
    #    other thing is unlikely to serve this project, while a dataset that
    #    resolves to several is simply broad.
    if (
        project_domain
        and len(candidate_domains) == 1
        and project_domain not in candidate_domains
    ):
        found.append(
            Contradiction(
                dimension="domain",
                penalty=DOMAIN_PENALTY,
                message=(
                    f"dataset is entirely about {sorted(candidate_domains)[0]} while the "
                    f"project is about {project_domain}"
                ),
            )
        )

    found.sort(key=lambda c: (-c.penalty, c.dimension))
    logger.debug("Found %d contradictions for %s", len(found), candidate.id)
    return found


def penalty_total(contradictions: Sequence[Contradiction]) -> float:
    """Combined incompatibility evidence, capped at full incompatibility."""
    return min(1.0, sum(c.penalty for c in contradictions))