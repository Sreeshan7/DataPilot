"""Deciding what the user actually pointed at: a file, a dataset name, a link, or nothing clear.

Offline and deterministic. It classifies the input and extracts identifiers; it does not
contact any registry (that is the external research service's job) and it never marks a
schema as available - only parsing a file or researching a registry can do that.

Precedence when several things are given: an uploaded file wins, because it is the only input
DataPilot can analyse directly. A name or link given alongside a file is kept as identity
context and said so, but the registry's copy is not analysed in place of the user's file.
"""

from __future__ import annotations

import re
from typing import Optional

from app.schemas.dataset_analysis import DatasetInputResolution, InputType, ResolutionStatus
from app.services.dataset_analysis.references import (
    UnsafeURLError,
    extract_dataset_name,
    extract_urls,
    normalize_dataset_name,
    parse_dataset_url,
)
from app.services.dataset_analysis.safety import sanitize_display_name

_SOURCE_WORDS = {
    "kaggle": "kaggle",
    "hugging face": "huggingface",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "uci": "uci",
    "openml": "openml",
    "open ml": "openml",
}
SOURCES = frozenset({"kaggle", "huggingface", "uci", "openml"})


def normalize_source(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return _SOURCE_WORDS.get(" ".join(value.lower().split()))


def _source_in_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    lowered = text.lower()
    for word, source in sorted(_SOURCE_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return source
    return None


class DatasetInputResolver:
    """Classifies the dataset reference in a request."""

    def resolve(
        self,
        *,
        has_file: bool = False,
        file_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_url: Optional[str] = None,
        message: Optional[str] = None,
        source_hint: Optional[str] = None,
    ) -> DatasetInputResolution:
        hint = normalize_source(source_hint) or _source_in_text(message) or None

        if has_file:
            return self._file(file_name, dataset_name, dataset_url)

        url = dataset_url or next(iter(extract_urls(message)), None)
        if url:
            return self._url(url, dataset_name, hint)

        if dataset_name:
            return self._name(dataset_name, hint, explicit=True)

        extracted = extract_dataset_name(message)
        if extracted is not None:
            return self._name(extracted.name, hint, explicit=False, extraction=extracted)

        return DatasetInputResolution(
            input_type=InputType.NO_DATASET_REFERENCE,
            resolution_status=ResolutionStatus.UNRESOLVED,
            evidence=["no file, dataset name or dataset link was supplied"],
            needs_confirmation=True,
            clarification=(
                "Upload the file, or give the dataset's name or its Kaggle / Hugging Face / UCI / "
                "OpenML link."
            ),
        )

    # ------------------------------------------------------------------

    def _file(
        self, file_name: Optional[str], dataset_name: Optional[str], dataset_url: Optional[str]
    ) -> DatasetInputResolution:
        display = sanitize_display_name(file_name)
        stem = display.rsplit(".", 1)[0] if "." in display else display
        evidence = ["a file was uploaded and will be analysed directly"]
        if dataset_name or dataset_url:
            evidence.append(
                "a dataset name or link was also given; the uploaded file was analysed, not the "
                "registry's copy"
            )
        return DatasetInputResolution(
            input_type=InputType.UPLOADED_FILE,
            normalized_dataset_name=(dataset_name or stem).strip() or None,
            source="upload",
            resolution_status=ResolutionStatus.NOT_APPLICABLE,
            resolution_confidence=1.0,
            evidence=evidence,
            raw_file_available=True,
        )

    def _url(self, url: str, dataset_name: Optional[str], hint: Optional[str]) -> DatasetInputResolution:
        try:
            parsed = parse_dataset_url(url)
        except UnsafeURLError as exc:
            return DatasetInputResolution(
                input_type=InputType.DATASET_URL,
                resolution_status=ResolutionStatus.UNSUPPORTED,
                evidence=[f"the link was refused: {exc}"],
                needs_confirmation=True,
                clarification=(
                    "Link a dataset page on Kaggle, Hugging Face, UCI or OpenML, or upload the file."
                ),
            )
        evidence = [f"a {parsed.source} dataset link identifies {parsed.identifier}"]
        if dataset_name:
            evidence.append(f"the name {dataset_name!r} was also given; the link takes precedence")
        return DatasetInputResolution(
            input_type=InputType.DATASET_URL,
            normalized_dataset_name=normalize_dataset_name(parsed.name_hint or parsed.identifier) or None,
            source=parsed.source,
            source_url=parsed.canonical_url,
            source_identifier=parsed.identifier,
            resolution_status=ResolutionStatus.RESOLVED,
            resolution_confidence=0.97,
            evidence=evidence,
        )

    def _name(
        self,
        name: str,
        hint: Optional[str],
        *,
        explicit: bool,
        extraction=None,
    ) -> DatasetInputResolution:
        cleaned = " ".join(name.split())
        check = extraction or extract_dataset_name(f"{cleaned} dataset")
        generic = bool(check is not None and check.generic)
        normalized = normalize_dataset_name(cleaned)
        evidence = [
            "the dataset name was supplied explicitly" if explicit
            else (extraction.evidence if extraction is not None else "the name was read from the message")
        ]
        if hint:
            evidence.append(f"the source {hint} was named, so only it will be searched")

        if not normalized or generic:
            return DatasetInputResolution(
                input_type=InputType.AMBIGUOUS_DATASET_REFERENCE,
                normalized_dataset_name=normalized or None,
                source=hint,
                resolution_status=ResolutionStatus.AMBIGUOUS,
                resolution_confidence=0.3,
                evidence=evidence + ["the name is a general topic word, too vague to identify one dataset"],
                needs_confirmation=True,
                clarification=(
                    f"'{cleaned}' could be thousands of datasets. Upload the file, give its link, or "
                    "use a more specific name."
                ),
            )
        confidence = 0.85 if explicit else min(0.9, extraction.confidence if extraction else 0.7)
        return DatasetInputResolution(
            input_type=InputType.PUBLIC_DATASET_NAME,
            normalized_dataset_name=normalized,
            source=hint,
            resolution_status=ResolutionStatus.UNRESOLVED,  # until a registry confirms it
            resolution_confidence=confidence,
            evidence=evidence,
        )
