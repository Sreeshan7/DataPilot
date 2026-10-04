"""Source-specific payload -> :class:`DatasetCandidate` normalization.

These are pure functions with no network access so they can be unit tested
directly against captured payloads.

Rules that must not be broken:

* Never fabricate. A field the payload does not contain stays ``None``.
* Never trust a model's self-assessment. GenAI output is always
  ``ai_suggested`` / ``unverified`` regardless of what the model claims.
* Malformed individual records are skipped, not fatal: one bad row must not
  discard an otherwise good search result.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    LLMDiscoveredCandidate,
    MetadataDepth,
    SourceType,
    VerificationStatus,
)
from app.services.datasets.evidence import assess_evidence

logger = get_logger(__name__)

# Kaggle returns ISO-8601 UTC timestamps such as "2025-04-12T10:49:08.663Z".
_ISO_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?"
    r"(Z|[+-]\d{2}:?\d{2})?$"
)
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Kaggle topic tags that unambiguously map onto a DataPilot dictionary domain.
# Deliberately conservative: an unmapped tag yields domain=None rather than a
# guess. Extend as new topics are observed.
KAGGLE_TAG_DOMAIN_ALIASES: dict[str, str] = {
    "education": "education",
    "school": "education",
    "university": "education",
    "student": "education",
    "students": "education",
    "academic": "education",
    "health": "healthcare",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "medicine": "healthcare",
    "disease": "healthcare",
    "heart": "healthcare",
    "diabetes": "healthcare",
    "cancer": "healthcare",
    "heart disease": "healthcare",
    "mental health": "healthcare",
    "finance": "finance",
    "financial": "finance",
    "banking": "finance",
    "credit": "finance",
    "fraud": "finance",
    "insurance": "finance",
    "stock": "finance",
    "trading": "finance",
    "loan": "finance",
    "cybersecurity": "cybersecurity",
    "security": "cybersecurity",
    "network": "cybersecurity",
    "malware": "cybersecurity",
    "intrusion detection": "cybersecurity",
    "phishing": "cybersecurity",
    "energy": "energy",
    "electricity": "energy",
    "power": "energy",
    "solar": "energy",
    "wind": "energy",
    "iot": "iot",
    "internet of things": "iot",
    "agriculture": "agriculture",
    "agricultural": "agriculture",
    "farming": "agriculture",
    "environment": "environment",
    "climate": "environment",
    "weather": "environment",
    "retail": "retail",
    "ecommerce": "retail",
    "shopping": "retail",
    "sports": "sports",
    "football": "sports",
    "nba": "sports",
    "real estate": "real_estate",
    "housing": "real_estate",
    "telecom": "telecom",
    "telecommunications": "telecom",
    "manufacturing": "manufacturing",
    "transportation": "transportation",
    "vehicle": "transportation",
    "traffic": "transportation",
    "geospatial": "geospatial_analytics",
    "satellite": "geospatial_analytics",
    "robotics": "robotics",
    "human resources": "human_resources",
    "recruitment": "human_resources",
    "social media": "social_media",
    "sentiment": "social_media",
    "entertainment": "entertainment",
    "movies": "entertainment",
    "public safety": "public_safety",
    "crime": "public_safety",
}


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> Optional[str]:
    """Return a clean non-empty string, or None for anything unusable."""
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return " ".join(text.split())


def coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_ratio(value: Any) -> Optional[float]:
    """Clamp a publisher score into 0.0-1.0, rejecting out-of-range junk."""
    number = coerce_float(value)
    if number is None:
        return None
    if number < 0.0 or number > 1.0:
        return None
    return number


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 or YYYY-MM-DD timestamp into an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = normalize_text(value)
    if not text:
        return None

    match = _ISO_RE.match(text)
    if match:
        year, month, day, hour, minute, second, _, offset = match.groups()
        micro = 0
        if match.group(7):
            micro = int((match.group(7) + "000000")[:6])
        # The wall-clock values belong to `offset`, so label the datetime with
        # that offset before converting to UTC. Labelling it UTC first would
        # silently discard the offset.
        source_tz = timezone.utc
        if offset and offset != "Z":
            sign = 1 if offset[0] == "+" else -1
            digits = offset[1:].replace(":", "")
            delta_hours = int(digits[:2])
            delta_minutes = int(digits[2:4]) if len(digits) >= 4 else 0
            source_tz = timezone(
                sign * timedelta(hours=delta_hours, minutes=delta_minutes)
            )
        try:
            stamp = datetime(
                int(year), int(month), int(day),
                int(hour), int(minute), int(second), micro,
                tzinfo=source_tz,
            )
        except ValueError:
            return None
        return stamp.astimezone(timezone.utc)

    match = _DATE_RE.match(text)
    if match:
        year, month, day = match.groups()
        try:
            return datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def normalize_url(value: Any) -> Optional[str]:
    """Accept only well-formed absolute http(s) URLs; reject anything else."""
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.lower()
    if not (lowered.startswith("http://") or lowered.startswith("https://")):
        return None
    if " " in text:
        return None
    return text


def slugify(value: str) -> str:
    return _SLUG_RE.sub("-", value.lower()).strip("-")[:80]


def extract_tags(raw_tags: Any) -> list[str]:
    """Kaggle returns tags as [{name, ref, fullPath, description}, ...]."""
    tags: list[str] = []
    if not isinstance(raw_tags, list):
        return tags
    for entry in raw_tags[:50]:
        if isinstance(entry, str):
            label = normalize_text(entry)
        elif isinstance(entry, dict):
            label = (
                normalize_text(entry.get("name"))
                or normalize_text(entry.get("ref"))
                or normalize_text(entry.get("fullPath"))
            )
        else:
            label = None
        if label and label not in tags:
            tags.append(label)
    return tags


def infer_domain_from_tags(
    tags: list[str], known_domains: Optional[set[str]] = None
) -> Optional[str]:
    """Map external topic tags onto a DataPilot dictionary domain, or None."""
    if known_domains is None:
        known_domains = _default_domain_keys()
    if not known_domains:
        return None

    normalized_known = {_normalize_key(d) for d in known_domains}
    for tag in tags:
        key = _normalize_key(tag)
        if not key:
            continue
        if key in normalized_known:
            original = next(d for d in known_domains if _normalize_key(d) == key)
            return original
        alias = KAGGLE_TAG_DOMAIN_ALIASES.get(key)
        if alias and _normalize_key(alias) in normalized_known:
            return alias
    return None


def infer_domain_from_text(
    text: Optional[str], known_domains: Optional[set[str]] = None
) -> Optional[str]:
    """Infer a domain from candidate-published free text, or None.

    Used for AI-suggested candidates, which carry no topic tags. Only the
    candidate's own metadata is inspected (never the model's rationale or a
    provenance hint), and only via the same conservative alias table used for
    Kaggle tags. Single significant words and adjacent two-word windows are
    tested; anything unmapped yields None rather than a guess, so a candidate
    is never assigned a domain without real evidence.
    """
    normalized = _normalize_key(text or "")
    if not normalized:
        return None
    words = normalized.split()
    candidates: list[str] = list(words)
    candidates.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))
    return infer_domain_from_tags(candidates, known_domains)


def _normalize_key(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").split())


_DOMAIN_KEY_CACHE: Optional[set[str]] = None


def _default_domain_keys() -> set[str]:
    global _DOMAIN_KEY_CACHE
    if _DOMAIN_KEY_CACHE is None:
        try:
            from app.services.keyword_service import KeywordService

            _DOMAIN_KEY_CACHE = set(KeywordService().get_domain_keys())
        except Exception as exc:  # pragma: no cover - dictionary is bundled
            logger.warning("Could not load dictionary domains for normalization: %s", exc)
            _DOMAIN_KEY_CACHE = set()
    return _DOMAIN_KEY_CACHE


def reset_domain_cache() -> None:
    """Test hook: clears the memoized dictionary domain keys."""
    global _DOMAIN_KEY_CACHE
    _DOMAIN_KEY_CACHE = None


def _with_evidence(candidate: DatasetCandidate) -> DatasetCandidate:
    """Attach the per-field provenance assessment.

    Done here, at the single point where a candidate's provenance is decided, so
    a candidate cannot reach the API layer without it. The assessment describes
    metadata only; it never changes ``source_type`` or ``verification_status``.
    """
    candidate.evidence = assess_evidence(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Kaggle
# ---------------------------------------------------------------------------

# Kaggle answers with a "<field>" and a "<field>Nullable" variant of every
# field. The nullable variant holds the real value; the plain one holds False.
def _kaggle_field(raw: dict, field: str) -> Any:
    value = raw.get(f"{field}Nullable")
    if value is None:
        value = raw.get(field)
    return value


def normalize_kaggle_record(
    raw: Any, known_domains: Optional[set[str]] = None
) -> Optional[DatasetCandidate]:
    """Convert one Kaggle ``datasets/list`` row into a DatasetCandidate.

    Returns None when the row lacks both a title and a ``ref``, which means it
    cannot be identified at all.
    """
    if not isinstance(raw, dict):
        logger.debug("Skipping non-dict Kaggle record: %s", type(raw).__name__)
        return None

    ref = normalize_text(_kaggle_field(raw, "ref"))
    title = normalize_text(_kaggle_field(raw, "title"))
    if not ref and not title:
        logger.debug("Skipping Kaggle record with no title and no ref")
        return None

    source_id = ref or f"{normalize_text(_kaggle_field(raw, 'ownerRef')) or 'unknown'}/{slugify(title or 'dataset')}"
    name = title or source_id.split("/")[-1]

    subtitle = normalize_text(_kaggle_field(raw, "subtitle"))
    description = normalize_text(_kaggle_field(raw, "description"))
    if not description:
        # Kaggle frequently ships an empty description but a useful subtitle.
        description = subtitle

    tags = extract_tags(_kaggle_field(raw, "tags"))
    owner = normalize_text(_kaggle_field(raw, "ownerRef")) or normalize_text(
        _kaggle_field(raw, "ownerName")
    )

    return _with_evidence(
        DatasetCandidate(
            id=f"kaggle:{source_id}",
            name=name,
            source="kaggle",
            source_type=SourceType.VERIFIED_EXTERNAL,
            source_id=source_id,
            verification_status=VerificationStatus.VERIFIED,
            url=normalize_url(_kaggle_field(raw, "url")),
            description=description,
            owner=owner,
            tags=tags,
            domain=infer_domain_from_tags(tags, known_domains),
            license=normalize_text(_kaggle_field(raw, "licenseName")),
            updated_at=parse_timestamp(_kaggle_field(raw, "lastUpdated")),
            # Kaggle's list endpoint does not expose a creation timestamp, and the
            # file list is empty there, so these stay None instead of being guessed.
            created_at=None,
            size_bytes=coerce_int(_kaggle_field(raw, "totalBytes")),
            file_count=None,
            file_types=[],
            download_count=coerce_int(_kaggle_field(raw, "downloadCount")),
            vote_count=coerce_int(_kaggle_field(raw, "voteCount")),
            usability=coerce_ratio(_kaggle_field(raw, "usabilityRating")),
        )
    )


def normalize_kaggle_records(
    raw_items: Any, known_domains: Optional[set[str]] = None
) -> list[DatasetCandidate]:
    """Normalize a Kaggle list response, skipping unusable rows."""
    if not isinstance(raw_items, list):
        logger.warning(
            "Kaggle returned %s instead of a list; returning no candidates",
            type(raw_items).__name__,
        )
        return []

    candidates: list[DatasetCandidate] = []
    for item in raw_items:
        candidate = normalize_kaggle_record(item, known_domains)
        if candidate is not None:
            candidates.append(candidate)
    logger.debug("Normalized %d/%d Kaggle records", len(candidates), len(raw_items))
    return candidates


#: Upper bound on an owner-written description kept on a candidate. Long enough
#: for a full column table, short enough that one dataset cannot dominate a
#: response or the ranker's text matching.
MAX_DESCRIPTION_CHARS = 8000

#: Published file names retained per candidate; the count is recorded separately.
MAX_FILE_NAMES = 20


def _file_extension(name: str) -> Optional[str]:
    if "." not in name:
        return None
    ext = "." + name.rsplit(".", 1)[-1].strip().lower()
    return ext if 1 < len(ext) <= 12 else None


def apply_kaggle_detail(
    candidate: DatasetCandidate, view: Any, files: Any
) -> DatasetCandidate:
    """Return a copy of ``candidate`` enriched with Kaggle's per-dataset record.

    ``view`` is the ``/datasets/view/{ref}`` payload and ``files`` the
    ``/datasets/list/{ref}`` payload. Either may be missing or malformed: each is
    optional, and a candidate with neither is returned unchanged at
    ``LISTING`` depth.

    Only what Kaggle states is added. The owner's description is the same kind of
    statement the listing's subtitle was, so it carries the same authority; file
    extensions are read straight off the published file names. Nothing here is
    inferred, and nothing about the *contents* of a file is claimed.
    """
    updates: dict[str, Any] = {}

    if isinstance(view, dict):
        subtitle = normalize_text(_kaggle_field(view, "subtitle"))
        body = normalize_text(_kaggle_field(view, "description"))
        parts = [p for p in (subtitle, body) if p]
        if len(parts) == 2 and parts[0].lower() in parts[1].lower():
            parts = [parts[1]]
        text = ". ".join(parts) if len(parts) == 2 else (parts[0] if parts else None)
        if text:
            updates["description"] = text[:MAX_DESCRIPTION_CHARS]
        if candidate.size_bytes is None:
            updates["size_bytes"] = coerce_int(_kaggle_field(view, "totalBytes"))
        if candidate.license is None:
            updates["license"] = normalize_text(_kaggle_field(view, "licenseName"))

    entries = files.get("datasetFiles") if isinstance(files, dict) else None
    if isinstance(entries, list):
        names = [
            n
            for n in (
                normalize_text(e.get("name")) for e in entries if isinstance(e, dict)
            )
            if n
        ]
        if names:
            extensions = sorted({e for e in (_file_extension(n) for n in names) if e})
            updates["file_count"] = len(names)
            updates["file_names"] = names[:MAX_FILE_NAMES]
            updates["file_types"] = extensions

    updates = {k: v for k, v in updates.items() if v is not None}
    if not updates:
        return candidate

    updates["metadata_depth"] = MetadataDepth.DETAILED
    enriched = candidate.model_copy(update=updates, deep=True)
    return _with_evidence(enriched)


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

#: Hub tag namespaces that describe packaging or provenance, not the subject.
_HF_NON_TOPIC_PREFIXES = (
    "annotations_creators:",
    "arxiv:",
    "doi:",
    "format:",
    "language:",
    "language_creators:",
    "language_details:",
    "library:",
    "license:",
    "modality:",
    "multilinguality:",
    "region:",
    "size_categories:",
    "source_datasets:",
    "task_categories:",
    "task_ids:",
)

#: Longest column list kept; the registry's own list may be far longer.
MAX_COLUMNS = 200


def _hf_tag_values(tags: list[str], prefix: str) -> list[str]:
    return [t[len(prefix):] for t in tags if t.startswith(prefix) and len(t) > len(prefix)]


def _hf_words(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split())


def _hf_as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _hf_columns(card: dict) -> list[str]:
    """Column names from ``cardData.dataset_info.features``, in published order.

    ``dataset_info`` is written by the Hub's own tooling from the data files, so a
    column list found there is a structured schema, not prose. A dataset whose card
    has none yields an empty list, never a guess.
    """
    info = card.get("dataset_info")
    infos = info if isinstance(info, list) else [info]
    names: list[str] = []
    for entry in infos:
        if not isinstance(entry, dict):
            continue
        for feature in entry.get("features") or []:
            if isinstance(feature, dict):
                name = normalize_text(feature.get("name"))
                if name and name not in names:
                    names.append(name)
    return names[:MAX_COLUMNS]


def _hf_dataset_size(card: dict) -> Optional[int]:
    info = card.get("dataset_info")
    infos = info if isinstance(info, list) else [info]
    for entry in infos:
        if isinstance(entry, dict):
            size = coerce_int(entry.get("dataset_size"))
            if size:
                return size
    return None


def normalize_huggingface_record(
    raw: Any, known_domains: Optional[set[str]] = None
) -> Optional[DatasetCandidate]:
    """Convert one Hugging Face ``/api/datasets`` row into a DatasetCandidate.

    Everything here is read from the Hub's own record: task categories become
    ``task_types``, ``format:`` tags become file extensions, ``modality:`` tags
    become published modalities, and a structured column list (when the card has
    one) becomes ``columns`` and a ``columns: ...`` feature description that the
    ranker already treats as a declared schema. Private or disabled datasets are
    skipped. Nothing is inferred beyond the topic domain, which DataPilot derives.
    """
    if not isinstance(raw, dict):
        return None
    if raw.get("private") or raw.get("disabled"):
        return None
    dataset_id = normalize_text(raw.get("id"))
    if not dataset_id:
        return None

    card = raw.get("cardData") if isinstance(raw.get("cardData"), dict) else {}
    tags = [t for t in raw.get("tags", []) if isinstance(t, str)]

    pretty = normalize_text(card.get("pretty_name"))
    name = pretty or _hf_words(dataset_id.split("/")[-1]) or dataset_id

    description = normalize_text(raw.get("description"))
    if description:
        description = description[:MAX_DESCRIPTION_CHARS]

    topical = [
        _hf_words(t)
        for t in [*tags, *_hf_as_list(card.get("tags"))]
        if t and not t.startswith(_HF_NON_TOPIC_PREFIXES)
    ]
    topical = list(dict.fromkeys(t for t in topical if t))[:20]

    task_types = list(
        dict.fromkeys(
            _hf_words(t)
            for t in (
                _hf_tag_values(tags, "task_categories:")
                + _hf_tag_values(tags, "task_ids:")
                + _hf_as_list(card.get("task_categories"))
            )
            if t
        )
    )

    file_types = []
    for fmt in _hf_tag_values(tags, "format:"):
        ext = "." + fmt.strip().lower()
        if len(ext) > 1 and ext not in file_types:
            file_types.append(ext)

    modalities = list(
        dict.fromkeys(_hf_words(m).lower() for m in _hf_tag_values(tags, "modality:") if m)
    )

    license_text = normalize_text((_hf_tag_values(tags, "license:") or [None])[0]) or normalize_text(
        (_hf_as_list(card.get("license")) or [None])[0]
    )

    columns = _hf_columns(card)
    feature_description = f"columns: {', '.join(columns)}" if columns else None

    domain = infer_domain_from_tags(topical, known_domains) or infer_domain_from_text(
        " ".join(p for p in (name, description or "") if p), known_domains
    )

    return _with_evidence(
        DatasetCandidate(
            id=f"huggingface:{dataset_id}",
            name=name,
            source="huggingface",
            source_type=SourceType.VERIFIED_EXTERNAL,
            source_id=dataset_id,
            verification_status=VerificationStatus.VERIFIED,
            url=f"https://huggingface.co/datasets/{dataset_id}",
            description=description,
            owner=normalize_text(raw.get("author")) or (
                dataset_id.split("/")[0] if "/" in dataset_id else None
            ),
            tags=topical,
            domain=domain,
            task_types=task_types,
            feature_description=feature_description,
            license=license_text,
            updated_at=parse_timestamp(raw.get("lastModified")),
            created_at=parse_timestamp(raw.get("createdAt")),
            size_bytes=_hf_dataset_size(card),
            file_types=file_types,
            modalities=modalities,
            columns=columns,
            download_count=coerce_int(raw.get("downloads")),
            vote_count=coerce_int(raw.get("likes")),
            usability=None,
            # The search response already carries the dataset card, so there is no
            # deeper per-dataset record to fetch.
            metadata_depth=MetadataDepth.DETAILED,
        )
    )


def normalize_huggingface_records(
    raw_items: Any, known_domains: Optional[set[str]] = None
) -> list[DatasetCandidate]:
    """Normalize a Hugging Face search response, skipping unusable rows."""
    if not isinstance(raw_items, list):
        logger.warning(
            "Hugging Face returned %s instead of a list; returning no candidates",
            type(raw_items).__name__,
        )
        return []
    out = [c for c in (normalize_huggingface_record(i, known_domains) for i in raw_items) if c]
    logger.debug("Normalized %d/%d Hugging Face records", len(out), len(raw_items))
    return out


# ---------------------------------------------------------------------------
# GenAI
# ---------------------------------------------------------------------------


def normalize_genai_candidate(
    raw: Any, index: int = 0
) -> Optional[DatasetCandidate]:
    """Convert one Gemini suggestion into an unverified DatasetCandidate.

    The model cannot verify that a dataset exists, so ``source_type`` and
    ``verification_status`` are forced here and are not read from the payload.
    """
    if isinstance(raw, LLMDiscoveredCandidate):
        item = raw
    elif isinstance(raw, dict):
        try:
            item = LLMDiscoveredCandidate(**raw)
        except Exception as exc:
            logger.warning("Discarding invalid GenAI candidate %d: %s", index, exc)
            return None
    else:
        logger.warning("Discarding GenAI candidate %d of type %s", index, type(raw).__name__)
        return None

    name = normalize_text(item.name)
    if not name:
        logger.warning("Discarding GenAI candidate %d with no name", index)
        return None

    suggested_features = [f for f in (normalize_text(s) for s in item.suggested_features) if f]

    description = normalize_text(item.description)
    target_description = normalize_text(item.target_description)
    # The domain is inferred only from text the model actually published about
    # the dataset. The model's rationale and provenance hint are excluded: they
    # describe the project, not the dataset, and must not become evidence.
    domain = infer_domain_from_text(
        " ".join(
            part
            for part in (name, description, " ".join(suggested_features), target_description)
            if part
        )
    )

    return _with_evidence(
        DatasetCandidate(
        id=f"genai:{slugify(name) or 'candidate'}",
            name=name,
            source="genai",
            source_type=SourceType.AI_SUGGESTED,
            source_id=slugify(name) or f"candidate-{index}",
            # Forced: a language model cannot externally verify a dataset.
            verification_status=VerificationStatus.UNVERIFIED,
            url=normalize_url(item.url),
            description=description,
            owner=None,
            tags=[],
            domain=domain,
            task_types=[t for t in (normalize_text(x) for x in item.ml_tasks) if t],
            target_description=target_description,
            feature_description="; ".join(suggested_features) if suggested_features else None,
            rationale=normalize_text(item.why_relevant),
            license=normalize_text(item.license),
            confidence_score=coerce_ratio(item.confidence),
            # Model provenance only. Recorded so a user can tell a well-known
            # dataset from a speculative one, never to imply verification.
            possible_source=normalize_text(item.possible_source),
            is_well_known=item.is_well_known,
        )
    )


def normalize_genai_candidates(raw_items: Any) -> list[DatasetCandidate]:
    if not isinstance(raw_items, list):
        logger.warning(
            "GenAI returned %s instead of a list; returning no candidates",
            type(raw_items).__name__,
        )
        return []

    candidates: list[DatasetCandidate] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw_items):
        candidate = normalize_genai_candidate(item, index)
        if candidate is None:
            continue
        if candidate.id in seen_ids:
            continue
        seen_ids.add(candidate.id)
        candidates.append(candidate)
    return candidates
