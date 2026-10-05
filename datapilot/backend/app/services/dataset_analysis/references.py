"""Reading dataset references out of user input: URLs and names.

Pure functions, no network. This is also the URL security boundary: anything that
will later be fetched passes through :func:`validate_public_url` first, which refuses
credentials in URLs, IP literals, non-default ports, non-HTTP(S) schemes and any host
that is not a registry DataPilot researches. Rejecting at the edge means no later code
has to remember to.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, unquote, urlsplit

MAX_URL_LENGTH = 2048

#: Registries whose APIs DataPilot researches, by host. Anything else is refused:
#: following arbitrary links from user input is how a server ends up fetching
#: internal addresses on someone else's behalf.
ALLOWED_HOSTS: dict[str, str] = {
    "kaggle.com": "kaggle",
    "www.kaggle.com": "kaggle",
    "huggingface.co": "huggingface",
    "datasets-server.huggingface.co": "huggingface",
    "www.huggingface.co": "huggingface",
    "archive.ics.uci.edu": "uci",
    "openml.org": "openml",
    "www.openml.org": "openml",
}

#: Path segments on Hugging Face that follow a dataset id; they are not part of it.
_HF_RESERVED = frozenset(
    {"viewer", "tree", "blob", "resolve", "discussions", "commits", "settings", "edit", "raw"}
)

#: Classic datasets people refer to by a bare name. Used only to raise confidence that a
#: phrase *is* a dataset name; it is a hint, not a registry, and nothing is resolved from it.
KNOWN_DATASET_NAMES: frozenset[str] = frozenset(
    {
        "titanic", "iris", "mnist", "fashion mnist", "cifar 10", "cifar 100", "imagenet",
        "boston housing", "california housing", "ames housing", "wine quality", "wine",
        "heart disease", "adult census", "breast cancer wisconsin", "pima indians diabetes",
        "palmer penguins", "penguins", "mall customers", "credit card fraud", "imdb",
        "bike sharing", "auto mpg", "house prices", "diabetes", "digits", "telco churn",
    }
)

#: The subset of KNOWN_DATASET_NAMES distinctive enough that merely *mentioning* one
#: (without the word "dataset") is evidence of a dataset reference. "predict diabetes" or
#: "house prices" name a topic, not a file, so those only count next to "dataset".
_DISTINCTIVE_NAMES: frozenset[str] = frozenset(
    {
        "titanic", "iris", "mnist", "fashion mnist", "cifar 10", "cifar 100", "imagenet",
        "boston housing", "california housing", "ames housing", "adult census",
        "breast cancer wisconsin", "pima indians diabetes", "palmer penguins", "imdb",
        "mall customers", "wine quality",
    }
)

_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?"

# Words that precede a dataset name without being part of it.
_LEADING_FILLER = frozenset(
    {
        "i", "ive", "i've", "have", "got", "has", "had", "the", "a", "an", "my", "this", "that",
        "our", "your", "their", "kaggle", "public", "open", "uci", "hugging", "face",
        "huggingface", "openml", "downloaded", "uploaded", "attached", "using", "use", "with",
        "on", "of", "analyze", "analyse", "profile", "inspect", "explore", "for", "about",
        "from", "in", "and", "to", "find", "need", "want", "me", "please", "can", "you", "what",
        "build", "do", "make", "if", "but", "new", "better", "other", "another", "different",
        "some", "any", "existing", "own", "famous", "classic", "popular", "well", "known",
        "is", "are", "was", "were", "it", "its", "also", "just", "only", "here", "there",
        "know", "knew", "found", "see", "check", "look", "at", "tell", "show", "give", "get",
        "recommend", "suggest", "search", "searching", "looking", "look", "locate", "discover",
        "source", "download", "give", "predict", "classify", "forecast", "detect", "train",
        "model", "project", "data",
        "dataset", "datasets", "csv", "excel", "file", "spreadsheet",
    }
)

#: Single words that name a topic, not a dataset. A name made only of these is too vague
#: to look up ("I downloaded a student dataset").
_GENERIC_TOPIC_WORDS = frozenset(
    {
        "student", "students", "customer", "customers", "sales", "health", "medical", "patient",
        "patients", "stock", "stocks", "weather", "sports", "financial", "finance", "retail",
        "education", "school", "survey", "employee", "employees", "product", "products",
        "transaction", "transactions", "user", "users", "company", "companies", "city",
        "population", "traffic", "energy", "housing", "house", "price", "prices", "text",
        "image", "images", "sensor", "network", "marketing", "research", "random", "test",
        "sample", "example", "demo", "big", "small", "large", "new", "old", "my", "your",
    }
)


class UnsafeURLError(ValueError):
    """The URL is malformed, hostile, or not one DataPilot researches."""


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def extract_urls(text: Optional[str]) -> list[str]:
    """URLs mentioned in free text, with trailing punctuation removed."""
    if not text:
        return []
    found = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(_TRAILING_PUNCT)
        if url and url not in found:
            found.append(url)
    return found


def validate_public_url(url: str) -> str:
    """Return a canonical https URL, or raise :class:`UnsafeURLError`.

    Refuses everything that is not an ordinary https URL to a named host on the
    allowlist. IP literals are refused outright, which closes the usual route to
    internal addresses (loopback, link-local metadata endpoints, private ranges).
    """
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("a URL is required")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURLError("the URL is too long")
    if any(ord(ch) < 32 or ch.isspace() for ch in url):
        raise UnsafeURLError("the URL contains whitespace or control characters")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError("the URL is malformed") from exc

    if parts.scheme.lower() not in {"http", "https"}:
        raise UnsafeURLError("only http(s) URLs are supported")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise UnsafeURLError("URLs containing credentials are refused")
    if port not in (None, 80, 443):
        raise UnsafeURLError("non-default ports are refused")

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise UnsafeURLError("the URL has no host")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise UnsafeURLError("IP-address hosts are refused")
    if host == "localhost" or "." not in host:
        raise UnsafeURLError("the host is not a public domain name")
    if host not in ALLOWED_HOSTS:
        raise UnsafeURLError(f"{host} is not a dataset registry DataPilot researches")

    # Canonical form: https, lowercase host, no port, no fragment, original path/query.
    query = f"?{parts.query}" if parts.query else ""
    return f"https://{host}{parts.path}{query}"


@dataclass(frozen=True)
class ParsedDatasetURL:
    source: str
    identifier: str
    name_hint: Optional[str]
    canonical_url: str


def _words(slug: str) -> str:
    return " ".join(re.sub(r"[-_+]+", " ", unquote(slug)).split())


def parse_dataset_url(url: str) -> ParsedDatasetURL:
    """Identify the registry and dataset a (valid) URL points at.

    Raises :class:`UnsafeURLError` for an invalid or unsupported URL, and for a
    supported host whose path is not a dataset page (a Kaggle competition, a user
    profile): those are refused with a specific reason instead of being guessed at.
    """
    canonical = validate_public_url(url)
    parts = urlsplit(canonical)
    source = ALLOWED_HOSTS[parts.hostname or ""]
    segments = [s for s in parts.path.split("/") if s]

    if source == "kaggle":
        if segments[:1] == ["datasets"] and len(segments) >= 3:
            owner, slug = segments[1], segments[2]
            return ParsedDatasetURL("kaggle", f"{owner}/{slug}", _words(slug), canonical)
        if segments[:1] in (["competitions"], ["c"]):
            raise UnsafeURLError(
                "Kaggle competition pages are not dataset records; link the dataset page "
                "(kaggle.com/datasets/owner/name) or upload the file"
            )
        raise UnsafeURLError("this Kaggle URL is not a dataset page (kaggle.com/datasets/owner/name)")

    if source == "huggingface":
        if segments[:1] != ["datasets"] or len(segments) < 2:
            raise UnsafeURLError(
                "this Hugging Face URL is not a dataset page (huggingface.co/datasets/owner/name)"
            )
        rest = segments[1:]
        if len(rest) >= 2 and rest[1] not in _HF_RESERVED:
            identifier = f"{rest[0]}/{rest[1]}"
            hint = _words(rest[1])
        else:
            identifier = rest[0]
            hint = _words(rest[0])
        return ParsedDatasetURL("huggingface", identifier, hint, canonical)

    if source == "uci":
        if segments[:1] == ["dataset"] and len(segments) >= 2 and segments[1].isdigit():
            hint = _words(segments[2]) if len(segments) >= 3 else None
            return ParsedDatasetURL("uci", segments[1], hint, canonical)
        if segments[:2] == ["ml", "datasets"] and len(segments) >= 3:
            name = _words(segments[2])
            return ParsedDatasetURL("uci", name, name, canonical)
        raise UnsafeURLError("this UCI URL is not a dataset page (archive.ics.uci.edu/dataset/<id>)")

    if source == "openml":
        if segments[:1] == ["d"] and len(segments) >= 2 and segments[1].isdigit():
            return ParsedDatasetURL("openml", segments[1], None, canonical)
        query = parse_qs(parts.query)
        ids = query.get("id", [])
        if segments[:1] == ["search"] and query.get("type", [""])[0] == "data" and ids and ids[0].isdigit():
            return ParsedDatasetURL("openml", ids[0], None, canonical)
        raise UnsafeURLError("this OpenML URL is not a dataset page (openml.org/d/<id>)")

    raise UnsafeURLError("unsupported source")  # pragma: no cover - allowlist is exhaustive


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetNameHint:
    name: str
    confidence: float
    evidence: str
    generic: bool = False


_FILE_RE = re.compile(r"(?<![\w/\\])([\w\-]{1,80})\.(csv|xlsx|xls|tsv)\b", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"“'‘]([^\"”'’\n]{2,60})[\"”'’]\s+(?:dataset|data ?set)", re.IGNORECASE)
_CALLED_RE = re.compile(
    r"(?:dataset|data ?set)\s+(?:called|named|titled)\s+[\"“'‘]?([A-Za-z0-9][^\"”'’\n.,;!?]{1,59})",
    re.IGNORECASE,
)
_NOUN_RE = re.compile(r"\b(dataset|data ?set)\b", re.IGNORECASE)


def normalize_dataset_name(name: str) -> str:
    """Lowercase, punctuation-free name with the generic word 'dataset' removed."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    kept = [w for w in words if w not in {"dataset", "datasets", "data", "set", "csv"}]
    return " ".join(kept or words)


def _is_known(name: str) -> bool:
    return normalize_dataset_name(name) in KNOWN_DATASET_NAMES


def _hint(name: str, evidence: str, base: float) -> Optional[DatasetNameHint]:
    cleaned = " ".join(name.split()).strip(" .,;:!?\"'")
    if not cleaned or len(cleaned) > 80:
        return None
    key = normalize_dataset_name(cleaned)
    words = key.split()
    if not words:
        return None
    if _is_known(cleaned):
        return DatasetNameHint(cleaned, 0.95, f"{evidence}; a well-known dataset name", False)
    generic = all(w in _GENERIC_TOPIC_WORDS for w in words)
    if generic:
        return DatasetNameHint(
            cleaned, 0.3, f"{evidence}; but only a topic word, too vague to look up", True
        )
    return DatasetNameHint(cleaned, base, evidence, False)


def extract_dataset_name(text: Optional[str]) -> Optional[DatasetNameHint]:
    """The dataset a user names in free text, or ``None``.

    Tried in order of how explicit the user was: a quoted title, "dataset called X", a
    file name, then the words immediately before the word "dataset". A name made only
    of topic words ("student") is returned at low confidence and flagged ``generic`` so
    the caller asks instead of guessing.
    """
    if not text:
        return None

    m = _QUOTED_RE.search(text)
    if m and (hint := _hint(m.group(1), "named in quotes", 0.9)):
        return hint

    m = _CALLED_RE.search(text)
    if m and (hint := _hint(m.group(1), "introduced as 'called/named'", 0.88)):
        return hint

    m = _FILE_RE.search(text)
    if m and (hint := _hint(m.group(1).replace("_", " ").replace("-", " "), f"file name {m.group(0)!r}", 0.85)):
        return hint

    for noun in _NOUN_RE.finditer(text):
        before = re.findall(r"[A-Za-z0-9][\w\-\.]*", text[: noun.start()])[-5:]
        while before and before[0].lower().strip(".") in _LEADING_FILLER:
            before = before[1:]
        # also trim trailing filler such as "the" directly before the noun
        while before and before[-1].lower() in _LEADING_FILLER:
            before = before[:-1]
        if before and (hint := _hint(" ".join(before), "the words before 'dataset'", 0.7)):
            return hint

    lowered = normalize_dataset_name(text)
    for known in sorted(_DISTINCTIVE_NAMES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(known)}\b", lowered):
            return DatasetNameHint(known, 0.8, "mentions a well-known dataset name", False)
    return None
