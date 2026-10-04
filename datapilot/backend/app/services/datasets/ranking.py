"""Deterministic dataset ranking engine.

No LLM participates in the numerical ranking. Every score comes from
structured, inspectable evidence: project/dataset term overlap, declared
metadata, and the candidate's verification status. The same inputs always
produce the same ranking, which is what makes the result testable and
reproducible.

Scoring model
-------------
Ten factors measure how well a dataset fits the project, one measures whether it
is externally confirmed::

    relevance_score = weighted mean of requirement factors   (0-100)
    quality_score   = weighted mean of publisher signals     (0-100)
    ranking_score   = weighted mean of every factor          (0-100)

Default weights, all configurable in ``Settings``::

    REQUIREMENTS (70)                        PUBLISHER (15)
        domain_relevance  16                     quality         8
        feature_match     16                     popularity      5
        target_match      13                     recency         2
        keyword_overlap    9
        task_match         8        TRUST (15)
        compatibility      8                     verification   15
        -------------------                       ---------------
        content total     70                       total         100

Three rules give the model its meaning:

* **Relevance dominates.** The requirement factors own 70 of 100 points.
  Quality, adoption and freshness together own 15, so a popular dataset cannot
  buy its way past a relevant one; ``relevance_score`` excludes them entirely so
  "relevant" and "popular" stay separately readable.
* **Missing evidence is excluded, not zeroed** - *unless* the source would have
  published the field and did not say anything matching. See below.
* **Unknown is not mismatch.** A registry that has no ``target_description``
  column cannot tell us a dataset lacks the target; it can only leave us
  uninformed. Requirement verdicts therefore distinguish ``match`` from
  ``unknown`` from ``mismatch``, and only ``mismatch`` - which requires the
  candidate to have *stated* something incompatible - is allowed to reach 0.

Trust
-----
``verification`` reports whether a registry really lists this dataset. It is a
separate axis from relevance by design: an AI suggestion with overwhelming
relevance is allowed to outrank a verified but weakly-relevant dataset, and the
ranking reasons say so in words.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.dataset import (
    DatasetCandidate,
    MatchQuality,
    MatchStatus,
    RankingFactor,
    RankingFactorName,
    RequirementMatch,
    SourceType,
    VerificationStatus,
)
from app.schemas.project import MatchType, ProjectAnalysisData
from app.services.datasets.compatibility import (
    Contradiction,
    find_contradictions,
    penalty_total,
)
from app.services.datasets.normalizer import infer_domain_from_tags
from app.services.datasets.target_concepts import (
    get_target_concept_index,
    singularize,
    value_tokens,
)
from app.services.keyword_service import KeywordService

logger = get_logger(__name__)

_PUNCT = re.compile(r"[^a-z0-9]+")

# Evidence hierarchy for a project concept, strongest first.
#
# A concept that the user wrote out, or that the dictionary matched exactly, is a
# statement about this project. A concept the model merely listed as "related" is
# a guess about the neighbourhood, and must not be able to outvote several strong
# signals. Ranking ``concept_overlap`` is a *weighted* coverage using these
# values, so a candidate that only matches generic related vocabulary scores
# close to zero even though it matched "something".
_CONCEPT_EVIDENCE_TIER: dict[MatchType, float] = {
    MatchType.EXACT: 1.0,
    MatchType.SYNONYM: 0.85,
    MatchType.ALIAS: 0.8,
    MatchType.PHRASE: 0.7,
    MatchType.ABBREVIATION: 0.6,
    MatchType.RELATED: 0.35,
}

# The model inferred the concept from the description without dictionary support.
_INFERRED_TIER = 0.55
_DEFAULT_TIER = 0.5

# Multi-word concept names are more specific evidence than single words, and the
# difference compounds with the tier above: an exact hit on "intrusion detection"
# is far better evidence than a bare "detection".
_PHRASE_SPECIFICITY = {1: 0.55, 2: 0.85, 3: 0.95}
_FULL_PHRASE_SPECIFICITY = 1.0
_ABBREVIATION_SPECIFICITY = 0.45


def _phrase_specificity(phrase: str, match_type: Optional[MatchType]) -> float:
    words = len(phrase.split())
    if match_type is MatchType.ABBREVIATION:
        return _ABBREVIATION_SPECIFICITY
    return _PHRASE_SPECIFICITY.get(words, _FULL_PHRASE_SPECIFICITY)


# Generic English and generic-ML words. A project term made only of these
# carries no retrieval signal. Note this list is deliberately different from the
# query builder's stop list: "student" is noise when forming a registry query
# but is strong ranking evidence when it appears in a dataset description.
_TERM_STOPWORDS = frozenset(
    {
        "a", "an", "and", "any", "are", "as", "at", "based", "be", "been",
        "build", "building", "but", "by", "can", "could", "create", "data",
        "dataset", "datasets", "deep", "detect", "detection", "develop",
        "each", "for", "from", "give", "has", "have", "how", "identify", "in",
        "into", "is", "it", "its", "learning", "like", "machine", "make",
        "method", "model", "need", "network", "of", "on", "or", "other",
        "our", "output", "predict", "prediction", "project", "should", "such",
        "supervised", "system", "than", "that", "the", "their", "then",
        "there", "these", "they", "this", "to", "unsupervised", "use", "used",
        "using", "via", "want", "was", "were", "what", "when", "which", "who",
        "will", "with", "would", "you",
    }
)

MAX_REASONS = 6

# Which factors answer which question. The split is what lets relevance be read
# independently of publisher signals, and it matches the three weight groups in
# the module docstring.
_REQUIREMENT_FACTORS = frozenset(
    {
        RankingFactorName.DOMAIN_RELEVANCE,
        RankingFactorName.KEYWORD_OVERLAP,
        RankingFactorName.FEATURE_MATCH,
        RankingFactorName.TARGET_MATCH,
        RankingFactorName.TASK_MATCH,
        RankingFactorName.COMPATIBILITY,
    }
)
_PUBLISHER_FACTORS = frozenset(
    {
        RankingFactorName.QUALITY,
        RankingFactorName.POPULARITY,
        RankingFactorName.RECENCY,
    }
)

#: Adoption blend. The two shares always sum to :data:`POPULARITY_BLEND_WEIGHT`,
#: which is the divisor ``_popularity`` uses. Keeping the nominal total explicit
#: means an unpublished part cannot rescale the part that was published.
POPULARITY_DOWNLOAD_SHARE = 0.7
POPULARITY_VOTE_SHARE = 0.3
POPULARITY_BLEND_WEIGHT = POPULARITY_DOWNLOAD_SHARE + POPULARITY_VOTE_SHARE

# Evidence availability for one requirement dimension.
#
# ``DECLARED`` - the source published this field, so a requirement that finds
#                nothing there really is absent.
# ``PROSE``    - only free text describes the dataset, so silence proves nothing.
# ``NONE``     - nothing at all is known; the requirement is undecidable.
#
# This distinction is the whole of "unknown is not mismatch". A Kaggle record has
# no ``target_description`` and no ``feature_description`` field, so its target
# and feature coverage is PROSE: a match is credit, a miss is UNKNOWN.
DECLARED = "declared"
PROSE = "prose"
NONE = "none"

# Cues that free text is actually enumerating the dataset's schema. "Contains
# columns sepal length, sepal width, petal length" is a declaration of what the
# dataset has, and a required feature missing from that list is then absent rather
# than unknown. Kept narrow: "feature" alone is far too common a word.
_SCHEMA_CUES = frozenset(
    {
        "attribute", "attributes", "column", "columns", "field", "fields",
        "schema", "schema", "variable", "variables", "row", "rows",
    }
)


def _declares_schema(text: Optional[str]) -> bool:
    if not text:
        return False
    tokens = set(_PUNCT.sub(" ", text.lower()).split())
    return bool(tokens & _SCHEMA_CUES)


# ---------------------------------------------------------------------------
# Text matching
# ---------------------------------------------------------------------------


def _normalize_phrase(text: Optional[str]) -> str:
    if not text:
        return ""
    return _PUNCT.sub(" ", text.lower().replace("_", " ")).strip()


# One shared plural-folding rule: evidence tokens and dictionary label values must
# be folded identically or target matching would compare "maliciou" with
# "malicious" and silently find nothing.
_singularize = singularize


def _term_tokens(term: str) -> list[str]:
    return [t for t in _normalize_phrase(term).split() if t]


def _match_tokens(text: str) -> list[str]:
    return [_singularize(t) for t in _normalize_phrase(text).split() if t]


def _is_meaningful(term: str) -> bool:
    tokens = _term_tokens(term)
    if not tokens:
        return False
    return any(len(t) > 2 and t not in _TERM_STOPWORDS for t in tokens)


class _Evidence:
    """Normalized searchable text belonging to one candidate.

    Term matching is deliberately tolerant but not loose, and it reports *how* a
    term was found, which is what the evidence hierarchy is built on:

    * ``EXACT``  - the whole phrase appears, after normalization and plural
      folding. The strongest evidence available.
    * ``ALIAS``  - every significant token of the term appears somewhere in the
      evidence, so "class participation" also matches "participation in class".
      The words are all present but in a different arrangement.
    * ``None``   - not present.

    Requiring *all* tokens of a multi-word term keeps the relaxation from
    producing false positives; synonym gaps are covered separately by the
    dictionary-backed alias expansion, which is reported as ``SEMANTIC``.

    ``declared`` holds the tokens of individual metadata fields, or the absence
    of the field itself. That distinction is what separates "the registry does
    not publish a target description" from "the target description says there is
    no target".
    """

    __slots__ = ("phrase", "tokens", "declared", "raw")

    def __init__(
        self,
        text: str,
        declared: Optional[dict[str, frozenset[str]]] = None,
        raw: Optional[dict[str, str]] = None,
    ) -> None:
        words = _match_tokens(text)
        self.phrase = " ".join(words)
        self.tokens = set(words)
        self.declared = declared or {}
        self.raw = raw or {}

    def quality(self, term: str) -> Optional[MatchQuality]:
        tokens = [_singularize(t) for t in _term_tokens(term)]
        if not tokens:
            return None
        if len(tokens) == 1:
            return MatchQuality.EXACT if tokens[0] in self.tokens else None
        if " ".join(tokens) in self.phrase:
            return MatchQuality.EXACT
        if all(token in self.tokens for token in tokens):
            return MatchQuality.ALIAS
        return None

    def contains(self, term: str) -> bool:
        return self.quality(term) is not None

    def field_tokens(self, field: str) -> Optional[frozenset[str]]:
        """Tokens of one metadata field, or ``None`` if it was never published."""
        return self.declared.get(field)

    def declares_schema(self) -> bool:
        """Whether the free text enumerates the dataset's own schema.

        "Contains columns sepal length, sepal width, petal length" is a
        statement about what the dataset has, so a required feature absent from
        that list is *absent* rather than merely unproven.
        """
        return any(_declares_schema(self.raw.get(f)) for f in ("description", "feature_description"))


def _dedupe_terms(terms: Sequence[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term:
            continue
        key = _normalize_phrase(term)
        if not key or key in seen or not _is_meaningful(term):
            continue
        seen.add(key)
        out.append(term.strip())
        if len(out) >= limit:
            break
    return out


# The dictionary is loaded once per process and reused across ranking runs.
_KEYWORD_SERVICE: Optional[KeywordService] = None


def _keyword_service() -> KeywordService:
    global _KEYWORD_SERVICE
    if _KEYWORD_SERVICE is None:
        _KEYWORD_SERVICE = KeywordService()
    return _KEYWORD_SERVICE


def _expand_aliases(terms: Sequence[str], include_features: bool = False) -> dict[str, list[str]]:
    """Map each base term to dictionary-backed equivalent phrases.

    A synonym only matches when the *project knowledge dictionary* already
    treats it as the same concept (canonical term, synonym, alias, or declared
    target requirement), which is what makes equivalence dictionary-derived
    instead of a hand-written list in the ranker. ``include_features`` adds a
    concept's declared ``feature_requirements`` so that "BMI" and "body mass
    index", or "salary" and "estimated salary", are known to be the same thing
    because the dictionary says so.
    """
    aliases: dict[str, list[str]] = {}
    service = _keyword_service()
    for term in terms:
        key = _normalize_phrase(term)
        if not key:
            continue
        matches = service.match(term)
        if matches:
            # A RELATED hit means the dictionary lists *this term* as a related
            # concept of another concept; it is the reverse direction, not an
            # equivalent, so it must not contribute aliases (otherwise "student
            # performance" would inherit "attendance", "examination", ...).
            matches = [m for m in matches if m.match_type is not MatchType.RELATED]
        if not matches:
            # Not a concept in its own right. A feature name can still be a
            # declared requirement of one - "BMI" is not a dictionary concept, but
            # the knowledge base says the concepts needing it also accept "body
            # mass index" - so the feature index is the fallback.
            if not include_features:
                continue
            entries = service.concepts_declaring_feature(term)
            if not entries:
                continue
            extra = _equivalents_of(entries, exclude=key, include_features=True)
            if extra:
                aliases[key] = extra
            continue
        # Use only the most specific dictionary phrase found for this term. A
        # short partial hit ("academic" -> the broad "education" concept) must
        # not drag in that concept's unrelated vocabulary; "academic
        # performance" should resolve to the "student performance" concept.
        best_len = max(len(_normalize_phrase(m.matched_text).split()) for m in matches)
        entries = [
            service.get_concept(m.dictionary_entry_id) or {}
            for m in matches
            if len(_normalize_phrase(m.matched_text).split()) == best_len
        ]
        extra = _equivalents_of(entries, exclude=key, include_features=include_features)
        if extra:
            aliases[key] = extra
    return aliases


def _equivalents_of(
    concepts: Sequence[dict],
    *,
    exclude: str,
    include_features: bool = False,
) -> list[str]:
    """Dictionary phrases that describe the same thing as ``exclude``.

    Built only from what the knowledge base declares: a concept's canonical
    term, synonyms, aliases and target vocabulary, plus - for feature terms -
    the concept's ``feature_aliases``.

    ``feature_aliases`` exists so that equivalence is *stated* rather than
    inferred. Two measurements being neighbours in one concept's feature list
    does not make them the same measurement - a dataset with blood-pressure
    readings does not satisfy a requirement for BMI - so sibling features are
    deliberately not offered as aliases.
    """
    out: list[str] = []
    seen: set[str] = set()
    for concept in concepts:
        if not concept:
            continue
        phrases = (
            [concept.get("canonical_term", "")]
            + list(concept.get("synonyms", []))
            + list(concept.get("aliases", []))
            + list(concept.get("target_requirements", []))
        )
        if include_features:
            # Only the aliases declared for *this* measurement. Reading every
            # concept's alias table would let "BMI" be satisfied by a dataset that
            # publishes glycated hemoglobin instead, which is a different test.
            phrases += list((concept.get("feature_aliases") or {}).get(exclude, []))
        for phrase in phrases:
            normalized = _normalize_phrase(phrase)
            if normalized and normalized != exclude and normalized not in seen:
                seen.add(normalized)
                out.append(str(phrase).strip())
    return out


def _concise_target(value: Optional[str]) -> str:
    """Reduce a model-written target to a short, matchable concept.

    A concise value is kept. A verbose, sentence-like value is resolved through
    the knowledge dictionary to its canonical concept ("Student performance
    metrics and support requirement status" -> "student performance"); when no
    concept matches, the value is dropped. Matching a whole sentence instead
    would make the target factor fire on almost any education dataset (every
    significant token appears somewhere), so the long form is never used.
    """
    if not value:
        return ""
    if len(_normalize_phrase(value).split()) <= 4:
        return value.strip()
    return _keyword_service().resolve_canonical_phrase(value) or ""


# ---------------------------------------------------------------------------
# Project profile
# ---------------------------------------------------------------------------


@dataclass
class ProjectProfile:
    """Structured project terms used for matching. Built once per ranking run."""

    domain_label: Optional[str] = None
    domain_terms: list[str] = field(default_factory=list)
    concept_terms: list[str] = field(default_factory=list)
    # normalized concept term -> how much evidence it carries. The evidence
    # hierarchy: a concept the user or dictionary named exactly outweighs one the
    # model only listed as "related", and a multi-word concept outweighs a bare
    # one. Coverage is a weighted mean over these, so one strong concept cannot
    # be outvoted by several weak guesses.
    concept_weights: dict[str, float] = field(default_factory=dict)
    feature_terms: list[str] = field(default_factory=list)
    target_terms: list[str] = field(default_factory=list)
    # The prediction target exactly as the model wrote it. Reducing a verbose
    # target to its canonical concept keeps a topic ("network traffic") and drops
    # the label it also names ("Normal vs Attack"), so the original text is kept
    # as a second, label-aware evidence source for target matching.
    raw_target: str = ""
    task_terms: list[str] = field(default_factory=list)
    project_domain_key: Optional[str] = None
    # base term (normalized) -> dictionary-backed equivalent phrases
    feature_aliases: dict[str, list[str]] = field(default_factory=dict)
    target_aliases: dict[str, list[str]] = field(default_factory=dict)
    # Free-text statements of the required data modality ("image", "tabular"),
    # kept verbatim because contradiction detection needs the model's own words.
    data_type_notes: list[str] = field(default_factory=list)
    # Every target string the project asked for, used to detect a candidate that
    # declares a different label space outright.
    target_labels: list[str] = field(default_factory=list)


def _concept_weights(keywords, terms: Sequence[str]) -> dict[str, float]:
    """Assign each project concept its evidence weight, keeping the strongest claim.

    Sources, in order of authority: the model's canonical terms (it restated the
    user's own words), dictionary matches at their tier, then the model's
    related concepts, which are suggestions about the neighbourhood rather than
    statements about this project.
    """
    weights: dict[str, float] = {}

    def record(term: str, weight: float) -> None:
        key = _normalize_phrase(term)
        if not key:
            return
        weights[key] = max(weights.get(key, 0.0), weight)

    for match in keywords.dictionary_matches:
        record(
            match.matched_text,
            _CONCEPT_EVIDENCE_TIER.get(match.match_type, _DEFAULT_TIER)
            * _phrase_specificity(match.matched_text, match.match_type),
        )
    for term in keywords.canonical:
        record(term, _CONCEPT_EVIDENCE_TIER[MatchType.EXACT])
    for term in keywords.related_concepts:
        record(
            term,
            _CONCEPT_EVIDENCE_TIER[MatchType.RELATED]
            * _phrase_specificity(term, MatchType.RELATED),
        )
    # Any term that survived de-duplication but matched nothing in the dictionary
    # is the model asserting a concept the knowledge base does not know. That is
    # still usable evidence - the model read the user's sentence - but it is the
    # weakest claim available, so it sits below a dictionary match.
    known = set(weights)
    for term in terms:
        if _normalize_phrase(term) not in known:
            record(
                term,
                _INFERRED_TIER * _phrase_specificity(term, MatchType.SYNONYM),
            )
    return weights


def build_project_profile(requirements: ProjectAnalysisData) -> ProjectProfile:
    understanding = requirements.project_understanding
    keywords = requirements.keywords
    dataset_requirements = requirements.dataset_requirements

    domain_label = understanding.domain.value if understanding.domain else None
    domain_terms = _dedupe_terms(
        [
            understanding.domain.value if understanding.domain else "",
            understanding.subdomain.value if understanding.subdomain else "",
            understanding.problem_type.value if understanding.problem_type else "",
        ],
        limit=10,
    )

    concept_terms = _dedupe_terms(
        list(keywords.canonical)
        + list(keywords.related_concepts)
        + [m.matched_text for m in keywords.dictionary_matches],
        limit=25,
    )

    # Match against the feature *name* (a short, stable label) rather than
    # "name description". Concatenating a model-written description turns the
    # term into a long phrase that no real dataset description ever contains,
    # which silently zeroes the feature factor.
    feature_terms = _dedupe_terms(
        [f.name for f in dataset_requirements.feature_requirements],
        limit=20,
    )

    raw_target = understanding.target.value if understanding.target else ""
    target_terms = _dedupe_terms(
        [_concise_target(t.name) for t in dataset_requirements.target_requirements]
        + ([_concise_target(raw_target)] if raw_target else []),
        limit=10,
    )

    task_terms = _dedupe_terms([t.task for t in understanding.ml_tasks], limit=8)

    return ProjectProfile(
        domain_label=domain_label,
        domain_terms=domain_terms,
        concept_terms=concept_terms,
        concept_weights=_concept_weights(keywords, concept_terms),
        feature_terms=feature_terms,
        target_terms=target_terms,
        raw_target=raw_target,
        task_terms=task_terms,
        # Reuses the normalizer's tag->domain mapping so the candidate's topic
        # domain and the project's domain are compared in one taxonomy.
        project_domain_key=infer_domain_from_tags([domain_label]) if domain_label else None,
        feature_aliases=_expand_aliases(feature_terms, include_features=True),
        target_aliases=_expand_aliases(target_terms),
        data_type_notes=[
            f"{d.type} {d.description}".strip() for d in dataset_requirements.data_type_requirements
        ],
        target_labels=_dedupe_terms(
            [t.name for t in dataset_requirements.target_requirements]
            + ([raw_target] if raw_target else []),
            limit=10,
        ),
    )


# ---------------------------------------------------------------------------
# Requirement assessment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Verdict:
    """One requirement's outcome, before it becomes a schema object."""

    term: str
    status: MatchStatus
    quality: MatchQuality
    weight: float
    via: Optional[str] = None


# A model asked to describe a target will often report that the dataset does not
# *have* one - "requires feature engineering to derive churn status based on time
# since last purchase". Such a sentence still contains the project's words, so a
# literal search scores it as a full match: the exact opposite of what it means.
# Only a declaration of absence is read this way, and it yields UNKNOWN rather
# than a mismatch, because "you would have to derive it" is not the same claim as
# "the label is a different value".
_TARGET_NOT_PROVIDED = re.compile(
    r"\b(?:no|without|missing|lacks?|lacking|absent)\b[^.;]{0,40}?"
    r"\b(?:label|labels|labell?ed|target|column|columns|field|fields|outcome|"
    r"annotation|ground[- ]truth|indicator)\b"
    r"|\bnot\s+(?:provided|included|available|supplied|labell?ed|annotated|"
    r"collected|measured|recorded|given|present)\b"
    r"|\b(?:requires?|needs?|must\s+be|can\s+be|could\s+be|would\s+need\s+to|"
    r"you\s+(?:can|could|would)\s+need\s+to)\b[^.;]{0,60}?"
    r"\b(?:deriv|comput|calculat|infer|inferenc|construct|engineer|creat|annotat|"
    r"label|assign|determin)\w*\b",
    re.IGNORECASE,
)

# The grammar a requirement uses to name a label is not part of what the label is.
# "customer churn status" and "customer churn" ask the same question, so a dataset
# whose title says "Customer Churn" satisfies it. The stripped form is reported as
# the equivalence that matched, so the weaker claim stays visible.
_GENERIC_LABEL_HEADS = frozenset(
    {
        "class", "classes", "column", "field", "flag", "indicator", "label",
        "outcome", "output", "prediction", "result", "results", "status", "target",
        "value", "variable",
    }
)


def _label_head_aliases(
    aliases: dict[str, list[str]], terms: Sequence[str]
) -> dict[str, list[str]]:
    """Add the label's head-noun-free form as an equivalent of each target term."""
    out = {key: list(values) for key, values in aliases.items()}
    for term in terms:
        key = _normalize_phrase(term)
        words = key.split()
        if len(words) < 2 or words[-1] not in _GENERIC_LABEL_HEADS:
            continue
        trimmed = " ".join(words[:-1])
        if trimmed and trimmed not in out.setdefault(key, []):
            out[key].append(trimmed)
    return out


def _assess_terms(
    terms: Sequence[str],
    evidence: _Evidence,
    aliases: Optional[dict[str, list[str]]],
    *,
    availability: str,
    schema_declared: bool = False,
    weights: Optional[dict[str, float]] = None,
    resolve_alias: Optional[Callable[[str], Optional[tuple[str, MatchQuality]]]] = None,
    resolve_concept: Optional[Callable[[str], Optional[tuple[str, MatchQuality]]]] = None,
) -> list[_Verdict]:
    """Decide MATCH / UNKNOWN / MISMATCH for every requirement, independently.

    The order of checks is the meaning of the function:

    1. the requirement's own words appear -> ``MATCH``;
    2. a dictionary-backed equivalent appears -> ``MATCH``, quality ``SEMANTIC``
       (or ``INFERRED`` when the equivalence came from the model rather than the
       dictionary);
    3. a target concept the requirement resolves to declares the value the
       candidate shows -> ``MATCH``, quality ``SEMANTIC``;
    4. nothing found. Now the availability of the evidence decides the verdict:
       a field the source publishes (``DECLARED``), or prose that enumerates the
       schema, makes the absence real and yields ``MISMATCH``; a source that
       simply never publishes this field yields ``UNKNOWN``, which the caller
       must not turn into a zero.
    """
    verdicts: list[_Verdict] = []
    for term in terms:
        weight = (weights or {}).get(_normalize_phrase(term), 1.0)
        quality = evidence.quality(term)
        if quality is not None:
            verdicts.append(_Verdict(term, MatchStatus.MATCH, quality, weight))
            continue

        via: Optional[tuple[str, MatchQuality]] = None
        inferred = False
        if aliases:
            for alias in aliases.get(_normalize_phrase(term), []):
                alias_quality = evidence.quality(alias)
                if alias_quality is not None:
                    via = (alias, MatchQuality.SEMANTIC)
                    break
        if via is None and resolve_alias is not None:
            via = resolve_alias(term)
        if via is None and resolve_concept is not None:
            resolved = resolve_concept(term)
            if resolved is not None:
                via = resolved
                inferred = resolved[1] is MatchQuality.INFERRED
        if via is not None:
            verdicts.append(
                _Verdict(
                    term,
                    MatchStatus.MATCH,
                    MatchQuality.INFERRED if inferred else via[1],
                    weight,
                    via=f"via {via[0]!r}",
                )
            )
            continue

        status = (
            MatchStatus.MISMATCH
            if availability is DECLARED or (availability is PROSE and schema_declared)
            else MatchStatus.UNKNOWN
        )
        verdicts.append(_Verdict(term, status, MatchQuality.INFERRED, weight))
    return verdicts


def _verdict_factor(
    name: RankingFactorName,
    weight: float,
    verdicts: list[_Verdict],
    *,
    dimension: str,
    basis: Optional[str] = None,
) -> RankingFactor:
    """Turn verdicts into a factor, weighting coverage by evidence strength.

    ``coverage`` is the weighted fraction of requirements that were met, so a
    match on a concept the project stated exactly counts for more than a match
    on something the model merely guessed at.

    When nothing matched, the difference between "absent" and "unjudgeable" is the
    whole point:

    * a requirement the source *could* have declared and did not is a real
      absence, and scores 0 - that is a genuine mismatch;
    * a requirement about metadata the source never publishes is unjudgeable, so
      the factor stays unscored and drops out of the weighted mean rather than
      registering as a zero. Nothing is invented in either direction.
    """
    matched = [v for v in verdicts if v.status is MatchStatus.MATCH]
    unknown = [v for v in verdicts if v.status is MatchStatus.UNKNOWN]
    mismatched = [v for v in verdicts if v.status is MatchStatus.MISMATCH]
    total_weight = sum(v.weight for v in verdicts)

    factor = RankingFactor(
        name=name,
        weight=weight,
requirements=[
                RequirementMatch(
                    term=v.term,
                    status=v.status,
                    quality=v.quality,
                    weight=round(v.weight, 4),
                    via=v.via,
                    detail=f"judged as a {dimension} requirement",
                )
                for v in verdicts
            ],
        unmatched_terms=[v.term for v in mismatched],
        unknown_terms=[v.term for v in unknown],
    )

    if not verdicts:
        factor.detail = f"No {dimension} requirements identified"
        return factor

    if matched:
        coverage = sum(v.weight for v in matched) / total_weight if total_weight else 0.0
        factor.score = round(min(coverage, 1.0), 4)
        factor.coverage = factor.score
    elif mismatched:
        # Every requirement was judged and none was found: a stated absence.
        factor.score = 0.0
        factor.coverage = 0.0
    else:
        factor.detail = (
            f"{dimension}: none of the {len(verdicts)} requirements could be judged - "
            "the source does not publish this metadata"
        )
        return factor

    factor.matched_terms = [v.term for v in matched]
    factor.matched_via = [v.via for v in matched if v.via]

    counts = [f"{len(matched)}/{len(verdicts)} {dimension} requirements met"]
    if basis:
        counts[0] += f" in {basis}"
    if unknown:
        counts.append(f"{len(unknown)} unjudgeable")
    if mismatched:
        counts.append(f"{len(mismatched)} not evidenced")
    factor.detail = "; ".join(counts)
    return factor


# ---------------------------------------------------------------------------
# Ranking engine
# ---------------------------------------------------------------------------


class DatasetRankingEngine:
    """Scores and orders candidates deterministically."""

    def __init__(self, settings: Optional[Settings] = None, now: Optional[datetime] = None) -> None:
        self._settings = settings or get_settings()
        # Injectable clock keeps ranking reproducible in tests.
        self._now = now

    # -- weights -----------------------------------------------------------

    def _weight(self, name: RankingFactorName) -> float:
        return {
            RankingFactorName.DOMAIN_RELEVANCE: self._settings.ranking_weight_domain_relevance,
            RankingFactorName.KEYWORD_OVERLAP: self._settings.ranking_weight_keyword_overlap,
            RankingFactorName.FEATURE_MATCH: self._settings.ranking_weight_feature_match,
            RankingFactorName.TARGET_MATCH: self._settings.ranking_weight_target_match,
            RankingFactorName.TASK_MATCH: self._settings.ranking_weight_task_match,
            RankingFactorName.COMPATIBILITY: self._settings.ranking_weight_compatibility,
            RankingFactorName.QUALITY: self._settings.ranking_weight_quality,
            RankingFactorName.POPULARITY: self._settings.ranking_weight_popularity,
            RankingFactorName.RECENCY: self._settings.ranking_weight_recency,
            RankingFactorName.VERIFICATION: self._settings.ranking_weight_verification,
        }[name]

    def weights(self) -> dict[str, float]:
        return {name.value: self._weight(name) for name in RankingFactorName}

    # -- entry point -------------------------------------------------------

    def rank(
        self,
        candidates: Sequence[DatasetCandidate],
        requirements: ProjectAnalysisData,
    ) -> list[DatasetCandidate]:
        """Score every candidate in place and return them best-first.

        Ordering is by ``ranking_score``, whose requirement factors hold 70 of
        the 100 available points, so relevance decides in practice; ties fall back
        to verification status, then name, then id, which keeps the result total
        and reproducible.
        """
        if not candidates:
            return []

        profile = build_project_profile(requirements)
        scored = [self._score(candidate, profile) for candidate in candidates]

        scored.sort(
            key=lambda c: (
                -float(c.ranking_score or 0.0),
                0 if c.verification_status is VerificationStatus.VERIFIED else 1,
                c.name.lower(),
                c.id,
            )
        )
        logger.debug("Ranked %d candidates", len(scored))
        return scored

    @staticmethod
    def _candidate_evidence(candidate: DatasetCandidate) -> _Evidence:
        """Assemble searchable text plus the per-field token maps.

        A field that was never published is recorded as an empty set; a field
        that was published is recorded with its tokens. The availability rules
        below read that difference to decide whether a missing requirement is a
        mismatch or merely unknown.
        """

        def tokens(text: Optional[str]) -> frozenset[str]:
            return frozenset(_match_tokens(text or ""))

        declared = {
            "name": tokens(candidate.name),
            "tags": tokens(", ".join(candidate.tags)),
            "domain": tokens(candidate.domain),
            "description": tokens(candidate.description),
            "feature_description": tokens(candidate.feature_description),
            "target_description": tokens(candidate.target_description),
            "task_types": tokens(" ".join(candidate.task_types)),
        }
        return _Evidence(
            candidate.evidence_text(),
            declared=declared,
            raw={
                "description": candidate.description,
                "target_description": candidate.target_description,
                "feature_description": candidate.feature_description,
            },
        )

    @staticmethod
    def _publishes_metadata(candidate: DatasetCandidate) -> bool:
        """Whether the candidate's fields are registry declarations, not model output.

        An AI suggestion's ``feature_description``, ``target_description`` and
        ``task_types`` are the generator's best guess. A guess may satisfy a
        requirement, but a guess cannot *contradict* one, and its silence cannot
        prove absence - the same reasoning that switches off contradiction
        detection also governs whether a missing requirement counts as a stated
        absence.
        """
        return candidate.verification_status is VerificationStatus.VERIFIED

    def _score(
        self, candidate: DatasetCandidate, profile: ProjectProfile
    ) -> DatasetCandidate:
        evidence = self._candidate_evidence(candidate)
        publishes = self._publishes_metadata(candidate)
        task_evidence = (
            _Evidence(" ".join(candidate.task_types))
            if publishes and candidate.task_types
            else None
        )
        candidate_domains = self._candidate_domains(candidate)

        factors: list[RankingFactor] = [
            self._domain_relevance(candidate, profile, evidence),
            self._keyword_overlap(profile, evidence),
            self._feature_match(profile, evidence, publishes),
            self._target_match(profile, evidence, candidate_domains, publishes),
            self._task_match(profile, evidence, task_evidence),
            self._compatibility(candidate, profile, candidate_domains),
            self._quality(candidate),
            self._popularity(candidate),
            self._recency(candidate),
            self._verification(candidate),
        ]

        candidate.ranking_factors = factors
        candidate.ranking_score = self._weighted_score(factors)
        candidate.relevance_score = self._relevance_score(factors)
        candidate.quality_score = self._quality_score(factors)
        candidate.ranking_reasons = self._build_reasons(factors, candidate)
        return candidate

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _judged_fraction(factor: RankingFactor) -> float:
        """How much of a factor's requirement set could be judged at all.

        ``1.0`` when the factor has no per-requirement verdicts (domain
        relevance and concept coverage are not requirement dimensions), otherwise
        the weighted share of requirements that reached a MATCH or MISMATCH
        verdict rather than staying UNKNOWN.
        """
        requirements = factor.requirements
        if not requirements:
            return 1.0
        total = sum(r.weight for r in requirements)
        if total <= 0:
            return 1.0
        judged = sum(r.weight for r in requirements if r.status is not MatchStatus.UNKNOWN)
        return judged / total

    def _contributions(
        self, factors: Sequence[RankingFactor]
    ) -> tuple[float, float]:
        """Return (denominator, weighted_sum) for the given factors.

        The denominator is the **nominal** total weight of the factors supplied -
        a constant for any given set - and never the weight of whichever factors
        happened to speak. Every factor therefore keeps its weight in the mean
        whether or not it could be judged, and a silent factor earns nothing.

        That is the whole "unknown is neutral" rule, and it has two consequences
        that are worth stating because they are what the previous denominator got
        wrong:

        * A silent factor cannot inflate anything else. When publisher metadata
          was absent its weight used to leave the denominator, and the survivors
          were rescaled by ``100 / counted_weight`` - so a dataset with *less*
          evidence scored *higher* (93.65 vs 90.40, paying verification 17.65
          against a nominal 15.00).
        * Removing evidence can never raise a score. Silence only ever zeroes a
          term that was already there; it never touches another factor's
          coefficient, because the denominator does not move.

        The requirement itself is still reported as UNKNOWN rather than MISMATCH -
        nothing is invented in either direction - and the shortfall stays visible
        in ``evidence_coverage``, ``unknown_terms`` and the reason text.
        """
        weighted = 0.0
        for factor in factors:
            if factor.score is None:
                continue
            weighted += factor.weight * float(factor.score) * self._judged_fraction(factor)
        denominator = sum(f.weight for f in factors)
        return denominator, weighted

    def _weighted_score(self, factors: list[RankingFactor]) -> float:
        denominator, weighted = self._contributions(factors)
        if denominator <= 0:
            return 0.0
        score = weighted / denominator * 100.0

        # Report each factor's actual point contribution so contributions sum
        # to the final score. Because the denominator is the nominal weight, a
        # factor's contribution is bounded above by its own nominal weight.
        for factor in factors:
            if factor.score is None or denominator <= 0:
                factor.contribution = 0.0
            else:
                judged = self._judged_fraction(factor)
                factor.contribution = round(
                    factor.weight * float(factor.score) * judged / denominator * 100.0, 2
                )
        return round(score, 2)

    def _relevance_score(self, factors: list[RankingFactor]) -> float:
        """How well the dataset fits the request, ignoring publisher and trust signals.

        Reported separately so "relevant" can never be mistaken for "popular" or
        "verified". Compatibility is included: a dataset that positively
        contradicts a requirement is not relevant, however well its description
        reads. Like ``ranking_score`` it discounts what could not be judged, so a
        candidate that stays silent about most requirements cannot present a high
        relevance score.
        """
        denominator, weighted = self._contributions(
            [f for f in factors if f.name in _REQUIREMENT_FACTORS]
        )
        if denominator <= 0:
            return 0.0
        return round(weighted / denominator * 100.0, 2)

    def _quality_score(self, factors: list[RankingFactor]) -> float:
        """Secondary axis: publisher quality, adoption and recency only."""
        denominator, weighted = self._contributions(
            [f for f in factors if f.name in _PUBLISHER_FACTORS]
        )
        if denominator <= 0:
            return 0.0
        return round(weighted / denominator * 100.0, 2)

    # -- individual factors ------------------------------------------------

    @staticmethod
    def _coverage(
        terms: list[str],
        evidence: _Evidence,
        aliases: Optional[dict[str, list[str]]] = None,
    ) -> tuple[Optional[float], list[str]]:
        """Fraction of base terms found in the evidence.

        A base term counts as matched when it appears directly *or* when one of
        its dictionary-backed equivalent phrases appears. The denominator stays
        the base term count, so alias matching rewards recognised synonyms
        without diluting the score.
        """
        if not terms:
            return None, []
        matched: list[str] = []
        for term in terms:
            if evidence.contains(term):
                matched.append(term)
                continue
            if aliases:
                extra = aliases.get(_normalize_phrase(term))
                if extra and any(evidence.contains(alias) for alias in extra):
                    matched.append(term)
        return len(matched) / len(terms), matched

    @staticmethod
    def _term_matched(
        term: str,
        evidence: _Evidence,
        aliases: Optional[dict[str, list[str]]] = None,
    ) -> bool:
        """Whether one term is present literally or through a dictionary alias."""
        if evidence.contains(term):
            return True
        if not aliases:
            return False
        extra = aliases.get(_normalize_phrase(term))
        return bool(extra) and any(evidence.contains(alias) for alias in extra or [])

    def _domain_relevance(
        self, candidate: DatasetCandidate, profile: ProjectProfile, evidence: _Evidence
    ) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.DOMAIN_RELEVANCE,
            weight=self._weight(RankingFactorName.DOMAIN_RELEVANCE),
        )
        if not profile.domain_terms and not profile.project_domain_key:
            factor.detail = "Project domain unknown"
            return factor

        coverage, matched = self._coverage(profile.domain_terms, evidence)

        domain_match = False
        if (
            profile.project_domain_key
            and candidate.domain
            and _normalize_phrase(candidate.domain)
            == _normalize_phrase(profile.project_domain_key)
        ):
            domain_match = True

        score = coverage if coverage is not None else 0.0
        if domain_match:
            score = max(score, 0.85)
            matched = matched + [f"domain:{profile.project_domain_key}"]
        if not profile.domain_terms and domain_match:
            score = 0.85

        factor.score = round(min(score, 1.0), 4)
        factor.matched_terms = matched
        factor.detail = (
            f"topic domain '{candidate.domain}' matches project domain "
            f"'{profile.project_domain_key}'"
            if domain_match
            else f"matched {len(matched)}/{len(profile.domain_terms)} domain terms"
        )
        return factor

    def _keyword_overlap(
        self, profile: ProjectProfile, evidence: _Evidence
    ) -> RankingFactor:
        """Evidence-weighted concept coverage.

        Concepts are not interchangeable evidence. A concept the dictionary
        matched exactly, in the user's own words, is worth more than one the
        model offered as "related"; a multi-word concept is more specific than a
        bare one. The score is therefore the weighted fraction of concept weight
        found, not a count - so a dataset that only matches a handful of weak
        related concepts cannot approach one that matches the concepts the
        project actually named.
        """
        factor = RankingFactor(
            name=RankingFactorName.KEYWORD_OVERLAP,
            weight=self._weight(RankingFactorName.KEYWORD_OVERLAP),
        )
        if not profile.concept_terms:
            factor.detail = "Project concepts unknown"
            return factor

        matched: list[str] = []
        found_weight = 0.0
        total_weight = 0.0
        for term in profile.concept_terms:
            weight = profile.concept_weights.get(_normalize_phrase(term), _DEFAULT_TIER)
            total_weight += weight
            if evidence.contains(term):
                matched.append(term)
                found_weight += weight
        factor.score = round(found_weight / total_weight, 4) if total_weight else None
        factor.coverage = factor.score
        factor.matched_terms = matched
        factor.detail = (
            f"matched {len(matched)}/{len(profile.concept_terms)} project concepts "
            f"({found_weight:.2f}/{total_weight:.2f} of the evidence weight)"
        )
        return factor

    def _feature_match(
        self, profile: ProjectProfile, evidence: _Evidence, publishes: bool = True
    ) -> RankingFactor:
        """How many required features the dataset's metadata supports.

        Three kinds of support, in descending order of trust:

        1. the feature's own words in the dataset metadata;
        2. a dictionary-declared equivalent - the knowledge base says "BMI" and
           "body mass index" describe the same thing - so semantically equivalent
           features count as matches without being listed here;
        3. an AI-inferred tag overlap, credited at ``INFERRED`` quality and only
           when the feature has no dictionary support at all. This is the one
           place a model's guess is allowed to count, so it is kept weak and
           labelled, never silently merged with a real match.
        """
        factor = RankingFactor(
            name=RankingFactorName.FEATURE_MATCH,
            weight=self._weight(RankingFactorName.FEATURE_MATCH),
        )
        if not profile.feature_terms:
            factor.detail = "No required features identified"
            return factor

        tag_tokens = evidence.field_tokens("tags") or frozenset()
        # Feature descriptions and name/tags are published fields; the description
        # is prose unless it enumerates a schema, and an inferred description is
        # prose by definition.
        availability = (
            DECLARED if publishes and evidence.field_tokens("feature_description") else PROSE
        )
        if not (evidence.field_tokens("feature_description") or tag_tokens or evidence.tokens):
            availability = NONE

        def infer_tag(term: str) -> Optional[tuple[str, MatchQuality]]:
            """An AI-suggested feature name echoed by the candidate's own tags."""
            tokens = [_singularize(t) for t in _term_tokens(term)]
            if not tokens or _normalize_phrase(term) in profile.feature_aliases:
                return None
            if all(token in tag_tokens for token in tokens):
                return (f"tag {', '.join(tokens)}", MatchQuality.INFERRED)
            return None

        verdicts = _assess_terms(
            profile.feature_terms,
            evidence,
            profile.feature_aliases,
            availability=availability,
            schema_declared=evidence.declares_schema(),
            resolve_alias=infer_tag,
        )
        return _verdict_factor(
            RankingFactorName.FEATURE_MATCH,
            self._weight(RankingFactorName.FEATURE_MATCH),
            verdicts,
            dimension="feature",
        )

    @staticmethod
    def _candidate_domains(candidate: DatasetCandidate) -> frozenset[str]:
        """Every domain the candidate's own metadata supports.

        The topic domain is only a tag-level guess, so the dictionary concepts the
        candidate's own text resolves to are added: a botnet dataset tagged ``iot``
        whose description names "network traffic" and "botnet attacks" is a
        security dataset. Only text the candidate published is used, never the AI
        model's rationale.

        ``RELATED`` hits are excluded: they mean "some other concept lists this
        term as related", which is the reverse direction and floods the set with
        every loosely connected domain.

        The declared domain is lower-cased rather than passed through verbatim,
        because the project side is keyed by ``infer_domain_from_tags`` output,
        which is lower case. Comparing the two raw would let ``"Healthcare"`` and
        ``"healthcare"`` register as different subjects, which produced a
        contradiction on every well-matched candidate and a reason string reading
        "dataset is entirely about Healthcare while the project is about
        healthcare". Case is the only difference normalised away: the dictionary
        key form (``network_security``) has to survive, because that is what the
        concept index compares against.
        """
        domains: set[str] = set()
        if candidate.domain:
            domains.add(candidate.domain.strip().lower())
        for match in _keyword_service().match(candidate.evidence_text()):
            if match.domain and match.match_type is not MatchType.RELATED:
                domains.add(match.domain)
        return frozenset(domains)

    def _target_match(
        self,
        profile: ProjectProfile,
        evidence: _Evidence,
        candidate_domains: frozenset[str],
        publishes: bool = True,
    ) -> RankingFactor:
        """Whether the dataset's prediction target is the one the project needs.

        Resolution runs from strongest evidence to weakest - the project's own
        words, a dictionary equivalent, then the label spaces the target concepts
        declare - and every step is recorded in ``matched_via`` so a reader can
        see *why* a differently-named target counted. The target a model wrote
        verbatim is also tried, which recovers the label that a concise reduction
        dropped ("Normal vs Attack").
        """
        factor = RankingFactor(
            name=RankingFactorName.TARGET_MATCH,
            weight=self._weight(RankingFactorName.TARGET_MATCH),
        )
        if not profile.target_terms:
            factor.detail = "No target variable identified"
            return factor

        index = get_target_concept_index()
        # ``target_description`` is a published field on some registries and does
        # not exist on others; only a registry's own word makes a miss a real
        # absence.
        availability = DECLARED if publishes and evidence.field_tokens("target_description") else PROSE
        if not evidence.tokens:
            availability = NONE

        # A statement that the dataset has no such label is a statement about the
        # label, so it must not also be counted as evidence for it.
        declared_absence = _TARGET_NOT_PROVIDED.search(
            " ".join(
                part
                for part in (
                    evidence.raw.get("target_description"),
                    evidence.raw.get("description"),
                )
                if part
            )
        )
        if declared_absence:
            factor = RankingFactor(
                name=RankingFactorName.TARGET_MATCH,
                weight=self._weight(RankingFactorName.TARGET_MATCH),
                unknown_terms=list(profile.target_terms),
                detail=(
                    "the candidate's own metadata says the label is not provided and "
                    f"would have to be derived (\"{declared_absence.group(0).strip()}\"), "
                    "so the target cannot be confirmed"
                ),
            )
            factor.requirements = [
                RequirementMatch(
                    term=term,
                    status=MatchStatus.UNKNOWN,
                    weight=1.0,
                    detail="the metadata says the label is not provided",
                )
                for term in profile.target_terms
            ]
            return factor

        matched_via: list[str] = []

        def resolve_concept(term: str) -> Optional[tuple[str, MatchQuality]]:
            found = index.match(term, evidence.tokens, profile.project_domain_key, candidate_domains)
            if found is None:
                return None
            return (found.explain(), MatchQuality.SEMANTIC)

        verdicts = _assess_terms(
            profile.target_terms,
            evidence,
            _label_head_aliases(profile.target_aliases, profile.target_terms),
            availability=availability,
            resolve_concept=resolve_concept,
        )

        # The reduced target can be a topic while the label it names was dropped
        # ("network traffic" from "Network traffic event classification (Normal
        # vs Attack)"). Resolving the original wording against the label spaces
        # recovers those terms.
        remaining = [v for v in verdicts if v.status is not MatchStatus.MATCH]
        if remaining and profile.raw_target:
            raw_match = index.match(
                profile.raw_target,
                evidence.tokens,
                profile.project_domain_key,
                candidate_domains,
            )
            if raw_match is not None:
                explanation = f"project target as written: {raw_match.explain()}"
                for verdict in remaining:
                    matched_via.append(explanation)
                verdicts = [
                    _Verdict(v.term, MatchStatus.MATCH, MatchQuality.EXACT, v.weight, explanation)
                    if v.status is not MatchStatus.MATCH
                    else v
                    for v in verdicts
                ]

        # Every matched term carries the equivalence that matched it, so the factor
        # explains all of them - dictionary alias, label-head-free form, or label
        # space - rather than only the ones a resolver happened to record.
        matched_via = [v.via for v in verdicts if v.status is MatchStatus.MATCH and v.via]

        factor = _verdict_factor(
            RankingFactorName.TARGET_MATCH,
            self._weight(RankingFactorName.TARGET_MATCH),
            verdicts,
            dimension="target",
        )
        # ``matched_via`` accumulates across every resolution attempt, including
        # the ones that ultimately did not resolve a term, so it is filtered to the
        # terms that actually matched.
        factor.matched_via = matched_via[: len(factor.matched_terms)] or matched_via
        return factor

    def _task_match(
        self,
        profile: ProjectProfile,
        evidence: _Evidence,
        task_evidence: Optional[_Evidence],
    ) -> RankingFactor:
        """Whether the dataset supports the ML task the project requires.

        ``task_types`` is a published enumeration, so a candidate that declares
        "regression" while the project needs "classification" is a stated
        mismatch. A candidate with no declared tasks is judged only on its prose,
        where silence proves nothing.
        """
        factor = RankingFactor(
            name=RankingFactorName.TASK_MATCH,
            weight=self._weight(RankingFactorName.TASK_MATCH),
        )
        if not profile.task_terms:
            factor.detail = "No ML tasks identified"
            return factor

        if task_evidence is not None:
            verdicts = _assess_terms(
                profile.task_terms, task_evidence, None, availability=DECLARED
            )
        else:
            # No declared task list. The schema cue deliberately does not apply
            # here: a dataset that enumerates its columns states what it contains,
            # not which modelling task it supports - that is a property of the
            # user's project, not of the data.
            verdicts = _assess_terms(
                profile.task_terms, evidence, None, availability=PROSE
            )
        return _verdict_factor(
            RankingFactorName.TASK_MATCH,
            self._weight(RankingFactorName.TASK_MATCH),
            verdicts,
            dimension="task",
            basis="declared tasks" if task_evidence is not None else "dataset text",
        )

    def _compatibility(
        self,
        candidate: DatasetCandidate,
        profile: ProjectProfile,
        candidate_domains: frozenset[str],
    ) -> RankingFactor:
        """Penalty for metadata that positively contradicts the request.

        Scored on declared metadata only, so an AI suggestion is penalised only
        where a registry statement would have been. A candidate with nothing to
        contradict scores 1.0: absence of conflict is information, not evidence
        of compatibility.
        """
        contradictions: list[Contradiction] = find_contradictions(
            candidate,
            project_domain=profile.project_domain_key,
            project_tasks=profile.task_terms,
            project_targets=profile.target_labels,
            project_data_types=profile.data_type_notes,
            candidate_domains=candidate_domains,
            target_index=get_target_concept_index(),
        )
        penalty = penalty_total(contradictions)
        return RankingFactor(
            name=RankingFactorName.COMPATIBILITY,
            weight=self._weight(RankingFactorName.COMPATIBILITY),
            score=round(1.0 - penalty, 4),
            coverage=round(1.0 - penalty, 4),
            contradictions=[c.message for c in contradictions],
            detail=(
                "; ".join(c.message for c in contradictions)
                if contradictions
                else "no declared metadata contradicts the project requirements"
            ),
        )

    def _quality(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.QUALITY,
            weight=self._weight(RankingFactorName.QUALITY),
        )
        if candidate.usability is None:
            factor.detail = "No published quality score"
            return factor
        factor.score = candidate.usability
        factor.detail = f"publisher quality score {candidate.usability:.2f}"
        return factor

    def _popularity(self, candidate: DatasetCandidate) -> RankingFactor:
        """Adoption, as a blend of downloads and votes on a log scale.

        The blend divides by the *nominal* blend weight (0.7 + 0.3), never by the
        weight of the parts a candidate happens to publish. Dividing by the
        present parts instead would scale the download term by 1 / 0.7 = 1.43 for
        any dataset with no vote count, so a registry that omits votes would hand
        every listing a popularity bonus. An absent part forfeits its share and
        inflates nothing - the same rule the score denominator follows.
        """
        factor = RankingFactor(
            name=RankingFactorName.POPULARITY,
            weight=self._weight(RankingFactorName.POPULARITY),
        )
        parts: list[tuple[float, float]] = []
        if candidate.download_count is not None and candidate.download_count >= 0:
            parts.append(
                (
                    POPULARITY_DOWNLOAD_SHARE,
                    self._log_scale(
                        candidate.download_count, self._settings.ranking_download_reference
                    ),
                )
            )
        if candidate.vote_count is not None and candidate.vote_count >= 0:
            parts.append(
                (
                    POPULARITY_VOTE_SHARE,
                    self._log_scale(
                        candidate.vote_count, self._settings.ranking_vote_reference
                    ),
                )
            )
        if not parts:
            factor.detail = "No download or vote counts published"
            return factor

        score = sum(w * v for w, v in parts) / POPULARITY_BLEND_WEIGHT
        factor.score = round(min(score, 1.0), 4)
        factor.detail = f"{candidate.download_count} downloads, {candidate.vote_count} votes"
        return factor

    @staticmethod
    def _log_scale(value: int, reference: int) -> float:
        if reference <= 1:
            return 1.0 if value > 0 else 0.0
        return min(1.0, math.log10(1 + max(0, value)) / math.log10(1 + reference))

    def _recency(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.RECENCY,
            weight=self._weight(RankingFactorName.RECENCY),
        )
        if candidate.updated_at is None:
            factor.detail = "No last-updated date published"
            return factor

        now = self._now or datetime.now(timezone.utc)
        updated = candidate.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)

        age_days = (now - updated).total_seconds() / 86400.0
        max_age = max(1, self._settings.ranking_recency_max_age_days)
        if age_days <= 0:
            score = 1.0
        else:
            score = max(0.0, 1.0 - age_days / max_age)
        factor.score = round(min(score, 1.0), 4)
        factor.detail = f"last updated {updated.date().isoformat()}"
        return factor

    def _verification(self, candidate: DatasetCandidate) -> RankingFactor:
        factor = RankingFactor(
            name=RankingFactorName.VERIFICATION,
            weight=self._weight(RankingFactorName.VERIFICATION),
        )
        trust = (
            self._settings.ranking_trust_verified
            if candidate.source_type is SourceType.VERIFIED_EXTERNAL
            else self._settings.ranking_trust_unverified
        )
        # Defence in depth: an unverified record can never earn full trust even
        # if it claims an external source.
        if candidate.verification_status is VerificationStatus.UNVERIFIED:
            trust = min(trust, self._settings.ranking_trust_unverified)

        factor.score = max(0.0, min(1.0, trust))
        if candidate.verification_status is VerificationStatus.VERIFIED:
            factor.matched_terms = [candidate.source]
            factor.detail = f"listed by external source '{candidate.source}'"
        else:
            factor.matched_terms = [candidate.source]
            factor.detail = (
                f"suggested by '{candidate.source}' and not confirmed by an external registry"
            )
        return factor

    # -- explanations ------------------------------------------------------

    def _build_reasons(
        self, factors: list[RankingFactor], candidate: DatasetCandidate
    ) -> list[str]:
        """Build reasons strictly from what the factors actually found.

        Three kinds of line, in this order:

        * what was found - a factor only earns a reason when its score is above
          zero, so no reason can describe evidence that was not found;
        * what could not be confirmed - requirements the source never published
          metadata for, stated as an absence of information rather than a flaw in
          the dataset;
        * whether the listing is real - always last and never dropped by the cap,
          because telling the user a result is an unconfirmed AI suggestion
          matters more than an extra match detail.
        """
        reasons: list[str] = []
        for factor in factors:
            if factor.name is RankingFactorName.VERIFICATION:
                continue
            if factor.score is None or factor.score <= 0:
                continue
            reason = self._reason_for(factor, candidate)
            if reason:
                reasons.append(reason)

        unknown = self._unknown_note(factors)
        if unknown:
            # Sits inside the cap: a confirmation gap must not push out the
            # positive evidence that justifies the recommendation.
            reasons = reasons[: MAX_REASONS - 1] + [unknown]
        reasons = reasons[:MAX_REASONS]

        verification = next(
            (f for f in factors if f.name is RankingFactorName.VERIFICATION), None
        )
        if verification is not None and verification.score is not None:
            trust_reason = self._reason_for(verification, candidate)
            if trust_reason:
                reasons.append(trust_reason)
        return reasons

    @staticmethod
    def _unknown_note(factors: list[RankingFactor]) -> Optional[str]:
        """One line naming what could not be confirmed, if anything.

        Reported even when the candidate scored well: a requirement that was
        never judged is a different situation from one that was judged satisfied,
        and hiding it would overstate what the ranking knows.
        """
        unconfirmed: list[str] = []
        for factor in factors:
            if factor.name not in _REQUIREMENT_FACTORS or not factor.unknown_terms:
                continue
            dimension = factor.name.value.split("_")[0]
            unconfirmed.append(f"{DatasetRankingEngine._join(factor.unknown_terms)} ({dimension})")
        if not unconfirmed:
            return None
        return (
            "Could not confirm from published metadata: "
            + "; ".join(unconfirmed)
            + " - the source does not declare these, so they are neither "
            "matched nor ruled out"
        )

    @staticmethod
    def _join(terms: list[str], limit: int = 5) -> str:
        shown = terms[:limit]
        suffix = ", ..." if len(terms) > limit else ""
        return ", ".join(shown) + suffix

    def _reason_for(
        self, factor: RankingFactor, candidate: DatasetCandidate
    ) -> Optional[str]:
        name = factor.name
        terms = factor.matched_terms

        if name is RankingFactorName.DOMAIN_RELEVANCE:
            domain_terms = [t for t in terms if not t.startswith("domain:")]
            if any(t.startswith("domain:") for t in terms):
                return f"Published topic domain matches the project domain: {candidate.domain}"
            if domain_terms:
                return f"Dataset metadata matches project domain terms: {self._join(domain_terms)}"
            return None

        if name is RankingFactorName.KEYWORD_OVERLAP:
            if terms:
                return f"Dataset metadata matches project concepts: {self._join(terms)}"
            return None

        if name is RankingFactorName.FEATURE_MATCH:
            if terms:
                return f"Dataset metadata references required features: {self._join(terms)}"
            return None

        if name is RankingFactorName.TARGET_MATCH:
            if terms:
                return f"Dataset metadata references the prediction target: {self._join(terms)}"
            return None

        if name is RankingFactorName.TASK_MATCH:
            if terms:
                return f"Compatible with the required ML task(s): {self._join(terms)}"
            return None

        if name is RankingFactorName.COMPATIBILITY:
            if factor.contradictions:
                return f"Conflicts with the request: {self._join(factor.contradictions)}"
            return None

        if name is RankingFactorName.QUALITY:
            if candidate.usability is not None:
                return f"Publisher quality score {candidate.usability:.2f} / 1.00"
            return None

        if name is RankingFactorName.POPULARITY:
            bits = []
            if candidate.download_count:
                bits.append(f"{candidate.download_count:,} downloads")
            if candidate.vote_count:
                bits.append(f"{candidate.vote_count:,} votes")
            if not bits:
                bits.append("Published download and vote counts")
            return f"Adoption signal: {', '.join(bits)}"

        if name is RankingFactorName.RECENCY:
            if candidate.updated_at is None:
                return None
            return f"Last updated {candidate.updated_at.date().isoformat()}"

        if name is RankingFactorName.VERIFICATION:
            if candidate.verification_status is VerificationStatus.VERIFIED:
                return f"Verified listing from an external dataset registry ({candidate.source})"
            return (
                f"AI-suggested candidate from {candidate.source}; not yet confirmed "
                "by an external dataset registry"
            )

        return None
