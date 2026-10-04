from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from app.core.logging_config import get_logger
from app.schemas.project import (
    KeywordMatchEvidence,
    KeywordProvenance,
    MatchType,
)

logger = get_logger(__name__)

DictionaryPath = Path(__file__).resolve().parent.parent.parent / "data" / "keyword_dictionary.json"

# Dictionary phrases that cannot select a domain on their own.
#
# The dictionary lists "learning" as a synonym of the ``education`` concept, but
# "learning" is also the head noun of the machine-learning compound family
# ("machine learning", "deep learning", "supervised learning", "unsupervised
# learning", "transfer learning", ...), every one of which belongs to the
# ``artificial_intelligence`` domain. So the bare word is polysemous: it only
# picks up "education" when a recognised phrase in that same domain consumes it
# ("learning analytics", "learning outcome prediction"), and never when an
# artificial_intelligence compound does.
#
# This is an explicit controlled vocabulary rather than a derived rule on
# purpose. The generic absorption rule below (``_absorbed_single_word_matches``)
# already removes "learning" from "machine learning"; what needs the list is the
# *uncorroborated* case. Any structural heuristic wide enough to catch that case
# also throws away legitimate single-word concepts such as "detection",
# "forecasting" and "IoT", all of which are the only signal for several real
# project descriptions. Keeping the set small and documented keeps matching
# explainable: every dropped match is traceable to a named entry.
_CONTEXT_DEPENDENT_SINGLE_WORDS = frozenset({"learning"})

# Shortest token that may resolve a concept on its own.
#
# Two-letter abbreviations ("AI", "ML", "AT", "SA", "DF", ...) are real dictionary
# vocabulary, but in running text they collide with ordinary words: the
# abbreviation of "algorithmic trading" is ``AT``, which appears in "... admitted
# at hospital" and "... scores measured at the end of term". Any word-boundary
# match therefore turns a healthcare description into a finance one.
#
# The rule is structural rather than per-concept, so no concept has to be listed:
# a token this short is only allowed to name a concept when the surrounding text
# independently supports that concept's subject matter. Support means either the
# same concept is also present as a real phrase ("algorithmic trading" in the text
# confirms the ``AT`` in it), or another concept of the *same domain* matched. A
# bare short token with no same-domain context names nothing, so the false
# positive cannot arise; a short token in a text that really is about that domain
# keeps working.
#
# The threshold is two characters rather than three on purpose. "IoT" is the only
# signal for a whole class of project descriptions and is itself an abbreviation
# of "internet of things", so a three-letter gate would throw it away - the same
# mistake as making the gate purely structural. Two letters is where the collision
# with ordinary English actually starts.
_MAX_UNCORROBORATED_TOKEN_LEN = 2


def is_short_token(phrase: str) -> bool:
    """Whether a single-word phrase is too short to stand on its own."""
    words = phrase.split()
    return len(words) == 1 and len(words[0]) <= _MAX_UNCORROBORATED_TOKEN_LEN


class _NormalizedPhrase:
    __slots__ = ("original", "normalized")

    def __init__(self, original: str, normalized: str) -> None:
        self.original = original
        self.normalized = normalized


def _normalize(text: str) -> str:
    """Normalize text for matching.

    Policy:
    - Lowercase
    - Strip whitespace
    - Convert underscores to spaces (so 'iot_sensors' matches 'iot sensors')
    - Remove punctuation except word chars and spaces
    - Collapse multiple spaces
    """
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_unmatchable(normalized: str) -> bool:
    """Whether a dictionary phrase can never be matched on its own.

    A single character matches ordinary text by accident far too often ("a",
    "I"), so one-character entries are not indexed at all. Two- and
    three-character phrases stay indexed and are gated by corroboration at match
    time instead; see ``_MAX_UNCORROBORATED_TOKEN_LEN``.
    """
    return len(normalized) == 1


class KeywordService:
    """Deterministic keyword/concept matching engine.

    Loads the dictionary once and builds indexes for:
      - canonical term lookup (exact phrase)
      - synonym lookup
      - alias lookup
      - abbreviation lookup
      - related concept lookup
    """

    def __init__(self, dictionary_path: Optional[Path] = None) -> None:
        self._path = dictionary_path or DictionaryPath
        self._raw: dict = {}
        self._version: str = "unknown"

        self._canonical_index: dict[str, list[dict]] = {}
        self._synonym_index: dict[str, list[dict]] = {}
        self._alias_index: dict[str, list[dict]] = {}
        self._abbreviation_index: dict[str, list[dict]] = {}
        self._related_index: dict[str, list[dict]] = {}
        self._feature_index: dict[str, list[dict]] = {}
        self._all_phrases: list[_NormalizedPhrase] = []

        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            logger.error("Keyword dictionary not found at %s", self._path)
            raise FileNotFoundError(f"Dictionary not found: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            self._raw = json.load(f)

        meta = self._raw.get("metadata", {})
        self._version = meta.get("version", "unknown")
        logger.info("Loaded keyword dictionary v%s", self._version)

        domains = self._raw.get("domains", {})
        count = 0
        for domain_key, domain_data in domains.items():
            subdomains = domain_data.get("subdomains", {})
            for sub_key, sub_data in subdomains.items():
                for concept in sub_data.get("concepts", []):
                    concept["domain"] = domain_key
                    concept["subdomain"] = sub_key
                    self._index_concept(concept)
                    count += 1

        logger.info("Indexed %d concepts across %d domains", count, len(domains))

    def _index_concept(self, concept: dict) -> None:
        domain = concept["domain"]
        subdomain = concept["subdomain"]
        canonical = _normalize(concept["canonical_term"])
        entry = {
            "id": concept["id"],
            "canonical_term": concept["canonical_term"],
            "domain": domain,
            "subdomain": subdomain,
            "importance": concept.get("importance", 0.5),
            "potential_ml_tasks": concept.get("potential_ml_tasks", []),
            "dataset_requirements": concept.get("dataset_requirements", []),
            "feature_requirements": concept.get("feature_requirements", []),
            "feature_aliases": concept.get("feature_aliases", {}),
            "target_requirements": concept.get("target_requirements", []),
        }

        # A one-character concept name is never searchable. The guard sits here as
        # well as on synonyms and aliases so that no index can reintroduce it.
        if not _is_unmatchable(canonical):
            self._canonical_index.setdefault(canonical, []).append(entry)
            self._all_phrases.append(_NormalizedPhrase(concept["canonical_term"], canonical))

        for syn in concept.get("synonyms", []):
            normed = _normalize(syn)
            if normed and not _is_unmatchable(normed):
                self._synonym_index.setdefault(normed, []).append(
                    {**entry, "_matched_synonym": syn}
                )
                self._all_phrases.append(_NormalizedPhrase(syn, normed))

        for alias in concept.get("aliases", []):
            normed = _normalize(alias)
            if normed and not _is_unmatchable(normed):
                self._alias_index.setdefault(normed, []).append(
                    {**entry, "_matched_alias": alias}
                )
                self._all_phrases.append(_NormalizedPhrase(alias, normed))

        for abbr in concept.get("abbreviations", []):
            normed = abbr.lower().strip()
            if normed and not _is_unmatchable(normed):
                self._abbreviation_index.setdefault(normed, []).append(
                    {**entry, "_matched_abbr": abbr}
                )

        for related in concept.get("related_concepts", []):
            normed = _normalize(related)
            # Skip single-word related concepts (too broad, cause false positives)
            if normed and len(normed.split()) > 1:
                self._related_index.setdefault(normed, []).append(
                    {**entry, "_matched_related": related}
                )
                self._all_phrases.append(_NormalizedPhrase(related, normed))

        # Reverse index over declared features, so a bare feature name ("BMI",
        # "blood glucose") can be resolved to the concepts that require it even
        # when the name is not itself a concept, synonym or alias. This is what
        # makes "BMI" and "body mass index" known to be the same measurement
        # without either being written into the ranking code.
        for feature in concept.get("feature_requirements", []):
            normed = _normalize(feature)
            if normed and not _is_unmatchable(normed):
                self._feature_index.setdefault(normed, []).append(entry)

        # An alias of a feature is the same measurement under another name, so it
        # resolves to the same concepts from either direction.
        for feature, equivalents in (concept.get("feature_aliases") or {}).items():
            for equivalent in equivalents:
                normed = _normalize(equivalent)
                if normed and not _is_unmatchable(normed):
                    self._feature_index.setdefault(normed, []).append(entry)

    def concepts_declaring_feature(self, phrase: str) -> list[dict]:
        """Concepts whose declared ``feature_requirements`` include ``phrase``."""
        normalized = _normalize(phrase)
        if not normalized:
            return []
        return list(self._feature_index.get(normalized, []))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    def match(
        self, text: str, *, case_sensitive: bool = False
    ) -> list[KeywordMatchEvidence]:
        """Match text against the knowledge base.

        Returns de-duplicated, evidence-tracked matches sorted by importance.

        Matching strategy:
        - Multi-word phrases: substring matching (e.g., 'machine learning' in text)
        - Single-word phrases: word-boundary matching to prevent false positives
          (e.g., 'learning' won't match inside 'machine learning')
        - Post-filter: a match whose word was consumed by a longer phrase from a
          *different* domain is absorbed by it, and a context-dependent word with
          no same-domain phrase backing it selects no domain at all. See
          :meth:`_apply_context_rules`.
        """
        if not case_sensitive:
            text = _normalize(text)
        else:
            text = text.strip()

        matched: dict[str, KeywordMatchEvidence] = {}
        # Canonical term -> number of words in the phrase that produced the match.
        # Needed by the context rules to tell a phrase hit from a bare word hit.
        phrase_length: dict[str, int] = {}

        # Pre-compile single-word boundary patterns
        single_word_pattern_cache: dict[str, re.Pattern] = {}

        for phrase_obj in self._all_phrases:
            words = phrase_obj.normalized.split()
            is_single_word = len(words) == 1

            if is_single_word:
                # Word-boundary matching for single words
                pattern = single_word_pattern_cache.get(phrase_obj.normalized)
                if pattern is None:
                    pattern = re.compile(
                        r"\b" + re.escape(phrase_obj.normalized) + r"\b",
                        re.IGNORECASE if not case_sensitive else 0,
                    )
                    single_word_pattern_cache[phrase_obj.normalized] = pattern
                if not pattern.search(text):
                    continue
            else:
                # Substring matching for multi-word phrases
                if phrase_obj.normalized not in text:
                    continue

            concept_entries = (
                self._canonical_index.get(phrase_obj.normalized, [])
                + self._synonym_index.get(phrase_obj.normalized, [])
                + self._alias_index.get(phrase_obj.normalized, [])
                + self._related_index.get(phrase_obj.normalized, [])
            )
            for entry in concept_entries:
                canonical = entry["canonical_term"]
                if canonical not in matched:
                    matched[canonical] = KeywordMatchEvidence(
                        canonical_term=canonical,
                        matched_text=phrase_obj.original,
                        match_type=self._determine_match_type(
                            phrase_obj.normalized, entry
                        ),
                        dictionary_entry_id=entry["id"],
                        domain=entry["domain"],
                        subdomain=entry["subdomain"],
                    )
                    phrase_length[canonical] = len(words)
                elif self._is_stronger_evidence(
                    matched[canonical].matched_text, phrase_obj.normalized
                ):
                    # Several phrases resolved to the same concept. Keep the most
                    # specific evidence: a longer phrase first, then a word that
                    # is not context-dependent. Without this, "teaching and
                    # learning in schools" would only ever be evidenced by
                    # "learning", which the context rules then (correctly) treat
                    # as unable to select a domain.
                    matched[canonical] = KeywordMatchEvidence(
                        canonical_term=canonical,
                        matched_text=phrase_obj.original,
                        match_type=self._determine_match_type(
                            phrase_obj.normalized, entry
                        ),
                        dictionary_entry_id=entry["id"],
                        domain=entry["domain"],
                        subdomain=entry["subdomain"],
                    )
                    phrase_length[canonical] = len(words)

        abbrev_pattern = re.compile(
            r"\b("
            + "|".join(re.escape(a) for a in self._abbreviation_index.keys())
            + r")\b",
            re.IGNORECASE,
        )
        for m in abbrev_pattern.finditer(text if case_sensitive else text):
            abbr_key = m.group(0).lower()
            for entry in self._abbreviation_index.get(abbr_key, []):
                canonical = entry["canonical_term"]
                if canonical not in matched:
                    matched[canonical] = KeywordMatchEvidence(
                        canonical_term=canonical,
                        matched_text=m.group(0),
                        match_type=MatchType.ABBREVIATION,
                        dictionary_entry_id=entry["id"],
                        domain=entry["domain"],
                        subdomain=entry["subdomain"],
                    )
                    phrase_length[canonical] = 1

        return self._apply_context_rules(matched, phrase_length)

    @staticmethod
    def _is_stronger_evidence(current_phrase: str, candidate_phrase: str) -> bool:
        """Whether ``candidate_phrase`` evidences a concept better than the
        evidence already recorded for it.

        A longer phrase is always better. Among equally long phrases, a word
        that is not context-dependent wins: "teaching and learning in schools"
        resolves to the ``education`` concept through both words, and only
        "teaching" can carry the domain on its own.
        """
        current_words = current_phrase.split()
        candidate_words = candidate_phrase.split()
        if len(candidate_words) != len(current_words):
            return len(candidate_words) > len(current_words)
        return (
            current_words[0] in _CONTEXT_DEPENDENT_SINGLE_WORDS
            and candidate_words[0] not in _CONTEXT_DEPENDENT_SINGLE_WORDS
        )

    def _apply_context_rules(
        self,
        matched: dict[str, KeywordMatchEvidence],
        phrase_length: dict[str, int],
    ) -> list[KeywordMatchEvidence]:
        """Drop matches that the surrounding text does not support.

        Three rules, all applied to single-word matches only (a multi-word phrase
        is specific enough to stand on its own):

        1. **Absorption.** If another matched phrase of two or more words from a
           *different* domain contains the same word, that longer phrase owns the
           word. "machine learning" (artificial_intelligence) therefore absorbs
           the bare "learning" synonym of "education": the user wrote an ML
           compound, not an education statement.
        2. **Corroboration.** A word from
           :data:`_CONTEXT_DEPENDENT_SINGLE_WORDS` is only kept when a matched
           phrase in its *own* domain also consumes it. Bare "learning" on its
           own is polysemous and must not resolve the domain.
        3. **Short-token corroboration.** A token no longer than
           :data:`_MAX_UNCORROBORATED_TOKEN_LEN` ("AI", "AT", "SA", ...) is only
           kept when the text already supports the same concept or the same
           domain by other means. Without this, the abbreviation of
           "algorithmic trading" turns "admitted at hospital" into a finance
           statement.

        Exact, abbreviation and multi-word matches are never touched by rules 1-2,
        so a domain only moves when the text genuinely contains a competing
        phrase. Rule 3 is the one deliberate exception: a short token is by
        definition ambiguous in running text, whatever its match type.
        """
        if not matched:
            return []

        phrase_hits: list[tuple[KeywordMatchEvidence, list[str]]] = []
        for canonical, evidence in matched.items():
            tokens = _normalize(evidence.matched_text).split()
            if phrase_length.get(canonical, len(tokens)) > 1:
                phrase_hits.append((evidence, tokens))

        kept: dict[str, KeywordMatchEvidence] = {}
        for canonical, evidence in matched.items():
            if phrase_length.get(canonical, len(_normalize(evidence.matched_text).split())) > 1:
                kept[canonical] = evidence
                continue

            word_tokens = _normalize(evidence.matched_text).split()
            if not word_tokens:
                kept[canonical] = evidence
                continue
            word = word_tokens[0]

            foreign_owner = any(
                other.domain != evidence.domain and word in tokens
                for other, tokens in phrase_hits
            )
            if foreign_owner:
                # The word is part of a longer concept from another domain, so
                # it carries that concept's meaning, not this one.
                continue

            if word in _CONTEXT_DEPENDENT_SINGLE_WORDS and not any(
                other.domain == evidence.domain and word in tokens
                for other, tokens in phrase_hits
            ):
                # Polysemous and uncorroborated: it names no domain.
                continue

            if is_short_token(word) and not self._short_token_supported(
                evidence, matched, phrase_hits
            ):
                continue

            kept[canonical] = evidence

        return sorted(kept.values(), key=lambda e: e.canonical_term)

    @staticmethod
    def _short_token_supported(
        evidence: KeywordMatchEvidence,
        matched: dict[str, KeywordMatchEvidence],
        phrase_hits: list[tuple[KeywordMatchEvidence, list[str]]],
    ) -> bool:
        """Whether a short token has independent support in the same text.

        Two ways to be supported, both derived from the text itself:

        * the same concept also appears as a real phrase ("algorithmic trading"
          confirms that the "AT" inside it means algorithmic trading), or
        * another concept of the same domain matched, so the text is demonstrably
          about that subject matter.

        With neither, the token is treated as an ordinary word.
        """
        if any(other.canonical_term == evidence.canonical_term for other, _ in phrase_hits):
            return True
        return any(
            other.domain == evidence.domain and other.canonical_term != evidence.canonical_term
            for other in matched.values()
        )

    def resolve_canonical_phrase(
        self, text: str, *, min_tokens: int = 2
    ) -> Optional[str]:
        """Return the most specific canonical concept contained in ``text``.

        Only phrases of at least ``min_tokens`` words are considered, so a bare
        generic word ("student", "academic") is never mistaken for a target.
        When ``text`` is a verbose sentence with no qualifying dictionary phrase
        the method returns ``None``, letting callers drop it rather than inject
        its word salad into a query. This is what turns a model-written target
        such as "Student performance metrics and support requirement status"
        into the concise, already-known concept "student performance".
        """
        normalized = _normalize(text or "")
        if not normalized:
            return None

        best_phrase = ""
        best_len = 0
        for phrase_obj in self._all_phrases:
            phrase = phrase_obj.normalized
            word_count = len(phrase.split())
            if word_count < min_tokens or word_count <= best_len:
                continue
            if phrase in normalized:
                best_phrase = phrase
                best_len = word_count

        if not best_phrase:
            return None
        entries = (
            self._canonical_index.get(best_phrase, [])
            + self._synonym_index.get(best_phrase, [])
            + self._alias_index.get(best_phrase, [])
            + self._related_index.get(best_phrase, [])
        )
        if not entries:
            return None
        return entries[0]["canonical_term"]

    def get_concept(self, concept_id: str) -> Optional[dict]:
        domains = self._raw.get("domains", {})
        for domain_data in domains.values():
            for sub_data in domain_data.get("subdomains", {}).values():
                for concept in sub_data.get("concepts", []):
                    if concept["id"] == concept_id:
                        return concept
        return None

    def iter_concepts(self):
        """Yield every concept with its ``domain``/``subdomain`` attached.

        Read-only view over the same objects :meth:`get_concept` returns. It lets
        other services derive indexes (target label spaces, for example) from
        the dictionary instead of maintaining a second copy of the vocabulary.
        """
        for domain_data in self._raw.get("domains", {}).values():
            for sub_data in domain_data.get("subdomains", {}).values():
                for concept in sub_data.get("concepts", []):
                    yield concept

    def get_all_canonical_terms(self) -> list[str]:
        return sorted(
            set(e["canonical_term"] for entries in self._canonical_index.values() for e in entries)
        )

    def get_domain_keys(self) -> list[str]:
        """Return the dictionary's domain keys, e.g. 'education', 'healthcare'.

        Used by dataset discovery to map external topic tags onto DataPilot's
        own domain taxonomy without duplicating it. Additive and read-only.
        """
        return sorted(self._raw.get("domains", {}).keys())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _determine_match_type(self, normalized: str, entry: dict) -> MatchType:
        if normalized == _normalize(entry["canonical_term"]):
            return MatchType.EXACT
        if "_matched_synonym" in entry:
            return MatchType.SYNONYM
        if "_matched_alias" in entry:
            return MatchType.ALIAS
        if "_matched_related" in entry:
            return MatchType.RELATED
        return MatchType.PHRASE
