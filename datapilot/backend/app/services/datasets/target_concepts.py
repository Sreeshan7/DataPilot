"""Canonical prediction-target concepts, derived from the knowledge dictionary.

A project states its target in its own words ("malicious network activity") and a
dataset states the label column it actually contains ("malicious vs benign
traffic", "normal vs malicious", "attack label"). Literal term matching cannot
see that these name the same outcome, so the target factor scored 0 on obviously
relevant datasets.

Both sides are therefore reduced to a *label space*: the target vocabulary a
dictionary concept already declares in its ``target_requirements`` entry.

    cyber.net_foundational.network_security  ->  "benign malicious labels"
    cyber.network.intrusion_detection        ->  "attack benign labels"
    cyber.net_foundational.malicious_activity->  "attack labels"
    ai.ml.anomaly_detection                  ->  "binary labels normal anomaly"
    edu.foundational.student_performance     ->  "GPA", "pass fail", "grade"

Three rules keep the relaxation honest and generic:

* **Dictionary-backed only.** A label space exists because some concept declares
  it. Nothing is invented here and no per-project word list is maintained.
* **Shared-value equivalence.** Label spaces that name the same value ("attack
  benign" and "benign malicious") describe the same outcome, so they belong to
  one equivalence family. That is what lets "attack label" satisfy a project
  target of "malicious network activity", while "malware family" (which shares
  no value) stays separate.
* **Context is mandatory.** A value only counts when the candidate's own metadata
  places it in the label space's domain; failing that, the project's domain must.
  So "heart attack" in a healthcare dataset never satisfies a cybersecurity
  target, and a value with no domain evidence at all is only accepted when the
  candidate names two values of the space outright ("malicious vs benign
  traffic").

  The candidate's context is the union of its topic domain and the domains of the
  dictionary concepts its own text resolves to. A botnet dataset tagged ``iot``
  whose description names "network traffic" and "botnet attacks" is a security
  dataset, and a single topic tag is too thin to say otherwise.

The result is reported in the ranking factor's ``matched_via`` field, so a
non-zero target score always carries its justification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from app.core.logging_config import get_logger

logger = get_logger(__name__)

_PUNCT = re.compile(r"[^a-z0-9]+")

# Words that describe the *container* of a label column rather than its values.
# Dropping them turns "binary labels normal anomaly" into the value set
# {normal, anomaly}, which is what actually has to be matched.
_LABEL_NOISE = frozenset(
    {
        "attribute", "attributes", "binary", "category", "categories", "class",
        "classes", "code", "coding", "column", "columns", "field", "fields",
        "flag", "indicator", "indicators", "label", "labeled", "labeling",
        "labelled", "labels", "level", "levels", "measure", "measurement",
        "output", "reading", "readings", "score", "scores", "status", "target",
        "targets", "type", "types", "value", "values", "variable", "variables",
        # Quantifiers say *how* a quantity is expressed, not *which* quantity.
        # "attendance rate" and "hospital readmission rate" share only "rate",
        # and letting that one word resolve a label space made an exam-results
        # dataset satisfy a readmission target.
        "amount", "count", "number", "percent", "percentage", "proportion",
        "rate", "ratio", "total",
    }
)

# Grammatical filler that carries no label value.
_CONNECTORS = frozenset(
    {"a", "an", "and", "for", "not", "of", "or", "per", "the", "to", "versus", "vs", "with"}
)

# Endings after which a trailing "s" is part of the word rather than a plural
# marker: "malicious", "analysis", "bias", "series", "class".
_NON_PLURAL_S = ("ss", "us", "is", "as", "os")


def singularize(word: str) -> str:
    """Fold a regular English plural to its singular form.

    Only regular plurals are folded, and only when the trailing "s" is actually a
    plural marker. Words such as "malicious", "analysis" or "bias" end in "s"
    without being plurals, and folding them would corrupt the value used for
    matching (and its display form), so those endings are left alone.
    """
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(_NON_PLURAL_S):
        return word[:-1]
    return word


def normalize_tokens(text: Optional[str]) -> list[str]:
    """Lowercase, de-punctuate and singularize a phrase into value tokens."""
    if not text:
        return []
    cleaned = text.lower().replace("_", " ").replace("-", " ")
    cleaned = _PUNCT.sub(" ", cleaned)
    return [singularize(t) for t in cleaned.split() if t]


def value_tokens(text: Optional[str]) -> list[str]:
    """Label *values* of a phrase, with container words and filler removed."""
    return [
        token
        for token in normalize_tokens(text)
        if token not in _LABEL_NOISE and token not in _CONNECTORS
    ]


@dataclass(frozen=True)
class TargetConcept:
    """One label space declared by the dictionary."""

    key: str
    members: frozenset[str]
    domains: frozenset[str]
    concept_ids: tuple[str, ...]

    @property
    def label(self) -> str:
        """Human-readable form, e.g. ``benign + malicious``."""
        return " + ".join(sorted(self.members))


@dataclass(frozen=True)
class TargetMatch:
    """Why one project target term was found in a candidate's metadata."""

    term: str
    concept: TargetConcept
    matched_values: tuple[str, ...]
    equivalent: bool
    """True when the candidate matched a sibling label space, not the exact one
    the project term named."""
    declared: bool = False
    """True when the matched values come from the target vocabulary the
    *concepts behind the term* declare ("final grade" for a project target of
    "student performance"), rather than from a label space the term names
    directly. Weaker than a direct space hit, stronger than nothing."""

    def explain(self) -> str:
        values = ", ".join(self.matched_values)
        if self.declared:
            return (
                f"{self.term!r} names a concept declaring the target "
                f"{self.concept.label} (matched {values})"
            )
        if self.equivalent:
            return (
                f"{self.term!r} resolved to target concept "
                f"'{self.concept.label}' via shared label values ({values})"
            )
        return (
            f"{self.term!r} resolved to target concept "
            f"'{self.concept.label}' (matched {values})"
        )


@dataclass(frozen=True)
class TargetExpectation:
    """Everything a target term means, in one value vocabulary.

    ``spaces`` are the label spaces the term names outright ("pass/fail",
    "normal/anomaly"). ``declared`` are the target values that the *concepts*
    the term resolves to declare - what those concepts say the label column
    looks like. That second set is what lets "student academic performance" be
    satisfied by a dataset whose target is "final grade", and "churn" by one
    labelled "retained", without either term being listed anywhere in code.
    """

    spaces: tuple[TargetConcept, ...]
    declared: frozenset[str]
    concept_ids: tuple[str, ...]

    @property
    def values(self) -> frozenset[str]:
        """Every value a candidate may use to satisfy this target."""
        out: set[str] = set(self.declared)
        for space in self.spaces:
            out |= space.members
        return frozenset(out)

    def __bool__(self) -> bool:
        return bool(self.spaces or self.declared)


class TargetConceptIndex:
    """Dictionary-derived index of prediction-target label spaces.

    Built once per process from the same dictionary the rest of the system uses,
    so adding a concept automatically extends target matching.
    """

    def __init__(
        self,
        concepts: Iterable[dict],
        service: Optional[object] = None,
    ) -> None:
        self._concepts: dict[str, TargetConcept] = {}
        self._family: dict[str, frozenset[str]] = {}
        # concept id -> the target values that concept declares
        self._declared: dict[str, frozenset[str]] = {}
        self._service = service
        self._build(concepts)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self, concepts: Iterable[dict]) -> None:
        members: dict[frozenset[str], set[str]] = {}
        domains: dict[frozenset[str], set[str]] = {}
        ids: dict[frozenset[str], list[str]] = {}
        declared: dict[str, set[str]] = {}

        for concept in concepts:
            domain = concept.get("domain") or ""
            concept_id = concept.get("id")
            values_for_concept: set[str] = set()
            for requirement in concept.get("target_requirements", []) or []:
                values = frozenset(value_tokens(requirement))
                values_for_concept |= values
                if not values:
                    continue
                members.setdefault(values, set())
                domains.setdefault(values, set())
                ids.setdefault(values, [])
                if domain:
                    domains[values].add(domain)
                if concept_id and concept_id not in ids[values]:
                    ids[values].append(concept_id)
            if concept_id and values_for_concept:
                declared.setdefault(concept_id, set()).update(values_for_concept)

        for values in members:
            self._concepts[self._key(values)] = TargetConcept(
                key=self._key(values),
                members=values,
                domains=frozenset(domains[values]),
                concept_ids=tuple(ids[values]),
            )
        self._declared = {k: frozenset(v) for k, v in declared.items()}
        self._build_families(members)
        logger.debug(
            "Built %d target label spaces in %d equivalence families",
            len(self._concepts),
            len({frozenset(f) for f in self._family.values()}),
        )

    def _build_families(self, members: dict[frozenset[str], set[str]]) -> None:
        """Union label spaces that share at least one label value.

        "attack benign" and "benign malicious" both name ``benign`` and therefore
        describe the same binary outcome; they are one concept as far as matching
        is concerned. The transitive closure is computed once, up front.
        """
        parent: dict[str, str] = {self._key(v): self._key(v) for v in members}

        def find(key: str) -> str:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                # Merge deterministically so the families do not depend on
                # dictionary iteration order.
                low, high = sorted((ra, rb))
                parent[high] = low

        by_value: dict[str, list[str]] = {}
        for values in members:
            key = self._key(values)
            for value in sorted(values):
                by_value.setdefault(value, []).append(key)
        for keys in by_value.values():
            for other in keys[1:]:
                union(keys[0], other)

        grouped: dict[str, list[str]] = {}
        for key in parent:
            grouped.setdefault(find(key), []).append(key)
        for keys in grouped.values():
            family = frozenset(keys)
            for key in keys:
                self._family[key] = family

    @staticmethod
    def _key(values: frozenset[str]) -> str:
        return " ".join(sorted(values))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._concepts)

    @property
    def concepts(self) -> dict[str, TargetConcept]:
        return dict(self._concepts)

    def resolve(self, term: Optional[str]) -> list[TargetConcept]:
        """Label spaces the project target term refers to, most specific first."""
        values = frozenset(value_tokens(term))
        if not values:
            return []
        hits = [
            concept
            for concept in self._concepts.values()
            if values & concept.members
        ]
        # A term naming several values of one space is stronger evidence than one
        # naming a single value, so order by overlap and then by space size.
        hits.sort(key=lambda c: (-len(values & c.members), -len(c.members), c.key))
        return hits

    def expectation(self, term: Optional[str]) -> TargetExpectation:
        """Every label space and declared target value ``term`` can mean.

        Two sources, both dictionary-derived:

        * label spaces the term names by value ("normal/anomaly", "pass/fail");
        * the target values declared by the concepts the term *is*
          ("student performance" declares "GPA", "pass fail", "grade"), which is
          what lets a dataset labelled "final grade" satisfy a project target of
          "student academic performance".

        Concept resolution goes through the normal matcher, so it is the same
        vocabulary the rest of the system uses rather than a second word list.
        """
        spaces = tuple(self.resolve(term))
        concept_ids = tuple(
            sorted({cid for space in spaces for cid in space.concept_ids})
        )
        declared: set[str] = set()
        if self._service is not None:
            for match in self._service.match(term or ""):  # type: ignore[attr-defined]
                if match.domain is None:
                    continue
                declared |= self._declared.get(match.dictionary_entry_id, frozenset())
        return TargetExpectation(
            spaces=spaces,
            declared=frozenset(declared),
            concept_ids=concept_ids,
        )

    def family(self, concept: TargetConcept) -> frozenset[TargetConcept]:
        """The concept plus every label space sharing one of its values."""
        keys = self._family.get(concept.key, frozenset({concept.key}))
        return frozenset(self._concepts[k] for k in keys if k in self._concepts)

    def match(
        self,
        term: str,
        evidence_tokens: frozenset[str] | set[str],
        project_domain: Optional[str],
        candidate_domains: Optional[Iterable[str]] = None,
    ) -> Optional[TargetMatch]:
        """Find a label space in which ``term`` and ``evidence_tokens`` overlap.

        ``candidate_domains`` is every domain the candidate's *own* metadata
        supports (its topic domain plus the domains of the dictionary concepts its
        text resolves to). Supporting at least one of them is what licenses the
        equivalence.

        Three kinds of hit, strongest first:

        1. a label space the term names outright, matched on its values;
        2. a sibling space sharing a value ("attack" for "malicious");
        3. a value the term's *concepts* declare ("grade" for a project target of
           "student performance"), reported as ``declared``.

        Returns ``None`` when nothing matches or when the surrounding domain does
        not support the equivalence. The best-supported space wins: the space the
        project actually named, then the one naming the most label values, then
        the largest space. The order is total and independent of set iteration
        order, so the same evidence always yields the same justification.
        """
        values = frozenset(value_tokens(term))
        if not values or not evidence_tokens:
            return None
        domains = frozenset(candidate_domains or ())
        expectation = self.expectation(term)

        # (is_declared, is_equivalent, -matched_values, -space_size, key) — a
        # fully specified sort key, because a frozenset of sibling spaces has no
        # stable order.
        ranked: list[
            tuple[tuple[bool, bool, int, int, str], TargetConcept, tuple[str, ...]]
        ] = []
        for concept in expectation.spaces:
            for sibling in self.family(concept):
                equivalent = sibling.key != concept.key
                hit = sibling.members & evidence_tokens
                if not hit:
                    continue
                if not self._domain_supports(sibling, hit, project_domain, domains):
                    continue
                key = (
                    False,
                    equivalent,
                    -len(hit),
                    -len(sibling.members),
                    sibling.key,
                )
                ranked.append((key, sibling, tuple(sorted(hit))))

        if ranked:
            key, concept, matched = min(ranked, key=lambda item: item[0])
            return TargetMatch(
                term=term,
                concept=concept,
                matched_values=matched,
                equivalent=key[1],
                declared=False,
            )

        # Fallback: the value vocabulary the term's concepts declare.
        if expectation.declared:
            hit = expectation.declared & evidence_tokens
            # Two values are self-evidencing ("pass and fail"); one is only
            # accepted when the candidate's subject matter agrees, or when the
            # project and candidate are in the same domain.
            if hit and (len(hit) >= 2 or not project_domain or project_domain in domains):
                label = ", ".join(sorted(hit))
                return TargetMatch(
                    term=term,
                    concept=TargetConcept(
                        key=f"declared:{label}",
                        members=frozenset(hit),
                        domains=frozenset(domains),
                        concept_ids=expectation.concept_ids,
                    ),
                    matched_values=tuple(sorted(hit)),
                    equivalent=False,
                    declared=True,
                )
        return None

    @staticmethod
    def _domain_supports(
        concept: TargetConcept,
        hit: frozenset[str],
        project_domain: Optional[str],
        candidate_domains: frozenset[str],
    ) -> bool:
        """Whether the candidate's own subject matter licenses reading ``hit``.

        The candidate's metadata decides, because it describes what the dataset is
        actually about; the project's domain is only the fallback for a candidate
        that carries no domain signal at all. With neither, the candidate must name
        two values of the space, which is self-evidencing.
        """
        if candidate_domains:
            return bool(candidate_domains & concept.domains)
        if project_domain:
            return project_domain in concept.domains
        return len(hit) >= 2


# The index is derived from a read-only dictionary file, so one instance per
# process is enough and keeps ranking free of repeated parsing.
_INDEX: Optional[TargetConceptIndex] = None


def get_target_concept_index() -> TargetConceptIndex:
    global _INDEX
    if _INDEX is None:
        from app.services.keyword_service import KeywordService

        service = KeywordService()
        _INDEX = TargetConceptIndex(service.iter_concepts(), service=service)
    return _INDEX


def reset_target_concept_index() -> None:
    """Test hook: clears the memoized index."""
    global _INDEX
    _INDEX = None


def explain_matches(matches: Sequence[TargetMatch]) -> list[str]:
    return [match.explain() for match in matches]