"""Phase 3A models: what we know about a dataset, and how well it supports a project.

Two representations, deliberately separated:

``DatasetCandidate``
    "What dataset is this?" Identity, published metadata, ranking output.

``DatasetIntelligence``
    "What do we know about it, and how well does it support *this* project?"
    Schema state, target state, feature compatibility, risks, limitations and a
    structured explanation.

The separation matters because they answer different questions. Ranking says
*how highly to recommend this*; compatibility says *whether it can actually
support the project*. Merging them would let a popular, unverifiable dataset
outrank a less famous but fully verified one, and would make "unknown" and
"absent" indistinguishable at the point where a user decides whether to trust a
recommendation.

Trust rules encoded here, not merely documented:

* ``AI_SUGGESTED`` never becomes ``SCHEMA_VERIFIED``. A model's guess that a
  dataset "contains BMI" is a guess about the dataset, published by nobody.
* ``INFERRED`` never becomes ``SOURCE_DECLARED``. DataPilot's own derivations
  are its reasoning, not the registry's statement.
* ``UNKNOWN`` never becomes ``MATCH``, and never becomes ``MISSING``. Absence of
  evidence is not evidence of absence: Kaggle has no ``target_description`` or
  ``feature_description`` field at all.
* ``MISSING`` requires trustworthy schema information. You can only declare a
  feature absent against a schema you actually verified.
* ``TARGET_PRESENT`` and ``TARGET_DERIVABLE`` both require declared evidence.
  Gemini speculation is not sufficient for either.

These are enforced by ``model_validator`` checks below, so a caller cannot
construct an intelligence record that violates them, and by the builder in
``app.services.datasets.intelligence``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.dataset import MatchQuality, MatchStatus, ProjectAnalysisData

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class EvidenceLayer(str, Enum):
    """How a single claim about a dataset came to be known, strongest first.

    ``SOURCE_DECLARED`` - published by the external registry the candidate came
    from. Somebody stated it on purpose.
    ``SCHEMA_VERIFIED`` - read from a schema DataPilot itself verified against
    the dataset. In Phase 3A this is a *contract*: no code path populates it yet,
    because verifying a schema means downloading and parsing the file, which is
    Phase 3B. It exists now so the ordering is fixed before anything can fill it.
    ``INFERRED`` - derived by DataPilot from declared metadata (a tag mapped onto
    a dictionary domain, an alias expansion). Real reasoning, published by nobody.
    ``AI_SUGGESTED`` - produced by a generative model. May be correct; carries no
    external evidence.
    ``UNKNOWN`` - nobody supplied anything, so nothing can be claimed.
    """

    SOURCE_DECLARED = "source_declared"
    SCHEMA_VERIFIED = "schema_verified"
    INFERRED = "inferred"
    AI_SUGGESTED = "ai_suggested"
    UNKNOWN = "unknown"


#: Trust order, strongest first. Used to pick a single layer for a record and to
#: answer "is this claim at least as strong as X?" without hand-written chains.
LAYER_STRENGTH: dict[EvidenceLayer, int] = {
    EvidenceLayer.SOURCE_DECLARED: 4,
    EvidenceLayer.SCHEMA_VERIFIED: 3,
    EvidenceLayer.INFERRED: 2,
    EvidenceLayer.AI_SUGGESTED: 1,
    EvidenceLayer.UNKNOWN: 0,
}

#: Layers strong enough to establish a fact about the dataset rather than a guess.
DECLARING_LAYERS: frozenset[EvidenceLayer] = frozenset(
    {EvidenceLayer.SOURCE_DECLARED, EvidenceLayer.SCHEMA_VERIFIED}
)


def strongest_layer(layers: "list[EvidenceLayer] | tuple[EvidenceLayer, ...]") -> EvidenceLayer:
    """The most authoritative layer present, or ``UNKNOWN`` when there is none."""
    if not layers:
        return EvidenceLayer.UNKNOWN
    return max(layers, key=lambda layer: LAYER_STRENGTH[layer])


def layer_at_least(layer: EvidenceLayer, floor: EvidenceLayer) -> bool:
    """Whether ``layer`` carries at least as much authority as ``floor``."""
    return LAYER_STRENGTH[layer] >= LAYER_STRENGTH[floor]


class EvidenceClaim(BaseModel):
    """One traceable statement about a dataset, with where it came from.

    ``value`` is recorded verbatim as the system holds it. It is never
    paraphrased into a stronger statement, which is what keeps
    "AI-suggested: contains BMI" from being rendered as "contains BMI".
    """

    subject: str = Field(..., description="What the claim is about, e.g. 'feature:age'.")
    value: Optional[str] = Field(
        None, description="The value as published or inferred; None when nothing was supplied."
    )
    layer: EvidenceLayer
    source: str = Field(
        ..., description="Registry name, 'gemini', or 'datapilot' for our own derivations."
    )

    @model_validator(mode="after")
    def unknown_claims_cannot_carry_a_value(self) -> "EvidenceClaim":
        """A claim with no evidence cannot assert anything.

        Without this, ``EvidenceClaim(layer=UNKNOWN, value="contains BMI")``
        would be constructible and would carry a fabricated fact through the
        explanation layer.
        """
        if self.layer is EvidenceLayer.UNKNOWN and self.value:
            raise ValueError(
                "an UNKNOWN claim cannot carry a value: nothing supplied it"
            )
        return self


# ---------------------------------------------------------------------------
# Schema state
# ---------------------------------------------------------------------------


class SchemaState(str, Enum):
    """How much trustworthy schema information exists for this dataset.

    This is a state, not a schema. An empty column list is not "no schema" and
    must never be presented as one.
    """

    VERIFIED = "verified"
    """Sufficient trustworthy schema information exists to decide feature-level
    questions."""

    PARTIAL = "partial"
    """Some trustworthy schema information exists, but not enough to conclude
    about every required feature."""

    UNKNOWN = "unknown"
    """No trustworthy schema information exists. Absence of evidence."""


#: States in which a required feature may legitimately be declared absent.
SCHEMA_STATES_PROVING_ABSENCE: frozenset[SchemaState] = frozenset(
    {SchemaState.VERIFIED, SchemaState.PARTIAL}
)


# ---------------------------------------------------------------------------
# Target state
# ---------------------------------------------------------------------------


class TargetState(str, Enum):
    """Whether the dataset can support the project's prediction target."""

    PRESENT = "present"
    """Declared evidence indicates the dataset contains a compatible target."""

    DERIVABLE = "derivable"
    """There is actual evidence the target can be constructed - e.g. the dataset
    states the label is absent but names the fields it would be derived from."""

    UNKNOWN = "unknown"
    """Insufficient evidence. Not a claim that the target is missing."""

    INCOMPATIBLE = "incompatible"
    """Evidence indicates the dataset cannot support the requested target."""


# ---------------------------------------------------------------------------
# Feature state
# ---------------------------------------------------------------------------


class FeatureMatchState(str, Enum):
    """How one required project feature stands against this dataset.

    The four positive states mirror ``MatchQuality`` exactly, so a feature that
    scored ``SEMANTIC`` for ranking reads as ``SEMANTIC`` here. ``UNKNOWN`` and
    ``MISSING`` are separate because collapsing them tells a user that a
    dataset lacks a feature when the truth is that nobody published whether it
    has one.
    """

    EXACT = "exact"
    ALIAS = "alias"
    SEMANTIC = "semantic"
    INFERRED = "inferred"
    UNKNOWN = "unknown"
    """Schema unavailable, so the feature can be neither confirmed nor denied."""

    MISSING = "missing"
    """Schema verified and the feature is absent."""


#: The four states in which the dataset is known to provide the feature.
FEATURE_STATES_SATISFIED: frozenset[FeatureMatchState] = frozenset(
    {
        FeatureMatchState.EXACT,
        FeatureMatchState.ALIAS,
        FeatureMatchState.SEMANTIC,
        FeatureMatchState.INFERRED,
    }
)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


class CompatibilityLevel(str, Enum):
    """Categorical summary of overall fit.

    Deliberately categorical. A percentage would imply a precision the evidence
    does not support: with three of six required features unverifiable, "50%
    compatible" and "83% compatible" are the same dataset, and the difference
    is entirely about how much nobody knows.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"
    """Not enough evidence to place it at all. Not the same as LOW."""


class DimensionCompatibility(BaseModel):
    """Verdict for one project dimension, reusing the ranking MATCH/UNKNOWN/MISMATCH."""

    dimension: str
    status: MatchStatus = Field(
        ..., description="Reused from MatchStatus so ranking and compatibility cannot drift apart."
    )
    evidence: str = Field(
        ..., description="Why this status, phrased so it names what was and was not seen."
    )
    layer: EvidenceLayer = EvidenceLayer.UNKNOWN


class FeatureCompatibility(BaseModel):
    """One required project feature against one dataset."""

    requirement: str
    state: FeatureMatchState
    layer: EvidenceLayer
    via: Optional[str] = Field(
        None, description="The equivalent phrase or field that satisfied it, when not literal."
    )
    detail: str = ""

    @model_validator(mode="after")
    def state_must_agree_with_its_evidence(self) -> "FeatureCompatibility":
        """Guard: the state cannot claim more knowledge than the layer supports.

        Two directions, both of which are ways of lying about a feature:

        * ``MISSING`` asserts absence, which needs trustworthy schema. A model
          guess or no evidence at all cannot support it.
        * ``UNKNOWN`` asserts that nobody knows. Verified schema is somebody
          knowing, so pairing them is incoherent in the other direction - it is
          how a real answer gets filed as a shrug.
        """
        if self.state is FeatureMatchState.MISSING and self.layer in {
            EvidenceLayer.AI_SUGGESTED,
            EvidenceLayer.UNKNOWN,
        }:
            raise ValueError(
                f"feature {self.requirement!r} cannot be MISSING on "
                f"{self.layer.value} evidence: absence needs trustworthy schema"
            )
        if self.state is FeatureMatchState.UNKNOWN and self.layer in {
            EvidenceLayer.SCHEMA_VERIFIED,
            EvidenceLayer.SOURCE_DECLARED,
        }:
            raise ValueError(
                f"feature {self.requirement!r} cannot be UNKNOWN on "
                f"{self.layer.value} evidence: that evidence is a positive answer"
            )
        return self


class FeatureCoverage(BaseModel):
    """Required features partitioned by what we actually know.

    ``matched + unknown + missing == required_count``, and the three lists are
    never merged into a single ratio. Reporting "4/6" would silently charge the
    unknown feature as if it were missing.
    """

    required_count: int = Field(..., ge=0)
    matched: list[FeatureCompatibility] = Field(default_factory=list)
    unknown: list[FeatureCompatibility] = Field(default_factory=list)
    missing: list[FeatureCompatibility] = Field(default_factory=list)

    @model_validator(mode="after")
    def buckets_partition_the_requirements(self) -> "FeatureCoverage":
        """Guard: no requirement may appear twice, vanish, or land in two buckets."""
        total = len(self.matched) + len(self.unknown) + len(self.missing)
        if total != self.required_count:
            raise ValueError(
                f"coverage lists hold {total} features but required_count is "
                f"{self.required_count}"
            )
        names = [f.requirement for f in (*self.matched, *self.unknown, *self.missing)]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"feature appears in more than one bucket: {duplicates}")
        return self

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def unknown_count(self) -> int:
        return len(self.unknown)

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def all_features(self) -> list[FeatureCompatibility]:
        """Every required feature, whatever we know about it.

        Ordered matched, then unknown, then missing: the order the buckets are
        declared in, so callers and serialized output are both stable.
        """
        return [*self.matched, *self.unknown, *self.missing]


class TargetAnalysis(BaseModel):
    """The project's target against this dataset, with the evidence that decided it."""

    requirement: Optional[str] = Field(
        None, description="The project's target; None when the project named none."
    )
    state: TargetState
    layer: EvidenceLayer
    evidence: str = ""
    detail: str = ""
    match_quality: Optional[MatchQuality] = Field(
        None,
        description=(
            "How a PRESENT target was matched. EXACT/ALIAS name the target; SEMANTIC "
            "reached it through a dictionary equivalent, so it is reported as "
            "inferred rather than verified."
        ),
    )

    @model_validator(mode="after")
    def positive_claims_need_declared_evidence(self) -> "TargetAnalysis":
        """Guard: neither PRESENT nor DERIVABLE may rest on a model or a guess.

        This is the rule that stops "Gemini says the dataset has a churn label"
        from being reported as "target is present".
        """
        if self.state in {TargetState.PRESENT, TargetState.DERIVABLE} and (
            self.layer not in DECLARING_LAYERS
        ):
            raise ValueError(
                f"target cannot be {self.state.value} on {self.layer.value} "
                f"evidence: {self.state.value} claims need declared evidence"
            )
        # INCOMPATIBLE is the mirror of PRESENT and is guarded for the same
        # reason: it is a positive claim that this dataset cannot serve the
        # project, and a model guessing it is as wrong as a model guessing the
        # opposite. Without this, "Gemini thinks these are different targets"
        # would render as a stated disagreement.
        if self.state is TargetState.INCOMPATIBLE and (
            self.layer not in DECLARING_LAYERS
        ):
            raise ValueError(
                "an INCOMPATIBLE target needs declared evidence, not speculation"
            )
        return self


class CompatibilityAssessment(BaseModel):
    """Can this dataset support the project? Kept separate from ranking."""

    overall: CompatibilityLevel
    domain: MatchStatus
    task: MatchStatus
    target: MatchStatus
    data_type: MatchStatus = MatchStatus.UNKNOWN
    feature_coverage: FeatureCoverage
    contradictions: list[str] = Field(
        default_factory=list,
        description="Project requirements the candidate's own metadata positively contradicts.",
    )
    unknown_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions that could not be decided, named explicitly.",
    )


# ---------------------------------------------------------------------------
# Risks and limitations
# ---------------------------------------------------------------------------


class RiskSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    """Structured risks.

    Only the first eleven categories are emitted in Phase 3A, and only where
    existing metadata supports the fact. The remaining five require reading the
    dataset itself - counting classes, measuring null rates, comparing rows,
    looking for an identifier column - which is Phase 3B raw-dataset profiling.
    They are declared here so the contract is fixed before anything fills them,
    and ``EMITTABLE_RISK_CATEGORIES`` keeps them out of Phase 3A output.
    """

    # -- emitted in Phase 3A
    TARGET_UNKNOWN = "target_unknown"
    TARGET_MISSING = "target_missing"
    TARGET_DERIVATION_REQUIRED = "target_derivation_required"
    FEATURE_UNKNOWN = "feature_unknown"
    FEATURE_MISSING = "feature_missing"
    SCHEMA_INCOMPLETE = "schema_incomplete"
    AI_UNVERIFIED = "ai_unverified"
    DOMAIN_MISMATCH = "domain_mismatch"
    TASK_MISMATCH = "task_mismatch"
    TARGET_MISMATCH = "target_mismatch"

    # -- Phase 3B only. Never emitted before raw profiling exists.
    POSSIBLE_LEAKAGE = "possible_leakage"
    CLASS_IMBALANCE = "class_imbalance"
    HIGH_MISSINGNESS = "high_missingness"
    DUPLICATES = "duplicates"
    IDENTIFIER_COLUMN = "identifier_column"
    SMALL_DATASET = "small_dataset"


#: The risks Phase 3A may emit. Anything outside this set needs raw data.
EMITTABLE_RISK_CATEGORIES: frozenset[RiskCategory] = frozenset(
    {
        RiskCategory.TARGET_UNKNOWN,
        RiskCategory.TARGET_MISSING,
        RiskCategory.TARGET_DERIVATION_REQUIRED,
        RiskCategory.FEATURE_UNKNOWN,
        RiskCategory.FEATURE_MISSING,
        RiskCategory.SCHEMA_INCOMPLETE,
        RiskCategory.AI_UNVERIFIED,
        RiskCategory.DOMAIN_MISMATCH,
        RiskCategory.TASK_MISMATCH,
        RiskCategory.TARGET_MISMATCH,
    }
)


def _require_text(value: str, what: str) -> str:
    """A risk or limitation with no stated reason is an unexplained warning."""
    if not value or not value.strip():
        raise ValueError(f"{what} must not be blank")
    return value


class DatasetRisk(BaseModel):
    """One risk, with the evidence that raised it."""

    category: RiskCategory
    severity: RiskSeverity
    detail: str
    subject: Optional[str] = Field(
        None, description="The specific requirement or field the risk is about."
    )
    layer: EvidenceLayer = EvidenceLayer.UNKNOWN

    @field_validator("detail")
    @classmethod
    def detail_must_state_the_evidence(cls, value: str) -> str:
        return _require_text(value, "risk detail")


class DatasetLimitation(BaseModel):
    """A caveat a user should know before relying on this dataset."""

    code: str = Field(
        ..., description="Stable machine-readable code, e.g. 'schema_not_published'."
    )
    detail: str
    layer: EvidenceLayer = EvidenceLayer.UNKNOWN

    @field_validator("code", "detail")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        return _require_text(value, "limitation field")


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


class RecommendationExplanation(BaseModel):
    """Structured explanation, ready for a renderer or an LLM to phrase.

    Every list is derived from typed state by
    ``build_recommendation_explanation``; nothing here is free text invented
    about the dataset. Section 13's contract is that Gemini receives *this* and
    turns it into language, rather than being asked what the dataset contains.
    """

    why_recommended: list[str] = Field(default_factory=list)
    what_matches: list[str] = Field(default_factory=list)
    what_is_unknown: list[str] = Field(default_factory=list)
    what_is_missing: list[str] = Field(default_factory=list)
    what_is_verified: list[str] = Field(default_factory=list)
    what_is_inferred: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_section_may_contain_blanks(self) -> "RecommendationExplanation":
        """Guard: a blank line in an explanation reads as a missing fact."""
        for name in type(self).model_fields:
            for entry in getattr(self, name):
                if not entry or not entry.strip():
                    raise ValueError(f"explanation section {name!r} contains a blank entry")
        return self


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


class DatasetIntelligence(BaseModel):
    """What we know about one candidate, and how well it supports the project.

    Identity is referenced by ``candidate_id``, not copied: the
    ``DatasetCandidate`` already holds it, and a second copy is a second thing
    that can drift.
    """

    candidate_id: str = Field(..., description="Matches DatasetCandidate.id.")
    evidence_layer: EvidenceLayer = Field(
        ..., description="Strongest evidence layer available anywhere on this candidate."
    )
    evidence: list[EvidenceClaim] = Field(default_factory=list)
    schema_state: SchemaState
    target_analysis: TargetAnalysis
    feature_coverage: FeatureCoverage
    compatibility: CompatibilityAssessment
    risks: list[DatasetRisk] = Field(default_factory=list)
    limitations: list[DatasetLimitation] = Field(default_factory=list)
    explanation: RecommendationExplanation = Field(
        default_factory=RecommendationExplanation
    )

    @model_validator(mode="after")
    def whole_record_is_internally_consistent(self) -> "DatasetIntelligence":
        """Cross-field guards the individual models cannot see.

        Each rule here corresponds to a failure that would otherwise be
        expressible: reporting an absent feature against a dataset whose schema
        nobody could see, claiming a verified schema with no verified claim to
        back it, or letting an unverified AI candidate pass for a registry
        result.
        """

        # 1. MISSING requires a schema state that can prove absence. Without
        #    this, "feature absent" could be reported against a dataset whose
        #    schema nobody has ever seen - which is UNKNOWN wearing a hat.
        if self.feature_coverage.missing and (
            self.schema_state not in SCHEMA_STATES_PROVING_ABSENCE
        ):
            raise ValueError(
                f"cannot report missing features against schema_state="
                f"{self.schema_state.value}"
            )

        # 2. VERIFIED schema needs at least one verified-schema claim. Otherwise
        #    the strongest available evidence is a declared description, which is
        #    PARTIAL at best.
        if self.schema_state is SchemaState.VERIFIED and not any(
            claim.layer is EvidenceLayer.SCHEMA_VERIFIED
            for claim in self.evidence
        ):
            raise ValueError(
                "schema_state=verified requires at least one SCHEMA_VERIFIED claim"
            )

        # 3. A verified schema claim implies a verified schema state. The
        #    converse direction of rule 2.
        if any(claim.layer is EvidenceLayer.SCHEMA_VERIFIED for claim in self.evidence):
            if self.schema_state is SchemaState.UNKNOWN:
                raise ValueError(
                    "a SCHEMA_VERIFIED claim exists but schema_state is unknown"
                )

        # 4. A missing target must never be reported as an unknown one, and an
        #    unknown target must never be reported as missing. The distinct
        #    states mean something only if each is reachable on its own evidence.
        if (
            self.target_analysis.state is TargetState.INCOMPATIBLE
            and self.target_analysis.layer not in DECLARING_LAYERS
        ):
            raise ValueError(
                "an INCOMPATIBLE target needs declared evidence, not speculation"
            )

        # 5. The record's own evidence layer must not overstate what its claims
        #    support. A record whose strongest claim is a model suggestion is
        #    itself AI_SUGGESTED.
        if self.evidence:
            strongest = strongest_layer([claim.layer for claim in self.evidence])
            if LAYER_STRENGTH[strongest] > LAYER_STRENGTH[self.evidence_layer]:
                raise ValueError(
                    f"evidence_layer={self.evidence_layer.value} understates the "
                    f"claims present (strongest is {strongest.value})"
                )

        # 6. Feature coverage is reported once, not twice, so the two references
        #    must agree. Compared by value rather than identity: a record read
        #    back from JSON holds two equal-but-distinct objects, and object
        #    identity would make an otherwise valid record unrevalidatable.
        if self.compatibility.feature_coverage != self.feature_coverage:
            raise ValueError(
                "compatibility.feature_coverage and feature_coverage must agree: "
                "coverage is reported once, not twice"
            )

        return self


def risks_emittable_in_phase_3a() -> frozenset[RiskCategory]:
    """The risks this phase may emit. Exposed so tests can assert the boundary."""
    return EMITTABLE_RISK_CATEGORIES