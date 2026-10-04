"""Controlled stress tests for the ranking function.

**This module measures. It must never change.**

Every experiment is built so that exactly one ranking dimension differs between
two otherwise identical candidates. That constraint is the whole point: the
benchmark corpus is made of realistic datasets where a dozen factors move at
once, so it can show that a factor *correlates* with a ranking outcome but can
never show that a factor *causes* one. Only a controlled pair can do that.

Two rules keep the measurements honest:

1. **Isolation is verified, not assumed.** Every pair measurement recomputes all
   ten factors for both candidates and checks that only the intended factor moved.
   If a supposedly controlled pair moved something else, the result carries
   ``isolated=False`` instead of being quietly presented as a clean result. Three
   of the first nine pairs were rejected this way, which is the check working.
2. **No normalisation.** Contributions are compared exactly as the ranker reports
   them. Nothing here divides, rescales or "corrects" a score. Where the ranker's
   own renormalisation is under test, it is measured and reported, not compensated
   for.

All clocks are pinned to :data:`BENCHMARK_NOW` and all inputs are synthetic
fixtures, so the module runs offline and reproducibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from app.schemas.dataset import DatasetCandidate, SourceType, VerificationStatus
from app.services.datasets.evaluation.fixtures import BENCHMARK_NOW, build_requirements
from app.services.datasets.evaluation.models import RequirementSpec
from app.services.datasets.ranking import DatasetRankingEngine

#: The project shared by every candidate-side experiment. Never varied, so the
#: project half of each comparison is constant and only candidate differences can
#: explain a score change.
STRESS_PROJECT = RequirementSpec(
    domain="retail",
    subdomain="customer analytics",
    keywords=["customer churn", "retention", "purchase frequency"],
    dictionary_keywords=["customer churn", "retention", "purchase frequency"],
    features=["purchase frequency", "tenure", "total spending"],
    target="churn",
    tasks=["classification"],
    data_types=["tabular"],
)

#: Evidence that never mentions the project target, so ``target_match`` responds
#: to ``target_description`` alone instead of matching "churn" in the prose.
TARGET_FREE_DESCRIPTION = (
    "Retail customer transaction records with purchase frequency, tenure and "
    "total spending per customer."
)
TARGET_FREE_TAGS = ("retail", "customer", "transactions")
NO_FEATURES = "aggregated monthly totals with no customer-level breakdown"

#: A project whose *concepts* deliberately exclude its own target word.
#:
#: ``keyword_overlap`` and ``target_match`` read the same published evidence, so if
#: "churn" were also a project concept then satisfying the target would always also
#: raise keyword_overlap and the two could never be varied separately. Removing
#: "churn" from the concept list lets the target vary on its own.
TARGET_ISOLATED_PROJECT = RequirementSpec(
    domain="retail",
    subdomain="customer analytics",
    keywords=["retention", "purchase frequency"],
    dictionary_keywords=["retention", "purchase frequency"],
    features=["purchase frequency", "tenure", "total spending"],
    target="churn",
    tasks=["classification"],
    data_types=["tabular"],
)

#: Evidence that satisfies every concept of :data:`TARGET_ISOLATED_PROJECT` but
#: never mentions the target, so ``target_match`` depends on ``target_description``
#: alone.
TARGET_ISOLATED_DESCRIPTION = (
    "Retail customer transaction records reporting purchase frequency, tenure and "
    "total spending per customer, used for retention analysis."
)

ALL_FEATURE_NAMES = (
    "compatibility",
    "domain_relevance",
    "feature_match",
    "keyword_overlap",
    "popularity",
    "quality",
    "recency",
    "target_match",
    "task_match",
    "verification",
)

_REQUIREMENT_FACTORS = frozenset(
    {"compatibility", "domain_relevance", "feature_match", "keyword_overlap",
     "target_match", "task_match"}
)

#: The score denominator is the nominal weight of whichever factor set is being
#: scored - all ten for ``ranking_score``, the six requirement factors for
#: ``relevance_score``, the three publisher factors for ``quality_score``. It never
#: varies with how much evidence a candidate happens to carry.
_DENOMINATOR_IS_PINNED = True

#: Dimensions whose evidence the ranker deliberately also reads through another
#: factor. Declaring the wrong task is simultaneously a task mismatch and a
#: compatibility contradiction, so both move together by design rather than by
#: accident. Recorded so the report can tell a designed coupling apart from a
#: pair that failed to be controlled.
COUPLED_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "task_match": ("compatibility",),
}


# ---------------------------------------------------------------------------
# candidate construction
# ---------------------------------------------------------------------------


_UNSET = object()


def make_candidate(
    key: str,
    *,
    name: Optional[str] = None,
    description: str = (
        "Retail customer transaction records with purchase frequency, tenure and "
        "total spending per customer for churn modelling."
    ),
    domain: Optional[str] = "retail",
    tags: tuple[str, ...] = ("retail", "customer", "churn", "transactions"),
    target_description: Optional[str] = "whether the customer churns",
    feature_description: Optional[str] = (
        "purchase frequency, tenure and total spending per customer"
    ),
    task_types: tuple[str, ...] = ("classification",),
    download_count: Optional[int] = 1_000,
    vote_count: Optional[int] = 100,
    usability: Optional[float] = 0.8,
    updated_at: object = _UNSET,
    source_type: SourceType = SourceType.VERIFIED_EXTERNAL,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
) -> DatasetCandidate:
    """Build one candidate with every field under explicit control.

    Defaults describe a fully-declared, verified, moderately popular dataset, so
    an experiment only has to name the single field it wants to move. Fields are
    ``Optional`` because "this datum is absent" is itself a condition worth
    measuring - so ``updated_at`` uses a sentinel default, which lets ``None`` mean
    a genuinely absent date rather than "caller said nothing".
    """
    return DatasetCandidate(
        id=f"stress:{key}",
        name=name or f"Stress fixture {key}",
        source="stress-fixture",
        source_id=f"stress/{key}",
        source_type=source_type,
        verification_status=verification_status,
        url=f"https://example.invalid/stress/{key}",
        description=description,
        owner="stress-fixture",
        tags=list(tags),
        domain=domain,
        license="CC0-1.0",
        download_count=download_count,
        vote_count=vote_count,
        usability=usability,
        updated_at=(
            BENCHMARK_NOW - timedelta(days=90)
            if updated_at is _UNSET
            else updated_at
        ),
        target_description=target_description,
        feature_description=feature_description,
        task_types=list(task_types),
        file_types=["csv"],
        possible_source="stress-fixture",
    )


def target_free_candidate(key: str, **overrides) -> DatasetCandidate:
    """A candidate whose evidence never contains the project target word.

    ``target_match`` therefore tracks ``target_description`` alone, which is what
    makes the MATCH/MISMATCH/UNKNOWN triple a controlled comparison.
    """
    fields: dict = {
        "description": TARGET_FREE_DESCRIPTION,
        "tags": TARGET_FREE_TAGS,
        "target_description": None,
    }
    fields.update(overrides)
    return make_candidate(key, **fields)


def make_engine(settings) -> DatasetRankingEngine:
    """An engine with the clock pinned, so recency is reproducible."""
    return DatasetRankingEngine(settings, now=BENCHMARK_NOW)


def make_project(overrides: Optional[RequirementSpec] = None):
    return build_requirements(overrides or STRESS_PROJECT, objective="Ranking stress test")


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorReading:
    """One factor as the ranker reported it, plus derived bookkeeping."""

    name: str
    score: Optional[float]
    contribution: float
    weight: float
    judged: float
    unknown_terms: tuple[str, ...]
    unmatched_terms: tuple[str, ...]

    @property
    def counted(self) -> bool:
        """Whether this factor's weight sits in the score denominator.

        Always true. The denominator is pinned to the nominal total weight of the
        factor set, so a factor that could not be judged forfeits its own credit
        and stays in the mean. This mirrors ``_contributions`` as fixed: an
        earlier version dropped silent publisher factors from the denominator,
        which let a dataset with *less* evidence outscore one with more.
        """
        return True

    @property
    def effective_weight(self) -> float:
        return self.weight

    @property
    def inflation(self) -> float:
        """Points earned as a multiple of this factor's nominal weight.

        Must never exceed 1.0. Any value above it means the denominator is
        rescaling the factor set.
        """
        return self.contribution / self.weight if self.weight else 0.0


@dataclass(frozen=True)
class Probe:
    """One ranked candidate with its per-factor breakdown."""

    label: str
    candidate: DatasetCandidate
    rank: int
    score: float
    relevance: float
    publisher: float
    factors: dict[str, FactorReading]

    def factor(self, name: str) -> FactorReading:
        return self.factors[name]

    @property
    def unknown_factors(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, f in self.factors.items() if f.score is None))

    @property
    def counted_weight(self) -> float:
        """The score denominator: nominal total weight, always.

        Named for continuity with earlier reports, but it no longer varies with
        evidence. A constant denominator is the property that makes silence
        unable to inflate anything.
        """
        return sum(f.effective_weight for f in self.factors.values())

    @property
    def normalisation(self) -> float:
        """The multiplier applied to every factor: 100 / denominator."""
        total = self.counted_weight
        return (100.0 / total) if total else 0.0

    @property
    def max_inflation(self) -> float:
        """Largest points-per-nominal-weight multiple across all factors."""
        values = [f.inflation for f in self.factors.values() if f.weight]
        return max(values) if values else 0.0

    @property
    def contributions_sum(self) -> float:
        return round(sum(f.contribution for f in self.factors.values()), 6)


@dataclass(frozen=True)
class PairMeasurement:
    """The measured difference between two candidates that should differ in one dimension."""

    dimension: str
    low: Probe
    high: Probe
    #: Factors whose score moved. Should contain exactly the intended factor.
    moved_factors: tuple[str, ...]
    #: True when only the intended factor moved.
    isolated: bool
    low_label: str
    high_label: str

    @property
    def winner(self) -> str:
        return self.high_label if self.high.rank < self.low.rank else self.low_label

    @property
    def expected_couplings(self) -> tuple[str, ...]:
        return COUPLED_DIMENSIONS.get(self.dimension, ())

    @property
    def coupling_explained(self) -> bool:
        """True when the extra factors that moved are documented couplings."""
        expected = set(self.expected_couplings) | {self.dimension}
        return set(self.moved_factors) == expected

    @property
    def controlled(self) -> bool:
        """Isolated, or moving exactly the documented coupled set."""
        return self.isolated or self.coupling_explained

    @property
    def flipped(self) -> bool:
        return (self.low.rank < self.high.rank) != (self.high.rank < self.low.rank)

    @property
    def score_delta(self) -> float:
        """Points gained by the high variant over the low variant."""
        return self.high.score - self.low.score

    @property
    def relevance_delta(self) -> float:
        return self.high.relevance - self.low.relevance

    @property
    def contribution_delta(self) -> float:
        return (
            self.high.factor(self.dimension).contribution
            - self.low.factor(self.dimension).contribution
        )

    @property
    def factor_score_delta(self) -> Optional[float]:
        return _delta(
            self.high.factor(self.dimension).score,
            self.low.factor(self.dimension).score,
        )


@dataclass(frozen=True)
class Finding:
    """A single measured statement. No interpretation attached."""

    section: str
    claim: str
    observed: dict


def _delta(high: Optional[float], low: Optional[float]) -> Optional[float]:
    if high is None or low is None:
        return None
    return high - low


def contribution_deltas(pair: PairMeasurement) -> dict[str, float]:
    """Points each factor gained going from the low variant to the high variant.

    Score deltas alone are not enough to explain a score delta. A factor can hold
    the same score and still contribute differently when its judged fraction or
    the shared denominator changes, which is exactly what happens when provenance
    changes a field's availability from declared to prose.
    """
    return {
        name: round(
            pair.high.factors[name].contribution - pair.low.factors[name].contribution, 4
        )
        for name in ALL_FEATURE_NAMES
        if name in pair.low.factors and name in pair.high.factors
    }


def moved_contributions(pair: PairMeasurement, exclude: str = "") -> dict[str, float]:
    return {
        name: delta
        for name, delta in contribution_deltas(pair).items()
        if abs(delta) > 1e-9 and name != exclude
    }


def read_factors(candidate: DatasetCandidate) -> dict[str, FactorReading]:
    """Extract the per-factor view the ranker itself published."""
    out: dict[str, FactorReading] = {}
    for factor in candidate.ranking_factors:
        requirements = factor.requirements or []
        total = sum(r.weight for r in requirements)
        judged = (
            1.0
            if not requirements or total <= 0
            else sum(
                r.weight
                for r in requirements
                if r.status.value != "unknown"
            )
            / total
        )
        out[factor.name.value] = FactorReading(
            name=factor.name.value,
            score=factor.score,
            contribution=factor.contribution,
            weight=factor.weight,
            judged=judged,
            unknown_terms=tuple(factor.unknown_terms or ()),
            unmatched_terms=tuple(factor.unmatched_terms or ()),
        )
    return out


def _to_probe(candidate: DatasetCandidate, position: int) -> Probe:
    return Probe(
        label=candidate.id.rsplit("/", 1)[-1],
        candidate=candidate,
        rank=position,
        score=candidate.ranking_score or 0.0,
        relevance=candidate.relevance_score or 0.0,
        publisher=candidate.quality_score or 0.0,
        factors=read_factors(candidate),
    )


def rank_all(
    candidates: list[DatasetCandidate],
    engine: DatasetRankingEngine,
    project=None,
) -> list[Probe]:
    project = project if project is not None else make_project()
    ranked = engine.rank(list(candidates), project)
    return [_to_probe(c, i) for i, c in enumerate(ranked, start=1)]


def measure_pair(
    dimension: str,
    low: DatasetCandidate,
    high: DatasetCandidate,
    engine: DatasetRankingEngine,
    project=None,
    low_label: Optional[str] = None,
    high_label: Optional[str] = None,
) -> PairMeasurement:
    """Rank one pair and report what actually moved.

    ``low`` and ``high`` must be identical apart from ``dimension``. Whether they
    really are is not assumed - it is measured, and ``isolated`` reports the
    answer.
    """
    ranked = rank_all([low, high], engine, project)
    probes = {p.candidate.id: p for p in ranked}
    low_probe, high_probe = probes[low.id], probes[high.id]
    moved = tuple(
        sorted(
            name
            for name in low_probe.factors
            if low_probe.factors[name].score != high_probe.factors[name].score
        )
    )
    return PairMeasurement(
        dimension=dimension,
        low=low_probe,
        high=high_probe,
        moved_factors=moved,
        isolated=moved == (dimension,),
        low_label=low_label or low.id.rsplit("/", 1)[-1],
        high_label=high_label or high.id.rsplit("/", 1)[-1],
    )


# ---------------------------------------------------------------------------
# section 1 - controlled single-dimension distractor pairs
# ---------------------------------------------------------------------------


def controlled_pairs(engine: DatasetRankingEngine) -> list[PairMeasurement]:
    """Eight candidate-side pairs, each differing in one dimension only.

    Publisher dimensions vary a single metadata field. Requirement dimensions vary
    a single piece of published evidence, with the surrounding evidence held
    target-free so the response is attributable. ``keyword_overlap`` is measured
    by :func:`keyword_project_sweep` instead - see the note below.
    """
    base = make_candidate("base")
    results: list[PairMeasurement] = []

    # popularity: one order of magnitude of downloads, everything else fixed.
    results.append(
        measure_pair("popularity", base, make_candidate("popular", download_count=10_000), engine)
    )

    # recency: same publication, six years apart.
    results.append(
        measure_pair(
            "recency",
            make_candidate("stale", updated_at=BENCHMARK_NOW - timedelta(days=1800)),
            make_candidate("fresh", updated_at=BENCHMARK_NOW - timedelta(days=30)),
            engine,
        )
    )

    # verification: provenance only.
    results.append(
        measure_pair(
            "verification",
            make_candidate(
                "ai",
                source_type=SourceType.AI_SUGGESTED,
                verification_status=VerificationStatus.UNVERIFIED,
            ),
            make_candidate("verified"),
            engine,
        )
    )

    # quality: publisher usability rating only.
    results.append(
        measure_pair("quality", make_candidate("poor", usability=0.1), base, engine)
    )

    # domain_relevance: declared topic domain only.
    results.append(
        measure_pair(
            "domain_relevance",
            make_candidate("wrong-domain", domain="agriculture"),
            base,
            engine,
        )
    )

    # target_match: the declared target is the only difference. The project keeps
    # its target word out of the concept list so that satisfying the target does
    # not also raise keyword_overlap.
    target_project = make_project(TARGET_ISOLATED_PROJECT)
    results.append(
        measure_pair(
            "target_match",
            make_candidate(
                "wrong-target",
                description=TARGET_ISOLATED_DESCRIPTION,
                tags=TARGET_FREE_TAGS,
                target_description="the number of support tickets opened",
            ),
            make_candidate(
                "right-target",
                description=TARGET_ISOLATED_DESCRIPTION,
                tags=TARGET_FREE_TAGS,
                target_description="whether the customer churns",
            ),
            engine,
            project=target_project,
            low_label="wrong-target",
            high_label="right-target",
        )
    )

    # feature_match: the dataset enumerates one of the three required features.
    results.append(
        measure_pair(
            "feature_match",
            make_candidate(
                "one-feature",
                feature_description="purchase frequency per customer",
                description="Retail customer transaction records reporting purchase frequency per customer.",
            ),
            base,
            engine,
        )
    )

    # keyword_overlap has no candidate-side pair on purpose. It reads the same
    # published evidence as feature_match and target_match, so every edit that
    # moves it also moves those. It is isolated on the project side instead, by
    # keyword_project_sweep() below.

    # task_match: declares regression where the project needs classification.
    results.append(
        measure_pair(
            "task_match",
            make_candidate("wrong-task", task_types=("regression",)),
            base,
            engine,
        )
    )

    return results


def keyword_project_sweep(engine: DatasetRankingEngine) -> list[Probe]:
    """Move only the project's concept count and watch only keyword_overlap react.

    The candidate is byte-identical in every probe, so any factor other than
    keyword_overlap moving would show up immediately.
    """
    candidate = make_candidate("kw-sweep")
    out: list[Probe] = []
    for count in (3, 2, 1):
        kws = ["customer churn", "retention", "purchase frequency"][:count]
        project = make_project(
            RequirementSpec(
                **{
                    **STRESS_PROJECT.model_dump(),
                    "keywords": kws,
                    "dictionary_keywords": kws,
                }
            )
        )
        out.append(rank_all([candidate], engine, project)[0])
    return out


# ---------------------------------------------------------------------------
# section 2-4 - where each factor stops being decisive
# ---------------------------------------------------------------------------

#: Relevance rungs built by varying how many required features a dataset declares.
RELEVANCE_RUNGS: tuple[tuple[str, dict], ...] = (
    ("full", {}),
    ("two-of-three", {"feature_description": "purchase frequency and tenure per customer",
                      "description": "Retail customer transaction records reporting purchase frequency and tenure per customer for churn modelling."}),
    ("one-of-three", {"feature_description": "tenure per customer",
                      "description": "Retail customer transaction records reporting tenure per customer for churn modelling."}),
)


def relevance_ladder(engine: DatasetRankingEngine) -> list[Probe]:
    return [
        (label, rank_all([make_candidate("rel", **fields)], engine)[0])
        for label, fields in RELEVANCE_RUNGS
    ]


def _bisect(
    predicate: Callable[[float], Optional[bool]],
    low: float,
    high: float,
    steps: int = 40,
) -> Optional[float]:
    """Find where ``predicate`` first returns True, or None if it never does.

    Used to locate a tipping point by measurement rather than by guessing a grid.
    """
    if predicate(high) is not True:
        return None
    for _ in range(steps):
        mid = (low + high) / 2
        if predicate(mid) is True:
            high = mid
        else:
            low = mid
    return high


def popularity_tipping(engine: DatasetRankingEngine) -> list[dict]:
    """How much download inflation does a relevance gap of a given size need?

    For each rung of the relevance ladder, hold the weaker candidate fixed and
    raise the stronger candidate's download count until the ordering flips.
    """
    rows: list[dict] = []
    runs = relevance_ladder(engine)
    for weaker_label, weaker in runs:
        for stronger_label, stronger in runs:
            if weaker.score >= stronger.score:
                continue
            gap = stronger.score - weaker.score
            weak_downloads = weaker.factor("popularity").contribution

            def wins(log_ratio: float, _s=stronger, _w=weaker) -> Optional[bool]:
                downloads = max(1, int(10**log_ratio * 1_000))
                pair = measure_pair(
                    "popularity",
                    _w.candidate,
                    _variant(_s.candidate, "strong-pop", download_count=downloads),
                    engine,
                )
                return pair.winner == "strong-pop"

            flip = _bisect(wins, 0.0, 12.0)
            rows.append(
                {
                    "weaker": weaker_label,
                    "stronger": stronger_label,
                    "relevance_gap": round(stronger.relevance - weaker.relevance, 2),
                    "score_gap": round(gap, 2),
                    "ratio_needed": round(10**flip, 1) if flip is not None else None,
                    "log10_ratio": round(flip, 3) if flip is not None else None,
                    "flips_within_log10_12": flip is not None,
                }
            )
    return rows


def recency_tipping(engine: DatasetRankingEngine) -> list[dict]:
    """How stale can the more relevant dataset be before recency rescues the other?

    For each rung pair, hold the weaker candidate fresh and age the stronger one
    until the ordering flips.
    """
    rows: list[dict] = []
    runs = relevance_ladder(engine)
    for weaker_label, weaker in runs:
        for stronger_label, stronger in runs:
            if weaker.score >= stronger.score:
                continue
            weak_age = 90

            def wins(age: float, _s=stronger, _w=weaker) -> Optional[bool]:
                stale = _variant(
                    _s.candidate, "strong-stale",
                    updated_at=BENCHMARK_NOW - timedelta(days=age),
                )
                fresh = _variant(
                    _w.candidate, "weak-fresh",
                    updated_at=BENCHMARK_NOW - timedelta(days=weak_age),
                )
                pair = measure_pair("recency", stale, fresh, engine)
                return pair.winner == "weak-fresh"

            flip = _bisect(wins, 0.0, 3_650.0, steps=30)
            rows.append(
                {
                    "weaker": weaker_label,
                    "stronger": stronger_label,
                    "relevance_gap": round(stronger.relevance - weaker.relevance, 2),
                    "score_gap": round(stronger.score - weaker.score, 2),
                    "age_at_flip_days": round(flip, 1) if flip is not None else None,
                    "horizon_days": 1825,
                }
            )
    return rows


def verification_tipping(engine: DatasetRankingEngine) -> list[dict]:
    """Can any relevance advantage overcome the verification step?

    Verification is binary (1.0 vs 0.25), so there is no tipping point to
    bisect. Instead each rung is paired against an unverified candidate and we
    report whether the verified candidate still wins.
    """
    rows: list[dict] = []
    runs = relevance_ladder(engine)
    for label, probe in runs:
        unverified = _variant(
            probe.candidate,
            "unverified",
            source_type=SourceType.AI_SUGGESTED,
            verification_status=VerificationStatus.UNVERIFIED,
        )
        pair = measure_pair(
            "verification", unverified, probe.candidate, engine,
            low_label="unverified", high_label="verified",
        )
        rows.append(
            {
                "rung": label,
                "relevance": round(probe.relevance, 2),
                "verified_score": round(probe.score, 2),
                "unverified_score": round(pair.low.score, 2),
                "step": round(pair.score_delta, 2),
                "verified_wins": pair.winner == "verified",
                "also_moved_points": "; ".join(
                    f"{k}{v:+.2f}" for k, v in moved_contributions(pair, "verification").items()
                ) or "-",
            }
        )
    return rows


def factor_ceilings(engine: DatasetRankingEngine) -> list[dict]:
    """The largest swing each publisher factor can physically produce.

    A factor that cannot move a score by more than X points cannot decide any
    contest whose gap exceeds X. This measures X directly instead of inferring it
    from a failed search.
    """
    rows: list[dict] = []

    # popularity: sweep downloads across twelve orders of magnitude, votes pinned.
    pop: list[float] = []
    for exponent in range(0, 13):
        candidate = make_candidate("ceil", download_count=10**exponent, vote_count=100)
        pop.append(rank_all([candidate], engine)[0].factor("popularity").contribution)
    rows.append(
        {
            "factor": "popularity",
            "swept": "downloads 10^0..10^12, votes pinned at 100",
            "min_points": round(min(pop), 2),
            "max_points": round(max(pop), 2),
            "total_swing": round(max(pop) - min(pop), 2),
        }
    )

    # popularity with votes free to move too - the widest case available.
    pop2: list[float] = []
    for exponent in range(0, 13):
        candidate = make_candidate("ceil", download_count=10**exponent, vote_count=10**exponent)
        pop2.append(rank_all([candidate], engine)[0].factor("popularity").contribution)
    rows.append(
        {
            "factor": "popularity (votes free)",
            "swept": "downloads and votes 10^0..10^12",
            "min_points": round(min(pop2), 2),
            "max_points": round(max(pop2), 2),
            "total_swing": round(max(pop2) - min(pop2), 2),
        }
    )

    # recency: sweep age well past the decay horizon.
    rec: list[float] = []
    for days in (0, 456, 912, 1368, 1825, 3650, 10000):
        candidate = make_candidate("ceil", updated_at=BENCHMARK_NOW - timedelta(days=days))
        rec.append(rank_all([candidate], engine)[0].factor("recency").contribution)
    rows.append(
        {
            "factor": "recency",
            "swept": "age 0..10000 days",
            "min_points": round(min(rec), 2),
            "max_points": round(max(rec), 2),
            "total_swing": round(max(rec) - min(rec), 2),
        }
    )

    # verification: the only two reachable values.
    ver = []
    for source, status in (
        (SourceType.VERIFIED_EXTERNAL, VerificationStatus.VERIFIED),
        (SourceType.VERIFIED_EXTERNAL, VerificationStatus.UNVERIFIED),
        (SourceType.AI_SUGGESTED, VerificationStatus.UNVERIFIED),
        (SourceType.AI_SUGGESTED, VerificationStatus.VERIFIED),
    ):
        candidate = make_candidate("ceil", source_type=source, verification_status=status)
        ver.append(
            (f"{source.value}/{status.value}",
             rank_all([candidate], engine)[0].factor("verification").contribution)
        )
    rows.append(
        {
            "factor": "verification",
            "swept": "; ".join(f"{k}={v}" for k, v in ver),
            "min_points": round(min(v for _, v in ver), 2),
            "max_points": round(max(v for _, v in ver), 2),
            "total_swing": round(max(v for _, v in ver) - min(v for _, v in ver), 2),
        }
    )

    return rows


def _fields(candidate: DatasetCandidate) -> dict:
    """Re-create a candidate's controllable fields, for one-field overrides."""
    return {
        "description": candidate.description,
        "domain": candidate.domain,
        "tags": tuple(candidate.tags or ()),
        "target_description": candidate.target_description,
        "feature_description": candidate.feature_description,
        "task_types": tuple(candidate.task_types or ()),
        "download_count": candidate.download_count,
        "vote_count": candidate.vote_count,
        "usability": candidate.usability,
        "updated_at": candidate.updated_at,
        "source_type": candidate.source_type,
        "verification_status": candidate.verification_status,
    }


def _variant(candidate: DatasetCandidate, key: str, **overrides) -> DatasetCandidate:
    """A copy of ``candidate`` with ``overrides`` applied.

    Overrides are merged rather than passed alongside the full field set, so a
    field can be varied without colliding with itself.
    """
    return make_candidate(key, **{**_fields(candidate), **overrides})


# ---------------------------------------------------------------------------
# section 5 - renormalisation
# ---------------------------------------------------------------------------


def renormalisation_cases(engine: DatasetRankingEngine) -> list[dict]:
    """What happens to the other nine factors when one goes silent.

    Case A: everything judgeable. Case B: publisher metadata absent, so all three
    publisher factors drop out of the denominator. Case C: publisher metadata
    absent *and* the target undeclared, so a requirement factor stays in the
    denominator holding its weight at zero credit. Case D: publisher metadata
    absent, no declared target, no declared features and no declared tasks.

    Cases C and D use target-free description text, otherwise the word "churn"
    sitting in the prose would satisfy the target anyway and the case would not
    test what it claims to test.
    """
    publisher_absent = dict(
        download_count=None, vote_count=None, usability=None, updated_at=None
    )
    cases = {
        "A-all-judgeable": {},
        "B-publisher-absent": dict(publisher_absent),
        "C-publisher-absent+target-unknown": dict(
            publisher_absent,
            description=TARGET_FREE_DESCRIPTION,
            tags=TARGET_FREE_TAGS,
            target_description=None,
        ),
        "D-all-absent": dict(
            publisher_absent,
            description=TARGET_FREE_DESCRIPTION,
            tags=TARGET_FREE_TAGS,
            target_description=None,
            feature_description=None,
            task_types=[],
        ),
    }
    rows: list[dict] = []
    for label, fields in cases.items():
        candidate = make_candidate("renorm", **fields)
        probe = rank_all([candidate], engine)[0]
        rows.append(
            {
                "case": label,
                "denominator": round(probe.counted_weight, 2),
                "normalisation": round(probe.normalisation, 4),
                "score": round(probe.score, 2),
                "silent_factors": ",".join(probe.unknown_factors) or "-",
                "verification_points": probe.factor("verification").contribution,
                "target_points": probe.factor("target_match").contribution,
                "max_inflation": round(probe.max_inflation, 4),
                "contributions_sum": probe.contributions_sum,
            }
        )
    return rows


def evidence_removal_monotonicity(engine: DatasetRankingEngine) -> list[dict]:
    """Removing evidence must never raise a score.

    The same candidate is measured repeatedly while one more field is taken away.
    Every step is a strict loss of information, so the score must not increase at
    any step. This is the property the old denominator violated.
    """
    steps = [
        ("complete evidence", {}),
        ("no popularity metadata", dict(download_count=None, vote_count=None)),
        ("no usability rating", dict(download_count=None, vote_count=None, usability=None)),
        ("no publication date", dict(download_count=None, vote_count=None,
                                     usability=None, updated_at=None)),
        # Target-free text, or "churn" in the prose would satisfy the target anyway
        # and this step would silently measure nothing.
        (
            "no declared target",
            dict(download_count=None, vote_count=None, usability=None, updated_at=None,
                 description=TARGET_FREE_DESCRIPTION, tags=TARGET_FREE_TAGS,
                 target_description=None),
        ),
        (
            "no target, features or tasks",
            dict(download_count=None, vote_count=None, usability=None, updated_at=None,
                 description=TARGET_FREE_DESCRIPTION, tags=TARGET_FREE_TAGS,
                 target_description=None, feature_description=None, task_types=[]),
        ),
    ]
    rows: list[dict] = []
    previous: Optional[float] = None
    for label, fields in steps:
        probe = rank_all([make_candidate("mono", **fields)], engine)[0]
        delta = None if previous is None else round(probe.score - previous, 2)
        rows.append(
            {
                "step": label,
                "denominator": round(probe.counted_weight, 2),
                "score": round(probe.score, 2),
                "change": delta,
                "increased": bool(delta is not None and delta > 1e-9),
                "max_inflation": round(probe.max_inflation, 4),
                "silent_factors": ",".join(probe.unknown_factors) or "-",
            }
        )
        previous = probe.score
    return rows


def inflation_profile(engine: DatasetRankingEngine) -> list[dict]:
    """Per-factor points and multiples of nominal weight under a thin-evidence case.

    Every ``multiple_of_weight`` must be at most 1.0. A larger value would mean
    the denominator is rescaling the factor set rather than measuring it.
    """
    thin = make_candidate(
        "thin",
        download_count=None,
        vote_count=None,
        usability=None,
        updated_at=None,
        target_description=None,
    )
    probe = rank_all([thin], engine)[0]
    rows: list[dict] = []
    for name in ALL_FEATURE_NAMES:
        if name not in probe.factors:
            continue
        factor = probe.factor(name)
        rows.append(
            {
                "factor": name,
                "score": factor.score,
                "judged": round(factor.judged, 4),
                "counted": factor.counted,
                "weight": factor.weight,
                "contribution": factor.contribution,
                "multiple_of_weight": round(factor.inflation, 4),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# section 6 - UNKNOWN vs MATCH vs MISMATCH
# ---------------------------------------------------------------------------


def unknown_vs_match_vs_mismatch(engine: DatasetRankingEngine) -> list[dict]:
    """Four candidates identical except for whether the target is declared.

    Same surrounding evidence throughout, so the only variable is whether
    ``target_description`` exists and what it says. Run against the project whose
    concepts exclude the target word, so keyword_overlap cannot move either.
    """
    rows: list[dict] = []
    project = make_project(TARGET_ISOLATED_PROJECT)
    specs = [
        ("MATCH", "whether the customer churns"),
        # Resolvable: "default" is a label space the concept index knows, so the
        # contradiction rule in compatibility.py fires and compatibility drops.
        ("MISMATCH-resolvable", "whether the applicant defaults on the loan"),
        # Unresolvable: support-ticket volume is genuinely not churn, but no label
        # space matches it, so the rule declines to call it a contradiction. Kept
        # in the report because it is the honest limit of the mechanism, not an
        # example that flatters it.
        ("MISMATCH-unresolved", "the number of support tickets opened"),
        ("UNKNOWN-absent", None),
        (
            "UNKNOWN-declared-absence",
            "the target label is not provided and would have to be derived",
        ),
    ]
    probes: dict[str, Probe] = {}
    for label, target in specs:
        candidate = make_candidate(
            "tvmm",
            description=TARGET_ISOLATED_DESCRIPTION,
            tags=TARGET_FREE_TAGS,
            target_description=target,
        )
        probe = rank_all([candidate], engine, project)[0]
        probes[label] = probe
        factor = probe.factor("target_match")
        rows.append(
            {
                "case": label,
                "target_score": factor.score,
                "target_points": factor.contribution,
                "judged": round(factor.judged, 3),
                "counted": factor.counted,
                "keyword_overlap": round(probe.factor("keyword_overlap").score or 0, 4),
                "compatibility": round(probe.factor("compatibility").score or 0, 4),
                "statuses": ",".join(
                    f"{r.term}={r.status.value}"
                    for r in factor_name_requirements(candidate, "target_match")
                ) or "-",
                "dataset_score": round(probe.score, 2),
            }
        )
    # A contradiction is only penalised more than silence if some other factor
    # reacts to it, so record which factors separate each mismatch from silence.
    silence = probes["UNKNOWN-absent"]
    for label in ("MISMATCH-resolvable", "MISMATCH-unresolved"):
        probe = probes[label]
        next(r for r in rows if r["case"] == label)["separating_factors"] = ",".join(
            sorted(
                n
                for n in ALL_FEATURE_NAMES
                if n != "target_match"
                and probe.factors[n].score != silence.factors[n].score
            )
        ) or "none"
    rows[0]["separating_factors"] = "none"
    return rows


def factor_name_requirements(candidate: DatasetCandidate, factor_name: str):
    for factor in candidate.ranking_factors:
        if factor.name.value == factor_name:
            return factor.requirements or []
    return []


def _factor_detail(candidate: DatasetCandidate, factor_name: str) -> str:
    for factor in candidate.ranking_factors:
        if factor.name.value == factor_name:
            return factor.detail or ""
    return ""


# ---------------------------------------------------------------------------
# section 7 - adversarial keyword behaviour
# ---------------------------------------------------------------------------


def keyword_adversarial(engine: DatasetRankingEngine) -> list[dict]:
    """Can keyword repetition buy a ranking win the content does not deserve?

    The adversarial candidate is in an unrelated domain but mentions every
    project concept verbatim. The honest candidate is in the right domain and
    mentions only one. Nothing else differs.
    """
    honest = make_candidate(
        "honest",
        description=(
            "Retail customer transaction history for store loyalty analysis, with "
            "per-customer spend and tenure."
        ),
        tags=("retail", "customer", "transactions"),
        target_description="whether the customer churns",
        feature_description="tenure and total spending per customer",
    )
    stuffed = make_candidate(
        "stuffed",
        description=(
            "Municipal bus timetable and depot maintenance records. "
            "customer churn retention purchase frequency customer churn retention "
            "purchase frequency customer churn retention purchase frequency."
        ),
        domain="transport",
        tags=("transport", "customer churn", "retention", "purchase frequency"),
        target_description="whether the customer churns",
        feature_description="customer churn retention purchase frequency",
    )
    pair = measure_pair("keyword_overlap", honest, stuffed, engine,
                        low_label="honest", high_label="stuffed")
    return [
        {
            "candidate": "honest",
            "domain": honest.domain,
            "keyword_overlap": pair.low.factor("keyword_overlap").score,
            "keyword_contribution": pair.low.factor("keyword_overlap").contribution,
            "domain_relevance": pair.low.factor("domain_relevance").score,
            "relevance": round(pair.low.relevance, 2),
            "score": round(pair.low.score, 2),
            "rank": pair.low.rank,
        },
        {
            "candidate": "stuffed",
            "domain": stuffed.domain,
            "keyword_overlap": pair.high.factor("keyword_overlap").score,
            "keyword_contribution": pair.high.factor("keyword_overlap").contribution,
            "domain_relevance": pair.high.factor("domain_relevance").score,
            "relevance": round(pair.high.relevance, 2),
            "score": round(pair.high.score, 2),
            "rank": pair.high.rank,
        },
    ]


# ---------------------------------------------------------------------------
# section 8 - same-relevance tie-breaks
# ---------------------------------------------------------------------------


def same_relevance_tiebreaks(engine: DatasetRankingEngine) -> list[dict]:
    """When relevance is identical, which factor decides the order, and by how much?

    Pairs are built to match on relevance_score and differ on one publisher
    dimension. Relevance equality is verified, not assumed.
    """
    reference = make_candidate("tie-ref")
    scenarios = {
        "popularity: 10x more downloads": dict(download_count=10_000),
        "recency: 2 months newer": dict(updated_at=BENCHMARK_NOW - timedelta(days=30)),
        "recency: 5 years older": dict(updated_at=BENCHMARK_NOW - timedelta(days=1825)),
        "quality: usability 0.8 -> 0.95": dict(usability=0.95),
        "verification: unverified provenance": dict(
            source_type=SourceType.AI_SUGGESTED,
            verification_status=VerificationStatus.UNVERIFIED,
        ),
    }
    rows: list[dict] = []
    for label, fields in scenarios.items():
        candidate = make_candidate("tie-other", **fields)
        ranked = rank_all([reference, candidate], engine)
        by_id = {p.candidate.id: p for p in ranked}
        ref, other = by_id[reference.id], by_id[candidate.id]
        factor = next(
            n for n in ALL_FEATURE_NAMES if ref.factors[n].score != other.factors[n].score
        )
        rows.append(
            {
                "scenario": label,
                "deciding_factor": factor,
                "relevance_equal": abs(ref.relevance - other.relevance) < 1e-9,
                "relevance_delta": round(other.relevance - ref.relevance, 6),
                "score_delta": round(other.score - ref.score, 2),
                "winner": other.label if other.rank < ref.rank else ref.label,
                "moved_factors": ",".join(
                    sorted(
                        n for n in ALL_FEATURE_NAMES
                        if ref.factors[n].score != other.factors[n].score
                    )
                ),
            }
        )
    return rows