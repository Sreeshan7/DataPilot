"""Run the controlled ranking stress tests and print the measurements.

    python -m scripts.run_stress

Measurement only: this script reads the ranker and prints numbers. It never
modifies the ranker, the weights, or the evaluation corpus.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
from app.services.datasets.evaluation import stress as S

BAR = "=" * 78


def show(title: str) -> None:
    print()
    print(BAR)
    print(title)
    print(BAR)


def table(rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows:
        print("  (no rows)")
        return
    columns = columns or list(rows[0])
    widths = {
        c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in columns
    }
    print("  " + "  ".join(c.ljust(widths[c]) for c in columns))
    print("  " + "  ".join("-" * widths[c] for c in columns))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in columns))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value) or "-"
    return str(value)


def main() -> int:
    engine = S.make_engine(Settings())

    show("SECTION 1  controlled single-dimension pairs")
    pairs = S.controlled_pairs(engine)
    table(
        [
            {
                "dimension": p.dimension,
                "low": p.low_label,
                "high": p.high_label,
                "isolated": p.isolated,
                "moved": ",".join(p.moved_factors),
                "factor_d": p.factor_score_delta,
                "points": round(p.contribution_delta, 2),
                "score_d": round(p.score_delta, 2),
                "winner": p.winner,
            }
            for p in pairs
        ]
    )
    bad = [p for p in pairs if not p.isolated]
    print(f"\n  isolated: {len(pairs) - len(bad)}/{len(pairs)}")
    for p in bad:
        print(f"  NOT isolated: {p.dimension} also moved {p.moved_factors}")

    show("SECTION 1b  keyword_overlap isolated on the project side")
    sweep = S.keyword_project_sweep(engine)
    table(
        [
            {
                "concepts": 3 - i,
                "keyword_overlap": round(p.factor("keyword_overlap").score or 0, 4),
                "points": p.factor("keyword_overlap").contribution,
                "domain": round(p.factor("domain_relevance").score or 0, 4),
                "feature": round(p.factor("feature_match").score or 0, 4),
                "target": round(p.factor("target_match").score or 0, 4),
                "score": round(p.score, 2),
            }
            for i, p in enumerate(sweep)
        ]
    )

    show("SECTION 2  relevance ladder (the denominator every tipping point sits on)")
    ladder = S.relevance_ladder(engine)
    table(
        [
            {
                "rung": label,
                "relevance": round(p.relevance, 2),
                "feature": round(p.factor("feature_match").score or 0, 4),
                "score": round(p.score, 2),
            }
            for label, p in ladder
        ]
    )

    show("SECTION 3  popularity tipping point (download inflation needed to flip)")
    table(S.popularity_tipping(engine))

    show("SECTION 4  recency tipping point (age at which the fresher weaker one wins)")
    table(S.recency_tipping(engine))

    show("SECTION 5  verification step")
    table(S.verification_tipping(engine))

    show("SECTION 5b  how far each publisher factor can physically move a score")
    table(S.factor_ceilings(engine))

    show("SECTION 6  renormalisation cases")
    table(S.renormalisation_cases(engine))

    show("SECTION 6a  removing evidence must never raise a score")
    mono = S.evidence_removal_monotonicity(engine)
    table(mono)
    bad_steps = [r for r in mono if r["increased"]]
    print(
        f"\n  monotonic: {len(mono) - len(bad_steps)}/{len(mono)} steps non-increasing"
        + ("" if not bad_steps else f"  VIOLATIONS: {[r['step'] for r in bad_steps]}")
    )

    show("SECTION 6b  per-factor inflation under thin evidence")
    table(S.inflation_profile(engine))

    show("SECTION 7  UNKNOWN vs MATCH vs MISMATCH")
    table(S.unknown_vs_match_vs_mismatch(engine))

    show("SECTION 8  adversarial keyword stuffing")
    table(S.keyword_adversarial(engine))

    show("SECTION 9  same-relevance tie-breaks")
    table(S.same_relevance_tiebreaks(engine))

    payload = {
        "controlled_pairs": [
            {
                "dimension": p.dimension,
                "isolated": p.isolated,
                "moved_factors": list(p.moved_factors),
                "factor_score_delta": p.factor_score_delta,
                "contribution_delta": p.contribution_delta,
                "score_delta": p.score_delta,
                "winner": p.winner,
            }
            for p in pairs
        ],
        "keyword_sweep": [
            {"keyword_overlap": p.factor("keyword_overlap").score,
             "contribution": p.factor("keyword_overlap").contribution}
            for p in sweep
        ],
        "relevance_ladder": [
            {"rung": label, "relevance": p.relevance, "score": p.score}
            for label, p in ladder
        ],
        "popularity_tipping": S.popularity_tipping(engine),
        "recency_tipping": S.recency_tipping(engine),
        "verification_tipping": S.verification_tipping(engine),
        "factor_ceilings": S.factor_ceilings(engine),
        "renormalisation_cases": S.renormalisation_cases(engine),
        "evidence_removal_monotonicity": S.evidence_removal_monotonicity(engine),
        "inflation_profile": S.inflation_profile(engine),
        "unknown_vs_match_vs_mismatch": S.unknown_vs_match_vs_mismatch(engine),
        "keyword_adversarial": S.keyword_adversarial(engine),
        "same_relevance_tiebreaks": S.same_relevance_tiebreaks(engine),
    }
    out = Path(__file__).resolve().parents[1] / "evaluation" / "stress_results.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())