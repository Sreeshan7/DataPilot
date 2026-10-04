# Ranking Stress Test — Controlled Measurements

**Date:** 2026-10-04 (Sections 6, 6a, 6b and 7 re-measured after the fix; see "What was changed")
**Scope:** Sections 1–5, 8 and 9 are measurement only. Section 6 originally recorded two
defects; both have since been **fixed**, and those sections now report before/after.
**Harness:** `app/services/datasets/evaluation/stress.py`, run via `python -m scripts.run_stress`
**Raw output:** `evaluation/stress_results.json`
**Guards:** `tests/unit/test_evaluation_stress.py` (60 tests)

Every number below comes from a synthetic, offline, fully deterministic run with the clock
pinned to `BENCHMARK_NOW = 2026-09-28 12:00 UTC`. The project side of every comparison is
held constant, so only candidate-side differences can explain a score change.

Findings are grouped into **OBSERVED FACT** (what the ranker did) and **INTERPRETATION** (what
that means). Section 6 additionally carries a **BEFORE/AFTER** pair because the original finding
has been acted on. Where a fix changed behaviour, the affected sections say so explicitly;
no factor weight was changed.

---

## Method: why the benchmark could not answer these questions

The existing 62-case benchmark is made of realistic datasets, so a dozen factors move at
once. It can show that a factor *correlates* with a ranking outcome; it cannot show that a
factor *causes* one. Every result in this document comes instead from a **controlled pair**:
two candidates identical except for the dimension under test.

The harness verifies isolation rather than assuming it. It recomputes all ten factors for
both candidates and reports which ones moved. This check was not decorative — it rejected
four of the first attempts:

| Attempted pair | Extra factor that moved | Fix applied |
| --- | --- | --- |
| keyword vs. domain distractor | `feature_match`, `target_match`, `compatibility` | moved to a project-side concept sweep |
| target: wrong vs. right | `keyword_overlap` | removed the target word from the project's concept list |
| task: regression vs. classification | `compatibility` | kept, and recorded as a designed coupling |
| task-only tie-break at a wide age gap | `recency` | pinned the comparison date |

`task_match` → `compatibility` is not a harness defect. Declaring the wrong task is
simultaneously a task mismatch and a compatibility contradiction, so both factors move by
design. That coupling is declared in `COUPLED_DIMENSIONS` so it is not mistaken for an
accident, and the rest of Section 1 is verified isolated.

Result: **7 of 8 candidate-side pairs are strictly isolated; the 8th moves exactly the
documented coupling set.**

---

## SECTION 1 — Controlled single-dimension pairs

### OBSERVED FACT

| dimension | low → high | factor Δ | points | score Δ | moved factors | isolated |
| --- | --- | --- | ---: | ---: | --- | --- |
| verification | ai → verified | +0.75 | **11.25** | 11.25 | verification | yes |
| target_match | wrong → right target | +1.00 | **13.00** | 13.00 | target_match | yes |
| feature_match | 1 of 3 → all 3 | +0.67 | **10.67** | 10.66 | feature_match | yes |
| recency | 6 yr old → 30 days | +0.97 | **1.94** | 1.94 | recency | yes |
| quality | usability 0.1 → 0.8 | +0.70 | **5.60** | 5.60 | quality | yes |
| domain_relevance | agriculture → retail | +0.35 | **5.60** | 5.60 | domain_relevance | yes |
| task_match | regression → classification | +1.00 | **8.00** | 12.80 | task_match, compatibility | no (designed) |
| popularity | 1k → 10k downloads | +0.12 | **0.58** | 0.59 | popularity | yes |

A wrong declared domain scores **0.50**, not 0 — a partial penalty for being off-topic.
A wrong declared target scores **0.00** with `unmatched_terms = ["churn"]`.

### INTERPRETATION

- The controlled evidence spans **0.58 to 13.00 points** for a single factor — a 22× range.
  Popularity and recency are at the bottom of that range, not because they are unimportant
  but because their formulas compress hard (Sections 3 and 4).
- `task_match` costs 8.00 points nominally but 12.80 observed, because the same wrong
  declaration is charged twice: once by `task_match` and once by `compatibility`. Any weight
  discussion that moves `task_match` must account for `compatibility` moving with it.

### RECOMMENDATION (not applied)

- Treat `task_match` + `compatibility` as one coupled lever when tuning; do not read a
  `task_match` ablation as a single-factor result.

---

## SECTION 1b — `keyword_overlap` cannot be isolated on the candidate side

### OBSERVED FACT

Holding the candidate byte-identical and varying only the project's concept count:

| project concepts | keyword_overlap | points | domain | feature | target | score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.6667 | 6.00 | 0.85 | 1.00 | 1.00 | 90.40 |
| 2 | 0.5000 | 4.50 | 0.85 | 1.00 | 1.00 | 88.90 |
| 1 | **1.0000** | **9.00** | 0.85 | 1.00 | 1.00 | **93.40** |

No other factor moved in any step. Every candidate-side attempt to move `keyword_overlap`
also moved `feature_match` or `target_match`, because all three read the same published
evidence.

### INTERPRETATION

- `keyword_overlap` is not independently controllable through candidate metadata. Its
  effective unit of variation is the project's own concept list, so candidate-side
  "keyword sensitivity" claims are not measurable in isolation.
- Narrowing a project's concept list *raises* the score of an unchanged candidate
  (88.90 → 93.40). The factor rewards matching a short concept list, so a sloppier project
  specification produces better keyword scores for the same data.

---

## SECTION 2 — Relevance ladder

### OBSERVED FACT

| rung | declared features | relevance | feature_match | score |
| --- | --- | ---: | ---: | ---: |
| full | 3 of 3 | 92.29 | 1.0000 | 90.40 |
| two-of-three | 2 of 3 | 84.67 | 0.6667 | 85.07 |
| one-of-three | 1 of 3 | 72.76 | 0.3333 | 76.74 |

Smallest adjacent gap: **5.33 points**.

---

## SECTION 3 — Popularity: no tipping point exists

### OBSERVED FACT

Bisecting the download ratio over **10⁰ … 10¹²** (a trillion-fold inflation) on each rung
pair, no ordering ever flips:

| weaker | stronger | relevance gap | score gap | ratio needed to flip |
| --- | --- | ---: | ---: | --- |
| two-of-three | full | 7.62 | 5.33 | **none within 10¹²** |
| one-of-three | two-of-three | 11.91 | 8.33 | **none within 10¹²** |
| one-of-three | full | 19.53 | 13.66 | **none within 10¹²** |

Physical ceiling of the factor, swept independently:

| sweep | min | max | total swing |
| --- | ---: | ---: | ---: |
| downloads 10⁰…10¹², votes pinned | 0.93 | 4.25 | **3.32** |
| downloads **and** votes 10⁰…10¹² | 0.29 | 5.00 | **4.71** |

Even with both download and vote counts inflated a trillion-fold, popularity never exceeds
its full nominal weight of 5.00 points.

### INTERPRETATION

- The reason no tipping point exists is arithmetic, not tuning: popularity's maximum
  possible swing (4.71) is smaller than the smallest relevance gap in the ladder (5.33).
  **On this ladder popularity cannot decide any contest, at any download count.**
- The earlier corpus observation that zeroing popularity changed no benchmark case is
  therefore explained. It is not evidence that the factor is useless — it is evidence that
  its dynamic range is too narrow to reach any decision boundary.

### RECOMMENDATION (not applied)

- If popularity is meant to break ties between similarly relevant datasets, its range must
  be widened; at present it cannot. If it is only a mild prior, its current behaviour is
  correct and should be documented as such rather than "fixed".

---

## SECTION 4 — Recency: no tipping point exists

### OBSERVED FACT

Bisecting candidate age over **0 … 3650 days** on each rung pair, no ordering ever flips:

| weaker | stronger | relevance gap | score gap | age at which the fresher weaker one wins |
| --- | --- | ---: | ---: | --- |
| two-of-three | full | 7.62 | 5.33 | **never** |
| one-of-three | two-of-three | 11.91 | 8.33 | **never** |
| one-of-three | full | 19.53 | 13.66 | **never** |

Ceiling: recency contributes **2.00** points at age 0 and **0.00** at and beyond the 1825-day
horizon. Total swing: **2.00 points**. Decay is exactly linear: 1.0 / 0.8 / 0.6 / 0.4 / 0.2 /
0.0 at 0 / 365 / 730 / 1095 / 1460 / 1825 days.

### INTERPRETATION

- Recency's entire dynamic range (2.00) is less than half the smallest relevance gap
  (5.33). Like popularity, it **cannot** flip any contest on this ladder. It can only act
  as a tie-break, and only between candidates already within 2 points.
- The 1825-day hard clip means recency cannot distinguish a 6-year-old dataset from a
  30-year-old one. Both score zero.

---

## SECTION 5 — Verification is the dominant lever

### OBSERVED FACT

| rung | relevance | verified | unverified | step | verified wins |
| --- | ---: | ---: | ---: | ---: | --- |
| full | 92.29 | 90.40 | 79.15 | **11.25** | yes |
| two-of-three | 84.67 | 85.07 | 70.27 | **14.80** | yes |
| one-of-three | 72.76 | 76.74 | 61.93 | **14.81** | yes |

Verification has exactly two reachable values: **15.00** and **3.75**.

Provenance combinations:

| source_type / status | verification score | points |
| --- | ---: | ---: |
| verified_external / verified | 1.00 | 15.00 |
| verified_external / unverified | 0.25 | 3.75 |
| ai_suggested / unverified | 0.25 | 3.75 |
| **ai_suggested / verified** | **0.25** | **3.75** |

The step exceeds 11.25 on the lower rungs because provenance also changes *evidence
availability*: with `ai_suggested`, a declared feature that is absent becomes `unknown`
instead of `mismatch`, judged coverage falls from 1.00 to 0.67, and `feature_match`
contributes **+3.56** points to the verified candidate. Verified total:
11.25 (verification) + 3.56 (feature_match) = **14.81**.

### INTERPRETATION

- Verification is the only factor that can single-handedly overturn a relevance gap on this
  ladder. It is also the factor with the fewest reachable values: it is a near-binary step,
  not a gradient.
- An AI-suggested record cannot reach full credit even when its status claims `verified`.
  Provenance caps it at 0.25, so the `verification_status` field alone cannot promote a
  model-generated record. This is a deliberate-looking guard, and it holds.
- The double effect above is a coupling worth naming: **losing verification costs both
  direct points and judged coverage**, so the cost of unverified provenance is larger than
  the weight suggests, and grows as the dataset declares more that it does not have.

---

## SECTION 5b — What each publisher factor can physically do

### OBSERVED FACT

| factor | swept | min | max | total swing |
| --- | --- | ---: | ---: | ---: |
| popularity | downloads 10⁰…10¹², votes pinned | 0.93 | 4.25 | 3.32 |
| popularity | downloads and votes 10⁰…10¹² | 0.29 | 5.00 | 4.71 |
| recency | age 0…10000 days | 0.00 | 2.00 | 2.00 |
| verification | all four provenance combinations | 3.75 | 15.00 | 11.25 |

---

## SECTION 6 — Denominator inflation: **fixed**

### BEFORE (observed, then corrected)

| case | counted weight | normalisation | score | silent factors | verification points |
| --- | ---: | ---: | ---: | --- | ---: |
| A — all judgeable | 100.00 | 1.0000 | **90.40** | — | 15.00 |
| B — publisher metadata absent | 85.00 | 1.1765 | **93.65** | popularity, quality, recency | 17.65 |
| C — B + target undeclared | 85.00 | 1.1765 | 74.82 | + target_match | 17.65 |
| D — B + no target, features or tasks | 85.00 | 1.1765 | 65.41 | + task_match | 17.65 |

Per-factor effect under thin evidence (case B):

| factor | score | counted | weight | contribution | × nominal |
| --- | ---: | --- | ---: | ---: | ---: |
| verification | 1.00 | yes | 15 | 17.65 | **1.176** |
| feature_match | 1.00 | yes | 16 | 18.82 | **1.176** |
| target_match | 1.00 | yes | 13 | 15.29 | **1.176** |
| compatibility | 1.00 | yes | 8 | 9.41 | **1.176** |
| task_match | 1.00 | yes | 8 | 9.41 | **1.176** |
| recency | — | **no** | 2 | 0.00 | 0 |
| quality | — | **no** | 8 | 0.00 | 0 |
| popularity | — | **no** | 5 | 0.00 | 0 |
| keyword_overlap | 0.6667 | yes | 9 | 7.06 | 0.784 |
| domain_relevance | 0.85 | yes | 16 | 16.00 | 1.000 |

The defect: **case B outranked case A by 3.25 points.** When 15 weight units went silent, the
surviving 85 were scaled by 100/85 = 1.176, so verification was paid 17.65 instead of 15.00 — a
17.6% bonus for having less to say. Silent *requirement* factors already behaved correctly
(stayed in the denominator at zero credit), so one function contained two contradictory rules.

### AFTER (current behaviour)

The denominator is the **nominal** weight of the factor set being scored — all 100 for
`ranking_score`, 61 for `relevance_score`, 15 for `quality_score` — and never varies with how
much evidence a candidate carries.

| case | denominator | normalisation | score | max × nominal | silent factors | verification points |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| A — all judgeable | 100 | 1.0000 | **90.40** | 1.0000 | — | 15.00 |
| B — publisher metadata absent | 100 | 1.0000 | **79.60** | 1.0000 | popularity, quality, recency | 15.00 |
| C — B + target undeclared | 100 | 1.0000 | 63.60 | 1.0000 | + target_match | 15.00 |
| D — B + no target, features or tasks | 100 | 1.0000 | 55.60 | 1.0000 | + task_match, task_match | 15.00 |

Per-factor effect under thin evidence — **no factor exceeds its nominal weight anywhere**:

| factor | score | weight | contribution | × nominal |
| --- | ---: | ---: | ---: | ---: |
| verification | 1.00 | 15 | 15.00 | 1.0000 |
| feature_match | 1.00 | 16 | 16.00 | 1.0000 |
| target_match | 1.00 | 13 | 13.00 | 1.0000 |
| compatibility | 1.00 | 8 | 8.00 | 1.0000 |
| task_match | 1.00 | 8 | 8.00 | 1.0000 |
| domain_relevance | 0.85 | 16 | 13.60 | 0.8500 |
| keyword_overlap | 0.6667 | 9 | 6.00 | 0.6667 |
| popularity | — | 5 | 0.00 | 0 |
| quality | — | 8 | 0.00 | 0 |
| recency | — | 2 | 0.00 | 0 |

Case B now scores **79.60 against case A's 90.40** — missing metadata costs the 10.80 points of
weight it forfeits, and inflates nothing.

### SECTION 6a — Removing evidence never raises a score

Each step removes one more piece of evidence from the same candidate:

| step | denominator | score | change | increased? |
| --- | ---: | ---: | ---: | --- |
| complete evidence | 100 | 90.40 | — | no |
| no popularity metadata | 100 | 87.90 | −2.50 | no |
| no usability rating | 100 | 81.50 | −6.40 | no |
| no publication date | 100 | 79.60 | −1.90 | no |
| no declared target | 100 | 63.60 | −16.00 | no |
| no target, features or tasks | 100 | 55.60 | −8.00 | no |

**6/6 steps non-increasing.** The denominator reads 100 in every row.

### A second, independent inflation path — also fixed

The monotonicity check found a defect the denominator fix had not covered. Removing `vote_count`
**raised** a score (71.33 → 71.47), because `_popularity` divided the 0.7/0.3 download/vote blend
by the weight of the parts a candidate happened to publish:

```python
total_weight = sum(w for w, _ in parts)          # 0.7 when votes are absent
score = sum(w * v for w, v in parts) / total_weight
```

Any dataset with no vote count had its download term multiplied by 1/0.7 = **1.43**. The fix
divides by the nominal blend weight (`POPULARITY_BLEND_WEIGHT = 1.0`), so an absent part forfeits
its share and inflates nothing — the same rule the score denominator follows. This is the same
defect class one level down: UNKNOWN rescaling the evidence that *was* present.

### Audit correction

The audit's inflation check compared each factor's share of the achieved score against
`weight / 100`. That fires on every candidate simply for scoring below 100, so it reported **all
ten factors as inflated** on arithmetic that was already correct — 11 warnings on the benchmark,
every one false. It now tests `contribution / weight`, which is 1.0 by construction for a
full-evidence candidate and cannot exceed 1.0 while the denominator is pinned. Benchmark audit
warnings: **11 → 0**, errors 0 → 0.

---

## SECTION 7 — UNKNOWN vs MATCH vs MISMATCH

### OBSERVED FACT

Candidates identical except for `target_description`, against a project whose target is `churn`:

| case | target score | target points | compatibility | reported status | dataset score | separates from silence |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| MATCH — "whether the customer churns" | 1.00 | 13.00 | 1.00 | `churn=match` | **93.40** | — |
| MISMATCH resolvable — "whether the applicant defaults on the loan" | 0.00 | 0.00 | **0.50** | `churn=mismatch` | **76.40** | `compatibility` |
| MISMATCH unresolved — "the number of support tickets opened" | 0.00 | 0.00 | 1.00 | `churn=mismatch` | **80.40** | none |
| UNKNOWN — field absent | `None` | 0.00 | 1.00 | `churn=unknown` | **80.40** | — |
| UNKNOWN — declared absence | `None` | 0.00 | 1.00 | `churn=unknown` | **80.40** | — |

`keyword_overlap` held at 1.00 across all five cases, confirming the isolation.

### INTERPRETATION

- **The three-way distinction is structurally intact.** MATCH scores 1.00, MISMATCH scores 0.00,
  UNKNOWN scores `None`, and the reported statuses are `match` / `mismatch` / `unknown`. UNKNOWN
  is never laundered into a match, and a statement that the label is not provided is read as
  absence rather than contradiction.
- **`MATCH > UNKNOWN > MISMATCH` holds**, at 93.40 > 80.40 > 76.40, with **no factor weight
  changed**. The separation does not come from `target_match`, which pays both zero — it comes
  from the pre-existing contradiction channel in `compatibility.py`. A declared target the
  concept index resolves to a *different* label space (`default` against the project's `churn`)
  costs 0.5 of compatibility; silence costs nothing. This is the mechanism the module was built
  for, not a new penalty invented to satisfy the ordering.
- **The honest limit:** a target the index cannot resolve to any label space is not separated.
  Support-ticket volume is genuinely not churn, but no label space matches it, so the
  contradiction rule declines to fire and that candidate ties with silence at 80.40. The case is
  reported rather than dropped because a friendlier example would hide where the mechanism stops
  working. Widening `TargetConceptIndex` is the fix, and it is a knowledge-base change, not a
  scoring change.
- Retracting a *declared contradiction* raises a score, and that is correct: going from
  "affirmatively the wrong label" to "says nothing about the label" is an improvement in evidence,
  not a loss of it. This is distinct from removing publisher metadata, which Section 6a shows can
  only lower a score.

---

## SECTION 8 — Adversarial keyword stuffing

### OBSERVED FACT

| candidate | domain | keyword_overlap | keyword points | domain_relevance | relevance | score | rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| honest (retail, mentions 1 concept) | retail | 0.3333 | 3.00 | 0.85 | **80.38** | **82.07** | **1** |
| stuffed (transport, repeats all 3 concepts) | transport | **1.0000** | **9.00** | **0.00** | 61.90 | 69.14 | 2 |

### INTERPRETATION

- Keyword stuffing **works on its own factor** — it reaches full 1.0 overlap and gains 6.00
  points over the honest candidate. The factor cannot be gamed out of existence.
- It **does not work overall**. The 6.00 points gained are more than erased by losing
  domain relevance (13.60 → 0.00) and 18.48 points of relevance, for a net **−12.93**. The
  stuffing candidate ranks second.
- The domain factor is what stops the attack, not the keyword factor. If a stuffing
  candidate were in the correct domain, the 6-point keyword gain would survive unopposed.

---

## SECTION 9 — Same-relevance tie-breaks

### OBSERVED FACT

Relevance exactly equal (Δ = 0.000000) in every scenario; each moves exactly one factor:

| scenario | deciding factor | score Δ | winner | relevance equal |
| --- | --- | ---: | --- | --- |
| unverified provenance | verification | **−11.25** | the verified candidate | yes |
| recency: 5 years older | recency | −1.90 | the newer candidate | yes |
| quality: usability 0.8 → 0.95 | quality | +1.20 | the more usable candidate | yes |
| popularity: 10× more downloads | popularity | +0.59 | the more popular candidate | yes |
| recency: 2 months newer | recency | +0.07 | the newer candidate | yes |

Measured tie-break strength: **verification (11.25) > recency (2.00) > quality (1.20) >
popularity (0.59)**.

### INTERPRETATION

- When relevance ties, the ordering is decided almost entirely by provenance. Verification
  alone is 5.6× stronger than recency and 19× stronger than popularity.
- Quality's tie-break is modest because the sweep moves usability from 0.8 to 0.95; its full
  usable range is 8.00 points.
- The practical consequence: two equally relevant datasets are ordered by **trust first,
  freshness second, polish third, popularity last**.

---

## Summary of OBSERVED FACT

1. Controlled single-factor response spans **0.58–13.00 points**; `task_match` is charged
   twice (8.00 nominal, 12.80 observed) via `compatibility`.
2. `keyword_overlap` has **no candidate-side isolation** — it shares its evidence source with
   `feature_match` and `target_match`.
3. Popularity cannot flip any relevance gap at any download count up to **10¹²**; its total
   swing is **4.71** against a smallest gap of **5.33**.
4. Recency cannot flip any relevance gap at any age up to **3650 days**; total swing
   **2.00**, hard-clipped to zero at the 1825-day horizon.
5. Verification is the only factor that overturns a relevance gap: step **11.25**, rising
   to **14.81** when lost provenance also lowers judged coverage.
6. An `ai_suggested` record cannot reach full verification credit even with
   `verification_status = verified`.
7. **Missing publisher metadata outranked present metadata by 3.25 points** (93.65 vs 90.40)
   because the shared denominator inflated the survivors by 100/85 = 1.176. **Now fixed**: the
   denominator is pinned at nominal weight and the same pair scores 79.60 vs 90.40.
8. Silent *requirement* and *publisher* factors now behave identically: each keeps its full
   weight in the denominator at zero credit and inflates nobody. Max observed ratio 1.0000.
9. Removing evidence never raises a score: **6/6 monotonicity steps non-increasing**, with the
   denominator reading 100 throughout. A second inflation path was found and fixed here too —
   absent votes scaled the download term by 1/0.7 = 1.43 inside `_popularity`.
10. UNKNOWN / MATCH / MISMATCH remain structurally distinct in reported status; UNKNOWN never
    becomes positive evidence. `MATCH > UNKNOWN > MISMATCH` now holds at 93.40 > 80.40 > 76.40,
    separated by the existing `compatibility` contradiction channel with **no weight change**.
    An unresolvable contradictory target still ties with silence at 80.40 — see Section 7.
11. Keyword stuffing reaches full keyword credit (+6.00) but loses 12.93 net and ranks
    second; the **domain** factor is what prevents the exploit, not the keyword factor.
12. At equal relevance the order is decided by **verification > recency > quality >
    popularity**.

## Summary of INTERPRETATION

- The 22× spread in controlled single-factor response is not a weight-tuning problem. Three
  factors (popularity, recency, quality) are compressed into a band too narrow to reach any
  decision boundary on this ladder, while verification behaves as a near-binary gate. The
  function is therefore much closer to "relevance, gated by trust" than to a weighted blend.
- Denominator renormalisation was the one place where the measurement contradicted the
  intuition the weights suggest: thin evidence inflated the factors that *were* present, so a
  dataset could buy relevance points by withholding publisher metadata. Pinning the denominator
  removes the reward, and **no ranking-quality metric moved** — NDCG@5, MRR, precision@1,
  recall and the unknown-as-mismatch rate are all unchanged to four decimals, because the
  benchmark's top candidates are fully evidenced. The defect was real but confined to sparse
  candidates.
- Two distinct inflation paths existed, at two levels. The score denominator rescaled across
  factors (peak 1.176×), and the popularity blend rescaled *within* a factor (1.43× on downloads
  when votes were absent). Both were the same mistake: dividing by the evidence that happened to
  be present rather than by the evidence that was declared possible.
- An audit is only as good as the invariant it tests. The inflation check compared against
  `weight / 100` and so reported all ten factors as inflated on correct arithmetic — 11 false
  warnings that would have buried a real 1.587× finding. Testing `contribution / weight`
  directly reduced the benchmark to 0 errors and 0 warnings while keeping the defect visible.
- UNKNOWN and MISMATCH are numerically separable **only where the contradiction machinery can
  resolve the conflicting target to a known label space**. That is a knowledge-base limit, not a
  scoring limit, and it is the boundary a future target-concept expansion would move.
- No result here justifies removing or re-weighting any factor. Sections 3 and 4 in
  particular show a factor being *unable to matter*, which is a statement about its dynamic
  range and not about its value.

## What was changed

Applied, with test coverage pinning each item:

1. **Score denominator pinned to nominal weight** (`_contributions`, `_weighted_score`,
   `_relevance_score`, `_quality_score` in `ranking.py`). Silent factors keep their weight and
   earn zero. This is the Section 6 fix; the `requirements` argument became unnecessary and was
   removed.
2. **Popularity blend divided by nominal blend weight** (`_popularity`). `POPULARITY_BLEND_WEIGHT`
   is now an explicit module constant.
3. **Audit inflation check moved to `contribution / weight`**, with a new
   `FactorAuditEntry.observed_max_ratio` field. Benchmark warnings 11 → 0.
4. **Stress probe extended**: `evidence_removal_monotonicity()`, `max_inflation` on every probe,
   and a second `MISMATCH-unresolved` case so the limit is visible.

**No factor weight changed.** Verification remains 15, domain 16, feature 16, target 13, and the
rest as before.

## Open items (not changed)

1. **Popularity and recency range (Sections 3, 4).** Decide deliberately whether these are
   intended as tie-breaks only. If they should ever matter more, their formulas must widen; no
   weight change alone can achieve it inside the current compression.
2. **`task_match` + `compatibility` coupling (Section 1).** Account for it in any future weight
   change; consider whether double-charging is intended.
3. **Unresolvable contradictions (Section 7).** Extending `TargetConceptIndex` so phrases like
   "the number of support tickets opened" resolve to a label space would make every MISMATCH
   strictly worse than silence. This is knowledge-base work, not scoring work.
4. **Keyword stuffing (Section 8).** Currently mitigated incidentally by the domain factor. Worth
   a regression fixture, since the protection is not designed.

---

## Reproducing

```
cd datapilot/backend
python -m scripts.run_stress                      # all sections, writes stress_results.json
python -m pytest tests/unit/test_evaluation_stress.py -q
```

The harness is offline and deterministic: no network, no clock reads outside
`BENCHMARK_NOW`, and no ranking state persists between runs. The 60 harness tests assert
pair isolation, score reconstruction, denominator bookkeeping, evidence-removal monotonicity,
the `MATCH > UNKNOWN > MISMATCH` ordering, sweep stability, and that the harness cannot alter the
configured weights.