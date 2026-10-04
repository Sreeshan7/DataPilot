# Dataset Ranking Evaluation

Offline evaluation of `DatasetRankingEngine` against the deterministic ranking benchmark (version 1.0.0). No network access, no model calls, no live APIs: every case is a fixture and every number below is reproducible from the committed corpus.

## Run configuration

- benchmark cases: **62**
- graded candidates: **173**
- evaluation clock: `2026-09-28T12:00:00+00:00` (fixed, so recency is stable)
- ranker: `DatasetRankingEngine`
- weights:
  - `ranking_weight_compatibility` = 8
  - `ranking_weight_domain_relevance` = 16
  - `ranking_weight_feature_match` = 16
  - `ranking_weight_keyword_overlap` = 9
  - `ranking_weight_popularity` = 5
  - `ranking_weight_quality` = 8
  - `ranking_weight_recency` = 2
  - `ranking_weight_target_match` = 13
  - `ranking_weight_task_match` = 8
  - `ranking_weight_verification` = 15

## 1. Ranking quality

Macro-averaged over cases. Relevance cut is grade >= 2; NDCG uses the full grade scale.

| metric | value |
| --- | ---: |
| `hit_rate@1` | 0.8226 |
| `hit_rate@10` | 0.8548 |
| `hit_rate@3` | 0.8548 |
| `hit_rate@5` | 0.8548 |
| `mrr` | 0.9113 |
| `mrr_highly_relevant` | 0.8360 |
| `ndcg@1` | 0.9124 |
| `ndcg@10` | 0.9551 |
| `ndcg@3` | 0.9537 |
| `ndcg@5` | 0.9546 |
| `precision@1` | 0.9032 |
| `precision@10` | 0.1355 |
| `precision@3` | 0.4516 |
| `precision@5` | 0.2710 |
| `recall@1` | 0.7016 |
| `recall@10` | 0.9194 |
| `recall@3` | 0.9194 |
| `recall@5` | 0.9194 |

Graded candidate distribution: grade 3: 59, grade 2: 25, grade 1: 36, grade 0: 43, grade -1: 10

### Retrieval recall vs ranking quality

- **retrieval recall** (pool contained a grade-3 dataset): 0.8548
- **ranking quality on answered cases** (NDCG@5): 0.9866
- **ranking quality across all cases** (NDCG@5): 0.9546

The gap between these two numbers is the part of the score that query construction, not ranking, is responsible for.

## 2. Diagnostic rates

| diagnostic | rate |
| --- | ---: |
| clearly incompatible datasets in the top 3 (lower is better) | 0.0430 |
| cases whose rank 1 is not a grade-3 dataset | 0.1774 |
| cases with no grade-3 dataset in the top 3 | 0.1452 |
| requirement verdicts read as MISMATCH without evidence | 0.0000 |

## 3. Quality by category

| category | cases | P@1 | NDCG@5 | hit@3 |
| --- | ---: | ---: | ---: | ---: |
| `computer_vision` | 5 | 1.0000 | 0.9966 | 1.0000 |
| `cybersecurity` | 8 | 0.8750 | 0.9539 | 0.7500 |
| `edge_case` | 8 | 0.7500 | 0.8750 | 0.3750 |
| `education` | 8 | 1.0000 | 0.9592 | 1.0000 |
| `finance` | 8 | 0.8750 | 0.9531 | 0.8750 |
| `healthcare` | 8 | 0.7500 | 0.9143 | 0.8750 |
| `nlp` | 6 | 1.0000 | 0.9925 | 1.0000 |
| `retail` | 6 | 1.0000 | 1.0000 | 1.0000 |
| `time_series` | 5 | 1.0000 | 1.0000 | 1.0000 |

## 4. Failure taxonomy

| category | cases |
| --- | ---: |
| `QUERY_RECALL_FAILURE` | 9 |
| `CONTRADICTION_MISSED` | 6 |
| `WRONG_DOMAIN` | 5 |
| `AI_EVIDENCE_OVERTRUST` | 3 |
| `MISSING_TARGET_EVIDENCE` | 3 |
| `SEMANTIC_EQUIVALENCE_MISSED` | 3 |
| `WRONG_TARGET` | 2 |
| `FEATURE_UNKNOWN` | 2 |
| `METADATA_INSUFFICIENT` | 2 |
| `FEATURE_MISMATCH` | 1 |

Retrieval vs ranking: `none` 40, `ranking` 21, `retrieval` 1

### Failing cases

- `cyber-adversarial-weak` [cybersecurity] - QUERY_RECALL_FAILURE
- `cyber-ai-surfaced-dataset` [cybersecurity] - AI_EVIDENCE_OVERTRUST, WRONG_DOMAIN
- `cyber-rare-attack-subtype` [cybersecurity] - CONTRADICTION_MISSED, QUERY_RECALL_FAILURE
- `edge-ai-rationale-trap` [edge_case] - QUERY_RECALL_FAILURE
- `edge-all-incompatible` [edge_case] - CONTRADICTION_MISSED, METADATA_INSUFFICIENT, QUERY_RECALL_FAILURE
- `edge-all-unknown-evidence` [edge_case] - QUERY_RECALL_FAILURE
- `edge-domain-only-project` [edge_case] - FEATURE_UNKNOWN, METADATA_INSUFFICIENT, MISSING_TARGET_EVIDENCE, QUERY_RECALL_FAILURE, SEMANTIC_EQUIVALENCE_MISSED
- `edge-retrieval-miss` [edge_case] - QUERY_RECALL_FAILURE
- `edu-final-grade-semantic` [education] - FEATURE_MISMATCH
- `edu-library-demand` [education] - WRONG_TARGET
- `edu-plagiarism-detection` [education] - CONTRADICTION_MISSED
- `edu-student-performance` [education] - AI_EVIDENCE_OVERTRUST
- `finance-target-conflict-approved` [finance] - CONTRADICTION_MISSED, QUERY_RECALL_FAILURE
- `health-adverse-effect-nlp` [healthcare] - WRONG_TARGET
- `health-heart-disease` [healthcare] - AI_EVIDENCE_OVERTRUST
- `health-readmission-sparse` [healthcare] - FEATURE_UNKNOWN, MISSING_TARGET_EVIDENCE, SEMANTIC_EQUIVALENCE_MISSED
- `health-target-conflict` [healthcare] - CONTRADICTION_MISSED, QUERY_RECALL_FAILURE
- `nlp-banking-intent` [nlp] - WRONG_DOMAIN
- `nlp-hate-speech` [nlp] - WRONG_DOMAIN
- `nlp-multilingual-sentiment` [nlp] - WRONG_DOMAIN
- `nlp-sarcasm` [nlp] - WRONG_DOMAIN
- `retail-basket-analysis` [retail] - MISSING_TARGET_EVIDENCE, SEMANTIC_EQUIVALENCE_MISSED

## 5. Factor influence

Average share of a final score, per factor, across every ranked candidate.

| factor | average share of score |
| --- | ---: |
| `verification` | 0.2561 |
| `domain_relevance` | 0.1432 |
| `compatibility` | 0.1232 |
| `quality` | 0.1096 |
| `target_match` | 0.0826 |
| `feature_match` | 0.0819 |
| `popularity` | 0.0686 |
| `task_match` | 0.0584 |
| `keyword_overlap` | 0.0480 |
| `recency` | 0.0284 |

### Score decomposition audit

- contributions sum to the displayed score: **True** (largest error 0.01)
- arithmetic errors: 0
- duplicate factors: 0
- factors exceeding their nominal share: 0


| factor | meaning | weight | nominal share | observed max share | unknown? | mismatch? |
| --- | --- | ---: | ---: | ---: | :---: | :---: |
| `domain_relevance` | Does the dataset's declared subject matter overlap the project's domain and subdomain? | 16 | 0.16 | 0.32 | no | no |
| `keyword_overlap` | How many of the project's query terms appear in the dataset's published text? Vocabulary overlap, not semantic equivalence. | 9 | 0.09 | 0.13 | no | no |
| `feature_match` | Which required features are visible in the dataset's published schema, feature description or prose? | 16 | 0.16 | 0.31 | yes | yes |
| `target_match` | Does the dataset publish a label space compatible with the project's target concept? | 13 | 0.13 | 0.27 | yes | yes |
| `task_match` | Is the requested ML task among the dataset's declared task types? | 8 | 0.08 | 0.19 | yes | yes |
| `compatibility` | Positive evidence that the dataset is something the project cannot use. Scored down from 1.0 by typed contradictions. | 8 | 0.08 | 0.68 | no | yes |
| `quality` | Registry usability rating for the dataset. | 8 | 0.08 | 0.23 | yes | no |
| `popularity` | Adoption signals: download and vote volume, log-scaled. | 5 | 0.05 | 0.15 | yes | no |
| `recency` | How recently the dataset was updated. | 2 | 0.02 | 0.06 | yes | no |
| `verification` | How the candidate was obtained: confirmed registry record versus a model's suggestion. | 15 | 0.15 | 0.52 | yes | no |

## 6. Ablation

Each signal is removed by setting its weight to zero and re-running the whole benchmark. Deltas are variant minus baseline; negative means the baseline was better, i.e. the removed signal was earning its place.

| signal removed | d(NDCG@5) | d(NDCG@10) | d(MRR) | d(P@1) | d(hit@3) | top-1 changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `popularity` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 |
| `recency` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 |
| `quality` | +0.0006 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 |
| `verification` | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 |
| `domain_relevance` | -0.0010 | -0.0010 | +0.0000 | +0.0000 | +0.0000 | 0 |
| `keyword_overlap` | -0.0078 | -0.0078 | -0.0081 | -0.0161 | +0.0000 | 1 |
| `feature_match` | -0.0117 | -0.0106 | -0.0081 | -0.0161 | +0.0000 | 2 |
| `target_match` | -0.0147 | -0.0137 | -0.0161 | -0.0323 | +0.0000 | 3 |
| `task_match` | -0.0042 | -0.0032 | +0.0000 | +0.0000 | +0.0000 | 2 |
| `compatibility` | +0.0004 | +0.0004 | +0.0000 | +0.0000 | +0.0000 | 0 |

## 7. Sensitivity

One input is perturbed at a time on a fixed sample of cases and the engine is re-run end to end. Measures what each piece of evidence is actually worth in ranking terms.

| perturbation | runs | mean abs score delta | max abs score delta | mean abs rank delta | runs that moved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `popularity x3` | 23 | 0.1935 | 0.2800 | 0.0000 | 0 |
| `verification removed` | 23 | 11.1630 | 11.2500 | 0.3043 | 6 |
| `target evidence removed` | 13 | 2.4646 | 25.0000 | 0.2308 | 1 |
| `feature evidence removed` | 11 | 7.7527 | 20.5000 | 0.2727 | 1 |
| `domain evidence removed` | 23 | 12.5661 | 16.0000 | 0.3043 | 6 |
| `task evidence removed` | 13 | 7.4046 | 10.4000 | 0.3077 | 2 |
| `recency penalty` | 23 | 1.6726 | 1.9700 | 0.1739 | 4 |
| `target contradicted` | 24 | 1.3350 | 25.0000 | 0.1250 | 1 |
| `evidence quality` | 1 | 7.2500 | 7.2500 | 0.0000 | 0 |

## 8. Per-case diagnostics

Every case, its ranking, and the factors that produced each position. `evidence/unknown/publisher` splits each candidate's score into the share driven by declared evidence, by unjudgeable factors, and by publisher signals.

### `edu-student-performance` - Predict university students' final academic performance from attendance, prior marks and study habits.

- category: `education` / difficulty: `standard`
- pooled candidates: 6
- note: Baseline case: rich metadata on the right dataset, plus generic lookalikes.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Higher Education Students Academic Performance | 3 | 95.63 | 94.52 | 96.43 | 0.69/0.00/0.31 |
| 2 | Student Grades Archive | 2 | 68.61 | 60.86 | 73.42 | 0.62/0.00/0.38 |
| 3 | Student Habits Survey Study | 2 | 51.77 | 68.61 | 0.00 | 0.93/0.00/0.07 |
| 4 | World University Rankings 2026 | 0 | 51.06 | 30.86 | 96.43 | 0.42/0.00/0.58 |
| 5 | University Enrollment Statistics | 0 | 49.40 | 30.86 | 85.36 | 0.44/0.00/0.56 |
| 6 | Examination Results Archive 2004-2018 | 1 | 47.76 | 34.45 | 57.64 | 0.48/0.02/0.50 |

**1. Higher Education Students Academic Performance** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.84 w=9 -> +7.56; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: student performance, attendance, examination, assignment scores, study hours; Dataset metadata references required features: attendance, previous examination marks, assignment scores, study hours; Dataset metadata references the prediction target: academic performance; Compatible with the required ML task(s): classification, regression; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Student Grades Archive** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.50 w=16 -> +8.00; `quality` 0.72 w=8 -> +5.76; `compatibility` 0.50 (contradiction: dataset target 'final grade' declares different label values than the project's 'academic performance') w=8 -> +4.00; `task_match` 0.50 w=8 -> +4.00; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `keyword_overlap` 0.00 w=9 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata references required features: previous examination marks, assignment scores; Dataset metadata references the prediction target: academic performance; Compatible with the required ML task(s): regression; Conflicts with the request: dataset target 'final grade' declares different label values than the project's 'academic performance'; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Student Habits Survey Study** (grade 2)
- factors: `feature_match` 1.00 w=16 -> +16.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `verification` 0.25 w=15 -> +3.75; `keyword_overlap` 0.34 w=9 -> +3.02; `domain_relevance` 0.00 w=16 -> +0.00; `popularity` UNKNOWN w=5 -> +0.00; `quality` UNKNOWN w=8 -> +0.00; `recency` UNKNOWN w=2 -> +0.00
- reasons: Dataset metadata matches project concepts: attendance, study hours; Dataset metadata references required features: attendance, previous examination marks, assignment scores, study hours; Dataset metadata references the prediction target: academic performance; Compatible with the required ML task(s): classification, regression; AI-suggested candidate from kaggle-fixture; not yet confirmed by an external dataset registry
- evidence note: unverified, so declared-field rules were not applied
**4. World University Rankings 2026** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (attendance, previous examination marks, assignment scores, study hours) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (academic performance) w=13 -> +0.00; `task_match` UNKNOWN (classification, regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: attendance, previous examination marks, assignment scores, study hours (feature); academic performance (target); classification, regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**5. University Enrollment Statistics** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (attendance, previous examination marks, assignment scores, study hours) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (academic performance) w=13 -> +0.00; `task_match` UNKNOWN (classification, regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: attendance, previous examination marks, assignment scores, study hours (feature); academic performance (target); classification, regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**6. Examination Results Archive 2004-2018** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `keyword_overlap` 0.17 w=9 -> +1.51; `feature_match` 0.25 w=16 -> +1.00; `recency` 0.47 w=2 -> +0.94; `target_match` UNKNOWN (academic performance) w=13 -> +0.00; `task_match` UNKNOWN (classification, regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: examination; Dataset metadata references required features: previous examination marks; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Could not confirm from published metadata: attendance, assignment scores, study hours (feature); academic performance (target); classification, regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-dropout-risk` - Predict whether a student drops out before graduation; the right dataset publishes almost no metadata.

- category: `education` / difficulty: `sparse_metadata`
- pooled candidates: 3
- note: Correct dataset has no feature/target descriptions, so the ranker must not treat silence as absence.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Higher Education Dropout Dataset | 2 | 90.07 | 91.51 | 73.42 | 0.71/0.00/0.29 |
| 2 | Primary School Enrolment by District | 0 | 49.40 | 30.86 | 85.36 | 0.44/0.00/0.56 |
| 3 | University Student Enrolment and Progression Records | 3 | 45.25 | 30.86 | 57.64 | 0.48/0.00/0.52 |

**1. Higher Education Dropout Dataset** (grade 2)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.61 w=9 -> +5.45; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: student dropout, enrolled credits, absences; Dataset metadata references required features: enrolled credits, absences, previous grades; Dataset metadata references the prediction target: dropout; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Primary School Enrolment by District** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (enrolled credits, absences, previous grades) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (dropout) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: enrolled credits, absences, previous grades (feature); dropout (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. University Student Enrolment and Progression Records** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `feature_match` UNKNOWN (enrolled credits, absences, previous grades) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (dropout) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: enrolled credits, absences, previous grades (feature); dropout (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-final-grade-semantic` - Target phrased as 'final grade'; the candidate publishes the same label under an institutional name.

- category: `education` / difficulty: `semantic`
- pooled candidates: 2
- note: Equivalence must be judged on the label space, not on the string 'final grade'.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Portuguese Secondary School Student Performance | 3 | 74.96 | 67.37 | 85.36 | 0.63/0.00/0.37 |
| 2 | GPA Collections by University | 0 | 64.06 | 49.43 | 96.43 | 0.54/0.00/0.46 |

**1. Portuguese Secondary School Student Performance** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.51 w=9 -> +4.56; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.00 w=16 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: final grade, secondary school; Dataset metadata references the prediction target: final grade; Compatible with the required ML task(s): classification, regression; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. GPA Collections by University** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (attendance) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `task_match` UNKNOWN (classification, regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata references the prediction target: final grade; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: attendance (feature); classification, regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-attendance-timeseries` - Predict daily attendance rate per course across a semester from dated register entries.

- category: `education` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Course Attendance Registers by Date | 3 | 93.97 | 94.52 | 85.36 | 0.70/0.00/0.30 |
| 2 | Monthly Course Attendance Summary | 1 | 74.31 | 68.99 | 73.42 | 0.65/0.00/0.35 |
| 3 | Campus Building Occupancy Counts | 0 | 51.06 | 30.86 | 96.43 | 0.42/0.00/0.58 |

**1. Course Attendance Registers by Date** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.84 w=9 -> +7.56; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: attendance, daily, course, semester, register; Dataset metadata references required features: attendance date, course id, student id; Dataset metadata references the prediction target: daily attendance rate; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Monthly Course Attendance Summary** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.67 w=16 -> +10.67; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.34 w=9 -> +3.02; `recency` 0.82 w=2 -> +1.64; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: attendance, course; Dataset metadata references required features: attendance date, student id; Dataset metadata references the prediction target: daily attendance rate; Publisher quality score 0.72 / 1.00; Could not confirm from published metadata: forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Campus Building Occupancy Counts** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (attendance date, course id, student id) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (daily attendance rate) w=13 -> +0.00; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: attendance date, course id, student id (feature); daily attendance rate (target); forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-course-recommendation` - Recommend the next course for a student from enrolment and rating interactions.

- category: `education` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | MOOC Course Enrolment and Rating Interactions | 3 | 86.18 | 85.95 | 73.42 | 0.70/0.00/0.30 |
| 2 | Library Book Loans Archive | 0 | 51.06 | 30.86 | 96.43 | 0.42/0.00/0.58 |
| 3 | Learning Management System Clickstream | 1 | 49.40 | 30.86 | 85.36 | 0.44/0.00/0.56 |

**1. MOOC Course Enrolment and Rating Interactions** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.75 w=16 -> +12.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: course recommendation, student, rating; Dataset metadata references required features: student id, course id, course rating; Dataset metadata references the prediction target: next recommended course; Compatible with the required ML task(s): recommendation; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Library Book Loans Archive** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (student id, course id, enrollment, course rating) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (next recommended course) w=13 -> +0.00; `task_match` UNKNOWN (recommendation) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: student id, course id, enrollment, course rating (feature); next recommended course (target); recommendation (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Learning Management System Clickstream** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (student id, course id, enrollment, course rating) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (next recommended course) w=13 -> +0.00; `task_match` UNKNOWN (recommendation) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: student id, course id, enrollment, course rating (feature); next recommended course (target); recommendation (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-plagiarism-detection` - Flag plagiarised submissions from submitted text, with a required label of plagiarism.

- category: `education` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Essay Corpus with Plagiarism Labels | 3 | 93.61 | 96.57 | 73.42 | 0.72/0.00/0.28 |
| 2 | Handwritten Digit Image Collection | -1 | 59.06 | 42.29 | 96.43 | 0.50/0.00/0.50 |
| 3 | Essay Topic Labelling Benchmark | 0 | 57.86 | 45.50 | 73.42 | 0.55/0.00/0.45 |

**1. Essay Corpus with Plagiarism Labels** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: plagiarism, submission, essay, token overlap; Dataset metadata references required features: submission text, token overlap; Dataset metadata references the prediction target: plagiarism; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Handwritten Digit Image Collection** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (submission text, token overlap) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Compatible with the required ML task(s): classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: submission text, token overlap (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Essay Topic Labelling Benchmark** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.25 w=9 -> +2.25; `recency` 0.82 w=2 -> +1.64; `feature_match` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: essay; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-library-demand` - Forecast future borrow counts per book title from checkout history.

- category: `education` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Public Library Loan History | 3 | 77.70 | 77.22 | 57.64 | 0.70/0.00/0.30 |
| 2 | Independent Bookstore Sales | 1 | 45.32 | 25.02 | 85.36 | 0.39/0.00/0.61 |

**1. Public Library Loan History** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.67 w=9 -> +6.05; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: library, borrow, checkout, book; Dataset metadata references required features: checkout history, book title, member id; Compatible with the required ML task(s): forecasting; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. Independent Bookstore Sales** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `keyword_overlap` 0.17 w=9 -> +1.51; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (checkout history, book title, member id) w=16 -> +0.00; `target_match` UNKNOWN (borrow count) w=13 -> +0.00
- reasons: Dataset metadata matches project concepts: book; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: checkout history, book title, member id (feature); borrow count (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edu-generic-term-trap` - Project description is dominated by generic education words; a popular aggregate dataset must not win.

- category: `education` / difficulty: `distractor`
- pooled candidates: 2
- note: Generic vocabulary (students, grades, performance) must not outweigh project-specific signals.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Course Withdrawal Microdata for Advising | 3 | 89.52 | 94.10 | 57.64 | 0.74/0.00/0.26 |
| 2 | Student Performance Indicators (Aggregated) | 0 | 52.88 | 33.45 | 96.43 | 0.44/0.00/0.56 |

**1. Course Withdrawal Microdata for Advising** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.81 w=9 -> +7.27; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: withdrawal, course load, advising, students; Dataset metadata references required features: withdrawal deadline, course load, advising contact; Dataset metadata references the prediction target: course withdrawal rate; Compatible with the required ML task(s): regression; Publisher quality score 0.62 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Student Performance Indicators (Aggregated)** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.20 w=9 -> +1.82; `feature_match` UNKNOWN (withdrawal deadline, course load, advising contact) w=16 -> +0.00; `target_match` UNKNOWN (course withdrawal rate) w=13 -> +0.00; `task_match` UNKNOWN (regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: students; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: withdrawal deadline, course load, advising contact (feature); course withdrawal rate (target); regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-diabetes-bmi` - Predict diabetes from BMI, glucose and insulin measurements.

- category: `healthcare` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Pima Indian Diabetes Dataset | 3 | 92.72 | 92.74 | 85.36 | 0.70/0.00/0.30 |
| 2 | Framingham Heart Study Cohort | 2 | 87.14 | 84.77 | 85.36 | 0.68/0.00/0.32 |
| 3 | Drug Review Sentiment Dataset | 0 | 43.06 | 19.43 | 96.43 | 0.32/0.00/0.68 |
| 4 | Medical Insurance Cost Dataset | 0 | 32.66 | 4.57 | 96.43 | 0.10/0.00/0.90 |

**1. Pima Indian Diabetes Dataset** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.70 w=9 -> +6.32; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: diabetes, bmi, blood glucose, insulin; Dataset metadata references required features: bmi, blood glucose, insulin, age; Dataset metadata references the prediction target: diabetes; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Framingham Heart Study Cohort** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.75 w=16 -> +12.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.53 w=9 -> +4.74; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: diabetes, bmi, blood glucose; Dataset metadata references required features: bmi, blood glucose, age; Dataset metadata references the prediction target: diabetes; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Drug Review Sentiment Dataset** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `compatibility` 0.00 (contradiction: dataset supports only sentiment analysis while the project requires classification) w=8 -> +0.00; `feature_match` UNKNOWN (bmi, blood glucose, insulin, age) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: bmi, blood glucose, insulin, age (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**4. Medical Insurance Cost Dataset** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.40 (contradiction: dataset supports only regression while the project requires classification) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (bmi, blood glucose, insulin, age) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (diabetes) w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Conflicts with the request: dataset supports only regression while the project requires classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: bmi, blood glucose, insulin, age (feature); diabetes (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-heart-disease` - Predict presence of heart disease from age, cholesterol and blood pressure.

- category: `healthcare` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Heart Disease Dataset (Cleveland) | 3 | 93.83 | 94.32 | 85.36 | 0.70/0.00/0.30 |
| 2 | Cardiology Cohort Study (unlisted) | 3 | 56.17 | 74.89 | 0.00 | 0.93/0.00/0.07 |
| 3 | Student Health and Lifestyle Survey | 0 | 49.40 | 30.86 | 85.36 | 0.44/0.00/0.56 |

**1. Heart Disease Dataset (Cleveland)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: heart disease, cholesterol, blood pressure, age; Dataset metadata references required features: age, cholesterol, resting blood pressure, maximum heart rate; Dataset metadata references the prediction target: heart disease; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Cardiology Cohort Study (unlisted)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `verification` 0.25 w=15 -> +3.75; `domain_relevance` 0.00 w=16 -> +0.00; `popularity` UNKNOWN w=5 -> +0.00; `quality` UNKNOWN w=8 -> +0.00; `recency` UNKNOWN w=2 -> +0.00
- reasons: Dataset metadata matches project concepts: heart disease, cholesterol, blood pressure, age; Dataset metadata references required features: age, cholesterol, resting blood pressure, maximum heart rate; Dataset metadata references the prediction target: heart disease; Compatible with the required ML task(s): classification; AI-suggested candidate from kaggle-fixture; not yet confirmed by an external dataset registry
- evidence note: unverified, so declared-field rules were not applied
**3. Student Health and Lifestyle Survey** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (age, cholesterol, resting blood pressure, maximum heart rate) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (heart disease) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: age, cholesterol, resting blood pressure, maximum heart rate (feature); heart disease (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-readmission-sparse` - Predict 30-day hospital readmission; the correct dataset publishes no features or target.

- category: `healthcare` / difficulty: `sparse_metadata`
- pooled candidates: 2
- note: Silence about target and features is UNKNOWN, never MISMATCH.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Care Planning Notes Collection | 1 | 53.25 | 42.29 | 57.64 | 0.56/0.00/0.44 |
| 2 | Hospital Encounters Longitudinal Extract | 3 | 50.70 | 38.65 | 57.64 | 0.53/0.00/0.47 |

**1. Care Planning Notes Collection** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `feature_match` 0.00 w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (readmission within 30 days) w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Compatible with the required ML task(s): classification; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: readmission within 30 days (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**2. Hospital Encounters Longitudinal Extract** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `keyword_overlap` 0.61 w=9 -> +5.45; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `feature_match` UNKNOWN (discharge disposition, encounter count, diagnosis count) w=16 -> +0.00; `target_match` UNKNOWN (readmission within 30 days) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: discharge, encounter, hospital; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: discharge disposition, encounter count, diagnosis count (feature); readmission within 30 days (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-retinal-imaging` - Detect diabetic retinopathy from retinal fundus photographs.

- category: `healthcare` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Diabetic Retinopathy Blindness Detection | 3 | 92.64 | 90.25 | 96.43 | 0.68/0.00/0.32 |
| 2 | Eye Disease Diagnosis Records (Tabular Only) | -1 | 59.93 | 45.89 | 85.36 | 0.54/0.00/0.46 |
| 3 | Heart Sound Recordings | 0 | 49.40 | 30.86 | 85.36 | 0.44/0.00/0.56 |
| 4 | Diabetes Clinical Indicators (Tabular) | 1 | 45.40 | 25.14 | 85.36 | 0.39/0.00/0.61 |

**1. Diabetic Retinopathy Blindness Detection** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.51 w=9 -> +4.58; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: retinal, fundus, diabetic retinopathy; Dataset metadata references required features: fundus photograph; Dataset metadata references the prediction target: diabetic retinopathy grade; Compatible with the required ML task(s): image classification; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Eye Disease Diagnosis Records (Tabular Only)** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `compatibility` 0.50 (contradiction: dataset target 'eye disease diagnosis' declares different label values than the project's 'diabetic retinopathy grade') w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85; `keyword_overlap` 0.17 w=9 -> +1.53; `feature_match` UNKNOWN (fundus photograph) w=16 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: fundus; Dataset metadata references the prediction target: diabetic retinopathy grade; Conflicts with the request: dataset target 'eye disease diagnosis' declares different label values than the project's 'diabetic retinopathy grade'; Publisher quality score 0.84 / 1.00; Could not confirm from published metadata: fundus photograph (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Heart Sound Recordings** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (fundus photograph) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: fundus photograph (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**4. Diabetes Clinical Indicators (Tabular)** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `compatibility` 0.50 (contradiction: dataset target 'diabetes risk level' declares different label values than the project's 'diabetic retinopathy grade') w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (fundus photograph) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Conflicts with the request: dataset target 'diabetes risk level' declares different label values than the project's 'diabetic retinopathy grade'; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: fundus photograph (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-target-conflict` - Project needs mortality; the candidate positively declares a different label (readmission).

- category: `healthcare` / difficulty: `distractor`
- pooled candidates: 2
- note: Declared label conflict is real incompatibility evidence, unlike silence.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Inpatient Readmission Study | -1 | 75.32 | 70.45 | 73.42 | 0.65/0.00/0.35 |
| 2 | Hospital Outcome Notes (Unlabelled) | 1 | 45.25 | 30.86 | 57.64 | 0.48/0.00/0.52 |

**1. Inpatient Readmission Study** (grade -1)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.41 w=9 -> +3.71; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: admission, diagnosis; Dataset metadata references required features: age, admission type, diagnosis; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. Hospital Outcome Notes (Unlabelled)** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `feature_match` UNKNOWN (age, admission type, diagnosis) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (mortality) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: age, admission type, diagnosis (feature); mortality (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-genomics-response` - Predict treatment response from gene expression; raw assay data needs substantial preprocessing.

- category: `healthcare` / difficulty: `semantic`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Drug Response with Labelled Outcomes | 3 | 94.29 | 97.53 | 73.42 | 0.72/0.00/0.28 |
| 2 | Raw Gene Expression Assay Dump | 1 | 51.24 | 39.42 | 57.64 | 0.50/0.03/0.46 |

**1. Drug Response with Labelled Outcomes** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.81 w=9 -> +7.27; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: gene expression, treatment response, dose, cell line; Dataset metadata references required features: gene expression, dose, cell line; Dataset metadata references the prediction target: treatment response; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Raw Gene Expression Assay Dump** (grade 1)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `keyword_overlap` 0.20 w=9 -> +1.82; `feature_match` 0.33 w=16 -> +1.78; `recency` 0.47 w=2 -> +0.94; `target_match` UNKNOWN (treatment response) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: gene expression; Dataset metadata references required features: gene expression; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Could not confirm from published metadata: dose, cell line (feature); treatment response (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `health-adverse-effect-nlp` - Classify adverse drug effects from patient review text.

- category: `healthcare` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Adverse Drug Event Annotation Corpus | 3 | 67.32 | 59.02 | 73.42 | 0.61/0.00/0.39 |
| 2 | Drug Review Sentiment Dataset | 2 | 57.98 | 40.73 | 96.43 | 0.49/0.00/0.51 |
| 3 | Large Movie Review Dataset | -1 | 42.52 | 18.65 | 96.43 | 0.31/0.00/0.69 |

**1. Adverse Drug Event Annotation Corpus** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `feature_match` 0.50 w=16 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.41 w=9 -> +3.71; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: drug, review; Dataset metadata references required features: review text; Compatible with the required ML task(s): text classification; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. Drug Review Sentiment Dataset** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `feature_match` 0.50 w=16 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.41 w=9 -> +3.71; `compatibility` 0.40 (contradiction: dataset supports only sentiment analysis while the project requires text classification) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: drug, review; Dataset metadata references required features: review text; Conflicts with the request: dataset supports only sentiment analysis while the project requires text classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Verified listing from an external dataset registry (kaggle-fixture)
**3. Large Movie Review Dataset** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `feature_match` 0.50 w=16 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.40 (contradiction: dataset supports only sentiment analysis while the project requires text classification) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.21 w=9 -> +1.86; `domain_relevance` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: review; Dataset metadata references required features: review text; Conflicts with the request: dataset supports only sentiment analysis while the project requires text classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)

### `health-age-bias-audit` - Audit age bias in an outcome predictor; age is the protected attribute and outcome the target.

- category: `healthcare` / difficulty: `semantic`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Clinical Outcome Fairness Benchmarks | 3 | 88.34 | 92.42 | 57.64 | 0.73/0.00/0.27 |
| 2 | Credit Scoring Records for Bias Review | 0 | 45.42 | 25.16 | 85.36 | 0.37/0.02/0.61 |

**1. Clinical Outcome Fairness Benchmarks** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `keyword_overlap` 0.85 w=9 -> +7.69; `quality` 0.62 w=8 -> +4.96; `task_match` 0.50 w=8 -> +4.00; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: age, bias, fairness, outcome, protected attribute; Dataset metadata references required features: age, sex, comorbidity count, length of stay; Dataset metadata references the prediction target: treatment outcome; Compatible with the required ML task(s): classification; Publisher quality score 0.62 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Credit Scoring Records for Bias Review** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.51 w=9 -> +4.62; `popularity` 0.85 w=5 -> +4.24; `task_match` 0.50 w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.25 w=16 -> +1.00; `domain_relevance` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Dataset metadata matches project concepts: age, bias, outcome; Dataset metadata references required features: age; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Could not confirm from published metadata: sex, comorbidity count, length of stay (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-iot-intrusion` - Classify intrusion type in IoT network traffic from flow statistics.

- category: `cybersecurity` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | BoT-IoT Dataset | 3 | 96.55 | 98.20 | 85.36 | 0.71/0.00/0.29 |
| 2 | N-BaIoT Device Anomaly Dataset | 2 | 56.94 | 44.18 | 73.42 | 0.54/0.00/0.46 |
| 3 | Smart Home Energy Consumption Dataset | 0 | 38.79 | 13.32 | 96.43 | 0.24/0.00/0.76 |
| 4 | FIFA Video Game Gameplay Network Data | -1 | 37.13 | 13.32 | 85.36 | 0.25/0.00/0.75 |

**1. BoT-IoT Dataset** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.86 w=9 -> +7.74; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: intrusion, iot, network traffic, protocol, byte count, ...; Dataset metadata references required features: source ip, destination ip, protocol, byte count; Dataset metadata references the prediction target: intrusion type; Compatible with the required ML task(s): classification, anomaly detection; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. N-BaIoT Device Anomaly Dataset** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `keyword_overlap` 0.15 w=9 -> +1.32; `feature_match` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: iot; Compatible with the required ML task(s): classification, anomaly detection; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Verified listing from an external dataset registry (kaggle-fixture)
**3. Smart Home Energy Consumption Dataset** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.15 w=9 -> +1.32; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (source ip, destination ip, protocol, byte count) w=16 -> +0.00; `target_match` UNKNOWN (intrusion type) w=13 -> +0.00; `task_match` UNKNOWN (classification, anomaly detection) w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: iot; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: source ip, destination ip, protocol, byte count (feature); intrusion type (target); classification, anomaly detection (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**4. FIFA Video Game Gameplay Network Data** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `keyword_overlap` 0.15 w=9 -> +1.32; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (source ip, destination ip, protocol, byte count) w=16 -> +0.00; `target_match` UNKNOWN (intrusion type) w=13 -> +0.00; `task_match` UNKNOWN (classification, anomaly detection) w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: network traffic; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: source ip, destination ip, protocol, byte count (feature); intrusion type (target); classification, anomaly detection (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-phishing-url` - Detect malicious URLs from raw URL strings.

- category: `cybersecurity` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Phishing Website URLs Dataset | 3 | 93.15 | 93.36 | 85.36 | 0.70/0.00/0.30 |
| 2 | Malware and Benign URL Strings | 1 | 74.45 | 69.19 | 73.42 | 0.65/0.00/0.35 |
| 3 | Adult Website Classification Dataset | 0 | 49.96 | 29.29 | 96.43 | 0.41/0.00/0.59 |

**1. Phishing Website URLs Dataset** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.75 w=9 -> +6.75; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: phishing, url, website; Dataset metadata references required features: url string, url length, num underscores; Dataset metadata references the prediction target: phishing; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Malware and Benign URL Strings** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `feature_match` 0.33 w=16 -> +5.33; `keyword_overlap` 0.50 w=9 -> +4.50; `compatibility` 0.50 (contradiction: dataset target 'malware label' declares different label values than the project's 'phishing') w=8 -> +4.00; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: phishing, url; Dataset metadata references required features: url string; Dataset metadata references the prediction target: phishing; Compatible with the required ML task(s): classification; Conflicts with the request: dataset target 'malware label' declares different label values than the project's 'phishing'; Verified listing from an external dataset registry (kaggle-fixture)
**3. Adult Website Classification Dataset** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.50 w=9 -> +4.50; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (url string, url length, num underscores) w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Dataset metadata matches project concepts: url, website; Compatible with the required ML task(s): classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: url string, url length, num underscores (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-adversarial-weak` - Adversarial detection needs attack generation; no public dataset ships attacks, so grade 1.

- category: `cybersecurity` / difficulty: `semantic`
- pooled candidates: 2
- note: Requires synthesising attacks: WEAK by definition, not MISSING.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Fashion-MNIST Image Collection | 0 | 32.66 | 4.57 | 96.43 | 0.10/0.00/0.90 |
| 2 | MNIST with Adversarial Samples (Small Split) | 1 | 28.76 | 7.31 | 57.64 | 0.18/0.00/0.82 |

**1. Fashion-MNIST Image Collection** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.40 (contradiction: dataset supports only image classification while the project requires anomaly detection) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (adversarial perturbation, clean input) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (adversarial attack) w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Conflicts with the request: dataset supports only image classification while the project requires anomaly detection; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: adversarial perturbation, clean input (feature); adversarial attack (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**2. MNIST with Adversarial Samples (Small Split)** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.62 w=8 -> +4.96; `compatibility` 0.40 (contradiction: dataset supports only classification while the project requires anomaly detection) w=8 -> +3.20; `popularity` 0.55 w=5 -> +2.75; `keyword_overlap` 0.21 w=9 -> +1.92; `recency` 0.47 w=2 -> +0.94; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (adversarial perturbation, clean input) w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: adversarial; Conflicts with the request: dataset supports only classification while the project requires anomaly detection; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: adversarial perturbation, clean input (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-rare-attack-subtype` - Need a specific attack type that is only a subset of a broader intrusion dataset.

- category: `cybersecurity` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Network Intrusion Flow Records (Broad Taxonomy) | 2 | 96.23 | 97.75 | 85.36 | 0.71/0.00/0.29 |
| 2 | Weather Station Telemetry Archive | -1 | 31.65 | 11.43 | 57.64 | 0.25/0.00/0.75 |

**1. Network Intrusion Flow Records (Broad Taxonomy)** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: attack type, packet, protocol, flag; Dataset metadata references required features: packet length, flag count, protocol type; Dataset metadata references the prediction target: specific attack type; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Weather Station Telemetry Archive** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (packet length, flag count, protocol type) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (specific attack type) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: packet length, flag count, protocol type (feature); specific attack type (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-code-vulnerability` - Detect source-code vulnerabilities from code snippets.

- category: `cybersecurity` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | National Vulnerability Database Extract | 3 | 91.97 | 91.67 | 85.36 | 0.70/0.00/0.30 |
| 2 | CVE Mention Corpus | 2 | 57.47 | 44.94 | 73.42 | 0.55/0.00/0.45 |
| 3 | Excel Vulnerability Reports Archive | 0 | 45.91 | 33.51 | 49.69 | 0.51/0.00/0.49 |

**1. National Vulnerability Database Extract** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: vulnerability, source code, cwe; Dataset metadata references required features: source code, language, cwe id; Dataset metadata references the prediction target: vulnerability presence; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. CVE Mention Corpus** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.82 w=2 -> +1.64; `feature_match` 0.00 w=16 -> +0.00; `target_match` UNKNOWN (vulnerability presence) w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: vulnerability; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Could not confirm from published metadata: vulnerability presence (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Excel Vulnerability Reports Archive** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.55 w=8 -> +4.40; `popularity` 0.61 w=5 -> +3.05; `keyword_overlap` 0.21 w=9 -> +1.86; `feature_match` UNKNOWN (source code, language, cwe id) w=16 -> +0.00; `recency` 0.00 w=2 -> +0.00; `target_match` UNKNOWN (vulnerability presence) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: vulnerability; Publisher quality score 0.55 / 1.00; Adoption signal: 9,800 downloads, 85 votes; Could not confirm from published metadata: source code, language, cwe id (feature); vulnerability presence (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-sms-spam` - Classify SMS messages as spam or ham.

- category: `cybersecurity` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | SMS Spam Collection (UCI) | 3 | 89.83 | 88.60 | 85.36 | 0.69/0.00/0.31 |
| 2 | Email Spam Enron Corpus | 2 | 85.83 | 82.89 | 85.36 | 0.68/0.00/0.32 |
| 3 | Fake News Article Corpus | 1 | 60.32 | 44.08 | 96.43 | 0.51/0.00/0.49 |

**1. SMS Spam Collection (UCI)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `task_match` 0.50 w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: sms, spam, ham, message; Dataset metadata references required features: message text, sender; Dataset metadata references the prediction target: spam; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Email Spam Enron Corpus** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `feature_match` 0.50 w=16 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: sms, spam, ham, message; Dataset metadata references required features: message text; Dataset metadata references the prediction target: spam; Compatible with the required ML task(s): classification, text classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Fake News Article Corpus** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.21 w=9 -> +1.86; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: spam; Dataset metadata references the prediction target: spam; Compatible with the required ML task(s): classification, text classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-audio-command` - Detect spoken voice commands from audio in a security system.

- category: `cybersecurity` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Audio-Triggered Intrusion Detection Dataset | 3 | 91.79 | 97.35 | 57.64 | 0.74/0.00/0.26 |
| 2 | Tabular Network Intrusion Features | 0 | 61.66 | 48.37 | 85.36 | 0.55/0.00/0.45 |

**1. Audio-Triggered Intrusion Detection Dataset** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.79 w=9 -> +7.14; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: audio, command, speaker, intrusion detection; Dataset metadata references required features: audio waveform, speaker id; Dataset metadata references the prediction target: spoken command; Compatible with the required ML task(s): classification; Publisher quality score 0.62 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Tabular Network Intrusion Features** (grade 0)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (audio waveform, speaker id) w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: audio; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Could not confirm from published metadata: audio waveform, speaker id (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cyber-ai-surfaced-dataset` - The strongest dataset is an unverified AI suggestion; verification weight must not bury it silently.

- category: `cybersecurity` / difficulty: `semantic`
- pooled candidates: 2
- note: Verification (weight 15) must not invert a clear relevance win.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Malware API Call Sequence Corpus | 3 | 56.17 | 74.89 | 0.00 | 0.93/0.00/0.07 |
| 2 | Verified Endpoint Telemetry Summary | 0 | 51.06 | 30.86 | 96.43 | 0.42/0.00/0.58 |

**1. Malware API Call Sequence Corpus** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `verification` 0.25 w=15 -> +3.75; `domain_relevance` 0.00 w=16 -> +0.00; `popularity` UNKNOWN w=5 -> +0.00; `quality` UNKNOWN w=8 -> +0.00; `recency` UNKNOWN w=2 -> +0.00
- reasons: Dataset metadata matches project concepts: malware, api calls, file hash, family; Dataset metadata references required features: api call sequence, file hash; Dataset metadata references the prediction target: malware family; Compatible with the required ML task(s): classification; AI-suggested candidate from kaggle-fixture; not yet confirmed by an external dataset registry
- evidence note: unverified, so declared-field rules were not applied
**2. Verified Endpoint Telemetry Summary** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (api call sequence, file hash) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (malware family) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Cybersecurity; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: api call sequence, file hash (feature); malware family (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-churn` - Predict bank customer churn (Exited = 1/0) from balance, tenure and product holdings.

- category: `finance` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Bank Customer Churn (Exited label) | 3 | 93.97 | 94.52 | 85.36 | 0.70/0.00/0.30 |
| 2 | Telecom Customer Churn | 3 | 52.49 | 32.89 | 96.43 | 0.44/0.00/0.56 |
| 3 | Customer Segmentation Profiles | 0 | 34.18 | 6.73 | 96.43 | 0.14/0.00/0.86 |
| 4 | Retail Loyalty Churn Survey | 1 | 31.61 | 8.00 | 73.42 | 0.18/0.00/0.82 |

**1. Bank Customer Churn (Exited label)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.84 w=9 -> +7.56; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: customer churn, exited, balance, tenure, bank; Dataset metadata references required features: balance, tenure, num products, estimated salary; Dataset metadata references the prediction target: Exited; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Telecom Customer Churn** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `feature_match` 0.25 w=16 -> +4.00; `keyword_overlap` 0.34 w=9 -> +3.02; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Dataset metadata matches project concepts: customer churn, tenure; Dataset metadata references required features: tenure; Compatible with the required ML task(s): classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)
**3. Customer Segmentation Profiles** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.40 (contradiction: dataset supports only clustering while the project requires classification) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.17 w=9 -> +1.51; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (balance, tenure, num products, estimated salary) w=16 -> +0.00; `target_match` UNKNOWN (Exited) w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: customer churn; Conflicts with the request: dataset supports only clustering while the project requires classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: balance, tenure, num products, estimated salary (feature); Exited (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**4. Retail Loyalty Churn Survey** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.72 w=8 -> +5.76; `compatibility` 0.70 (contradiction: dataset is entirely about retail while the project is about finance) w=8 -> +5.60; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (balance, tenure, num products, estimated salary) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about retail while the project is about finance; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: balance, tenure, num products, estimated salary (feature); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-credit-default` - Predict loan default from income, credit amount and duration.

- category: `finance` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | German Credit Risk Dataset | 3 | 92.23 | 92.03 | 85.36 | 0.70/0.00/0.30 |
| 2 | Loan Approval Prediction Playground | 2 | 79.78 | 71.87 | 96.43 | 0.63/0.00/0.37 |
| 3 | Credit Card Transaction Records | 1 | 53.92 | 34.94 | 96.43 | 0.44/0.02/0.55 |

**1. German Credit Risk Dataset** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.75 w=16 -> +12.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: credit default, loan, income, risk; Dataset metadata references required features: income, credit amount, duration; Dataset metadata references the prediction target: default; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Loan Approval Prediction Playground** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `feature_match` 0.25 w=16 -> +4.00; `keyword_overlap` 0.41 w=9 -> +3.71; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: loan, income; Dataset metadata references required features: income; Dataset metadata references the prediction target: default; Compatible with the required ML task(s): classification; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Credit Card Transaction Records** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.21 w=9 -> +1.86; `feature_match` 0.25 w=16 -> +1.00; `target_match` UNKNOWN (default) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: credit default; Dataset metadata references required features: income; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Could not confirm from published metadata: credit amount, duration, savings account (feature); default (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-fraud` - Detect fraudulent card transactions (isFraud target).

- category: `finance` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | IEEE-CIS Fraud Detection | 3 | 97.74 | 97.53 | 96.43 | 0.70/0.00/0.30 |
| 2 | Credit Card Fraud Detection (ULB) | 2 | 70.59 | 61.13 | 85.36 | 0.61/0.00/0.39 |
| 3 | Bank Transaction Fraud Annotations | 2 | 68.77 | 58.53 | 85.36 | 0.60/0.00/0.40 |
| 4 | Online Retail Sales Transactions | 0 | 49.10 | 28.05 | 96.43 | 0.40/0.00/0.60 |

**1. IEEE-CIS Fraud Detection** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.81 w=9 -> +7.27; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: fraud, transaction, isFraud, card; Dataset metadata references required features: transaction amount, card brand, merchant category; Dataset metadata references the prediction target: isFraud; Compatible with the required ML task(s): classification; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Credit Card Fraud Detection (ULB)** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.61 w=9 -> +5.45; `feature_match` 0.33 w=16 -> +5.33; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: fraud, transaction, card; Dataset metadata references required features: transaction amount; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**3. Bank Transaction Fraud Annotations** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `feature_match` 0.33 w=16 -> +5.33; `popularity` 0.85 w=5 -> +4.24; `keyword_overlap` 0.40 w=9 -> +3.64; `recency` 0.92 w=2 -> +1.85; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: fraud, transaction; Dataset metadata references required features: transaction amount; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**4. Online Retail Sales Transactions** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `domain_relevance` 0.50 w=16 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.40 w=9 -> +3.64; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (transaction amount, card brand, merchant category) w=16 -> +0.00; `target_match` UNKNOWN (isFraud) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Dataset metadata matches project domain terms: Fraud; Dataset metadata matches project concepts: fraud, transaction; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: transaction amount, card brand, merchant category (feature); isFraud (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-stock-forecast` - Forecast next-day closing price from historical prices and volume.

- category: `finance` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Daily Stock Price History | 3 | 90.11 | 86.64 | 96.43 | 0.67/0.00/0.33 |
| 2 | Bitcoin Hourly Price Series | 2 | 86.94 | 84.48 | 85.36 | 0.68/0.00/0.32 |
| 3 | Monthly Retail Sales Series | 1 | 39.53 | 19.30 | 73.42 | 0.34/0.00/0.66 |

**1. Daily Stock Price History** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.67 w=9 -> +6.05; `popularity` 0.98 w=5 -> +4.89; `task_match` 0.50 w=8 -> +4.00; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: stock, close, price, volume; Dataset metadata references required features: open, high, low, volume; Dataset metadata references the prediction target: next day close; Compatible with the required ML task(s): forecasting; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Bitcoin Hourly Price Series** (grade 2)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.50 w=9 -> +4.54; `popularity` 0.85 w=5 -> +4.24; `task_match` 0.50 w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: close, price, volume; Dataset metadata references required features: open, high, low, volume; Dataset metadata references the prediction target: next day close; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Monthly Retail Sales Series** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `task_match` 0.50 w=8 -> +4.00; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `keyword_overlap` 0.17 w=9 -> +1.51; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (open, high, low, volume) w=16 -> +0.00; `target_match` UNKNOWN (next day close) w=13 -> +0.00
- reasons: Dataset metadata matches project concepts: price; Compatible with the required ML task(s): forecasting; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: open, high, low, volume (feature); next day close (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-insurance-fraud` - Detect fraudulent insurance claims from claim and policy records.

- category: `finance` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Insurance Claims Fraud Dataset | 3 | 96.08 | 97.53 | 85.36 | 0.71/0.00/0.29 |
| 2 | Auto Insurance Claim Severity | 2 | 79.59 | 73.98 | 85.36 | 0.65/0.00/0.35 |
| 3 | Medical Insurance Charges | 0 | 52.30 | 32.62 | 96.43 | 0.44/0.00/0.56 |

**1. Insurance Claims Fraud Dataset** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.81 w=9 -> +7.27; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: insurance, fraud, claim, policy; Dataset metadata references required features: claim amount, policy type, prior claims; Dataset metadata references the prediction target: claim fraud; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Auto Insurance Claim Severity** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.61 w=9 -> +5.45; `feature_match` 0.33 w=16 -> +5.33; `popularity` 0.85 w=5 -> +4.24; `compatibility` 0.50 (contradiction: dataset target 'claim severity class' declares different label values than the project's 'claim fraud') w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: insurance, fraud, claim; Dataset metadata references required features: claim amount; Dataset metadata references the prediction target: claim fraud; Compatible with the required ML task(s): classification; Conflicts with the request: dataset target 'claim severity class' declares different label values than the project's 'claim fraud'; Verified listing from an external dataset registry (kaggle-fixture)
**3. Medical Insurance Charges** (grade 0)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.40 w=9 -> +3.64; `compatibility` 0.40 (contradiction: dataset supports only regression while the project requires classification) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (claim amount, policy type, prior claims) w=16 -> +0.00; `target_match` UNKNOWN (claim fraud) w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: insurance, fraud; Conflicts with the request: dataset supports only regression while the project requires classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Could not confirm from published metadata: claim amount, policy type, prior claims (feature); claim fraud (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-target-conflict-approved` - Project needs whether a loan was approved; candidate positively declares a default label.

- category: `finance` / difficulty: `distractor`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Loan Default Registry | -1 | 77.18 | 73.10 | 73.42 | 0.66/0.00/0.34 |
| 2 | Loan Application Notes (No Labels) | 1 | 47.10 | 33.51 | 57.64 | 0.50/0.00/0.50 |

**1. Loan Default Registry** (grade -1)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: loan, applicant, amount; Dataset metadata references required features: applicant income, requested amount; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. Loan Application Notes (No Labels)** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.47 w=2 -> +0.94; `feature_match` UNKNOWN (applicant income, requested amount) w=16 -> +0.00; `target_match` UNKNOWN (approved) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: loan; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: applicant income, requested amount (feature); approved (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-delinquent-semantic` - Target phrased as 'delinquent'; the candidate uses 'charged off / past due'.

- category: `finance` / difficulty: `semantic`
- pooled candidates: 2
- note: Requires knowing delinquent ≈ charged off ≈ past due.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Charge-off and Delinquency Status Corpus | 3 | 92.07 | 97.75 | 57.64 | 0.74/0.00/0.26 |
| 2 | General Credit Card Accounts Summary | 0 | 53.46 | 34.29 | 96.43 | 0.45/0.00/0.55 |

**1. Charge-off and Delinquency Status Corpus** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: delinquent, charged off, past due, balance; Dataset metadata references required features: days past due, balance; Dataset metadata references the prediction target: delinquent; Compatible with the required ML task(s): classification; Publisher quality score 0.62 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. General Credit Card Accounts Summary** (grade 0)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (days past due, balance) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (delinquent) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: days past due, balance (feature); delinquent (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `finance-ai-inferred-target` - AI suggestion claims a 'revenue' target when the project needs profit; the rationale must not be matched.

- category: `finance` / difficulty: `semantic`
- pooled candidates: 2
- note: rationale contains 'net profit margin'; only declared fields may be matched.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Unit Economics with Net Profit Margin | 3 | 92.01 | 94.29 | 73.42 | 0.72/0.00/0.28 |
| 2 | Retail Revenue Trend Report | 0 | 13.75 | 14.29 | 0.00 | 0.58/0.15/0.27 |

**1. Unit Economics with Net Profit Margin** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `task_match` 0.50 w=8 -> +4.00; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: profit margin, unit cost, unit price, overhead; Dataset metadata references required features: unit cost, unit price, overhead; Dataset metadata references the prediction target: net profit margin; Compatible with the required ML task(s): regression; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Retail Revenue Trend Report** (grade 0)
- factors: `compatibility` 1.00 w=8 -> +8.00; `verification` 0.25 w=15 -> +3.75; `task_match` 0.50 w=8 -> +2.00; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (unit cost, unit price, overhead) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `popularity` UNKNOWN w=5 -> +0.00; `quality` UNKNOWN w=8 -> +0.00; `recency` UNKNOWN w=2 -> +0.00; `target_match` UNKNOWN (net profit margin) w=13 -> +0.00
- reasons: Compatible with the required ML task(s): forecasting; Could not confirm from published metadata: unit cost, unit price, overhead (feature); net profit margin (target); regression (task) - the source does not declare these, so they are neither matched nor ruled out; AI-suggested candidate from kaggle-fixture; not yet confirmed by an external dataset registry
- evidence note: unverified, so declared-field rules were not applied

### `retail-demand-forecast` - Forecast weekly demand per store and SKU from price, promotion and calendar features.

- category: `retail` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Walmart Recruiting II: Sales in Stormy Weather | 3 | 96.26 | 95.43 | 96.43 | 0.69/0.00/0.31 |
| 2 | Grocery Item Demand Weekly | 2 | 74.34 | 66.48 | 85.36 | 0.63/0.00/0.37 |
| 3 | Fashion Ecommerce Sales Snapshot | 1 | 65.87 | 56.94 | 73.42 | 0.60/0.01/0.39 |

**1. Walmart Recruiting II: Sales in Stormy Weather** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.80 w=16 -> +12.80; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: demand, sales, store, sku, promotion, ...; Dataset metadata references required features: store id, sku, price, week; Dataset metadata references the prediction target: weekly demand; Compatible with the required ML task(s): forecasting; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Grocery Item Demand Weekly** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `keyword_overlap` 0.17 w=9 -> +1.54; `feature_match` 0.00 w=16 -> +0.00
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: demand; Dataset metadata references the prediction target: weekly demand; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**3. Fashion Ecommerce Sales Snapshot** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.51 w=9 -> +4.62; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `feature_match` 0.20 w=16 -> +0.64; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: demand, sales, sku; Dataset metadata references required features: sku; Dataset metadata references the prediction target: weekly demand; Publisher quality score 0.72 / 1.00; Could not confirm from published metadata: store id, price, promotion flag, week (feature); forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `retail-basket-analysis` - Unsupervised market-basket analysis over grocery transactions (no target).

- category: `retail` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Online Grocery Transactions (Basket Ready) | 3 | 78.97 | 73.10 | 85.36 | 0.65/0.00/0.35 |
| 2 | Store Inventory Levels | 1 | 47.10 | 33.51 | 57.64 | 0.50/0.00/0.50 |
| 3 | Music Listening Sessions | 0 | 35.06 | 8.00 | 96.43 | 0.16/0.00/0.84 |

**1. Online Grocery Transactions (Basket Ready)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `target_match` UNKNOWN w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: grocery, transaction, item; Dataset metadata references required features: transaction id, item id, quantity; Compatible with the required ML task(s): clustering, association rule mining; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. Store Inventory Levels** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.47 w=2 -> +0.94; `feature_match` UNKNOWN (transaction id, item id, quantity) w=16 -> +0.00; `target_match` UNKNOWN w=13 -> +0.00; `task_match` UNKNOWN (clustering, association rule mining) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: item; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: transaction id, item id, quantity (feature); clustering, association rule mining (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Music Listening Sessions** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `compatibility` 0.70 (contradiction: dataset is entirely about entertainment while the project is about retail) w=8 -> +5.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (transaction id, item id, quantity) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN w=13 -> +0.00; `task_match` UNKNOWN (clustering, association rule mining) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about entertainment while the project is about retail; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: transaction id, item id, quantity (feature); clustering, association rule mining (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `retail-price-elasticity` - Estimate units sold from price and promotion, needing both price and sales columns.

- category: `retail` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Retail Price and Unit Sales Panel | 3 | 94.37 | 95.10 | 85.36 | 0.71/0.00/0.29 |
| 2 | Aggregate Units Sold by Category | 1 | 60.11 | 43.78 | 96.43 | 0.51/0.00/0.49 |

**1. Retail Price and Unit Sales Panel** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: units sold, price, promotion; Dataset metadata references required features: price, promotion flag, week; Dataset metadata references the prediction target: units sold; Compatible with the required ML task(s): regression; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Aggregate Units Sold by Category** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `feature_match` 0.33 w=16 -> +5.33; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.41 w=9 -> +3.71; `recency` 0.99 w=2 -> +1.97; `target_match` UNKNOWN (units sold) w=13 -> +0.00; `task_match` UNKNOWN (regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: units sold, price; Dataset metadata references required features: price; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Could not confirm from published metadata: units sold (target); regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `retail-segmentation` - Label customer segments for targeted retail marketing.

- category: `retail` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Ecommerce Customer Segmentation | 3 | 88.22 | 86.31 | 85.36 | 0.68/0.00/0.32 |
| 2 | Online Shoppers Purchasing Intention | 2 | 53.40 | 36.57 | 85.36 | 0.48/0.00/0.52 |
| 3 | Gym Membership Attendance | 0 | 34.01 | 11.43 | 73.42 | 0.24/0.00/0.76 |

**1. Ecommerce Customer Segmentation** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `keyword_overlap` 0.20 w=9 -> +1.82
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: customer segment; Dataset metadata references required features: recency, frequency, monetary; Dataset metadata references the prediction target: customer segment; Compatible with the required ML task(s): clustering, classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Online Shoppers Purchasing Intention** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `task_match` 0.50 w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN (recency, frequency, monetary) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Retail; Compatible with the required ML task(s): classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: recency, frequency, monetary (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Gym Membership Attendance** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (recency, frequency, monetary) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (customer segment) w=13 -> +0.00; `task_match` UNKNOWN (clustering, classification) w=8 -> +0.00
- reasons: Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: recency, frequency, monetary (feature); customer segment (target); clustering, classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `retail-review-rating` - Predict the numeric product rating from review text.

- category: `retail` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Amazon Product Review Ratings | 3 | 99.46 | 100.00 | 96.43 | 0.70/0.00/0.30 |
| 2 | Product Ratings Without Review Text | 2 | 83.34 | 79.33 | 85.36 | 0.58/0.09/0.33 |
| 3 | Movie Review Polarity | 0 | 76.28 | 66.87 | 96.43 | 0.61/0.00/0.39 |

**1. Amazon Product Review Ratings** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: product review, rating, review text, stars, sentiment analysis; Dataset metadata references required features: review text, product id, star rating; Dataset metadata references the prediction target: rating; Compatible with the required ML task(s): regression, sentiment analysis; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Product Ratings Without Review Text** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `feature_match` 0.67 w=16 -> +7.11; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `task_match` 0.50 w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: product review, rating, review text, stars; Dataset metadata references required features: review text, star rating; Dataset metadata references the prediction target: rating; Compatible with the required ML task(s): regression; Could not confirm from published metadata: product id (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Movie Review Polarity** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.67 w=16 -> +10.67; `domain_relevance` 0.50 w=16 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.79 w=9 -> +7.14; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.50 (contradiction: dataset target 'sentiment polarity' declares different label values than the project's 'rating') w=8 -> +4.00; `task_match` 0.50 w=8 -> +4.00; `recency` 0.99 w=2 -> +1.97
- reasons: Dataset metadata matches project domain terms: Reviews; Dataset metadata matches project concepts: rating, review text, stars, sentiment analysis; Dataset metadata references required features: review text, star rating; Dataset metadata references the prediction target: rating; Compatible with the required ML task(s): sentiment analysis; Conflicts with the request: dataset target 'sentiment polarity' declares different label values than the project's 'rating'; Verified listing from an external dataset registry (kaggle-fixture)

### `retail-cart-abandonment` - Predict abandoned shopping carts from session events.

- category: `retail` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Shopping Cart Session Events | 3 | 87.63 | 88.02 | 73.42 | 0.70/0.00/0.30 |
| 2 | Generic Web Analytics Pageviews | 1 | 43.06 | 19.43 | 96.43 | 0.32/0.00/0.68 |

**1. Shopping Cart Session Events** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `keyword_overlap` 0.78 w=9 -> +7.01; `quality` 0.72 w=8 -> +5.76; `task_match` 0.50 w=8 -> +4.00; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: cart abandonment, session, cart value; Dataset metadata references required features: session id, cart value, last event timestamp; Dataset metadata references the prediction target: cart abandonment; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Generic Web Analytics Pageviews** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.50 w=16 -> +8.00; `quality` 0.95 w=8 -> +7.60; `compatibility` 0.70 (contradiction: dataset is entirely about web while the project is about retail) w=8 -> +5.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (session id, cart value, last event timestamp) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (cart abandonment) w=13 -> +0.00; `task_match` UNKNOWN (classification, forecasting) w=8 -> +0.00
- reasons: Dataset metadata matches project domain terms: Web Analytics; Conflicts with the request: dataset is entirely about web while the project is about retail; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: session id, cart value, last event timestamp (feature); cart abandonment (target); classification, forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `nlp-multilingual-sentiment` - Multilingual sentiment analysis across English, French and Arabic reviews.

- category: `nlp` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Multilingual Product Reviews | 3 | 81.80 | 77.14 | 85.36 | 0.66/0.00/0.34 |
| 2 | French Movie Review Corpus | 1 | 63.43 | 50.89 | 85.36 | 0.56/0.00/0.44 |
| 3 | Sentiment140 Tweet Corpus | 2 | 57.90 | 40.62 | 96.43 | 0.49/0.00/0.51 |
| 4 | Structured Star Ratings Without Text | 0 | 40.58 | 20.81 | 73.42 | 0.18/0.18/0.64 |

**1. Multilingual Product Reviews** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: sentiment, multilingual, reviews, language, sentiment analysis; Dataset metadata references required features: review text, language, star rating; Dataset metadata references the prediction target: sentiment; Compatible with the required ML task(s): sentiment analysis, text classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. French Movie Review Corpus** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `feature_match` 0.33 w=16 -> +5.33; `keyword_overlap` 0.59 w=9 -> +5.29; `popularity` 0.85 w=5 -> +4.24; `task_match` 0.50 w=8 -> +4.00; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: sentiment, reviews, sentiment analysis; Dataset metadata references required features: review text; Dataset metadata references the prediction target: sentiment; Compatible with the required ML task(s): sentiment analysis; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**3. Sentiment140 Tweet Corpus** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `task_match` 0.50 w=8 -> +4.00; `keyword_overlap` 0.38 w=9 -> +3.43; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: sentiment, sentiment analysis; Dataset metadata references the prediction target: sentiment; Compatible with the required ML task(s): sentiment analysis; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)
**4. Structured Star Ratings Without Text** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `feature_match` 0.67 w=16 -> +7.11; `quality` 0.72 w=8 -> +5.76; `compatibility` 0.70 (contradiction: dataset is entirely about retail while the project is about artificial_intelligence) w=8 -> +5.60; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.82 w=2 -> +1.64; `domain_relevance` 0.00 w=16 -> +0.00; `target_match` UNKNOWN (sentiment) w=13 -> +0.00; `task_match` UNKNOWN (sentiment analysis, text classification) w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: reviews; Dataset metadata references required features: review text, star rating; Conflicts with the request: dataset is entirely about retail while the project is about artificial_intelligence; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Could not confirm from published metadata: language (feature); sentiment (target); sentiment analysis, text classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `nlp-banking-intent` - Classify banking customer-service utterances into intents.

- category: `nlp` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Banking77 Intent Dataset | 3 | 81.80 | 77.14 | 85.36 | 0.66/0.00/0.34 |
| 2 | SMS Spam Utterances | 1 | 78.16 | 74.49 | 73.42 | 0.67/0.00/0.33 |
| 3 | BBC News Topic Corpus | 0 | 61.90 | 46.33 | 96.43 | 0.52/0.00/0.48 |

**1. Banking77 Intent Dataset** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: intent, banking, utterance, customer service, text classification; Dataset metadata references required features: utterance text; Dataset metadata references the prediction target: intent; Compatible with the required ML task(s): text classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. SMS Spam Utterances** (grade 1)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.79 w=9 -> +7.14; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `domain_relevance` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: intent, banking, utterance, text classification; Dataset metadata references required features: utterance text; Dataset metadata references the prediction target: intent; Compatible with the required ML task(s): text classification; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Verified listing from an external dataset registry (kaggle-fixture)
**3. BBC News Topic Corpus** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.38 w=9 -> +3.43; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: intent, text classification; Dataset metadata references the prediction target: intent; Compatible with the required ML task(s): text classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)

### `nlp-clinical-ner` - Extract clinical entities from discharge summaries.

- category: `nlp` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | De-identified Clinical Notes (MIMIC-style) | 3 | 89.43 | 93.97 | 57.64 | 0.74/0.00/0.26 |
| 2 | Newswire Named Entity Corpus | 1 | 62.17 | 49.09 | 85.36 | 0.55/0.00/0.45 |
| 3 | Contract Clause Collection | 0 | 29.25 | 8.00 | 57.64 | 0.19/0.00/0.81 |

**1. De-identified Clinical Notes (MIMIC-style)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.80 w=9 -> +7.18; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: clinical entity, discharge summary, ner, named entity recognition; Dataset metadata references required features: discharge summary text; Dataset metadata references the prediction target: clinical entities; Compatible with the required ML task(s): named entity recognition; Publisher quality score 0.62 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Newswire Named Entity Corpus** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.60 w=9 -> +5.36; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: clinical entity, ner, named entity recognition; Dataset metadata references the prediction target: clinical entities; Compatible with the required ML task(s): named entity recognition; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)
**3. Contract Clause Collection** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 0.70 (contradiction: dataset is entirely about legal while the project is about healthcare) w=8 -> +5.60; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (discharge summary text) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (clinical entities) w=13 -> +0.00; `task_match` UNKNOWN (named entity recognition) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about legal while the project is about healthcare; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: discharge summary text (feature); clinical entities (target); named entity recognition (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `nlp-squad` - Answer extraction from Wikipedia passages (SQuAD-style).

- category: `nlp` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | SQuAD Reading Comprehension (v1) | 3 | 97.06 | 96.57 | 96.43 | 0.70/0.00/0.30 |
| 2 | SQuAD v2 with Unanswerable Questions | 3 | 94.81 | 93.36 | 96.43 | 0.69/0.00/0.31 |
| 3 | News Headline Summarization Pairs | 1 | 54.25 | 37.79 | 85.36 | 0.49/0.00/0.51 |

**1. SQuAD Reading Comprehension (v1)** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: question answering, answer span, wikipedia, squad; Dataset metadata references required features: question text, passage text; Dataset metadata references the prediction target: answer span; Compatible with the required ML task(s): question answering; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. SQuAD v2 with Unanswerable Questions** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.75 w=9 -> +6.75; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: question answering, answer span, squad; Dataset metadata references required features: question text, passage text; Dataset metadata references the prediction target: answer span; Compatible with the required ML task(s): question answering; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. News Headline Summarization Pairs** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.50 w=16 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `compatibility` 0.40 (contradiction: dataset supports only text summarization while the project requires question answering) w=8 -> +3.20; `keyword_overlap` 0.25 w=9 -> +2.25; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: answer span; Dataset metadata references required features: question text; Dataset metadata references the prediction target: answer span; Conflicts with the request: dataset supports only text summarization while the project requires question answering; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)

### `nlp-hate-speech` - Detect hate speech in social media posts.

- category: `nlp` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Hate Speech Twitter Corpus | 3 | 78.37 | 72.24 | 85.36 | 0.65/0.00/0.35 |
| 2 | IMDB Large Movie Reviews | 0 | 47.52 | 25.79 | 96.43 | 0.38/0.00/0.62 |

**1. Hate Speech Twitter Corpus** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: hate speech, posts, social media; Dataset metadata references required features: post text, platform; Dataset metadata references the prediction target: hate speech; Compatible with the required ML task(s): text classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. IMDB Large Movie Reviews** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.40 (contradiction: dataset supports only sentiment analysis while the project requires text classification) w=8 -> +3.20; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.21 w=9 -> +1.86; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` 0.00 w=16 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: hate speech; Dataset metadata references the prediction target: hate speech; Conflicts with the request: dataset supports only sentiment analysis while the project requires text classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)

### `nlp-sarcasm` - Detect sarcasm in online forum posts.

- category: `nlp` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Reddit Sarcasm Corpus | 3 | 81.80 | 77.14 | 85.36 | 0.66/0.00/0.34 |
| 2 | News Headline Archive | 0 | 33.47 | 10.65 | 73.42 | 0.22/0.00/0.78 |
| 3 | Product Reviews with Sarcasm Notes | 1 | 32.86 | 7.22 | 85.36 | 0.15/0.00/0.85 |

**1. Reddit Sarcasm Corpus** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00
- reasons: Dataset metadata matches project concepts: sarcasm, forum, thread, reddit, text classification; Dataset metadata references required features: forum post text, thread; Dataset metadata references the prediction target: sarcasm; Compatible with the required ML task(s): text classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Verified listing from an external dataset registry (kaggle-fixture)
**2. News Headline Archive** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.72 w=8 -> +5.76; `compatibility` 0.70 (contradiction: dataset is entirely about social media while the project is about artificial_intelligence) w=8 -> +5.60; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.82 w=2 -> +1.64; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (forum post text, thread) w=16 -> +0.00; `target_match` UNKNOWN (sarcasm) w=13 -> +0.00; `task_match` UNKNOWN (text classification) w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: sarcasm; Conflicts with the request: dataset is entirely about social media while the project is about artificial_intelligence; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: forum post text, thread (feature); sarcasm (target); text classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Product Reviews with Sarcasm Notes** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `compatibility` 0.40 (contradiction: dataset supports only sentiment analysis while the project requires text classification) w=8 -> +3.20; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` 0.00 w=16 -> +0.00; `target_match` UNKNOWN (sarcasm) w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: sarcasm; Conflicts with the request: dataset supports only sentiment analysis while the project requires text classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: sarcasm (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cv-digit-classification` - Classify handwritten digits from 28x28 grayscale images.

- category: `computer_vision` / difficulty: `standard`
- pooled candidates: 5

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | MNIST Handwritten Digits | 3 | 97.06 | 96.57 | 96.43 | 0.70/0.00/0.30 |
| 2 | EMNIST Handwritten Letters and Digits | 3 | 91.76 | 93.92 | 73.42 | 0.72/0.00/0.28 |
| 3 | Fashion-MNIST | 2 | 68.61 | 55.92 | 96.43 | 0.57/0.00/0.43 |
| 4 | CIFAR-10 Colour Images | 0 | 60.64 | 44.54 | 96.43 | 0.51/0.00/0.49 |
| 5 | Iris Flower Measurements (Tabular) | -1 | 37.46 | 11.43 | 96.43 | 0.21/0.00/0.79 |

**1. MNIST Handwritten Digits** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: handwritten digit, mnist, 28x28, grayscale, image classification; Dataset metadata references required features: 28x28 grayscale image; Dataset metadata references the prediction target: digit; Compatible with the required ML task(s): image classification; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. EMNIST Handwritten Letters and Digits** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.79 w=9 -> +7.14; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: handwritten digit, 28x28, grayscale, image classification; Dataset metadata references required features: 28x28 grayscale image; Dataset metadata references the prediction target: digit; Compatible with the required ML task(s): image classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Fashion-MNIST** (grade 2)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.79 w=9 -> +7.14; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Dataset metadata matches project concepts: mnist, 28x28, grayscale, image classification; Dataset metadata references required features: 28x28 grayscale image; Compatible with the required ML task(s): image classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)
**4. CIFAR-10 Colour Images** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `keyword_overlap` 0.18 w=9 -> +1.58; `feature_match` UNKNOWN (28x28 grayscale image) w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: image classification; Compatible with the required ML task(s): image classification; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Could not confirm from published metadata: 28x28 grayscale image (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**5. Iris Flower Measurements (Tabular)** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (28x28 grayscale image) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: 28x28 grayscale image (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cv-object-detection` - Detect objects with bounding boxes.

- category: `computer_vision` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | COCO Detection Annotations | 3 | 97.06 | 96.57 | 96.43 | 0.70/0.00/0.30 |
| 2 | ImageNet Classification (ILSVRC) | 0 | 66.26 | 52.57 | 96.43 | 0.50/0.06/0.44 |
| 3 | Cityscapes Semantic Segmentation | 1 | 50.01 | 34.29 | 73.42 | 0.48/0.00/0.52 |

**1. COCO Detection Annotations** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: object detection, bounding box, images; Dataset metadata references required features: image, bounding box coordinates; Dataset metadata references the prediction target: bounding boxes; Compatible with the required ML task(s): object detection; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. ImageNet Classification (ILSVRC)** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `feature_match` 0.50 w=16 -> +4.00; `compatibility` 0.40 (contradiction: dataset supports only image classification while the project requires object detection) w=8 -> +3.20; `keyword_overlap` 0.33 w=9 -> +3.00; `recency` 0.99 w=2 -> +1.97; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: images; Dataset metadata references required features: image; Dataset metadata references the prediction target: bounding boxes; Conflicts with the request: dataset supports only image classification while the project requires object detection; Could not confirm from published metadata: bounding box coordinates (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Cityscapes Semantic Segmentation** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `feature_match` 0.50 w=16 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.33 w=9 -> +3.00; `recency` 0.82 w=2 -> +1.64; `compatibility` 0.00 (contradiction: dataset supports only semantic segmentation while the project requires object detection) w=8 -> +0.00; `domain_relevance` 0.00 w=16 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: images; Dataset metadata references required features: image; Dataset metadata references the prediction target: bounding boxes; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Verified listing from an external dataset registry (kaggle-fixture)

### `cv-face-detection` - Count faces in images.

- category: `computer_vision` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | WIDER FACE Dataset | 3 | 95.40 | 96.57 | 85.36 | 0.71/0.00/0.29 |
| 2 | Celebrity Identity Labels | 1 | 50.37 | 32.24 | 85.36 | 0.37/0.08/0.55 |
| 3 | Iris Image Database | 0 | 31.87 | 8.37 | 73.42 | 0.06/0.13/0.82 |

**1. WIDER FACE Dataset** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: face, count, images, bounding box, face detection; Dataset metadata references required features: image, bounding box; Dataset metadata references the prediction target: face count; Compatible with the required ML task(s): face detection; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Celebrity Identity Labels** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.85 w=5 -> +4.24; `feature_match` 0.50 w=16 -> +4.00; `recency` 0.92 w=2 -> +1.85; `compatibility` 0.00 (contradiction: dataset supports only image classification while the project requires face detection) w=8 -> +0.00; `domain_relevance` 0.00 w=16 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: face, count, images; Dataset metadata references required features: image; Dataset metadata references the prediction target: face count; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Could not confirm from published metadata: bounding box (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Iris Image Database** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.72 w=8 -> +5.76; `feature_match` 0.50 w=16 -> +4.00; `popularity` 0.72 w=5 -> +3.61; `keyword_overlap` 0.21 w=9 -> +1.86; `recency` 0.82 w=2 -> +1.64; `compatibility` 0.00 (contradiction: dataset supports only image classification while the project requires face detection) w=8 -> +0.00; `domain_relevance` 0.00 w=16 -> +0.00; `target_match` 0.00 w=13 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Dataset metadata matches project concepts: images; Dataset metadata references required features: image; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: bounding box (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cv-tumor-mri` - Detect brain tumour presence in MRI scans.

- category: `computer_vision` / difficulty: `standard`
- pooled candidates: 4

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Brain Tumour MRI Images | 3 | 92.03 | 91.75 | 85.36 | 0.70/0.00/0.30 |
| 2 | Chest X-ray Pneumonia Images | 1 | 57.40 | 42.29 | 85.36 | 0.52/0.00/0.48 |
| 3 | Skin Lesion Photos (HAM10000) | 1 | 57.40 | 42.29 | 85.36 | 0.52/0.00/0.48 |
| 4 | Satellite Imagery Tiles | 0 | 41.46 | 17.14 | 96.43 | 0.29/0.00/0.71 |

**1. Brain Tumour MRI Images** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.62 w=9 -> +5.62; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: brain tumour, mri, scan; Dataset metadata references required features: mri scan; Dataset metadata references the prediction target: tumour presence; Compatible with the required ML task(s): image classification; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Chest X-ray Pneumonia Images** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.00 w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Compatible with the required ML task(s): image classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)
**3. Skin Lesion Photos (HAM10000)** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.00 w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Healthcare; Compatible with the required ML task(s): image classification; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)
**4. Satellite Imagery Tiles** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `compatibility` 0.50 (contradiction: dataset target 'land cover class' declares different label values than the project's 'tumour presence') w=8 -> +4.00; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (mri scan) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Compatible with the required ML task(s): image classification; Conflicts with the request: dataset target 'land cover class' declares different label values than the project's 'tumour presence'; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: mri scan (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `cv-action-video` - Recognise actions from video clips.

- category: `computer_vision` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | UCF101 Action Recognition | 3 | 95.40 | 96.57 | 85.36 | 0.71/0.00/0.29 |
| 2 | Still Image Action Dataset | 1 | 57.60 | 42.57 | 85.36 | 0.52/0.00/0.48 |
| 3 | Sports Match Commentary Text | 0 | 29.25 | 8.00 | 57.64 | 0.19/0.00/0.81 |

**1. UCF101 Action Recognition** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `keyword_overlap` 1.00 w=9 -> +9.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata matches project concepts: action recognition, video, clip; Dataset metadata references required features: video clip; Dataset metadata references the prediction target: action class; Compatible with the required ML task(s): action recognition; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Still Image Action Dataset** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `compatibility` 0.40 (contradiction: dataset supports only image classification while the project requires action recognition) w=8 -> +3.20; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.00 w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `task_match` 0.00 w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Artificial Intelligence; Dataset metadata references the prediction target: action class; Conflicts with the request: dataset supports only image classification while the project requires action recognition; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)
**3. Sports Match Commentary Text** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 0.70 (contradiction: dataset is entirely about sports analytics while the project is about artificial_intelligence) w=8 -> +5.60; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (video clip) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (action class) w=13 -> +0.00; `task_match` UNKNOWN (action recognition) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about sports analytics while the project is about artificial_intelligence; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: video clip (feature); action class (target); action recognition (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `ts-solar-output` - Forecast next-hour solar power output from irradiance and weather features.

- category: `time_series` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Solar Power Plant Generation | 3 | 91.86 | 91.51 | 85.36 | 0.70/0.00/0.30 |
| 2 | Electricity Load Diagrams | 2 | 70.40 | 60.86 | 85.36 | 0.61/0.00/0.39 |
| 3 | Weather Station Observations | 1 | 39.61 | 19.43 | 73.42 | 0.34/0.00/0.66 |

**1. Solar Power Plant Generation** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.61 w=9 -> +5.45; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Energy; Dataset metadata matches project concepts: solar, power output, irradiance; Dataset metadata references required features: irradiance, temperature, timestamp; Dataset metadata references the prediction target: solar power output; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Electricity Load Diagrams** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.00 w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00
- reasons: Published topic domain matches the project domain: Energy; Dataset metadata references the prediction target: solar power output; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)
**3. Weather Station Observations** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.50 w=16 -> +8.00; `quality` 0.72 w=8 -> +5.76; `compatibility` 0.70 (contradiction: dataset is entirely about environment while the project is about energy) w=8 -> +5.60; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `feature_match` UNKNOWN (irradiance, temperature, timestamp) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (solar power output) w=13 -> +0.00; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Dataset metadata matches project domain terms: Energy; Conflicts with the request: dataset is entirely about environment while the project is about energy; Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: irradiance, temperature, timestamp (feature); solar power output (target); forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `ts-airline-passengers` - Forecast monthly airline passenger totals.

- category: `time_series` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Airline Passengers Time Series | 3 | 93.63 | 91.67 | 96.43 | 0.69/0.00/0.31 |
| 2 | Daily Flight Counts | 2 | 57.40 | 42.29 | 85.36 | 0.52/0.00/0.48 |

**1. Airline Passengers Time Series** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `keyword_overlap` 0.62 w=9 -> +5.57; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97
- reasons: Published topic domain matches the project domain: Transportation; Dataset metadata matches project concepts: airline, passengers, monthly; Dataset metadata references required features: month, year, passenger total; Dataset metadata references the prediction target: monthly passengers; Compatible with the required ML task(s): forecasting; Publisher quality score 0.95 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Daily Flight Counts** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.00 w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Transportation; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)

### `ts-traffic-flow` - Forecast hourly traffic volume at intersections.

- category: `time_series` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Hourly Traffic Volume Sensors | 3 | 96.23 | 97.75 | 85.36 | 0.71/0.00/0.29 |
| 2 | Taxi Trip Records | 1 | 70.18 | 58.16 | 96.43 | 0.58/0.00/0.42 |
| 3 | Parking Occupancy Sensors | 0 | 34.01 | 11.43 | 73.42 | 0.24/0.00/0.76 |

**1. Hourly Traffic Volume Sensors** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.82 w=9 -> +7.42; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Transportation; Dataset metadata matches project concepts: traffic, volume, hourly, intersection; Dataset metadata references required features: hour, intersection id, occupancy rate; Dataset metadata references the prediction target: traffic volume; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Taxi Trip Records** (grade 1)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `keyword_overlap` 0.41 w=9 -> +3.71; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (hour, intersection id, occupancy rate) w=16 -> +0.00; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Transportation; Dataset metadata matches project concepts: traffic, volume; Dataset metadata references the prediction target: traffic volume; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Could not confirm from published metadata: hour, intersection id, occupancy rate (feature); forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Parking Occupancy Sensors** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (hour, intersection id, occupancy rate) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (traffic volume) w=13 -> +0.00; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Publisher quality score 0.72 / 1.00; Adoption signal: 38,000 downloads, 320 votes; Last updated 2025-11-12; Could not confirm from published metadata: hour, intersection id, occupancy rate (feature); traffic volume (target); forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `ts-energy-consumption` - Forecast hourly building energy consumption.

- category: `time_series` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Building Energy Consumption Hourly | 3 | 91.57 | 93.65 | 73.42 | 0.72/0.00/0.28 |
| 2 | Hourly Electricity Load Time Series | 3 | 82.96 | 78.80 | 85.36 | 0.66/0.00/0.34 |
| 3 | Natural Gas Consumption (Monthly) | 1 | 70.80 | 67.37 | 57.64 | 0.67/0.00/0.33 |

**1. Building Energy Consumption Hourly** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.51 w=9 -> +4.56; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Energy; Dataset metadata matches project concepts: energy consumption, hourly; Dataset metadata references required features: hour, outdoor temperature; Dataset metadata references the prediction target: hourly energy consumption; Compatible with the required ML task(s): forecasting; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Hourly Electricity Load Time Series** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `feature_match` 0.50 w=16 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.51 w=9 -> +4.56; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Energy; Dataset metadata matches project concepts: energy consumption, hourly; Dataset metadata references required features: hour; Dataset metadata references the prediction target: hourly energy consumption; Compatible with the required ML task(s): forecasting; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Natural Gas Consumption (Monthly)** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `keyword_overlap` 0.51 w=9 -> +4.56; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `feature_match` UNKNOWN (hour, outdoor temperature) w=16 -> +0.00
- reasons: Published topic domain matches the project domain: Energy; Dataset metadata matches project concepts: energy consumption, hourly; Dataset metadata references the prediction target: hourly energy consumption; Compatible with the required ML task(s): forecasting; Publisher quality score 0.62 / 1.00; Could not confirm from published metadata: hour, outdoor temperature (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `ts-market-volatility` - Forecast 30-day market volatility from price history.

- category: `time_series` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Options Implied Volatility Surface | 3 | 78.79 | 72.83 | 85.36 | 0.65/0.00/0.35 |
| 2 | Daily Stock Returns | 2 | 68.78 | 56.17 | 96.43 | 0.57/0.00/0.43 |
| 3 | Foreign Exchange Rate History | 1 | 55.06 | 36.57 | 96.43 | 0.46/0.00/0.54 |

**1. Options Implied Volatility Surface** (grade 3)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `feature_match` 0.33 w=16 -> +5.33; `popularity` 0.85 w=5 -> +4.24; `keyword_overlap` 0.34 w=9 -> +3.05; `recency` 0.92 w=2 -> +1.85
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: volatility, returns; Dataset metadata references required features: daily returns; Dataset metadata references the prediction target: 30 day volatility; Compatible with the required ML task(s): forecasting, regression; Publisher quality score 0.84 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Daily Stock Returns** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `feature_match` 0.67 w=16 -> +10.67; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `task_match` 0.50 w=8 -> +4.00; `keyword_overlap` 0.34 w=9 -> +3.05; `recency` 0.99 w=2 -> +1.97; `target_match` UNKNOWN (30 day volatility) w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: volatility, returns; Dataset metadata references required features: daily returns, rolling window; Compatible with the required ML task(s): forecasting; Publisher quality score 0.95 / 1.00; Could not confirm from published metadata: 30 day volatility (target) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. Foreign Exchange Rate History** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `task_match` 0.50 w=8 -> +4.00; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (daily returns, rolling window, volume) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` 0.00 w=13 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Compatible with the required ML task(s): forecasting; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: daily returns, rolling window, volume (feature) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-all-unknown-evidence` - Every candidate publishes no target and no features; grades must still be assigned by content.

- category: `edge_case` / difficulty: `sparse_metadata`
- pooled candidates: 3
- note: UNKNOWN must not collapse into MISMATCH for any candidate here.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Student Result Summaries | 2 | 65.25 | 59.43 | 57.64 | 0.58/0.06/0.36 |
| 2 | Exam Score Lists | 1 | 52.25 | 40.86 | 57.64 | 0.47/0.08/0.45 |
| 3 | School Bus Routes | 0 | 29.25 | 8.00 | 57.64 | 0.19/0.00/0.81 |

**1. Student Result Summaries** (grade 2)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `feature_match` 0.50 w=16 -> +4.00; `keyword_overlap` 0.33 w=9 -> +3.00; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: pass fail; Dataset metadata references required features: exam score; Dataset metadata references the prediction target: pass fail; Publisher quality score 0.62 / 1.00; Could not confirm from published metadata: attendance (feature); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**2. Exam Score Lists** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.62 w=8 -> +4.96; `feature_match` 0.50 w=16 -> +4.00; `keyword_overlap` 0.33 w=9 -> +3.00; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `target_match` UNKNOWN (pass fail) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Education; Dataset metadata matches project concepts: exam score; Dataset metadata references required features: exam score; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Could not confirm from published metadata: attendance (feature); pass fail (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**3. School Bus Routes** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `compatibility` 0.70 (contradiction: dataset is entirely about transportation while the project is about education) w=8 -> +5.60; `quality` 0.62 w=8 -> +4.96; `popularity` 0.55 w=5 -> +2.75; `recency` 0.47 w=2 -> +0.94; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (exam score, attendance) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (pass fail) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about transportation while the project is about education; Publisher quality score 0.62 / 1.00; Adoption signal: 4,200 downloads, 48 votes; Last updated 2024-02-09; Could not confirm from published metadata: exam score, attendance (feature); pass fail (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-single-candidate` - One candidate only; every metric must be well defined at k=10 for a list of length 1.

- category: `edge_case` / difficulty: `standard`
- pooled candidates: 1

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Weekly Unit Sales Panel | 3 | 93.01 | 95.71 | 73.42 | 0.72/0.00/0.28 |

**1. Weekly Unit Sales Panel** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.67 w=9 -> +6.00; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Retail; Dataset metadata matches project concepts: weekly units, store; Dataset metadata references required features: store id, week; Dataset metadata references the prediction target: weekly units; Compatible with the required ML task(s): forecasting; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-all-incompatible` - No relevant candidate exists; recall denominators are zero and must not divide by zero.

- category: `edge_case` / difficulty: `standard`
- pooled candidates: 2
- note: Relevant-in-pool = 0; recall and NDCG must be defined, not NaN.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Retail Seasonality Sales | -1 | 35.06 | 8.00 | 96.43 | 0.16/0.00/0.84 |
| 2 | Climate Reanalysis Archive | -1 | 33.40 | 8.00 | 85.36 | 0.17/0.00/0.83 |

**1. Retail Seasonality Sales** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `compatibility` 0.70 (contradiction: dataset is entirely about retail while the project is about healthcare) w=8 -> +5.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (variant allele frequency, clinvar significance) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (genomic variant pathogenicity) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about retail while the project is about healthcare; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: variant allele frequency, clinvar significance (feature); genomic variant pathogenicity (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**2. Climate Reanalysis Archive** (grade -1)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.84 w=8 -> +6.72; `compatibility` 0.70 (contradiction: dataset is entirely about environment while the project is about healthcare) w=8 -> +5.60; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (variant allele frequency, clinvar significance) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (genomic variant pathogenicity) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about environment while the project is about healthcare; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: variant allele frequency, clinvar significance (feature); genomic variant pathogenicity (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-domain-only-project` - Project specifies only a domain; every requirement factor is unjudgeable but publisher signals still apply.

- category: `edge_case` / difficulty: `standard`
- pooled candidates: 2
- note: Ranks on domain + publisher signals only; must not crash.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Cybersecurity Incident Archive | 2 | 56.67 | 41.23 | 85.36 | 0.51/0.00/0.49 |
| 2 | Culinary Recipe Archive | 0 | 35.06 | 8.00 | 96.43 | 0.16/0.00/0.84 |

**1. Cybersecurity Incident Archive** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `keyword_overlap` 0.54 w=9 -> +4.86; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `feature_match` UNKNOWN w=16 -> +0.00; `target_match` UNKNOWN w=13 -> +0.00; `task_match` UNKNOWN w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Cybersecurity; Dataset metadata matches project concepts: cybersecurity; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Verified listing from an external dataset registry (kaggle-fixture)
**2. Culinary Recipe Archive** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `compatibility` 0.70 (contradiction: dataset is entirely about food while the project is about cybersecurity) w=8 -> +5.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN w=13 -> +0.00; `task_match` UNKNOWN w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about food while the project is about cybersecurity; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-unicode-and-punctuation` - Non-ASCII names and punctuation must rank deterministically and not crash normalisation.

- category: `edge_case` / difficulty: `standard`
- pooled candidates: 2

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Taux d'incidence —district sanitaire | 3 | 89.29 | 90.39 | 73.42 | 0.71/0.00/0.29 |
| 2 | Recettes de cuisine :: données | 0 | 35.06 | 8.00 | 96.43 | 0.16/0.00/0.84 |

**1. Taux d'incidence —district sanitaire** (grade 3)
- factors: `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `quality` 0.72 w=8 -> +5.76; `keyword_overlap` 0.52 w=9 -> +4.68; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Healthcare; Dataset metadata matches project concepts: incidence, district; Dataset metadata references required features: date, district; Dataset metadata references the prediction target: incidence rate; Compatible with the required ML task(s): forecasting; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Recettes de cuisine :: données** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.95 w=8 -> +7.60; `compatibility` 0.70 (contradiction: dataset is entirely about food while the project is about healthcare) w=8 -> +5.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (date, district) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (incidence rate) w=13 -> +0.00; `task_match` UNKNOWN (forecasting) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about food while the project is about healthcare; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: date, district (feature); incidence rate (target); forecasting (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-ai-rationale-trap` - Only the AI rationale mentions the required target; a ranker matching rationale would invert this case.

- category: `edge_case` / difficulty: `semantic`
- pooled candidates: 2
- note: rationale text names pm2.5 concentration; declared fields do not.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | PM2.5 Air Quality Monitor Records | 2 | 93.85 | 96.91 | 73.42 | 0.72/0.00/0.28 |
| 2 | Urban Sensing Programme Data | 1 | 11.75 | 11.43 | 0.00 | 0.68/0.00/0.32 |

**1. PM2.5 Air Quality Monitor Records** (grade 2)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.76 w=9 -> +6.84; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Environment; Dataset metadata matches project concepts: pm2.5, air quality, wind speed; Dataset metadata references required features: pm2.5, wind speed, timestamp; Dataset metadata references the prediction target: pm2.5 concentration; Compatible with the required ML task(s): regression; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Urban Sensing Programme Data** (grade 1)
- factors: `compatibility` 1.00 w=8 -> +8.00; `verification` 0.25 w=15 -> +3.75; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (pm2.5, wind speed, timestamp) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `popularity` UNKNOWN w=5 -> +0.00; `quality` UNKNOWN w=8 -> +0.00; `recency` UNKNOWN w=2 -> +0.00; `target_match` UNKNOWN (pm2.5 concentration) w=13 -> +0.00; `task_match` UNKNOWN (regression) w=8 -> +0.00
- reasons: Could not confirm from published metadata: pm2.5, wind speed, timestamp (feature); pm2.5 concentration (target); regression (task) - the source does not declare these, so they are neither matched nor ruled out; AI-suggested candidate from kaggle-fixture; not yet confirmed by an external dataset registry
- evidence note: unverified, so declared-field rules were not applied

### `edge-retrieval-miss` - The pool contains no grade-3 dataset: a retrieval failure, not a ranking failure.

- category: `edge_case` / difficulty: `retrieval`
- pooled candidates: 2
- note: Metric ceiling is bounded by the pool; classification must say RETRIEVAL.

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Regional Harvest Statistics | 1 | 53.46 | 36.65 | 85.36 | 0.45/0.03/0.52 |
| 2 | Weather History Archive | 0 | 33.40 | 8.00 | 85.36 | 0.17/0.00/0.83 |

**1. Regional Harvest Statistics** (grade 1)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.84 w=8 -> +6.72; `popularity` 0.85 w=5 -> +4.24; `keyword_overlap` 0.25 w=9 -> +2.28; `recency` 0.92 w=2 -> +1.85; `feature_match` 0.33 w=16 -> +1.78; `target_match` UNKNOWN (crop yield) w=13 -> +0.00; `task_match` UNKNOWN (regression) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Agriculture; Dataset metadata matches project concepts: rainfall; Dataset metadata references required features: rainfall; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Could not confirm from published metadata: soil moisture, crop type (feature); crop yield (target); regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)
**2. Weather History Archive** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `quality` 0.84 w=8 -> +6.72; `compatibility` 0.70 (contradiction: dataset is entirely about environment while the project is about agriculture) w=8 -> +5.60; `popularity` 0.85 w=5 -> +4.24; `recency` 0.92 w=2 -> +1.85; `domain_relevance` 0.00 w=16 -> +0.00; `feature_match` UNKNOWN (soil moisture, rainfall, crop type) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (crop yield) w=13 -> +0.00; `task_match` UNKNOWN (regression) w=8 -> +0.00
- reasons: Conflicts with the request: dataset is entirely about environment while the project is about agriculture; Publisher quality score 0.84 / 1.00; Adoption signal: 210,000 downloads, 1,050 votes; Last updated 2026-05-18; Could not confirm from published metadata: soil moisture, rainfall, crop type (feature); crop yield (target); regression (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

### `edge-dense-and-identical` - Two near-identical candidates must rank deterministically with no hidden ordering by object identity.

- category: `edge_case` / difficulty: `standard`
- pooled candidates: 3

| rank | candidate | grade | score | relevance | quality | evidence/unknown/publisher |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Credit Risk Applicants (Part A) | 3 | 94.03 | 97.16 | 73.42 | 0.72/0.00/0.28 |
| 2 | Credit Risk Applicants (Part B) | 3 | 94.03 | 97.16 | 73.42 | 0.72/0.00/0.28 |
| 3 | Consumer Finance Overview | 0 | 51.06 | 30.86 | 96.43 | 0.42/0.00/0.58 |

**1. Credit Risk Applicants (Part A)** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.78 w=9 -> +7.01; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: default, income, duration; Dataset metadata references required features: income, duration; Dataset metadata references the prediction target: default; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**2. Credit Risk Applicants (Part B)** (grade 3)
- factors: `domain_relevance` 1.00 w=16 -> +16.00; `feature_match` 1.00 w=16 -> +16.00; `verification` 1.00 w=15 -> +15.00; `target_match` 1.00 w=13 -> +13.00; `compatibility` 1.00 w=8 -> +8.00; `task_match` 1.00 w=8 -> +8.00; `keyword_overlap` 0.78 w=9 -> +7.01; `quality` 0.72 w=8 -> +5.76; `popularity` 0.72 w=5 -> +3.61; `recency` 0.82 w=2 -> +1.64
- reasons: Published topic domain matches the project domain: Finance; Dataset metadata matches project concepts: default, income, duration; Dataset metadata references required features: income, duration; Dataset metadata references the prediction target: default; Compatible with the required ML task(s): classification; Publisher quality score 0.72 / 1.00; Verified listing from an external dataset registry (kaggle-fixture)
**3. Consumer Finance Overview** (grade 0)
- factors: `verification` 1.00 w=15 -> +15.00; `domain_relevance` 0.85 w=16 -> +13.60; `compatibility` 1.00 w=8 -> +8.00; `quality` 0.95 w=8 -> +7.60; `popularity` 0.98 w=5 -> +4.89; `recency` 0.99 w=2 -> +1.97; `feature_match` UNKNOWN (income, duration) w=16 -> +0.00; `keyword_overlap` 0.00 w=9 -> +0.00; `target_match` UNKNOWN (default) w=13 -> +0.00; `task_match` UNKNOWN (classification) w=8 -> +0.00
- reasons: Published topic domain matches the project domain: Finance; Publisher quality score 0.95 / 1.00; Adoption signal: 1,450,000 downloads, 5,200 votes; Last updated 2026-09-08; Could not confirm from published metadata: income, duration (feature); default (target); classification (task) - the source does not declare these, so they are neither matched nor ruled out; Verified listing from an external dataset registry (kaggle-fixture)

## 9. Limitations

- The candidate pools are fixtures, not live registry results. They are built to exercise the ranker, so absolute scores describe this corpus and not Kaggle at large.
- Retrieval quality is measured only as 'did the pool contain the right answer'. Real query performance against a live registry is not represented.
- Only 5 candidates are unverified, so the verification ablation is weakly identified: a null result there means 'not enough unverified candidates to flip a ranking', not 'verification does not matter'.
- Grades are human judgements made by one author against written criteria. They are defensible, not objective, and the criteria are in `labels.py` so a disagreement can be resolved rather than argued about.
