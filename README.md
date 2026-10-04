# DataPilot

**DataPilot** is a GenAI-powered platform intended to help users move from project ideas to dataset discovery, evaluation, ranking, and recommendation.

> **Current Status:** The **User A** workflow is implemented end to end: a project idea becomes structured requirements, then ranked and explained dataset recommendations from Kaggle and Hugging Face (plus clearly labelled, unverified AI suggestions). **User B** (analysing a dataset you upload), the frontend, a database, authentication, and embeddings/vector search are **not implemented**. See `DEVELOPMENT_STATE.md` for exact status, test counts, and known limitations.

## Currently Implemented

### 1. Project Requirement Analyzer

The Project Requirement Analyzer takes a natural-language ML/AI project description (max 150 words) and converts it into structured requirements.

It combines:
1. **Deterministic keyword/concept dictionary matching** — a 147-concept knowledge base covering 22 domains
2. **Natural language understanding using Gemini** — structured LLM output via the Google GenAI SDK
3. **Dictionary + LLM reconciliation** — deduplication, provenance tracking
4. **Ambiguity detection** — does not hallucinate missing information
5. **Missing-information detection** — flags unclear objectives, missing targets
6. **Dataset requirement extraction** — describes what future datasets should contain

## Architecture

```
User Description
    ↓
FastAPI POST /api/v1/projects/analyze
    ↓
Word Count Validation (backend-enforced, 150-word limit)
    ↓
Dictionary Matching (deterministic, instant)
    ↓
Gemini LLM Analysis (semantic understanding, structured output)
    ↓
Reconciliation (merge, deduplicate, track provenance)
    ↓
Structured Response
```

### 2. Dataset discovery, intelligence and recommendation (User A)

`POST /api/v1/datasets/recommend` takes the same description and returns:

- **Ranked datasets** from Kaggle and Hugging Face (registry-verified) and Gemini suggestions
  (`ai_suggested`, `unverified`), deduplicated, with a deterministic ten-factor score that
  always equals the sum of its listed contributions.
- **Dataset intelligence** per dataset: what the registry published versus what DataPilot
  inferred, schema state, whether the target is present / derivable / unknown /
  incompatible, per-feature coverage (matched, unknown and missing kept apart), data-type
  and time-axis checks, risks and limitations.
- **Confidence** per dataset, and a **final recommendation**: best match, alternatives with
  the one reason each ranks lower, what is verified / unknown / missing, and what to watch
  out for.

Rules the system enforces in code, not just in documentation: an AI suggestion is never
reported as verified or recommended; unknown is never treated as missing or as a match; a
dataset found incompatible is never put forward as the best match; no dataset file is ever
downloaded, so "verified schema" means a column list the registry itself published.

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.13+, FastAPI, Pydantic v2 |
| AI | Google GenAI SDK (`google-genai`) with Gemini 3.1 Flash Lite |
| Sources | Kaggle API and Hugging Face Hub API via `httpx` |
| Configuration | pydantic-settings, .env |
| Testing | pytest, httpx, FastAPI TestClient |
| Server | Uvicorn |

## Setup

### 1. Clone and enter the backend directory

```bash
git clone https://github.com/Sreeshan7/DataPilot.git
cd DataPilot/datapilot/backend
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and fill in your Gemini API key (a Kaggle token is optional; Kaggle's public search and the Hugging Face Hub work without credentials):

```bash
cp .env.example .env
```

Edit `.env`:
```
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
KAGGLE_API_TOKEN=            # optional
APP_ENV=development
LOG_LEVEL=INFO
```

### 5. Run the server

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 6. Access documentation

- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc
- **OpenAPI schema:** http://127.0.0.1:8000/openapi.json

## API Endpoints

### Health & Readiness

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (no Gemini call) |
| GET | `/ready` | Readiness check (shows config state) |

### Datasets

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/datasets/search` | Keyword search across the verified sources (no Gemini) |
| POST | `/api/v1/datasets/recommend` | Full pipeline: requirements, discovery, ranking, intelligence, final recommendation |

`recommend` request: `{"description": "...", "limit": 10}`. Each entry of `recommendations`
carries `dataset`, `ranking_score`, `ranking_factors`, `intelligence` and `confidence`;
the response also has `final_recommendation` and `meta` (which reports the extra enrichment
and confirmation requests made). Errors use the same `ErrorResponse` envelope under
`detail`.

### Project Analysis

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/projects/analyze` | Analyze a project description |

#### Request

```json
{
  "description": "I want to build a phishing detection system using machine learning on URL features."
}
```

#### Success Response

```json
{
  "success": true,
  "request_id": "uuid",
  "api_version": "v1",
  "data": {
    "input": { "word_count": 12 },
    "project_understanding": {
      "domain": { "value": "Cybersecurity", "confidence": 0.9 },
      "subdomain": { "value": "Web Security", "confidence": 0.85 },
      "problem_type": { "value": "Machine Learning", "confidence": 0.9 },
      "ml_tasks": [{ "task": "classification", "confidence": 0.87 }],
      "objective": "...",
      "target": { "value": "phishing/legitimate", "confidence": 0.7, "status": "inferred" }
    },
    "keywords": { "canonical": [...], "dictionary_matches": [...], "provenance": [...] },
    "dataset_requirements": { "feature_requirements": [...], ... },
    "analysis": { "overall_confidence": 0.85, "ambiguities": [...], "needs_clarification": false }
  },
  "meta": { "model": "gemini-3.1-flash-lite", "dictionary_version": "0.2.0", "processing_time_ms": 1234.5, "timestamp": "..." }
}
```

#### Error Response

```json
{
  "success": false,
  "error": {
    "code": "PROJECT_DESCRIPTION_TOO_LONG",
    "message": "Project description must not exceed 150 words.",
    "details": { "word_count": 151, "maximum_allowed": 150 }
  },
  "request_id": "uuid"
}
```

#### Error Codes

| Code | Description |
|------|-------------|
| `PROJECT_DESCRIPTION_REQUIRED` | Description field is missing |
| `PROJECT_DESCRIPTION_EMPTY` | Description is empty or whitespace only |
| `PROJECT_DESCRIPTION_TOO_LONG` | Exceeds 150-word limit |
| `LLM_AUTHENTICATION_ERROR` | Gemini API key is invalid |
| `LLM_RATE_LIMITED` | Gemini rate limit exceeded |
| `LLM_TIMEOUT` | Gemini request timed out |
| `LLM_UNAVAILABLE` | Gemini service unavailable |
| `LLM_INVALID_RESPONSE` | Gemini returned invalid structured data |
| `INTERNAL_ERROR` | Unexpected server error |

## Keyword Knowledge Base

The dictionary at `backend/data/keyword_dictionary.json` is a project-to-dataset knowledge base with **148 concepts** across **22 domains**:

- Artificial Intelligence (ML, Deep Learning, NLP, Computer Vision, Foundational ML)
- Cybersecurity (Network Security, Web Security, Fraud Detection)
- IoT (Internet of Things, Sensor Data, Edge Computing)
- Healthcare (Medical Imaging, Clinical Analysis, Genomics)
- Finance (Trading, Risk & Compliance)
- Education (Student Performance, Learning Analytics)
- Energy (Consumption, Forecasting)
- Agriculture, Environment, Manufacturing, Transportation
- Retail, Social Media, Sports, Real Estate, HR
- Telecom, Robotics, Operations Research
- Geospatial Analytics, Public Safety, Entertainment

Each concept includes: canonical term, synonyms, aliases, abbreviations, related concepts, ML tasks, dataset requirements, feature requirements, and target requirements.

## Testing

### Run unit tests

```bash
cd datapilot/backend
python -m pytest tests/unit/ -v
```

### Run with shorter output

```bash
python -m pytest tests/unit/ -v --tb=short
```

**Current test status:** 1056 unit tests, all passing, and identical under `PYTHONHASHSEED` 0 / 5 / 88 / 1234.

Other verification (all in `datapilot/backend`):

```bash
python -m scripts.run_evaluation   # offline ranking benchmark (no network)
python -m scripts.run_stress       # offline controlled stress test
python -m scripts.run_e2e          # LIVE: real Kaggle, Hugging Face and Gemini; spends a few Gemini calls
```

The benchmark figures in `evaluation/RANKING_EVALUATION.md` come from a corpus this project
wrote; they are not real-world accuracy.

## Postman

Import the collection from:
```
postman/DataPilot-Project-Analyzer.postman_collection.json
```

Set the `base_url` variable to `http://localhost:8000`.

## Security

- API keys are stored in `.env` (never committed)
- `.env` is included in `.gitignore`
- API keys are never logged, printed, or returned in responses
- User descriptions are treated as untrusted data
- Prompt injection attempts are mitigated

## Currently NOT Implemented

- User B: uploading and analysing your own dataset (CSV/Excel profiling, statistics, quality)
- Raw-data risks (class imbalance, leakage, missing values, duplicates) - they need the file
- Sources beyond Kaggle and Hugging Face (UCI, data.gov, GitHub, papers)
- Frontend
- Database
- RAG / Vector Search / Embeddings
- User Authentication

See `DEVELOPMENT_STATE.md` for current status, known limitations and the next task, and
`DEVELOPMENT_CONTEXT.md` for the longer technical history of Phases 1 and 2.

## File Structure

```
DataPilot/
├── README.md
├── DEVELOPMENT_STATE.md          # canonical status / handoff
├── DEVELOPMENT_CONTEXT.md        # long-form Phase 1-2 history
├── .gitignore
├── datapilot/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── api/v1/project_analysis.py, datasets.py
│   │   │   ├── core/config.py, logging_config.py
│   │   │   ├── schemas/project.py, dataset.py, intelligence.py, recommendation.py
│   │   │   ├── services/
│   │   │   │   ├── keyword_service.py, llm_service.py
│   │   │   │   ├── reconciliation_service.py, project_analyzer.py
│   │   │   │   └── datasets/      # sources, normalizer, dedup, ranking,
│   │   │   │                      # intelligence, recommendation, evaluation/
│   │   │   └── prompts/project_analysis.py, dataset_discovery.py
│   │   ├── data/keyword_dictionary.json
│   │   ├── scripts/               # run_evaluation, run_stress, run_e2e
│   │   ├── evaluation/            # benchmark reports and the live E2E report
│   │   ├── tests/unit/
│   │   ├── requirements.txt
│   │   ├── .env.example
│   │   └── .env
│   └── postman/
│       └── DataPilot-Project-Analyzer.postman_collection.json
```
