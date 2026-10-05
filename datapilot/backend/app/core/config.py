from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    #: A second key doubles the free-tier allowance; calls spread across both and fail over on 429/503.
    gemini_api_key_2: str = ""
    #: DataPilot stops itself below the published free-tier limits (per key).
    gemini_requests_per_minute: int = 8
    gemini_requests_per_day: int = 900
    gemini_model: str = "gemini-3.1-flash-lite"
    app_env: str = "development"
    log_level: str = "INFO"

    api_v1_prefix: str = "/api/v1"
    app_title: str = "DataPilot Backend"
    app_version: str = "0.1.0"

    max_project_description_words: int = 150
    max_project_description_chars: int = 5000

    llm_request_timeout: float = 60.0
    llm_max_retries: int = 1

    # --- Phase 2: Kaggle dataset discovery ---------------------------------
    # The token is read from KAGGLE_API_TOKEN. It is never logged or returned.
    kaggle_api_token: str = ""
    kaggle_api_base_url: str = "https://www.kaggle.com/api/v1"
    kaggle_request_timeout: float = 20.0
    kaggle_enabled: bool = True
    kaggle_default_limit: int = 10
    kaggle_max_limit: int = 50
    # Kaggle ignores `pageSize` on datasets/list and always returns 20 rows,
    # so the limit is satisfied by walking a bounded number of pages.
    kaggle_max_pages: int = 3

    # --- Phase 2: GenAI dataset candidate discovery ------------------------
    dataset_discovery_genai_enabled: bool = True
    dataset_discovery_genai_max_candidates: int = 8
    dataset_discovery_max_query_terms: int = 6

    # --- Phase 3B: metadata enrichment ---------------------------------------
    # Kaggle's list endpoint returns an empty description for every dataset, so a
    # first-pass ranking only sees a title, subtitle and tags. The top candidates
    # are therefore re-read from the per-dataset endpoints (two calls each) and
    # re-ranked. Bounded by top_k so the extra traffic is predictable.
    dataset_enrichment_enabled: bool = True
    dataset_enrichment_top_k: int = 5

    # --- Phase 3C: registry verification of AI suggestions -------------------
    # At most this many unconfirmed AI suggestions are looked up by name in the
    # verified registries. A hit is merged (the registry record wins); a miss
    # leaves the suggestion unverified. 0 disables the lookups.
    dataset_verification_lookups: int = 3

    # --- Phase 3C: Hugging Face Hub (public, no credentials) -----------------
    huggingface_enabled: bool = True
    huggingface_api_base_url: str = "https://huggingface.co/api"
    huggingface_request_timeout: float = 20.0
    huggingface_max_limit: int = 50

    # --- Phase 4: analysing a dataset the user already has -------------------
    # Every limit exists to keep one request from exhausting memory or time. They
    # are settings, not constants, so a deployment can trade capacity for safety.
    #
    # 25 MiB: pandas holds one Python object per cell, so peak parse memory is
    # roughly ten times the file size; this keeps a worst-case upload in the low
    # hundreds of MB.
    analysis_max_upload_bytes: int = 25 * 1024 * 1024
    # Rows/columns/cells actually analysed. A file beyond them is truncated and the
    # analysis says so; it is never silently treated as complete.
    analysis_max_rows: int = 200_000
    analysis_max_columns: int = 500
    # rows x columns. Bounds memory independently of file size (a file of one-character
    # cells packs far more cells per byte than a typical one).
    analysis_max_cells: int = 2_000_000
    analysis_max_excel_sheets: int = 20
    # An .xlsx is a zip. These two stop a small upload that expands enormously
    # (a "zip bomb") before any of it is parsed.
    analysis_max_excel_uncompressed_bytes: int = 200 * 1024 * 1024
    analysis_max_excel_compression_ratio: int = 100
    # csv.field_size_limit: stops a single runaway cell from consuming memory.
    analysis_max_field_chars: int = 32_768
    # Rows used for the expensive analyses (duplicate detection, association); a
    # fixed seed makes the sample, and so every result, reproducible.
    analysis_sample_rows: int = 20_000
    analysis_random_seed: int = 42
    # Wall-clock budget for one analysis, checked between stages.
    analysis_time_limit_seconds: float = 30.0

    # Quality / target thresholds. Each is a rule of thumb, stated here so it can be
    # challenged, and reported alongside any finding that uses it.
    #  - 0.40 missing: beyond this, imputation is rarely trustworthy.
    quality_high_missing_ratio: float = 0.40
    #  - 0.99 share for one value: the column carries almost no information.
    quality_near_constant_ratio: float = 0.99
    #  - 0.98 distinct values per row: indistinguishable from a row identifier.
    quality_identifier_unique_ratio: float = 0.98
    #  - Tukey's 1.5 x IQR fences, the conventional "possible outlier" rule.
    quality_outlier_iqr_multiplier: float = 1.5
    #  - minority class below 10% of rows is usually worth handling explicitly.
    quality_imbalance_minority_ratio: float = 0.10
    #  - a single feature this predictive of the target is a leakage signal, not a find.
    quality_leakage_association: float = 0.98
    #  - below this many rows a supervised task cannot be validated meaningfully.
    task_min_rows_supervised: int = 30
    #  - a target with more distinct values than this is treated as continuous.
    task_max_classes: int = 20

    # --- Phase 4: external dataset research (metadata only, never the data) ---
    #: Read only the first 64 KB of a Kaggle dataset's main csv to get its real column header (never the dataset).
    external_peek_enabled: bool = True
    #: Last resort when no registry knows a dataset's structure: one cached Gemini + Google Search lookup.
    web_reader_enabled: bool = True
    web_reader_pages: int = 4                 # pages fetched and read per dataset lookup
    web_reader_page_bytes: int = 400_000
    web_lookup_enabled: bool = True
    web_lookup_timeout_seconds: float = 45.0
    external_enabled: bool = True
    external_request_timeout: float = 15.0
    # Responses are streamed and cut off here; a registry record is a few KB, so a
    # larger body is hostile or broken.
    external_max_response_bytes: int = 2 * 1024 * 1024
    external_max_redirects: int = 3
    uci_api_base_url: str = "https://archive.ics.uci.edu/api"
    openml_api_base_url: str = "https://www.openml.org/api/v1/json"

    # --- Phase 5: dataset -> project intelligence ----------------------------
    # Importance is estimated on a bounded, seeded sample so it cannot hang on a
    # large file; the sample size is reported with the result.
    importance_max_rows: int = 5_000
    importance_max_features: int = 40
    importance_n_repeats: int = 5
    importance_n_estimators: int = 100
    importance_test_fraction: float = 0.25
    # One structured Gemini call per analysis, for phrasing and ideation only. The
    # deterministic result is always produced first and is returned unchanged if the
    # call is disabled, unavailable or rejected.
    analysis_gemini_enabled: bool = True
    analysis_blueprint_enabled: bool = True   # detailed AI-written build plans for the top ideas (one batched call)
    analysis_blueprint_ideas: int = 3
    analysis_ai_name_enabled: bool = True     # read the dataset name out of a message the rules cannot parse (one small call)
    analysis_ai_target_enabled: bool = True   # goal-aware target pick (one small call, names and types only)
    analysis_gemini_max_columns: int = 60

    # Optional "Research this problem" (web search only; Gemini is at most ONE synthesis call).
    research_enabled: bool = True
    research_gemini_enabled: bool = True
    research_ai_assist_enabled: bool = True
    research_deep_analysis_enabled: bool = True   # per-paper and per-algorithm in-depth analysis (two parallel AI calls)
    research_deep_analysis_papers: int = 6
    research_fulltext_enabled: bool = True        # read open-access PDFs of the top papers
    research_fulltext_papers: int = 4
    research_fulltext_max_bytes: int = 4_000_000
    research_fulltext_timeout_seconds: float = 15.0
    research_search_url: str = "https://html.duckduckgo.com/html/"
    research_timeout_seconds: float = 12.0
    research_max_queries: int = 5
    research_results_per_query: int = 10
    research_query_delay_seconds: float = 1.0   # the keyless endpoint rate-limits bursts
    research_rate_limit_retry_seconds: float = 3.0
    research_max_response_bytes: int = 1_000_000

    # --- Phase 2: deterministic ranking weights ----------------------------
    # Three groups, and the split is the point: requirement factors own 70 of the
    # 100 points, publisher-side signals own 15, trust owns 15. Relevance
    # therefore decides the ranking, and popularity cannot buy its way past a
    # dataset that actually fits the project.
    #
    #   requirements (70)          publisher (15)
    #     domain_relevance  16       quality       8
    #     feature_match     16       popularity    5
    #     target_match      13       recency       2
    #     keyword_overlap    9
    #     task_match         8       trust (15)
    #     compatibility      8         verification 15
    ranking_weight_domain_relevance: float = 16.0
    ranking_weight_keyword_overlap: float = 9.0
    ranking_weight_feature_match: float = 16.0
    ranking_weight_target_match: float = 13.0
    ranking_weight_task_match: float = 8.0
    # Penalty for metadata that positively contradicts the request (a declared ML
    # task, label space or file modality the project cannot use).
    ranking_weight_compatibility: float = 8.0
    ranking_weight_quality: float = 8.0
    ranking_weight_popularity: float = 5.0
    ranking_weight_recency: float = 2.0
    ranking_weight_verification: float = 15.0

    # Trust applied to the verification factor depending on status.
    ranking_trust_verified: float = 1.0
    ranking_trust_unverified: float = 0.25

    # Popularity is log-scaled against these reference values.
    ranking_download_reference: int = 1_000_000
    ranking_vote_reference: int = 10_000

    # Recency decays linearly from 1.0 at age 0 to 0.0 at ranking_recency_max_age_days.
    ranking_recency_max_age_days: int = 1825

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
