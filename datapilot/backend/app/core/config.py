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
