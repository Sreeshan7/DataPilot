"""Shared fixtures for the Phase 5 (project intelligence) tests.

Importance is computed with light settings (fewer trees and shuffles) so the suite stays fast; the
benchmark and the API defaults exercise the production settings. Outcomes are cached per dataset:
analysis is deterministic and the callers never mutate it.
"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache
from typing import Optional

from app.core.config import Settings
from app.services.dataset_analysis.evaluation.datasets import GENERATORS, HELDOUT_GENERATORS, SyntheticDataset
from app.services.dataset_analysis.service import AnalysisOutcome, AnalysisRequest, DatasetAnalysisService
from app.services.project_intelligence import ProjectIntelligenceService

ALL = {**GENERATORS, **HELDOUT_GENERATORS}


def light_settings(**overrides) -> Settings:
    values = dict(
        gemini_api_key="", kaggle_api_token="", app_env="testing", external_enabled=False,
        importance_max_rows=800, importance_n_estimators=40, importance_n_repeats=3, importance_max_features=20,
    )
    values.update(overrides)
    return Settings(**values)


SETTINGS = light_settings()


def run(coro):
    return asyncio.run(coro)


def analyze_dataset(dataset: SyntheticDataset, settings: Settings = SETTINGS, target: Optional[str] = None) -> AnalysisOutcome:
    service = DatasetAnalysisService(settings)
    return run(service.analyze(AnalysisRequest(
        stream=io.BytesIO(dataset.csv_bytes()), file_name=f"{dataset.name}.csv", target_hint=target)))


@lru_cache(maxsize=None)
def outcome_of(name: str, target: Optional[str] = None) -> AnalysisOutcome:
    return analyze_dataset(ALL[name](), SETTINGS, target)


def build(outcome: AnalysisOutcome, goal: Optional[str] = None, use_ai: bool = False, writer=None,
          settings: Settings = SETTINGS):
    return run(ProjectIntelligenceService(settings, writer=writer).build(outcome, goal=goal, use_ai=use_ai))


@lru_cache(maxsize=None)
def intelligence_of(name: str, target: Optional[str] = None):
    return build(outcome_of(name, target))
