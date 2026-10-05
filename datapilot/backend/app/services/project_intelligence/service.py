"""Phase 5 orchestration: from an analysed dataset to project intelligence.

    analysis -> understanding (target + task) -> feature intelligence (actual rows only)
             -> model recommendations -> grounded project ideas -> [optional Gemini prose]
             -> [optional compatibility with the user's goal]

Every step reads the Phase 4 analysis (and, for importance, the in-memory parsed table). Nothing
is downloaded, no file is re-read and no column is invented. A failure in any enrichment step
degrades that step with a stated reason; it never fails the analysis the user asked for.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from app.core import progress
from app.core.config import Settings, get_settings
from app.core.logging_config import get_logger
from app.schemas.dataset_analysis import AnalysisMode
from app.schemas.project_intelligence import (
    CAUSALITY_CAVEAT,
    FeatureIntelligence,
    ProjectIntelligence,
)
from app.services.dataset_analysis.errors import AnalysisError
from app.services.dataset_analysis.safety import Deadline
from app.services.dataset_analysis.service import AnalysisOutcome
from app.services.project_intelligence import features
from app.services.project_intelligence.compatibility import assess_compatibility
from app.services.project_intelligence.gemini import IdeaWriter, enhance_ideas
from app.services.project_intelligence.ideas import MAX_IDEAS, generate_ideas, validate_ideas
from app.services.project_intelligence.models import NO_PROMISE, recommend_models
from app.services.project_intelligence.shape import shape_of
from app.services.project_intelligence.understanding import understand

logger = get_logger(__name__)

MIN_GOOD_IDEAS = 3


class ProjectIntelligenceService:
    def __init__(self, settings: Optional[Settings] = None, writer: Optional[IdeaWriter] = None) -> None:
        self._settings = settings or get_settings()
        self._writer = writer

    def _blueprint_writer(self):
        """The writer used for build plans: an injected one (tests) or the real Gemini client."""
        if self._writer is not None:
            return self._writer
        from app.services.llm_service import LLMService

        return LLMService(self._settings)

    def _idea_writer(self) -> Optional[IdeaWriter]:
        """The Gemini client, built only if it is ever needed (so a plain analysis never touches it)."""
        if self._writer is None:
            from app.services.llm_service import LLMService

            self._writer = LLMService(self._settings)
        return self._writer

    async def build(
        self, outcome: AnalysisOutcome, goal: Optional[str] = None, use_ai: bool = False
    ) -> ProjectIntelligence:
        analysis = outcome.analysis
        understanding = understand(analysis)
        shape = shape_of(analysis)
        warnings: list[str] = []
        target = understanding.best_target.column if understanding.best_target else None

        # ---- feature intelligence: rows only, never for a registry-only dataset ----------
        if analysis.mode is AnalysisMode.FILE_PARSED and outcome.parsed is not None:
            progress.start("signal", "Measuring which columns carry signal")
            deadline = Deadline(self._settings.analysis_time_limit_seconds)
            try:
                intel = await asyncio.to_thread(
                    features.build_feature_intelligence, outcome.parsed, understanding, self._settings, deadline)
            except AnalysisError as exc:
                intel = features.unavailable(
                    "Feature analysis stopped because it took longer than the allowed time.", target)
                warnings.append(f"Feature importance was skipped ({exc.code.value}).")
            except Exception as exc:  # an enrichment must never fail the analysis
                logger.error("Feature intelligence failed: %s", type(exc).__name__)
                intel = features.unavailable("Feature analysis could not be completed for this file.", target)
                warnings.append("Feature importance was skipped because of an internal error.")
        else:
            intel = features.external_feature_intelligence(analysis, target)

        progress.done("signal", "measured" if intel.available else "not available for this data")
        progress.start("ideas", "Drafting project ideas from your columns")
        models = recommend_models(understanding, shape)
        ideas = validate_ideas(generate_ideas(analysis, understanding, intel, shape), shape.names)[:MAX_IDEAS]

        progress.done("ideas", f"{len(ideas)} ideas")
        writer = self._idea_writer() if use_ai and self._settings.analysis_gemini_enabled else None
        ideas, gemini, gemini_warnings = await enhance_ideas(
            ideas, shape, goal, writer, self._settings, requested=use_ai)
        ideas = validate_ideas(ideas, shape.names)  # the guard runs again on whatever came back
        from app.services.research.offer import request_for_idea

        ideas = [i.model_copy(update={"research_request": request_for_idea(i, analysis, goal)}) for i in ideas]
        from app.services.project_intelligence.blueprint import attach_blueprints, rank_ideas

        progress.start("blueprint", "Writing detailed build plans")
        ideas, ai_plans, plan_note = await attach_blueprints(rank_ideas(ideas), shape, intel, goal, self._blueprint_writer(), self._settings)
        progress.done("blueprint", f"{len(ideas)} plans" + (f", {ai_plans} AI-written" if ai_plans else ""))
        if plan_note:
            warnings.append(plan_note)
        warnings.extend(gemini_warnings)

        compatibility = assess_compatibility(goal, analysis, understanding, shape) if goal else None

        limitations = [NO_PROMISE]
        if intel.available:
            limitations.append(CAUSALITY_CAVEAT)
        elif analysis.mode is AnalysisMode.EXTERNAL_METADATA:
            limitations.append(
                "Everything here is drawn from the registry's published schema: no rows were read, so no "
                "feature importance, class balance or data quality is claimed.")
        if len(ideas) < MIN_GOOD_IDEAS:
            limitations.append(
                f"Only {len(ideas)} project idea(s) could be grounded in what is known about this dataset; "
                "more need its rows, a confirmed target or a known schema.")

        return ProjectIntelligence(
            understanding=understanding, feature_intelligence=intel, project_ideas=ideas,
            model_recommendations=models, compatibility=compatibility, gemini=gemini,
            limitations=limitations, warnings=warnings)
