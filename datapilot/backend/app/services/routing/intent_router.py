"""Automatic user-journey detection: USER A (find a dataset) or USER B (analyse one).

Deterministic and offline: no model is called to decide where a request goes, so the
decision is reproducible, free, and explainable by listing the evidence that drove it.

How it decides
--------------
Each rule that fires contributes a weight to one side. A side's score combines its
weights with a noisy-OR (``1 - prod(1 - w)``), so independent evidence accumulates but
never exceeds 1 and no single weak cue dominates. The winner needs both a minimum score
and a clear margin over the other side; otherwise the answer is ``AMBIGUOUS`` together
with a question to ask. A wrong guess sends the user down the wrong journey, which costs
more than one clarifying question.

Two judgments are deliberate:

* **Ownership beats intent.** "I want to predict X" opens both journeys ("...with my
  data" and "...find me data"), so the project-description cue is weak whenever the user
  also says they have a dataset, and strong only when nothing says so.
* **A request for an alternative is a follow-up, not a switch.** "I have this dataset but
  need a better one" is USER_B first; the wish for another dataset is reported as
  ``follow_up_intent`` so the journeys connect instead of excluding each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging_config import get_logger
from app.schemas.routing import (
    DatasetReferenceHint,
    IntentRequest,
    IntentResolution,
    UserIntent,
)
from app.services.dataset_analysis.references import (
    UnsafeURLError,
    extract_dataset_name,
    extract_urls,
    parse_dataset_url,
)

logger = get_logger(__name__)

#: Both sides need at least this much evidence to be chosen at all.
MIN_DECISIVE_SCORE = 0.5
#: ...and must beat the other side by this much, or the request is ambiguous.
MIN_MARGIN = 0.25

_DATA_NOUN = r"(?:data ?sets?|data|csv|excel|spreadsheet|xlsx|xls|file|table|database|workbook)"

# ---- USER B cues ----------------------------------------------------------

_POSSESSION = re.compile(
    rf"\b(?:i|we)\s*(?:'ve|ve|\s+have|\s+had|\s+got|\s+own|\s+downloaded|\s+uploaded|\s+attached|"
    rf"\s+collected|\s+already\s+have|\s+already\s+know|\s+am\s+using|\s+was\s+given)\b"
    rf"[^.?!]{{0,60}}?\b{_DATA_NOUN}\b",
    re.IGNORECASE,
)
_POSSESSIVE = re.compile(rf"\bmy\s+(?:own\s+)?(?:\w+\s+)?{_DATA_NOUN}\b", re.IGNORECASE)
_DEICTIC = re.compile(
    rf"\b(?:this|the\s+attached|the\s+uploaded|attached|uploaded|the\s+following)\s+(?:\w+\s+)?{_DATA_NOUN}\b",
    re.IGNORECASE,
)
_FILE_MENTION = re.compile(r"(?<![\w/\\])[\w\-]{1,80}\.(?:csv|xlsx|xls|tsv)\b", re.IGNORECASE)
_ANALYSIS_VERB = re.compile(
    r"\b(?:analy[sz]e|profile|inspect|explore|summari[sz]e|examine|review|audit|clean|check)\b",
    re.IGNORECASE,
)
_WHAT_CAN_I = re.compile(
    r"\bwhat\s+(?:can|could|should)\s+(?:i|we)\s+(?:build|do|make|create|train|predict)\b|"
    r"\bwhat\s+(?:projects?|ml\s+projects?|models?)\b[^.?!]{0,40}\b(?:with|from|using)\b",
    re.IGNORECASE,
)
_COLUMN_TALK = re.compile(r"\b(?:target\s+column|which\s+columns?|use\s+column|the\s+columns?)\b", re.IGNORECASE)

# ---- USER A cues ----------------------------------------------------------

_DISCOVERY = re.compile(
    rf"\b(?:find|search\s+for|look(?:ing)?\s+for|recommend|suggest|locate|discover|source)\b"
    rf"[^.?!]{{0,40}}?\b(?:data ?sets?|data)\b|"
    rf"\b(?:need|want|require)\s+(?:a|an|some|the\s+right|good|suitable)?\s*(?:good\s+|suitable\s+)?(?:data ?sets?)\b|"
    rf"\bwhere\s+(?:can|do)\s+(?:i|we)\s+(?:find|get|download)\b|"
    rf"\b(?:which|what|any)\s+data ?sets?\b",
    re.IGNORECASE,
)
#: "If you can't find ..." is a fallback clause, not a request to find a dataset now.
_CONDITIONAL_FIND = re.compile(
    r"\bif\s+you\s+(?:can(?:'t|not)|could(?:n't|\s+not)|are\s+unable\s+to|don't|do\s+not)\s+find\b[^.?!;]*",
    re.IGNORECASE,
)
_ALTERNATIVE = re.compile(
    r"\b(?:better|another|different|alternative|other|new)\s+(?:one|data ?sets?|data|option)\b|"
    r"\bif\s+(?:it|this|that)\s+(?:is|isn't|is\s+not|doesn't|does\s+not)\b[^.?!]{0,40}\b(?:suitable|fit|work)\b|"
    r"\bif\s+you\s+can(?:'t|not)\s+find\b",
    re.IGNORECASE,
)
_WANT_TO = re.compile(
    r"\b(?:i|we)\s+(?:want|would\s+like|plan|am\s+planning|need|am\s+trying|intend|hope)\s+to\s+"
    r"(?:build|develop|create|make|predict|detect|classify|forecast|identify|design|train|implement|estimate|recommend|cluster)\b",
    re.IGNORECASE,
)
_IMPERATIVE_PROJECT = re.compile(
    r"^\s*(?:predict|build|develop|detect|forecast|classify|identify|estimate|recommend|cluster)\b",
    re.IGNORECASE,
)


@dataclass
class _Scores:
    a: list[tuple[float, str]] = field(default_factory=list)
    b: list[tuple[float, str]] = field(default_factory=list)
    alternative: list[tuple[float, str]] = field(default_factory=list)

    @staticmethod
    def combine(items: list[tuple[float, str]]) -> float:
        remaining = 1.0
        for weight, _ in items:
            remaining *= 1.0 - weight
        return round(1.0 - remaining, 4)


def _confidence(winner: float, loser: float) -> float:
    return round(min(0.99, max(0.5, 0.5 + 0.5 * (winner - loser))), 2)


class IntentRouter:
    """Decides which journey a request belongs to. Stateless and deterministic."""

    def resolve(self, request: IntentRequest) -> IntentResolution:
        text = " ".join(part for part in (request.message, request.project_goal) if part)
        scores = _Scores()
        reference = self._collect_b(request, text, scores)
        has_b_evidence = bool(scores.b)
        self._collect_a(request, text, scores, has_b_evidence)

        sa, sb = _Scores.combine(scores.a), _Scores.combine(scores.b)
        decisive_b = sb >= MIN_DECISIVE_SCORE
        # A usable dataset link counts the same whether it came in its own field or
        # was pasted into the message.
        has_link = bool(request.dataset_url) or bool(reference and reference.url)

        # An alternative-dataset request only counts as a primary USER_A signal when the
        # user never said they have a dataset; otherwise it is a follow-up.
        follow_up: Optional[UserIntent] = None
        if scores.alternative:
            if decisive_b:
                follow_up = UserIntent.USER_A_DATASET_DISCOVERY
            else:
                scores.a.extend(scores.alternative)
                sa = _Scores.combine(scores.a)

        evidence = [text for _, text in scores.b] + [text for _, text in scores.a]
        logger.info(
            "Intent routing: A=%.2f B=%.2f follow_up=%s", sa, sb, follow_up.value if follow_up else None
        )

        # An explicit file or dataset link is a statement of possession; a discovery
        # phrase next to it only adds a follow-up.
        winner_b = sb > sa
        winner_score, loser_score = (sb, sa) if winner_b else (sa, sb)

        if (
            winner_score < MIN_DECISIVE_SCORE
            or (winner_score - loser_score) < MIN_MARGIN
        ) and not (decisive_b and (request.has_file or has_link)):
            return self._ambiguous(sa, sb, evidence, reference)

        if decisive_b and (request.has_file or has_link):
            winner_b = True
            winner_score = sb
            if sa >= MIN_DECISIVE_SCORE and follow_up is None:
                follow_up = UserIntent.USER_A_DATASET_DISCOVERY
            # A discovery phrase beside a file or link is the follow-up, not a rival
            # reading, so it must not be subtracted from confidence in the file.
            loser_score = 0.0 if follow_up is not None else sa

        if winner_b:
            return self._user_b(request, reference, evidence, winner_score, loser_score, follow_up)
        return self._user_a(request, text, evidence, winner_score, loser_score, reference)

    # ------------------------------------------------------------------
    # Evidence collection
    # ------------------------------------------------------------------

    def _collect_b(
        self, request: IntentRequest, text: str, scores: _Scores
    ) -> Optional[DatasetReferenceHint]:
        reference: Optional[DatasetReferenceHint] = None

        if request.has_file:
            scores.b.append((0.95, "A file accompanies the request"))
        if request.file_name:
            scores.b.append((0.2, f"A file name was given ({request.file_name})"))

        urls = ([request.dataset_url] if request.dataset_url else []) + extract_urls(text)
        for url in urls:
            try:
                parsed = parse_dataset_url(url)
            except UnsafeURLError:
                scores.b.append((0.6, "The user supplied a link to a dataset-like page"))
                continue
            scores.b.append((0.92, f"The user supplied a {parsed.source} dataset URL"))
            reference = reference or DatasetReferenceHint(
                name=parsed.name_hint,
                url=parsed.canonical_url,
                source=parsed.source,
                evidence=f"{parsed.source} dataset URL",
            )

        if request.dataset_name:
            scores.b.append((0.88, f"A dataset name was supplied ({request.dataset_name})"))
            reference = reference or DatasetReferenceHint(
                name=request.dataset_name, evidence="dataset_name field"
            )

        if _POSSESSION.search(text):
            scores.b.append((0.88, "The user says they have/uploaded/downloaded a dataset or file"))
        if _POSSESSIVE.search(text):
            scores.b.append((0.8, "The user refers to 'my' dataset or file"))
        if _DEICTIC.search(text):
            scores.b.append((0.65, "The user refers to 'this/the attached' dataset or file"))
        if _FILE_MENTION.search(text):
            scores.b.append((0.8, "The message names a data file (.csv/.xlsx)"))
        if _WHAT_CAN_I.search(text):
            scores.b.append((0.7, "The user asks what can be built from it"))
        if _COLUMN_TALK.search(text):
            scores.b.append((0.45, "The user talks about columns of a dataset"))

        name_hint = extract_dataset_name(text)
        if name_hint is not None:
            reference = reference or DatasetReferenceHint(
                name=name_hint.name, evidence=name_hint.evidence
            )
            if not name_hint.generic and name_hint.confidence >= 0.8:
                scores.b.append((0.4, f"A named dataset is mentioned ({name_hint.name})"))

        if _ANALYSIS_VERB.search(text) and (
            scores.b or name_hint is not None or re.search(rf"\b{_DATA_NOUN}\b", text, re.IGNORECASE)
        ):
            scores.b.append((0.6, "The user asks for analysis of an existing dataset"))
        return reference

    def _collect_a(
        self, request: IntentRequest, text: str, scores: _Scores, has_b_evidence: bool
    ) -> None:
        if _DISCOVERY.search(_CONDITIONAL_FIND.sub(" ", text)):
            scores.a.append((0.9, "The user asks to find or recommend a dataset"))
        if _ALTERNATIVE.search(text):
            scores.alternative.append((0.6, "The user wants a better or alternative dataset"))

        b_score = _Scores.combine(scores.b)
        if _WANT_TO.search(text):
            weight = 0.75 if b_score < 0.3 else 0.35
            scores.a.append((weight, "The user describes a project they want to build"))
        elif _IMPERATIVE_PROJECT.search(text) and b_score < 0.3:
            scores.a.append((0.5, "The message reads as a project description"))

        if request.project_goal and not has_b_evidence:
            scores.a.append((0.85, "A project goal was given and no dataset was mentioned"))

    # ------------------------------------------------------------------
    # Outcomes
    # ------------------------------------------------------------------

    def _user_b(
        self,
        request: IntentRequest,
        reference: Optional[DatasetReferenceHint],
        evidence: list[str],
        winner: float,
        loser: float,
        follow_up: Optional[UserIntent],
    ) -> IntentResolution:
        needs: Optional[str] = None
        has_link = bool(request.dataset_url) or bool(reference and reference.url)
        has_usable_reference = bool(
            request.has_file
            or has_link
            or request.dataset_name
            or (reference is not None and (reference.url or reference.name))
        )
        specific_name = False
        if reference is not None and reference.name and not reference.url:
            hint = extract_dataset_name(reference.name) or None
            specific_name = bool(request.dataset_name) or (
                hint is not None and not hint.generic
            )
        if not has_usable_reference or (
            not request.has_file and not has_link and not specific_name
        ):
            needs = (
                "Upload the file, or give the dataset's name or its Kaggle / Hugging Face / "
                "UCI / OpenML URL."
            )
        return IntentResolution(
            intent=UserIntent.USER_B_DATASET_ANALYSIS,
            confidence=_confidence(winner, loser),
            evidence=evidence,
            reason="The user already has, or has named, a dataset and wants it analysed.",
            required_next_input=needs,
            follow_up_intent=follow_up,
            dataset_reference=reference,
        )

    def _user_a(
        self,
        request: IntentRequest,
        text: str,
        evidence: list[str],
        winner: float,
        loser: float,
        reference: Optional[DatasetReferenceHint],
    ) -> IntentResolution:
        needs: Optional[str] = None
        description = request.project_goal or request.message or ""
        if len(re.findall(r"\w+", description)) < 6:
            needs = (
                "Describe the project: what you want to predict or detect, and from what "
                "kind of data."
            )
        return IntentResolution(
            intent=UserIntent.USER_A_DATASET_DISCOVERY,
            confidence=_confidence(winner, loser),
            evidence=evidence,
            reason="The user wants a dataset for a project and has not said they have one.",
            required_next_input=needs,
            dataset_reference=reference,
        )

    def _ambiguous(
        self,
        sa: float,
        sb: float,
        evidence: list[str],
        reference: Optional[DatasetReferenceHint],
    ) -> IntentResolution:
        if sa > 0 and sb > 0:
            clarification = (
                "It looks like you may already have a dataset. Do you want me to analyze that "
                "dataset, or find a different dataset for your project?"
            )
            reason = "The input contains evidence for both journeys and neither is clearly stronger."
            confidence = 0.7
        else:
            clarification = (
                "Do you want me to find a dataset for a project you are working on, or analyze a "
                "dataset you already have? If you have a file, upload it; if you have a project "
                "idea, describe what you want to predict."
            )
            reason = "Nothing in the input says whether you need a dataset or already have one."
            confidence = 0.9
        return IntentResolution(
            intent=UserIntent.AMBIGUOUS,
            confidence=confidence,
            evidence=evidence,
            reason=reason,
            clarification=clarification,
            required_next_input="Say whether you want to find a dataset or analyze one you have.",
            dataset_reference=reference,
        )
