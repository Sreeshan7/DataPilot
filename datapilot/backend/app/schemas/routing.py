"""Models for automatic user-journey detection.

DataPilot serves two journeys that a client should never have to choose between:

* ``USER_A_DATASET_DISCOVERY`` - "I need a dataset for my project."
* ``USER_B_DATASET_ANALYSIS`` - "I already have / know a dataset; analyse it."

The router reads the input and decides. When the evidence is not decisive it says so
and asks, rather than guessing: sending someone down the wrong journey is a worse
outcome than one clarifying question.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UserIntent(str, Enum):
    USER_A_DATASET_DISCOVERY = "USER_A_DATASET_DISCOVERY"
    USER_B_DATASET_ANALYSIS = "USER_B_DATASET_ANALYSIS"
    AMBIGUOUS = "AMBIGUOUS"


class IntentRequest(BaseModel):
    """Everything the router may inspect. Every field is optional."""

    message: Optional[str] = Field(
        None, max_length=5000, description="The user's free-text message or project description."
    )
    has_file: bool = Field(False, description="Whether a file accompanies the request.")
    file_name: Optional[str] = Field(None, max_length=300)
    dataset_name: Optional[str] = Field(None, max_length=300)
    dataset_url: Optional[str] = Field(None, max_length=2048)
    project_goal: Optional[str] = Field(
        None, max_length=5000, description="What the user wants to build, if stated separately."
    )

    @field_validator("message", "file_name", "dataset_name", "dataset_url", "project_goal")
    @classmethod
    def blank_means_absent(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None


class DatasetReferenceHint(BaseModel):
    """A dataset the user appears to have named, extracted from their input."""

    name: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None
    evidence: str = ""


class IntentResolution(BaseModel):
    """The routing decision and the reasons for it."""

    intent: UserIntent
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reason: str
    required_next_input: Optional[str] = Field(
        None, description="What the user must still supply for the chosen journey to run."
    )
    clarification: Optional[str] = Field(
        None, description="A question to ask the user; set only when intent is AMBIGUOUS."
    )
    follow_up_intent: Optional[UserIntent] = Field(
        None,
        description=(
            "A journey that may follow the primary one, e.g. USER_B first and then "
            "USER_A when the user's dataset turns out to be unsuitable."
        ),
    )
    dataset_reference: Optional[DatasetReferenceHint] = None
