"""
state_machine.py — Investigation state machine for POLICYX.

Tracks every step of the "find the previous policy" workflow.
States are plain strings so they serialise to JSON cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# All valid states (in order)
STATES = [
    "IDLE",
    "COLLECTING_CONTEXT",
    "EXTRACTING_METADATA",
    "SEARCHING_SOURCES",
    "RANKING_CANDIDATES",
    "VERIFYING_PROVENANCE",
    "AWAITING_CONFIRMATION",
    "DOWNLOADING_DOCUMENT",
    "COMPARING_POLICIES",
    "COMPLETED",
    "FAILED",
]

STATE_DESCRIPTIONS = {
    "IDLE":                   "Waiting for user input.",
    "COLLECTING_CONTEXT":     "Collecting institution and policy information.",
    "EXTRACTING_METADATA":    "Analysing the uploaded policy document.",
    "SEARCHING_SOURCES":      "Searching authoritative sources for the previous version.",
    "RANKING_CANDIDATES":     "Ranking candidate documents by relevance.",
    "VERIFYING_PROVENANCE":   "Evaluating source authority and provenance signals.",
    "AWAITING_CONFIRMATION":  "Waiting for you to confirm the candidate document.",
    "DOWNLOADING_DOCUMENT":   "Retrieving the confirmed candidate document.",
    "COMPARING_POLICIES":     "Running the ML comparison pipeline.",
    "COMPLETED":              "Investigation and comparison complete.",
    "FAILED":                 "Investigation could not be completed.",
}


@dataclass
class InvestigationState:
    """Full mutable state for one investigation session."""

    # Core state
    state: str = "IDLE"
    error: str = ""

    # User-supplied context (Q1 + Q2)
    institution_name: str = ""
    affiliating_university: str = ""
    policy_title: str = ""
    department: str = ""
    academic_year: str = ""

    # Extracted from uploaded newer PDF
    extracted_metadata: dict = field(default_factory=dict)

    # Search + ranking
    candidates: list[dict] = field(default_factory=list)

    # Confirmed candidate
    confirmed_candidate: dict | None = None

    # Audit trail
    audit: list[dict] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def transition(self, new_state: str, note: str = "") -> None:
        assert new_state in STATES, f"Unknown state: {new_state}"
        self.audit.append({
            "from": self.state,
            "to": new_state,
            "note": note,
            "ts": datetime.utcnow().isoformat(),
        })
        self.state = new_state
        self.updated_at = datetime.utcnow().isoformat()

    def fail(self, reason: str) -> None:
        self.error = reason
        self.transition("FAILED", reason)

    def description(self) -> str:
        return STATE_DESCRIPTIONS.get(self.state, self.state)

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "description": self.description(),
            "error": self.error,
            "institution_name": self.institution_name,
            "affiliating_university": self.affiliating_university,
            "policy_title": self.policy_title,
            "department": self.department,
            "academic_year": self.academic_year,
            "extracted_metadata": self.extracted_metadata,
            "candidates": self.candidates,
            "confirmed_candidate": self.confirmed_candidate,
            "audit": self.audit,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }
