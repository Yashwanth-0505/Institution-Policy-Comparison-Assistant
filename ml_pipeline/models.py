"""
models.py — Pydantic v2 data models for the POLICYX ML pipeline.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Supporting value objects
# ---------------------------------------------------------------------------


class SectionMetadata(BaseModel):
    """Metadata about the section a clause belongs to."""

    section_number: Optional[str] = None
    section_title: Optional[str] = None
    depth: int = 0  # nesting depth (0 = top-level)

    model_config = {"arbitrary_types_allowed": True}


class ExtractedClause(BaseModel):
    """A single extracted clause from a policy document."""

    document_id: str
    document_name: str
    version: str = ""
    clause_id: str
    heading: str = ""
    original_text: str
    normalized_text: str
    page_number: int
    section_metadata: SectionMetadata = Field(default_factory=SectionMetadata)
    flagged_for_review: bool = False
    review_reason: str = ""

    model_config = {"arbitrary_types_allowed": True}


class ClauseFeatures(BaseModel):
    """Feature vectors computed for a clause."""

    clause: ExtractedClause
    tfidf_vector: Any  # np.ndarray — not JSON-serialisable, kept as Any
    embedding: Optional[Any] = None  # np.ndarray | None

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class MatchType(str, Enum):
    EXACT = "exact"
    HYBRID = "hybrid"
    BASELINE = "baseline"
    ADDED_CANDIDATE = "added_candidate"
    REMOVED_CANDIDATE = "removed_candidate"
    UNMATCHED = "unmatched"
    NEEDS_REVIEW = "needs_review"


class ChangeType(str, Enum):
    UNCHANGED = "unchanged"
    WORDING_ONLY = "wording_only"
    SUBSTANTIVE = "substantive"
    ADDED = "added"
    REMOVED = "removed"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Matching models
# ---------------------------------------------------------------------------


class ClauseMatch(BaseModel):
    """Result of matching a clause from document A to document B."""

    clause_a: ExtractedClause
    clause_b: Optional[ExtractedClause] = None  # None for REMOVED_CANDIDATE
    match_score: float = 0.0
    number_score: float = 0.0
    heading_score: float = 0.0
    embedding_score: float = 0.0
    tfidf_score: float = 0.0
    match_type: MatchType = MatchType.UNMATCHED
    needs_review: bool = False
    review_reason: str = ""

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Change detection models
# ---------------------------------------------------------------------------


class ChangeSignal(BaseModel):
    """A single detected semantic signal between two clause versions."""

    signal_type: str  # e.g. "numeric_change", "date_change"
    description: str
    old_value: str = ""
    new_value: str = ""


class Citation(BaseModel):
    """A verifiable reference back to a source clause."""

    document_id: str
    document_name: str
    clause_id: str
    heading: str = ""
    page_number: int
    exact_text: str
    citation_valid: bool = True
    warning: str = ""


class BoundedSummary(BaseModel):
    """LLM- or deterministically-generated change summary."""

    summary: str
    classification_suggestion: str = ""  # hint back to the caller
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provider_used: str = "deterministic"
    fallback_reason: str = ""


class ChangeResult(BaseModel):
    """Full change analysis for one matched (or unmatched) clause pair."""

    match: ClauseMatch
    change_type: ChangeType = ChangeType.UNCERTAIN
    signals: list[ChangeSignal] = Field(default_factory=list)
    needs_review: bool = False
    review_reason: str = ""
    summary: Optional[BoundedSummary] = None
    citations: list[Citation] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Extraction models
# ---------------------------------------------------------------------------


class PageData(BaseModel):
    """Text and metadata extracted from a single PDF page."""

    page_number: int  # 1-based
    original_text: str
    normalized_text: str
    has_sufficient_text: bool = True
    ocr_warning: bool = False

    model_config = {"arbitrary_types_allowed": True}


class DocumentExtraction(BaseModel):
    """Full extraction result for one PDF document."""

    document_id: str
    document_name: str
    version: str = ""
    pages: list[PageData] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Top-level comparison result
# ---------------------------------------------------------------------------


class ComparisonResult(BaseModel):
    """End-to-end comparison output for two policy documents."""

    document_a: DocumentExtraction
    document_b: DocumentExtraction
    clauses_a: list[ExtractedClause] = Field(default_factory=list)
    clauses_b: list[ExtractedClause] = Field(default_factory=list)
    matches: list[ClauseMatch] = Field(default_factory=list)
    added_clauses: list[ExtractedClause] = Field(default_factory=list)
    removed_clauses: list[ExtractedClause] = Field(default_factory=list)
    changes: list[ChangeResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    processing_complete: bool = False

    model_config = {"arbitrary_types_allowed": True}
