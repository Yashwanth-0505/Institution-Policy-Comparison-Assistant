"""
conftest.py — Shared fixtures for the POLICYX ML pipeline test suite.

All fixtures are synthetic — no internet access or real PDF files required.
Session-scoped model preloading for faster test execution.
"""
from __future__ import annotations

import io
import logging

import fitz  # PyMuPDF
import pytest

from ml_pipeline.models import (
    ClauseMatch,
    ExtractedClause,
    MatchType,
    SectionMetadata,
)
from ml_pipeline.features import FeatureExtractor
from ml_pipeline.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clause(
    document_id: str = "doc_a",
    clause_id: str = "doc_a_clause_0000",
    heading: str = "1. Introduction",
    text: str = "All employees must complete mandatory training within 30 days.",
    page: int = 1,
) -> ExtractedClause:
    """
    Create a minimal :class:`~ml_pipeline.models.ExtractedClause` for testing.

    Parameters
    ----------
    document_id:   Stable document identifier.
    clause_id:     Unique clause identifier.
    heading:       Section heading string.
    text:          Clause body text (used for both original and normalised).
    page:          1-based page number.

    Returns
    -------
    ExtractedClause
    """
    return ExtractedClause(
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        version="v1",
        clause_id=clause_id,
        heading=heading,
        original_text=text,
        normalized_text=text,
        page_number=page,
        section_metadata=SectionMetadata(section_number="1", section_title="Introduction", depth=0),
        flagged_for_review=False,
        review_reason="",
    )


def make_pdf_bytes(text: str = "Hello, world!\nThis is a test PDF page.") -> bytes:
    """
    Create a minimal valid PDF in memory using fitz (PyMuPDF).

    No file I/O is performed — the PDF is written entirely to a bytes buffer.

    Parameters
    ----------
    text:
        Text content to embed on the single page.

    Returns
    -------
    bytes
        Raw PDF bytes that can be written to a temporary file or passed to
        ``fitz.open(stream=..., filetype='pdf')``.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 dimensions
    page.insert_text(
        (72, 100),
        text,
        fontsize=11,
        fontname="helv",
    )
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def preload_embedding_model():
    """
    Preload the embedding model once per test session to avoid repeated
    expensive downloads and initialization during individual tests.
    
    This fixture is session-scoped and autouse=True, so it runs automatically
    before any test in the session.
    """
    logger.info("Preloading embedding model for test session...")
    config = load_config()
    try:
        extractor = FeatureExtractor(config=config)
        # Trigger model load by calling the getter
        if extractor._embedding_model is not None:
            logger.info("Embedding model loaded successfully")
        else:
            logger.warning("Embedding model is None (fallback to TF-IDF)")
    except Exception as e:
        logger.warning(f"Failed to preload model: {e}")
    yield
    logger.info("Test session complete")


@pytest.fixture
def sample_clause_a() -> ExtractedClause:
    return make_clause(
        document_id="doc_a",
        clause_id="doc_a_clause_0000",
        heading="1. Data Protection",
        text="All staff must complete mandatory data protection training within 30 days of joining.",
        page=1,
    )


@pytest.fixture
def sample_clause_b() -> ExtractedClause:
    return make_clause(
        document_id="doc_b",
        clause_id="doc_b_clause_0000",
        heading="1. Data Protection",
        text="All employees must complete mandatory data protection training within 30 days of joining.",
        page=1,
    )


@pytest.fixture
def sample_match(sample_clause_a: ExtractedClause, sample_clause_b: ExtractedClause) -> ClauseMatch:
    return ClauseMatch(
        clause_a=sample_clause_a,
        clause_b=sample_clause_b,
        match_score=0.92,
        number_score=1.0,
        heading_score=1.0,
        embedding_score=0.85,
        tfidf_score=0.88,
        match_type=MatchType.HYBRID,
        needs_review=False,
        review_reason="",
    )
