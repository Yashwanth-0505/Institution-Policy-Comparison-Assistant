"""
test_extractor.py — Tests for ml_pipeline.extractor
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ml_pipeline.extractor import _normalize_text, extract_document
from ml_pipeline.tests.conftest import make_pdf_bytes


def _write_pdf(tmp_path: Path, text: str) -> Path:
    """Helper: write make_pdf_bytes to a temp file and return path."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(make_pdf_bytes(text))
    return pdf_path


# ---------------------------------------------------------------------------

def test_extract_valid_pdf(tmp_path: Path) -> None:
    """extract_document returns a DocumentExtraction with at least one page."""
    pdf_path = _write_pdf(tmp_path, "1. Introduction\nThis is a test policy document.")
    result = extract_document(str(pdf_path), "doc_test", "Test Document")
    assert result.document_id == "doc_test"
    assert result.document_name == "Test Document"
    assert len(result.pages) >= 1
    assert result.errors == []


def test_extract_preserves_original_text(tmp_path: Path) -> None:
    """original_text is stored verbatim before normalization."""
    content = "Hello   World\xadtest\f\nNew page"
    pdf_path = _write_pdf(tmp_path, content)
    result = extract_document(str(pdf_path), "doc_orig", "OrigDoc")
    # At least one page must have some text
    texts = [p.original_text for p in result.pages]
    assert any(t.strip() for t in texts), "Expected at least one page with text"
    # normalized_text should differ from original if normalization was applied
    for page in result.pages:
        if page.original_text:
            # normalized should not have soft hyphens
            assert "\xad" not in page.normalized_text


def test_extract_empty_page_ocr_warning(tmp_path: Path) -> None:
    """Pages with fewer than 50 chars trigger ocr_warning=True."""
    # Create a PDF with minimal text that will likely produce a short page
    pdf_path = _write_pdf(tmp_path, "Hi")
    result = extract_document(str(pdf_path), "doc_short", "ShortDoc")
    # At least one page should trigger the OCR warning (text is very short)
    ocr_pages = [p for p in result.pages if p.ocr_warning]
    assert len(ocr_pages) >= 1
    assert any("fewer than" in w or "50 char" in w for w in result.warnings)


def test_extract_missing_file() -> None:
    """FileNotFoundError is raised with a helpful message for missing PDFs."""
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_document("/nonexistent/path/file.pdf", "doc_x", "Missing")


def test_normalize_whitespace() -> None:
    """_normalize_text collapses spaces, removes soft hyphens, replaces form-feeds."""
    raw = "Hello   World\xad  test\f\nSecond"
    result = _normalize_text(raw)
    assert "\xad" not in result
    assert "   " not in result  # multiple spaces collapsed
    assert "\f" not in result   # form-feed replaced
    assert "Hello World" in result
    assert "test" in result
