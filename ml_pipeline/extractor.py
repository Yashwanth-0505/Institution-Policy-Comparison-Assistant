"""
extractor.py — PDF text extraction for the POLICYX ML pipeline.
Uses PyMuPDF (fitz) to extract per-page text with normalisation.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from ml_pipeline.models import DocumentExtraction, PageData

# Minimum character count for a page to be considered non-empty.
_OCR_WARNING_THRESHOLD = 50

# Soft-hyphen character (U+00AD) used in some PDFs.
_SOFT_HYPHEN = "\xad"


def _normalize_text(text: str) -> str:
    """
    Apply lightweight normalisation to extracted page text.

    Rules (applied in order):
    1. Remove soft hyphens (\\xad).
    2. Replace form-feed characters with newlines.
    3. Collapse runs of more than two spaces into a single space.
    4. Strip leading/trailing whitespace from every line.
    5. Strip leading/trailing whitespace from the whole string.
    """
    text = text.replace(_SOFT_HYPHEN, "")
    text = text.replace("\f", "\n")
    # Collapse multiple spaces (but not newlines) to one space.
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip trailing spaces on each line.
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def extract_document(
    pdf_path: str | Path,
    document_id: str,
    document_name: str,
    version: str = "",
) -> DocumentExtraction:
    """
    Extract text from a PDF file and return a :class:`DocumentExtraction`.

    Parameters
    ----------
    pdf_path:
        Absolute or relative path to the PDF file.
    document_id:
        Stable identifier for this document (used in clause IDs).
    document_name:
        Human-readable document name.
    version:
        Optional version string (e.g. ``"2024-01"``).

    Returns
    -------
    DocumentExtraction
        Per-page text with original and normalised variants, plus warnings
        and errors collected during extraction.

    Raises
    ------
    FileNotFoundError
        When *pdf_path* does not exist on the filesystem.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path!r}. "
            "Please verify the path and try again."
        )

    pages: list[PageData] = []
    warnings: list[str] = []
    errors: list[str] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Failed to open PDF '{pdf_path}': {exc}")
        return DocumentExtraction(
            document_id=document_id,
            document_name=document_name,
            version=version,
            pages=[],
            warnings=warnings,
            errors=errors,
        )

    try:
        for i, page in enumerate(doc):
            page_number = i + 1  # 1-based; never fabricated
            try:
                original_text: str = page.get_text("text")  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Page {page_number}: failed to extract text — {exc}"
                errors.append(error_msg)
                # Still create a page entry so callers know it existed.
                pages.append(
                    PageData(
                        page_number=page_number,
                        original_text="",
                        normalized_text="",
                        has_sufficient_text=False,
                        ocr_warning=True,
                    )
                )
                continue

            normalized_text = _normalize_text(original_text)

            has_sufficient = len(normalized_text.strip()) >= _OCR_WARNING_THRESHOLD
            ocr_warning = not has_sufficient

            if ocr_warning:
                warnings.append(
                    f"Page {page_number}: fewer than {_OCR_WARNING_THRESHOLD} "
                    "characters extracted — possible scanned/image-only page."
                )

            pages.append(
                PageData(
                    page_number=page_number,
                    original_text=original_text,
                    normalized_text=normalized_text,
                    has_sufficient_text=has_sufficient,
                    ocr_warning=ocr_warning,
                )
            )
    finally:
        doc.close()

    return DocumentExtraction(
        document_id=document_id,
        document_name=document_name,
        version=version,
        pages=pages,
        warnings=warnings,
        errors=errors,
    )
