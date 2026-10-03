"""
segmenter.py — Clause segmentation for the POLICYX ML pipeline.

Splits institutional policy document text (attendance, examination, lecturer,
disciplinary, placement, financial/admin, hostel, library) into discrete
clauses using heading-pattern detection.
Short clauses are flagged but never discarded.
"""
from __future__ import annotations

import re
from typing import Optional

from ml_pipeline.config import PipelineConfig, load_config
from ml_pipeline.models import DocumentExtraction, ExtractedClause, SectionMetadata

# ---------------------------------------------------------------------------
# Heading patterns — compiled once at module load time for performance.
# Covers standard institutional policy document formats used by universities,
# colleges, and other educational/administrative bodies.
# ---------------------------------------------------------------------------
_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # Numeric hierarchies: 1.1.1 / 1.1 / 1.
    re.compile(r"^\d+\.\d+\.\d+\s"),
    re.compile(r"^\d+\.\d+\s"),
    re.compile(r"^\d+\.\s"),
    # Labelled sections: Section 3, Clause 4.2, Article IV, Rule 5
    re.compile(r"^Section\s+\d+", re.IGNORECASE),
    re.compile(r"^Clause\s+[\d.]+", re.IGNORECASE),
    re.compile(r"^ARTICLE\s+[IVX\d]+", re.IGNORECASE),
    re.compile(r"^Rule\s+\d+", re.IGNORECASE),
    # Institutional policy domain headings
    re.compile(r"^Chapter\s+\d+", re.IGNORECASE),
    re.compile(r"^Part\s+[IVX\d]+", re.IGNORECASE),
    re.compile(r"^Schedule\s+\d+", re.IGNORECASE),
    re.compile(r"^Annexure\s+[A-Z\d]+", re.IGNORECASE),
    re.compile(r"^Regulation\s+\d+", re.IGNORECASE),
    re.compile(r"^Policy\s+\d+", re.IGNORECASE),
    # ALL CAPS headings (e.g. ATTENDANCE POLICY, HOSTEL RULES)
    re.compile(r"^[A-Z][A-Z\s]{4,}$"),
]


def _is_heading_line(line: str) -> bool:
    """Return *True* if *line* matches any known heading pattern."""
    line = line.strip()
    if not line:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


def _extract_section_metadata(heading: str) -> SectionMetadata:
    """
    Derive :class:`SectionMetadata` from a heading string.

    Extracts the leading numbering (if any) as *section_number* and the
    remainder as *section_title*.
    """
    heading = heading.strip()
    # Match leading numbering like "1.", "1.1", "1.1.1", "Section 3", etc.
    number_match = re.match(r"^([\d.]+\.?|Section\s+\d+|Clause\s+[\d.]+|ARTICLE\s+[IVX\d]+)\s*", heading, re.IGNORECASE)
    if number_match:
        section_number = number_match.group(1).strip()
        section_title = heading[number_match.end():].strip()
        # Estimate depth from dot count
        depth = section_number.count(".")
        return SectionMetadata(
            section_number=section_number,
            section_title=section_title or None,
            depth=depth,
        )
    # ALL CAPS headings have no number
    return SectionMetadata(section_number=None, section_title=heading, depth=0)


def segment_clauses_from_text(
    text: str,
    document_id: str,
    document_name: str,
    version: str = "",
    page_number: int = 1,
    config: Optional[PipelineConfig] = None,
) -> list[ExtractedClause]:
    """
    Segment *text* into a list of :class:`ExtractedClause` objects.

    Parameters
    ----------
    text:
        The normalised page/document text to segment.
    document_id:
        Stable document identifier used to build clause IDs.
    document_name:
        Human-readable document name.
    version:
        Optional version string.
    page_number:
        Page number to record on each clause (1-based).
    config:
        Pipeline configuration.  If *None*, defaults are used.

    Returns
    -------
    list[ExtractedClause]
        One entry per detected clause.  Short clauses are flagged but
        **not** discarded.
    """
    if config is None:
        config = load_config()

    lines = text.splitlines()
    if not lines:
        return []

    # -----------------------------------------------------------------------
    # First pass: identify heading positions
    # -----------------------------------------------------------------------
    # Each element is (line_index, heading_text).
    heading_positions: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        if _is_heading_line(line):
            heading_positions.append((idx, line.strip()))

    # -----------------------------------------------------------------------
    # Build raw blocks: (heading, body_lines)
    # -----------------------------------------------------------------------
    blocks: list[tuple[str, list[str], bool]] = []  # (heading, body, is_preamble)

    if not heading_positions:
        # No headings found — treat whole text as a single preamble clause.
        blocks.append(("", lines, True))
    else:
        # Preamble: lines before the first heading.
        first_heading_idx = heading_positions[0][0]
        preamble_lines = lines[:first_heading_idx]
        if preamble_lines and any(l.strip() for l in preamble_lines):
            blocks.append(("", preamble_lines, True))

        # Sections between headings.
        for i, (h_idx, h_text) in enumerate(heading_positions):
            if i + 1 < len(heading_positions):
                next_h_idx = heading_positions[i + 1][0]
                body_lines = lines[h_idx + 1 : next_h_idx]
            else:
                body_lines = lines[h_idx + 1 :]
            blocks.append((h_text, body_lines, False))

    # -----------------------------------------------------------------------
    # Convert blocks to ExtractedClause objects
    # -----------------------------------------------------------------------
    clauses: list[ExtractedClause] = []
    clause_counter = 0

    for heading, body_lines, is_preamble in blocks:
        body_text = "\n".join(body_lines).strip()
        # Include the heading itself as part of the clause text when present.
        if heading:
            full_text = f"{heading}\n{body_text}".strip()
        else:
            full_text = body_text

        if not full_text:
            continue

        clause_id = f"{document_id}_clause_{clause_counter:04d}"
        clause_counter += 1

        is_short = len(full_text) < config.min_clause_length
        flagged = is_preamble or is_short
        review_reasons: list[str] = []
        if is_preamble:
            review_reasons.append("preamble (before first heading)")
        if is_short:
            review_reasons.append(
                f"short clause ({len(full_text)} < {config.min_clause_length} chars)"
            )

        section_meta = _extract_section_metadata(heading) if heading else SectionMetadata()

        clauses.append(
            ExtractedClause(
                document_id=document_id,
                document_name=document_name,
                version=version,
                clause_id=clause_id,
                heading=heading,
                original_text=full_text,
                normalized_text=full_text,
                page_number=page_number,
                section_metadata=section_meta,
                flagged_for_review=flagged,
                review_reason="; ".join(review_reasons),
            )
        )

    return clauses


def segment_clauses(
    doc_extraction: DocumentExtraction,
    config: Optional[PipelineConfig] = None,
) -> list[ExtractedClause]:
    """
    Segment all pages of *doc_extraction* into clauses.

    Iterates over every :class:`PageData` in *doc_extraction* and calls
    :func:`segment_clauses_from_text` for each page's normalised text.
    Clause IDs are unique across the entire document.

    Parameters
    ----------
    doc_extraction:
        The result of :func:`~ml_pipeline.extractor.extract_document`.
    config:
        Pipeline configuration.  If *None*, defaults are used.

    Returns
    -------
    list[ExtractedClause]
        All clauses from all pages, with globally unique ``clause_id``
        values within this document.
    """
    if config is None:
        config = load_config()

    all_clauses: list[ExtractedClause] = []
    global_counter = 0

    for page in doc_extraction.pages:
        if not page.normalized_text.strip():
            continue

        page_clauses = segment_clauses_from_text(
            text=page.normalized_text,
            document_id=doc_extraction.document_id,
            document_name=doc_extraction.document_name,
            version=doc_extraction.version,
            page_number=page.page_number,
            config=config,
        )

        # Re-number clause IDs to be globally unique across pages.
        for clause in page_clauses:
            clause.clause_id = f"{doc_extraction.document_id}_clause_{global_counter:04d}"
            global_counter += 1

        all_clauses.extend(page_clauses)

    return all_clauses
