"""
test_segmenter.py — Tests for ml_pipeline.segmenter
"""
from __future__ import annotations

import pytest

from ml_pipeline.segmenter import segment_clauses_from_text


def _seg(text: str, doc_id: str = "doc_a") -> list:
    return segment_clauses_from_text(text, doc_id, f"{doc_id}.pdf")


def test_numbered_sections() -> None:
    """Numbered headings (1. ) are detected as clause boundaries."""
    text = "1. Introduction\nThis is the intro.\n2. Scope\nThis defines scope."
    clauses = _seg(text)
    headings = [c.heading for c in clauses]
    assert any("1." in h for h in headings)
    assert any("2." in h for h in headings)


def test_decimal_sections() -> None:
    """Decimal headings (1.1 , 1.1.1 ) are detected."""
    text = "1.1 Policy Statement\nAll staff must comply.\n1.2 Applicability\nApplies to all."
    clauses = _seg(text)
    headings = [c.heading for c in clauses]
    assert any("1.1" in h for h in headings)
    assert any("1.2" in h for h in headings)


def test_section_keyword() -> None:
    """'Section N' headings are detected."""
    text = "Section 3 Data Protection\nAll data must be protected.\nSection 4 Security\nSecurity controls apply."
    clauses = _seg(text)
    headings = [c.heading for c in clauses]
    assert any("Section 3" in h or "Section" in h for h in headings)


def test_clause_keyword() -> None:
    """'Clause N' headings are detected."""
    text = "Clause 2.1 Responsibilities\nManagers are responsible.\nClause 2.2 Reporting\nAll incidents must be reported."
    clauses = _seg(text)
    headings = [c.heading for c in clauses]
    assert any("Clause" in h for h in headings)


def test_allcaps_heading() -> None:
    """ALL CAPS headings (>=5 chars) are detected."""
    text = "INTRODUCTION\nThis introduces the policy.\nSCOPE\nThis applies to all staff."
    clauses = _seg(text)
    headings = [c.heading for c in clauses]
    assert any(h.isupper() and len(h) >= 5 for h in headings)


def test_no_headings_flagged() -> None:
    """Text with no headings produces a preamble clause flagged for review."""
    text = "This is just plain text without any headings. It continues for a while."
    clauses = _seg(text)
    assert len(clauses) >= 1
    assert all(c.flagged_for_review for c in clauses)
    assert all("preamble" in c.review_reason for c in clauses)


def test_short_clause_not_discarded() -> None:
    """Short clauses (below min_clause_length) are flagged but NOT discarded."""
    text = "1. Introduction\nOK.\n2. Scope\nThis is a longer scope section with more content."
    clauses = _seg(text)
    # The short "OK." clause must still be present
    assert len(clauses) >= 2
    short_clauses = [c for c in clauses if len(c.normalized_text) < 20]
    for c in short_clauses:
        assert c.flagged_for_review is True
        assert "short" in c.review_reason


def test_clause_ids_unique() -> None:
    """All clause_ids within one document are unique."""
    text = "\n".join([
        "1. Introduction\nIntro text here for testing purposes.",
        "2. Scope\nScope text here for testing purposes.",
        "3. Policy\nPolicy text here for testing purposes.",
        "4. Enforcement\nEnforcement text here for testing purposes.",
    ])
    clauses = _seg(text)
    ids = [c.clause_id for c in clauses]
    assert len(ids) == len(set(ids)), "Clause IDs must be unique"
