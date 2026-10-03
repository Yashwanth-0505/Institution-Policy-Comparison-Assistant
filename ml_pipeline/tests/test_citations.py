"""
test_citations.py — Tests for ml_pipeline.citations
"""
from __future__ import annotations

import pytest

from ml_pipeline.citations import attach_citations, build_citation, validate_citation
from ml_pipeline.models import (
    ChangeResult,
    ChangeType,
    ClauseMatch,
    MatchType,
)
from ml_pipeline.tests.conftest import make_clause


def _result(ct: ChangeType, mt: MatchType, text_a: str = "Old text here.", text_b: str = "New text here.") -> ChangeResult:
    ca = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", text_a, page=2)
    cb = make_clause("doc_b", "doc_b_clause_0000", "1. Policy", text_b, page=3)

    if mt in (MatchType.REMOVED_CANDIDATE,):
        match = ClauseMatch(clause_a=ca, clause_b=None, match_type=mt)
    elif mt in (MatchType.ADDED_CANDIDATE,):
        match = ClauseMatch(clause_a=ca, clause_b=None, match_type=mt)
    else:
        match = ClauseMatch(clause_a=ca, clause_b=cb, match_type=mt)

    return ChangeResult(match=match, change_type=ct)


# ---------------------------------------------------------------------------

def test_build_citation_fields_match_clause() -> None:
    """build_citation copies all fields from the clause verbatim."""
    clause = make_clause("doc_a", "doc_a_clause_0001", "2. Scope", "Applies to all staff.", page=5)
    citation = build_citation(clause)
    assert citation.document_id == "doc_a"
    assert citation.clause_id == "doc_a_clause_0001"
    assert citation.page_number == 5
    assert citation.exact_text == "Applies to all staff."
    assert citation.citation_valid is True


def test_validate_valid_passes() -> None:
    """validate_citation returns citation_valid=True for a correct citation."""
    clause = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", "Staff must comply.", page=1)
    citation = build_citation(clause)
    validated = validate_citation(citation, [clause])
    assert validated.citation_valid is True
    assert validated.warning == ""


def test_validate_missing_clause_invalid() -> None:
    """citation_valid=False when clause_id not found in corpus."""
    clause = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", "Staff must comply.", page=1)
    citation = build_citation(clause)
    # Pass an empty list — clause_id won't be found
    validated = validate_citation(citation, [])
    assert validated.citation_valid is False
    assert "not found" in validated.warning


def test_validate_page_mismatch() -> None:
    """citation_valid=False when page_number doesn't match the clause."""
    clause = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", "Staff must comply.", page=1)
    citation = build_citation(clause)
    citation.page_number = 99  # deliberately wrong
    validated = validate_citation(citation, [clause])
    assert validated.citation_valid is False
    assert "Page number" in validated.warning or "page" in validated.warning.lower()


def test_validate_text_mismatch() -> None:
    """citation_valid=False when exact_text doesn't match original_text."""
    clause = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", "Staff must comply.", page=1)
    citation = build_citation(clause)
    citation.exact_text = "This is different text entirely."
    validated = validate_citation(citation, [clause])
    assert validated.citation_valid is False
    assert "mismatch" in validated.warning.lower() or "text" in validated.warning.lower()


def test_matched_pair_two_citations() -> None:
    """Matched pair (UNCHANGED/WORDING_ONLY/SUBSTANTIVE) gets citations from both docs."""
    ca = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", "Old policy text.", page=1)
    cb = make_clause("doc_b", "doc_b_clause_0000", "1. Policy", "New policy text.", page=2)
    match = ClauseMatch(clause_a=ca, clause_b=cb, match_type=MatchType.HYBRID)
    result = ChangeResult(match=match, change_type=ChangeType.SUBSTANTIVE)
    results = attach_citations([result], [ca], [cb])
    assert len(results[0].citations) == 2
    doc_ids = {c.document_id for c in results[0].citations}
    assert "doc_a" in doc_ids
    assert "doc_b" in doc_ids


def test_added_cites_doc_b_only() -> None:
    """ADDED clause is cited from doc_b only."""
    cb = make_clause("doc_b", "doc_b_clause_0000", "2. New Section", "Newly added clause text.", page=3)
    match = ClauseMatch(clause_a=cb, clause_b=None, match_type=MatchType.ADDED_CANDIDATE)
    result = ChangeResult(match=match, change_type=ChangeType.ADDED)
    results = attach_citations([result], [], [cb])
    assert len(results[0].citations) == 1
    assert results[0].citations[0].document_id == "doc_b"


def test_removed_cites_doc_a_only() -> None:
    """REMOVED clause is cited from doc_a only."""
    ca = make_clause("doc_a", "doc_a_clause_0000", "3. Old Section", "Removed clause text.", page=2)
    match = ClauseMatch(clause_a=ca, clause_b=None, match_type=MatchType.REMOVED_CANDIDATE)
    result = ChangeResult(match=match, change_type=ChangeType.REMOVED)
    results = attach_citations([result], [ca], [])
    assert len(results[0].citations) == 1
    assert results[0].citations[0].document_id == "doc_a"
