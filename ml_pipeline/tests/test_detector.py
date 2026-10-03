"""
test_detector.py — Tests for ml_pipeline.detector
"""
from __future__ import annotations

import pytest

from ml_pipeline.detector import detect_changes, _extract_signals
from ml_pipeline.models import ChangeType, ClauseMatch, MatchType
from ml_pipeline.tests.conftest import make_clause


def _make_match(
    text_a: str,
    text_b: str,
    match_type: MatchType = MatchType.HYBRID,
    heading_a: str = "1. Policy",
    heading_b: str = "1. Policy",
) -> ClauseMatch:
    ca = make_clause("doc_a", "doc_a_clause_0000", heading_a, text_a)
    cb = make_clause("doc_b", "doc_b_clause_0000", heading_b, text_b)
    return ClauseMatch(
        clause_a=ca,
        clause_b=cb,
        match_score=0.9,
        match_type=match_type,
    )


def _make_unilateral(text: str, match_type: MatchType) -> ClauseMatch:
    """Make an ADDED or REMOVED match (no clause_b)."""
    ca = make_clause("doc_a", "doc_a_clause_0000", "1. Policy", text)
    return ClauseMatch(
        clause_a=ca,
        clause_b=None,
        match_score=0.0,
        match_type=match_type,
    )


# ---------------------------------------------------------------------------

def test_identical_unchanged() -> None:
    """Identical texts produce UNCHANGED."""
    text = "All staff must complete mandatory cybersecurity training within 30 days of joining."
    match = _make_match(text, text)
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.UNCHANGED


def test_minor_wording_only() -> None:
    """Minor paraphrase with no modal/numeric/negation signals is WORDING_ONLY."""
    # Use text where the same modal word appears in both sides so no modal_change fires.
    match = _make_match(
        "All students must submit their assignment before the deadline.",
        "All students must hand in their assignment before the deadline.",
    )
    results = detect_changes([match])
    assert results[0].change_type in (ChangeType.WORDING_ONLY, ChangeType.UNCHANGED)


def test_numeric_change_substantive() -> None:
    """Changed number triggers SUBSTANTIVE classification."""
    match = _make_match(
        "Invoices must be paid within 30 days of the invoice date.",
        "Invoices must be paid within 60 days of the invoice date.",
    )
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.SUBSTANTIVE


def test_negation_added_substantive() -> None:
    """Adding a negation word produces SUBSTANTIVE."""
    match = _make_match(
        "Staff may access the server room when accompanied by an authorised member.",
        "Staff may not access the server room under any circumstances.",
    )
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.SUBSTANTIVE


def test_modal_shift_substantive() -> None:
    """Changing 'may' to 'must' is SUBSTANTIVE."""
    match = _make_match(
        "Employees may submit a grievance within 10 working days.",
        "Employees must submit a grievance within 10 working days.",
    )
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.SUBSTANTIVE


def test_date_change_substantive() -> None:
    """Changing a date produces SUBSTANTIVE."""
    match = _make_match(
        "Annual statements must be submitted by 31 March 2024.",
        "Annual statements must be submitted by 30 June 2025.",
    )
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.SUBSTANTIVE


def test_added_type() -> None:
    """ADDED_CANDIDATE match produces ADDED ChangeType."""
    match = _make_unilateral(
        "All remote workers must use a VPN when accessing internal systems.",
        MatchType.ADDED_CANDIDATE,
    )
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.ADDED


def test_removed_type() -> None:
    """REMOVED_CANDIDATE match produces REMOVED ChangeType."""
    match = _make_unilateral(
        "Employees may request flexible working after 26 weeks of service.",
        MatchType.REMOVED_CANDIDATE,
    )
    results = detect_changes([match])
    assert results[0].change_type == ChangeType.REMOVED


def test_uncertain_ambiguous() -> None:
    """Very low similarity ratio with no clear signals -> UNCERTAIN."""
    match = _make_match(
        "The committee will consider applications on a case-by-case basis.",
        "Applications will be assessed according to the published scoring rubric approved by the Board.",
    )
    results = detect_changes([match])
    # Low similarity -> UNCERTAIN or WORDING_ONLY depending on ratio
    assert results[0].change_type in (ChangeType.UNCERTAIN, ChangeType.WORDING_ONLY)


def test_signals_populated() -> None:
    """Signal list is populated when substantive changes are detected."""
    match = _make_match(
        "Staff are entitled to 20 days of annual leave per year.",
        "Staff are entitled to 25 days of annual leave per year.",
    )
    results = detect_changes([match])
    assert len(results[0].signals) >= 1
    signal_types = [s.signal_type for s in results[0].signals]
    assert "numeric_change" in signal_types
