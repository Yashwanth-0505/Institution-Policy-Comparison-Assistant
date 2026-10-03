"""
citations.py — Citation building and validation for the POLICYX ML pipeline.

Citations provide verifiable references back to the source clause in the
original document.  This module never fabricates citation data — every field
is derived directly from the :class:`~ml_pipeline.models.ExtractedClause`.
"""
from __future__ import annotations

import logging

from ml_pipeline.models import (
    ChangeResult,
    ChangeType,
    Citation,
    ExtractedClause,
    MatchType,
)

logger = logging.getLogger(__name__)


def build_citation(clause: ExtractedClause) -> Citation:
    """
    Build a :class:`~ml_pipeline.models.Citation` from an
    :class:`~ml_pipeline.models.ExtractedClause`.

    All fields are taken verbatim from the clause — nothing is inferred or
    fabricated.

    Parameters
    ----------
    clause:
        The source clause to cite.

    Returns
    -------
    Citation
        A citation whose ``exact_text`` is the clause's ``original_text``
        and whose ``citation_valid`` is initially *True*.
    """
    return Citation(
        document_id=clause.document_id,
        document_name=clause.document_name,
        clause_id=clause.clause_id,
        heading=clause.heading,
        page_number=clause.page_number,
        exact_text=clause.original_text,
        citation_valid=True,
        warning="",
    )


def validate_citation(
    citation: Citation,
    clauses: list[ExtractedClause],
) -> Citation:
    """
    Validate a :class:`~ml_pipeline.models.Citation` against the clause corpus.

    Checks performed (in order):
    1. ``clause_id`` must exist in *clauses*.
    2. ``page_number`` must match the clause's recorded page number.
    3. ``exact_text`` must match the clause's ``original_text``.

    Any failing check sets ``citation_valid=False`` and populates ``warning``
    with a human-readable description of the mismatch.

    Parameters
    ----------
    citation:
        The citation to validate.
    clauses:
        The full clause list for the document (used as ground truth).

    Returns
    -------
    Citation
        The same object with ``citation_valid`` and ``warning`` updated.
    """
    # Build lookup by clause_id for O(1) access.
    clause_lookup: dict[str, ExtractedClause] = {c.clause_id: c for c in clauses}

    if citation.clause_id not in clause_lookup:
        citation.citation_valid = False
        citation.warning = (
            f"clause_id '{citation.clause_id}' not found in the provided clause list."
        )
        logger.warning("Citation invalid: %s", citation.warning)
        return citation

    source_clause = clause_lookup[citation.clause_id]

    if citation.page_number != source_clause.page_number:
        citation.citation_valid = False
        citation.warning = (
            f"Page number mismatch: citation says page {citation.page_number} "
            f"but clause records page {source_clause.page_number}."
        )
        logger.warning("Citation invalid: %s", citation.warning)
        return citation

    if citation.exact_text != source_clause.original_text:
        citation.citation_valid = False
        citation.warning = (
            f"Text mismatch: citation text does not match the clause's original_text "
            f"(clause_id={citation.clause_id})."
        )
        logger.warning("Citation invalid: %s", citation.warning)
        return citation

    return citation


def attach_citations(
    change_results: list[ChangeResult],
    clauses_a: list[ExtractedClause],
    clauses_b: list[ExtractedClause],
) -> list[ChangeResult]:
    """
    Build and attach validated citations to every
    :class:`~ml_pipeline.models.ChangeResult`.

    Citation strategy by change type:

    - **UNCHANGED / WORDING_ONLY / SUBSTANTIVE / UNCERTAIN / NEEDS_REVIEW**:
      cite both ``clause_a`` (from doc A) and ``clause_b`` (from doc B).
    - **ADDED**: cite ``clause_b`` only (the new clause).
    - **REMOVED**: cite ``clause_a`` only (the removed clause).

    Citations are validated against the provided clause lists.  Invalid
    citations are kept but flagged with ``citation_valid=False``.

    Parameters
    ----------
    change_results:
        Output of :func:`~ml_pipeline.detector.detect_changes` (or after
        summary generation).
    clauses_a:
        All clauses from document A.
    clauses_b:
        All clauses from document B.

    Returns
    -------
    list[ChangeResult]
        The same list with ``citations`` populated on each result.
    """
    for result in change_results:
        citations: list[Citation] = []
        match = result.match
        change_type = result.change_type

        if change_type == ChangeType.ADDED:
            # Clause exists only in document B.
            # For ADDED, clause_b holds the new clause (stored in clause_a slot
            # for ADDED_CANDIDATE entries from HybridMatcher/BaselineMatcher).
            # We inspect match_type to determine which slot to use.
            if match.match_type == MatchType.ADDED_CANDIDATE and match.clause_b is None:
                # clause_a slot holds the added clause from doc B.
                cite = build_citation(match.clause_a)
                cite = validate_citation(cite, clauses_b)
            elif match.clause_b is not None:
                cite = build_citation(match.clause_b)
                cite = validate_citation(cite, clauses_b)
            else:
                cite = build_citation(match.clause_a)
                cite = validate_citation(cite, clauses_b)
            citations.append(cite)

        elif change_type == ChangeType.REMOVED:
            # Clause exists only in document A.
            cite = build_citation(match.clause_a)
            cite = validate_citation(cite, clauses_a)
            citations.append(cite)

        else:
            # Matched pair — cite both sides.
            cite_a = build_citation(match.clause_a)
            cite_a = validate_citation(cite_a, clauses_a)
            citations.append(cite_a)

            if match.clause_b is not None:
                cite_b = build_citation(match.clause_b)
                cite_b = validate_citation(cite_b, clauses_b)
                citations.append(cite_b)

        result.citations = citations

    return change_results
