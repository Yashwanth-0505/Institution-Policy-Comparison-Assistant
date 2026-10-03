"""
detector.py — Change detection for the POLICYX ML pipeline.
"""
from __future__ import annotations

import difflib
import logging
import re

from ml_pipeline.models import (
    ChangeResult,
    ChangeSignal,
    ChangeType,
    ClauseMatch,
    MatchType,
)

logger = logging.getLogger(__name__)

_NUMERIC_RE  = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_DATE_RE     = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{4})\b", re.IGNORECASE)
_MONETARY_RE = re.compile(r"(?:Rs\.?\s*[\d,]+|\$[\d,]+(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s*(?:USD|GBP|EUR)\b)")

_NEGATION_WORDS: frozenset[str] = frozenset({"not", "no", "never", "neither", "nor", "except", "unless"})
_MODAL_WORDS:    frozenset[str] = frozenset({"shall", "must", "may", "should", "will", "can", "cannot"})
_CONDITION_PHRASES: list[str]   = ["if ", "unless ", "provided that", "subject to", "except where"]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def _extract_signals(old_text: str, new_text: str) -> tuple[list[ChangeSignal], float]:
    ratio   = difflib.SequenceMatcher(None, old_text, new_text).ratio()
    signals: list[ChangeSignal] = []

    # numeric
    # Strip leading section numbers (e.g., "3.1 ", "2. ", "Section 5:") before comparing
    # This prevents renumbering from being flagged as a substantive change
    def _strip_section_number(text: str) -> str:
        """Remove leading section number/label (e.g., '3.1', '2.', 'Section 5:')."""
        t = text.lstrip()
        # Match patterns like "3.1 ", "2. ", "Section 5: ", etc.
        t = re.sub(r"^(?:Section\s+)?[\d.]+(?:\s*[:.]?\s+)?", "", t)
        return t

    old_nums = _NUMERIC_RE.findall(_strip_section_number(old_text))
    new_nums = _NUMERIC_RE.findall(_strip_section_number(new_text))
    if set(old_nums) != set(new_nums):
        signals.append(ChangeSignal(
            signal_type="numeric_change",
            description="Numeric values differ between versions",
            old_value=", ".join(old_nums) or "(none)",
            new_value=", ".join(new_nums) or "(none)",
        ))

    # date
    old_dates = _DATE_RE.findall(old_text)
    new_dates = _DATE_RE.findall(new_text)
    if set(old_dates) != set(new_dates):
        signals.append(ChangeSignal(
            signal_type="date_change",
            description="Date references differ between versions",
            old_value=", ".join(old_dates) or "(none)",
            new_value=", ".join(new_dates) or "(none)",
        ))

    # monetary
    old_money = _MONETARY_RE.findall(old_text)
    new_money = _MONETARY_RE.findall(new_text)
    if set(old_money) != set(new_money):
        signals.append(ChangeSignal(
            signal_type="monetary_change",
            description="Monetary values differ between versions",
            old_value=", ".join(old_money) or "(none)",
            new_value=", ".join(new_money) or "(none)",
        ))

    # negation
    old_tok  = _tokenize(old_text)
    new_tok  = _tokenize(new_text)
    old_negs = _NEGATION_WORDS & old_tok
    new_negs = _NEGATION_WORDS & new_tok
    if old_negs != new_negs:
        added   = new_negs - old_negs
        removed = old_negs - new_negs
        signals.append(ChangeSignal(
            signal_type="negation_change",
            description="Negation words added or removed",
            old_value=", ".join(sorted(removed)) or "(none)",
            new_value=", ".join(sorted(added))   or "(none)",
        ))

    # modal — only fire when actual modal words are added or removed
    old_modals = _MODAL_WORDS & old_tok
    new_modals = _MODAL_WORDS & new_tok
    added_m    = new_modals - old_modals
    removed_m  = old_modals - new_modals
    if added_m or removed_m:          # strict: must have actual changes
        signals.append(ChangeSignal(
            signal_type="modal_change",
            description="Modal/obligation words changed",
            old_value=", ".join(sorted(removed_m)) or "(none)",
            new_value=", ".join(sorted(added_m))   or "(none)",
        ))

    # condition
    old_conds = [p for p in _CONDITION_PHRASES if p in old_text.lower()]
    new_conds = [p for p in _CONDITION_PHRASES if p in new_text.lower()]
    if set(old_conds) != set(new_conds):
        signals.append(ChangeSignal(
            signal_type="condition_change",
            description="Conditional phrases added or removed",
            old_value=", ".join(old_conds) or "(none)",
            new_value=", ".join(new_conds) or "(none)",
        ))

    return signals, ratio


def _classify(signals: list[ChangeSignal], ratio: float, match: ClauseMatch) -> tuple[ChangeType, bool, str]:
    substantive_types = {"numeric_change","date_change","monetary_change","negation_change","modal_change"}
    has_substantive   = any(s.signal_type in substantive_types for s in signals)

    # Substantive signals win FIRST — even if ratio is very high
    if has_substantive:
        return ChangeType.SUBSTANTIVE, False, ""

    if ratio >= 0.98:
        return ChangeType.UNCHANGED, False, ""

    if ratio >= 0.85:
        return ChangeType.WORDING_ONLY, False, ""

    if ratio < 0.60:
        return (ChangeType.UNCERTAIN, True,
                f"Low similarity ({ratio:.2f}) with ambiguous signals — manual review required.")

    return ChangeType.WORDING_ONLY, False, ""


def detect_changes(matches: list[ClauseMatch]) -> list[ChangeResult]:
    results: list[ChangeResult] = []

    for match in matches:
        if match.match_type == MatchType.ADDED_CANDIDATE:
            results.append(ChangeResult(match=match, change_type=ChangeType.ADDED))
            continue

        if match.match_type == MatchType.REMOVED_CANDIDATE:
            results.append(ChangeResult(match=match, change_type=ChangeType.REMOVED))
            continue

        if match.clause_b is None:
            results.append(ChangeResult(match=match, change_type=ChangeType.UNCERTAIN,
                                        needs_review=True, review_reason="No clause_b."))
            continue

        old_text = match.clause_a.normalized_text
        new_text = match.clause_b.normalized_text

        try:
            signals, ratio = _extract_signals(old_text, new_text)
        except Exception as exc:
            logger.warning("Signal extraction failed: %s", exc)
            signals, ratio = [], 0.5

        change_type, needs_review, review_reason = _classify(signals, ratio, match)

        if match.needs_review:
            needs_review  = True
            review_reason = (review_reason + " " + match.review_reason).strip()

        results.append(ChangeResult(
            match=match,
            change_type=change_type,
            signals=signals,
            needs_review=needs_review,
            review_reason=review_reason,
        ))

    return results
