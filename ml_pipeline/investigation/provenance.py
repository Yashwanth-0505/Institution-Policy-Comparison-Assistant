"""
provenance.py — Source provenance verification for POLICYX investigation mode.

Evaluates each candidate document against a set of deterministic evidence
signals.  NEVER claims AUTHENTIC — only assigns a provenance tier based on
the weight of evidence.

Provenance tiers (best → worst):
  VERIFIED_PROVENANCE  — official domain + title match + year match + authority match
  STRONG_PROVENANCE    — official domain + title match + (year OR authority match)
  LIMITED_PROVENANCE   — some signals match but not enough for STRONG
  UNVERIFIED           — no meaningful signals
"""
from __future__ import annotations

import re
import urllib.parse
from difflib import SequenceMatcher


# ── Domain authority helpers ──────────────────────────────────────────────────

_EDU_GOV_RE = re.compile(r"\.(edu|ac\.[a-z]{2}|gov|nic\.in|gov\.in)(/|$)", re.I)
_THIRD_PARTY = re.compile(
    r"(scribd|slideshare|academia\.edu|docplayer|studocu|coursehero|"
    r"quora|reddit|facebook|twitter|instagram|youtube)", re.I
)


def _is_official_domain(url: str) -> bool:
    return bool(_EDU_GOV_RE.search(url)) and not bool(_THIRD_PARTY.search(url))


def _is_third_party(url: str) -> bool:
    return bool(_THIRD_PARTY.search(url))


# ── Text similarity ───────────────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _contains_words(text: str, words: list[str]) -> bool:
    tl = text.lower()
    return any(w.lower() in tl for w in words if w)


# ── Year helpers ──────────────────────────────────────────────────────────────

def _extract_years(text: str) -> set[str]:
    return set(re.findall(r"20\d{2}", text))


def _is_previous_year(candidate_text: str, newer_year: str) -> bool:
    """Return True if candidate text mentions a year before the newer document year."""
    try:
        ny = int(newer_year)
        years = {int(y) for y in re.findall(r"20\d{2}", candidate_text)}
        return any(y < ny for y in years)
    except Exception:
        return False


# ── Main verification ─────────────────────────────────────────────────────────

def verify_provenance(
    candidate: dict,
    institution: str,
    university: str,
    policy_title: str,
    newer_metadata: dict,
) -> dict:
    """
    Evaluate provenance of a single candidate dict.
    Returns enriched candidate with source_verification added.
    """
    url         = candidate.get("source_url", "")
    doc_title   = candidate.get("document_title", "")
    snippet     = candidate.get("snippet", "")
    domain      = candidate.get("source_domain", "")
    combined    = f"{doc_title} {snippet}".lower()
    newer_year  = newer_metadata.get("academic_year", "")

    signals: list[str] = []
    warnings: list[str] = []
    score = 0

    # A — Domain authority
    if _is_official_domain(url):
        signals.append(f"Document hosted on official domain ({domain})")
        score += 3
    elif _is_third_party(url):
        warnings.append(f"Third-party hosting site ({domain}) — not an authoritative source")
        score -= 2
    else:
        warnings.append(f"Domain authority unknown ({domain})")

    # B — Institution name consistency
    if institution and _contains_words(combined, institution.split()):
        signals.append(f"Institution name '{institution}' found in candidate document")
        score += 2
    elif institution:
        warnings.append(f"Institution name '{institution}' not found in candidate snippet")

    # C — University consistency
    if university and _contains_words(combined, university.split()):
        signals.append(f"Affiliating university '{university}' found in candidate document")
        score += 2
    elif university:
        warnings.append(f"University name '{university}' not found in candidate snippet")

    # D — Policy title consistency
    title_sim = _similarity(policy_title, doc_title)
    if title_sim > 0.7:
        signals.append(f"Policy title closely matches (similarity {title_sim:.0%})")
        score += 3
    elif title_sim > 0.4 or _contains_words(combined, policy_title.split()[:3]):
        signals.append(f"Policy title partially matches candidate")
        score += 1
    else:
        warnings.append("Policy title does not closely match candidate document title")

    # E — Version / year consistency  (candidate should be from an earlier year)
    if newer_year and _is_previous_year(combined, newer_year[:4]):
        signals.append(f"Candidate appears to be from a year before {newer_year}")
        score += 2
    elif newer_year:
        warnings.append("Could not confirm candidate is from a previous year")

    # F — PDF link signal
    if url.lower().endswith(".pdf"):
        signals.append("Direct PDF link")
        score += 1

    # G — Policy keyword in snippet
    policy_kws = ["regulations", "policy", "rules", "guidelines", "ordinance", "statute"]
    if _contains_words(combined, policy_kws):
        signals.append("Policy-related keywords found in document snippet")
        score += 1

    # ── Determine tier ────────────────────────────────────────────────────────
    if score >= 9 and _is_official_domain(url):
        status = "VERIFIED_PROVENANCE"
    elif score >= 6 and not _is_third_party(url):
        status = "STRONG_PROVENANCE"
    elif score >= 3:
        status = "LIMITED_PROVENANCE"
    else:
        status = "UNVERIFIED"

    candidate["source_verification"] = {
        "status": status,
        "score":  score,
        "signals": signals,
        "warnings": warnings,
    }
    candidate["provenance_status"] = status
    candidate["provenance_score"]  = score

    return candidate


def rank_candidates(
    candidates: list[dict],
    institution: str,
    university: str,
    policy_title: str,
    newer_metadata: dict,
) -> list[dict]:
    """
    Verify provenance for all candidates and return them sorted best first.
    """
    verified = [
        verify_provenance(c, institution, university, policy_title, newer_metadata)
        for c in candidates
    ]

    tier_order = {
        "VERIFIED_PROVENANCE": 0,
        "STRONG_PROVENANCE":   1,
        "LIMITED_PROVENANCE":  2,
        "UNVERIFIED":          3,
    }

    verified.sort(key=lambda c: (
        tier_order.get(c.get("provenance_status", "UNVERIFIED"), 3),
        -c.get("provenance_score", 0),
    ))

    return verified
