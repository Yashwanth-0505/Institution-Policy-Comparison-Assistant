"""
metadata_extractor.py — Extract policy metadata from an uploaded PDF.

Pulls: title, issuing authority, effective/academic year, version string,
policy category, and document references — all without an LLM.
"""
from __future__ import annotations

import re
from pathlib import Path


# ── Regex helpers ────────────────────────────────────────────────────────────

_YEAR_RE      = re.compile(r"\b(20\d{2})\b")
_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})[–\-](20\d{2}|\d{2})\b")
_VERSION_RE   = re.compile(r"\b(v\d+(?:\.\d+)*|version\s*\d+[\d.]*|rev(?:ision)?\s*\d+)\b", re.I)

_HEADING_PATTERNS = [
    re.compile(r"^(\d+\.\s+[A-Z][A-Z\s&/\-]{3,})", re.M),
    re.compile(r"^([A-Z][A-Z\s&/\-]{5,})$", re.M),
]

_AUTHORITY_KEYWORDS = [
    "university", "college", "institute", "board", "registrar",
    "academic council", "senate", "examination", "department",
]

_CATEGORY_MAP = {
    "attendance":     "Attendance Policy",
    "examination":    "Examination Rules",
    "grading":        "Grading Policy",
    "academic":       "Academic Regulations",
    "placement":      "Placement Policy",
    "library":        "Library Policy",
    "hostel":         "Hostel Rules",
    "scholarship":    "Scholarship Policy",
    "fee":            "Fee Structure",
    "disciplinary":   "Disciplinary Policy",
    "admission":      "Admission Policy",
    "syllabus":       "Syllabus / Curriculum",
}


def extract_metadata(pdf_path: str | Path) -> dict:
    """
    Return a metadata dict extracted from the PDF text.
    Never raises — on any failure returns partial results.
    """
    pdf_path = Path(pdf_path)
    meta: dict = {
        "filename": pdf_path.name,
        "title": "",
        "issuing_authority": "",
        "academic_year": "",
        "version": "",
        "policy_category": "",
        "years_mentioned": [],
        "headings": [],
        "raw_first_500": "",
    }

    try:
        text = _read_pdf_text(pdf_path)
    except Exception as exc:
        meta["error"] = str(exc)
        return meta

    first500 = text[:500].strip()
    meta["raw_first_500"] = first500

    # Years
    years = sorted(set(_YEAR_RE.findall(text)), reverse=True)
    meta["years_mentioned"] = years[:6]

    # Academic year range
    yr_range = _YEAR_RANGE_RE.search(text)
    if yr_range:
        meta["academic_year"] = yr_range.group(0)
    elif years:
        meta["academic_year"] = years[0]

    # Version
    ver = _VERSION_RE.search(text)
    meta["version"] = ver.group(0) if ver else ""

    # Title — first non-blank line of meaningful length
    for line in first500.splitlines():
        line = line.strip()
        if 6 <= len(line) <= 120 and not line.startswith("http"):
            meta["title"] = line
            break

    # Issuing authority — look for lines with authority keywords near top
    top = text[:800].lower()
    for kw in _AUTHORITY_KEYWORDS:
        idx = top.find(kw)
        if idx != -1:
            # grab the surrounding line
            start = top.rfind("\n", 0, idx) + 1
            end   = top.find("\n", idx)
            line  = text[start: end if end != -1 else start + 120].strip()
            if line:
                meta["issuing_authority"] = line[:120]
                break

    # Policy category
    text_lower = text.lower()
    for kw, cat in _CATEGORY_MAP.items():
        if kw in text_lower:
            meta["policy_category"] = cat
            break

    # Headings
    headings = []
    for pat in _HEADING_PATTERNS:
        headings += pat.findall(text[:3000])
    meta["headings"] = list(dict.fromkeys(h.strip() for h in headings))[:15]

    return meta


def _read_pdf_text(path: Path) -> str:
    """Read text from a PDF using pymupdf (fitz)."""
    try:
        import pymupdf as fitz  # preferred import
    except ImportError:
        import fitz              # fallback

    text_parts = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)
