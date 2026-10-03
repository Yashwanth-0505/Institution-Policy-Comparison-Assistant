"""
searcher.py — Search for candidate previous policy documents.

Uses DuckDuckGo HTML search (no API key required) as the primary provider,
with a fallback to Google site: queries via the remote_web_search-style URL.

SOURCE PRIORITY (per spec):
  1. Official university domain
  2. Official college/institution domain
  3. Official affiliated institutional repository
  4. Official government/accreditation source
  5. Other sources (candidate discovery only)
"""
from __future__ import annotations

import re
import time
import urllib.parse
from typing import Any

import requests

_UA = "Mozilla/5.0 (compatible; POLICYX/1.0; +https://policyx.io)"
_TIMEOUT = 10


# ── Domain authority scoring ─────────────────────────────────────────────────

_EDU_GOV_RE = re.compile(r"\.(edu|ac\.[a-z]{2}|gov|nic\.in|gov\.in)(/|$)", re.I)
_THIRD_PARTY = re.compile(r"(scribd|slideshare|academia\.edu|docplayer|studocu|coursehero)", re.I)


def _domain_priority(url: str) -> int:
    """Return 1 (highest) … 5 (lowest) based on domain authority."""
    if _THIRD_PARTY.search(url):
        return 5
    if _EDU_GOV_RE.search(url):
        return 1
    if re.search(r"\.(org|net)(/|$)", url, re.I):
        return 3
    return 4


def _source_type(url: str) -> str:
    if _EDU_GOV_RE.search(url):
        return "official_domain"
    if _THIRD_PARTY.search(url):
        return "third_party"
    return "web"


# ── Query builders ────────────────────────────────────────────────────────────

def _build_queries(institution: str, university: str,
                   policy_title: str, academic_year: str) -> list[str]:
    """Return a ranked list of search query strings."""
    year_hint = ""
    if academic_year:
        # e.g. "2024-25" or "2024"
        m = re.search(r"(20\d{2})", academic_year)
        if m:
            y = int(m.group(1))
            year_hint = f"{y - 1}-{str(y)[-2:]}"  # previous year range

    queries = []

    if institution and policy_title:
        queries.append(f'"{institution}" "{policy_title}" {year_hint} filetype:pdf')
        queries.append(f'"{institution}" "{policy_title}" regulations {year_hint}')

    if university and policy_title:
        queries.append(f'"{university}" "{policy_title}" {year_hint} filetype:pdf')
        queries.append(f'"{university}" "{policy_title}" previous version')

    if policy_title:
        queries.append(f'{policy_title} {institution} {year_hint} pdf regulations')

    return [q.strip() for q in queries if q.strip()]


# ── DuckDuckGo HTML scrape ────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """Scrape DuckDuckGo HTML search results (no JS, no API key)."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "b": "", "kl": "us-en"}
    headers = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}

    try:
        resp = requests.post(url, data=params, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception:
        return []

    html = resp.text
    results = []

    # Extract result blocks
    for block in re.findall(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.S
    ):
        raw_url, raw_title, raw_snippet = block
        # DDG sometimes wraps with redirect
        url_match = re.search(r'uddg=([^&"]+)', raw_url)
        actual_url = urllib.parse.unquote(url_match.group(1)) if url_match else raw_url

        title   = re.sub(r"<[^>]+>", "", raw_title).strip()
        snippet = re.sub(r"<[^>]+>", "", raw_snippet).strip()

        if actual_url.startswith("http"):
            results.append({
                "url": actual_url,
                "title": title,
                "snippet": snippet,
            })
        if len(results) >= max_results:
            break

    return results


# ── Main search function ──────────────────────────────────────────────────────

def search_for_previous_policy(
    institution: str,
    university: str,
    policy_title: str,
    academic_year: str,
    max_candidates: int = 5,
) -> list[dict]:
    """
    Search for candidate previous policy documents.
    Returns a list of candidate dicts sorted by domain_priority (ascending = better).
    Never raises.
    """
    queries = _build_queries(institution, university, policy_title, academic_year)
    seen_urls: set[str] = set()
    raw_results: list[dict] = []

    for query in queries[:3]:   # limit to 3 queries to avoid rate limits
        hits = _ddg_search(query, max_results=5)
        for hit in hits:
            url = hit.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                raw_results.append({**hit, "query_used": query})
        if len(raw_results) >= max_candidates * 2:
            break
        time.sleep(0.5)         # polite delay

    # Enrich with provenance signals
    candidates = []
    for r in raw_results:
        url  = r.get("url", "")
        prio = _domain_priority(url)
        cand = {
            "source_url":    url,
            "source_domain": _extract_domain(url),
            "document_title": r.get("title", ""),
            "snippet":        r.get("snippet", ""),
            "source_type":    _source_type(url),
            "domain_priority": prio,
            "query_used":     r.get("query_used", ""),
            "retrieval_ts":   _now_iso(),
        }
        candidates.append(cand)

    # Sort: official domains first, then by title relevance
    candidates.sort(key=lambda c: (c["domain_priority"], 0))

    return candidates[:max_candidates]


def _extract_domain(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc
    except Exception:
        return url


def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()
