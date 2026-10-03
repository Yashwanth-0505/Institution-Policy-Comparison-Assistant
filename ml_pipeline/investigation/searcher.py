"""
searcher.py — AI-powered search for candidate previous policy documents.

Primary: Gemini API with Google Search grounding (real web results, no API key limits)
Fallback: Groq LLM to generate likely URLs + metadata
Last resort: DuckDuckGo HTML scrape

SOURCE PRIORITY (per spec):
  1. Official university domain (.ac.in, .edu, .gov)
  2. Official college/institution domain
  3. Official affiliated institutional repository
  4. Official government/accreditation source
  5. Other sources (candidate discovery only)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

try:
    import requests as _requests_lib
except ImportError:
    _requests_lib = None

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_TIMEOUT = 15

# ── Domain authority scoring ──────────────────────────────────────────────────

_EDU_GOV_RE  = re.compile(r"\.(edu|ac\.[a-z]{2}|gov|nic\.in|gov\.in)(/|$)", re.I)
_THIRD_PARTY = re.compile(
    r"(scribd|slideshare|academia\.edu|docplayer|studocu|coursehero|quora|reddit)", re.I
)


def _domain_priority(url: str) -> int:
    if _THIRD_PARTY.search(url): return 5
    if _EDU_GOV_RE.search(url):  return 1
    if re.search(r"\.(org|net)(/|$)", url, re.I): return 3
    return 4


def _source_type(url: str) -> str:
    if _EDU_GOV_RE.search(url):  return "official_domain"
    if _THIRD_PARTY.search(url): return "third_party"
    return "web"


def _extract_domain(url: str) -> str:
    try:    return urllib.parse.urlparse(url).netloc
    except: return url


def _now_iso() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()


def _enrich(raw: list[dict]) -> list[dict]:
    """Add domain_priority, source_type, retrieval_ts to raw results."""
    seen: set[str] = set()
    out  = []
    for r in raw:
        url = r.get("source_url", r.get("url", ""))
        if not url or url in seen: continue
        seen.add(url)
        out.append({
            "source_url":      url,
            "source_domain":   _extract_domain(url),
            "document_title":  r.get("title", r.get("document_title", "")),
            "snippet":         r.get("snippet", ""),
            "source_type":     _source_type(url),
            "domain_priority": _domain_priority(url),
            "query_used":      r.get("query_used", ""),
            "retrieval_ts":    _now_iso(),
        })
    return sorted(out, key=lambda c: c["domain_priority"])


# ── 1. Gemini with Google Search grounding ────────────────────────────────────

def _gemini_search(institution: str, university: str, policy_title: str,
                   programme: str, role: str, academic_year: str) -> list[dict]:
    """
    Use Gemini API with Google Search grounding to find real policy document URLs.
    Returns list of raw result dicts.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []

    # Build a rich search prompt
    prog_hint = f"{programme} " if programme and programme != "General" else ""
    year_hint = ""
    if academic_year:
        m = re.search(r"(20\d{2})", academic_year)
        if m:
            y = int(m.group(1))
            year_hint = f" for the year {y-1}-{str(y)[-2:]}"

    prompt = f"""You are a research assistant helping find official academic policy documents.

Search for the PREVIOUS version of the following policy document:

Institution: {institution}
Affiliating University: {university}
Policy/Regulation: {prog_hint}{policy_title}{year_hint}
For: {role if role else "students"}

Search the web and find 3-5 real URLs pointing to official PDF or web pages containing this policy from the university or institution's official domain.

Prioritise:
1. Official university domain (e.g. osmania.ac.in, ou.ac.in)
2. Official college domain
3. Government/accreditation sources (.gov.in, nic.in)

Return ONLY a JSON array. Each object must have:
- "url": the exact working URL
- "title": document title
- "snippet": brief description of what was found

Example:
[
  {{"url": "https://osmania.ac.in/regulations/btech2023.pdf", "title": "B.Tech Regulations 2023-24", "snippet": "Official B.Tech academic regulations from Osmania University"}}
]

Return ONLY the JSON array, no other text."""

    try:
        import google.genai as genai
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={
                "tools": [{"google_search": {}}],
                "temperature": 0.1,
            }
        )

        raw_text = response.text.strip() if response.text else ""
        logger.info(f"[Gemini Search] Raw response length: {len(raw_text)}")

        # Extract JSON array from response
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_text, re.S)
        if json_match:
            data = json.loads(json_match.group(0))
            results = []
            for item in data:
                url = item.get("url", "").strip()
                if url and url.startswith("http"):
                    results.append({
                        "source_url": url,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "query_used": f"gemini_search:{institution} {policy_title}",
                    })
            logger.info(f"[Gemini Search] Found {len(results)} candidates")
            return results

        # If Gemini didn't return JSON but has grounding metadata, extract those URLs
        if hasattr(response, 'candidates') and response.candidates:
            urls = []
            for candidate in response.candidates:
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    gm = candidate.grounding_metadata
                    if hasattr(gm, 'grounding_chunks'):
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, 'web') and chunk.web:
                                urls.append({
                                    "source_url": chunk.web.uri,
                                    "title": chunk.web.title or "",
                                    "snippet": "",
                                    "query_used": "gemini_grounding",
                                })
            if urls:
                logger.info(f"[Gemini Grounding] Found {len(urls)} URLs from grounding metadata")
                return urls

    except Exception as exc:
        logger.warning(f"[Gemini Search] Failed: {exc}")

    return []


# ── 2. Groq LLM fallback ──────────────────────────────────────────────────────

def _groq_search(institution: str, university: str, policy_title: str,
                 programme: str, role: str, academic_year: str) -> list[dict]:
    """Use Groq to suggest likely official URLs for the policy."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return []

    prog_hint = f"{programme} " if programme and programme != "General" else ""
    year_hint = ""
    if academic_year:
        m = re.search(r"(20\d{2})", academic_year)
        if m:
            y = int(m.group(1))
            year_hint = f" {y-1}-{str(y)[-2:]}"

    prompt = f"""Find the official web page or PDF URL for the previous version of this academic policy:

Institution: {institution}
University: {university}
Policy: {prog_hint}{policy_title}{year_hint}
Target: {role if role else "students"}

Return a JSON array of 3-5 likely real URLs from official domains (.ac.in, .edu, .gov.in).
Each object: {{"url": "...", "title": "...", "snippet": "..."}}
Only return the JSON array."""

    try:
        import openai
        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": "You are a research assistant. Return only valid JSON."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        raw = response.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)

        json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.S)
        if json_match:
            data = json.loads(json_match.group(0))
            results = []
            for item in data:
                url = item.get("url", "").strip()
                if url and url.startswith("http"):
                    results.append({
                        "source_url": url,
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "query_used": f"groq_llm:{institution} {policy_title}",
                    })
            logger.info(f"[Groq Search] Found {len(results)} candidates")
            return results

    except Exception as exc:
        logger.warning(f"[Groq Search] Failed: {exc}")

    return []


# ── 3. DuckDuckGo HTML scrape (last resort) ───────────────────────────────────

def _ddg_search(query: str, max_results: int = 5) -> list[dict]:
    """Scrape DuckDuckGo HTML search (no API key, last resort)."""
    search_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    headers = {
        "User-Agent": _BROWSER_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        if _requests_lib:
            resp = _requests_lib.get(search_url, headers=headers, timeout=_TIMEOUT)
            resp.raise_for_status()
            html = resp.text
        else:
            req = urllib.request.Request(search_url, headers=headers)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                html = r.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.warning(f"[DDG] Request failed: {exc}")
        return []

    raw_urls   = re.findall(r"uddg=([^&\"'\s>]+)", html)
    titles_raw = re.findall(r'class="result__a"[^>]*>(.*?)</a>', html, re.S)
    snips_raw  = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)>', html, re.S)

    results = []
    seen: set[str] = set()
    for i, raw_url in enumerate(raw_urls):
        url = urllib.parse.unquote(raw_url)
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        title   = re.sub(r"<[^>]+>", "", titles_raw[i] if i < len(titles_raw) else "").strip()
        snippet = re.sub(r"<[^>]+>", "", snips_raw[i]  if i < len(snips_raw)  else "").strip()
        title   = title.replace("&amp;", "&").replace("&#x27;", "'")
        snippet = snippet.replace("&amp;", "&").replace("&#x27;", "'")
        results.append({"url": url, "title": title, "snippet": snippet, "query_used": query})
        if len(results) >= max_results:
            break

    logger.info(f"[DDG] Found {len(results)} results for: {query[:60]}")
    return results


# ── Main public function ──────────────────────────────────────────────────────

def search_for_previous_policy(
    institution: str,
    university: str,
    policy_title: str,
    academic_year: str,
    max_candidates: int = 5,
    role: str = "",
    programme: str = "",
) -> list[dict]:
    """
    Search for candidate previous policy documents using AI-powered search.

    Provider order:
      1. Gemini API with Google Search grounding (best)
      2. Groq LLM URL suggestions (fallback)
      3. DuckDuckGo HTML scrape (last resort)

    Returns a list of enriched candidate dicts sorted by domain priority.
    Never raises.
    """
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

    if not institution and university:
        institution = university
    elif not university and institution:
        university = institution

    raw: list[dict] = []

    # Try Gemini first
    logger.info("[Search] Trying Gemini API with Google Search grounding...")
    raw = _gemini_search(institution, university, policy_title, programme, role, academic_year)

    # Fallback to Groq
    if not raw:
        logger.info("[Search] Gemini returned nothing, trying Groq LLM...")
        raw = _groq_search(institution, university, policy_title, programme, role, academic_year)

    # Last resort: DuckDuckGo
    if not raw:
        logger.info("[Search] LLMs returned nothing, falling back to DuckDuckGo...")
        prog_hint = f"{programme} " if programme and programme != "General" else ""
        year_hint = ""
        if academic_year:
            m = re.search(r"(20\d{2})", academic_year)
            if m:
                y = int(m.group(1))
                year_hint = str(y - 1)

        queries = [
            f"{institution} {prog_hint}{policy_title} pdf {year_hint}".strip(),
            f"{university} {prog_hint}{policy_title} regulations pdf".strip(),
        ]
        for q in queries:
            hits = _ddg_search(q, max_results=5)
            raw.extend(hits)
            if len(raw) >= max_candidates * 2:
                break
            time.sleep(0.5)

    if not raw:
        logger.warning("[Search] All providers returned no results.")
        return []

    enriched = _enrich(raw)
    return enriched[:max_candidates]
