# -*- coding: utf-8 -*-
"""
test_investigation.py
Standalone terminal test for the Investigation Mode.
Searches for an Osmania University B.Tech Academic Regulations PDF
WITHOUT needing an uploaded newer policy.

Run: python test_investigation.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ml_pipeline.investigation.searcher import search_for_previous_policy
from ml_pipeline.investigation.provenance import rank_candidates

INSTITUTION   = "Osmania University"
UNIVERSITY    = "Osmania University"
POLICY_TITLE  = "Academic Regulations"
PROGRAMME     = "B.Tech"
ROLE          = "student"
ACADEMIC_YEAR = "2024-25"

def run():
    print("\n" + "="*65)
    print("  POLICYX Investigation Mode — Terminal Test")
    print("="*65)
    print(f"  Institution  : {INSTITUTION}")
    print(f"  University   : {UNIVERSITY}")
    print(f"  Policy       : {POLICY_TITLE}")
    print(f"  Programme    : {PROGRAMME}")
    print(f"  Role         : {ROLE}")
    print(f"  Year         : {ACADEMIC_YEAR}")
    print("="*65)

    print("\n[1/3] Searching for candidate documents (Gemini -> Groq -> DDG)...")
    # Enable logging so we can see which provider fires
    import logging
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
    candidates = search_for_previous_policy(
        institution=INSTITUTION,
        university=UNIVERSITY,
        policy_title=POLICY_TITLE,
        academic_year=ACADEMIC_YEAR,
        max_candidates=5,
        role=ROLE,
        programme=PROGRAMME,
    )

    if not candidates:
        print("\n[FAIL] No candidates found.")
        print("       This could be a network/rate-limit issue.")
        print("       Try again in a few seconds.")
        sys.exit(1)

    print(f"\n[2/3] Found {len(candidates)} candidate(s). Verifying provenance...")

    # Fake newer_metadata (no PDF needed for this test)
    newer_metadata = {
        "academic_year": ACADEMIC_YEAR,
        "title": f"{POLICY_TITLE} {ACADEMIC_YEAR}",
    }

    ranked = rank_candidates(
        candidates,
        institution=INSTITUTION,
        university=UNIVERSITY,
        policy_title=POLICY_TITLE,
        newer_metadata=newer_metadata,
    )

    print(f"\n[3/3] Results — ranked by provenance:\n")
    print(f"  {'#':<3} {'PROVENANCE':<22} {'SCORE':<6} {'DOMAIN':<35} TITLE")
    print(f"  {'-'*3} {'-'*22} {'-'*6} {'-'*35} {'-'*30}")

    for i, c in enumerate(ranked, 1):
        prov   = c.get("provenance_status", "UNVERIFIED")
        score  = c.get("provenance_score", 0)
        domain = c.get("source_domain", "")[:34]
        title  = c.get("document_title", "")[:50]
        print(f"  {i:<3} {prov:<22} {score:<6} {domain:<35} {title}")

    print("\n  --- Detailed signals for top candidate ---")
    top = ranked[0]
    sv  = top.get("source_verification", {})
    print(f"  Status  : {sv.get('status')}")
    print(f"  Score   : {sv.get('score')}")
    print(f"  URL     : {top.get('source_url','')[:80]}")

    signals  = sv.get("signals", [])
    warnings = sv.get("warnings", [])

    if signals:
        print("  Signals :")
        for s in signals:
            print(f"    [OK]  {s}")
    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    [!!]  {w}")

    # Overall result
    top_status = top.get("provenance_status", "UNVERIFIED")
    if top_status in ("VERIFIED_PROVENANCE", "STRONG_PROVENANCE"):
        print(f"\n[RESULT] SUCCESS — top candidate has {top_status}")
        print(f"         Ready to use in comparison pipeline.")
    elif top_status == "LIMITED_PROVENANCE":
        print(f"\n[RESULT] PARTIAL — top candidate has LIMITED_PROVENANCE")
        print(f"         User should manually verify before using.")
    else:
        print(f"\n[RESULT] UNVERIFIED — candidates found but provenance is weak.")
        print(f"         Recommend manual upload instead.")

    print()

if __name__ == "__main__":
    run()
