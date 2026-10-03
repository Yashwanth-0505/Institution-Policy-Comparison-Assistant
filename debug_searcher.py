"""Debug why search_for_previous_policy returns empty."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch to add verbose output
import ml_pipeline.investigation.searcher as s
import re, urllib.parse, urllib.request

query = "Osmania University B.Tech Academic Regulations pdf"

# Step 1 — raw DDG
print("[1] Calling _ddg_search directly...")
results = s._ddg_search(query, max_results=5)
print(f"    Raw results: {len(results)}")
for r in results:
    print(f"    URL   : {r['url'][:80]}")
    print(f"    Title : {r['title'][:60]}")
    print()

# Step 2 — full search
print("[2] Calling search_for_previous_policy...")
candidates = s.search_for_previous_policy(
    institution="Osmania University",
    university="Osmania University",
    policy_title="Academic Regulations",
    academic_year="2024-25",
    role="student",
    programme="B.Tech",
    max_candidates=5,
)
print(f"    Candidates returned: {len(candidates)}")
for c in candidates:
    print(f"    {c.get('source_domain')} | {c.get('document_title','')[:50]}")
