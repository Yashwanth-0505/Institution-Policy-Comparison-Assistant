# -*- coding: utf-8 -*-
"""
Test script: Upload and compare sample policy PDFs to the Flask app.
Verifies the ML pipeline works end-to-end with real PDF uploads.
Run: python test_pdf_upload.py
"""
import sys
import requests
from pathlib import Path

BASE_URL = "http://localhost:5000"
SAMPLE_DIR = Path("sample_policies")

def test_pdf_upload():
    pdf_2025 = SAMPLE_DIR / "Academic_Regulations_2025.pdf"
    pdf_2026 = SAMPLE_DIR / "Academic_Regulations_2026.pdf"

    if not pdf_2025.exists() or not pdf_2026.exists():
        print("[ERROR] Sample PDFs not found.")
        return False

    print("\n" + "=" * 70)
    print("  Testing POLICYX PDF Upload & Comparison (Task #3)")
    print("=" * 70)
    print(f"\n[UPLOAD] Policy A: {pdf_2025.name}")
    print(f"[UPLOAD] Policy B: {pdf_2026.name}")

    try:
        with open(pdf_2025, "rb") as f_a, open(pdf_2026, "rb") as f_b:
            files = {
                "policy_a": (pdf_2025.name, f_a, "application/pdf"),
                "policy_b": (pdf_2026.name, f_b, "application/pdf"),
            }
            response = requests.post(f"{BASE_URL}/compare", files=files, timeout=60)
    except Exception as exc:
        print(f"[ERROR] Request failed: {exc}")
        return False

    if response.status_code != 200:
        print(f"[ERROR] Server returned {response.status_code}: {response.text}")
        return False

    data  = response.json()
    stats = data.get("stats", {})
    rows  = data.get("rows", [])

    print(f"\n[OK] Request successful!")
    print(f"\n[STATS]")
    print(f"  Total clauses in 2025 : {stats.get('total_clauses_a', 0)}")
    print(f"  Total clauses in 2026 : {stats.get('total_clauses_b', 0)}")
    print(f"  Total changes detected: {stats.get('total_changes', 0)}")
    print(f"\n[BREAKDOWN]")
    print(f"  Substantive  : {stats.get('substantive', 0)}")
    print(f"  Added        : {stats.get('added', 0)}")
    print(f"  Removed      : {stats.get('removed', 0)}")
    print(f"  Wording-only : {stats.get('wording_only', 0)}")
    print(f"  Unchanged    : {stats.get('unchanged', 0)}")

    print(f"\n[CHANGES]")
    for i, row in enumerate(rows[:10], 1):
        ct    = row.get("change_type", "unknown").upper()
        ha    = row.get("heading_a", "(no heading)")
        score = row.get("match_score", 0)
        print(f"  [{i}] {ct} | Match {score}% | {ha}")
        for sig in row.get("signals", []):
            print(f"       Signal: {sig.get('type')} '{sig.get('old')}' -> '{sig.get('new')}'")

    # --- Verification ---
    checks = {
        "Attendance 75->80"       : False,
        "Pass score 40->50"       : False,
        "Books 3->5"              : False,
        "Back-paper negation"     : False,
        "Training 2->4"           : False,
        "Academic Integrity added": False,
    }

    for row in rows:
        ta  = row.get("text_a", "").lower()
        tb  = row.get("text_b", "").lower()
        ha  = row.get("heading_a", "").lower()
        ct  = row.get("change_type", "")

        if "75" in ta and "80" in tb and "attendance" in ta:
            checks["Attendance 75->80"] = True
        if "40" in ta and "50" in tb and "pass" in ta:
            checks["Pass score 40->50"] = True
        if "3 books" in ta and "5 books" in tb:
            checks["Books 3->5"] = True
        if "permitted" in ta and "not permitted" in tb and "back" in ta:
            checks["Back-paper negation"] = True
        if "2 pre-placement" in ta and "4 pre-placement" in tb:
            checks["Training 2->4"] = True
        # For ADDED rows heading_a holds the new clause heading; text_b is empty
        if ct == "added" and "academic integrity" in ha:
            checks["Academic Integrity added"] = True

    print(f"\n[VERIFICATION]")
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    all_pass = all(checks.values())
    if all_pass:
        print("\n[RESULT] ALL CHECKS PASSED - Task #3 complete!")
    else:
        print("\n[RESULT] Some checks failed - see above.")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if test_pdf_upload() else 1)
