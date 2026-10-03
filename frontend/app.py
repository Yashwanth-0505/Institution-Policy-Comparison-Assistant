"""
app.py — Flask web application for POLICYX Institution Policy Comparison Assistant.
Side-by-side comparison with semantic diff, citations, summaries, auth, and admin.
"""
from __future__ import annotations

import os
import sys
import tempfile
from functools import wraps
from pathlib import Path

from flask import (Flask, flash, jsonify, make_response, redirect,
                   render_template, request, url_for)

# Ensure ml_pipeline is importable from d:\jig
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline.pipeline import compare_policies
from ml_pipeline.config import load_config
from database import (
    init_db, register_user, login_user, logout_user,
    get_user_by_token, verify_admin_code,
    get_all_users, get_comparison_stats, log_comparison,
)

app = Flask(__name__)
app.secret_key = os.environ.get("POLICYX_SECRET", "dev-secret-change-in-production-2026")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

CHANGE_COLORS = {
    "unchanged":   "#e8f5e9",
    "wording_only":"#fff9c4",
    "substantive": "#ffebee",
    "added":       "#e3f2fd",
    "removed":     "#fce4ec",
    "uncertain":   "#f3e5f5",
}

CHANGE_BADGES = {
    "unchanged":   ("UNCHANGED",    "#4caf50"),
    "wording_only":("WORDING ONLY", "#ff9800"),
    "substantive": ("SUBSTANTIVE",  "#f44336"),
    "added":       ("ADDED",        "#2196f3"),
    "removed":     ("REMOVED",      "#e91e63"),
    "uncertain":   ("NEEDS REVIEW", "#9c27b0"),
}

# ──────────────────────────────────────────────
# Auth helpers
# ──────────────────────────────────────────────

SESSION_COOKIE = "px_session"

def get_current_user():
    token = request.cookies.get(SESSION_COOKIE)
    return get_user_by_token(token)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def _set_session_cookie(response, token: str):
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=7 * 24 * 3600,
        httponly=True, samesite="Lax",
    )
    return response


# ──────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────

@app.route("/")
def home():
    """Public landing page."""
    return render_template("home.html")


@app.route("/login")
def login_page():
    """Login / Sign-up page."""
    if get_current_user():
        return redirect(url_for("app_page"))
    return render_template("login.html")


@app.route("/app")
@login_required
def app_page():
    """Main comparison app — requires login."""
    return render_template("index.html")


@app.route("/logout")
def logout():
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        logout_user(token)
    resp = make_response(redirect(url_for("home")))
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ──────────────────────────────────────────────
# Auth API
# ──────────────────────────────────────────────

@app.route("/auth/register", methods=["POST"])
def auth_register():
    first_name  = request.form.get("first_name", "").strip()
    last_name   = request.form.get("last_name", "").strip()
    email       = request.form.get("email", "").strip()
    institution = request.form.get("institution", "").strip()
    password    = request.form.get("password", "")

    if not all([first_name, last_name, email, password]):
        return jsonify({"ok": False, "error": "All required fields must be filled in."})

    result = register_user(first_name, last_name, email, institution, password)
    if not result["ok"]:
        return jsonify(result)

    # Auto-login after registration
    login_result = login_user(email, password)
    if not login_result["ok"]:
        return jsonify({"ok": True, "redirect": url_for("login_page")})

    resp = make_response(jsonify({"ok": True, "redirect": url_for("app_page")}))
    _set_session_cookie(resp, login_result["token"])
    return resp


@app.route("/auth/login", methods=["POST"])
def auth_login():
    email    = request.form.get("email", "").strip()
    password = request.form.get("password", "")

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password are required."})

    result = login_user(email, password)
    if not result["ok"]:
        return jsonify(result)

    resp = make_response(jsonify({"ok": True, "redirect": url_for("app_page")}))
    _set_session_cookie(resp, result["token"])
    return resp


# ──────────────────────────────────────────────
# Admin
# ──────────────────────────────────────────────

ADMIN_SESSION_COOKIE = "px_admin"


def is_admin_authed():
    return request.cookies.get(ADMIN_SESSION_COOKIE) == "granted"


@app.route("/admin")
def admin_page():
    authed = is_admin_authed()
    stats  = get_comparison_stats() if authed else {
        "total_users": 0, "total_comparisons": 0,
        "total_institutions": 0, "active_today": 0,
        "recent_comparisons": [],
    }
    users = get_all_users() if authed else []
    return render_template("admin.html", admin_authed=authed, stats=stats, users=users)


@app.route("/admin/verify", methods=["POST"])
def admin_verify():
    code = request.form.get("code", "")
    if not verify_admin_code(code):
        return jsonify({"ok": False, "error": "Invalid access code. Try again."})

    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie(
        ADMIN_SESSION_COOKIE, "granted",
        max_age=4 * 3600,
        httponly=True, samesite="Strict",
    )
    return resp


@app.route("/admin/data")
def admin_data():
    """Return live stats + users as JSON — only when admin cookie is set."""
    if not is_admin_authed():
        return jsonify({"error": "Unauthorized"}), 403

    stats = get_comparison_stats()
    users = get_all_users()
    return jsonify({"stats": stats, "users": users})


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    resp = make_response(jsonify({"ok": True}))
    resp.delete_cookie(ADMIN_SESSION_COOKIE)
    return resp


# ──────────────────────────────────────────────
# Investigation Mode — Missing Previous Policy
# ──────────────────────────────────────────────

from ml_pipeline.investigation import (
    InvestigationState,
    extract_metadata,
    search_for_previous_policy,
    rank_candidates,
    download_candidate,
)

# In-memory session store (replace with Redis/DB in production)
_INVESTIGATIONS: dict[str, InvestigationState] = {}


def _get_investigation(session_id: str) -> InvestigationState | None:
    return _INVESTIGATIONS.get(session_id)


@app.route("/investigate/start", methods=["POST"])
def investigate_start():
    """
    Start a new investigation session.
    User uploads the NEWER policy PDF only.
    Returns: {"session_id": "...", "state": {...}}
    """
    if "newer_policy" not in request.files:
        return jsonify({"ok": False, "error": "Newer policy PDF is required."}), 400

    file = request.files["newer_policy"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"ok": False, "error": "Please upload a PDF file."}), 400

    # Save to temp
    tmp_dir = tempfile.mkdtemp(prefix="policyx_inv_")
    newer_path = os.path.join(tmp_dir, "newer_policy.pdf")
    file.save(newer_path)

    # Create investigation session
    import uuid
    session_id = str(uuid.uuid4())
    inv = InvestigationState()
    inv.transition("COLLECTING_CONTEXT", "Uploaded newer policy, ready for user input.")

    # Store temp path for later
    inv.extracted_metadata["_tmp_newer_path"] = newer_path
    inv.extracted_metadata["newer_filename"] = file.filename

    _INVESTIGATIONS[session_id] = inv

    return jsonify({
        "ok": True,
        "session_id": session_id,
        "state": inv.to_dict(),
    })


@app.route("/investigate/context", methods=["POST"])
def investigate_context():
    """
    Submit Q1 + Q2 answers, extract metadata, search, rank candidates.
    Body: {session_id, institution_name, affiliating_university, policy_title, department?, academic_year?}
    Returns: {"ok": True, "state": {...}, "candidates": [...]}
    """
    data = request.get_json() or {}
    session_id = data.get("session_id")
    inv = _get_investigation(session_id)
    if not inv:
        return jsonify({"ok": False, "error": "Invalid session."}), 404

    inv.institution_name        = data.get("institution_name", "").strip()
    inv.affiliating_university  = data.get("affiliating_university", "").strip()
    inv.policy_title            = data.get("policy_title", "").strip()
    inv.department              = data.get("department", "").strip()
    inv.academic_year           = data.get("academic_year", "").strip()

    if not inv.institution_name or not inv.policy_title:
        return jsonify({"ok": False, "error": "Institution and policy title are required."}), 400

    # EXTRACTING_METADATA
    inv.transition("EXTRACTING_METADATA", "Analysing the uploaded policy document.")
    newer_path = inv.extracted_metadata.get("_tmp_newer_path")
    if newer_path and os.path.exists(newer_path):
        try:
            meta = extract_metadata(newer_path)
            inv.extracted_metadata.update(meta)
        except Exception as exc:
            inv.fail(f"Metadata extraction failed: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 500

    # SEARCHING_SOURCES
    inv.transition("SEARCHING_SOURCES", "Searching for the previous policy version.")
    try:
        raw_candidates = search_for_previous_policy(
            inv.institution_name,
            inv.affiliating_university,
            inv.policy_title,
            inv.academic_year or inv.extracted_metadata.get("academic_year", ""),
            max_candidates=5,
        )
    except Exception as exc:
        inv.fail(f"Search failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500

    if not raw_candidates:
        inv.fail("No candidate documents found. Please upload the previous policy manually.")
        return jsonify({
            "ok": False,
            "error": "No candidates found.",
            "state": inv.to_dict(),
        })

    # RANKING_CANDIDATES
    inv.transition("RANKING_CANDIDATES", f"Found {len(raw_candidates)} candidate(s), ranking by provenance.")

    # VERIFYING_PROVENANCE
    inv.transition("VERIFYING_PROVENANCE", "Evaluating source authority and provenance signals.")
    try:
        ranked = rank_candidates(
            raw_candidates,
            inv.institution_name,
            inv.affiliating_university,
            inv.policy_title,
            inv.extracted_metadata,
        )
        inv.candidates = ranked
    except Exception as exc:
        inv.fail(f"Provenance verification failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500

    # AWAITING_CONFIRMATION
    inv.transition("AWAITING_CONFIRMATION", "Candidates ranked. Awaiting user confirmation.")

    return jsonify({
        "ok": True,
        "state": inv.to_dict(),
        "candidates": inv.candidates,
    })


@app.route("/investigate/confirm", methods=["POST"])
def investigate_confirm():
    """
    User confirms a candidate. Download it, run comparison.
    Body: {session_id, candidate_index}
    Returns: {"ok": True, "state": {...}, "comparison_result": {...}}
    """
    data = request.get_json() or {}
    session_id = data.get("session_id")
    idx = data.get("candidate_index")

    inv = _get_investigation(session_id)
    if not inv:
        return jsonify({"ok": False, "error": "Invalid session."}), 404

    if inv.state != "AWAITING_CONFIRMATION":
        return jsonify({"ok": False, "error": "Not in confirmation state."}), 400

    if idx is None or not (0 <= idx < len(inv.candidates)):
        return jsonify({"ok": False, "error": "Invalid candidate index."}), 400

    candidate = inv.candidates[idx]
    inv.confirmed_candidate = candidate

    # DOWNLOADING_DOCUMENT
    inv.transition("DOWNLOADING_DOCUMENT", f"Downloading candidate from {candidate['source_url']}")
    try:
        dl_result = download_candidate(candidate["source_url"])
    except Exception as exc:
        inv.fail(f"Download failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500

    if not dl_result.get("ok"):
        inv.fail(f"Download failed: {dl_result.get('error')}")
        return jsonify({"ok": False, "error": dl_result.get("error")}), 500

    earlier_path = dl_result["path"]

    # COMPARING_POLICIES
    inv.transition("COMPARING_POLICIES", "Running the ML comparison pipeline.")
    newer_path = inv.extracted_metadata.get("_tmp_newer_path")
    if not newer_path or not os.path.exists(newer_path):
        inv.fail("Newer policy file not found.")
        return jsonify({"ok": False, "error": "Newer policy file missing."}), 500

    try:
        config = load_config()
        result = compare_policies(earlier_path, newer_path, config=config)
    except Exception as exc:
        inv.fail(f"Comparison failed: {exc}")
        return jsonify({"ok": False, "error": str(exc)}), 500

    # Build rows
    rows = _build_rows(result)
    stats = {
        "total_clauses_a":   len(result.clauses_a),
        "total_clauses_b":   len(result.clauses_b),
        "total_changes":     len(result.changes),
        "added":             sum(1 for r in rows if r["change_type"] == "added"),
        "removed":           sum(1 for r in rows if r["change_type"] == "removed"),
        "substantive":       sum(1 for r in rows if r["change_type"] == "substantive"),
        "wording_only":      sum(1 for r in rows if r["change_type"] == "wording_only"),
        "unchanged":         sum(1 for r in rows if r["change_type"] == "unchanged"),
        "needs_review":      sum(1 for r in rows if r["needs_review"]),
        "doc_a_name":        f"[DISCOVERED] {dl_result['filename']}",
        "doc_b_name":        inv.extracted_metadata.get("newer_filename", "newer_policy.pdf"),
        "source_url":        candidate["source_url"],
        "provenance_status": candidate.get("provenance_status", "UNVERIFIED"),
        "errors":            result.errors,
        "warnings":          result.warnings,
        "processing_complete": result.processing_complete,
    }

    # Log to DB
    user = get_current_user()
    log_comparison(user["id"] if user else None, stats["doc_a_name"], stats["doc_b_name"], stats)

    # COMPLETED
    inv.transition("COMPLETED", "Investigation and comparison completed successfully.")

    return jsonify({
        "ok": True,
        "state": inv.to_dict(),
        "stats": stats,
        "rows": rows,
    })


@app.route("/investigate/cancel", methods=["POST"])
def investigate_cancel():
    """Cancel an investigation session and clean up temp files."""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    inv = _get_investigation(session_id)
    if inv:
        # Clean up temp files
        newer_path = inv.extracted_metadata.get("_tmp_newer_path")
        if newer_path and os.path.exists(newer_path):
            try:
                os.remove(newer_path)
            except Exception:
                pass
        _INVESTIGATIONS.pop(session_id, None)
    return jsonify({"ok": True})


# ──────────────────────────────────────────────
# Comparison API  (unchanged logic)
# ──────────────────────────────────────────────

def _build_rows(result):
    """Serialise comparison result to frontend-friendly list."""
    rows = []
    for change in result.changes:
        ct       = change.change_type.value
        match    = change.match
        clause_a = match.clause_a
        clause_b = match.clause_b

        citations_a = [c for c in change.citations if c.document_id == clause_a.document_id]
        citations_b = [c for c in change.citations if clause_b and c.document_id == clause_b.document_id]

        signals = [
            {"type": s.signal_type.replace("_", " ").title(),
             "desc": s.description, "old": s.old_value, "new": s.new_value}
            for s in change.signals
        ]

        rows.append({
            "change_type":  ct,
            "badge_label":  CHANGE_BADGES.get(ct, ("UNKNOWN", "#999"))[0],
            "badge_color":  CHANGE_BADGES.get(ct, ("UNKNOWN", "#999"))[1],
            "bg_color":     CHANGE_COLORS.get(ct, "#fff"),
            "heading_a":    clause_a.heading or "(no heading)",
            "heading_b":    clause_b.heading if clause_b else "—",
            "text_a":       clause_a.normalized_text,
            "text_b":       clause_b.normalized_text if clause_b else "",
            "page_a":       clause_a.page_number,
            "page_b":       clause_b.page_number if clause_b else None,
            "doc_a":        clause_a.document_name,
            "doc_b":        clause_b.document_name if clause_b else "",
            "match_score":  round(match.match_score * 100, 1),
            "signals":      signals,
            "summary":      change.summary.summary if change.summary else "",
            "provider":     change.summary.provider_used if change.summary else "",
            "needs_review": change.needs_review,
            "citations_a":  [{"page": c.page_number, "valid": c.citation_valid} for c in citations_a],
            "citations_b":  [{"page": c.page_number, "valid": c.citation_valid} for c in citations_b],
        })
    return rows


@app.route("/compare", methods=["POST"])
def compare():
    if "policy_a" not in request.files or "policy_b" not in request.files:
        return jsonify({"error": "Both policy files are required."}), 400

    file_a = request.files["policy_a"]
    file_b = request.files["policy_b"]

    if not file_a.filename or not file_b.filename:
        return jsonify({"error": "Please select both PDF files."}), 400

    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "policy_a.pdf")
        path_b = os.path.join(tmp, "policy_b.pdf")
        file_a.save(path_a)
        file_b.save(path_b)

        try:
            config = load_config()
            result = compare_policies(path_a, path_b, config=config)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    rows = _build_rows(result)

    stats = {
        "total_clauses_a":   len(result.clauses_a),
        "total_clauses_b":   len(result.clauses_b),
        "total_changes":     len(result.changes),
        "added":             sum(1 for r in rows if r["change_type"] == "added"),
        "removed":           sum(1 for r in rows if r["change_type"] == "removed"),
        "substantive":       sum(1 for r in rows if r["change_type"] == "substantive"),
        "wording_only":      sum(1 for r in rows if r["change_type"] == "wording_only"),
        "unchanged":         sum(1 for r in rows if r["change_type"] == "unchanged"),
        "needs_review":      sum(1 for r in rows if r["needs_review"]),
        "doc_a_name":        file_a.filename,
        "doc_b_name":        file_b.filename,
        "errors":            result.errors,
        "warnings":          result.warnings,
        "processing_complete": result.processing_complete,
    }

    # Log to DB (user optional — works without login too)
    user = get_current_user()
    log_comparison(user["id"] if user else None, file_a.filename, file_b.filename, stats)

    return jsonify({"stats": stats, "rows": rows})


@app.route("/sample")
def sample():
    """Run comparison on the bundled Greenfield sample PDFs."""
    base   = Path(__file__).parent.parent / "sample_policies"
    path_a = str(base / "Greenfield_Policy_2023.pdf")
    path_b = str(base / "Greenfield_Policy_2024.pdf")

    try:
        config = load_config()
        result = compare_policies(path_a, path_b, config=config)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    rows  = _build_rows(result)
    stats = {
        "total_clauses_a":   len(result.clauses_a),
        "total_clauses_b":   len(result.clauses_b),
        "total_changes":     len(result.changes),
        "added":             sum(1 for r in rows if r["change_type"] == "added"),
        "removed":           sum(1 for r in rows if r["change_type"] == "removed"),
        "substantive":       sum(1 for r in rows if r["change_type"] == "substantive"),
        "wording_only":      sum(1 for r in rows if r["change_type"] == "wording_only"),
        "unchanged":         sum(1 for r in rows if r["change_type"] == "unchanged"),
        "needs_review":      sum(1 for r in rows if r["needs_review"]),
        "doc_a_name":        "Greenfield_Policy_2023.pdf",
        "doc_b_name":        "Greenfield_Policy_2024.pdf",
        "errors":            result.errors,
        "warnings":          result.warnings,
        "processing_complete": result.processing_complete,
    }
    return jsonify({"stats": stats, "rows": rows})


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import threading, time, webbrowser

    def _open_browser():
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")

    threading.Thread(target=_open_browser, daemon=True).start()
    print("\n" + "=" * 55)
    print("  POLICYX — Institution Policy Comparison Assistant")
    print("  http://localhost:5000")
    print("  Admin code: 200616")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000, use_reloader=False)
