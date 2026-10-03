"""
database.py — SQLite3 database module for POLICYX auth system.
Handles user registration, login, session management.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "policyx.db"

# Admin access code — change this to your preferred code
ADMIN_CODE = os.environ.get("POLICYX_ADMIN_CODE", "200616")


def get_db() -> sqlite3.Connection:
    """Return a database connection with row_factory set."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name  TEXT    NOT NULL,
                last_name   TEXT    NOT NULL,
                email       TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                institution TEXT    NOT NULL DEFAULT '',
                password_hash TEXT  NOT NULL,
                role        TEXT    NOT NULL DEFAULT 'user',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                last_login  TEXT
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token       TEXT    PRIMARY KEY,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
                expires_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS comparison_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
                doc_a_name  TEXT,
                doc_b_name  TEXT,
                total_changes INTEGER DEFAULT 0,
                substantive   INTEGER DEFAULT 0,
                added         INTEGER DEFAULT 0,
                removed       INTEGER DEFAULT 0,
                wording_only  INTEGER DEFAULT 0,
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
    print(f"[DB] Initialised database at {DB_PATH}")


# ──────────────────────────────────────────────
# Password helpers
# ──────────────────────────────────────────────

def _hash_password(password: str, salt: str | None = None) -> str:
    """Return 'salt$hash' string using PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260_000)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a password against the stored 'salt$hash' string."""
    try:
        salt, _ = stored.split("$", 1)
        return secrets.compare_digest(_hash_password(password, salt), stored)
    except Exception:
        return False


# ──────────────────────────────────────────────
# User operations
# ──────────────────────────────────────────────

def register_user(first_name: str, last_name: str, email: str,
                  institution: str, password: str) -> dict:
    """
    Register a new user.
    Returns {'ok': True, 'user_id': int} or {'ok': False, 'error': str}.
    """
    if len(password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters."}

    password_hash = _hash_password(password)
    try:
        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO users (first_name, last_name, email, institution, password_hash)
                   VALUES (?, ?, ?, ?, ?)""",
                (first_name.strip(), last_name.strip(), email.strip().lower(),
                 institution.strip(), password_hash),
            )
            return {"ok": True, "user_id": cur.lastrowid}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "An account with this email already exists."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def login_user(email: str, password: str) -> dict:
    """
    Authenticate a user and create a session token.
    Returns {'ok': True, 'token': str, 'user': dict} or {'ok': False, 'error': str}.
    """
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()

        if row is None or not _verify_password(password, row["password_hash"]):
            return {"ok": False, "error": "Invalid email or password."}

        # Create session token (valid for 7 days)
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat()
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, row["id"], expires_at),
        )
        # Update last login
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), row["id"]),
        )

        return {
            "ok": True,
            "token": token,
            "user": {
                "id": row["id"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "email": row["email"],
                "institution": row["institution"],
                "role": row["role"],
            },
        }


def get_user_by_token(token: str) -> dict | None:
    """Return the user dict for a valid session token, or None if invalid/expired."""
    if not token:
        return None
    with get_db() as conn:
        row = conn.execute(
            """SELECT u.* FROM users u
               JOIN sessions s ON s.user_id = u.id
               WHERE s.token = ? AND s.expires_at > datetime('now')""",
            (token,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def logout_user(token: str) -> None:
    """Delete a session token."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


# ──────────────────────────────────────────────
# Admin helpers
# ──────────────────────────────────────────────

def verify_admin_code(code: str) -> bool:
    """Check if the supplied code matches the admin access code."""
    return secrets.compare_digest(code.strip(), ADMIN_CODE)


def get_all_users() -> list[dict]:
    """Return all users (admin use only)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, first_name, last_name, email, institution, role, created_at, last_login FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_comparison_stats() -> dict:
    """Return aggregate stats for the admin dashboard."""
    with get_db() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_comparisons = conn.execute("SELECT COUNT(*) FROM comparison_log").fetchone()[0]
        total_institutions = conn.execute("SELECT COUNT(DISTINCT institution) FROM users WHERE institution != ''").fetchone()[0]
        active_today = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM sessions WHERE created_at >= date('now')"
        ).fetchone()[0]
        recent = conn.execute(
            """SELECT cl.*, u.first_name || ' ' || u.last_name AS user_name, u.institution
               FROM comparison_log cl
               LEFT JOIN users u ON u.id = cl.user_id
               ORDER BY cl.created_at DESC LIMIT 10"""
        ).fetchall()
    return {
        "total_users": total_users,
        "total_comparisons": total_comparisons,
        "total_institutions": total_institutions,
        "active_today": active_today,
        "recent_comparisons": [dict(r) for r in recent],
    }


def log_comparison(user_id: int | None, doc_a: str, doc_b: str, stats: dict) -> None:
    """Log a completed comparison to the database."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO comparison_log
               (user_id, doc_a_name, doc_b_name, total_changes, substantive, added, removed, wording_only)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, doc_a, doc_b,
             stats.get("total_changes", 0), stats.get("substantive", 0),
             stats.get("added", 0), stats.get("removed", 0), stats.get("wording_only", 0)),
        )


# Auto-init on import
init_db()
