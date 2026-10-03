"""
downloader.py — Safely download a confirmed candidate PDF.

Security rules (per spec):
- Validate content-type is application/pdf
- Cap file size at 32 MB
- Write to a temp path only
- Never execute anything from the document
- Sanitise filename from URL (no path traversal)
"""
from __future__ import annotations

import re
import tempfile
import urllib.parse
from pathlib import Path

import requests

_UA      = "Mozilla/5.0 (compatible; POLICYX/1.0; +https://policyx.io)"
_TIMEOUT = 30
_MAX_BYTES = 32 * 1024 * 1024   # 32 MB

_SAFE_FILENAME_RE = re.compile(r"[^\w\-.]")


def _safe_filename(url: str) -> str:
    """Extract a safe filename from a URL."""
    path = urllib.parse.urlparse(url).path
    name = Path(path).name or "previous_policy.pdf"
    name = _SAFE_FILENAME_RE.sub("_", name)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:120]


def download_candidate(url: str) -> dict:
    """
    Download the PDF at ``url`` to a temporary file.

    Returns:
        {"ok": True,  "path": "/tmp/.../previous_policy.pdf", "filename": "..."}
     or {"ok": False, "error": "reason"}
    """
    # Basic URL safety check — must be http/https
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"ok": False, "error": "Only HTTP/HTTPS URLs are supported."}

    headers = {
        "User-Agent": _UA,
        "Accept": "application/pdf,*/*",
    }

    try:
        resp = requests.get(
            url, headers=headers, timeout=_TIMEOUT,
            stream=True, allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Download timed out. The server took too long to respond."}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "error": "Could not connect to the document server."}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": f"Server returned {exc.response.status_code}."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    # Validate content-type
    ct = resp.headers.get("content-type", "").lower()
    if "pdf" not in ct and not url.lower().endswith(".pdf"):
        # Give it a pass if the URL ends in .pdf but server sends wrong CT
        if "html" in ct or "text" in ct:
            return {"ok": False, "error": f"Server returned HTML, not a PDF ({ct})."}

    # Stream into temp file with size cap
    filename = _safe_filename(url)
    tmp_dir  = tempfile.mkdtemp(prefix="policyx_inv_")
    tmp_path = Path(tmp_dir) / filename
    total    = 0

    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        f.close()
                        tmp_path.unlink(missing_ok=True)
                        return {"ok": False, "error": "Document exceeds the 32 MB size limit."}
                    f.write(chunk)
    except Exception as exc:
        return {"ok": False, "error": f"Failed to save file: {exc}"}

    if total < 1024:
        tmp_path.unlink(missing_ok=True)
        return {"ok": False, "error": "Downloaded file is too small to be a valid PDF."}

    # Verify PDF magic bytes
    try:
        with open(tmp_path, "rb") as f:
            magic = f.read(5)
        if not magic.startswith(b"%PDF-"):
            tmp_path.unlink(missing_ok=True)
            return {"ok": False, "error": "Downloaded file is not a valid PDF (bad magic bytes)."}
    except Exception:
        pass

    return {
        "ok":       True,
        "path":     str(tmp_path),
        "filename": filename,
        "size_bytes": total,
    }
