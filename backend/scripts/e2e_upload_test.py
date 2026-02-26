"""
E2E Upload Test — PrivaSee

Verifies that POST /api/upload stores the document in the Unity Catalog volume.
Runs two test cases back-to-back:
  1. Digital PDF  — text embedded as PDF text objects (machine-readable, no OCR needed)
  2. Scanned PDF  — text rasterised to an image then embedded in PDF (requires OCR)

Usage:
    cd backend
    python scripts/e2e_upload_test.py

Prerequisites:
    - .env file populated with DATABRICKS_HOST, DATABRICKS_TOKEN, UC_VOLUME_PATH
    - FastAPI server running (default: http://localhost:8000)
      cd backend && uvicorn app.main:app --reload
"""

from __future__ import annotations

import io
import json
import os
import sys

import fitz  # pymupdf
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
UC_VOLUME_PATH = os.getenv("UC_VOLUME_PATH", "").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

FILES_API = "/api/2.0/fs/files"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(condition: bool, msg: str) -> None:
    if not condition:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)
    print(f"  OK   {msg}")


def _db_headers() -> dict:
    return {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}


def _db_url(uc_path: str) -> str:
    return f"{DATABRICKS_HOST}{FILES_API}{uc_path}"


def _uc_path(session_id: str, filename: str) -> str:
    return f"{UC_VOLUME_PATH}/{session_id}/{filename}"


# ---------------------------------------------------------------------------
# PDF factories
# ---------------------------------------------------------------------------

def make_digital_pdf() -> bytes:
    """
    Digital PDF: text is embedded as real PDF text objects.
    Text is directly selectable/extractable — no OCR required.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    page.insert_text((72, 100), "PrivaSee E2E upload test — digital document")
    page.insert_text((72, 130), "Name: Jane Doe")
    page.insert_text((72, 160), "Email: jane@example.com")
    page.insert_text((72, 190), "Phone: +44 7700 900123")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def make_scanned_pdf() -> bytes:
    """
    Scanned PDF: text is first rasterised to a pixel image, then that image
    is embedded as the sole content of a PDF page — there is no text layer.
    Extracting text requires OCR (as with a real scanner output).
    """
    # Step 1 — render text onto a pixmap (simulate a scanned page)
    tmp = fitz.open()
    tmp_page = tmp.new_page(width=595, height=842)
    tmp_page.insert_text((72, 100), "PrivaSee E2E upload test — scanned document")
    tmp_page.insert_text((72, 130), "Name: John Smith")
    tmp_page.insert_text((72, 160), "Email: john@example.com")
    tmp_page.insert_text((72, 190), "Phone: +44 7700 900456")
    pix = tmp_page.get_pixmap(dpi=150)
    tmp.close()

    # Step 2 — embed the pixmap as an image-only PDF page (no text layer)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# Core test steps (reused for both test cases)
# ---------------------------------------------------------------------------

def step_upload(pdf_bytes: bytes, filename: str, label: str) -> str:
    print(f"  POST /api/upload ({label}) ...")
    resp = requests.post(
        f"{API_BASE_URL}/api/upload",
        files={"file": (filename, io.BytesIO(pdf_bytes), "application/pdf")},
        timeout=30,
    )
    _check(resp.status_code == 200, f"HTTP {resp.status_code} (expected 200) — body: {resp.text[:200]}")
    body = resp.json()
    session_id = body.get("session_id", "")
    _check(bool(session_id), f"session_id present: {session_id}")
    _check(body.get("filename") == filename, f"filename={body.get('filename')}")
    _check(body.get("file_size") == len(pdf_bytes), f"file_size={body.get('file_size')}")
    print(f"  INFO session_id={session_id}")
    return session_id


def step_verify_file(session_id: str, pdf_bytes: bytes) -> None:
    print("  Verifying original.pdf in UC volume ...")
    url = _db_url(_uc_path(session_id, "original.pdf"))
    resp = requests.get(url, headers=_db_headers(), timeout=30)
    _check(resp.status_code == 200, f"GET original.pdf → HTTP {resp.status_code}")
    _check(resp.content == pdf_bytes, f"File bytes match ({len(resp.content)} bytes)")


def step_verify_metadata(session_id: str) -> None:
    print("  Verifying metadata.json in UC volume ...")
    url = _db_url(_uc_path(session_id, "metadata.json"))
    resp = requests.get(url, headers=_db_headers(), timeout=30)
    _check(resp.status_code == 200, f"GET metadata.json → HTTP {resp.status_code}")
    try:
        meta = resp.json()
    except json.JSONDecodeError:
        _check(False, f"metadata.json is not valid JSON: {resp.text[:200]}")
        return
    _check(meta.get("status") == "uploaded", f"status={meta.get('status')}")
    _check("session_id" in meta, "session_id field present")


def step_cleanup(session_id: str) -> None:
    print("  Cleaning up session artefacts from UC ...")
    for fname in ("original.pdf", "metadata.json"):
        url = _db_url(_uc_path(session_id, fname))
        resp = requests.delete(url, headers=_db_headers(), timeout=30)
        _check(resp.status_code in (200, 204, 404), f"DELETE {fname} → HTTP {resp.status_code}")


def run_test_case(label: str, pdf_bytes: bytes, filename: str) -> bool:
    """
    Run a single upload test case. Returns True on pass, False on fail.
    Artefacts are left in UC on failure so they can be inspected.
    """
    print(f"\n{'─' * 60}")
    print(f"Test case: {label}")
    print(f"{'─' * 60}")
    session_id = None
    try:
        _check(len(pdf_bytes) > 0, f"PDF built ({len(pdf_bytes)} bytes)")
        session_id = step_upload(pdf_bytes, filename, label)
        step_verify_file(session_id, pdf_bytes)
        step_verify_metadata(session_id)
        step_cleanup(session_id)
        print(f"  ✅ PASSED")
        return True
    except AssertionError:
        if session_id:
            print(f"  INFO: artefacts left in UC — session_id={session_id}")
            print(f"  Path: {UC_VOLUME_PATH}/{session_id}/")
        print(f"  ❌ FAILED")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Preflight
    missing = [v for v in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "UC_VOLUME_PATH") if not os.getenv(v)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}\nCopy .env.template → .env and fill in values.")

    print(f"API_BASE_URL  : {API_BASE_URL}")
    print(f"UC_VOLUME_PATH: {UC_VOLUME_PATH}")

    results: list[tuple[str, bool]] = []

    try:
        results.append(("Digital PDF", run_test_case(
            label="Digital PDF (text layer — no OCR needed)",
            pdf_bytes=make_digital_pdf(),
            filename="e2e_digital.pdf",
        )))
        results.append(("Scanned PDF", run_test_case(
            label="Scanned PDF (image-only — OCR required)",
            pdf_bytes=make_scanned_pdf(),
            filename="e2e_scanned.pdf",
        )))
    except requests.ConnectionError:
        sys.exit(f"\nCould not connect to {API_BASE_URL} — is the server running?")

    # Summary
    print(f"\n{'═' * 60}")
    print("Summary")
    print(f"{'═' * 60}")
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}  {name}")

    if not all(ok for _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
