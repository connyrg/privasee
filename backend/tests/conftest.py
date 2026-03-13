"""
Shared pytest fixtures for the PrivaSee backend test suite.

Fixture summary
---------------
create_pdf_with_text   helper function (not a fixture) — create a minimal PDF
                       with text inserted at caller-specified (x, y) positions.

test_pdf_bytes         single-page A4 PDF containing four PII strings at known
                       coordinates; used by masking and integration tests.

test_image_bytes       200×200 white PNG; used by masking tests that do not
                       need a multi-page PDF (avoids the poppler dependency).

sample_field_definitions
                       four FieldDefinition-compatible dicts matching the PII
                       text in test_pdf_bytes.

sample_entities        four Entity-compatible dicts with normalised bounding
                       boxes derived from the exact text positions in
                       test_pdf_bytes.

mock_session_manager   MagicMock(spec=UCSessionManager); all methods available,
                       no real UC / Databricks I/O.

override_databricks_dependency
                       Monkeypatches app.main._session_manager with
                       mock_session_manager and enables MOCK_DATABRICKS mode.
                       Returns the mock so callers can configure return values.

client                 httpx.AsyncClient backed by ASGITransport(app=app);
                       depends on override_databricks_dependency so every
                       API call hits the mock UCSessionManager.

PDF coordinate notes
--------------------
fitz (PyMuPDF) uses a top-left origin.  insert_text(point, text, fontsize=FS)
places the *baseline* of the text at `point`.  The visible bounding box is
therefore:

    top    = y - FS   (one fontsize above the baseline)
    bottom = y        (at the baseline)
    left   = x
    right  ≈ x + estimated_width

Normalised coordinates are divided by page width (W=595) and height (H=842).
"""

from __future__ import annotations

import io
from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import fitz  # PyMuPDF
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.session_manager import UCSessionManager
import app.main as main_module
from app.main import app


# ---------------------------------------------------------------------------
# Helper (not a fixture)
# ---------------------------------------------------------------------------

def create_pdf_with_text(text_items: List[Tuple[str, int, int]]) -> bytes:
    """
    Create a minimal single-page A4 PDF with text at known positions.

    Args:
        text_items: list of (text, x, y) tuples.
                    x, y are in PDF points from the top-left corner.
                    y is the text baseline position.

    Returns:
        Raw PDF bytes.
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4
    for text, x, y in text_items:
        page.insert_text((x, y), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# File fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_pdf_bytes() -> bytes:
    """
    A minimal single-page A4 PDF containing four PII strings at known positions.

    Text layout (baseline positions):
        "John Smith"              x=50,  y=100
        "john.smith@example.com"  x=50,  y=150
        "01/01/1990"              x=50,  y=200
        "123-45-6789"             x=50,  y=250

    These coordinates are mirrored in the sample_entities fixture so masking
    tests can verify redactions land on the correct regions.
    """
    return create_pdf_with_text([
        ("John Smith",             50, 100),
        ("john.smith@example.com", 50, 150),
        ("01/01/1990",             50, 200),
        ("123-45-6789",            50, 250),
    ])


@pytest.fixture
def test_image_bytes() -> bytes:
    """
    A 200×200 white PNG image for masking tests that do not require a PDF.

    Using a PNG avoids the poppler system dependency (pdf2image), making
    masking-service unit tests runnable in any CI environment.
    """
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_field_definitions() -> List[dict]:
    """
    Four FieldDefinition-compatible dicts corresponding to the PII in
    test_pdf_bytes.
    """
    return [
        {
            "name": "Full Name",
            "description": "Person's full name",
            "strategy": "Fake Data",
        },
        {
            "name": "Email",
            "description": "Email address",
            "strategy": "Black Out",
        },
        {
            "name": "Date of Birth",
            "description": "Date of birth",
            "strategy": "Entity Label",
        },
        {
            "name": "SSN",
            "description": "Social Security Number",
            "strategy": "Black Out",
        },
    ]


@pytest.fixture
def sample_entities() -> List[dict]:
    """
    Four Entity-compatible dicts with normalised bounding boxes that match
    the text positions in test_pdf_bytes.

    Coordinate derivation (A4 page: W=595pt, H=842pt, fontsize FS=12pt):

        text                      x   baseline-y  est_width  top = y - FS
        ─────────────────────────────────────────────────────────────────
        "John Smith"              50  100         80         88
        "john.smith@example.com"  50  150        155        138
        "01/01/1990"              50  200         75        188
        "123-45-6789"             50  250         80        238

    bounding_box = [x/W, top/H, width/W, FS/H]  (all values in [0, 1])
    """
    W, H, FS = 595, 842, 12

    rows = [
        # (id, text, x, baseline_y, est_width, field, replacement, strategy)
        ("entity-1", "John Smith",             50, 100,  80,  "Full Name",    "Jane Doe",    "fake_name"),
        ("entity-2", "john.smith@example.com", 50, 150, 155,  "Email",        "[REDACTED]",  "redact"),
        ("entity-3", "01/01/1990",             50, 200,  75,  "Date of Birth","[DATE OF BIRTH]", "entity_label"),
        ("entity-4", "123-45-6789",            50, 250,  80,  "SSN",          "[REDACTED]",  "redact"),
    ]

    entities = []
    for eid, text, x, y_base, w, field, replacement, strategy in rows:
        top = y_base - FS
        entities.append(
            {
                "id": eid,
                "entity_type": field,
                "original_text": text,
                "replacement_text": replacement,
                "bounding_box": [
                    round(x / W, 4),
                    round(top / H, 4),
                    round(w / W, 4),
                    round(FS / H, 4),
                ],
                "confidence": 0.95,
                "approved": True,
                "page_number": 1,
                "strategy": strategy,
            }
        )
    return entities


# ---------------------------------------------------------------------------
# SessionManager mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session_manager() -> MagicMock:
    """
    A MagicMock spec'd against SessionManager.

    All SessionManager methods are available as mock attributes.  Tests can
    configure return values, side effects, and assert call counts:

        mock_session_manager.get_session.return_value = some_session_data
        mock_session_manager.create_session.side_effect = Exception("boom")
    """
    mock = MagicMock(spec=UCSessionManager)
    mock.create_session.return_value = "mock-session-id-1234-5678"
    return mock


# ---------------------------------------------------------------------------
# Dependency override (module-level monkeypatch, not FastAPI Depends)
# ---------------------------------------------------------------------------

@pytest.fixture
def override_databricks_dependency(mock_session_manager, monkeypatch) -> MagicMock:
    """
    Swap app.main._session_manager for the mock and enable MOCK_DATABRICKS.

    _require_session_manager() reads the module-level _session_manager
    variable directly (it is NOT a FastAPI Depends()), so we use
    monkeypatch.setattr rather than app.dependency_overrides.

    MOCK_DATABRICKS is also set to True so that POST /api/process returns
    mock entities without attempting a real Databricks HTTP call.

    Returns the mock_session_manager so callers can configure it:

        def test_something(override_databricks_dependency):
            sm = override_databricks_dependency
            sm.get_session.return_value = ...
    """
    monkeypatch.setattr(main_module, "_session_manager", mock_session_manager)
    monkeypatch.setattr(main_module, "MOCK_DATABRICKS", True)
    return mock_session_manager


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(override_databricks_dependency) -> AsyncClient:
    """
    An async HTTP client wired directly to the FastAPI app via ASGITransport.

    Depends on override_databricks_dependency so every request hits the mock
    SessionManager and mock Databricks mode — no live services required.

    Usage:
        async def test_health(client):
            response = await client.get("/api/health")
            assert response.status_code == 200
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
