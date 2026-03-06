"""
Unit tests for MaskingService.apply_pdf_masks.

All tests use real PyMuPDF operations — fitz is NOT mocked.  Test PDFs are
created with the create_pdf_with_text helper from conftest.py so the exact
text positions are known in advance.

Coordinate system (A4 page: W=595 pt, H=842 pt, fontsize FS=12 pt)
-------------------------------------------------------------------
insert_text(point, text, fontsize=12) places the text baseline at *point*.

For text at baseline (x, y):
    visible top    = y - FS  (one fontsize above baseline)
    visible bottom = y
    visible left   = x
    visible right  ≈ x + estimated_width

The redaction rect used in each test is chosen to fully cover the visible
glyph area while being easy to reason about.  It is deliberately a little
generous (a few extra points on each edge) so that floating-point layout
differences between fitz versions do not cause flakes.
"""

from __future__ import annotations

import re

import fitz
import pytest

from app.services.masking_service import MaskingService
from tests.conftest import create_pdf_with_text

# ---------------------------------------------------------------------------
# Page / font constants (must match the create_pdf_with_text helper)
# ---------------------------------------------------------------------------

W: float = 595.0   # A4 width in points
H: float = 842.0   # A4 height in points
FS: int = 12       # fontsize used by create_pdf_with_text

# Tolerance (points) for the filled-rectangle position check
_RECT_TOLERANCE = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> MaskingService:
    return MaskingService()


def _bbox(x: float, baseline_y: float, width: float) -> list[float]:
    """
    Return a normalised [x, y, w, h] bounding box that covers text placed
    by insert_text at (x, baseline_y) with the module-level fontsize.

    A 3-point vertical pad on both sides keeps tests stable across
    minor fitz layout variations.
    """
    top = baseline_y - FS - 3
    return [x / W, top / H, width / W, (FS + 6) / H]


def _redact_entity(
    entity_type: str,
    original_text: str,
    x: float,
    baseline_y: float,
    width: float,
    strategy: str,
    replacement_text: str = "",
    approved: bool = True,
    page_number: int = 1,
) -> dict:
    return {
        "entity_type": entity_type,
        "original_text": original_text,
        "replacement_text": replacement_text,
        "bounding_box": _bbox(x, baseline_y, width),
        "strategy": strategy,
        "approved": approved,
        "page_number": page_number,
    }


def _apply(
    texts: list[tuple[str, int, int]],
    entities: list[dict],
) -> fitz.Document:
    """Create a PDF, apply masks, return the resulting open fitz.Document."""
    svc = _make_service()
    pdf_bytes = create_pdf_with_text(texts)
    result_bytes = svc.apply_pdf_masks(pdf_bytes, entities)
    return fitz.open(stream=result_bytes, filetype="pdf")


# ===========================================================================
# Group 1 — Black Out strategy
# ===========================================================================


@pytest.mark.unit
def test_black_out_removes_text_from_output():
    """Original text must not appear anywhere in the page's text layer."""
    entity = _redact_entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
    doc = _apply([("John Smith", 50, 100)], [entity])
    page_text = doc[0].get_text()
    doc.close()

    assert "John Smith" not in page_text


@pytest.mark.unit
def test_black_out_draws_filled_rectangle():
    """
    After redaction the page drawings must contain a filled black rectangle
    positioned within _RECT_TOLERANCE points of the annotation rect.
    """
    entity = _redact_entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
    doc = _apply([("John Smith", 50, 100)], [entity])
    page = doc[0]

    # Expected rect in page-unit coordinates
    x0_exp = 50.0
    y0_exp = 100.0 - FS - 3   # top of the bbox (see _bbox())

    black_rects = [
        d["rect"]
        for d in page.get_drawings()
        if d.get("fill") == (0.0, 0.0, 0.0)
    ]
    doc.close()

    assert black_rects, "No black-filled rectangle found in page drawings"
    near = any(
        abs(r.x0 - x0_exp) <= _RECT_TOLERANCE
        and abs(r.y0 - y0_exp) <= _RECT_TOLERANCE
        for r in black_rects
    )
    assert near, (
        f"No black rect within {_RECT_TOLERANCE} pt of ({x0_exp}, {y0_exp}); "
        f"found: {black_rects}"
    )


@pytest.mark.unit
def test_black_out_does_not_affect_other_text():
    """Only the targeted text is removed; unrelated text is fully preserved."""
    entity = _redact_entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
    doc = _apply(
        [("John Smith", 50, 100), ("keep this safe", 50, 200)],
        [entity],
    )
    page_text = doc[0].get_text()
    doc.close()

    assert "John Smith" not in page_text
    assert "keep this safe" in page_text


# ===========================================================================
# Group 2 — Fake Data strategy
# ===========================================================================


@pytest.mark.unit
def test_fake_data_removes_original_text():
    """Original text must not appear in the output when Fake Data is applied."""
    entity = _redact_entity(
        "Full Name", "John Smith", 50, 100, 90,
        strategy="Fake Data", replacement_text="Jane Doe",
    )
    doc = _apply([("John Smith", 50, 100)], [entity])
    page_text = doc[0].get_text()
    doc.close()

    assert "John Smith" not in page_text


@pytest.mark.unit
def test_fake_data_inserts_replacement_text():
    """The replacement_text value must appear in the masked output's text layer."""
    entity = _redact_entity(
        "Full Name", "John Smith", 50, 100, 90,
        strategy="Fake Data", replacement_text="Jane Doe",
    )
    doc = _apply([("John Smith", 50, 100)], [entity])
    page_text = doc[0].get_text()
    doc.close()

    assert "Jane Doe" in page_text


@pytest.mark.unit
def test_fake_data_uses_consistent_replacement():
    """
    Two entities with identical original_text must receive identical
    replacement text — this mirrors the guarantee that MappingManager provides.
    """
    original = "John Smith"
    replacement = "Jane Doe"

    # Both entities reference the same original_text but different bounding boxes
    entities = [
        _redact_entity(
            "Full Name", original, 50, 100, 90,
            strategy="Fake Data", replacement_text=replacement,
        ),
        _redact_entity(
            "Full Name", original, 50, 200, 90,
            strategy="Fake Data", replacement_text="Should Be Overridden",
        ),
    ]

    pdf_bytes = create_pdf_with_text([(original, 50, 100), (original, 50, 200)])
    svc = _make_service()
    result_bytes = svc.apply_pdf_masks(pdf_bytes, entities)
    doc = fitz.open(stream=result_bytes, filetype="pdf")
    page_text = doc[0].get_text()
    doc.close()

    # The replacement used for the *first* entity must win for both occurrences
    assert page_text.count("Jane Doe") == 2
    assert "Should Be Overridden" not in page_text


# ===========================================================================
# Group 3 — Entity Label strategy
# ===========================================================================


@pytest.mark.unit
def test_entity_label_uses_type_and_counter_format():
    """
    The generated label must match the pattern ``{EntityType}_{N}`` where
    spaces/hyphens are replaced with underscores and N starts at 1.
    """
    entity = _redact_entity(
        "Full Name", "John Smith", 50, 100, 90, strategy="Entity Label",
    )
    doc = _apply([("John Smith", 50, 100)], [entity])
    page_text = doc[0].get_text()
    doc.close()

    pattern = re.compile(r"Full_Name_[A-Z\d]+")
    assert pattern.search(page_text), (
        f"Expected a label matching 'Full_Name_A' in output text; got: {page_text!r}"
    )


@pytest.mark.unit
def test_entity_label_increments_counter_per_type():
    """
    Two distinct Full Name entities must produce Full_Name_A and Full_Name_B,
    not the same label twice.
    """
    entities = [
        _redact_entity(
            "Full Name", "John Smith", 50, 100, 90, strategy="Entity Label",
        ),
        _redact_entity(
            "Full Name", "Jane Doe", 50, 200, 90, strategy="Entity Label",
        ),
    ]
    doc = _apply(
        [("John Smith", 50, 100), ("Jane Doe", 50, 200)],
        entities,
    )
    page_text = doc[0].get_text()
    doc.close()

    assert "Full_Name_A" in page_text, f"Expected Full_Name_A in: {page_text!r}"
    assert "Full_Name_B" in page_text, f"Expected Full_Name_B in: {page_text!r}"
    assert page_text.count("Full_Name_A") == 1, "Full_Name_A should appear exactly once"
    assert page_text.count("Full_Name_B") == 1, "Full_Name_B should appear exactly once"


# ===========================================================================
# Group 4 — Edge cases
# ===========================================================================


@pytest.mark.unit
def test_skips_entity_with_approved_false():
    """Entities flagged approved=False must not be redacted at all."""
    entity = _redact_entity(
        "Full Name", "John Smith", 50, 100, 90,
        strategy="Black Out", approved=False,
    )
    doc = _apply([("John Smith", 50, 100)], [entity])
    page_text = doc[0].get_text()
    doc.close()

    # Text must be completely untouched
    assert "John Smith" in page_text, (
        "Text with approved=False should remain in the output"
    )


@pytest.mark.unit
def test_handles_empty_bounding_boxes_without_crashing():
    """
    An entity with bounding_box=[] and no bounding_boxes must be silently
    skipped — no exception should be raised.
    """
    entity = {
        "entity_type": "Full Name",
        "original_text": "John Smith",
        "replacement_text": "",
        "bounding_box": [],
        "strategy": "Black Out",
        "approved": True,
        "page_number": 1,
    }
    # Must not raise; text is left intact because the entity has no valid box
    doc = _apply([("John Smith", 50, 100)], [entity])
    doc.close()


@pytest.mark.unit
def test_handles_multi_box_entity():
    """
    An entity with bounding_boxes (plural) must have every listed box
    redacted in the output.
    """
    # Simulate a name split across two bounding regions
    bbox1 = _bbox(50, 100, 45)    # "John" on one line
    bbox2 = _bbox(50, 120, 45)    # "Smith" on the next

    entity = {
        "entity_type": "Full Name",
        "original_text": "John",
        "replacement_text": "",
        "bounding_boxes": [bbox1, bbox2],
        "strategy": "Black Out",
        "approved": True,
        "page_number": 1,
    }

    doc = _apply([("John", 50, 100), ("Smith", 50, 120)], [entity])
    page = doc[0]

    # Both boxes should have produced black filled drawings
    black_rects = [
        d["rect"]
        for d in page.get_drawings()
        if d.get("fill") == (0.0, 0.0, 0.0)
    ]
    doc.close()

    assert len(black_rects) >= 2, (
        f"Expected at least 2 black rects for a multi-box entity; got {len(black_rects)}"
    )


@pytest.mark.unit
def test_returns_valid_pdf_bytes():
    """
    Output bytes must be parseable as a valid PDF by fitz.open() — a broken
    output would crash the browser preview.
    """
    entity = _redact_entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
    svc = _make_service()
    pdf_bytes = create_pdf_with_text([("John Smith", 50, 100)])
    result_bytes = svc.apply_pdf_masks(pdf_bytes, [entity])

    assert result_bytes[:4] == b"%PDF", "Output does not start with PDF magic bytes"

    # fitz.open() raises if the stream is malformed
    try:
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        page_count = len(doc)
        doc.close()
    except Exception as exc:
        pytest.fail(f"fitz.open() raised on masked output: {exc}")

    assert page_count >= 1


@pytest.mark.unit
def test_handles_single_page_pdf():
    """
    The service must work correctly on the common single-page document case.
    """
    entity = _redact_entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
    svc = _make_service()
    pdf_bytes = create_pdf_with_text([("John Smith", 50, 100)])
    result_bytes = svc.apply_pdf_masks(pdf_bytes, [entity])

    doc = fitz.open(stream=result_bytes, filetype="pdf")
    assert len(doc) == 1, f"Expected 1 page, got {len(doc)}"
    page_text = doc[0].get_text()
    doc.close()

    assert "John Smith" not in page_text
