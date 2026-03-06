"""
Unit tests for databricks.model.MaskingService

Uses real PyMuPDF operations — fitz is NOT mocked.  Test PDFs are created
with the create_pdf_with_text helper so the exact text positions are known.

Mirrors the structure of backend/tests/unit/test_masking_service.py.
"""

import io
import re
import unittest

import fitz

from databricks.model.masking_service import MaskingService

# ---------------------------------------------------------------------------
# Page / font constants
# ---------------------------------------------------------------------------

W: float = 595.0   # A4 width in points
H: float = 842.0   # A4 height in points
FS: int = 12       # fontsize used by create_pdf_with_text

_RECT_TOLERANCE = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_pdf_with_text(text_items: list) -> bytes:
    """
    Create a minimal single-page A4 PDF with text inserted at given positions.

    Args:
        text_items: list of (text, x, baseline_y) tuples

    Returns:
        PDF bytes
    """
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    for text, x, y in text_items:
        page.insert_text((x, y), text, fontsize=FS)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _bbox(x: float, baseline_y: float, width: float) -> list:
    """Normalised [x, y, w, h] covering text at (x, baseline_y) with FS fontsize."""
    top = baseline_y - FS - 3
    return [x / W, top / H, width / W, (FS + 6) / H]


def _entity(
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


def _apply(text_items: list, entities: list) -> fitz.Document:
    """Create a PDF, apply masks, return the resulting fitz.Document."""
    svc = MaskingService()
    pdf_bytes = create_pdf_with_text(text_items)
    result_bytes = svc.apply_pdf_masks(pdf_bytes, entities)
    return fitz.open(stream=result_bytes, filetype="pdf")


# ===========================================================================
# Black Out strategy
# ===========================================================================

class TestBlackOut(unittest.TestCase):

    def test_removes_text_from_output(self):
        """Original text must not appear in the page text layer."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        doc = _apply([("John Smith", 50, 100)], [entity])
        page_text = doc[0].get_text()
        doc.close()
        self.assertNotIn("John Smith", page_text)

    def test_draws_black_filled_rectangle(self):
        """A black-filled rectangle must appear at the entity location."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        doc = _apply([("John Smith", 50, 100)], [entity])
        page = doc[0]

        x0_exp = 50.0
        y0_exp = 100.0 - FS - 3

        black_rects = [
            d["rect"]
            for d in page.get_drawings()
            if d.get("fill") == (0.0, 0.0, 0.0)
        ]
        doc.close()

        self.assertTrue(black_rects, "No black-filled rectangle found")
        near = any(
            abs(r.x0 - x0_exp) <= _RECT_TOLERANCE
            and abs(r.y0 - y0_exp) <= _RECT_TOLERANCE
            for r in black_rects
        )
        self.assertTrue(near, f"No black rect near ({x0_exp}, {y0_exp}); found: {black_rects}")

    def test_does_not_affect_other_text(self):
        """Unrelated text must be preserved."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        doc = _apply([("John Smith", 50, 100), ("keep this safe", 50, 200)], [entity])
        page_text = doc[0].get_text()
        doc.close()
        self.assertNotIn("John Smith", page_text)
        self.assertIn("keep this safe", page_text)


# ===========================================================================
# Fake Data strategy
# ===========================================================================

class TestFakeData(unittest.TestCase):

    def test_removes_original_text(self):
        entity = _entity(
            "Full Name", "John Smith", 50, 100, 90,
            strategy="Fake Data", replacement_text="Jane Doe",
        )
        doc = _apply([("John Smith", 50, 100)], [entity])
        page_text = doc[0].get_text()
        doc.close()
        self.assertNotIn("John Smith", page_text)

    def test_inserts_replacement_text(self):
        entity = _entity(
            "Full Name", "John Smith", 50, 100, 90,
            strategy="Fake Data", replacement_text="Jane Doe",
        )
        doc = _apply([("John Smith", 50, 100)], [entity])
        page_text = doc[0].get_text()
        doc.close()
        self.assertIn("Jane Doe", page_text)

    def test_consistent_replacement_for_same_original(self):
        """Two entities with the same original_text must use the same replacement."""
        entities = [
            _entity("Full Name", "John Smith", 50, 100, 90,
                    strategy="Fake Data", replacement_text="Jane Doe"),
            _entity("Full Name", "John Smith", 50, 200, 90,
                    strategy="Fake Data", replacement_text="Should Be Overridden"),
        ]
        pdf_bytes = create_pdf_with_text([("John Smith", 50, 100), ("John Smith", 50, 200)])
        svc = MaskingService()
        result_bytes = svc.apply_pdf_masks(pdf_bytes, entities)
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        page_text = doc[0].get_text()
        doc.close()
        self.assertEqual(page_text.count("Jane Doe"), 2)
        self.assertNotIn("Should Be Overridden", page_text)


# ===========================================================================
# Entity Label strategy
# ===========================================================================

class TestEntityLabel(unittest.TestCase):

    def test_uses_type_and_counter_format(self):
        """Label must match {EntityType}_{Letter} e.g. Full_Name_A."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90, strategy="Entity Label")
        doc = _apply([("John Smith", 50, 100)], [entity])
        page_text = doc[0].get_text()
        doc.close()
        self.assertRegex(page_text, r"Full_Name_[A-Z\d]+")

    def test_increments_counter_per_type(self):
        """Two distinct entities of the same type get _A and _B."""
        entities = [
            _entity("Full Name", "John Smith", 50, 100, 90, strategy="Entity Label"),
            _entity("Full Name", "Jane Doe",   50, 200, 90, strategy="Entity Label"),
        ]
        doc = _apply([("John Smith", 50, 100), ("Jane Doe", 50, 200)], entities)
        page_text = doc[0].get_text()
        doc.close()
        self.assertIn("Full_Name_A", page_text)
        self.assertIn("Full_Name_B", page_text)
        self.assertEqual(page_text.count("Full_Name_A"), 1)
        self.assertEqual(page_text.count("Full_Name_B"), 1)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_skips_unapproved_entity(self):
        """Entities with approved=False must not be redacted."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90,
                         strategy="Black Out", approved=False)
        doc = _apply([("John Smith", 50, 100)], [entity])
        page_text = doc[0].get_text()
        doc.close()
        self.assertIn("John Smith", page_text)

    def test_handles_empty_bounding_box_without_crashing(self):
        """An entity with bounding_box=[] must be silently skipped."""
        entity = {
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "replacement_text": "",
            "bounding_box": [],
            "strategy": "Black Out",
            "approved": True,
            "page_number": 1,
        }
        doc = _apply([("John Smith", 50, 100)], [entity])
        doc.close()  # must not raise

    def test_handles_multi_box_entity(self):
        """An entity with bounding_boxes (plural) redacts every listed box."""
        bbox1 = _bbox(50, 100, 45)
        bbox2 = _bbox(50, 120, 45)
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
        black_rects = [d["rect"] for d in page.get_drawings() if d.get("fill") == (0.0, 0.0, 0.0)]
        doc.close()
        self.assertGreaterEqual(len(black_rects), 2, "Expected at least 2 black rects for multi-box entity")

    def test_returns_valid_pdf_bytes(self):
        """Output must be parseable as valid PDF."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        svc = MaskingService()
        pdf_bytes = create_pdf_with_text([("John Smith", 50, 100)])
        result_bytes = svc.apply_pdf_masks(pdf_bytes, [entity])

        self.assertEqual(result_bytes[:4], b"%PDF")
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertGreaterEqual(len(doc), 1)
        doc.close()

    def test_empty_entities_list_returns_unchanged_pdf(self):
        """apply_pdf_masks with no entities must return a valid PDF."""
        svc = MaskingService()
        pdf_bytes = create_pdf_with_text([("Hello World", 50, 100)])
        result_bytes = svc.apply_pdf_masks(pdf_bytes, [])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertIn("Hello World", doc[0].get_text())
        doc.close()


if __name__ == "__main__":
    unittest.main()
