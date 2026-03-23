"""
Unit tests for databricks.model.MaskingService

Uses real PyMuPDF operations — fitz is NOT mocked.  Test PDFs are created
with the create_pdf_with_text helper so the exact text positions are known.

Entity format (canonical):
    occurrences[].page_number, occurrences[].original_text,
    occurrences[].bounding_boxes  ([[x, y, w, h], ...], normalised 0–1)
No entity-level bounding_box, bounding_boxes, or page_number fields.
"""

import io
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
    """Create a minimal single-page A4 PDF with text at given positions."""
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
    occ_text: str = None,
) -> dict:
    """Build a canonical entity dict with a single occurrence."""
    return {
        "entity_type": entity_type,
        "original_text": original_text,
        "replacement_text": replacement_text,
        "strategy": strategy,
        "approved": approved,
        "occurrences": [
            {
                "page_number": page_number,
                "original_text": occ_text if occ_text is not None else original_text,
                "bounding_boxes": [_bbox(x, baseline_y, width)],
            }
        ],
    }


def _entity_multi_occ(
    entity_type: str,
    original_text: str,
    strategy: str,
    replacement_text: str = "",
    approved: bool = True,
    occurrences: list = None,
) -> dict:
    """Build a canonical entity dict with multiple occurrences."""
    return {
        "entity_type": entity_type,
        "original_text": original_text,
        "replacement_text": replacement_text,
        "strategy": strategy,
        "approved": approved,
        "occurrences": occurrences or [],
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

    def test_consistent_replacement_across_occurrences(self):
        """Two occurrences of the same entity must use the same replacement."""
        entity = _entity_multi_occ(
            "Full Name", "John Smith", "Fake Data",
            replacement_text="Jane Doe",
            occurrences=[
                {"page_number": 1, "original_text": "John Smith",
                 "bounding_boxes": [_bbox(50, 100, 90)]},
                {"page_number": 1, "original_text": "John Smith",
                 "bounding_boxes": [_bbox(50, 200, 90)]},
            ],
        )
        pdf_bytes = create_pdf_with_text([("John Smith", 50, 100), ("John Smith", 50, 200)])
        svc = MaskingService()
        result_bytes = svc.apply_pdf_masks(pdf_bytes, [entity])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        page_text = doc[0].get_text()
        doc.close()
        self.assertEqual(page_text.count("Jane Doe"), 2)

    def test_occurrence_text_token_aligned_to_full_name(self):
        """An occurrence where original_text is a first name derives the replacement
        slice from the entity's full replacement_text via token alignment."""
        entity = _entity_multi_occ(
            "Full Name", "John Smith", "Fake Data",
            replacement_text="Jane Doe",
            occurrences=[
                {"page_number": 1, "original_text": "John Smith",
                 "bounding_boxes": [_bbox(50, 100, 90)]},
                {"page_number": 1, "original_text": "John",
                 "bounding_boxes": [_bbox(50, 200, 40)]},
            ],
        )
        pdf_bytes = create_pdf_with_text([("John Smith", 50, 100), ("John", 50, 200)])
        svc = MaskingService()
        result_bytes = svc.apply_pdf_masks(pdf_bytes, [entity])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        page_text = doc[0].get_text()
        doc.close()
        # Full name occurrence → "Jane Doe"
        self.assertIn("Jane Doe", page_text)
        # First name occurrence → "Jane" (first token slice)
        self.assertIn("Jane", page_text)
        self.assertNotIn("John", page_text)

    def test_middle_name_contiguous_slice(self):
        """First+middle and middle+last slices derive replacement from the full name."""
        entity = _entity_multi_occ(
            "Full Name", "John Michael Smith", "Fake Data",
            replacement_text="Jane Alice Doe",
            occurrences=[
                {"page_number": 1, "original_text": "John Michael Smith",
                 "bounding_boxes": [_bbox(50, 100, 130)]},
                {"page_number": 1, "original_text": "John Michael",
                 "bounding_boxes": [_bbox(50, 200, 90)]},
                {"page_number": 1, "original_text": "Michael Smith",
                 "bounding_boxes": [_bbox(50, 300, 90)]},
            ],
        )
        pdf = create_pdf_with_text([
            ("John Michael Smith", 50, 100),
            ("John Michael", 50, 200),
            ("Michael Smith", 50, 300),
        ])
        svc = MaskingService()
        result_bytes = svc.apply_pdf_masks(pdf, [entity])
        doc = fitz.open(stream=result_bytes, filetype="pdf")
        text = doc[0].get_text()
        doc.close()
        self.assertIn("Jane Alice", text)   # John Michael → Jane Alice
        self.assertIn("Alice Doe", text)    # Michael Smith → Alice Doe
        self.assertNotIn("John Michael", text)
        self.assertNotIn("Michael Smith", text)


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

    def test_handles_entity_with_no_occurrences_without_crashing(self):
        """An entity with occurrences=[] must be silently skipped."""
        entity = {
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "replacement_text": "",
            "strategy": "Black Out",
            "approved": True,
            "occurrences": [],
        }
        doc = _apply([("John Smith", 50, 100)], [entity])
        doc.close()  # must not raise

    def test_handles_multi_bbox_occurrence(self):
        """An occurrence with two bounding_boxes must redact both regions."""
        bbox1 = _bbox(50, 100, 45)
        bbox2 = _bbox(50, 120, 45)
        entity = {
            "entity_type": "Full Name",
            "original_text": "John",
            "replacement_text": "",
            "strategy": "Black Out",
            "approved": True,
            "occurrences": [
                {
                    "page_number": 1,
                    "original_text": "John",
                    "bounding_boxes": [bbox1, bbox2],
                }
            ],
        }
        doc = _apply([("John", 50, 100), ("Smith", 50, 120)], [entity])
        page = doc[0]
        black_rects = [d["rect"] for d in page.get_drawings() if d.get("fill") == (0.0, 0.0, 0.0)]
        doc.close()
        self.assertGreaterEqual(len(black_rects), 2, "Expected at least 2 black rects for multi-bbox occurrence")

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


# ===========================================================================
# Multi-page documents
# ===========================================================================


class TestMultiPage(unittest.TestCase):

    def _two_page_pdf(self) -> bytes:
        """Create a 2-page PDF: page 1 has 'Safe text page 1', page 2 has 'John Smith'."""
        doc = fitz.open()
        # Insert text immediately after creating each page to avoid stale page refs
        doc.new_page(width=W, height=H)
        doc[0].insert_text((50, 100), "Safe text page 1", fontsize=FS)
        doc.new_page(width=W, height=H)
        doc[1].insert_text((50, 100), "John Smith", fontsize=FS)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()

    def test_masks_entity_on_correct_page(self):
        """An occurrence with page_number=2 must be redacted on page 2, not page 1."""
        two_page_pdf = self._two_page_pdf()

        entity = {
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "replacement_text": "",
            "strategy": "Black Out",
            "approved": True,
            "occurrences": [
                {
                    "page_number": 2,
                    "original_text": "John Smith",
                    "bounding_boxes": [_bbox(50, 100, 90)],
                }
            ],
        }

        svc = MaskingService()
        result_bytes = svc.apply_pdf_masks(two_page_pdf, [entity])
        result = fitz.open(stream=result_bytes, filetype="pdf")

        # Page 1 text must be untouched
        self.assertIn("Safe text page 1", result[0].get_text())
        # Page 2 text must be removed
        self.assertNotIn("John Smith", result[1].get_text())
        result.close()

    def test_same_entity_on_multiple_pages(self):
        """An entity with occurrences on both pages must be redacted on both."""
        doc = fitz.open()
        for i in range(2):
            doc.new_page(width=W, height=H)
            doc[i].insert_text((50, 100), "John Smith", fontsize=FS)
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()

        entity = {
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "replacement_text": "",
            "strategy": "Black Out",
            "approved": True,
            "occurrences": [
                {"page_number": 1, "original_text": "John Smith", "bounding_boxes": [_bbox(50, 100, 90)]},
                {"page_number": 2, "original_text": "John Smith", "bounding_boxes": [_bbox(50, 100, 90)]},
            ],
        }

        svc = MaskingService()
        result_bytes = svc.apply_pdf_masks(buf.getvalue(), [entity])
        result = fitz.open(stream=result_bytes, filetype="pdf")
        self.assertNotIn("John Smith", result[0].get_text())
        self.assertNotIn("John Smith", result[1].get_text())
        result.close()


# ===========================================================================
# Mixed strategies on the same page
# ===========================================================================


class TestMixedStrategies(unittest.TestCase):

    def test_black_out_and_fake_data_on_same_page(self):
        """Two entities with different strategies can coexist on the same page."""
        entities = [
            _entity("Full Name", "John Smith", 50, 100, 90,
                    strategy="Fake Data", replacement_text="Jane Doe"),
            _entity("Email", "john@example.com", 50, 200, 140,
                    strategy="Black Out"),
        ]
        doc = _apply(
            [("John Smith", 50, 100), ("john@example.com", 50, 200)],
            entities,
        )
        text = doc[0].get_text()
        doc.close()

        self.assertNotIn("John Smith", text)
        self.assertIn("Jane Doe", text)
        self.assertNotIn("john@example.com", text)

    def test_entity_label_and_black_out_on_same_page(self):
        """Entity Label and Black Out can both be applied to the same page."""
        entities = [
            _entity("Full Name", "Alice Brown",       50, 100, 90, strategy="Entity Label"),
            _entity("Email",     "alice@example.com", 50, 200, 150, strategy="Black Out"),
        ]
        doc = _apply(
            [("Alice Brown", 50, 100), ("alice@example.com", 50, 200)],
            entities,
        )
        text = doc[0].get_text()
        doc.close()

        # Entity Label must have inserted a label for Alice Brown
        self.assertRegex(text, r"Full_Name_[A-Z\d]+")
        # Black Out must have removed the email
        self.assertNotIn("alice@example.com", text)

    def test_unapproved_entity_not_masked_alongside_approved(self):
        """An unapproved entity is left intact even when other entities are masked."""
        entities = [
            _entity("Full Name", "John Smith", 50, 100, 90,
                    strategy="Black Out", approved=True),
            _entity("Email", "john@example.com", 50, 200, 140,
                    strategy="Black Out", approved=False),
        ]
        doc = _apply(
            [("John Smith", 50, 100), ("john@example.com", 50, 200)],
            entities,
        )
        text = doc[0].get_text()
        doc.close()

        self.assertNotIn("John Smith", text)
        self.assertIn("john@example.com", text)


if __name__ == "__main__":
    unittest.main()
