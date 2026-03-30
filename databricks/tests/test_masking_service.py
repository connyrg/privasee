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


# ===========================================================================
# Form widget masking
# ===========================================================================


def _create_pdf_with_widgets(widgets_spec: list) -> tuple:
    """
    Create a PDF page with AcroForm text widgets.

    widgets_spec: list of (field_name, field_value, rect_tuple)
        rect_tuple: (x0, y0, x1, y1) in PDF points

    Returns (pdf_bytes, widgets_rects) where widgets_rects maps
    field_name -> fitz.Rect (so callers can build bboxes from them).
    """
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    widget_rects = {}
    for field_name, field_value, (x0, y0, x1, y1) in widgets_spec:
        wgt = fitz.Widget()
        wgt.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        wgt.field_name = field_name
        wgt.field_value = field_value
        wgt.rect = fitz.Rect(x0, y0, x1, y1)
        page.add_widget(wgt)
        widget_rects[field_name] = fitz.Rect(x0, y0, x1, y1)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue(), widget_rects


def _widget_values(result_bytes: bytes) -> dict:
    """Return {field_name: field_value} from the first page of result_bytes."""
    doc = fitz.open(stream=result_bytes, filetype="pdf")
    vals = {}
    for wgt in doc[0].widgets():
        vals[wgt.field_name] = wgt.field_value or ""
    doc.close()
    return vals


def _normalised_bbox(rect: fitz.Rect) -> list:
    """Convert a fitz.Rect (PDF points) to a normalised [x, y, w, h] bbox."""
    return [rect.x0 / W, rect.y0 / H, rect.width / W, rect.height / H]


class TestFormWidgetMasking(unittest.TestCase):
    """
    Tests that form widget field values are masked in place, not just painted over.

    Key invariant: when a bounding box from the LLM covers a form widget, the
    widget's field_value must be updated directly.  A redact-annotation painted
    on top of a widget is invisible because the widget renders above the content
    stream layer.
    """

    # ------------------------------------------------------------------
    # Single widget — bbox precisely covers widget rect (sanity check)
    # ------------------------------------------------------------------

    def test_widget_masked_via_exact_bbox_overlap_black_out(self):
        """Black Out: widget value cleared when bbox exactly covers widget rect."""
        spec = [("name_field", "John Smith", (50, 100, 200, 120))]
        pdf_bytes, rects = _create_pdf_with_widgets(spec)

        bbox = _normalised_bbox(rects["name_field"])
        entity = {
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "replacement_text": "",
            "strategy": "Black Out",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "John Smith",
                             "bounding_boxes": [bbox]}],
        }

        svc = MaskingService()
        result = svc.apply_pdf_masks(pdf_bytes, [entity])
        vals = _widget_values(result)
        self.assertNotEqual(vals.get("name_field", "John Smith"), "John Smith",
                            "Widget value must be cleared by Black Out")

    def test_widget_masked_via_exact_bbox_overlap_fake_data(self):
        """Fake Data: widget value replaced when bbox exactly covers widget rect."""
        spec = [("name_field", "John Smith", (50, 100, 200, 120))]
        pdf_bytes, rects = _create_pdf_with_widgets(spec)

        bbox = _normalised_bbox(rects["name_field"])
        entity = {
            "entity_type": "Full Name",
            "original_text": "John Smith",
            "replacement_text": "Jane Doe",
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "John Smith",
                             "bounding_boxes": [bbox]}],
        }

        svc = MaskingService()
        result = svc.apply_pdf_masks(pdf_bytes, [entity])
        vals = _widget_values(result)
        self.assertEqual(vals.get("name_field"), "Jane Doe",
                         "Widget value must be replaced with fake data")

    # ------------------------------------------------------------------
    # Single widget — bbox is offset and misses the widget rect
    # (value-based fallback required)
    # ------------------------------------------------------------------

    def test_widget_masked_via_value_fallback_when_bbox_misses(self):
        """
        When the LLM bbox does not intersect the widget rect the service must
        fall back to matching by field value so the widget is still masked.

        This test fails with the current code (no fallback) and passes once
        the value-based fallback is in place.
        """
        spec = [("address_field", "80 Ann St", (50, 300, 300, 320))]
        pdf_bytes, _ = _create_pdf_with_widgets(spec)

        # bbox is positioned way above the widget (simulates LLM coordinate mismatch)
        misplaced_bbox = [50 / W, 10 / H, 250 / W, 15 / H]  # top of page, not near widget

        entity = {
            "entity_type": "Address",
            "original_text": "80 Ann St",
            "replacement_text": "99 Fake Rd",
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "80 Ann St",
                             "bounding_boxes": [misplaced_bbox]}],
        }

        svc = MaskingService()
        result = svc.apply_pdf_masks(pdf_bytes, [entity])
        vals = _widget_values(result)
        self.assertNotEqual(vals.get("address_field"), "80 Ann St",
                            "Widget value must be replaced even when bbox misses the widget rect")

    def test_widget_not_masked_when_value_does_not_match(self):
        """
        Value-based fallback must NOT mask a widget whose value doesn't match
        the occurrence text.  Unrelated fields must be left intact.
        """
        spec = [
            ("address_field",  "80 Ann St",   (50, 300, 300, 320)),
            ("unrelated_field", "Some Other",  (50, 350, 300, 370)),
        ]
        pdf_bytes, _ = _create_pdf_with_widgets(spec)

        # bbox misses both widgets
        misplaced_bbox = [50 / W, 10 / H, 250 / W, 15 / H]

        entity = {
            "entity_type": "Address",
            "original_text": "80 Ann St",
            "replacement_text": "99 Fake Rd",
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "80 Ann St",
                             "bounding_boxes": [misplaced_bbox]}],
        }

        svc = MaskingService()
        result = svc.apply_pdf_masks(pdf_bytes, [entity])
        vals = _widget_values(result)
        # Unrelated field must be untouched
        self.assertEqual(vals.get("unrelated_field"), "Some Other",
                         "Unrelated widget must not be affected by value-based fallback")

    # ------------------------------------------------------------------
    # Multi-widget span — occurrence bbox covers multiple widgets
    # (e.g. day/month/year date fields)
    # ------------------------------------------------------------------

    def test_multi_widget_dob_all_fields_masked(self):
        """
        A date-of-birth split across day/month/year widgets is fully masked when
        the occurrence bbox covers all three widgets.
        """
        spec = [
            ("dob_day",   "31",   (50,  200, 100, 220)),
            ("dob_month", "07",   (110, 200, 160, 220)),
            ("dob_year",  "1980", (170, 200, 250, 220)),
        ]
        pdf_bytes, rects = _create_pdf_with_widgets(spec)

        # Single bbox spanning all three widgets
        combined = rects["dob_day"] | rects["dob_month"] | rects["dob_year"]
        combined_bbox = _normalised_bbox(combined)

        entity = {
            "entity_type": "Date of Birth",
            "original_text": "31 07 1980",
            "replacement_text": "12 03 1990",  # different year so all three change
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "31 07 1980",
                             "bounding_boxes": [combined_bbox]}],
        }

        svc = MaskingService()
        result = svc.apply_pdf_masks(pdf_bytes, [entity])
        vals = _widget_values(result)

        self.assertNotEqual(vals.get("dob_day"),   "31",   "Day widget must be masked")
        self.assertNotEqual(vals.get("dob_month"), "07",   "Month widget must be masked")
        self.assertNotEqual(vals.get("dob_year"),  "1980", "Year widget must be masked")

    def test_multi_widget_address_street_and_suburb_masked(self):
        """
        Address split across street and suburb widgets — both must be masked
        when the occurrence text spans both field values.
        """
        spec = [
            ("street",  "80 Ann St",  (50, 400, 300, 420)),
            ("suburb",  "Brisbane",   (50, 430, 300, 450)),
        ]
        pdf_bytes, rects = _create_pdf_with_widgets(spec)

        combined = rects["street"] | rects["suburb"]
        combined_bbox = _normalised_bbox(combined)

        entity = {
            "entity_type": "Address",
            "original_text": "80 Ann St Brisbane",
            "replacement_text": "99 Fake Rd Sydney",
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "80 Ann St Brisbane",
                             "bounding_boxes": [combined_bbox]}],
        }

        svc = MaskingService()
        result = svc.apply_pdf_masks(pdf_bytes, [entity])
        vals = _widget_values(result)

        self.assertNotEqual(vals.get("street"),  "80 Ann St",  "Street widget must be masked")
        self.assertNotEqual(vals.get("suburb"),  "Brisbane",   "Suburb widget must be masked")


# ===========================================================================
# _resolve_component_replacement — unit tests
# ===========================================================================


def _mock_widget(field_value: str, x0: float, y0: float = 100.0, width: float = 40.0):
    """Minimal duck-type widget with .field_value and .rect for unit tests."""
    class _W:
        def __init__(self, fv, r):
            self.field_value = fv
            self.rect = r
    return _W(field_value, fitz.Rect(x0, y0, x0 + width, y0 + 20))


class TestResolveComponentReplacement(unittest.TestCase):
    """Unit tests for MaskingService._resolve_component_replacement."""

    # ------------------------------------------------------------------
    # Separator-based — unambiguous
    # ------------------------------------------------------------------

    def test_slash_date_day(self):
        widgets = [_mock_widget("15", 50), _mock_widget("03", 100), _mock_widget("1985", 150)]
        result = MaskingService._resolve_component_replacement(
            "15", "15/03/1985", "22/07/1985", fitz.Rect(50, 100, 90, 120), widgets
        )
        self.assertEqual(result, "22")

    def test_slash_date_month(self):
        widgets = [_mock_widget("15", 50), _mock_widget("03", 100), _mock_widget("1985", 150)]
        result = MaskingService._resolve_component_replacement(
            "03", "15/03/1985", "22/07/1985", fitz.Rect(100, 100, 140, 120), widgets
        )
        self.assertEqual(result, "07")

    def test_slash_date_year(self):
        widgets = [_mock_widget("15", 50), _mock_widget("03", 100), _mock_widget("1985", 150)]
        result = MaskingService._resolve_component_replacement(
            "1985", "15/03/1985", "22/07/1985", fitz.Rect(150, 100, 210, 120), widgets
        )
        self.assertEqual(result, "1985")

    def test_space_sep_medicare_group1(self):
        widgets = [_mock_widget("2023", 50), _mock_widget("45678", 140), _mock_widget("1", 250)]
        result = MaskingService._resolve_component_replacement(
            "2023", "2023 45678 1", "9876 54321 0", fitz.Rect(50, 100, 130, 120), widgets
        )
        self.assertEqual(result, "9876")

    def test_space_sep_medicare_group2(self):
        widgets = [_mock_widget("2023", 50), _mock_widget("45678", 140), _mock_widget("1", 250)]
        result = MaskingService._resolve_component_replacement(
            "45678", "2023 45678 1", "9876 54321 0", fitz.Rect(140, 100, 240, 120), widgets
        )
        self.assertEqual(result, "54321")

    # ------------------------------------------------------------------
    # Separator-based — ambiguous (DD==MM=="01"), resolved by x-position
    # ------------------------------------------------------------------

    def test_slash_ambiguous_day_resolved_by_position(self):
        widgets = [
            _mock_widget("01", 50),   # leftmost → DD → index 0
            _mock_widget("01", 100),  # middle   → MM → index 1
            _mock_widget("2025", 150),
        ]
        result = MaskingService._resolve_component_replacement(
            "01", "01/01/2025", "05/03/2025", fitz.Rect(50, 100, 90, 120), widgets
        )
        self.assertEqual(result, "05")  # DD replacement

    def test_slash_ambiguous_month_resolved_by_position(self):
        widgets = [
            _mock_widget("01", 50),
            _mock_widget("01", 100),  # → MM → index 1
            _mock_widget("2025", 150),
        ]
        result = MaskingService._resolve_component_replacement(
            "01", "01/01/2025", "05/03/2025", fitz.Rect(100, 100, 140, 120), widgets
        )
        self.assertEqual(result, "03")  # MM replacement

    # ------------------------------------------------------------------
    # Character-level fallback (10 single-char widgets)
    # ------------------------------------------------------------------

    def test_char_level_first_char(self):
        widgets = [_mock_widget(c, 50 + i * 20) for i, c in enumerate("2023456781")]
        result = MaskingService._resolve_component_replacement(
            "2", "2023456781", "9876543210", fitz.Rect(50, 100, 70, 120), widgets
        )
        self.assertEqual(result, "9")

    def test_char_level_middle_char(self):
        # Index 4 in "2023456781" is "4"; index 4 in "9876543210" is "5"
        widgets = [_mock_widget(c, 50 + i * 20) for i, c in enumerate("2023456781")]
        result = MaskingService._resolve_component_replacement(
            "4", "2023456781", "9876543210", fitz.Rect(130, 100, 150, 120), widgets
        )
        self.assertEqual(result, "5")

    def test_char_level_last_char(self):
        # Index 9 in "2023456781" is "1"; index 9 in "9876543210" is "0"
        widgets = [_mock_widget(c, 50 + i * 20) for i, c in enumerate("2023456781")]
        result = MaskingService._resolve_component_replacement(
            "1", "2023456781", "9876543210", fitz.Rect(230, 100, 250, 120), widgets
        )
        self.assertEqual(result, "0")

    # ------------------------------------------------------------------
    # Fallback — no split possible
    # ------------------------------------------------------------------

    def test_returns_full_replacement_when_part_counts_differ(self):
        # original has 3 whitespace-tokens, replacement has 4 → no separator matches
        result = MaskingService._resolve_component_replacement(
            "Ann", "80 Ann St", "99 Fake Road North",
            fitz.Rect(50, 100, 200, 120), []
        )
        self.assertEqual(result, "99 Fake Road North")


# ===========================================================================
# Form widget masking — aligned component replacement (integration)
# ===========================================================================


class TestFormWidgetComponentReplacement(unittest.TestCase):
    """
    Verifies that when Fake Data masks multi-widget occurrences (e.g. date split
    across DD/MM/YYYY fields), each widget receives its aligned replacement
    component — not the full replacement string.
    """

    def test_slash_date_each_widget_gets_aligned_component(self):
        """DD/MM/YYYY widgets each get their own replacement component."""
        spec = [
            ("dob_day",   "15",   (50,  200, 90,  220)),
            ("dob_month", "03",   (100, 200, 140, 220)),
            ("dob_year",  "1985", (150, 200, 230, 220)),
        ]
        pdf_bytes, rects = _create_pdf_with_widgets(spec)
        combined = rects["dob_day"] | rects["dob_month"] | rects["dob_year"]

        entity = {
            "entity_type": "Date of Birth",
            "original_text": "15/03/1985",
            "replacement_text": "22/07/1985",
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "15/03/1985",
                             "bounding_boxes": [_normalised_bbox(combined)]}],
        }

        svc = MaskingService()
        vals = _widget_values(svc.apply_pdf_masks(pdf_bytes, [entity]))

        self.assertEqual(vals.get("dob_day"),   "22",   "Day must get replacement day")
        self.assertEqual(vals.get("dob_month"), "07",   "Month must get replacement month")
        self.assertEqual(vals.get("dob_year"),  "1985", "Year must get replacement year")

    def test_space_sep_medicare_grouped_gets_aligned_components(self):
        """Medicare in 3 grouped widgets (5+4+1) gets per-group replacement."""
        spec = [
            ("mc1", "2023",  (50,  300, 130, 320)),
            ("mc2", "45678", (140, 300, 240, 320)),
            ("mc3", "1",     (250, 300, 280, 320)),
        ]
        pdf_bytes, rects = _create_pdf_with_widgets(spec)
        combined = rects["mc1"] | rects["mc2"] | rects["mc3"]

        entity = {
            "entity_type": "Medicare Number",
            "original_text": "2023 45678 1",
            "replacement_text": "9876 54321 0",
            "strategy": "Fake Data",
            "approved": True,
            "occurrences": [{"page_number": 1, "original_text": "2023 45678 1",
                             "bounding_boxes": [_normalised_bbox(combined)]}],
        }

        svc = MaskingService()
        vals = _widget_values(svc.apply_pdf_masks(pdf_bytes, [entity]))

        self.assertEqual(vals.get("mc1"), "9876",  "Group 1 must get aligned replacement")
        self.assertEqual(vals.get("mc2"), "54321", "Group 2 must get aligned replacement")
        self.assertEqual(vals.get("mc3"), "0",     "Group 3 must get aligned replacement")


if __name__ == "__main__":
    unittest.main()
