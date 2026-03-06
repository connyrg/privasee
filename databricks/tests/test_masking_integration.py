"""
Integration tests for the full MaskingModel predict pipeline.

Unlike the unit tests, these use the *real* MaskingService (PyMuPDF — fitz
is NOT mocked).  Only the UC REST API calls (_fetch_original_file and
_write_masked_file) are replaced with lightweight mocks so no live Databricks
workspace or UC volume is required.

This verifies the end-to-end path:
  predict() → _process_masking() → real MaskingService → verified PDF output

Test PDFs are created with create_pdf_with_text (same helper as
test_masking_service.py) so exact text positions are known.
"""

import io
import json
import sys
import unittest
from unittest.mock import Mock, patch

import fitz
import pandas as pd

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing the module
# ---------------------------------------------------------------------------

class _MockPythonModel:
    def load_context(self, context): pass
    def predict(self, context, model_input): pass

class _MockPyfunc:
    PythonModel = _MockPythonModel

class _MockMlflow:
    pyfunc = _MockPyfunc()

sys.modules.setdefault("mlflow", _MockMlflow())
sys.modules.setdefault("mlflow.pyfunc", _MockPyfunc())

from databricks.model.masking_model import MaskingModel   # noqa: E402
from databricks.model.masking_service import MaskingService  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrors test_masking_service.py)
# ---------------------------------------------------------------------------

W: float = 595.0
H: float = 842.0
FS: int = 12


def create_pdf_with_text(text_items: list) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=W, height=H)
    for text, x, y in text_items:
        page.insert_text((x, y), text, fontsize=FS)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _bbox(x: float, baseline_y: float, width: float) -> list:
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


def _open_result(masked_bytes: bytes) -> fitz.Document:
    return fitz.open(stream=masked_bytes, filetype="pdf")


# ---------------------------------------------------------------------------
# Factory: ready MaskingModel with real MaskingService and mocked UC I/O
# ---------------------------------------------------------------------------

def _make_model(pdf_bytes: bytes, filename: str = "original.pdf") -> MaskingModel:
    """Return a MaskingModel wired with a real MaskingService.

    _fetch_original_file returns *pdf_bytes* as if it came from UC.
    _write_masked_file captures the written bytes into model._written_bytes.
    """
    model = MaskingModel()
    model.masking_service = MaskingService()   # real service — no mock
    model.uc_volume_path = "/Volumes/cat/schema/sessions"
    model.databricks_host = "https://test.databricks.com"
    model.databricks_token = "test-token"

    written: list = []

    def fake_fetch(session_id):
        return (pdf_bytes, filename)

    def fake_write(session_id, masked_bytes):
        written.append(masked_bytes)

    model._fetch_original_file = Mock(side_effect=fake_fetch)
    model._write_masked_file = Mock(side_effect=fake_write)
    model._written = written
    return model


# ===========================================================================
# Group 1 — Black Out through full pipeline
# ===========================================================================

class TestIntegrationBlackOut(unittest.TestCase):

    def setUp(self):
        self.entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        self.pdf_bytes = create_pdf_with_text([("John Smith", 50, 100)])
        self.model = _make_model(self.pdf_bytes)

    def _run(self, entities=None):
        entities = entities if entities is not None else [self.entity]
        df = pd.DataFrame([{
            "session_id": "sess-blackout",
            "entities_to_mask": entities,
        }])
        return self.model.predict(context=None, model_input=df)

    def test_status_is_complete(self):
        result = self._run()
        self.assertEqual(result.iloc[0]["status"], "complete")

    def test_entities_masked_count(self):
        result = self._run()
        self.assertEqual(result.iloc[0]["entities_masked"], 1)

    def test_original_text_removed_from_pdf(self):
        self._run()
        masked_bytes = self.model._written[0]
        doc = _open_result(masked_bytes)
        text = doc[0].get_text()
        doc.close()
        self.assertNotIn("John Smith", text)

    def test_write_called_with_valid_pdf(self):
        self._run()
        masked_bytes = self.model._written[0]
        self.assertTrue(masked_bytes.startswith(b"%PDF"))
        doc = _open_result(masked_bytes)
        self.assertGreaterEqual(len(doc), 1)
        doc.close()

    def test_unrelated_text_preserved(self):
        pdf = create_pdf_with_text([("John Smith", 50, 100), ("keep this", 50, 200)])
        model = _make_model(pdf)
        entity2 = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        df = pd.DataFrame([{"session_id": "s", "entities_to_mask": [entity2]}])
        model.predict(context=None, model_input=df)
        doc = _open_result(model._written[0])
        text = doc[0].get_text()
        doc.close()
        self.assertNotIn("John Smith", text)
        self.assertIn("keep this", text)


# ===========================================================================
# Group 2 — Fake Data through full pipeline
# ===========================================================================

class TestIntegrationFakeData(unittest.TestCase):

    def test_replacement_text_appears_in_output(self):
        entity = _entity("Full Name", "John Smith", 50, 100, 90,
                         strategy="Fake Data", replacement_text="Jane Doe")
        pdf = create_pdf_with_text([("John Smith", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([{"session_id": "s-fake", "entities_to_mask": [entity]}])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "complete")
        doc = _open_result(model._written[0])
        text = doc[0].get_text()
        doc.close()
        self.assertNotIn("John Smith", text)
        self.assertIn("Jane Doe", text)

    def test_json_string_entities_parsed_correctly(self):
        """MLflow passes entities_to_mask as a JSON string — verify full parse."""
        entity = _entity("Full Name", "John Smith", 50, 100, 90,
                         strategy="Fake Data", replacement_text="Jane Doe")
        pdf = create_pdf_with_text([("John Smith", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([{
            "session_id": "s-json",
            "entities_to_mask": json.dumps([entity]),   # JSON string, not list
        }])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "complete")
        doc = _open_result(model._written[0])
        text = doc[0].get_text()
        doc.close()
        self.assertIn("Jane Doe", text)


# ===========================================================================
# Group 3 — Entity Label through full pipeline
# ===========================================================================

class TestIntegrationEntityLabel(unittest.TestCase):

    def test_label_format_in_output_pdf(self):
        entity = _entity("Full Name", "John Smith", 50, 100, 90, strategy="Entity Label")
        pdf = create_pdf_with_text([("John Smith", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([{"session_id": "s-label", "entities_to_mask": [entity]}])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "complete")
        doc = _open_result(model._written[0])
        text = doc[0].get_text()
        doc.close()
        import re
        self.assertRegex(text, r"Full_Name_\d+")


# ===========================================================================
# Group 4 — Multiple rows, mixed success and failure
# ===========================================================================

class TestIntegrationMultiRow(unittest.TestCase):

    def test_all_rows_complete(self):
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        pdf = create_pdf_with_text([("John Smith", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([
            {"session_id": "s1", "entities_to_mask": [entity]},
            {"session_id": "s2", "entities_to_mask": [entity]},
        ])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(list(result["status"]), ["complete", "complete"])
        self.assertEqual(len(model._written), 2)

    def test_error_row_does_not_block_subsequent_row(self):
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        pdf = create_pdf_with_text([("John Smith", 50, 100)])

        model = MaskingModel()
        model.masking_service = MaskingService()
        model.uc_volume_path = "/Volumes/cat/schema/sessions"
        model.databricks_host = "https://test.databricks.com"
        model.databricks_token = "test-token"

        written: list = []
        call_count = [0]

        def fetch_side_effect(session_id):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FileNotFoundError("first row not found")
            return (pdf, "original.pdf")

        model._fetch_original_file = Mock(side_effect=fetch_side_effect)
        model._write_masked_file = Mock(side_effect=lambda sid, b: written.append(b))
        model._written = written

        df = pd.DataFrame([
            {"session_id": "bad", "entities_to_mask": [entity]},
            {"session_id": "ok",  "entities_to_mask": [entity]},
        ])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "error")
        self.assertEqual(result.iloc[1]["status"], "complete")
        # Only the second row should have produced masked output
        self.assertEqual(len(written), 1)
        doc = _open_result(written[0])
        doc.close()  # valid PDF

    def test_zero_entities_produces_unchanged_pdf(self):
        pdf = create_pdf_with_text([("Hello World", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([{"session_id": "s-empty", "entities_to_mask": []}])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "complete")
        self.assertEqual(result.iloc[0]["entities_masked"], 0)
        doc = _open_result(model._written[0])
        self.assertIn("Hello World", doc[0].get_text())
        doc.close()


# ===========================================================================
# Group 5 — _write_masked_file receives the correct bytes
# ===========================================================================

class TestIntegrationWritePayload(unittest.TestCase):

    def test_write_called_once_per_row(self):
        entity = _entity("Full Name", "John Smith", 50, 100, 90, "Black Out")
        pdf = create_pdf_with_text([("John Smith", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([{"session_id": "sess-w", "entities_to_mask": [entity]}])
        model.predict(context=None, model_input=df)

        model._write_masked_file.assert_called_once()
        call_args = model._write_masked_file.call_args
        session_arg = call_args[0][0]
        bytes_arg = call_args[0][1]

        self.assertEqual(session_arg, "sess-w")
        self.assertTrue(bytes_arg.startswith(b"%PDF"))

    def test_fetch_called_with_session_id(self):
        entity = _entity("Full Name", "Jane Doe", 50, 100, 70,
                         strategy="Fake Data", replacement_text="John Smith")
        pdf = create_pdf_with_text([("Jane Doe", 50, 100)])
        model = _make_model(pdf)

        df = pd.DataFrame([{"session_id": "specific-sess", "entities_to_mask": [entity]}])
        model.predict(context=None, model_input=df)

        model._fetch_original_file.assert_called_once_with("specific-sess")


if __name__ == "__main__":
    unittest.main()
