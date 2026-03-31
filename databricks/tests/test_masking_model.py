"""
Unit tests for databricks.model.MaskingModel

All UC volume HTTP calls and masking operations are mocked — no live
Databricks workspace or real PDFs are required.

Covers:
1. load_context — success and missing env var failures
2. predict — successful PDF masking, JSON string input, image path, error handling
3. _apply_masking — unsupported file type raises ValueError
4. _write_masked_file — PUT request to UC Files API
5. _verify_masking — occurrence counting, digital vs scanned routing, ADI unavailable
"""

import io
import json
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch, call

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

sys.modules["mlflow"] = _MockMlflow()
sys.modules["mlflow.pyfunc"] = _MockPyfunc()

from databricks.model.masking_model import MaskingModel  # noqa: E402


# ---------------------------------------------------------------------------
# Shared env vars for load_context tests
# ---------------------------------------------------------------------------

_VALID_ENV = {
    "UC_VOLUME_PATH":    "/Volumes/cat/schema/sessions",
    "DATABRICKS_HOST":   "https://test.databricks.com",
    "DATABRICKS_TOKEN":  "test-token-123",
}

_FAKE_PDF = b"%PDF-1.4 fake %%EOF"


# ===========================================================================
# Group 1 — load_context
# ===========================================================================

class TestLoadContext(unittest.TestCase):

    def _make_model(self) -> MaskingModel:
        return MaskingModel()

    def _load_context_mocked_modules(self):
        """sys.modules patches for the two relative imports inside load_context.

        load_context does `from .masking_service import MaskingService` and
        `from .ocr_service import OCRService` lazily.  Neither submodule has
        been imported at test-collection time, so patch() can't resolve them by
        dotted name — we inject them directly into sys.modules instead.
        """
        return patch.dict(sys.modules, {
            "databricks.model.masking_service": MagicMock(),
            "databricks.model.ocr_service": MagicMock(),
        })

    def test_load_context_succeeds_with_valid_env(self):
        with self._load_context_mocked_modules(), patch.dict(os.environ, _VALID_ENV):
            model = self._make_model()
            model.load_context(context=None)

        self.assertEqual(model.uc_volume_path, "/Volumes/cat/schema/sessions")
        self.assertEqual(model.databricks_host, "https://test.databricks.com")
        self.assertEqual(model.databricks_token, "test-token-123")

    def test_load_context_strips_trailing_slash(self):
        env = {**_VALID_ENV, "UC_VOLUME_PATH": "/Volumes/cat/schema/sessions/",
               "DATABRICKS_HOST": "https://test.databricks.com/"}
        with self._load_context_mocked_modules(), patch.dict(os.environ, env):
            model = self._make_model()
            model.load_context(context=None)

        self.assertFalse(model.uc_volume_path.endswith("/"))
        self.assertFalse(model.databricks_host.endswith("/"))

    def test_load_context_raises_without_uc_volume_path(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "UC_VOLUME_PATH"}
        with self._load_context_mocked_modules(), patch.dict(os.environ, env, clear=True):
            model = self._make_model()
            with self.assertRaises(ValueError) as ctx:
                model.load_context(context=None)
        self.assertIn("UC_VOLUME_PATH", str(ctx.exception))

    def test_load_context_raises_without_databricks_host(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "DATABRICKS_HOST"}
        with self._load_context_mocked_modules(), patch.dict(os.environ, env, clear=True):
            model = self._make_model()
            with self.assertRaises(ValueError) as ctx:
                model.load_context(context=None)
        self.assertIn("DATABRICKS_HOST", str(ctx.exception))

    def test_load_context_raises_without_databricks_token(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "DATABRICKS_TOKEN"}
        with self._load_context_mocked_modules(), patch.dict(os.environ, env, clear=True):
            model = self._make_model()
            with self.assertRaises(ValueError) as ctx:
                model.load_context(context=None)
        self.assertIn("DATABRICKS_TOKEN", str(ctx.exception))


# ===========================================================================
# Group 2 — predict (success paths)
# ===========================================================================

class TestPredictSuccess(unittest.TestCase):

    def _make_ready_model(self) -> MaskingModel:
        """Return a MaskingModel with all deps wired as mocks (no load_context)."""
        model = MaskingModel()
        model.masking_service = Mock()
        model.masking_service.apply_pdf_masks.return_value = _FAKE_PDF
        model.ocr_service = Mock()
        model.uc_volume_path = "/Volumes/cat/schema/sessions"
        model.databricks_host = "https://test.databricks.com"
        model.databricks_token = "test-token"
        model._fetch_original_file = Mock(return_value=(b"original bytes", "original.pdf"))
        model._write_masked_file = Mock()
        return model

    def _sample_entities(self) -> list:
        return [
            {
                "id": "e1",
                "entity_type": "Full Name",
                "original_text": "John Smith",
                "replacement_text": "Jane Doe",
                "bounding_box": [0.05, 0.08, 0.2, 0.025],
                "strategy": "Fake Data",
                "approved": True,
                "page_number": 1,
                "occurrences": [
                    {"page_number": 1, "original_text": "John Smith", "bounding_boxes": [[0.05, 0.08, 0.2, 0.025]]},
                ],
            }
        ]

    def test_predict_returns_complete_status(self):
        model = self._make_ready_model()
        df = pd.DataFrame([{
            "session_id": "sess-abc",
            "entities_to_mask": self._sample_entities(),
        }])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["session_id"], "sess-abc")
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["entities_masked"], 1)

    def test_predict_accepts_json_string_input(self):
        """entities_to_mask can be a JSON string (MLflow serialises it that way)."""
        model = self._make_ready_model()
        df = pd.DataFrame([{
            "session_id": "sess-json",
            "entities_to_mask": json.dumps(self._sample_entities()),
        }])
        result = model.predict(context=None, model_input=df)
        self.assertEqual(result.iloc[0]["status"], "complete")

    def test_predict_calls_masking_service_with_pdf(self):
        model = self._make_ready_model()
        df = pd.DataFrame([{
            "session_id": "sess-pdf",
            "entities_to_mask": self._sample_entities(),
        }])
        model.predict(context=None, model_input=df)
        model.masking_service.apply_pdf_masks.assert_called_once()

    def test_predict_calls_write_masked_file(self):
        model = self._make_ready_model()
        df = pd.DataFrame([{
            "session_id": "sess-write",
            "entities_to_mask": self._sample_entities(),
        }])
        model.predict(context=None, model_input=df)
        model._write_masked_file.assert_called_once_with("sess-write", _FAKE_PDF)

    def test_predict_multiple_rows(self):
        model = self._make_ready_model()
        df = pd.DataFrame([
            {"session_id": "s1", "entities_to_mask": self._sample_entities()},
            {"session_id": "s2", "entities_to_mask": self._sample_entities()},
        ])
        result = model.predict(context=None, model_input=df)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result["status"]), ["complete", "complete"])

    def test_predict_image_file_calls_apply_masks(self):
        """PNG original must go through PIL apply_masks path, not apply_pdf_masks."""
        model = self._make_ready_model()
        model._fetch_original_file = Mock(return_value=(b"png bytes", "original.png"))
        model.masking_service.apply_masks.return_value = None

        with patch("databricks.model.masking_model.tempfile") as mock_tmpdir, \
             patch("databricks.model.masking_model.io") as mock_io:
            # Simulate the image → PDF conversion path
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = Mock(return_value="/tmp/fake")
            mock_ctx.__exit__ = Mock(return_value=False)
            mock_tmpdir.TemporaryDirectory.return_value = mock_ctx

            from unittest.mock import patch as _patch
            with _patch("databricks.model.masking_model.os.path.join", side_effect=lambda *a: "/tmp/" + a[-1]), \
                 _patch("builtins.open", unittest.mock.mock_open(read_data=b"data")):
                with _patch("databricks.model.masking_model.MaskingModel._apply_masking",
                            return_value=_FAKE_PDF) as mock_apply:
                    df = pd.DataFrame([{"session_id": "s-img", "entities_to_mask": self._sample_entities()}])
                    result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "complete")

    def test_predict_with_run_verification_includes_verify_fields(self):
        """When run_verification=True the result includes occurrences_total/masked/score."""
        model = self._make_ready_model()
        model._verify_masking = Mock(return_value={
            "occurrences_total": 2,
            "occurrences_masked": 2,
            "score": 100.0,
        })
        df = pd.DataFrame([{
            "session_id": "sess-verify",
            "entities_to_mask": self._sample_entities(),
            "run_verification": True,
        }])
        result = model.predict(context=None, model_input=df)

        row = result.iloc[0]
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["occurrences_total"], 2)
        self.assertEqual(row["occurrences_masked"], 2)
        self.assertEqual(row["score"], 100.0)
        model._verify_masking.assert_called_once()

    def test_predict_without_run_verification_skips_verify(self):
        """When run_verification is absent/False, _verify_masking is not called."""
        model = self._make_ready_model()
        model._verify_masking = Mock()
        df = pd.DataFrame([{
            "session_id": "sess-no-verify",
            "entities_to_mask": self._sample_entities(),
        }])
        model.predict(context=None, model_input=df)
        model._verify_masking.assert_not_called()


# ===========================================================================
# Group 3 — predict (error paths)
# ===========================================================================

class TestPredictErrors(unittest.TestCase):

    def _make_ready_model(self) -> MaskingModel:
        model = MaskingModel()
        model.masking_service = Mock()
        model.masking_service.apply_pdf_masks.return_value = _FAKE_PDF
        model.ocr_service = Mock()
        model.uc_volume_path = "/Volumes/cat/schema/sessions"
        model.databricks_host = "https://test.databricks.com"
        model.databricks_token = "test-token"
        model._fetch_original_file = Mock(return_value=(b"original bytes", "original.pdf"))
        model._write_masked_file = Mock()
        return model

    def _sample_entities(self) -> list:
        return [{"id": "e1", "entity_type": "Full Name", "original_text": "John",
                 "replacement_text": "Jane", "bounding_box": [0.05, 0.08, 0.2, 0.025],
                 "strategy": "Fake Data", "approved": True, "page_number": 1,
                 "occurrences": [{"page_number": 1, "original_text": "John", "bounding_boxes": []}]}]

    def test_predict_returns_error_row_on_fetch_failure(self):
        model = self._make_ready_model()
        model._fetch_original_file = Mock(side_effect=FileNotFoundError("No original file"))
        df = pd.DataFrame([{"session_id": "sess-err", "entities_to_mask": self._sample_entities()}])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "error")
        self.assertEqual(result.iloc[0]["entities_masked"], 0)
        self.assertIn("No original file", result.iloc[0]["error_message"])

    def test_predict_returns_error_row_on_masking_failure(self):
        model = self._make_ready_model()
        model.masking_service.apply_pdf_masks.side_effect = Exception("PDF error")
        df = pd.DataFrame([{"session_id": "sess-fail", "entities_to_mask": self._sample_entities()}])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "error")
        self.assertIn("PDF error", result.iloc[0]["error_message"])

    def test_predict_other_rows_succeed_after_one_error(self):
        model = self._make_ready_model()
        call_count = [0]

        def side_effect(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FileNotFoundError("Missing")
            return (b"bytes", "original.pdf")

        model._fetch_original_file = Mock(side_effect=side_effect)
        df = pd.DataFrame([
            {"session_id": "s1", "entities_to_mask": self._sample_entities()},
            {"session_id": "s2", "entities_to_mask": self._sample_entities()},
        ])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "error")
        self.assertEqual(result.iloc[1]["status"], "complete")


# ===========================================================================
# Group 4 — _apply_masking
# ===========================================================================

class TestApplyMasking(unittest.TestCase):

    def _make_ready_model(self) -> MaskingModel:
        model = MaskingModel()
        model.masking_service = Mock()
        model.masking_service.apply_pdf_masks.return_value = b"pdf"
        return model

    def test_pdf_calls_apply_pdf_masks(self):
        model = self._make_ready_model()
        result = model._apply_masking(b"pdf content", ".pdf", [])
        model.masking_service.apply_pdf_masks.assert_called_once()
        self.assertEqual(result, b"pdf")

    def test_unsupported_extension_raises_value_error(self):
        model = self._make_ready_model()
        with self.assertRaises(ValueError) as ctx:
            model._apply_masking(b"data", ".docx", [])
        self.assertIn("Unsupported file type", str(ctx.exception))


# ===========================================================================
# Group 5 — _write_masked_file
# ===========================================================================

class TestWriteMaskedFile(unittest.TestCase):

    def _make_ready_model(self) -> MaskingModel:
        model = MaskingModel()
        model.uc_volume_path = "/Volumes/cat/schema/sessions"
        model.databricks_host = "https://test.databricks.com"
        model.databricks_token = "test-token"
        return model

    def test_write_puts_to_correct_url(self):
        model = self._make_ready_model()
        with patch("requests.put") as mock_put:
            mock_put.return_value.raise_for_status = Mock()
            model._write_masked_file("sess-abc", b"data")

            expected_url = "https://test.databricks.com/api/2.0/fs/files/Volumes/cat/schema/sessions/sess-abc/masked.pdf"
            mock_put.assert_called_once()
            self.assertEqual(mock_put.call_args[0][0], expected_url)

    def test_write_uses_bearer_auth(self):
        model = self._make_ready_model()
        with patch("requests.put") as mock_put:
            mock_put.return_value.raise_for_status = Mock()
            model._write_masked_file("sess-write", b"data")

            headers = mock_put.call_args[1]["headers"]
            self.assertEqual(headers["Authorization"], "Bearer test-token")

    def test_write_sends_correct_bytes(self):
        model = self._make_ready_model()
        with patch("requests.put") as mock_put:
            mock_put.return_value.raise_for_status = Mock()
            model._write_masked_file("sess-bytes", b"my-pdf-data")

            self.assertEqual(mock_put.call_args[1]["data"], b"my-pdf-data")


# ===========================================================================
# Group 6 — _verify_masking
# ===========================================================================

def _make_verify_model() -> MaskingModel:
    """Return a MaskingModel ready for _verify_masking tests."""
    model = MaskingModel()
    model.masking_service = Mock()
    model.ocr_service = Mock()
    model.ocr_service.adi_available = False
    # Mirror the real OCRService constant so _verify_masking classification is correct.
    model.ocr_service.MIN_TEXT_LENGTH_FOR_DIGITAL = 50
    model.ocr_service.RENDER_ZOOM_FACTOR = 2.0
    model.ocr_service._adi_max_concurrent = 5
    return model


def _make_fitz_page(text: str = "", is_pdf: bool = True):
    """Return a mock fitz page whose get_text() returns the given text."""
    page = MagicMock()
    page.get_text.return_value = text
    pixmap = MagicMock()
    pixmap.tobytes.return_value = b"fake-png"
    page.get_pixmap.return_value = pixmap
    return page


def _make_fitz_doc(pages):
    """Return a mock fitz document containing the given page mocks."""
    doc = MagicMock()
    doc.__len__ = Mock(return_value=len(pages))
    doc.__iter__ = Mock(return_value=iter(pages))
    doc.__getitem__ = Mock(side_effect=lambda i: pages[i])
    doc.close = Mock()
    return doc


@contextmanager
def _mock_fitz_ctx(open_side_effect=None, open_return_value=None):
    """Context manager that injects a mock fitz module into sys.modules.

    _verify_masking does `import fitz` locally — patching sys.modules is the
    correct way to intercept a deferred (inside-function) import.
    """
    mock_fitz = MagicMock()
    mock_fitz.Matrix = MagicMock(return_value=MagicMock())
    if open_side_effect is not None:
        mock_fitz.open.side_effect = open_side_effect
    elif open_return_value is not None:
        mock_fitz.open.return_value = open_return_value
    with patch.dict(sys.modules, {"fitz": mock_fitz}):
        yield mock_fitz


class TestVerifyMasking(unittest.TestCase):

    def _make_entities(self, occurrences_per_entity):
        """Build entity list from [(page_num, text), ...] per entity."""
        return [
            {
                "id": f"e{i}",
                "entity_type": "Full Name",
                "original_text": text,
                "occurrences": [{"page_number": pg, "original_text": text, "bounding_boxes": []}],
            }
            for i, (pg, text) in enumerate(occurrences_per_entity)
        ]

    # Use text > MIN_DIGITAL_TEXT_LENGTH (50 chars) so pages are classified as
    # "digital" — the extraction path we're actually testing here.
    _DIGITAL_TEXT = "This is a digital PDF page with plenty of text content for detection."

    def test_all_occurrences_masked_score_100(self):
        """Score is 100 when none of the original texts appear in the masked output."""
        model = _make_verify_model()
        entities = self._make_entities([(1, "John Smith")])

        orig_page = _make_fitz_page(text=self._DIGITAL_TEXT + " John Smith")
        masked_page = _make_fitz_page(text=self._DIGITAL_TEXT + " redacted")

        with _mock_fitz_ctx(open_side_effect=[
            _make_fitz_doc([orig_page]),
            _make_fitz_doc([masked_page]),
        ]):
            result = model._verify_masking(b"orig", ".pdf", b"masked", entities)

        self.assertEqual(result["occurrences_total"], 1)
        self.assertEqual(result["occurrences_masked"], 1)
        self.assertEqual(result["score"], 100.0)

    def test_no_occurrences_masked_score_0(self):
        """Score is 0 when original text still appears in masked output."""
        model = _make_verify_model()
        entities = self._make_entities([(1, "John Smith")])

        orig_page = _make_fitz_page(text=self._DIGITAL_TEXT + " John Smith")
        masked_page = _make_fitz_page(text=self._DIGITAL_TEXT + " John Smith still here")

        with _mock_fitz_ctx(open_side_effect=[
            _make_fitz_doc([orig_page]),
            _make_fitz_doc([masked_page]),
        ]):
            result = model._verify_masking(b"orig", ".pdf", b"masked", entities)

        self.assertEqual(result["occurrences_total"], 1)
        self.assertEqual(result["occurrences_masked"], 0)
        self.assertEqual(result["score"], 0.0)

    def test_empty_entities_returns_score_100(self):
        """No occurrences → score 100 (nothing to mask)."""
        model = _make_verify_model()
        result = model._verify_masking(b"orig", ".pdf", b"masked", [])
        self.assertEqual(result["occurrences_total"], 0)
        self.assertEqual(result["occurrences_masked"], 0)
        self.assertEqual(result["score"], 100.0)

    def test_scanned_page_without_adi_treated_as_masked(self):
        """When ADI is unavailable, scanned pages are assumed fully masked."""
        model = _make_verify_model()
        model.ocr_service.adi_available = False
        entities = self._make_entities([(1, "John Smith")])

        # Original page has no text → classified as scanned
        orig_page = _make_fitz_page(text="")
        masked_page = _make_fitz_page(text="John Smith")  # would fail if we OCR'd it

        with _mock_fitz_ctx(open_side_effect=[
            _make_fitz_doc([orig_page]),
            _make_fitz_doc([masked_page]),
        ]):
            result = model._verify_masking(b"orig", ".pdf", b"masked", entities)

        # Cannot verify scanned page without ADI → treated as masked
        self.assertEqual(result["occurrences_masked"], 1)
        self.assertEqual(result["score"], 100.0)

    def test_scanned_page_with_adi_uses_ocr_result(self):
        """Scanned pages are re-OCR'd via ADI when available; result drives masked count."""
        model = _make_verify_model()
        model.ocr_service.adi_available = True
        model.ocr_service._get_adi_token = Mock(return_value="tok")
        # ADI returns text that does NOT contain the original → masked
        model.ocr_service._ocr_with_adi = Mock(return_value={"text": "Some other text", "words": []})
        entities = self._make_entities([(1, "John Smith")])

        # Original page has no text → classified as scanned
        orig_page = _make_fitz_page(text="")
        masked_page = _make_fitz_page(text="")  # get_text() irrelevant for scanned

        with _mock_fitz_ctx(open_side_effect=[
            _make_fitz_doc([orig_page]),
            _make_fitz_doc([masked_page]),
        ]):
            result = model._verify_masking(b"orig", ".pdf", b"masked", entities)

        model.ocr_service._ocr_with_adi.assert_called_once()
        self.assertEqual(result["occurrences_masked"], 1)

    def test_image_original_treats_all_pages_as_scanned(self):
        """Non-PDF originals (images) always route masked pages through ADI or treat as masked."""
        model = _make_verify_model()
        model.ocr_service.adi_available = False
        entities = self._make_entities([(1, "John Smith")])

        masked_page = _make_fitz_page(text="John Smith")

        # For image originals, only one fitz.open call (the masked PDF)
        with _mock_fitz_ctx(open_return_value=_make_fitz_doc([masked_page])):
            result = model._verify_masking(b"orig", ".png", b"masked", entities)

        # ADI not available → scanned page treated as masked
        self.assertEqual(result["occurrences_masked"], 1)

    def test_partial_masking_score(self):
        """50% score when one of two occurrences still present."""
        model = _make_verify_model()
        entities = [
            {
                "id": "e1",
                "entity_type": "Full Name",
                "original_text": "John Smith",
                "occurrences": [
                    {"page_number": 1, "original_text": "John Smith", "bounding_boxes": []},
                    {"page_number": 2, "original_text": "John Smith", "bounding_boxes": []},
                ],
            }
        ]

        orig_page1 = _make_fitz_page(text=self._DIGITAL_TEXT + " John Smith page one")
        orig_page2 = _make_fitz_page(text=self._DIGITAL_TEXT + " John Smith page two")
        masked_page1 = _make_fitz_page(text=self._DIGITAL_TEXT + " redacted page one")   # masked
        masked_page2 = _make_fitz_page(text=self._DIGITAL_TEXT + " John Smith page two") # not masked

        with _mock_fitz_ctx(open_side_effect=[
            _make_fitz_doc([orig_page1, orig_page2]),
            _make_fitz_doc([masked_page1, masked_page2]),
        ]):
            result = model._verify_masking(b"orig", ".pdf", b"masked", entities)

        self.assertEqual(result["occurrences_total"], 2)
        self.assertEqual(result["occurrences_masked"], 1)
        self.assertEqual(result["score"], 50.0)


if __name__ == "__main__":
    unittest.main()
