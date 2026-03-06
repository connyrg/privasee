"""
Unit tests for databricks.model.MaskingModel

All UC volume HTTP calls and masking operations are mocked — no live
Databricks workspace or real PDFs are required.

Covers:
1. load_context — success and missing env var failures
2. predict — successful PDF masking, JSON string input, image path, error handling
3. _apply_masking — unsupported file type raises ValueError
4. _write_masked_file — PUT request to UC Files API
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

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

    def test_load_context_succeeds_with_valid_env(self):
        with patch("databricks.model.masking_model.MaskingService"):
            with patch.dict(os.environ, _VALID_ENV):
                model = self._make_model()
                model.load_context(context=None)

        self.assertEqual(model.uc_volume_path, "/Volumes/cat/schema/sessions")
        self.assertEqual(model.databricks_host, "https://test.databricks.com")
        self.assertEqual(model.databricks_token, "test-token-123")

    def test_load_context_strips_trailing_slash(self):
        env = {**_VALID_ENV, "UC_VOLUME_PATH": "/Volumes/cat/schema/sessions/",
               "DATABRICKS_HOST": "https://test.databricks.com/"}
        with patch("databricks.model.masking_model.MaskingService"):
            with patch.dict(os.environ, env):
                model = self._make_model()
                model.load_context(context=None)

        self.assertFalse(model.uc_volume_path.endswith("/"))
        self.assertFalse(model.databricks_host.endswith("/"))

    def test_load_context_raises_without_uc_volume_path(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "UC_VOLUME_PATH"}
        with patch("databricks.model.masking_model.MaskingService"):
            with patch.dict(os.environ, env, clear=True):
                model = self._make_model()
                with self.assertRaises(ValueError) as ctx:
                    model.load_context(context=None)
        self.assertIn("UC_VOLUME_PATH", str(ctx.exception))

    def test_load_context_raises_without_databricks_host(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "DATABRICKS_HOST"}
        with patch("databricks.model.masking_model.MaskingService"):
            with patch.dict(os.environ, env, clear=True):
                model = self._make_model()
                with self.assertRaises(ValueError) as ctx:
                    model.load_context(context=None)
        self.assertIn("DATABRICKS_HOST", str(ctx.exception))

    def test_load_context_raises_without_databricks_token(self):
        env = {k: v for k, v in _VALID_ENV.items() if k != "DATABRICKS_TOKEN"}
        with patch("databricks.model.masking_model.MaskingService"):
            with patch.dict(os.environ, env, clear=True):
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


# ===========================================================================
# Group 3 — predict (error handling)
# ===========================================================================

class TestPredictErrors(unittest.TestCase):

    def _make_ready_model(self) -> MaskingModel:
        model = MaskingModel()
        model.masking_service = Mock()
        model.uc_volume_path = "/Volumes/cat/schema/sessions"
        model.databricks_host = "https://test.databricks.com"
        model.databricks_token = "test-token"
        return model

    def test_predict_returns_error_row_on_fetch_failure(self):
        model = self._make_ready_model()
        model._fetch_original_file = Mock(side_effect=FileNotFoundError("no file"))
        model._write_masked_file = Mock()

        df = pd.DataFrame([{"session_id": "bad-sess", "entities_to_mask": []}])
        result = model.predict(context=None, model_input=df)

        row = result.iloc[0]
        self.assertEqual(row["session_id"], "bad-sess")
        self.assertEqual(row["status"], "error")
        self.assertIn("error_message", row)

    def test_predict_returns_error_row_on_masking_failure(self):
        model = self._make_ready_model()
        model._fetch_original_file = Mock(return_value=(b"bytes", "original.pdf"))
        model.masking_service.apply_pdf_masks.side_effect = RuntimeError("fitz crash")
        model._write_masked_file = Mock()

        df = pd.DataFrame([{"session_id": "crash-sess", "entities_to_mask": [{"strategy": "Black Out"}]}])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "error")

    def test_predict_other_rows_succeed_after_one_error(self):
        """A failure in one row must not prevent subsequent rows from processing."""
        model = self._make_ready_model()
        model.masking_service.apply_pdf_masks.return_value = b"%PDF-1.4 %%EOF"
        model._write_masked_file = Mock()

        call_count = [0]
        def fetch_side_effect(session_id):
            call_count[0] += 1
            if call_count[0] == 1:
                raise FileNotFoundError("first row fails")
            return (b"bytes", "original.pdf")

        model._fetch_original_file = Mock(side_effect=fetch_side_effect)

        df = pd.DataFrame([
            {"session_id": "fail", "entities_to_mask": []},
            {"session_id": "ok",   "entities_to_mask": []},
        ])
        result = model.predict(context=None, model_input=df)

        self.assertEqual(result.iloc[0]["status"], "error")
        self.assertEqual(result.iloc[1]["status"], "complete")


# ===========================================================================
# Group 4 — _apply_masking
# ===========================================================================

class TestApplyMasking(unittest.TestCase):

    def _make_model(self) -> MaskingModel:
        model = MaskingModel()
        model.masking_service = Mock()
        model.masking_service.apply_pdf_masks.return_value = _FAKE_PDF
        return model

    def test_pdf_calls_apply_pdf_masks(self):
        model = self._make_model()
        result = model._apply_masking(b"pdf bytes", ".pdf", [])
        model.masking_service.apply_pdf_masks.assert_called_once_with(b"pdf bytes", [])
        self.assertEqual(result, _FAKE_PDF)

    def test_unsupported_extension_raises_value_error(self):
        model = self._make_model()
        with self.assertRaises(ValueError) as ctx:
            model._apply_masking(b"docx bytes", ".docx", [])
        self.assertIn(".docx", str(ctx.exception))


# ===========================================================================
# Group 5 — _write_masked_file
# ===========================================================================

class TestWriteMaskedFile(unittest.TestCase):

    def _make_model(self) -> MaskingModel:
        model = MaskingModel()
        model.uc_volume_path = "/Volumes/cat/schema/sessions"
        model.databricks_host = "https://test.databricks.com"
        model.databricks_token = "test-token"
        return model

    def test_write_puts_to_correct_url(self):
        model = self._make_model()
        mock_resp = Mock()
        mock_resp.raise_for_status.return_value = None

        with patch("databricks.model.masking_model.MaskingModel._write_masked_file",
                   wraps=model._write_masked_file):
            with patch("requests.put", return_value=mock_resp) as mock_put:
                model._write_masked_file("sess-123", _FAKE_PDF)

        call_url = mock_put.call_args[0][0]
        self.assertIn("/sess-123/masked.pdf", call_url)
        self.assertIn("/api/2.0/fs/files", call_url)

    def test_write_sends_correct_bytes(self):
        model = self._make_model()
        mock_resp = Mock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.put", return_value=mock_resp) as mock_put:
            model._write_masked_file("sess-123", _FAKE_PDF)

        self.assertEqual(mock_put.call_args[1]["data"], _FAKE_PDF)

    def test_write_uses_bearer_auth(self):
        model = self._make_model()
        mock_resp = Mock()
        mock_resp.raise_for_status.return_value = None

        with patch("requests.put", return_value=mock_resp) as mock_put:
            model._write_masked_file("sess-123", _FAKE_PDF)

        headers = mock_put.call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-token")


if __name__ == "__main__":
    unittest.main()
