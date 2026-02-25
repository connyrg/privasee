"""
Tests for DocumentIntelligenceModel.

TODO: implement tests covering:
  - predict() returns a list of entity dicts for a valid input
  - predict() raises ValueError for missing required input fields
  - load_context() initialises OCRService and ClaudeService correctly
  - End-to-end: a synthetic single-page image produces at least one entity
    when OCR and Claude are mocked to return fixture data
  - Model signature matches DatabricksInferenceRequest / DatabricksInferenceResponse

Use pytest fixtures with mocked Azure and Anthropic clients so tests run
without live API credentials.
"""

import pytest


# Placeholder — tests to be written in a subsequent step.
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
