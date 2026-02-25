"""
Unit tests for MaskingService.

Tests run entirely in-process with no filesystem I/O beyond a temp directory.
The test_pdf_bytes and test_image_bytes fixtures from conftest.py supply
known inputs; sample_entities supply normalised bounding boxes that match
the text positions in test_pdf_bytes.

TODO: implement tests covering:
  - apply_masks() writes an output file at the given path
  - apply_masks() with "redact" strategy renders a filled rectangle (no text)
  - apply_masks() with replacement text centres text within the bounding box
  - apply_masks() with entity_label strategy renders the entity_type string
  - _normalize_bbox() converts normalised [0,1] coords correctly to pixels
  - _normalize_bbox() clamps values to image boundaries (no out-of-bounds rects)
  - apply_masks() raises FileNotFoundError for a missing input path
  - apply_masks() with an empty entity list leaves the image unchanged
"""

import pytest


@pytest.mark.unit
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
