"""
Tests for BBoxMatcher.

TODO: implement tests covering:
  - match_entity() returns the exact Claude bbox when no OCR words overlap
  - match_entity() returns the union of two overlapping OCR words
  - match_entity() handles multi-word entities that span line breaks
  - _overlaps() correctly identifies overlapping and non-overlapping boxes
  - _bbox_union() returns the correct union for a list of boxes
  - match_all() updates bounding_box in-place for every entity in the list
  - tolerance parameter expands the search region as expected
"""

import pytest


# Placeholder — tests to be written in a subsequent step.
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
