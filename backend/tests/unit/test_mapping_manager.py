"""
Unit tests for MappingManager.

All tests are pure in-process — no filesystem, network, or Databricks I/O.
MappingManager is seeded with a fixed integer so outputs are deterministic.

TODO: implement tests covering:
  - get_replacement("fake_name", text, "fake_name") returns a different string
  - get_replacement("ssn", text, "redact") returns "[REDACTED]"
  - get_replacement("email", text, "entity_label") returns "[EMAIL]" (or similar)
  - get_replacement() is idempotent for the same (entity_type, original_text) pair
    when using the same seed
  - get_replacement() returns a string for every strategy variant
    ("Fake Data", "fake_name", "Black Out", "redact", "Entity Label", "entity_label")
  - Unknown strategy falls back gracefully (no exception)
"""

import pytest


@pytest.mark.unit
def test_placeholder():
    """Remove once real tests are in place."""
    assert True
