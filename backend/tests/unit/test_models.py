"""
Unit tests for DatabricksProcessResponse.from_mlflow_response().

Verifies that the response parser correctly handles the pages-based format
returned by the Databricks model and the flat-entities fallback used in
local mocking.
"""

import pytest

from app.models import DatabricksProcessResponse


# ---------------------------------------------------------------------------
# Minimal valid entity dict (matches Entity model required fields)
# ---------------------------------------------------------------------------

_ENTITY = {
    "id": "e1",
    "entity_type": "Full Name",
    "original_text": "John Smith",
    "replacement_text": "Jane Doe",
}


def _entity(**overrides) -> dict:
    """Return a fresh copy of _ENTITY, avoiding cross-test mutation."""
    return {**_ENTITY, **overrides}


# ===========================================================================
# pages[].entities format — primary model output
# ===========================================================================


@pytest.mark.unit
def test_from_mlflow_response_flattens_pages_entities():
    """Entities from all pages are merged into a single flat list."""
    raw = {
        "predictions": [
            {
                "session_id": "abc",
                "status": "complete",
                "pages": [
                    {"page_num": 1, "entities": [_entity()]},
                    {"page_num": 2, "entities": [_entity(id="e2")]},
                ],
            }
        ]
    }
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert len(result.entities) == 2
    assert result.entities[0].id == "e1"
    assert result.entities[1].id == "e2"


@pytest.mark.unit
def test_from_mlflow_response_handles_empty_pages():
    """A pages list with no entities must produce an empty entity list."""
    raw = {
        "predictions": [
            {
                "session_id": "abc",
                "status": "complete",
                "pages": [{"page_num": 1, "entities": []}],
            }
        ]
    }

    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert result.entities == []


# ===========================================================================
# Flat {"entities": [...]} fallback — used by mock / local testing
# ===========================================================================


@pytest.mark.unit
def test_from_mlflow_response_handles_flat_entities_fallback():
    """Bare predictions[0].entities list (no pages key) is still accepted."""
    raw = {"predictions": [{"entities": [_entity()]}]}
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert len(result.entities) == 1
    assert result.entities[0].id == "e1"


@pytest.mark.unit
def test_from_mlflow_response_bare_dict_fallback():
    """A response with no predictions wrapper is parsed as a bare dict."""
    raw = {"entities": [_entity()]}
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert len(result.entities) == 1


# ===========================================================================
# model_version passthrough
# ===========================================================================


@pytest.mark.unit
def test_from_mlflow_response_extracts_model_version():
    """model_version is extracted from the prediction record when present."""
    raw = {
        "predictions": [
            {
                "pages": [{"page_num": 1, "entities": [_ENTITY]}],
                "model_version": "7",
            }
        ]
    }
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert result.model_version == "7"


@pytest.mark.unit
def test_from_mlflow_response_model_version_defaults_to_none():
    """model_version is None when not present in the response."""
    raw = {"predictions": [{"pages": [{"page_num": 1, "entities": [_ENTITY]}]}]}
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert result.model_version is None


# ===========================================================================
# occurrences round-trip
# ===========================================================================


@pytest.mark.unit
def test_from_mlflow_response_preserves_occurrences():
    """occurrences (with bounding_boxes) must survive Entity parsing intact."""
    occurrences = [
        {
            "page_number": 1,
            "original_text": "John Smith",
            "bounding_boxes": [[0.1, 0.2, 0.3, 0.04], [0.5, 0.6, 0.3, 0.04]],
        }
    ]
    entity = _entity(occurrences=occurrences)
    raw = {"predictions": [{"pages": [{"page_num": 1, "entities": [entity]}]}]}

    result = DatabricksProcessResponse.from_mlflow_response(raw)

    occ = result.entities[0].occurrences[0]
    assert occ.page_number == 1
    assert occ.bounding_boxes == [[0.1, 0.2, 0.3, 0.04], [0.5, 0.6, 0.3, 0.04]]
