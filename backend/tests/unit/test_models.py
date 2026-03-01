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
    "bounding_box": [0.05, 0.08, 0.45, 0.025],
}


def _entity(**overrides) -> dict:
    """Return a fresh copy of _ENTITY, avoiding cross-test mutation via setdefault."""
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
def test_from_mlflow_response_sets_page_number_from_page_num():
    """page_number on each entity is sourced from its page's page_num."""
    raw = {
        "predictions": [
            {
                "pages": [
                    {"page_num": 3, "entities": [_entity()]},
                ]
            }
        ]
    }
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert result.entities[0].page_number == 3


@pytest.mark.unit
def test_from_mlflow_response_preserves_explicit_page_number():
    """An entity that already carries page_number must not be overwritten by page_num."""
    entity_with_page = _entity(page_number=5)
    raw = {
        "predictions": [
            {
                "pages": [{"page_num": 1, "entities": [entity_with_page]}]
            }
        ]
    }
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    # setdefault does not overwrite an existing page_number
    assert result.entities[0].page_number == 5


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


@pytest.mark.unit
def test_from_mlflow_response_multi_page_preserves_page_numbers():
    """Entities from different pages carry distinct page_number values."""
    raw = {
        "predictions": [
            {
                "pages": [
                    {"page_num": 1, "entities": [_entity()]},
                    {"page_num": 2, "entities": [_entity(id="e2")]},
                ]
            }
        ]
    }
    result = DatabricksProcessResponse.from_mlflow_response(raw)

    assert result.entities[0].page_number == 1
    assert result.entities[1].page_number == 2


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
