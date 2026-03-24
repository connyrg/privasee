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


# ===========================================================================
# Error status handling
# ===========================================================================


@pytest.mark.unit
def test_from_mlflow_response_raises_runtime_error_on_error_status():
    """status='error' in the prediction record must raise RuntimeError with the error_message."""
    raw = {
        "predictions": [
            {
                "status": "error",
                "error_message": "ADI timeout after 30s",
            }
        ]
    }
    with pytest.raises(RuntimeError, match="ADI timeout after 30s"):
        DatabricksProcessResponse.from_mlflow_response(raw)


@pytest.mark.unit
def test_from_mlflow_response_error_status_uses_default_message_when_no_error_message():
    """status='error' with no error_message uses a generic fallback RuntimeError message."""
    raw = {"predictions": [{"status": "error"}]}
    with pytest.raises(RuntimeError, match="Entity extraction failed"):
        DatabricksProcessResponse.from_mlflow_response(raw)


@pytest.mark.unit
def test_from_mlflow_response_non_error_status_is_not_raised():
    """status='complete' (not 'error') must not raise; entities are parsed normally."""
    raw = {
        "predictions": [
            {
                "status": "complete",
                "entities": [_entity()],
            }
        ]
    }
    result = DatabricksProcessResponse.from_mlflow_response(raw)
    assert len(result.entities) == 1


# ===========================================================================
# FieldDefinition validation
# ===========================================================================


@pytest.mark.unit
def test_field_definition_rejects_empty_name():
    """FieldDefinition must raise when name is empty or whitespace-only."""
    from pydantic import ValidationError
    from app.models import FieldDefinition
    with pytest.raises(ValidationError):
        FieldDefinition(name="", description="A description", strategy="Fake Data")


@pytest.mark.unit
def test_field_definition_rejects_whitespace_only_description():
    """FieldDefinition must raise when description is blank."""
    from pydantic import ValidationError
    from app.models import FieldDefinition
    with pytest.raises(ValidationError):
        FieldDefinition(name="Full Name", description="   ", strategy="Fake Data")


@pytest.mark.unit
def test_field_definition_strips_name_whitespace():
    """Leading/trailing whitespace in name is stripped, not rejected."""
    from app.models import FieldDefinition
    fd = FieldDefinition(name="  Full Name  ", description="A name", strategy="Fake Data")
    assert fd.name == "Full Name"


@pytest.mark.unit
def test_field_definition_default_strategy_is_fake_data():
    """Strategy defaults to Fake Data when omitted."""
    from app.models import FieldDefinition
    fd = FieldDefinition(name="Email", description="An email address")
    assert fd.strategy.value == "Fake Data"


# ===========================================================================
# Entity confidence bounds
# ===========================================================================


@pytest.mark.unit
def test_entity_confidence_out_of_range_raises():
    """confidence must be 0.0–1.0; values outside that range must raise."""
    from pydantic import ValidationError
    from app.models import Entity
    with pytest.raises(ValidationError):
        Entity(id="e1", entity_type="Full Name", original_text="Alice", confidence=1.5)


@pytest.mark.unit
def test_entity_confidence_at_boundary_values():
    """confidence=0.0 and confidence=1.0 are both valid."""
    from app.models import Entity
    e0 = Entity(id="e1", entity_type="Full Name", original_text="Alice", confidence=0.0)
    e1 = Entity(id="e2", entity_type="Full Name", original_text="Bob",   confidence=1.0)
    assert e0.confidence == 0.0
    assert e1.confidence == 1.0


# ===========================================================================
# BoundingBox validation
# ===========================================================================


@pytest.mark.unit
def test_bounding_box_rejects_values_outside_unit_range():
    """BoundingBox coordinates must all be in [0, 1]."""
    from pydantic import ValidationError
    from app.models import BoundingBox
    with pytest.raises(ValidationError):
        BoundingBox(x=1.5, y=0.0, width=0.3, height=0.1)


@pytest.mark.unit
def test_bounding_box_from_list_requires_four_values():
    """BoundingBox.from_list must raise for lists that are not length 4."""
    from app.models import BoundingBox
    with pytest.raises(ValueError):
        BoundingBox.from_list([0.1, 0.2, 0.3])


@pytest.mark.unit
def test_bounding_box_to_list_round_trips():
    """to_list and from_list must be inverses of each other."""
    from app.models import BoundingBox
    values = [0.1, 0.2, 0.3, 0.04]
    bb = BoundingBox.from_list(values)
    assert bb.to_list() == values
