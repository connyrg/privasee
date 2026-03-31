"""
Contract tests: MockDatabricksClient output conforms to DATABRICKS_RESPONSE_SCHEMA.

Run these tests whenever you update MockDatabricksClient OR whenever you update
the Databricks model's predict() return value.  A failure here means the mock
has drifted from the real endpoint and any passing unit tests may be giving
false confidence.

These tests instantiate MockDatabricksClient directly — not through FastAPI —
so they exercise the Databricks contract layer in complete isolation from
routing, authentication, and session storage.
"""

import pytest

from tests.contracts.databricks_response_schema import (
    DATABRICKS_RESPONSE_SCHEMA,
    validate_databricks_response,
)
from tests.contracts.mock_databricks_client import MockDatabricksClient


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------

@pytest.mark.contract
def test_mock_success_response_matches_schema(sample_field_definitions):
    """
    The full response for a typical field-definition set passes schema
    validation and satisfies the structural invariants that jsonschema alone
    cannot enforce.
    """
    client = MockDatabricksClient()
    response = client.process_document("test-session-schema-001", sample_field_definitions)

    # Schema validation (raises jsonschema.ValidationError on failure)
    validate_databricks_response(response)

    # Flatten entities across all pages for assertion convenience
    all_entities = [e for page in response["pages"] for e in page["entities"]]

    # One entity per field definition
    assert len(all_entities) == len(sample_field_definitions), (
        f"Expected {len(sample_field_definitions)} entities, got {len(all_entities)}"
    )

    # Each entity_type matches one of the field definition names
    field_names = {fd["name"] for fd in sample_field_definitions}
    for entity in all_entities:
        assert entity["entity_type"] in field_names, (
            f"entity_type '{entity['entity_type']}' not in field definitions {field_names}"
        )

    # Every entity has at least one occurrence with bounding_boxes
    for entity in all_entities:
        assert entity.get("occurrences"), (
            f"Entity '{entity['id']}' has no occurrences"
        )
        for occ in entity["occurrences"]:
            assert occ.get("bounding_boxes"), (
                f"Occurrence of '{entity['id']}' has no bounding_boxes"
            )
            for bbox in occ["bounding_boxes"]:
                assert len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox), (
                    f"bounding_box {bbox} is not a valid [x,y,w,h] list for entity '{entity['id']}'"
                )

    # All confidence values are in [0, 1]
    for entity in all_entities:
        assert 0.0 <= entity["confidence"] <= 1.0, (
            f"confidence {entity['confidence']} out of range for entity '{entity['id']}'"
        )


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------

@pytest.mark.contract
def test_mock_entity_types_match_field_definitions():
    """
    MockDatabricksClient is dynamic: it reflects the field definitions it is
    given, not a hardcoded list.  Passing three distinct field names produces
    exactly those three entity types in the response.
    """
    client = MockDatabricksClient()

    field_defs = [
        {
            "name": "Employee ID",
            "description": "Internal employee identifier",
            "strategy": "Black Out",
        },
        {
            "name": "Bank Account",
            "description": "Bank account number",
            "strategy": "Black Out",
        },
        {
            "name": "Passport Number",
            "description": "Passport document number",
            "strategy": "Entity Label",
        },
    ]

    response = client.process_document("test-session-schema-002", field_defs)

    # Response must still satisfy the contract
    validate_databricks_response(response)

    all_entities = [e for page in response["pages"] for e in page["entities"]]
    returned_types = {e["entity_type"] for e in all_entities}
    expected_types = {"Employee ID", "Bank Account", "Passport Number"}

    assert returned_types == expected_types, (
        f"Entity types mismatch.\n"
        f"  Expected: {sorted(expected_types)}\n"
        f"  Got:      {sorted(returned_types)}"
    )


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------

@pytest.mark.contract
def test_mock_handles_empty_field_definitions():
    """
    An empty field_definitions list is valid input: the mock returns a
    schema-compliant response with an empty entity list rather than raising.

    The "pages" array must still contain at least one page object (minItems=1
    in the schema), but that page's entities list may be empty.
    """
    client = MockDatabricksClient()
    response = client.process_document("test-session-schema-003", [])

    # Must still pass schema validation (pages: minItems=1, entities can be [])
    validate_databricks_response(response)

    assert response["status"] == "complete"
    assert len(response["pages"]) >= 1

    all_entities = [e for page in response["pages"] for e in page["entities"]]
    assert all_entities == [], (
        f"Expected empty entity list for empty field_definitions, got {all_entities}"
    )


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------

@pytest.mark.contract
def test_mock_session_id_is_preserved():
    """
    The session_id passed to process_document is echoed unchanged in the
    response.  FastAPI relies on this to write entities to the correct UC
    session folder after the Databricks call returns.
    """
    client = MockDatabricksClient()

    specific_session_id = "privasee-session-abc-123-xyz"
    field_defs = [
        {
            "name": "Full Name",
            "description": "Person's full legal name",
            "strategy": "Fake Data",
        }
    ]

    response = client.process_document(specific_session_id, field_defs)

    validate_databricks_response(response)

    assert response["session_id"] == specific_session_id, (
        f"session_id was not preserved.\n"
        f"  Sent:     '{specific_session_id}'\n"
        f"  Received: '{response['session_id']}'"
    )
