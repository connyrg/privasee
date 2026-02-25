"""
MockDatabricksClient — local reference implementation of the Databricks endpoint.

This class produces responses that are structurally identical to what the real
DocumentIntelligenceModel.predict() will return.  It validates every response
it generates against DATABRICKS_RESPONSE_SCHEMA so that any drift between the
mock and the real endpoint schema is caught immediately when contract tests run.

Usage (directly, not through FastAPI):

    client = MockDatabricksClient()
    response = client.process_document(session_id, field_definitions)
    # response is guaranteed to conform to DATABRICKS_RESPONSE_SCHEMA

Field definition input format
------------------------------
field_definitions may be a list of:
  - dicts with either "strategy" (display name) or "replacement_strategy"
    (internal code) keys, e.g.:
      {"name": "Full Name", "strategy": "Fake Data"}
      {"name": "Full Name", "replacement_strategy": "fake_name"}
  - FieldDefinition model objects (strategy is a ReplacementStrategy enum)
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

from tests.contracts.databricks_response_schema import validate_databricks_response

# ---------------------------------------------------------------------------
# Strategy normalisation
# ---------------------------------------------------------------------------

# Maps every accepted strategy representation → the three canonical display names
# required by the Databricks contract schema.
_STRATEGY_DISPLAY: Dict[str, str] = {
    # Internal codes (used in conftest sample_field_definitions)
    "fake_name":    "Fake Data",
    "redact":       "Black Out",
    "entity_label": "Entity Label",
    # Display names (used when FieldDefinition.strategy is already normalised)
    "Fake Data":    "Fake Data",
    "Black Out":    "Black Out",
    "Entity Label": "Entity Label",
}

_DEFAULT_STRATEGY = "Black Out"

# ---------------------------------------------------------------------------
# Representative mock values per field type
# ---------------------------------------------------------------------------

# Keys are lowercase substrings; first key that is contained in (or contains)
# the field name's lowercase form wins.
_MOCK_FIELD_DATA: Dict[str, tuple] = {
    "full name":       ("John Smith",                   "Jane Doe"),
    "first name":      ("John",                         "Jane"),
    "last name":       ("Smith",                        "Doe"),
    "name":            ("John Smith",                   "Jane Doe"),
    "date of birth":   ("01/15/1985",                   "07/22/1990"),
    "dob":             ("01/15/1985",                   "07/22/1990"),
    "ssn":             ("123-45-6789",                  "987-65-4321"),
    "social security": ("123-45-6789",                  "987-65-4321"),
    "email":           ("john.smith@example.com",        "j.doe@example.org"),
    "phone":           ("(555) 123-4567",               "(555) 987-6543"),
    "address":         ("123 Main Street, Springfield", "456 Oak Ave, Shelbyville"),
    "employer":        ("Acme Corporation",             "Globex Industries"),
    "company":         ("Acme Corporation",             "Globex Industries"),
    "job title":       ("Senior Analyst",               "Principal Consultant"),
    "license":         ("D1234567",                     "X9876543"),
    "passport":        ("AB1234567",                    "CD9876543"),
    "credit card":     ("4111 1111 1111 1111",          "5500 0000 0000 0004"),
    "bank account":    ("0001234567890",                "0009876543210"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_name(field_def: Union[Dict[str, Any], Any]) -> str:
    if isinstance(field_def, dict):
        return field_def.get("name", "Unknown")
    return getattr(field_def, "name", "Unknown")


def _extract_strategy(field_def: Union[Dict[str, Any], Any]) -> str:
    """Return a canonical strategy display name for the field definition."""
    if isinstance(field_def, dict):
        # Accept either key; "strategy" takes precedence if both present.
        raw = field_def.get("strategy") or field_def.get("replacement_strategy", "redact")
    else:
        strategy_attr = getattr(field_def, "strategy", None)
        if strategy_attr is None:
            raw = "redact"
        elif hasattr(strategy_attr, "value"):
            # ReplacementStrategy enum → use the .value string
            raw = strategy_attr.value
        else:
            raw = str(strategy_attr)

    return _STRATEGY_DISPLAY.get(raw, _DEFAULT_STRATEGY)


def _mock_values(field_name: str) -> tuple:
    """Return (original_text, replacement_text) for a field name."""
    lookup = field_name.lower()
    for key, pair in _MOCK_FIELD_DATA.items():
        if key in lookup or lookup in key:
            return pair
    # Fallback for unrecognised field names
    return (f"Sample {field_name}", f"[{field_name.upper()}]")


# ---------------------------------------------------------------------------
# MockDatabricksClient
# ---------------------------------------------------------------------------

class MockDatabricksClient:
    """
    Produces Databricks-schema-compliant responses without calling Databricks.

    Every response is validated against DATABRICKS_RESPONSE_SCHEMA before
    being returned, so a structural drift will raise jsonschema.ValidationError
    immediately rather than causing a silent false pass in consumer code.
    """

    def process_document(
        self,
        session_id: str,
        field_definitions: List[Union[Dict[str, Any], Any]],
    ) -> Dict[str, Any]:
        """
        Generate a mock entity-extraction response for the given session.

        One entity is generated per field definition, placed on page 1 at
        vertically distributed bounding boxes so they do not overlap visually.

        Args:
            session_id:        The session identifier; echoed back in the response.
            field_definitions: List of FieldDefinition dicts or model objects.

        Returns:
            A dict conforming to DATABRICKS_RESPONSE_SCHEMA.

        Raises:
            jsonschema.ValidationError: if the generated response does not
                match the schema (indicates a bug in this mock).
        """
        entities = []
        for i, field_def in enumerate(field_definitions):
            name = _extract_name(field_def)
            strategy = _extract_strategy(field_def)
            original, replacement = _mock_values(name)

            entities.append(
                {
                    "id": f"{session_id}_mock_{i}",
                    "entity_type": name,
                    "original_text": original,
                    "replacement_text": replacement,
                    "strategy": strategy,
                    "confidence": 0.95,
                    "approved": True,
                    "bounding_boxes": [
                        {
                            "x": 0.05,
                            "y": round(0.08 + i * 0.07, 4),
                            "width": 0.45,
                            "height": 0.025,
                        }
                    ],
                }
            )

        response: Dict[str, Any] = {
            "session_id": session_id,
            "status": "complete",
            "pages": [
                {
                    "page_num": 1,
                    "entities": entities,
                }
            ],
        }

        # Self-validate: any schema drift raises immediately.
        validate_databricks_response(response)

        return response
