"""
Single source of truth for the Databricks Model Serving response contract.

Both MockDatabricksClient and any future integration tests against the real
endpoint must validate against DATABRICKS_RESPONSE_SCHEMA.

Schema version: bump SCHEMA_VERSION whenever the Databricks model's predict()
return value changes so that contract tests fail loudly on any mismatch.

Response format (success)
--------------------------
{
    "session_id": "<uuid>",
    "status": "complete",
    "pages": [
        {
            "page_num": 1,
            "entities": [
                {
                    "id": "<string>",
                    "entity_type": "<field name>",
                    "original_text": "<non-empty string>",
                    "replacement_text": "<string, may be empty>",
                    "strategy": "Fake Data" | "Black Out" | "Entity Label",
                    "confidence": <float 0-1>,
                    "approved": <bool>,
                    "bounding_boxes": [
                        {"x": <float>, "y": <float>,
                         "width": <float ≥ 0>, "height": <float ≥ 0>}
                    ]
                }
            ]
        }
    ]
}

Response format (error)
-----------------------
{
    "session_id": "<uuid>",
    "status": "error",
    "error_message": "<non-empty description>"
}
"""

import jsonschema

SCHEMA_VERSION = "2.0.0"

# ---------------------------------------------------------------------------
# Success response schema
# ---------------------------------------------------------------------------

DATABRICKS_RESPONSE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DatabricksProcessResponse",
    "description": (
        "Schema for the PrivaSee Databricks Model Serving success response. "
        f"Version {SCHEMA_VERSION}."
    ),
    "type": "object",
    "required": ["session_id", "status", "pages"],
    "additionalProperties": True,
    "properties": {
        "session_id": {
            "type": "string",
            "minLength": 1,
        },
        "status": {
            "type": "string",
            "enum": ["complete", "error"],
        },
        "error_message": {
            "type": "string",
            "description": "Only present when status is 'error'.",
        },
        "pages": {
            "type": "array",
            "minItems": 1,
            "items": {"$ref": "#/$defs/page"},
        },
    },
    "$defs": {
        "page": {
            "type": "object",
            "required": ["page_num", "entities"],
            "additionalProperties": True,
            "properties": {
                "page_num": {
                    "type": "integer",
                    "minimum": 1,
                },
                "entities": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/entity"},
                },
            },
        },
        "entity": {
            "type": "object",
            "required": [
                "id",
                "entity_type",
                "original_text",
                "replacement_text",
                "strategy",
                "confidence",
                "approved",
                "bounding_boxes",
            ],
            "additionalProperties": True,
            "properties": {
                "id": {
                    "type": "string",
                    "minLength": 1,
                },
                "entity_type": {
                    "type": "string",
                    "minLength": 1,
                },
                "original_text": {
                    "type": "string",
                    "minLength": 1,
                },
                "replacement_text": {
                    # May be empty string when replacement has not yet been generated
                    "type": "string",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["Fake Data", "Black Out", "Entity Label"],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "approved": {
                    "type": "boolean",
                },
                "bounding_boxes": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/bounding_box"},
                },
            },
        },
        "bounding_box": {
            "type": "object",
            "required": ["x", "y", "width", "height"],
            "additionalProperties": True,
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number", "minimum": 0},
                "height": {"type": "number", "minimum": 0},
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Error response schema
# ---------------------------------------------------------------------------

DATABRICKS_ERROR_RESPONSE_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DatabricksErrorResponse",
    "description": "Schema for the PrivaSee Databricks Model Serving error response.",
    "type": "object",
    "required": ["session_id", "status", "error_message"],
    "additionalProperties": True,
    "properties": {
        "session_id": {
            "type": "string",
            "minLength": 1,
        },
        "status": {
            "type": "string",
            "enum": ["error"],
        },
        "error_message": {
            "type": "string",
            "minLength": 1,
        },
    },
}


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_databricks_response(response: dict) -> None:
    """
    Validate a Databricks endpoint response against the contract schema.

    Selects DATABRICKS_ERROR_RESPONSE_SCHEMA when status == "error",
    otherwise uses DATABRICKS_RESPONSE_SCHEMA.

    Args:
        response: The parsed JSON dict returned by the endpoint.

    Raises:
        jsonschema.ValidationError: with a prefixed message that identifies
            this as a contract violation, including the failing field path.
    """
    schema = (
        DATABRICKS_ERROR_RESPONSE_SCHEMA
        if response.get("status") == "error"
        else DATABRICKS_RESPONSE_SCHEMA
    )
    try:
        jsonschema.validate(instance=response, schema=schema)
    except jsonschema.ValidationError as exc:
        path = " → ".join(str(p) for p in exc.absolute_path) or "<root>"
        raise jsonschema.ValidationError(
            f"Databricks contract violation at '{path}': {exc.message}"
        ) from exc
