"""
Claude Service — Anthropic Claude Vision wrapper.

Migrated from the PoC (backend/app/services/claude_service.py) for use
inside the Databricks model.  The interface is intentionally identical to
the PoC so that existing tests can be reused.

Key changes vs. the PoC:
  - API key is injected at construction time by
    DocumentIntelligenceModel.load_context() (sourced from Databricks secrets)
  - Input is a base64-encoded PNG, matching the Model Serving REST interface

TODO:
  - Copy implementation from backend/app/services/claude_service.py in the PoC
  - Update model ID constant if a newer Claude model is preferred
  - Unit-test with a mocked Anthropic client
"""

# Default model — update to the latest available Claude model as needed.
DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeService:
    """Wraps the Anthropic Claude Vision API for entity extraction."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: Anthropic API key
            model:   Claude model ID
        """
        self.api_key = api_key
        self.model = model
        # TODO: initialise anthropic.Anthropic client

    def extract_entities(
        self,
        image_base64: str,
        ocr_data: dict,
        field_definitions: list,
    ) -> list:
        """
        Identify sensitive entities in the document image.

        Args:
            image_base64:      Base64-encoded PNG of a single page
            ocr_data:          Structured OCR output from OCRService
            field_definitions: List of field definition dicts

        Returns:
            List of entity dicts (type, original_text, replacement_text,
            bounding_box, confidence, page)

        TODO: implement
        """
        raise NotImplementedError

    def _build_extraction_prompt(self, ocr_data: dict, field_definitions: list) -> str:
        """Build the system/user prompt for Claude. TODO: implement."""
        raise NotImplementedError

    def _parse_claude_response(self, response_text: str) -> list:
        """Parse JSON entity list from Claude response. TODO: implement."""
        raise NotImplementedError
