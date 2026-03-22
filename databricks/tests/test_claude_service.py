"""
Tests for Claude Vision Service

This test suite verifies:
1. Service initialization with API key validation
2. Entity extraction with Claude Vision API
3. Prompt building with OCR context
4. JSON response parsing — occurrences-based format
5. Error handling for file operations and API failures

All Anthropic API calls are mocked.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch, MagicMock, mock_open
import json

# Mock the anthropic module before importing claude_service
import sys
sys.modules['anthropic'] = MagicMock()
sys.modules['anthropic.types'] = MagicMock()

from databricks.model.claude_service import ClaudeVisionService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bbox(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def make_occurrence(original_text, bboxes):
    return {"original_text": original_text, "bounding_boxes": bboxes}


def make_entity(entity_type, canonical_text, occurrences, confidence=0.9):
    return {
        "entity_type": entity_type,
        "original_text": canonical_text,
        "confidence": confidence,
        "occurrences": occurrences,
    }


# ===========================================================================
# Initialization
# ===========================================================================

class TestClaudeServiceInit(unittest.TestCase):

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def test_init_success(self, mock_anthropic):
        service = ClaudeVisionService(api_key="test-api-key-123")
        mock_anthropic.assert_called_once_with(api_key="test-api-key-123")
        self.assertIsNotNone(service.client)

    def test_init_missing_api_key(self):
        with self.assertRaises(ValueError) as ctx:
            ClaudeVisionService(api_key="")
        self.assertIn("API key must be provided", str(ctx.exception))

    def test_init_none_api_key(self):
        with self.assertRaises(ValueError) as ctx:
            ClaudeVisionService(api_key=None)
        self.assertIn("API key must be provided", str(ctx.exception))


# ===========================================================================
# Prompt building
# ===========================================================================

class TestBuildExtractionPrompt(unittest.TestCase):

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        self.service = ClaudeVisionService(api_key="test-key")

    def test_prompt_includes_field_definitions(self):
        field_definitions = [
            {"name": "Full Name", "description": "Person's full name"},
            {"name": "SSN", "description": "Social Security Number"}
        ]
        ocr_data = {"text": "Test document", "words": []}
        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)
        self.assertIn("Full Name", prompt)
        self.assertIn("Person's full name", prompt)
        self.assertIn("SSN", prompt)
        self.assertIn("Social Security Number", prompt)

    def test_prompt_includes_ocr_text(self):
        field_definitions = [{"name": "Name", "description": "Person name"}]
        ocr_data = {
            "text": "John Doe lives at 123 Main St",
            "words": [
                {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.02}},
                {"text": "Doe", "bounding_box": {"x": 0.16, "y": 0.1, "width": 0.04, "height": 0.02}}
            ]
        }
        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)
        self.assertIn("John Doe lives at 123 Main St", prompt)
        self.assertIn("word_count", prompt)
        self.assertIn("words", prompt)

    def test_prompt_truncates_long_text(self):
        field_definitions = [{"name": "Name", "description": "Person name"}]
        long_text = "A" * 5000
        ocr_data = {"text": long_text, "words": []}
        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)
        self.assertIn("A" * 3000, prompt)
        self.assertNotIn("A" * 3001, prompt)

    def test_prompt_includes_all_words(self):
        field_definitions = [{"name": "Name", "description": "Person name"}]
        words = [{"text": f"word{i}", "bounding_box": {}} for i in range(100)]
        ocr_data = {"text": "A" * 5000, "words": words}
        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)
        self.assertIn('"word_count": 100', prompt)
        self.assertIn('"words":', prompt)

    def test_prompt_includes_occurrences_format(self):
        """Prompt output format example must use occurrences/bounding_boxes structure."""
        field_definitions = [{"name": "Name", "description": "A name"}]
        ocr_data = {"text": "", "words": []}
        prompt = self.service._build_extraction_prompt(field_definitions, ocr_data)
        self.assertIn("occurrences", prompt)
        self.assertIn("bounding_boxes", prompt)


# ===========================================================================
# _parse_claude_response
# ===========================================================================

class TestParseClaudeResponse(unittest.TestCase):

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        self.service = ClaudeVisionService(api_key="test-key")
        self.ocr_data = {"text": "Test", "words": []}

    def test_parse_single_entity_single_occurrence(self):
        payload = [
            make_entity("Full Name", "John Doe", [
                make_occurrence("John Doe", [make_bbox(0.1, 0.2, 0.05, 0.02), make_bbox(0.16, 0.2, 0.04, 0.02)])
            ], confidence=0.95)
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data, page_number=1)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Full Name")
        self.assertEqual(entities[0]['original_text'], "John Doe")
        self.assertEqual(entities[0]['confidence'], 0.95)
        self.assertEqual(len(entities[0]['occurrences']), 1)
        occ = entities[0]['occurrences'][0]
        self.assertEqual(occ['page_number'], 1)
        # Two same-line tokens merged into one rect
        self.assertEqual(len(occ['bounding_boxes']), 1)

    def test_parse_single_entity_two_occurrences(self):
        payload = [
            make_entity("Full Name", "Stephen Parrot", [
                make_occurrence("Stephen Parrot", [make_bbox(0.1, 0.1, 0.1, 0.02)]),
                make_occurrence("Stephen Parrot", [make_bbox(0.1, 0.5, 0.1, 0.02)]),
            ])
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data, page_number=1)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['original_text'], "Stephen Parrot")
        self.assertEqual(len(entities[0]['occurrences']), 2)

    def test_parse_typo_variant_uses_occurrence_original_text(self):
        """Occurrences with different spellings retain their own original_text."""
        payload = [
            make_entity("Full Name", "Kranthi", [
                make_occurrence("Kranthi", [make_bbox(0.1, 0.1, 0.05, 0.02)]),
                make_occurrence("Kranti",  [make_bbox(0.1, 0.5, 0.05, 0.02)]),
            ])
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data, page_number=1)
        self.assertEqual(len(entities), 1)
        occs = entities[0]['occurrences']
        self.assertEqual(len(occs), 2)
        self.assertEqual(occs[0]['original_text'], "Kranthi")
        self.assertEqual(occs[1]['original_text'], "Kranti")

    def test_parse_occurrence_missing_original_text_falls_back_to_canonical(self):
        payload = [
            make_entity("Full Name", "John Doe", [
                {"bounding_boxes": [make_bbox(0.1, 0.2, 0.1, 0.02)]}  # no original_text
            ])
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data, page_number=2)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['original_text'], "John Doe")
        self.assertEqual(entities[0]['occurrences'][0]['page_number'], 2)

    def test_parse_line_break_produces_two_bboxes(self):
        """Tokens on different lines stay as two separate merged rects."""
        payload = [
            make_entity("Full Name", "John Doe", [
                make_occurrence("John Doe", [
                    make_bbox(0.8, 0.20, 0.05, 0.02),  # "John" end of line 1
                    make_bbox(0.1, 0.25, 0.04, 0.02),  # "Doe"  start of line 2
                ])
            ])
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data, page_number=1)
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0]['occurrences'][0]['bounding_boxes']), 2)

    def test_parse_date_five_tokens_merged_into_one_bbox(self):
        """Five tokens on the same line are merged into one rect."""
        bboxes = [
            make_bbox(0.10, 0.30, 0.02, 0.02),
            make_bbox(0.12, 0.30, 0.005, 0.02),
            make_bbox(0.13, 0.30, 0.02, 0.02),
            make_bbox(0.15, 0.30, 0.005, 0.02),
            make_bbox(0.16, 0.30, 0.03, 0.02),
        ]
        payload = [make_entity("Date of Birth", "15/03/1985", [
            make_occurrence("15/03/1985", bboxes)
        ])]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data, page_number=1)
        self.assertEqual(len(entities), 1)
        self.assertEqual(len(entities[0]['occurrences'][0]['bounding_boxes']), 1)

    def test_parse_json_with_markdown_json_fence(self):
        payload = [make_entity("SSN", "123-45-6789", [
            make_occurrence("123-45-6789", [make_bbox(0.5, 0.5, 0.1, 0.02)])
        ], confidence=0.98)]
        response = f"```json\n{json.dumps(payload)}\n```"
        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=2)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "SSN")
        self.assertEqual(entities[0]['occurrences'][0]['page_number'], 2)

    def test_parse_json_with_generic_fence(self):
        payload = [make_entity("Address", "123 Main St", [
            make_occurrence("123 Main St", [make_bbox(0.2, 0.3, 0.2, 0.04)])
        ])]
        response = f"```\n{json.dumps(payload)}\n```"
        entities = self.service._parse_claude_response(response, self.ocr_data, page_number=1)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Address")

    def test_parse_empty_array(self):
        entities = self.service._parse_claude_response("[]", self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_invalid_json_returns_empty_list(self):
        entities = self.service._parse_claude_response("not valid json {[}", self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_non_array_json_returns_empty_list(self):
        entities = self.service._parse_claude_response('{"entity_type": "Name"}', self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_entity_missing_entity_type_is_skipped(self):
        payload = [
            {"original_text": "John", "confidence": 0.9, "occurrences": [
                make_occurrence("John", [make_bbox(0.1, 0.1, 0.05, 0.02)])
            ]}
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_entity_missing_original_text_is_skipped(self):
        payload = [
            {"entity_type": "Full Name", "confidence": 0.9, "occurrences": [
                make_occurrence("John", [make_bbox(0.1, 0.1, 0.05, 0.02)])
            ]}
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_entity_no_occurrences_is_skipped(self):
        payload = [make_entity("Full Name", "John Doe", [])]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_occurrence_empty_bboxes_is_skipped(self):
        payload = [make_entity("Full Name", "John Doe", [
            make_occurrence("John Doe", [])
        ])]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertEqual(entities, [])

    def test_parse_default_confidence(self):
        payload = [{
            "entity_type": "Name",
            "original_text": "Test",
            "occurrences": [make_occurrence("Test", [make_bbox(0.1, 0.2, 0.05, 0.02)])]
        }]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['confidence'], 0.9)

    def test_parse_multiple_entities(self):
        payload = [
            make_entity("Full Name", "Alice Smith", [
                make_occurrence("Alice Smith", [make_bbox(0.1, 0.1, 0.1, 0.02)])
            ]),
            make_entity("Full Name", "Bob Jones", [
                make_occurrence("Bob Jones", [make_bbox(0.1, 0.2, 0.1, 0.02)])
            ]),
        ]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]['original_text'], "Alice Smith")
        self.assertEqual(entities[1]['original_text'], "Bob Jones")

    def test_output_uses_occurrences_with_bounding_boxes(self):
        """Output must use occurrences[].bounding_boxes — no entity-level bbox keys."""
        payload = [make_entity("Full Name", "John Doe", [
            make_occurrence("John Doe", [make_bbox(0.1, 0.2, 0.1, 0.02)])
        ])]
        entities = self.service._parse_claude_response(json.dumps(payload), self.ocr_data)
        self.assertIn('occurrences', entities[0])
        self.assertNotIn('bounding_box', entities[0])
        self.assertNotIn('bounding_boxes', entities[0])
        self.assertNotIn('page_number', entities[0])
        occ = entities[0]['occurrences'][0]
        self.assertIn('bounding_boxes', occ)
        self.assertIn('page_number', occ)


# ===========================================================================
# extract_entities (file-based, async)
# ===========================================================================

class TestExtractEntities(unittest.TestCase):

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        self.service = ClaudeVisionService(api_key="test-key")
        self.mock_client = Mock()
        self.service.client = self.mock_client

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @patch('databricks.model.claude_service.base64.standard_b64encode')
    def test_extract_entities_success(self, mock_b64encode, mock_file):
        mock_b64encode.return_value = b'encoded_image_data'

        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps([
            make_entity("Full Name", "John Doe", [
                make_occurrence("John Doe", [make_bbox(0.1, 0.2, 0.1, 0.02)])
            ], confidence=0.95)
        ])
        mock_message.content = [mock_content]
        self.mock_client.messages.create.return_value = mock_message

        entities = asyncio.run(self.service.extract_entities(
            "/fake/image.png",
            {"text": "John Doe", "words": []},
            [{"name": "Full Name", "description": "Person's full name"}],
            page_number=1,
        ))

        mock_file.assert_called_once_with("/fake/image.png", "rb")
        self.mock_client.messages.create.assert_called_once()
        call_args = self.mock_client.messages.create.call_args
        self.assertEqual(call_args.kwargs['model'], 'claude-sonnet-4-6')
        content = call_args.kwargs['messages'][0]['content']
        self.assertEqual(content[0]['type'], 'image')
        self.assertEqual(content[1]['type'], 'text')

        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]['entity_type'], "Full Name")
        self.assertEqual(entities[0]['original_text'], "John Doe")
        self.assertEqual(entities[0]['occurrences'][0]['page_number'], 1)

    @patch('builtins.open', side_effect=FileNotFoundError("File not found"))
    def test_extract_entities_file_not_found(self, mock_file):
        with self.assertRaises(FileNotFoundError):
            asyncio.run(self.service.extract_entities(
                "/nonexistent/image.png", {"text": "Test", "words": []},
                [{"name": "Name", "description": "Name"}]
            ))

    @patch('builtins.open', new_callable=mock_open, read_data=b'fake_image_data')
    @patch('databricks.model.claude_service.base64.standard_b64encode')
    def test_extract_entities_api_error(self, mock_b64encode, mock_file):
        mock_b64encode.return_value = b'encoded_image_data'
        self.mock_client.messages.create.side_effect = Exception("API Error")
        with self.assertRaises(Exception) as ctx:
            asyncio.run(self.service.extract_entities(
                "/fake/image.png", {"text": "Test", "words": []},
                [{"name": "Name", "description": "Name"}]
            ))
        self.assertIn("Entity extraction failed", str(ctx.exception))


# ===========================================================================
# extract_entities_from_base64_async
# ===========================================================================

class TestExtractEntitiesFromBase64Async(unittest.TestCase):

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        self.service = ClaudeVisionService(api_key="test-key")
        self.mock_async_client = AsyncMock()
        self.service.async_client = self.mock_async_client

    def _make_api_response(self, entities):
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps(entities)
        mock_message.content = [mock_content]
        return mock_message

    def test_returns_entities_on_success(self):
        payload = [make_entity("Full Name", "Alice Brown", [
            make_occurrence("Alice Brown", [make_bbox(0.1, 0.2, 0.15, 0.03)])
        ], confidence=0.96)]
        self.mock_async_client.messages.create = AsyncMock(
            return_value=self._make_api_response(payload)
        )
        entities = asyncio.run(self.service.extract_entities_from_base64_async(
            "img_b64", "png", {"text": "Alice Brown", "words": []},
            [{"name": "Full Name", "description": "Person's full name"}], page_number=2
        ))
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["entity_type"], "Full Name")
        self.assertEqual(entities[0]["original_text"], "Alice Brown")
        self.assertEqual(entities[0]["occurrences"][0]["page_number"], 2)
        self.assertAlmostEqual(entities[0]["confidence"], 0.96)

    def test_passes_correct_model_and_image(self):
        self.mock_async_client.messages.create = AsyncMock(
            return_value=self._make_api_response([])
        )
        asyncio.run(self.service.extract_entities_from_base64_async(
            "encoded_img", "jpeg", {"text": "test", "words": []},
            [{"name": "Name", "description": "A name"}], page_number=1
        ))
        call_kwargs = self.mock_async_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "claude-sonnet-4-6")
        content = call_kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["source"]["media_type"], "image/jpeg")
        self.assertEqual(content[0]["source"]["data"], "encoded_img")
        self.assertEqual(content[1]["type"], "text")

    def test_returns_empty_list_on_api_error(self):
        self.mock_async_client.messages.create = AsyncMock(side_effect=Exception("Rate limit"))
        entities = asyncio.run(self.service.extract_entities_from_base64_async(
            "img_b64", "png", {}, [], page_number=1
        ))
        self.assertEqual(entities, [])

    def test_returns_empty_list_on_invalid_json_response(self):
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "this is not json"
        mock_message.content = [mock_content]
        self.mock_async_client.messages.create = AsyncMock(return_value=mock_message)
        entities = asyncio.run(self.service.extract_entities_from_base64_async(
            "img_b64", "png", {}, [], page_number=1
        ))
        self.assertEqual(entities, [])


# ===========================================================================
# test_connection
# ===========================================================================

class TestConnectionTest(unittest.TestCase):

    @patch('databricks.model.claude_service.anthropic.Anthropic')
    def setUp(self, mock_anthropic):
        self.service = ClaudeVisionService(api_key="test-key")
        self.mock_client = Mock()
        self.service.client = self.mock_client

    def test_connection_success(self):
        self.mock_client.messages.create.return_value = Mock()
        self.assertTrue(self.service.test_connection())
        self.mock_client.messages.create.assert_called_once()

    def test_connection_failure(self):
        self.mock_client.messages.create.side_effect = Exception("Connection failed")
        self.assertFalse(self.service.test_connection())


if __name__ == '__main__':
    unittest.main()
