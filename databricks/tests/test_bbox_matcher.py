"""
Tests for Bounding Box Matcher

This test suite verifies:
1. Single word entity matching
2. Multi-word entity matching
3. Duplicate entity occurrences (multiple matches)
4. Entity not found (fallback with empty bounding_boxes)
5. Case-insensitive matching
6. Whitespace normalization
7. Bounding box merging for multi-word entities
8. Invalid bounding box handling

The matcher is critical for entity redaction accuracy.
"""

import unittest
from databricks.model.bbox_matcher import BBoxMatcher, match_entities_to_words


class TestSingleWordEntity(unittest.TestCase):
    """Test matching single word entities to OCR words"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_single_word_exact_match(self):
        """Test exact match for single word entity"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "Hello", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.02}},
            {"text": "John", "bounding_box": {"x": 0.2, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.25, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['bounding_boxes']), 1)
        
        bbox = result[0]['bounding_boxes'][0]
        self.assertAlmostEqual(bbox['x'], 0.2, places=10)
        self.assertAlmostEqual(bbox['y'], 0.1, places=10)
        self.assertAlmostEqual(bbox['width'], 0.04, places=10)
        self.assertAlmostEqual(bbox['height'], 0.02, places=10)

    def test_single_word_case_insensitive(self):
        """Test case-insensitive matching for single word"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "JOHN",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "john", "bounding_box": {"x": 0.2, "y": 0.1, "width": 0.04, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result[0]['bounding_boxes']), 1)


class TestMultiWordEntity(unittest.TestCase):
    """Test matching multi-word entities"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_two_word_entity(self):
        """Test matching two-word entity"""
        entities = [
            {
                "entity_type": "Full Name",
                "original_text": "John Smith",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "Hello", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.02}},
            {"text": "John", "bounding_box": {"x": 0.2, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.25, "y": 0.1, "width": 0.05, "height": 0.02}},
            {"text": "here", "bounding_box": {"x": 0.31, "y": 0.1, "width": 0.04, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['bounding_boxes']), 1)
        
        # Bounding box should merge "John" and "Smith"
        bbox = result[0]['bounding_boxes'][0]
        self.assertAlmostEqual(bbox['x'], 0.2, places=10)  # min x
        self.assertAlmostEqual(bbox['y'], 0.1, places=10)  # min y
        self.assertAlmostEqual(bbox['width'], 0.1, places=10)  # (0.25 + 0.05) - 0.2
        self.assertAlmostEqual(bbox['height'], 0.02, places=10)  # max y + height - min y

    def test_three_word_entity(self):
        """Test matching three-word entity"""
        entities = [
            {
                "entity_type": "Full Name",
                "original_text": "Dr. John Smith",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "Dr.", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.03, "height": 0.02}},
            {"text": "John", "bounding_box": {"x": 0.14, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.19, "y": 0.1, "width": 0.05, "height": 0.02}},
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result[0]['bounding_boxes']), 1)
        
        # Bounding box should merge all three words
        bbox = result[0]['bounding_boxes'][0]
        self.assertAlmostEqual(bbox['x'], 0.1, places=10)  # min x
        self.assertAlmostEqual(bbox['width'], 0.14, places=10)  # (0.19 + 0.05) - 0.1


class TestDuplicateOccurrences(unittest.TestCase):
    """Test finding multiple occurrences of the same entity"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_duplicate_single_word(self):
        """Test finding duplicate single word entity"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "and", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.03, "height": 0.02}},
            {"text": "John", "bounding_box": {"x": 0.2, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "met", "bounding_box": {"x": 0.25, "y": 0.1, "width": 0.03, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result), 1)
        # Should find both occurrences
        self.assertEqual(len(result[0]['bounding_boxes']), 2)
        
        bbox1 = result[0]['bounding_boxes'][0]
        bbox2 = result[0]['bounding_boxes'][1]
        
        self.assertEqual(bbox1['x'], 0.1)
        self.assertEqual(bbox2['x'], 0.2)

    def test_duplicate_multi_word(self):
        """Test finding duplicate multi-word entity"""
        entities = [
            {
                "entity_type": "Full Name",
                "original_text": "John Smith",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.05, "height": 0.02}},
            {"text": "and", "bounding_box": {"x": 0.21, "y": 0.1, "width": 0.03, "height": 0.02}},
            {"text": "John", "bounding_box": {"x": 0.25, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.3, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result), 1)
        # Should find both occurrences
        self.assertEqual(len(result[0]['bounding_boxes']), 2)
        
        bbox1 = result[0]['bounding_boxes'][0]
        bbox2 = result[0]['bounding_boxes'][1]
        
        # First occurrence: "John Smith" at 0.1-0.2
        self.assertAlmostEqual(bbox1['x'], 0.1, places=10)
        self.assertAlmostEqual(bbox1['width'], 0.1, places=10)  # (0.15 + 0.05) - 0.1
        
        # Second occurrence: "John Smith" at 0.25-0.35
        self.assertAlmostEqual(bbox2['x'], 0.25, places=10)
        self.assertAlmostEqual(bbox2['width'], 0.1, places=10)  # (0.3 + 0.05) - 0.25

    def test_three_occurrences(self):
        """Test finding three occurrences of the same entity"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "Bob",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "Bob", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.03, "height": 0.02}},
            {"text": "Bob", "bounding_box": {"x": 0.2, "y": 0.1, "width": 0.03, "height": 0.02}},
            {"text": "Bob", "bounding_box": {"x": 0.3, "y": 0.1, "width": 0.03, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should find all three occurrences
        self.assertEqual(len(result[0]['bounding_boxes']), 3)


class TestEntityNotFound(unittest.TestCase):
    """Test fallback behavior when entity is not found"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_entity_not_found(self):
        """Test entity not found returns empty bounding_boxes"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "NotHere",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result), 1)
        # Should return empty bounding_boxes list
        self.assertEqual(result[0]['bounding_boxes'], [])

    def test_partial_match_not_found(self):
        """Test partial match does not return results"""
        entities = [
            {
                "entity_type": "Full Name",
                "original_text": "John Smith Jr",
                "page_number": 1
            }
        ]
        
        # Only "John Smith" exists, not "John Smith Jr"
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should not match partial text
        self.assertEqual(result[0]['bounding_boxes'], [])

    def test_empty_entity_text(self):
        """Test entity with empty original_text"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        self.assertEqual(result[0]['bounding_boxes'], [])


class TestCaseWhitespaceNormalization(unittest.TestCase):
    """Test case and whitespace normalization"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_case_differences(self):
        """Test matching with various case differences"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "JoHn SmItH",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "john", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "SMITH", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should match despite case differences
        self.assertEqual(len(result[0]['bounding_boxes']), 1)

    def test_extra_whitespace(self):
        """Test matching with extra whitespace"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John  Smith",  # Two spaces
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should match despite whitespace differences
        self.assertEqual(len(result[0]['bounding_boxes']), 1)

    def test_leading_trailing_whitespace(self):
        """Test matching with leading/trailing whitespace"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "  John Smith  ",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.15, "y": 0.1, "width": 0.05, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should match despite leading/trailing whitespace
        self.assertEqual(len(result[0]['bounding_boxes']), 1)


class TestBoundingBoxMerging(unittest.TestCase):
    """Test bounding box merging logic"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_merge_horizontal_boxes(self):
        """Test merging horizontally adjacent boxes"""
        words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.04, "height": 0.02}},
            {"text": "Smith", "bounding_box": {"x": 0.15, "y": 0.2, "width": 0.05, "height": 0.02}}
        ]

        merged = self.matcher._merge_bounding_boxes(words)

        self.assertAlmostEqual(merged['x'], 0.1, places=10)  # min x
        self.assertAlmostEqual(merged['y'], 0.2, places=10)  # min y
        self.assertAlmostEqual(merged['width'], 0.1, places=10)  # (0.15 + 0.05) - 0.1
        self.assertAlmostEqual(merged['height'], 0.02, places=10)  # (0.2 + 0.02) - 0.2

    def test_merge_vertical_boxes(self):
        """Test merging vertically stacked boxes"""
        words = [
            {"text": "First", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.02}},
            {"text": "Line", "bounding_box": {"x": 0.1, "y": 0.13, "width": 0.04, "height": 0.02}}
        ]

        merged = self.matcher._merge_bounding_boxes(words)

        self.assertAlmostEqual(merged['x'], 0.1, places=10)  # min x
        self.assertAlmostEqual(merged['y'], 0.1, places=10)  # min y
        self.assertAlmostEqual(merged['width'], 0.05, places=10)  # (0.1 + 0.05) - 0.1
        self.assertAlmostEqual(merged['height'], 0.05, places=10)  # (0.13 + 0.02) - 0.1

    def test_merge_different_heights(self):
        """Test merging boxes with different heights"""
        words = [
            {"text": "Small", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.05, "height": 0.02}},
            {"text": "TALL", "bounding_box": {"x": 0.16, "y": 0.09, "width": 0.04, "height": 0.04}}
        ]

        merged = self.matcher._merge_bounding_boxes(words)

        self.assertAlmostEqual(merged['x'], 0.1, places=10)  # min x
        self.assertAlmostEqual(merged['y'], 0.09, places=10)  # min y (TALL is higher)
        self.assertAlmostEqual(merged['width'], 0.1, places=10)  # (0.16 + 0.04) - 0.1
        self.assertAlmostEqual(merged['height'], 0.04, places=10)  # (0.1 + 0.02) - 0.09


class TestInvalidBoundingBoxes(unittest.TestCase):
    """Test handling of invalid bounding boxes"""

    def setUp(self):
        """Set up test matcher"""
        self.matcher = BBoxMatcher()

    def test_missing_bbox_field(self):
        """Test word missing bounding_box field"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John"}  # Missing bounding_box
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should handle gracefully
        self.assertEqual(result[0]['bounding_boxes'], [])

    def test_incomplete_bbox(self):
        """Test word with incomplete bounding_box"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.2}}  # Missing width/height
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should handle gracefully
        self.assertEqual(result[0]['bounding_boxes'], [])

    def test_non_numeric_bbox_values(self):
        """Test word with non-numeric bounding box values"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": "invalid", "y": 0.2, "width": 0.04, "height": 0.02}}
        ]

        result = self.matcher.match_entities_to_words(entities, ocr_words)

        # Should handle gracefully
        self.assertEqual(result[0]['bounding_boxes'], [])


class TestConvenienceFunction(unittest.TestCase):
    """Test the module-level convenience function"""

    def test_convenience_function(self):
        """Test match_entities_to_words convenience function"""
        entities = [
            {
                "entity_type": "Name",
                "original_text": "John",
                "page_number": 1
            }
        ]
        
        ocr_words = [
            {"text": "John", "bounding_box": {"x": 0.1, "y": 0.1, "width": 0.04, "height": 0.02}}
        ]

        result = match_entities_to_words(entities, ocr_words)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]['bounding_boxes']), 1)


if __name__ == '__main__':
    unittest.main()
