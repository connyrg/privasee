"""
Bounding Box Matcher for Databricks Model Serving

This module matches Claude-extracted entities to ADI OCR words and assigns
bounding boxes. Uses sliding window matching to find all occurrences of
multi-word entities and merges their individual word bounding boxes.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BBoxMatcher:
    """Matches entity text to OCR words and assigns bounding boxes."""

    def __init__(self):
        """Initialize bounding box matcher."""
        logger.info("BBox Matcher initialized")

    def match_entities_to_words(
        self,
        entities: List[Dict],
        ocr_words: List[Dict]
    ) -> List[Dict]:
        """
        Match entities to OCR words and assign bounding boxes.

        Uses sliding window approach to find all occurrences of each entity
        in the OCR words, then merges bounding boxes for multi-word entities.

        Args:
            entities: List of entities from Claude Vision
                [
                    {
                        "entity_type": "Full Name",
                        "original_text": "John Smith",
                        "page_number": 1,
                        ...
                    }
                ]
            ocr_words: List of words from OCR service
                [
                    {
                        "text": "John",
                        "bounding_box": {"x": 0.1, "y": 0.2, "width": 0.05, "height": 0.02}
                    },
                    ...
                ]

        Returns:
            List of entities with bounding_boxes field added:
            [
                {
                    "entity_type": "Full Name",
                    "original_text": "John Smith",
                    "page_number": 1,
                    "bounding_boxes": [
                        {"x": 0.1, "y": 0.2, "width": 0.15, "height": 0.02},
                        {"x": 0.3, "y": 0.5, "width": 0.15, "height": 0.02}  # second occurrence
                    ],
                    ...
                }
            ]
        """
        logger.info(
            f"Matching {len(entities)} entities to {len(ocr_words)} OCR words"
        )

        matched_entities = []

        for entity in entities:
            entity_text = entity.get('original_text', '')
            
            if not entity_text:
                logger.warning(f"Entity has empty original_text: {entity}")
                entity['bounding_boxes'] = []
                matched_entities.append(entity)
                continue

            # Find all matching bounding boxes for this entity
            bounding_boxes = self._find_all_matches(entity_text, ocr_words)

            if not bounding_boxes:
                logger.warning(
                    f"No match found for entity '{entity_text}' "
                    f"(page {entity.get('page_number', 'unknown')})"
                )

            # Add bounding boxes to entity
            entity['bounding_boxes'] = bounding_boxes
            matched_entities.append(entity)

        logger.info(
            f"Matched {sum(1 for e in matched_entities if e['bounding_boxes'])} "
            f"out of {len(entities)} entities"
        )

        return matched_entities

    def _find_all_matches(
        self,
        entity_text: str,
        ocr_words: List[Dict]
    ) -> List[Dict]:
        """
        Find all occurrences of entity text in OCR words using sliding window.

        Args:
            entity_text: Text to find (e.g., "John Smith")
            ocr_words: List of OCR word dictionaries

        Returns:
            List of merged bounding boxes for all occurrences
        """
        # Normalize entity text for matching
        normalized_entity = self._normalize_text(entity_text)
        entity_word_count = len(normalized_entity.split())

        bounding_boxes = []

        # Slide window across OCR words
        for i in range(len(ocr_words) - entity_word_count + 1):
            window_words = ocr_words[i:i + entity_word_count]
            
            # Join window words and normalize
            window_text = ' '.join([w.get('text', '') for w in window_words])
            normalized_window = self._normalize_text(window_text)

            # Check if window matches entity text
            if normalized_window == normalized_entity:
                # Merge bounding boxes for all words in the window
                merged_bbox = self._merge_bounding_boxes(window_words)
                if merged_bbox:
                    bounding_boxes.append(merged_bbox)

        return bounding_boxes

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching.

        Args:
            text: Text to normalize

        Returns:
            Normalized text (lowercase, stripped whitespace)
        """
        return ' '.join(text.lower().split())

    def _merge_bounding_boxes(self, words: List[Dict]) -> Optional[Dict]:
        """
        Merge multiple word bounding boxes into a single bounding box.

        The merged box covers all words:
        - x: minimum x across all boxes
        - y: minimum y across all boxes
        - width: (max x + width) - min x
        - height: (max y + height) - min y

        Args:
            words: List of word dictionaries with bounding_box field

        Returns:
            Merged bounding box dictionary or None if no valid boxes
        """
        valid_boxes = []
        
        for word in words:
            bbox = word.get('bounding_box')
            if bbox and self._is_valid_bbox(bbox):
                valid_boxes.append(bbox)

        if not valid_boxes:
            logger.warning("No valid bounding boxes to merge")
            return None

        # Extract coordinates
        x_values = [box['x'] for box in valid_boxes]
        y_values = [box['y'] for box in valid_boxes]
        x_max_values = [box['x'] + box['width'] for box in valid_boxes]
        y_max_values = [box['y'] + box['height'] for box in valid_boxes]

        # Calculate merged bounding box
        min_x = min(x_values)
        min_y = min(y_values)
        max_x = max(x_max_values)
        max_y = max(y_max_values)

        merged_bbox = {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }

        return merged_bbox

    def _is_valid_bbox(self, bbox: Dict) -> bool:
        """
        Validate bounding box has required fields with numeric values.

        Args:
            bbox: Bounding box dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['x', 'y', 'width', 'height']
        
        try:
            for field in required_fields:
                if field not in bbox:
                    return False
                # Check if value is numeric
                float(bbox[field])
            return True
        except (TypeError, ValueError):
            return False


# Convenience function for direct usage
def match_entities_to_words(
    entities: List[Dict],
    ocr_words: List[Dict]
) -> List[Dict]:
    """
    Match entities to OCR words and assign bounding boxes.

    This is a convenience function that creates a BBoxMatcher instance
    and calls its match_entities_to_words method.

    Args:
        entities: List of entities from Claude Vision
        ocr_words: List of words from OCR service

    Returns:
        List of entities with bounding_boxes field added
    """
    matcher = BBoxMatcher()
    return matcher.match_entities_to_words(entities, ocr_words)
