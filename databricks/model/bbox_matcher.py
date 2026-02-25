"""
Bounding Box Matcher

Reconciles entity bounding boxes returned by Claude Vision with the precise
word-level bounding boxes from Azure Document Intelligence OCR.

Why this is needed:
  Claude's bounding boxes are approximate (estimated from the image).  The OCR
  service provides pixel-accurate, word-level boxes.  BBoxMatcher snaps each
  Claude entity to the tightest union of OCR word boxes that overlap the
  Claude box, improving masking precision.

Planned algorithm:
  1. For each Claude entity, find all OCR words whose centres fall within the
     entity's bounding box (with a configurable tolerance).
  2. Compute the union bounding box of the matched OCR words.
  3. If no OCR words match (e.g. the entity spans a non-text region), fall back
     to the original Claude bounding box.

TODO:
  - Implement match_entity(entity_bbox, ocr_words, tolerance) -> list[float]
  - Implement match_all(entities, ocr_data) -> list[dict]  (mutates bbox in place)
  - Add tolerance parameter (default 0.01 in normalised coords)
  - Unit-test edge cases: no overlap, partial overlap, multi-word entities
"""


class BBoxMatcher:
    """Snaps Claude entity boxes to precise OCR word boxes."""

    def __init__(self, tolerance: float = 0.01):
        """
        Args:
            tolerance: Extra margin (in normalised coords) added around each
                       Claude bounding box when searching for OCR word matches.
        """
        self.tolerance = tolerance

    def match_all(self, entities: list, ocr_data: dict) -> list:
        """
        Refine bounding boxes for all entities.

        Args:
            entities: List of entity dicts with 'bounding_box' key
            ocr_data: OCR output dict with 'words' list

        Returns:
            Same entity list with updated 'bounding_box' values

        TODO: implement
        """
        raise NotImplementedError

    def match_entity(self, entity_bbox: list, ocr_words: list) -> list:
        """
        Find the tightest OCR word union for a single entity bbox.

        Args:
            entity_bbox: [x, y, w, h] in normalised coords
            ocr_words:   List of OCR word dicts, each with 'bounding_box' key

        Returns:
            Refined [x, y, w, h] in normalised coords

        TODO: implement
        """
        raise NotImplementedError

    def _bbox_union(self, boxes: list) -> list:
        """Compute the union of a list of [x, y, w, h] boxes. TODO: implement."""
        raise NotImplementedError

    def _overlaps(self, box_a: list, box_b: list) -> bool:
        """Return True if two normalised boxes overlap. TODO: implement."""
        raise NotImplementedError
