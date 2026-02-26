"""
Masking Service
Applies visual masks to document images by drawing rectangles over identified entities.
"""

import cv2
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging
from typing import Any, Dict, List, Tuple, Optional
import os

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class MaskingService:
    """Service for applying visual masks to document images."""

    def __init__(self, font_path: Optional[str] = None):
        """
        Initialize masking service.

        Args:
            font_path: Path to TrueType font file (optional)
        """
        self.font_path = font_path
        logger.info("Masking Service initialized")

    def apply_masks(
        self,
        image_path: str,
        entities: List[Dict],
        output_path: str,
        mask_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        border_color: Tuple[int, int, int] = (200, 200, 200)
    ) -> str:
        """
        Apply masks to image for specified entities.

        Args:
            image_path: Path to input image
            entities: List of entities to mask, each with:
                - bounding_box: [x, y, width, height]
                - replacement_text: Text to display
                - entity_type: Type of entity
            output_path: Path for output image
            mask_color: RGB color for mask background (default white)
            text_color: RGB color for replacement text (default black)
            border_color: RGB color for mask border (default gray)

        Returns:
            Path to masked image

        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: For processing errors
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            logger.info(f"Applying masks to {len(entities)} entities")

            # Load image with PIL for better quality
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Get image dimensions
            img_width, img_height = image.size

            # Sort entities by bounding box area (largest first) to avoid overlaps
            sorted_entities = sorted(
                entities,
                key=lambda e: e['bounding_box'][2] * e['bounding_box'][3],
                reverse=True
            )

            # Apply each mask
            for entity in sorted_entities:
                bbox = entity['bounding_box']
                replacement_text = entity.get('replacement_text', '[REDACTED]')

                # Convert bounding box to pixel coordinates
                x, y, width, height = self._normalize_bbox(bbox, img_width, img_height)

                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    logger.warning(f"Invalid bounding box: {bbox}")
                    continue

                # Draw white rectangle (mask)
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=mask_color,
                    outline=border_color,
                    width=1
                )

                # Draw replacement text if not complete redaction
                if replacement_text != "[REDACTED]" and replacement_text != "":
                    self._draw_text(
                        draw,
                        replacement_text,
                        x, y, width, height,
                        text_color
                    )

            # Save masked image
            image.save(output_path, 'PNG', quality=95)

            logger.info(f"Successfully saved masked image to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error applying masks: {str(e)}")
            raise Exception(f"Failed to apply masks: {str(e)}")

    def _normalize_bbox(
        self,
        bbox: List[float],
        img_width: int,
        img_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Normalize bounding box coordinates.

        Args:
            bbox: Bounding box [x, y, width, height]
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Normalized bounding box as integers (x, y, width, height)
        """
        x, y, width, height = bbox

        # Bounding boxes are always stored as 0-1 normalized values
        # (normalized against page dimensions by the OCR service).
        x = int(x * img_width)
        y = int(y * img_height)
        width = int(width * img_width)
        height = int(height * img_height)

        # Add small padding
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(img_width - x, width + 2 * padding)
        height = min(img_height - y, height + 2 * padding)

        return x, y, width, height

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        box_width: int,
        box_height: int,
        text_color: Tuple[int, int, int]
    ):
        """
        Draw text centered in bounding box.

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            x, y: Top-left corner of bounding box
            box_width, box_height: Dimensions of bounding box
            text_color: RGB color for text
        """
        # Try to load font
        font = self._get_font(box_height)

        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Check if text fits in box
        if text_width > box_width * 0.95:
            # Try smaller font
            font = self._get_font(int(box_height * 0.8))
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        # Calculate center position
        text_x = x + (box_width - text_width) // 2
        text_y = y + (box_height - text_height) // 2

        # Draw text
        draw.text(
            (text_x, text_y),
            text,
            fill=text_color,
            font=font
        )

    def _get_font(self, height: int) -> ImageFont.ImageFont:
        """
        Get appropriate font for text height.

        Args:
            height: Desired text height in pixels

        Returns:
            PIL ImageFont object
        """
        # Calculate font size (approximate)
        font_size = max(8, int(height * 0.7))

        try:
            if self.font_path and os.path.exists(self.font_path):
                return ImageFont.truetype(self.font_path, font_size)
            else:
                # Try common system fonts
                font_paths = [
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                    "C:\\Windows\\Fonts\\arial.ttf",  # Windows
                ]

                for path in font_paths:
                    if os.path.exists(path):
                        return ImageFont.truetype(path, font_size)

                # Fallback to default font
                return ImageFont.load_default()

        except Exception as e:
            logger.warning(f"Could not load font: {e}, using default")
            return ImageFont.load_default()

    def preview_masks(
        self,
        image_path: str,
        entities: List[Dict],
        output_path: str
    ) -> str:
        """
        Create preview with colored bounding boxes (no masking).

        Args:
            image_path: Path to input image
            entities: List of entities to preview
            output_path: Path for output preview image

        Returns:
            Path to preview image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load image with OpenCV for color manipulation
            image = cv2.imread(image_path)

            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            img_height, img_width = image.shape[:2]

            # Define colors for different entity types
            colors = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
            ]

            entity_types = list(set(e['entity_type'] for e in entities))
            color_map = {et: colors[i % len(colors)] for i, et in enumerate(entity_types)}

            # Draw boxes
            for entity in entities:
                bbox = entity['bounding_box']
                entity_type = entity['entity_type']
                color = color_map.get(entity_type, (255, 0, 0))

                x, y, width, height = self._normalize_bbox(bbox, img_width, img_height)

                # Draw rectangle
                cv2.rectangle(
                    image,
                    (x, y),
                    (x + width, y + height),
                    color,
                    2
                )

                # Draw label
                label = entity_type
                cv2.putText(
                    image,
                    label,
                    (x, max(y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

            # Save preview
            cv2.imwrite(output_path, image)

            logger.info(f"Preview saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating preview: {str(e)}")
            raise Exception(f"Failed to create preview: {str(e)}")

    # ------------------------------------------------------------------
    # PDF-native masking (PyMuPDF)
    # ------------------------------------------------------------------

    _STRATEGY_MAP: Dict[str, str] = {
        "Black Out":    "redact",
        "redact":       "redact",
        "Fake Data":    "fake_name",
        "fake_name":    "fake_name",
        "Entity Label": "entity_label",
        "entity_label": "entity_label",
    }

    def apply_pdf_masks(
        self,
        pdf_bytes: bytes,
        entities: List[Dict[str, Any]],
    ) -> bytes:
        """
        Apply redactions to PDF bytes using PyMuPDF and return masked PDF bytes.

        Each entity may specify a single ``bounding_box`` or a list of
        ``bounding_boxes`` (takes priority when both are present).  Bounding
        box values are normalised [0, 1] relative to page dimensions.

        Entities with ``approved=False`` are silently skipped.

        Strategies
        ----------
        "Black Out" / "redact"
            Black-filled rectangle.  Original text permanently removed.
        "Fake Data" / "fake_name"
            White-filled rectangle.  Original text replaced with the entity's
            ``replacement_text``.  The same ``original_text`` always maps to the
            same replacement within a single call.
        "Entity Label" / "entity_label"
            White-filled rectangle.  Original text replaced with an auto-generated
            label of the form ``{EntityType}_{N}`` (e.g. ``Full_Name_1``) where
            ``N`` is a per-type incrementing integer.  Identical ``original_text``
            values receive the same label.

        Parameters
        ----------
        pdf_bytes:
            Raw bytes of the input PDF.
        entities:
            List of entity dicts.  Recognised keys:
            ``approved``, ``page_number``, ``strategy``, ``entity_type``,
            ``original_text``, ``replacement_text``,
            ``bounding_box``, ``bounding_boxes``.

        Returns
        -------
        bytes
            Redacted PDF bytes, parseable by ``fitz.open()``.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Per-call state (reset for every apply_pdf_masks invocation)
        label_counters: Dict[str, int] = {}
        consistency_map: Dict[str, str] = {}  # normalised_original → replacement

        # Group approved entities by 0-indexed page number
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        for entity in entities:
            if not entity.get("approved", True):
                continue
            page_idx = entity.get("page_number", 1) - 1
            by_page.setdefault(page_idx, []).append(entity)

        for page_idx, page_entities in sorted(by_page.items()):
            if page_idx >= len(doc):
                continue

            page = doc[page_idx]
            W = page.rect.width
            H = page.rect.height

            # Collect (rect, replacement_text) pairs so we can batch all
            # add_redact_annot calls before a single apply_redactions().
            inserts: List[Tuple[fitz.Rect, str]] = []

            for entity in page_entities:
                raw_strategy = entity.get("strategy", "redact")
                strategy = self._STRATEGY_MAP.get(raw_strategy, "redact")
                bboxes = self._resolve_bboxes(entity)

                if not bboxes:
                    continue

                replacement = self._resolve_replacement(
                    entity, strategy, label_counters, consistency_map
                )
                fill_color = (0.0, 0.0, 0.0) if strategy == "redact" else (1.0, 1.0, 1.0)

                for bbox in bboxes:
                    if len(bbox) != 4:
                        continue
                    x, y, w, h = bbox
                    rect = fitz.Rect(x * W, y * H, (x + w) * W, (y + h) * H)
                    page.add_redact_annot(rect, fill=fill_color)
                    if replacement:
                        inserts.append((rect, replacement))

            page.apply_redactions()

            # Insert replacement text after redactions have been applied so
            # the text lands in the now-cleared area and is extractable.
            for rect, text in inserts:
                fontsize = max(6, min(10, int(rect.height * 0.7)))
                page.insert_text(
                    (rect.x0 + 2, rect.y1 - 2),
                    text,
                    fontsize=fontsize,
                )

        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Private helpers for apply_pdf_masks
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bboxes(entity: Dict[str, Any]) -> List[List[float]]:
        """Return the list of bounding boxes for an entity.

        Prefers ``bounding_boxes`` (plural) when present; falls back to
        ``bounding_box`` (singular) wrapped in a one-element list.  Entries
        that are not 4-element lists are silently discarded.
        """
        multi = entity.get("bounding_boxes")
        if multi:
            return [b for b in multi if isinstance(b, (list, tuple)) and len(b) == 4]
        single = entity.get("bounding_box")
        if single and len(single) == 4:
            return [list(single)]
        return []

    @staticmethod
    def _resolve_replacement(
        entity: Dict[str, Any],
        strategy: str,
        label_counters: Dict[str, int],
        consistency_map: Dict[str, str],
    ) -> str:
        """Return the replacement string for an entity, maintaining consistency."""
        if strategy == "redact":
            return ""

        original_key = entity.get("original_text", "").lower().strip()

        # Both fake_name and entity_label honour the consistency map so that
        # duplicate occurrences of the same original text always get the same
        # replacement within one apply_pdf_masks call.
        if original_key in consistency_map:
            return consistency_map[original_key]

        if strategy == "fake_name":
            text = entity.get(
                "replacement_text",
                f"[{entity.get('entity_type', 'UNKNOWN')}]",
            )
            consistency_map[original_key] = text
            return text

        if strategy == "entity_label":
            etype = (
                entity.get("entity_type", "Unknown")
                .replace(" ", "_")
                .replace("-", "_")
            )
            label_counters[etype] = label_counters.get(etype, 0) + 1
            label = f"{etype}_{label_counters[etype]}"
            consistency_map[original_key] = label
            return label

        return ""
