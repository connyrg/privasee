"""
Masking Service
Applies visual masks to documents using PyMuPDF (PDF) or PIL (images).

Identical logic to backend/app/services/masking_service.py — both use the
same coordinate system and strategy names so entities produced by the
extraction model can be passed directly to apply_pdf_masks.
"""

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class MaskingService:
    """Apply visual masks to documents."""

    _STRATEGY_MAP: Dict[str, str] = {
        "Black Out":    "redact",
        "redact":       "redact",
        "Fake Data":    "fake_name",
        "fake_name":    "fake_name",
        "Entity Label": "entity_label",
        "entity_label": "entity_label",
    }

    def __init__(self, font_path: Optional[str] = None):
        self.font_path = font_path

    # ------------------------------------------------------------------
    # PDF-native masking (PyMuPDF)
    # ------------------------------------------------------------------

    def apply_pdf_masks(
        self,
        pdf_bytes: bytes,
        entities: List[Dict[str, Any]],
    ) -> bytes:
        """
        Apply redactions to PDF bytes and return masked PDF bytes.

        Bounding box values are normalised [0, 1] relative to page dimensions.
        Entities with approved=False are skipped.

        Strategies:
            "Black Out"    — black-filled rectangle, text permanently removed.
            "Fake Data"    — white-filled rectangle, replacement_text inserted.
            "Entity Label" — white-filled rectangle, auto label e.g. Full_Name_1.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        label_counters: Dict[str, int] = {}
        consistency_map: Dict[str, str] = {}

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

            inserts: List[Tuple[fitz.Rect, str]] = []

            # Collect widgets once per page into a plain list so we can
            # safely look them up without re-iterating the live generator.
            page_widgets = list(page.widgets())

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

                    # If this bbox covers a form widget, update the widget
                    # value directly instead of painting a redact rect on top.
                    # Painting over a widget annotation does not clear it —
                    # the widget renders its value on top of whatever is in
                    # the content stream, causing the original value to show
                    # through.  Updating the value in place avoids that while
                    # keeping the form field structure intact.
                    overlapping_widget = next(
                        (wgt for wgt in page_widgets if not (wgt.rect & rect).is_empty),
                        None,
                    )
                    if overlapping_widget is not None:
                        overlapping_widget.field_value = replacement or ""
                        overlapping_widget.update()
                    else:
                        page.add_redact_annot(rect, fill=fill_color)
                        if replacement:
                            inserts.append((rect, replacement))

            page.apply_redactions()

            for rect, text in inserts:
                fontsize = max(6, min(10, int(rect.height * 0.7)))
                page.insert_text(
                    (rect.x0 + 2, rect.y1 - 2),
                    text,
                    fontsize=fontsize,
                )

        buf = io.BytesIO()
        # garbage=4: full cross-reference rebuild — purges all unreferenced
        # objects, including old widget appearance streams that contained the
        # original (sensitive) field values before they were overwritten.
        doc.save(buf, garbage=4, deflate=True)
        doc.close()
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Image masking (PIL) — used when original file is PNG/JPG
    # ------------------------------------------------------------------

    def apply_masks(
        self,
        image_path: str,
        entities: List[Dict],
        output_path: str,
        mask_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        border_color: Tuple[int, int, int] = (200, 200, 200),
    ) -> str:
        """Apply masks to an image file and save the result."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size

        for entity in entities:
            raw_strategy = entity.get("strategy", "")
            strategy = self._STRATEGY_MAP.get(raw_strategy, "redact")
            replacement_text = entity.get("replacement_text", "")
            bboxes = self._resolve_bboxes(entity)

            fill = (0, 0, 0) if strategy == "redact" else mask_color

            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                x, y, width, height = self._normalize_bbox(bbox, img_width, img_height)

                if width <= 0 or height <= 0:
                    continue

                # Black Out → solid black rectangle, no text
                # Fake Data / Entity Label → white rectangle + replacement text
                draw.rectangle(
                    [x, y, x + width, y + height],
                    fill=fill,
                    outline=border_color,
                    width=1,
                )

                if strategy != "redact" and replacement_text:
                    self._draw_text(draw, replacement_text, x, y, width, height, text_color)

        image.save(output_path, "PNG", quality=95)
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bboxes(entity: Dict[str, Any]) -> List[List[float]]:
        multi = entity.get("bounding_boxes")
        if multi:
            result = []
            for b in multi:
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    result.append(list(b))
                elif isinstance(b, dict) and all(k in b for k in ("x", "y", "width", "height")):
                    result.append([b["x"], b["y"], b["width"], b["height"]])
            return result
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
        if strategy == "redact":
            return ""

        original_key = entity.get("original_text", "").lower().strip()

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
            # Respect a label pre-generated by the document intelligence model
            # (stored in replacement_text).  Fall back to generating one here
            # only if no pre-set value is available.
            pre_set = entity.get("replacement_text", "")
            if pre_set:
                consistency_map[original_key] = pre_set
                return pre_set
            etype = (
                entity.get("entity_type", "Unknown")
                .replace(" ", "_")
                .replace("-", "_")
            )
            label_counters[etype] = label_counters.get(etype, 0) + 1
            count = label_counters[etype]
            suffix = chr(64 + count) if count <= 26 else str(count)
            label = f"{etype}_{suffix}"
            consistency_map[original_key] = label
            return label

        return ""

    def _normalize_bbox(
        self, bbox: List[float], img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        x, y, width, height = bbox
        x = int(x * img_width)
        y = int(y * img_height)
        width = int(width * img_width)
        height = int(height * img_height)
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(img_width - x, width + 2 * padding)
        height = min(img_height - y, height + 2 * padding)
        return x, y, width, height

    def _draw_text(self, draw, text, x, y, box_width, box_height, text_color):
        font = self._get_font(box_height)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        if text_width > box_width * 0.95:
            font = self._get_font(int(box_height * 0.8))
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

        text_x = x + (box_width - text_width) // 2
        text_y = y + (box_height - text_height) // 2
        draw.text((text_x, text_y), text, fill=text_color, font=font)

    def _get_font(self, height: int) -> ImageFont.ImageFont:
        font_size = max(8, int(height * 0.7))
        try:
            if self.font_path and os.path.exists(self.font_path):
                return ImageFont.truetype(self.font_path, font_size)
            for path in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux (Databricks)
                "/System/Library/Fonts/Helvetica.ttc",               # macOS
                "C:\\Windows\\Fonts\\arial.ttf",                     # Windows
            ]:
                if os.path.exists(path):
                    return ImageFont.truetype(path, font_size)
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()
