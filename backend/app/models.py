"""
Pydantic data models for PrivaSee.

All request/response shapes are kept wire-compatible with the PoC so the
React frontend requires no changes.  New fields added for the target
architecture are either optional or only appear in server-to-server calls.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReplacementStrategy(str, Enum):
    FAKE_DATA    = "Fake Data"
    BLACK_OUT    = "Black Out"
    ENTITY_LABEL = "Entity Label"


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class FieldDefinition(BaseModel):
    """A field the user wants to detect and de-identify."""

    name: str = Field(..., description="Short label, e.g. 'Full Name'")
    description: str = Field(..., description="Natural-language description for Claude")
    strategy: ReplacementStrategy = ReplacementStrategy.FAKE_DATA

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field description cannot be empty")
        return v.strip()


class BoundingBox(BaseModel):
    """Normalised bounding box — all values in the [0, 1] range."""

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., ge=0.0, le=1.0)
    height: float = Field(..., ge=0.0, le=1.0)

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.width, self.height]

    @classmethod
    def from_list(cls, bbox: List[float]) -> "BoundingBox":
        if len(bbox) != 4:
            raise ValueError("Bounding box must have exactly 4 values")
        return cls(x=bbox[0], y=bbox[1], width=bbox[2], height=bbox[3])


class Entity(BaseModel):
    """
    A single sensitive entity detected in the document.

    Field names match the PoC exactly so the React ReviewTable and
    approveAndMask() call work without modification.
    """

    id: str = Field(..., description="Unique identifier for this entity")
    entity_type: str = Field(..., description="Type of entity (field name)")
    original_text: str = Field(..., description="Original text found in the document")
    replacement_text: str = Field(default="", description="Text to replace it with")
    bounding_box: List[float] = Field(
        ..., description="Normalised [x, y, width, height] — first occurrence, used for preview"
    )
    bounding_boxes: Optional[List[Any]] = Field(
        default=None,
        description="All occurrences in the document: list of [x,y,w,h] or {x,y,width,height}. "
                    "Passed through to the masking model so every appearance is redacted.",
    )
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    approved: bool = Field(default=True)
    page_number: int = Field(default=1, ge=1)
    strategy: Optional[str] = Field(
        default=None, description="Masking strategy: 'Fake Data', 'Black Out', 'Entity Label'"
    )

    @field_validator("bounding_box")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bounding_box must have exactly 4 values [x, y, w, h]")
        return v


class OCRData(BaseModel):
    """Structured OCR output — mirrors the shape returned by Azure Document Intelligence."""

    words: List[Dict[str, Any]] = []
    lines: List[Dict[str, Any]] = []
    page_width: float = 0.0
    page_height: float = 0.0


class SessionData(BaseModel):
    """All persisted state for a single processing session."""

    session_id: str
    filename: str
    file_size: int
    page_count: int = 1
    status: str = "uploaded"          # uploaded | processing | awaiting_review | completed
    entities: List[Entity] = []
    field_definitions: List[FieldDefinition] = []

    # UC volume path for this session, e.g. /Volumes/cat/schema/sessions/<id>/
    session_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response to POST /api/upload — wire-compatible with the PoC."""

    session_id: str
    filename: str
    file_size: int
    page_count: int = 1
    preview_url: Optional[str] = None
    message: str = "File uploaded successfully"


# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    """Request body for POST /api/process."""

    session_id: str
    field_definitions: List[FieldDefinition] = Field(..., min_length=1)

    @field_validator("field_definitions")
    @classmethod
    def unique_field_names(cls, v: List[FieldDefinition]) -> List[FieldDefinition]:
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            raise ValueError("Field names must be unique")
        return v


class ProcessResponse(BaseModel):
    """Response from POST /api/process — wire-compatible with the PoC."""

    session_id: str
    entities: List[Entity]
    total_entities: int
    message: str = "Document processed successfully"


# ---------------------------------------------------------------------------
# Approve and mask
# ---------------------------------------------------------------------------

class EntityUpdate(BaseModel):
    """Minimal payload for a user-edited entity — only the fields the backend uses."""

    id: str
    replacement_text: str


class ApprovalRequest(BaseModel):
    """Request body for POST /api/approve-and-mask."""

    session_id: str
    # IDs of entities the user approved for masking
    approved_entity_ids: List[str]
    # Optional entities with user-edited replacement_text
    updated_entities: Optional[List[EntityUpdate]] = None


class ApprovalResponse(BaseModel):
    """Response from POST /api/approve-and-mask — wire-compatible with the PoC."""

    session_id: str
    original_pdf_url: str
    masked_pdf_url: str
    masked_image_url: Optional[str] = None
    entities_masked: int
    message: str = "Masked PDF generated successfully"


# ---------------------------------------------------------------------------
# Session info (GET /api/sessions/{session_id})
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    """Public session metadata — no file paths or entity details."""

    session_id: str
    filename: str
    file_size: int
    status: str
    entity_count: int
    has_masked_output: bool


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response from GET /api/health."""

    status: str = "ok"
    version: str = "2.0.0"
    mock_databricks: bool = False
    databricks_endpoint_configured: bool = False
    databricks_masking_endpoint_configured: bool = False
    uc_volume_configured: bool = False


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope returned by exception handlers."""

    error: str
    detail: Optional[str] = None
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Databricks Model Serving — server-to-server contract
# ---------------------------------------------------------------------------

class DatabricksProcessRequest(BaseModel):
    """
    Payload sent to the Databricks Model Serving endpoint.

    Uses MLflow's standard dataframe_records envelope so the PyFunc model
    can receive it via the standard serving interface.  The model reads the
    actual document bytes from the shared UC volume using session_id.
    """

    session_id: str
    field_definitions: List[FieldDefinition]

    def to_mlflow_payload(self) -> Dict[str, Any]:
        """Wrap in MLflow Model Serving dataframe_records format."""
        return {
            "dataframe_records": [
                {
                    "session_id": self.session_id,
                    "field_definitions": [f.model_dump() for f in self.field_definitions],
                }
            ]
        }


class DatabricksProcessResponse(BaseModel):
    """
    Response from the Databricks Model Serving endpoint.

    MLflow wraps model output in a "predictions" list; the model returns one
    prediction object per input record.
    """

    entities: List[Entity]
    model_version: Optional[str] = None

    @classmethod
    def from_mlflow_response(cls, raw: Dict[str, Any]) -> "DatabricksProcessResponse":
        """
        Parse the MLflow Model Serving JSON envelope.

        Handles both the standard {"predictions": [...]} shape and a bare
        {"entities": [...]} shape for easier local mocking.

        The model returns entities nested under a "pages" list:
            {"pages": [{"page_num": 1, "entities": [...]}, ...]}
        This method flattens them into a single entity list, using each
        page's "page_num" as the entity's page_number.
        """
        predictions = raw.get("predictions") or raw.get("dataframe_records")
        if predictions and isinstance(predictions, list):
            record = predictions[0]
        else:
            record = raw  # bare dict fallback

        # Flatten pages[].entities into a single list
        pages = record.get("pages")
        if pages and isinstance(pages, list):
            raw_entities: list = []
            for page in pages:
                page_num = page.get("page_num", 1)
                for entity in page.get("entities", []):
                    entity.setdefault("page_number", page_num)
                    raw_entities.append(entity)
        else:
            # Fallback: bare {"entities": [...]} shape (mock / local testing)
            raw_entities = record.get("entities", [])

        entities = [Entity(**e) for e in raw_entities]
        return cls(
            entities=entities,
            model_version=record.get("model_version"),
        )
