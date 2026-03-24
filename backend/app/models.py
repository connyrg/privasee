"""
Pydantic data models for PrivaSee.

Covers all FastAPI request/response shapes and the server-to-server contract
with the Databricks Model Serving endpoints.
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


class Occurrence(BaseModel):
    """
    A single positional occurrence of an entity in the document.

    Populated after entity-variant merging so that partial name references
    (e.g. "Stephen" merged into "Stephen Parrot") are tracked alongside the
    canonical entity across all pages.
    """

    page_number: int = Field(default=1, ge=1)
    original_text: str = Field(default="", description="Exact text at this location")
    bounding_boxes: List[List[float]] = Field(
        default_factory=list,
        description="Line-level bboxes [[x, y, w, h], ...], normalised 0–1",
    )


class Entity(BaseModel):
    """A single sensitive entity detected in the document."""

    id: str = Field(..., description="Unique identifier for this entity")
    entity_type: str = Field(..., description="Type of entity (field name)")
    original_text: str = Field(..., description="Original text found in the document")
    replacement_text: str = Field(default="", description="Text to replace it with")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    approved: bool = Field(default=True)
    strategy: Optional[str] = Field(
        default=None, description="Masking strategy: 'Fake Data', 'Black Out', 'Entity Label'"
    )
    occurrences: Optional[List[Occurrence]] = Field(
        default=None,
        description="All positional appearances across pages. Each occurrence carries "
                    "page_number, original_text (exact), and bounding_boxes ([[x,y,w,h]...]).",
    )


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
    status: str = "uploaded"          # uploaded | processing | awaiting_review | completed | error
    entities: List[Entity] = []
    field_definitions: List[FieldDefinition] = []
    error_message: Optional[str] = None

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


class ProcessAcceptedResponse(BaseModel):
    """202 response from POST /api/process — entity extraction running in background."""

    session_id: str
    status: str = "processing"
    message: str = "Entity extraction started. Poll GET /api/sessions/{session_id} for status."


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
    """Public session metadata returned by GET /api/sessions/{session_id}."""

    session_id: str
    filename: str
    file_size: int
    status: str
    entity_count: int
    has_masked_output: bool
    entities: List[Entity] = []
    error_message: Optional[str] = None


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
# Config management
# ---------------------------------------------------------------------------

class SaveConfigRequest(BaseModel):
    """Request body for POST /api/configs."""
    config_name: str = Field(..., min_length=1, max_length=80)
    field_definitions: List[FieldDefinition] = Field(..., min_length=1)


class ConfigSummary(BaseModel):
    """Lightweight config metadata returned by GET /api/configs."""
    config_name: str
    key: str
    saved_at: str


class ConfigDetail(ConfigSummary):
    """Full config including field definitions, returned by GET /api/configs/{key}."""
    field_definitions: List[FieldDefinition]


# ---------------------------------------------------------------------------
# System templates
# ---------------------------------------------------------------------------

class SystemTemplateSummary(BaseModel):
    """Lightweight system template metadata returned by GET /api/templates."""
    key: str
    template_name: str
    description: str
    field_count: int


class SystemTemplateDetail(SystemTemplateSummary):
    """Full system template with field definitions, returned by GET /api/templates/{key}."""
    field_definitions: List[FieldDefinition]


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

class EntityVerifyResult(BaseModel):
    """Verification result for a single entity."""
    id: str
    original_text: str
    masked: bool


class VerifyRequest(BaseModel):
    """Request body for POST /api/sessions/{session_id}/verify."""
    entities: List[Entity]


class VerifyResponse(BaseModel):
    """Response from POST /api/sessions/{session_id}/verify."""
    session_id: str
    score: float = Field(..., ge=0.0, le=100.0, description="Masking score 0–100")
    masked_count: int
    total: int
    entities: List[EntityVerifyResult]


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

        Handles the standard ``{"predictions": [...]}`` shape and a bare
        ``{"entities": [...]}`` shape for easier local mocking.

        The model response includes both a top-level ``entities`` key (merged
        flat list) and a ``pages`` key (per-page unmerged list).  This method
        prefers ``entities`` — using the merged list avoids overwriting the
        UC-persisted merged result with unmerged per-page data.  The ``pages``
        key is only used as a fallback when ``entities`` is absent.
        """
        predictions = raw.get("predictions") or raw.get("dataframe_records")
        if predictions and isinstance(predictions, list):
            record = predictions[0]
        else:
            record = raw  # bare dict fallback

        # Surface errors returned by the Databricks model (e.g. ADI timeout,
        # vision API failure) so the backend can mark the session as "error"
        # instead of silently treating the result as 0 entities found.
        if record.get("status") == "error":
            msg = record.get("error_message") or "Entity extraction failed in the Databricks model."
            raise RuntimeError(msg)

        # Prefer top-level "entities" (merged flat list written by the model)
        # over "pages" (per-page unmerged list). The model sets both; using
        # "entities" avoids overwriting the UC-merged result with unmerged data.
        if record.get("entities") is not None:
            raw_entities: list = record["entities"]
        elif record.get("pages") and isinstance(record["pages"], list):
            raw_entities = []
            for page in record["pages"]:
                for entity in page.get("entities", []):
                    raw_entities.append(entity)
        else:
            raw_entities = []

        entities = [Entity(**e) for e in raw_entities]
        return cls(
            entities=entities,
            model_version=record.get("model_version"),
        )
