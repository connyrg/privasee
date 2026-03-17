"""
Session Manager

Manages session artefacts on a Databricks Unity Catalog volume using the
Files REST API (``/api/2.0/fs/files``).

Session artefacts are stored under:
    {UC_VOLUME_PATH}/{session_id}/
        metadata.json          — session state (status, filename, field_definitions, …)
        entities.json          — extracted entity list (written by Databricks model)
        masking_decisions.json — audit record of what was approved and masked
        original.{ext}         — the uploaded document
        masked.pdf             — the de-identified output (after approve-and-mask)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from app.models import SessionData

logger = logging.getLogger(__name__)

_VALID_STATUSES = ["uploaded", "processing", "awaiting_review", "completed", "error"]


class UCSessionManager:
    """
    Manages session artefacts on a Databricks Unity Catalog volume using the
    Files REST API (``/api/2.0/fs/files``).

    All methods make HTTP calls authenticated with a Databricks personal access
    token.  The caller is responsible for catching ``requests.HTTPError`` for
    unexpected non-2xx responses.
    """

    _FILES_API = "/api/2.0/fs/files"

    def __init__(self, databricks_host: str, token: str, volume_path: str) -> None:
        """
        Args:
            databricks_host: Workspace URL, e.g. https://adb-xxxx.azuredatabricks.net
            token:           Personal access token with Files API permissions.
            volume_path:     UC volume root, e.g. /Volumes/catalog/schema/sessions
        """
        self.host = databricks_host.rstrip("/")
        self.token = token
        self.volume_path = volume_path.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _url(self, uc_path: str) -> str:
        """Build a full Files API URL from a UC volume path."""
        return f"{self.host}{self._FILES_API}{uc_path}"

    def _session_path(self, session_id: str, filename: str) -> str:
        return f"{self.volume_path}/{session_id}/{filename}"

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    def create_session(self, original_filename: str) -> str:
        """
        Create a new session, persist metadata.json, and return the session_id.

        Args:
            original_filename: Name of the uploaded document.

        Returns:
            A new UUID v4 string identifying the session.
        """
        session_id = str(uuid.uuid4())
        metadata: Dict[str, Any] = {
            "session_id": session_id,
            "original_filename": original_filename,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "uploaded",
        }
        path = self._session_path(session_id, "metadata.json")
        response = requests.put(
            self._url(path),
            headers=self._headers(),
            data=json.dumps(metadata),
        )
        response.raise_for_status()
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Load session metadata from metadata.json on the UC volume.

        Returns:
            Parsed SessionData, or None if the session does not exist (404).

        Raises:
            ValueError: If metadata.json contains malformed JSON.
        """
        path = self._session_path(session_id, "metadata.json")
        response = requests.get(self._url(path), headers=self._headers())
        if response.status_code == 404:
            return None
        response.raise_for_status()
        try:
            meta = response.json()
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed metadata.json for session {session_id}"
            ) from exc
        return SessionData(
            session_id=meta["session_id"],
            filename=meta.get("original_filename", ""),
            file_size=meta.get("file_size", 0),
            status=meta.get("status", "uploaded"),
            error_message=meta.get("error_message"),
        )

    def update_status(self, session_id: str, status: str) -> None:
        """
        Read existing metadata.json, update its status field, and write it back.

        Args:
            session_id: Session to update.
            status:     New status — must be one of ``_VALID_STATUSES``.

        Raises:
            ValueError: If ``status`` is not in the valid progression.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. Must be one of {_VALID_STATUSES}"
            )
        path = self._session_path(session_id, "metadata.json")
        url = self._url(path)

        read_response = requests.get(url, headers=self._headers())
        read_response.raise_for_status()
        metadata = read_response.json()

        metadata["status"] = status
        write_response = requests.put(
            url, headers=self._headers(), data=json.dumps(metadata)
        )
        write_response.raise_for_status()

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        """
        Merge arbitrary kwargs into metadata.json and write it back.

        Supported kwargs mirror SessionData fields (status, field_definitions,
        page_count, …).  If ``status`` is included it must be one of
        ``_VALID_STATUSES``.

        Args:
            session_id: Session to update.
            **kwargs:   Fields to merge into the stored metadata.

        Raises:
            ValueError: If a ``status`` kwarg is not a valid status value.
        """
        if "status" in kwargs and kwargs["status"] not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {kwargs['status']!r}. Must be one of {_VALID_STATUSES}"
            )
        path = self._session_path(session_id, "metadata.json")
        url = self._url(path)

        read_response = requests.get(url, headers=self._headers())
        read_response.raise_for_status()
        metadata = read_response.json()

        metadata.update(kwargs)
        write_response = requests.put(
            url, headers=self._headers(), data=json.dumps(metadata)
        )
        write_response.raise_for_status()

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def save_entities(self, session_id: str, entities: List[Dict[str, Any]]) -> None:
        """
        Persist the entity list for a session as entities.json.

        Args:
            session_id: Session to update.
            entities:   List of entity dicts.
        """
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "status": "awaiting_review",
            "entities": entities,
        }
        path = self._session_path(session_id, "entities.json")
        response = requests.put(
            self._url(path),
            headers=self._headers(),
            data=json.dumps(payload),
        )
        response.raise_for_status()

    def get_entities(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load entities.json for a session.

        Returns:
            List of entity dicts stored under the ``"entities"`` key.

        Raises:
            FileNotFoundError: If the session has no saved entities (404).
            ValueError:        If the stored file contains malformed JSON.
        """
        path = self._session_path(session_id, "entities.json")
        response = requests.get(self._url(path), headers=self._headers())
        if response.status_code == 404:
            raise FileNotFoundError(
                f"No entities found for session {session_id}"
            )
        response.raise_for_status()
        try:
            payload = response.json()
            return payload.get("entities", [])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in entities for session {session_id}"
            ) from exc

    def save_masking_decisions(
        self,
        session_id: str,
        all_entities: List[Dict[str, Any]],
        approved_ids: set,
    ) -> None:
        """
        Write masking_decisions.json as an audit record before approve-and-mask.

        Records every entity with an ``approved`` flag and the final
        ``replacement_text`` that was used, plus a timestamp. Written before
        the masking call so the record is captured even if masking fails.

        Args:
            session_id:   Session to update.
            all_entities: Full entity list with any user edits already applied.
            approved_ids: Set of entity IDs approved for masking.
        """
        decisions = [
            {**e, "approved": e.get("id") in approved_ids}
            for e in all_entities
        ]
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "entities": decisions,
        }
        path = self._session_path(session_id, "masking_decisions.json")
        response = requests.put(
            self._url(path),
            headers=self._headers(),
            data=json.dumps(payload),
        )
        response.raise_for_status()

    # ------------------------------------------------------------------
    # Binary file storage
    # ------------------------------------------------------------------

    def save_file(self, session_id: str, filename: str, data: bytes) -> None:
        """
        Upload raw bytes to the UC volume under the session directory.

        Args:
            session_id: Session the file belongs to.
            filename:   Storage name, e.g. ``"original.pdf"``.
            data:       Raw file bytes sent as the request body.
        """
        path = self._session_path(session_id, filename)
        response = requests.put(
            self._url(path),
            headers=self._headers(),
            data=data,
        )
        response.raise_for_status()

    def get_file(self, session_id: str, filename: str) -> bytes:
        """
        Download raw bytes from the UC volume.

        Args:
            session_id: Session the file belongs to.
            filename:   Name stored under the session directory.

        Returns:
            Raw file bytes (``response.content``).

        Raises:
            FileNotFoundError: If the file does not exist in the volume (404).
        """
        path = self._session_path(session_id, filename)
        response = requests.get(self._url(path), headers=self._headers())
        if response.status_code == 404:
            raise FileNotFoundError(
                f"File '{filename}' not found for session {session_id}"
            )
        response.raise_for_status()
        return response.content

    def delete_session(self, session_id: str) -> None:
        """
        Delete all artefacts for a session from the UC volume.

        Attempts to delete each known file individually.  404 responses are
        silently ignored (the file was never created or was already removed).
        Any other non-2xx response raises ``requests.HTTPError``.

        Args:
            session_id: Session whose artefacts should be deleted.
        """
        # Determine the original file's extension so we can delete it by name.
        original_ext = ".pdf"
        try:
            meta_path = self._session_path(session_id, "metadata.json")
            meta_response = requests.get(self._url(meta_path), headers=self._headers())
            if meta_response.ok:
                meta = meta_response.json()
                original_filename = meta.get("original_filename", "")
                from pathlib import Path as _Path
                ext = _Path(original_filename).suffix.lower()
                if ext:
                    original_ext = ext
        except Exception:
            pass  # Best-effort — fall back to .pdf

        candidates = [
            "metadata.json",
            "entities.json",
            f"original{original_ext}",
            "masked.pdf",
        ]
        for filename in candidates:
            path = self._session_path(session_id, filename)
            response = requests.delete(self._url(path), headers=self._headers())
            if response.status_code == 404:
                continue  # File never existed — not an error
            response.raise_for_status()
