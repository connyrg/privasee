"""
Session Manager

Manages session lifecycle on a Databricks Unity Catalog volume.

Session artefacts are stored under:
    {UC_VOLUME_PATH}/{session_id}/
        session.json    — serialised SessionData
        original.{ext}  — the uploaded document
        masked.pdf      — the de-identified output (after approve-and-mask)

Access is via the Databricks SDK Files API, authenticated with
DATABRICKS_HOST + DATABRICKS_TOKEN.

TODO: implement all methods using databricks-sdk WorkspaceClient.
      Until then every method raises NotImplementedError so callers
      in main.py can catch it and return a clear 501 response.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from app.models import Entity, FieldDefinition, SessionData

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session lifecycle on a Databricks Unity Catalog volume."""

    def __init__(
        self,
        uc_volume_path: str,
        databricks_host: str,
        databricks_token: str,
    ) -> None:
        """
        Args:
            uc_volume_path:   Base path, e.g. /Volumes/catalog/schema/sessions
            databricks_host:  Workspace URL, e.g. https://adb-xxxx.azuredatabricks.net
            databricks_token: Personal access token with Files API permissions
        """
        self.uc_volume_path = uc_volume_path.rstrip("/")
        self.databricks_host = databricks_host.rstrip("/")
        self.databricks_token = databricks_token
        # TODO: initialise databricks-sdk WorkspaceClient
        # from databricks.sdk import WorkspaceClient
        # self._client = WorkspaceClient(
        #     host=databricks_host,
        #     token=databricks_token,
        # )
        logger.info("SessionManager initialised (UC volume: %s)", uc_volume_path)

    # ------------------------------------------------------------------
    # Internal helpers (to be implemented)
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str) -> str:
        """Return the UC volume directory for a session."""
        return f"{self.uc_volume_path}/{session_id}"

    def _session_json_path(self, session_id: str) -> str:
        return f"{self._session_dir(session_id)}/session.json"

    def _file_path(self, session_id: str, filename: str) -> str:
        return f"{self._session_dir(session_id)}/{filename}"

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, metadata: Dict[str, Any]) -> SessionData:
        """
        Create a new session entry and persist it to the UC volume.

        Args:
            session_id: Caller-generated UUID string.
            metadata:   Dict containing at minimum: filename, file_size, status.

        Returns:
            The newly created SessionData.

        TODO: write session.json to UC using self._client.files.upload()
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.create_session is not yet implemented")

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Load a session from the UC volume.

        Returns None if the session does not exist.

        TODO: read and parse session.json via self._client.files.download()
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.get_session is not yet implemented")

    def update_session(self, session_id: str, **kwargs: Any) -> SessionData:
        """
        Merge kwargs into the persisted session and write it back.

        Supported kwargs match SessionData fields (status, entities,
        field_definitions, page_count, …).

        TODO: read → merge → write session.json
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.update_session is not yet implemented")

    def delete_session(self, session_id: str) -> None:
        """
        Remove all artefacts for a session from the UC volume.

        TODO: list and delete all files under _session_dir() via
              self._client.files.delete()
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.delete_session is not yet implemented")

    # ------------------------------------------------------------------
    # File storage
    # ------------------------------------------------------------------

    def save_file(self, session_id: str, filename: str, data: bytes) -> str:
        """
        Write binary artefact to the UC volume.

        Args:
            session_id: Session the file belongs to.
            filename:   Name to store under, e.g. "original.pdf" or "masked.pdf".
            data:       Raw file bytes.

        Returns:
            Full UC volume path of the stored file.

        TODO: upload via self._client.files.upload(path, io.BytesIO(data))
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.save_file is not yet implemented")

    def get_file(self, session_id: str, filename: str) -> bytes:
        """
        Read a binary artefact from the UC volume.

        Args:
            session_id: Session the file belongs to.
            filename:   Name stored under, e.g. "original.pdf".

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the file does not exist in the volume.

        TODO: download via self._client.files.download(path)
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.get_file is not yet implemented")

    # ------------------------------------------------------------------
    # Entity convenience methods
    # ------------------------------------------------------------------

    def save_entities(self, session_id: str, entities: List[Dict[str, Any]]) -> None:
        """
        Persist the entity list for a session.

        Entities are serialised as part of session.json — this is a
        convenience wrapper around update_session(entities=...).

        Args:
            session_id: Session to update.
            entities:   List of entity dicts (model_dump() output from Entity).

        TODO: delegates to update_session once that is implemented.
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.save_entities is not yet implemented")

    def get_entities(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load the stored entity list for a session.

        Returns:
            List of entity dicts, or [] if none have been saved yet.

        TODO: read from session.json via get_session()
        """
        # TODO: implement
        raise NotImplementedError("SessionManager.get_entities is not yet implemented")

    # ------------------------------------------------------------------
    # Legacy aliases (kept for backward compatibility with older call sites)
    # ------------------------------------------------------------------

    def upload_file(self, session_id: str, filename: str, data: bytes) -> str:
        """Alias for save_file()."""
        return self.save_file(session_id, filename, data)

    def download_file(self, session_id: str, filename: str) -> bytes:
        """Alias for get_file()."""
        return self.get_file(session_id, filename)


# ---------------------------------------------------------------------------
# UCSessionManager — concrete implementation via Databricks Files REST API
# ---------------------------------------------------------------------------

_VALID_STATUSES = ["uploaded", "processing", "awaiting_review", "completed"]


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

        Status is always written as ``"awaiting_review"``.

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
