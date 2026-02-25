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
from typing import Any, Dict, List, Optional

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
