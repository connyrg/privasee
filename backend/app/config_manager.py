"""
Config Manager

Persists named field-definition configs on a Databricks Unity Catalog volume
using the Files REST API (``/api/2.0/fs/files``).

Configs are stored alongside sessions:
    {UC_VOLUME_PATH}/../configs/{key}.json

where ``key`` is the sanitised config name (lowercase, alphanumeric + hyphens/underscores).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _sanitise_key(name: str) -> str:
    """Convert a human-readable config name to a safe filename stem.

    Examples:
        "Patient Record (v2)" → "patient_record__v2_"
        "Full Name / DOB"     → "full_name___dob"
    """
    return re.sub(r"[^\w\-]", "_", name.strip().lower())


class ConfigManager:
    """
    Manages named field-definition configs on a Databricks UC volume.

    Configs live at ``{configs_path}/{key}.json`` where ``configs_path`` is
    derived from the sessions volume path by replacing the last segment with
    ``configs``.
    """

    _FILES_API = "/api/2.0/fs/files"

    def __init__(self, databricks_host: str, token: str, sessions_volume_path: str) -> None:
        """
        Args:
            databricks_host:      Workspace URL, e.g. https://adb-xxxx.azuredatabricks.net
            token:                Personal access token with Files API permissions.
            sessions_volume_path: UC volume path used for sessions, e.g.
                                  /Volumes/catalog/schema/sessions.  The configs
                                  directory is derived from this automatically.
        """
        self.host = databricks_host.rstrip("/")
        self.token = token
        # Derive configs path: replace last path segment with "configs"
        base = sessions_volume_path.rstrip("/").rsplit("/", 1)[0]
        self.configs_path = f"{base}/configs"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    def _url(self, uc_path: str) -> str:
        return f"{self.host}{self._FILES_API}{uc_path}"

    def save_config(self, config_name: str, field_definitions: List[Dict[str, Any]]) -> str:
        """Persist a named config. Overwrites an existing config with the same key.

        Returns the sanitised key. Raises requests.HTTPError on storage failure.
        """
        key = _sanitise_key(config_name)
        payload = {
            "config_name": config_name,
            "key": key,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "field_definitions": field_definitions,
        }
        r = requests.put(
            self._url(f"{self.configs_path}/{key}.json"),
            headers=self._headers(),
            data=json.dumps(payload),
        )
        r.raise_for_status()
        logger.info("Saved config %r as %s.json", config_name, key)
        return key

    def list_configs(self) -> List[Dict[str, Any]]:
        """Return summary metadata (config_name, key, saved_at) for all saved configs.

        Returns [] if the configs directory does not yet exist.
        Raises requests.HTTPError on unexpected storage errors.
        """
        r = requests.get(self._url(f"{self.configs_path}/"), headers=self._headers())
        if r.status_code in (400, 404):
            # 404 = directory doesn't exist yet; Databricks Files API may also
            # return 400 for a path that has never been written to.
            return []
        r.raise_for_status()

        summaries = []
        for entry in r.json().get("contents", []):
            # The Files API returns the full path; derive the filename from it.
            # e.g. "/Volumes/catalog/schema/configs/patient_record.json" → "patient_record.json"
            full_path = entry.get("path", "") or entry.get("name", "")
            filename = full_path.rstrip("/").split("/")[-1]
            if not filename.endswith(".json"):
                continue
            config = self.get_config(filename[:-5])  # strip .json to get key
            if config:
                summaries.append({
                    "config_name": config["config_name"],
                    "key": config["key"],
                    "saved_at": config["saved_at"],
                })
        return summaries

    def get_config(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a config by its sanitised key.

        Returns None if not found (404). Raises requests.HTTPError on other errors.
        """
        r = requests.get(
            self._url(f"{self.configs_path}/{key}.json"),
            headers=self._headers(),
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
