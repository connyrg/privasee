# Databricks notebook: Clean up stale sessions from the Unity Catalog volume
#
# Schedule this notebook as a Databricks Job (e.g. daily) to remove session
# artefacts that are older than MAX_AGE_HOURS.
#
# What it deletes:
#   - {UC_VOLUME_PATH}/{session_id}/ directories older than MAX_AGE_HOURS
#   - This includes uploaded PDFs, page images, masked PDFs, and session.json
#
# Prerequisites:
#   - UC_VOLUME_PATH Databricks secret (or widget) pointing to the sessions volume
#   - The job principal has Files API delete permissions on the volume
#
# TODO:
#   1. Set UC_VOLUME_PATH and MAX_AGE_HOURS (or read from job parameters / widgets)
#   2. Implement listing + deletion using databricks-sdk Files API
#   3. Add logging / alerting for cleanup failures

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — update or replace with dbutils.widgets
# ---------------------------------------------------------------------------
UC_VOLUME_PATH = "TODO:/Volumes/catalog/schema/privasee_sessions"
MAX_AGE_HOURS  = 24

# ---------------------------------------------------------------------------
# Cleanup logic
# ---------------------------------------------------------------------------
cutoff = datetime.utcnow() - timedelta(hours=MAX_AGE_HOURS)

# TODO: list session directories under UC_VOLUME_PATH
# TODO: for each directory, check creation/modification time
# TODO: delete directories older than cutoff using WorkspaceClient().files.delete()

print(f"TODO: delete sessions older than {cutoff.isoformat()} under {UC_VOLUME_PATH}")
