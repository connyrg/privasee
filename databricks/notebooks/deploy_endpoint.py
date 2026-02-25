# Databricks notebook: Deploy a Model Serving endpoint for PrivaSee
#
# Run this notebook after register_model.py has successfully registered
# the model in Unity Catalog.
#
# The endpoint URL produced here should be set as DATABRICKS_MODEL_ENDPOINT
# in the backend's .env file.
#
# Prerequisites:
#   - Model registered in UC (run register_model.py first)
#   - Workspace has Model Serving enabled
#   - The calling identity has "Can Manage" on the serving endpoint
#
# TODO:
#   1. Fill in CATALOG, SCHEMA, MODEL_NAME, ENDPOINT_NAME
#   2. Choose compute size (workload_size) appropriate for expected load
#   3. Add secret scope reference so the endpoint can access API keys at runtime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)

# ---------------------------------------------------------------------------
# Configuration — update before running
# ---------------------------------------------------------------------------
CATALOG       = "TODO_catalog"
SCHEMA        = "TODO_schema"
MODEL_NAME    = "privasee_document_intelligence"
MODEL_VERSION = "1"           # update to the UC model version to deploy
ENDPOINT_NAME = "privasee"

# ---------------------------------------------------------------------------
# Create or update the serving endpoint
# ---------------------------------------------------------------------------

# TODO: uncomment and run once configuration is filled in
# w = WorkspaceClient()
# w.serving_endpoints.create_and_wait(
#     name=ENDPOINT_NAME,
#     config=EndpointCoreConfigInput(
#         served_models=[
#             ServedModelInput(
#                 model_name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
#                 model_version=MODEL_VERSION,
#                 workload_size=ServedModelInputWorkloadSize.SMALL,
#                 scale_to_zero_enabled=True,
#             )
#         ]
#     ),
# )
# print(f"Endpoint '{ENDPOINT_NAME}' deployed.")
# print(f"Set DATABRICKS_MODEL_ENDPOINT to the endpoint's invocation URL.")

print(f"TODO: deploy endpoint '{ENDPOINT_NAME}' for {CATALOG}.{SCHEMA}.{MODEL_NAME} v{MODEL_VERSION}")
