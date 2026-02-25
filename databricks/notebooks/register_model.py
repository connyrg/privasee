# Databricks notebook: Register DocumentIntelligenceModel in Unity Catalog
#
# Run this notebook in a Databricks cluster after updating the TODO sections.
# It logs the model to MLflow and registers it under the UC model registry.
#
# Prerequisites:
#   - Databricks Runtime ML 14.x or later
#   - Unity Catalog enabled with a target catalog/schema
#   - Databricks secrets configured for Azure and Anthropic credentials
#   - All files under databricks/model/ uploaded to the cluster or a repo
#
# TODO:
#   1. Set CATALOG, SCHEMA, MODEL_NAME to your target UC path
#   2. Confirm the MLflow model signature matches DatabricksInferenceRequest
#   3. Update conda_env / pip_requirements from databricks/model/requirements.txt
#   4. Add any model artefacts (e.g. font files) to the logged model artefacts

import mlflow
import mlflow.pyfunc

# ---------------------------------------------------------------------------
# Configuration — update before running
# ---------------------------------------------------------------------------
CATALOG    = "TODO_catalog"
SCHEMA     = "TODO_schema"
MODEL_NAME = "privasee_document_intelligence"
UC_MODEL_PATH = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# ---------------------------------------------------------------------------
# Log and register the model
# ---------------------------------------------------------------------------

# TODO: import DocumentIntelligenceModel from databricks/model/document_intelligence.py
# from document_intelligence import DocumentIntelligenceModel

# TODO: define MLflow model signature
# from mlflow.models.signature import ModelSignature
# from mlflow.types.schema import Schema, ColSpec
# signature = ModelSignature(inputs=..., outputs=...)

# TODO: log model and register in UC
# with mlflow.start_run():
#     mlflow.pyfunc.log_model(
#         artifact_path="model",
#         python_model=DocumentIntelligenceModel(),
#         signature=signature,
#         pip_requirements="databricks/model/requirements.txt",
#         registered_model_name=UC_MODEL_PATH,
#     )

print(f"TODO: register model at {UC_MODEL_PATH}")
