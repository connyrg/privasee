"""
DocumentIntelligence MLflow PyFunc model.

This is the top-level model class that will be registered in Unity Catalog
and deployed to Databricks Model Serving.

Responsibilities:
  1. Receive a single-page document image (base64 PNG) + OCR data + field
     definitions via the Model Serving REST interface.
  2. Orchestrate OCRService (Azure Document Intelligence) and ClaudeService
     (Anthropic Claude Vision) to extract sensitive entities.
  3. Return a list of Entity objects with bounding boxes and confidence scores.

MLflow model signature (to be finalised):
  Input:  { "session_id": str, "image_base64": str,
             "ocr_data": dict, "field_definitions": list[dict] }
  Output: { "entities": list[dict], "model_version": str }

Usage:
  Registered and deployed via databricks/notebooks/register_model.py and
  databricks/notebooks/deploy_endpoint.py.

TODO:
  - Define mlflow.pyfunc.PythonModel subclass with predict() method
  - Implement load_context() to load credentials from Databricks secrets
  - Wire up OCRService and ClaudeService (see ocr_service.py, claude_service.py)
  - Define and log the MLflow model signature
  - Add input validation / error handling
"""

import mlflow.pyfunc


class DocumentIntelligenceModel(mlflow.pyfunc.PythonModel):
    """MLflow PyFunc wrapper for the PrivaSee entity-extraction pipeline."""

    def load_context(self, context):
        """
        Load secrets and initialise downstream services.

        Args:
            context: mlflow.pyfunc.PythonModelContext — provides access to
                     artefacts and model config stored alongside the model.

        TODO: retrieve AZURE_DOC_INTEL_* and ANTHROPIC_API_KEY from
              Databricks secret scope via dbutils.secrets.get().
        """
        # TODO: implement
        pass

    def predict(self, context, model_input):
        """
        Run the entity-extraction pipeline for a single page.

        Args:
            context: mlflow.pyfunc.PythonModelContext
            model_input: pandas DataFrame or dict with keys:
                         session_id, image_base64, ocr_data, field_definitions

        Returns:
            dict with key "entities" (list of entity dicts) and "model_version"

        TODO: implement
        """
        raise NotImplementedError
