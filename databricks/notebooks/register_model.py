# Databricks notebook source
# MAGIC %md
# MAGIC # Document Intelligence Model - Registration
# MAGIC
# MAGIC This notebook logs and registers the DocumentIntelligenceModel in Unity Catalog with MLflow.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC * Databricks Runtime ML 14.x or later
# MAGIC * Unity Catalog enabled with a target catalog/schema
# MAGIC * Model code in `../model/` directory
# MAGIC * Requirements file at `../requirements.txt`
# MAGIC
# MAGIC ## Next Steps
# MAGIC After running this notebook:
# MAGIC 1. Deploy the model using `deploy_endpoint.py`
# MAGIC 2. Manage sessions using `cleanup_sessions.py`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# Install required packages
# %pip install -q mlflow>=2.10.0 pandas>=2.0.0 PyMuPDF>=1.23.0 python-docx>=1.1.0 \
#     azure-ai-documentintelligence>=1.0.0 azure-core>=1.29.0 \
#     anthropic>=0.25.0 openai>=1.12.0
%pip install -r ../requirements.txt
# Restart Python to ensure packages are loaded
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Update these values before running the notebook

# COMMAND ----------

# Model registration configuration
CATALOG = "datascience_dev_bronze_sandbox"
SCHEMA = "ds_document_deidentification"
MODEL_NAME = "doc_deidentification"
UC_MODEL_PATH = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# UC Volume for session storage (will be used by deployed model)
UC_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/sessions"

print(f"✅ Configuration loaded:")
print(f"   Model: {UC_MODEL_PATH}")
print(f"   Volume: {UC_VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Model and Setup MLflow

# COMMAND ----------

# DBTITLE 1,Cell 7
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import sys
import os

# Add model directory to path
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
module_dir = os.path.join("/Workspace" + os.path.dirname(notebook_path), "../../..")
sys.path.insert(0, module_dir)

print(f"✅ Model directory added to path: {module_dir}")

# COMMAND ----------

# Import the DocumentIntelligenceModel
try:
    from privasee.databricks.model.document_intelligence import DocumentIntelligenceModel
    print("✅ DocumentIntelligenceModel imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DocumentIntelligenceModel: {e}")
    print(f"   Make sure the model files are in: {module_dir}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Model Signature

# COMMAND ----------

# DBTITLE 1,Cell 10
# Define MLflow model signature
# Input: DataFrame with document request data
# Output: DataFrame with processing results
# Note: Output fields may contain None values - this is handled by the model's predict() method

input_schema = Schema([
    ColSpec("string", "session_id"),
    ColSpec("string", "document_bytes_b64"),
    ColSpec("string", "document_filename"),
    ColSpec("string", "field_definitions_json"),
])

output_schema = Schema([
    ColSpec("string", "session_id"),
    ColSpec("string", "status"),
    ColSpec("string", "ocr_result_json", required=False),
    ColSpec("string", "entity_detection_json", required=False),
    ColSpec("string", "bbox_matches_json", required=False),
    ColSpec("string", "masked_pdf_path", required=False),
    ColSpec("string", "error_message", required=False),
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)
print("✅ Model signature defined")

# COMMAND ----------

# Create input example for model card
import pandas as pd
import json

input_example = pd.DataFrame({
    "session_id": ["example_session_123"],
    "document_bytes_b64": ["base64_encoded_document_bytes_here"],
    "document_filename": ["sample_document.pdf"],
    "field_definitions_json": [json.dumps({
        "PERSON_NAME": {"label": "Person Name", "category": "PII"},
        "ADDRESS": {"label": "Address", "category": "PII"},
        "SSN": {"label": "Social Security Number", "category": "SENSITIVE"}
    })]
})

print("✅ Input example created")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log and Register Model to Unity Catalog

# COMMAND ----------

# Set MLflow registry to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Set or create experiment
experiment_path = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/privasee_experiments"
mlflow.set_experiment(experiment_path)

print(f"✅ Using experiment: {experiment_path}")

# COMMAND ----------

# DBTITLE 1,Cell 14
# Get requirements from file
requirements_path = os.path.join(os.path.dirname(module_dir), "databricks/requirements.txt")
print(f"📦 Loading requirements from: {requirements_path}")

# Log the model
with mlflow.start_run(run_name="privasee_model_registration") as run:
    print("📝 Logging model to MLflow...")
    
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=DocumentIntelligenceModel(),
        signature=signature,
        input_example=input_example,
        pip_requirements=requirements_path,
        # code_paths=[os.path.join(module_dir, "privasee/databricks/model")],
        code_paths=[os.path.join(module_dir, "privasee")],
    )
    
    # Log additional metadata
    mlflow.log_param("catalog", CATALOG)
    mlflow.log_param("schema", SCHEMA)
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("uc_volume_path", UC_VOLUME_PATH)
    
    print(f"✅ Model logged: {model_info.model_uri}")

# COMMAND ----------

# Register model in Unity Catalog
print(f"📋 Registering model to Unity Catalog: {UC_MODEL_PATH}")

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=UC_MODEL_PATH,
    tags={"project": "privasee", "type": "document_intelligence"}
)

print(f"✅ Model registered successfully!")
print(f"   Name: {registered_model.name}")
print(f"   Version: {registered_model.version}")

# Save version for use in other notebooks
MODEL_VERSION = str(registered_model.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Registered Model (Optional)

# COMMAND ----------

# DBTITLE 1,Untitled
# Load and test the registered model
print(f"🧪 Testing registered model: {UC_MODEL_PATH} v{MODEL_VERSION}")

import os
os.environ['AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT']='endpoint'
os.environ['AZURE_DOCUMENT_INTELLIGENCE_KEY']='apikey'
os.environ['WORKSPACE_URL']='url'
os.environ['WORKSPACE_ID']='id'
os.environ['PROXY_CLUSTER_ID']='cluster_id'
os.environ['PROXY_CLIENT_ID']='client_id'
os.environ['PROXY_CLIENT_SECRET']='client_secret'
os.environ['PROXY_PORT']='port'
os.environ['PROXY_ROUTE']='route'
os.environ['AZURE_OPENAI_ENDPOINT']='endpoint'
os.environ['AZURE_OPENAI_API_KEY']='apikey'
os.environ['UC_VOLUME_PATH']='/Volumes/datascience_dev_bronze_sandbox/ds_document_deidentification/sessions'
loaded_model = mlflow.pyfunc.load_model(f"models:/{UC_MODEL_PATH}/{MODEL_VERSION}")
print("✅ Model loaded successfully")

# Note: Full testing requires valid Azure credentials and a real document
# The actual predict() call is skipped here to avoid credential errors
print("ℹ️  Skipping predict test (requires Azure credentials)")
print("   Test the model via the endpoint once deployed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Model Registration Complete!**
# MAGIC
# MAGIC * **Model Name:** `{UC_MODEL_PATH}`
# MAGIC * **Version:** `{MODEL_VERSION}`
# MAGIC * **Experiment:** `{experiment_path}`
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC 1. **Deploy the model** using `deploy_endpoint.py`
# MAGIC 2. **Manage sessions** using `cleanup_sessions.py`
# MAGIC 3. **Configure secrets** for Azure Document Intelligence and OpenAI/Claude
# MAGIC 4. **Test the endpoint** with real documents

# COMMAND ----------

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🎉 Registration Complete! 🎉                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Model Path:     {UC_MODEL_PATH:<56}║
║  Model Version:  {MODEL_VERSION:<56}║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

Next: Run deploy_endpoint.py to create a serving endpoint for this model.
""")

# COMMAND ----------

# DBTITLE 1,Run All Tests - Verify Fixes

