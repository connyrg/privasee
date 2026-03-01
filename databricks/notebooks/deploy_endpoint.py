# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Model Serving Endpoint
# MAGIC
# MAGIC This notebook deploys the registered DocumentIntelligenceModel to a Model Serving endpoint.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC * Model registered in Unity Catalog (run `register_model.py` first)
# MAGIC * Workspace has Model Serving enabled
# MAGIC * Calling identity has "Can Manage" permissions on serving endpoints
# MAGIC * Databricks secrets configured for Azure and OpenAI/Claude credentials
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC 1. Configures endpoint settings (model, compute size, secrets)
# MAGIC 2. Creates or updates the serving endpoint
# MAGIC 3. Waits for endpoint to become ready
# MAGIC 4. Tests endpoint availability
# MAGIC 5. Provides endpoint URL for integration
# MAGIC
# MAGIC ## Previous Step
# MAGIC Run `register_model.py` to register the model first.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# MAGIC %pip install -q databricks-sdk>=0.20.0
# MAGIC
# MAGIC # Restart Python to ensure packages are loaded
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Update these values to match your registered model

# COMMAND ----------

# Model configuration - must match values from register_model.py
CATALOG = "datascience_dev_bronze_sandbox"
SCHEMA = "ds_document_deidentification"
MODEL_NAME = "doc_deidentification"
UC_MODEL_PATH = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Endpoint configuration
ENDPOINT_NAME = "privasee_endpoint_local"
MODEL_VERSION = "7"  # Update to your registered model version
WORKLOAD_SIZE = "Small"  # Options: Small, Medium, Large
SCALE_TO_ZERO = True  # Enable scale-to-zero to save costs

# UC Volume for session storage
UC_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/sessions"

# Vision service provider
VISION_PROVIDER = "openai"  # Options: "openai" or "claude"

print(f"✅ Configuration loaded:")
print(f"   Model: {UC_MODEL_PATH} v{MODEL_VERSION}")
print(f"   Endpoint: {ENDPOINT_NAME}")
print(f"   Workload: {WORKLOAD_SIZE}")
print(f"   Scale to Zero: {SCALE_TO_ZERO}")
print(f"   Vision Provider: {VISION_PROVIDER}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configure Endpoint Environment Variables
# MAGIC
# MAGIC The endpoint needs access to Azure credentials via Databricks secrets.
# MAGIC Update the secret scope and key names to match your configuration.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    EndpointStateReady,
)

# Initialize Workspace client
w = WorkspaceClient()
print("✅ Workspace client initialized")

# COMMAND ----------

# Environment variables for the endpoint
# These reference Databricks secrets - update scope and key names
OPENAI_SECRET_SCOPE = "openai_00010_1"  # Update to your secret scope
PROXY_SECRET_SCOPE = "nginx_proxy_sp"

env_vars = {
    "VISION_SERVICE_PROVIDER": VISION_PROVIDER,
    "UC_VOLUME_PATH": UC_VOLUME_PATH,
    # Azure Document Intelligence secrets
    # "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": f"{{{{secrets/{SECRET_SCOPE}/adi_endpoint}}}}",
    # "AZURE_DOCUMENT_INTELLIGENCE_KEY": f"{{{{secrets/{SECRET_SCOPE}/adi_key}}}}",
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "dummy_endpoint",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY": "dummy_key",
}

# Add OpenAI or Claude secrets based on provider
if VISION_PROVIDER == "openai":
    env_vars.update({
        "AZURE_OPENAI_API_KEY": f"{{{{secrets/{OPENAI_SECRET_SCOPE}/apikey}}}}",
        "AZURE_OPENAI_ENDPOINT": "https://openai-00010-non-prod-1.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5-global",
        "WORKSPACE_URL": "https://suncorp-dev.cloud.databricks.com/",
        "WORKSPACE_ID": "1238531023703058",
        "PROXY_CLUSTER_ID": "0503-061117-o2rl78n9",
        "PROXY_PORT": "8110",
        "PROXY_ROUTE": "openai-00010-1",
        "PROXY_CLIENT_ID": f"{{{{secrets/{PROXY_SECRET_SCOPE}/client_id}}}}",
        "PROXY_CLIENT_SECRET": f"{{{{secrets/{PROXY_SECRET_SCOPE}/client_secret}}}}",
    })
    print("✅ OpenAI environment variables configured")
elif VISION_PROVIDER == "claude":
    env_vars.update({
        "ANTHROPIC_API_KEY": "dummy_apikey",
    })
    print("✅ Claude environment variables configured")

print(f"\n📋 Environment variables:")
for key, value in env_vars.items():
    if "secrets" in str(value):
        print(f"   {key}: {value}")
    else:
        print(f"   {key}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ⚠️ **Important: Verify Secrets Exist**
# MAGIC
# MAGIC Before deploying, ensure these secrets exist in your Databricks workspace:
# MAGIC
# MAGIC For Azure Document Intelligence:
# MAGIC * `{SECRET_SCOPE}/adi_endpoint`
# MAGIC * `{SECRET_SCOPE}/adi_key`
# MAGIC
# MAGIC For OpenAI (if using):
# MAGIC * `{SECRET_SCOPE}/openai_key`
# MAGIC * `{SECRET_SCOPE}/openai_endpoint`
# MAGIC
# MAGIC For Claude (if using):
# MAGIC * `{SECRET_SCOPE}/anthropic_key`
# MAGIC
# MAGIC Run this cell to verify secrets (optional):

# COMMAND ----------

# # Verify secrets exist (optional)
# try:
#     # Try to list secrets in the scope
#     secrets = dbutils.secrets.list(SECRET_SCOPE)
#     print(f"✅ Secret scope '{SECRET_SCOPE}' exists with {len(secrets)} secrets:")
#     for secret in secrets:
#         print(f"   - {secret.key}")
# except Exception as e:
#     print(f"⚠️  Warning: Could not access secret scope '{SECRET_SCOPE}'")
#     print(f"   Error: {e}")
#     print("   Make sure the secret scope exists and contains required secrets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Endpoint

# COMMAND ----------

# DBTITLE 1,Cell 12
from databricks.sdk.errors import ResourceDoesNotExist

print(f"🚀 Deploying endpoint: {ENDPOINT_NAME}")
print(f"   Model: {UC_MODEL_PATH} v{MODEL_VERSION}")
print(f"   Workload size: {WORKLOAD_SIZE}")
print(f"   Scale to zero: {SCALE_TO_ZERO}")

try:
    # Check if endpoint already exists
    existing_endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)
    print(f"\nℹ️  Endpoint '{ENDPOINT_NAME}' already exists")
    print(f"   Current state: {existing_endpoint.state.config_update}")
    print(f"   Updating to model version {MODEL_VERSION}...")
    
    # Update existing endpoint
    w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=UC_MODEL_PATH,
                entity_version=MODEL_VERSION,
                workload_size=WORKLOAD_SIZE,
                scale_to_zero_enabled=SCALE_TO_ZERO,
                environment_vars=env_vars,
            )
        ],
    )
    print(f"✅ Endpoint '{ENDPOINT_NAME}' updated successfully!")
    
except ResourceDoesNotExist:
    print(f"\nℹ️  Endpoint does not exist, creating new endpoint...")
    print(f"   This may take 5-10 minutes...")
    
    # Create new endpoint
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=UC_MODEL_PATH,
                    entity_version=MODEL_VERSION,
                    workload_size=WORKLOAD_SIZE,
                    scale_to_zero_enabled=SCALE_TO_ZERO,
                    environment_vars=env_vars,
                )
            ]
        ),
    )
    print(f"✅ Endpoint '{ENDPOINT_NAME}' created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for Endpoint to Be Ready

# COMMAND ----------

import time

print("⏳ Waiting for endpoint to be ready...")
max_wait = 600  # 10 minutes
wait_interval = 15
elapsed = 0

while elapsed < max_wait:
    endpoint_status = w.serving_endpoints.get(name=ENDPOINT_NAME)
    
    if endpoint_status.state.ready == EndpointStateReady.READY:
        print(f"✅ Endpoint is ready! (waited {elapsed}s)")
        break
    else:
        status = endpoint_status.state.config_update
        print(f"   Status: {status} (waited {elapsed}s / {max_wait}s)")
        time.sleep(wait_interval)
        elapsed += wait_interval

if elapsed >= max_wait:
    print(f"⚠️  Endpoint is taking longer than expected ({max_wait}s)")
    print("   Check the endpoint status in the Databricks UI")
    print("   The endpoint may still become ready - monitor in the Serving UI")
else:
    print(f"\n🎉 Endpoint deployment complete in {elapsed} seconds!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Endpoint Details

# COMMAND ----------

# Get endpoint details
endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)

# Get workspace URL
workspace_url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}"
endpoint_url = f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"

print(f"📊 Endpoint Details:")
print(f"   Name: {endpoint.name}")
print(f"   State: {endpoint.state.ready}")
print(f"   Config Update: {endpoint.state.config_update}")
print(f"\n🔗 Endpoint URL:")
print(f"   {endpoint_url}")
print(f"\n💡 Use this URL in your application to invoke the model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Endpoint Availability

# COMMAND ----------

import requests
import json

# Get API token
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("🧪 Testing endpoint availability...")

# Create minimal test payload
test_payload = {
    "dataframe_records": [
        {
            "session_id": "test_session_001",
            "document_bytes_b64": "VGVzdCBkb2N1bWVudA==",  # "Test document" in base64
            "document_filename": "test.pdf",
            "field_definitions_json": json.dumps({
                "PERSON_NAME": {"label": "Person Name", "category": "PII"}
            })
        }
    ]
}

try:
    print(f"📤 Sending test request to: {endpoint_url}")
    response = requests.post(
        endpoint_url,
        headers=headers,
        json=test_payload,
        timeout=120
    )
    
    print(f"\n📥 Response received:")
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Endpoint is responding correctly!")
        print(f"   Response preview: {json.dumps(result, indent=2)[:500]}...")
    else:
        print(f"⚠️  Endpoint returned non-200 status:")
        print(f"   Response: {response.text[:500]}")
        
except requests.exceptions.Timeout:
    print("⚠️  Request timed out (120s)")
    print("   This may be normal for first request if endpoint is scaling from zero")
    print("   Try again in a few minutes")
except Exception as e:
    print(f"⚠️  Error testing endpoint: {e}")
    print("   This may be expected if:")
    print("   - Secrets are not properly configured")
    print("   - Test data is invalid")
    print("   - Model is still initializing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Next Steps

# COMMAND ----------

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🎉 Endpoint Deployed! 🎉                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Endpoint Name:  {ENDPOINT_NAME:<56}║
║  Model Path:     {UC_MODEL_PATH:<56}║
║  Model Version:  {MODEL_VERSION:<56}║
║  Workload Size:  {WORKLOAD_SIZE:<56}║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Endpoint URL:                                                           ║
║  {endpoint_url:<73}║
╠══════════════════════════════════════════════════════════════════════════╣
║  Next Steps:                                                             ║
║  1. Test with real documents using the endpoint URL above                ║
║  2. Monitor endpoint in Databricks Serving UI                            ║
║  3. Run cleanup_sessions.py periodically to manage storage               ║
║  4. Configure alerting for endpoint errors                               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Invocation Examples
# MAGIC
# MAGIC ### Python Example
# MAGIC ```python
# MAGIC import requests
# MAGIC import json
# MAGIC import base64
# MAGIC
# MAGIC # Read and encode document
# MAGIC with open("document.pdf", "rb") as f:
# MAGIC     doc_bytes = base64.b64encode(f.read()).decode()
# MAGIC
# MAGIC # Create request payload
# MAGIC payload = {
# MAGIC     "dataframe_records": [{
# MAGIC         "session_id": "session_123",
# MAGIC         "document_bytes_b64": doc_bytes,
# MAGIC         "document_filename": "document.pdf",
# MAGIC         "field_definitions_json": json.dumps({
# MAGIC             "PERSON_NAME": {"label": "Person Name", "category": "PII"},
# MAGIC             "ADDRESS": {"label": "Address", "category": "PII"}
# MAGIC         })
# MAGIC     }]
# MAGIC }
# MAGIC
# MAGIC # Invoke endpoint
# MAGIC response = requests.post(
# MAGIC     "{endpoint_url}",
# MAGIC     headers={{"Authorization": f"Bearer {{DATABRICKS_TOKEN}}"}},
# MAGIC     json=payload
# MAGIC )
# MAGIC
# MAGIC result = response.json()
# MAGIC ```
# MAGIC
# MAGIC ### cURL Example
# MAGIC ```bash
# MAGIC curl -X POST {endpoint_url} \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d @request.json
# MAGIC ```
