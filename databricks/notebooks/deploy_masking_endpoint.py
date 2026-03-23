# Databricks notebook source
# DBTITLE 1,Cell 1
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
# MAGIC * **Documents must be uploaded to UC volume** at `/Volumes/{catalog}/{schema}/sessions/{session_id}/original.{ext}` before invoking the endpoint
# MAGIC
# MAGIC ## What This Notebook Does
# MAGIC 1. Configures endpoint settings (model, compute size, secrets)
# MAGIC 2. Creates or updates the serving endpoint
# MAGIC 3. Waits for endpoint to become ready
# MAGIC 4. Tests endpoint availability
# MAGIC 5. Provides endpoint URL for integration
# MAGIC
# MAGIC ## How the Model Works
# MAGIC The model expects a `session_id` and `field_definitions` as input. It fetches the original document from the UC volume using the session_id, processes it through OCR and vision AI, and returns detected entities with bounding boxes.
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
MODEL_NAME = "doc_masking"
UC_MODEL_PATH = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Endpoint configuration
ENDPOINT_NAME = "doc_masking"
MODEL_VERSION = "20"  # Update to your registered model version
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
# OPENAI_SECRET_SCOPE = "openai_00010_1"  # Update to your secret scope
# PROXY_SECRET_SCOPE = "nginx_proxy_sp"

env_vars = {
    # "VISION_SERVICE_PROVIDER": VISION_PROVIDER,
    "UC_VOLUME_PATH": UC_VOLUME_PATH,
    "DATABRICKS_HOST": "https://suncorp-dev.cloud.databricks.com/",
    "DATABRICKS_TOKEN": f"{{{{secrets/Conny.GUNADI@suncorp.com.au/DATABRICKS_TOKEN_DEV}}}}",
    # Azure Document Intelligence secrets
    # # "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": f"{{{{secrets/{SECRET_SCOPE}/adi_endpoint}}}}",
    # # "AZURE_DOCUMENT_INTELLIGENCE_KEY": f"{{{{secrets/{SECRET_SCOPE}/adi_key}}}}",
    # "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": "dummy_endpoint",
    # "AZURE_DOCUMENT_INTELLIGENCE_KEY": "dummy_key",
}


# Add OpenAI or Claude secrets based on provider
# if VISION_PROVIDER == "openai":
#     env_vars.update({
#         "DATABRICKS_HOST": "https://suncorp-dev.cloud.databricks.com/",
#         "DATABRICKS_TOKEN": f"{{{{secrets/Conny.GUNADI@suncorp.com.au/DATABRICKS_TOKEN_DEV}}}}",
#         "AZURE_OPENAI_API_KEY": f"{{{{secrets/{OPENAI_SECRET_SCOPE}/apikey}}}}",
#         "AZURE_OPENAI_ENDPOINT": "https://openai-00010-non-prod-1.openai.azure.com/",
#         "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
#         "AZURE_OPENAI_DEPLOYMENT_NAME": "gpt-5-global",
#         "WORKSPACE_URL": "https://suncorp-dev.cloud.databricks.com/",
#         "WORKSPACE_ID": "1238531023703058",
#         "PROXY_CLUSTER_ID": "0503-061117-o2rl78n9",
#         "PROXY_PORT": "8110",
#         "PROXY_ROUTE": "openai-00010-1",
#         "PROXY_CLIENT_ID": f"{{{{secrets/{PROXY_SECRET_SCOPE}/client_id}}}}",
#         "PROXY_CLIENT_SECRET": f"{{{{secrets/{PROXY_SECRET_SCOPE}/client_secret}}}}",
#     })
#     print("✅ OpenAI environment variables configured")
# elif VISION_PROVIDER == "claude":
#     env_vars.update({
#         "ANTHROPIC_API_KEY": "dummy_apikey",
#     })
#     print("✅ Claude environment variables configured")

print(f"\n📋 Environment variables:")
for key, value in env_vars.items():
    if "secrets" in str(value):
        print(f"   {key}: {value}")
    else:
        print(f"   {key}: {value}")

# COMMAND ----------

# DBTITLE 1,Cell 9
# MAGIC %md
# MAGIC ⚠️ **Important: Verify Secrets Exist**
# MAGIC
# MAGIC Before deploying, ensure these secrets exist in your Databricks workspace.
# MAGIC
# MAGIC **For Azure Document Intelligence:**
# MAGIC * Note: Currently using dummy values in the code - update Cell 8 with real secret references if needed
# MAGIC
# MAGIC **For OpenAI (if VISION_PROVIDER = "openai"):**
# MAGIC * `{SECRET_SCOPE}/apikey` - Azure OpenAI API key
# MAGIC * `{SECRET_SCOPE}/client_id` - Proxy client ID
# MAGIC * `{SECRET_SCOPE}/client_secret` - Proxy client secret
# MAGIC * `{SECRET_SCOPE}/DATABRICKS_TOKEN_DEV` - Databricks token
# MAGIC
# MAGIC **For Claude (if VISION_PROVIDER = "claude"):**
# MAGIC * Note: Currently using dummy value in the code - update Cell 8 with real secret reference if needed
# MAGIC
# MAGIC Run the cell below to verify secrets (optional):

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

            "session_id": "integration_test_001",
            "entities_to_mask": json.dumps([
                {
                    "entity_type": "Claimant Name",
                    "original_text": "Stephen Parrot",
                    "bounding_box": [
                        0.14965359893857966,
                        0.16382730180575378,
                        0.11323988798943743,
                        0.01599838422713382
                    ],
                    "confidence": 0.99,
                    "page_number": 1,
                    "bounding_boxes": [
                        {
                            "x": 0.14965359893857966,
                            "y": 0.16382730180575378,
                            "width": 0.11323988798943746,
                            "height": 0.015998384227133816
                        }
                    ],
                    "id": "76673463-773e-4100-bf63-7d31f1b3c72d",
                    "approved": True,
                    "strategy": "Fake Data",
                    "replacement_text": "William Long",
                    "occurrences": [
                        {
                            "page_number": 1,
                            "bounding_box": [
                            0.14965359893857966,
                            0.16382730180575378,
                            0.11323988798943746,
                            0.015998384227133816
                            ],
                            "original_text": "Stephen Parrot"
                        },
                        {
                            "page_number": 1,
                            "bounding_box": [
                            0.12094335670594705,
                            0.20772694728975996,
                            0.062211488372183626,
                            0.015998384227133816
                            ],
                            "original_text": "Stephen"
                        }
                    ]
                },
                {
                    "id": "test_e2",
                    "entity_type": "incident_date",
                    "bounding_box": [
                        0.6438296945689354,
                        0.286119207615939,
                        0.1270384139471667,
                        0.01599838422713382
                    ],
                    "bounding_boxes": [
                        {
                            "x": 0.6438296945689354,
                            "y": 0.286119207615939,
                            "width": 0.1270384139471667,
                            "height": 0.01599838422713382
                        }
                    ],
                    "original_text": "12 January 2020,",
                    "replacement_text": "incident_date_A",
                    "strategy": "Entity Label",
                    "approved": True,
                    "occurences": [
                        {
                            "page_number": 1,
                            "bounding_box": [
                                0.6438296945689354,
                                0.286119207615939,
                                0.1270384139471667,
                                0.01599838422713382
                            ],
                            "original_text": "12 January 2020,",
                        },
                    ]
                }
            ])
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

print(test_payload['dataframe_records'][0]['entities_to_mask'])

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

# DBTITLE 1,Cell 21
# MAGIC %md
# MAGIC ## Endpoint Invocation Examples
# MAGIC
# MAGIC ### Important: Document Upload Workflow
# MAGIC
# MAGIC Before invoking the endpoint:
# MAGIC 1. Upload your document to UC volume: `/Volumes/{catalog}/{schema}/sessions/{session_id}/original.{ext}`
# MAGIC 2. Send request with `session_id` and `field_definitions` to the endpoint
# MAGIC 3. The model fetches the document from UC volume, processes it, and returns entities
# MAGIC
# MAGIC ### Python Example
# MAGIC ```python
# MAGIC import requests
# MAGIC import json
# MAGIC import shutil
# MAGIC import os
# MAGIC
# MAGIC # Step 1: Upload document to UC volume
# MAGIC session_id = "my_session_123"
# MAGIC uc_volume_path = "/Volumes/datascience_dev_bronze_sandbox/ds_document_deidentification/sessions"
# MAGIC session_dir = f"{uc_volume_path}/{session_id}"
# MAGIC os.makedirs(session_dir, exist_ok=True)
# MAGIC
# MAGIC # Copy document to session directory as 'original.pdf'
# MAGIC shutil.copy2("path/to/document.pdf", f"{session_dir}/original.pdf")
# MAGIC
# MAGIC # Step 2: Define field definitions
# MAGIC field_definitions = [
# MAGIC     {
# MAGIC         "name": "claimant_name",
# MAGIC         "description": "The name of the claimant",
# MAGIC         "strategy": "Black Out"
# MAGIC     },
# MAGIC     {
# MAGIC         "name": "incident_date",
# MAGIC         "description": "The date when an incident happened",
# MAGIC         "strategy": "Black Out"
# MAGIC     }
# MAGIC ]
# MAGIC
# MAGIC # Step 3: Create request payload
# MAGIC payload = {
# MAGIC     "dataframe_records": [{
# MAGIC         "session_id": session_id,
# MAGIC         "field_definitions": field_definitions
# MAGIC     }]
# MAGIC }
# MAGIC
# MAGIC # Step 4: Invoke endpoint
# MAGIC response = requests.post(
# MAGIC     "{endpoint_url}",
# MAGIC     headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
# MAGIC     json=payload,
# MAGIC     timeout=120
# MAGIC )
# MAGIC
# MAGIC result = response.json()
# MAGIC print(json.dumps(result, indent=2))
# MAGIC ```
# MAGIC
# MAGIC ### Expected Response Format
# MAGIC ```json
# MAGIC {
# MAGIC   "predictions": [{
# MAGIC     "session_id": "my_session_123",
# MAGIC     "status": "complete",
# MAGIC     "pages": [
# MAGIC       {
# MAGIC         "page_num": 1,
# MAGIC         "entities": [
# MAGIC           {
# MAGIC             "id": "uuid",
# MAGIC             "entity_type": "claimant_name",
# MAGIC             "original_text": "John Doe",
# MAGIC             "bounding_box": [0.1, 0.2, 0.3, 0.4],
# MAGIC             "confidence": 0.95,
# MAGIC             "page_number": 1,
# MAGIC             "approved": true,
# MAGIC             "strategy": "Black Out"
# MAGIC           }
# MAGIC         ]
# MAGIC       }
# MAGIC     ]
# MAGIC   }]
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ### cURL Example
# MAGIC ```bash
# MAGIC # Step 1: Upload document using Files API (or use volume directly)
# MAGIC cp document.pdf /Volumes/catalog/schema/sessions/session_123/original.pdf
# MAGIC
# MAGIC # Step 2: Create request.json
# MAGIC cat > request.json << EOF
# MAGIC {
# MAGIC   "dataframe_records": [{
# MAGIC     "session_id": "session_123",
# MAGIC     "field_definitions": [
# MAGIC       {
# MAGIC         "name": "claimant_name",
# MAGIC         "description": "The name of the claimant",
# MAGIC         "strategy": "Black Out"
# MAGIC       }
# MAGIC     ]
# MAGIC   }]
# MAGIC }
# MAGIC EOF
# MAGIC
# MAGIC # Step 3: Invoke endpoint
# MAGIC curl -X POST {endpoint_url} \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" \
# MAGIC   -H "Content-Type: application/json" \
# MAGIC   -d @request.json
# MAGIC ```

# COMMAND ----------

import json
print(json.dumps(json.loads("[{\"entity_type\": \"Claimant Name\", \"original_text\": \"Stephen Parrot\", \"bounding_box\": [0.14965359893857966, 0.16382730180575378, 0.11323988798943743, 0.01599838422713382], \"confidence\": 0.99, \"page_number\": 1, \"bounding_boxes\": [{\"x\": 0.14965359893857966, \"y\": 0.16382730180575378, \"width\": 0.11323988798943746, \"height\": 0.015998384227133816}], \"id\": \"76673463-773e-4100-bf63-7d31f1b3c72d\", \"approved\": true, \"strategy\": \"Fake Data\", \"replacement_text\": \"William Long\", \"occurrences\": [{\"page_number\": 1, \"bounding_box\": [0.14965359893857966, 0.16382730180575378, 0.11323988798943746, 0.015998384227133816], \"original_text\": \"Stephen Parrot\"}, {\"page_number\": 1, \"bounding_box\": [0.12094335670594705, 0.20772694728975996, 0.062211488372183626, 0.015998384227133816], \"original_text\": \"Stephen\"}]}]"), indent=2))

# COMMAND ----------


