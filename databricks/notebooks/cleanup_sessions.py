# Databricks notebook source
# MAGIC %md
# MAGIC # Cleanup Sessions and Resources
# MAGIC
# MAGIC This notebook provides utilities for cleaning up PrivaSee resources:
# MAGIC 1. **Session Cleanup** - Remove stale session data from Unity Catalog volumes
# MAGIC 2. **Endpoint Management** - Delete or manage serving endpoints
# MAGIC 3. **Storage Monitoring** - Monitor volume usage and statistics
# MAGIC
# MAGIC ## Use Cases
# MAGIC * **Scheduled Job**: Run this notebook daily/weekly to clean up old sessions
# MAGIC * **Manual Cleanup**: Run on-demand to free up storage space
# MAGIC * **Endpoint Deletion**: Remove endpoints when no longer needed
# MAGIC
# MAGIC ## Prerequisites
# MAGIC * Unity Catalog volume with session data
# MAGIC * Permissions to delete files from UC volumes
# MAGIC * "Can Manage" permissions on serving endpoints (for endpoint deletion)
# MAGIC
# MAGIC ## Scheduling
# MAGIC This notebook can be scheduled as a Databricks Job:
# MAGIC 1. Create a new job in Databricks
# MAGIC 2. Add this notebook as a task
# MAGIC 3. Set schedule (e.g., daily at 2 AM)
# MAGIC 4. Configure retention parameters via widgets or job parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC Update these values or use widgets for job scheduling

# COMMAND ----------

# Session cleanup configuration
CATALOG = "datascience_dev_bronze_sandbox"
SCHEMA = "privasee"
UC_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/privasee_sessions"

# Cleanup settings
MAX_SESSION_AGE_HOURS = 24  # Delete sessions older than this
DRY_RUN = True  # Set to False to actually delete files

# Endpoint configuration (for optional endpoint deletion)
ENDPOINT_NAME = "privasee_document_intelligence"

print(f"✅ Configuration loaded:")
print(f"   Volume Path: {UC_VOLUME_PATH}")
print(f"   Max Session Age: {MAX_SESSION_AGE_HOURS} hours")
print(f"   Dry Run Mode: {DRY_RUN}")
print(f"   Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Widgets for Job Parameters (Optional)
# MAGIC
# MAGIC Uncomment to create widgets for parameterized job execution

# COMMAND ----------

# Uncomment to use widgets for job parameters
# dbutils.widgets.text("uc_volume_path", UC_VOLUME_PATH, "UC Volume Path")
# dbutils.widgets.text("max_age_hours", str(MAX_SESSION_AGE_HOURS), "Max Age (hours)")
# dbutils.widgets.dropdown("dry_run", "True", ["True", "False"], "Dry Run")
# 
# # Read widget values
# UC_VOLUME_PATH = dbutils.widgets.get("uc_volume_path")
# MAX_SESSION_AGE_HOURS = int(dbutils.widgets.get("max_age_hours"))
# DRY_RUN = dbutils.widgets.get("dry_run") == "True"
# 
# print(f"✅ Widget values loaded:")
# print(f"   Volume: {UC_VOLUME_PATH}")
# print(f"   Max Age: {MAX_SESSION_AGE_HOURS}h")
# print(f"   Dry Run: {DRY_RUN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 1: Session Cleanup
# MAGIC
# MAGIC Clean up stale session directories from Unity Catalog volumes

# COMMAND ----------

from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Volume Exists

# COMMAND ----------

try:
    # Check if volume path exists
    dbutils.fs.ls(UC_VOLUME_PATH)
    print(f"✅ Volume path exists: {UC_VOLUME_PATH}")
except Exception as e:
    print(f"❌ Volume path does not exist or is not accessible: {UC_VOLUME_PATH}")
    print(f"   Error: {e}")
    print("   Please create the volume or update the path")
    dbutils.notebook.exit("Volume path not accessible")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analyze Session Storage

# COMMAND ----------

def analyze_session_storage(volume_path: str):
    """
    Analyze session storage usage and provide statistics.
    
    Args:
        volume_path: Path to Unity Catalog volume
    
    Returns:
        Dictionary with storage statistics
    """
    print(f"📊 Analyzing session storage: {volume_path}\n")
    
    try:
        session_dirs = dbutils.fs.ls(volume_path)
        
        total_sessions = 0
        total_size = 0
        sessions_by_age = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0}
        
        now = datetime.utcnow()
        
        for dir_info in session_dirs:
            if dir_info.isDir():
                total_sessions += 1
                
                # Calculate age
                mod_time = datetime.fromtimestamp(dir_info.modificationTime / 1000)
                age_hours = (now - mod_time).total_seconds() / 3600
                
                # Categorize by age
                if age_hours < 1:
                    sessions_by_age["<1h"] += 1
                elif age_hours < 6:
                    sessions_by_age["1-6h"] += 1
                elif age_hours < 24:
                    sessions_by_age["6-24h"] += 1
                else:
                    sessions_by_age[">24h"] += 1
                
                # Calculate size (approximate)
                try:
                    session_files = dbutils.fs.ls(dir_info.path)
                    for file in session_files:
                        total_size += file.size
                except:
                    pass
        
        # Convert size to MB
        total_size_mb = total_size / (1024 * 1024)
        
        print(f"📈 Storage Statistics:")
        print(f"   Total Sessions: {total_sessions}")
        print(f"   Total Size: {total_size_mb:.2f} MB")
        print(f"\n📅 Sessions by Age:")
        for age_range, count in sessions_by_age.items():
            print(f"   {age_range:>8}: {count:>4} sessions")
        
        return {
            "total_sessions": total_sessions,
            "total_size_mb": total_size_mb,
            "sessions_by_age": sessions_by_age
        }
        
    except Exception as e:
        print(f"❌ Error analyzing storage: {e}")
        return None

# COMMAND ----------

# Run storage analysis
storage_stats = analyze_session_storage(UC_VOLUME_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clean Up Stale Sessions

# COMMAND ----------

def cleanup_stale_sessions(volume_path: str, max_age_hours: int, dry_run: bool = True):
    """
    Clean up session directories older than max_age_hours.
    
    Args:
        volume_path: Path to Unity Catalog volume containing sessions
        max_age_hours: Maximum age in hours before deletion
        dry_run: If True, only print what would be deleted without actually deleting
    
    Returns:
        Dictionary with cleanup results
    """
    print(f"🧹 Cleaning up sessions from: {volume_path}")
    print(f"   Max age: {max_age_hours} hours")
    print(f"   Mode: {'DRY RUN (no actual deletion)' if dry_run else 'DELETE'}")
    print("=" * 80)
    
    cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
    print(f"   Cutoff time: {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
    
    try:
        # List directories in volume
        session_dirs = dbutils.fs.ls(volume_path)
        
        deleted_sessions = []
        kept_sessions = []
        deleted_size = 0
        
        for dir_info in session_dirs:
            if dir_info.isDir():
                session_path = dir_info.path
                session_id = session_path.rstrip('/').split('/')[-1]
                
                # Get modification time (in milliseconds)
                mod_time = datetime.fromtimestamp(dir_info.modificationTime / 1000)
                age_hours = (datetime.utcnow() - mod_time).total_seconds() / 3600
                
                # Calculate session size
                session_size = 0
                try:
                    session_files = dbutils.fs.ls(session_path)
                    for file in session_files:
                        session_size += file.size
                except:
                    pass
                
                if mod_time < cutoff_time:
                    # Session is old enough to delete
                    print(f"   {'🗑️  Would delete' if dry_run else '🗑️  Deleting'}: {session_id}")
                    print(f"      Age: {age_hours:.1f}h, Size: {session_size / (1024*1024):.2f} MB")
                    print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if not dry_run:
                        try:
                            dbutils.fs.rm(session_path, recurse=True)
                            print(f"      ✅ Deleted")
                        except Exception as e:
                            print(f"      ❌ Error: {e}")
                            continue
                    
                    deleted_sessions.append(session_id)
                    deleted_size += session_size
                else:
                    kept_sessions.append(session_id)
        
        # Print summary
        deleted_size_mb = deleted_size / (1024 * 1024)
        print("\n" + "=" * 80)
        print(f"✅ Cleanup {'simulation' if dry_run else 'complete'}:")
        print(f"   Sessions {'to delete' if dry_run else 'deleted'}: {len(deleted_sessions)}")
        print(f"   Sessions kept: {len(kept_sessions)}")
        print(f"   Space {'to free' if dry_run else 'freed'}: {deleted_size_mb:.2f} MB")
        
        return {
            "deleted_count": len(deleted_sessions),
            "kept_count": len(kept_sessions),
            "deleted_size_mb": deleted_size_mb,
            "deleted_sessions": deleted_sessions,
        }
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        raise

# COMMAND ----------

# Run cleanup
print("🧪 Running cleanup...\n")
cleanup_result = cleanup_stale_sessions(
    volume_path=UC_VOLUME_PATH,
    max_age_hours=MAX_SESSION_AGE_HOURS,
    dry_run=DRY_RUN
)

# COMMAND ----------

# MAGIC %md
# MAGIC ⚠️ **To actually delete sessions, set `DRY_RUN = False` in the configuration cell and re-run**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Endpoint Management
# MAGIC
# MAGIC Manage and optionally delete serving endpoints

# COMMAND ----------

# MAGIC %pip install -q databricks-sdk>=0.20.0
# MAGIC
# MAGIC # Restart Python to load packages
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointStateReady

w = WorkspaceClient()
print("✅ Workspace client initialized")

# COMMAND ----------

# MAGIC %md
# MAGIC ### List Serving Endpoints

# COMMAND ----------

def list_endpoints():
    """List all serving endpoints in the workspace."""
    print("📋 Serving Endpoints:\n")
    
    try:
        endpoints = w.serving_endpoints.list()
        
        for endpoint in endpoints:
            print(f"   Name: {endpoint.name}")
            print(f"   State: {endpoint.state.ready if endpoint.state else 'Unknown'}")
            print(f"   Config: {endpoint.state.config_update if endpoint.state else 'Unknown'}")
            
            if endpoint.config and endpoint.config.served_entities:
                for entity in endpoint.config.served_entities:
                    print(f"   Model: {entity.entity_name} v{entity.entity_version}")
            
            print()
        
        return list(endpoints)
        
    except Exception as e:
        print(f"❌ Error listing endpoints: {e}")
        return []

# COMMAND ----------

# List all endpoints
endpoints = list_endpoints()
print(f"✅ Found {len(endpoints)} endpoint(s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Endpoint Details

# COMMAND ----------

def get_endpoint_details(endpoint_name: str):
    """Get detailed information about a specific endpoint."""
    print(f"🔍 Endpoint Details: {endpoint_name}\n")
    
    try:
        endpoint = w.serving_endpoints.get(name=endpoint_name)
        
        print(f"   Name: {endpoint.name}")
        print(f"   ID: {endpoint.id}")
        print(f"   State: {endpoint.state.ready if endpoint.state else 'Unknown'}")
        print(f"   Config Update: {endpoint.state.config_update if endpoint.state else 'Unknown'}")
        
        if endpoint.config:
            print(f"\n   Configuration:")
            if endpoint.config.served_entities:
                for entity in endpoint.config.served_entities:
                    print(f"      Model: {entity.entity_name}")
                    print(f"      Version: {entity.entity_version}")
                    print(f"      Workload Size: {entity.workload_size}")
                    print(f"      Scale to Zero: {entity.scale_to_zero_enabled}")
        
        return endpoint
        
    except Exception as e:
        print(f"❌ Error getting endpoint details: {e}")
        return None

# COMMAND ----------

# Get details for our endpoint
endpoint_details = get_endpoint_details(ENDPOINT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete Endpoint (Optional)

# COMMAND ----------

def delete_endpoint(endpoint_name: str, confirm: bool = False):
    """
    Delete a model serving endpoint.
    
    Args:
        endpoint_name: Name of the endpoint to delete
        confirm: Must be True to actually delete (safety check)
    
    Returns:
        True if deleted, False if not confirmed
    """
    if not confirm:
        print(f"ℹ️  Endpoint deletion not confirmed: {endpoint_name}")
        print("   To delete, call: delete_endpoint('{endpoint_name}', confirm=True)")
        return False
    
    print(f"⚠️  Deleting endpoint: {endpoint_name}")
    
    try:
        w.serving_endpoints.delete(name=endpoint_name)
        print(f"✅ Endpoint '{endpoint_name}' deleted successfully")
        return True
    except Exception as e:
        print(f"❌ Error deleting endpoint: {e}")
        raise

# COMMAND ----------

# Uncomment to delete the endpoint (use with caution!)
# delete_endpoint(ENDPOINT_NAME, confirm=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ⚠️ **Warning: Endpoint deletion is permanent!**
# MAGIC
# MAGIC To actually delete an endpoint, run:
# MAGIC ```python
# MAGIC delete_endpoint(ENDPOINT_NAME, confirm=True)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary and Recommendations

# COMMAND ----------

print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🧹 Cleanup Summary                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Session Cleanup:                                                        ║
║    Volume Path:    {UC_VOLUME_PATH:<54}║
║    Max Age:        {MAX_SESSION_AGE_HOURS} hours{' ' * 54}║
║    Sessions:       {cleanup_result['deleted_count']} to delete, {cleanup_result['kept_count']} kept{' ' * (53 - len(str(cleanup_result['deleted_count'])) - len(str(cleanup_result['kept_count'])))}║
║    Space:          {cleanup_result['deleted_size_mb']:.2f} MB to free{' ' * (53 - len(f"{cleanup_result['deleted_size_mb']:.2f}"))}║
║    Mode:           {'DRY RUN' if DRY_RUN else 'DELETE'}{' ' * (54 - len('DRY RUN' if DRY_RUN else 'DELETE'))}║
║                                                                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Recommendations:                                                        ║
║  • Schedule this notebook as a daily/weekly job                          ║
║  • Set DRY_RUN=False to enable actual deletion                           ║
║  • Monitor storage usage regularly                                       ║
║  • Adjust MAX_SESSION_AGE_HOURS based on retention requirements          ║
║  • Archive important sessions before cleanup if needed                   ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scheduling This Notebook
# MAGIC
# MAGIC To schedule this notebook as a job:
# MAGIC
# MAGIC 1. **Create a Job**:
# MAGIC    * Go to Workflows → Jobs → Create Job
# MAGIC    * Add this notebook as a task
# MAGIC
# MAGIC 2. **Configure Schedule**:
# MAGIC    * Set schedule (e.g., "0 2 * * *" for daily at 2 AM)
# MAGIC    * Or use "Cron" for custom scheduling
# MAGIC
# MAGIC 3. **Set Parameters**:
# MAGIC    * Use widgets or set variables in job parameters
# MAGIC    * Important: Set `DRY_RUN = False` for actual cleanup
# MAGIC
# MAGIC 4. **Configure Alerts**:
# MAGIC    * Add email notifications on job failure
# MAGIC    * Monitor job run history
# MAGIC
# MAGIC 5. **Test First**:
# MAGIC    * Run manually with `DRY_RUN = True`
# MAGIC    * Verify what would be deleted
# MAGIC    * Then enable actual deletion
