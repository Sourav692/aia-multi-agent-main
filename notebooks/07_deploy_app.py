# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Deploy App & Grant Unity Catalog Permissions
# MAGIC
# MAGIC This notebook handles the complete app deployment lifecycle:
# MAGIC 1. **Creates or updates** the Databricks App (`aia-agent-360`)
# MAGIC 2. **Deploys** the app source code from the workspace
# MAGIC 3. **Discovers** the auto-created app service principal
# MAGIC 4. **Grants** Unity Catalog permissions so the app can read/write tables
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - The app source code must be synced to the workspace (via `databricks bundle deploy` or manually)
# MAGIC - The serving endpoint `agents_aia_multi_agent_catalog-ai_ops-supervisor_agent` must exist
# MAGIC - The SQL warehouse must be accessible

# COMMAND ----------

catalog = "aia_multi_agent_catalog"
app_name = "aia-agent-360"
app_description = "AIA Agent 360 – Multi-Agent Insurance Intelligence Chat UI (Dash)"
schemas = ["bronze", "silver", "gold", "ai_ops", "ai_insights"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 — Resolve the app source code path
# MAGIC The app source must already be in the workspace. We auto-detect the path
# MAGIC relative to this notebook.

# COMMAND ----------

import os

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
project_root = os.path.dirname(os.path.dirname(notebook_path))  # up from notebooks/
source_code_path = f"/Workspace{project_root}/app"

print(f"Notebook path:    {notebook_path}")
print(f"Project root:     {project_root}")
print(f"App source path:  {source_code_path}")

# Verify the source exists
try:
    files = dbutils.fs.ls(source_code_path.replace("/Workspace", "dbfs:"))
    print(f"✅ Source directory found ({len(files)} files)")
except Exception:
    # dbutils.fs may not work for workspace paths; try workspace API
    print("ℹ Could not verify via dbutils.fs — will proceed with deployment.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 — Create or Update the Databricks App

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppResource, AppResourceServingEndpoint, AppResourceSqlWarehouse

w = WorkspaceClient()

# Check if the app already exists
app_exists = False
try:
    existing_app = w.apps.get(app_name)
    app_exists = True
    print(f"ℹ App '{app_name}' already exists — will update and deploy.")
except Exception:
    print(f"ℹ App '{app_name}' not found — will create a new app.")

# Define app resources (these map to the env vars in app.yaml)
app_resources = [
    AppResource(
        name="serving-endpoint",
        serving_endpoint=AppResourceServingEndpoint(
            name="agents_aia_multi_agent_catalog-ai_ops-supervisor_agent",
            permission="CAN_QUERY",
        ),
    ),
    AppResource(
        name="sql-warehouse",
        sql_warehouse=AppResourceSqlWarehouse(
            name="Shared Autoscaling Warehouse",
            permission="CAN_USE",
        ),
    ),
]

if not app_exists:
    # Create the app
    print(f"Creating app '{app_name}'...")
    app_info = w.apps.create_and_wait(
        App(
            name=app_name,
            description=app_description,
            resources=app_resources,
        )
    )
    print(f"✅ App created: {app_info.name}")
else:
    # Update resources if needed
    try:
        app_info = w.apps.update(
            name=app_name,
            app=App(
                name=app_name,
                description=app_description,
                resources=app_resources,
            ),
        )
        print(f"✅ App updated: {app_info.name}")
    except Exception as e:
        print(f"ℹ App update skipped: {e}")
        app_info = existing_app

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 — Deploy the app source code

# COMMAND ----------

print(f"Deploying source code to app '{app_name}'...")
print(f"Source: {source_code_path}")

deployment = w.apps.deploy_and_wait(
    app_name=app_name,
    source_code_path=source_code_path,
)

print(f"\n✅ App deployed successfully!")
print(f"  Deployment ID:  {deployment.deployment_id}")
print(f"  Status:         {deployment.status}")

# Get the app URL
try:
    app_info = w.apps.get(app_name)
    app_url = getattr(app_info, "url", None) or getattr(app_info, "app_url", None)
    if app_url:
        print(f"  App URL:        {app_url}")
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 — Discover the App Service Principal

# COMMAND ----------

app_info = w.apps.get(app_name)

app_sp = (
    getattr(app_info, "service_principal_name", None)
    or getattr(app_info, "effective_service_principal_name", None)
)

# Fallback: search service principals list for the app pattern
if not app_sp:
    for sp in w.service_principals.list(filter=f"displayName co '{app_name}'"):
        app_sp = sp.display_name
        break

if not app_sp:
    raise RuntimeError(
        f"Could not find a service principal for app '{app_name}'. "
        f"The app was deployed but SP discovery failed — grant permissions manually."
    )

print(f"✅ App Service Principal: {app_sp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 — Grant Unity Catalog Permissions to App SP

# COMMAND ----------

quoted_sp = f"`{app_sp}`"

# USE CATALOG
spark.sql(f"GRANT USE CATALOG ON CATALOG {catalog} TO {quoted_sp}")
print(f"✅ USE CATALOG on {catalog}")

# Per-schema: USE SCHEMA + SELECT
for schema in schemas:
    fq_schema = f"{catalog}.{schema}"
    spark.sql(f"GRANT USE SCHEMA ON SCHEMA {fq_schema} TO {quoted_sp}")
    spark.sql(f"GRANT SELECT ON SCHEMA {fq_schema} TO {quoted_sp}")
    print(f"✅ USE SCHEMA + SELECT on {fq_schema}")

# MODIFY on ai_ops (session persistence, memory writes, prompt updates)
spark.sql(f"GRANT MODIFY ON SCHEMA {catalog}.ai_ops TO {quoted_sp}")
print(f"✅ MODIFY on {catalog}.ai_ops")

# MODIFY on ai_insights (agent-generated insight writes)
spark.sql(f"GRANT MODIFY ON SCHEMA {catalog}.ai_insights TO {quoted_sp}")
print(f"✅ MODIFY on {catalog}.ai_insights")

print(f"\n{'='*60}")
print(f"All UC permissions granted to '{app_sp}'")
print(f"{'='*60}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 — Verify Permissions

# COMMAND ----------

grants = spark.sql(f"SHOW GRANTS ON CATALOG {catalog}").collect()
app_grants = [g for g in grants if app_sp in str(g)]

print(f"Grants for '{app_sp}' on catalog '{catalog}':")
for g in app_grants:
    print(f"  {g}")

if not app_grants:
    print("⚠ No grants found — you may not have permission to view grants.")
else:
    print(f"\n✅ All {len(app_grants)} grants confirmed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The app is now deployed and authorized. Next steps:
# MAGIC - Open the app URL above to test the chat UI
# MAGIC - Try: *"What is the total number of claims by region for the last three months?"*
# MAGIC - If you see permission errors, verify you are the catalog owner or have `MANAGE` privilege
