# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Setup Genie Space for Claims Analytics
# MAGIC Creates a single Genie Space covering the claims domain with metric views and gold tables.
# MAGIC
# MAGIC **Note:** Genie Spaces are created via the UI or API. This notebook provides
# MAGIC the configuration and instructions for setup.

# COMMAND ----------

catalog = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genie Space Configuration
# MAGIC
# MAGIC **Name:** Claims Analytics
# MAGIC
# MAGIC **Description:** Ask questions about insurance claims across all regions and products.
# MAGIC Covers claim counts, amounts, processing times, approval rates, fraud analysis,
# MAGIC and trends by region, product category, claim type, and time period.
# MAGIC
# MAGIC **Tables to include:**
# MAGIC - `aia_multi_agent_catalog.gold.claims_summary`
# MAGIC - `aia_multi_agent_catalog.gold.fraud_analysis`
# MAGIC - `aia_multi_agent_catalog.silver.enriched_claims`
# MAGIC - `aia_multi_agent_catalog.gold.mv_claims_count`
# MAGIC - `aia_multi_agent_catalog.gold.mv_claims_amount`
# MAGIC - `aia_multi_agent_catalog.gold.mv_fraud_summary`
# MAGIC - `aia_multi_agent_catalog.gold.policy_performance`
# MAGIC - `aia_multi_agent_catalog.gold.mv_policy_premium`
# MAGIC - `aia_multi_agent_catalog.gold.mv_customer_segments`
# MAGIC
# MAGIC **Sample Questions:**
# MAGIC 1. What is the total number of claims by region for the last 12 months?
# MAGIC 2. What is the average claim processing time by product category?
# MAGIC 3. Which regions have the highest fraud scores?
# MAGIC 4. Show me the monthly trend of hospitalization claims in Hong Kong
# MAGIC 5. What is the claim approval rate by claim type?
# MAGIC 6. How many suspicious claims were filed in Q4 2024?
# MAGIC 7. What is the total premium by distribution channel?
# MAGIC 8. Which customer segments have the highest claim frequency?

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Genie Space via API

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

genie_tables = [
    f"{catalog}.gold.claims_summary",
    f"{catalog}.gold.fraud_analysis",
    f"{catalog}.silver.enriched_claims",
    f"{catalog}.gold.policy_performance",
    f"{catalog}.silver.customer_360",
]

# Attempt to create via Genie API
try:
    space = w.genie.create_space(
        title="Claims Analytics",
        description=(
            "Ask questions about insurance claims across all AIA regions and products. "
            "Covers claim counts, amounts, processing times, approval rates, fraud analysis, "
            "customer segments, and policy premium trends."
        ),
        table_identifiers=genie_tables,
    )
    genie_space_id = space.space_id
    print(f"Created Genie Space: {genie_space_id}")
except Exception as e:
    print(f"Genie Space creation via SDK: {e}")
    print("\nPlease create the Genie Space manually via the Databricks UI:")
    print("  1. Go to Genie in the left sidebar")
    print("  2. Click 'New Space'")
    print("  3. Name: 'Claims Analytics'")
    print(f"  4. Add tables: {', '.join(genie_tables)}")
    print("  5. Add the sample questions from the markdown above")
    genie_space_id = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Genie Space ID for Agent Use

# COMMAND ----------

if genie_space_id:
    # Store in a config table for agents to reference
    spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.agent_config (
        config_key STRING,
        config_value STRING,
        updated_at TIMESTAMP
    )
    """)

    spark.sql(f"""
    MERGE INTO {catalog}.ai_ops.agent_config t
    USING (SELECT 'bajaj_genie_space_id' AS config_key, '{genie_space_id}' AS config_value, current_timestamp() AS updated_at) s
    ON t.config_key = s.config_key
    WHEN MATCHED THEN UPDATE SET t.config_value = s.config_value, t.updated_at = s.updated_at
    WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"Stored Genie Space ID: {genie_space_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Tables Are Accessible

# COMMAND ----------

for table in genie_tables:
    count = spark.table(table).count()
    print(f"  {table}: {count} rows")
