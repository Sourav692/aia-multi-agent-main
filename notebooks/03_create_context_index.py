# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Create Context Index (Vector Search for Asset Discovery)
# MAGIC Creates the `ai_ops.context_index` table and Vector Search index
# MAGIC used by the Supervisor Agent to discover relevant data assets.

# COMMAND ----------

catalog = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Context Index Table
# MAGIC Catalog of all discoverable assets: Genie Spaces, Metric Views, Tables, Dashboards

# COMMAND ----------

from pyspark.sql import Row

assets = [
    # Genie Spaces
    Row(asset_type="genie_space", asset_id="01f0d6ff25da1f229950bb97c1ec974c",
        display_name="Bajaj Demo Genie Space",
        text="Genie Space for Bajaj demo analytics. Ask questions about claims, policies, customers, and business performance metrics.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "genie_space", "space_id": "01f0d6ff25da1f229950bb97c1ec974c", "warehouse_id": "4b9b953939869799"}'),

    # Metric Views
    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_claims_count",
        display_name="Claims Count Metric View",
        text="Total count of insurance claims by month, region, product category, claim type, and status. Use for claims volume analysis and trend monitoring.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "grain": "monthly", "dimensions": ["region", "product_category", "claim_type", "claim_status"]}'),

    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_claims_amount",
        display_name="Claims Amount Metric View",
        text="Total and average claim amounts in USD by month, region, product category, and claim type. Includes approved amounts and average processing days.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "grain": "monthly", "measures": ["total_claim_amount_usd", "avg_claim_amount_usd", "avg_processing_days"]}'),

    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_fraud_summary",
        display_name="Fraud Summary Metric View",
        text="Fraud risk analysis showing suspicious claims count and average fraud scores by month, region, and product category. Use for fraud monitoring and anomaly detection.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "measures": ["suspicious_claims_count", "avg_fraud_score", "suspicious_pct"]}'),

    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_policy_premium",
        display_name="Policy Premium Metric View",
        text="Total and average premium by region, product category, channel, and policy status. Use for premium analysis and distribution channel effectiveness.",
        domain="policies",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "measures": ["total_premium_usd", "avg_premium_usd", "total_sum_assured_usd"]}'),

    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_policy_mix",
        display_name="Policy Mix Metric View",
        text="Policy distribution and premium mix by product category and distribution channel. Shows premium share percentage across product lines.",
        domain="policies",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "measures": ["total_policies", "total_premium_usd", "premium_share_pct"]}'),

    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_agent_productivity",
        display_name="Agent Productivity Metric View",
        text="Agent productivity metrics including policies sold, premium generated, churn rate by agent, region, and channel. Use for agent performance analysis and incentive planning.",
        domain="distribution",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "measures": ["policies_sold", "total_premium_sold_usd", "churn_rate_pct"]}'),

    Row(asset_type="metric_view", asset_id=f"{catalog}.gold.mv_customer_segments",
        display_name="Customer Segments Metric View",
        text="Customer segment analysis showing policy count, premium, claims, and NPS by region and customer segment (Mass, Mass Affluent, High Net Worth, Ultra High Net Worth).",
        domain="customers",
        endorsement_level="endorsed",
        metadata='{"type": "metric_view", "dimensions": ["region", "segment"], "measures": ["customer_count", "avg_premium_per_customer", "avg_nps_score"]}'),

    # Gold Tables
    Row(asset_type="table", asset_id=f"{catalog}.gold.claims_summary",
        display_name="Claims Summary Table",
        text="Monthly aggregated claims metrics by region, product category, claim type, and status. Pre-aggregated for fast BI queries on claim trends.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "gold", "grain": "monthly"}'),

    Row(asset_type="table", asset_id=f"{catalog}.gold.policy_performance",
        display_name="Policy Performance Table",
        text="Policy performance metrics aggregated by region, product category, channel, and status. Includes policy counts, premium totals, and customer counts.",
        domain="policies",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "gold"}'),

    Row(asset_type="table", asset_id=f"{catalog}.gold.agent_performance",
        display_name="Agent Performance Table",
        text="Agent-level performance KPIs: policies sold, premium generated, customer count, active vs churned policies, and churn rate percentage.",
        domain="distribution",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "gold"}'),

    Row(asset_type="table", asset_id=f"{catalog}.gold.fraud_analysis",
        display_name="Fraud Analysis Table",
        text="Claims flagged with elevated fraud risk scores. Contains detailed claim information with fraud risk tiers (High, Medium, Low) and claim-to-income ratios.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "gold"}'),

    # Silver Tables
    Row(asset_type="table", asset_id=f"{catalog}.silver.enriched_claims",
        display_name="Enriched Claims Table",
        text="Detailed claim-level data enriched with customer demographics, policy details, and product information. Use for deep claim analysis, ad-hoc queries, and data science.",
        domain="claims",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "silver", "row_count_approx": 5000}'),

    Row(asset_type="table", asset_id=f"{catalog}.silver.enriched_policies",
        display_name="Enriched Policies Table",
        text="Policy-level data enriched with customer, product, and agent details. Use for underwriting analysis, portfolio management, and detailed policy queries.",
        domain="policies",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "silver", "row_count_approx": 3000}'),

    Row(asset_type="table", asset_id=f"{catalog}.silver.customer_360",
        display_name="Customer 360 Table",
        text="360-degree customer view with aggregated policy and claims metrics. Use for customer segmentation, lifetime value analysis, and retention strategies.",
        domain="customers",
        endorsement_level="endorsed",
        metadata='{"type": "table", "layer": "silver", "row_count_approx": 2000}'),

    # Policy Documents (for RAG)
    Row(asset_type="document_index", asset_id=f"{catalog}.bronze.policy_documents",
        display_name="Policy Documents Index",
        text="Collection of insurance policy documents including policy wordings, product disclosure sheets, benefit schedules, exclusion lists, claims procedure guides, FAQs, and underwriting guidelines. Use for answering questions about specific product coverage, benefits, exclusions, and procedures.",
        domain="documents",
        endorsement_level="endorsed",
        metadata='{"type": "document_index", "doc_types": ["Policy Wording", "Product Disclosure Sheet", "Benefit Schedule", "Exclusion List", "Claims Procedure Guide", "FAQ", "Underwriting Guidelines"]}'),
]

context_df = spark.createDataFrame(assets)
context_df.write.mode("overwrite").saveAsTable(f"{catalog}.ai_ops.context_index")
print(f"Created {catalog}.ai_ops.context_index with {len(assets)} assets")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enable Change Data Feed (required for Vector Search Delta Sync)

# COMMAND ----------

spark.sql(f"""
ALTER TABLE {catalog}.ai_ops.context_index
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print("Enabled CDF on context_index")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Vector Search Endpoint & Index

# COMMAND ----------

import requests, time

TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
HOST  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
HEADERS = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

VS_ENDPOINT = "aia_context_index_vs"

# Create Vector Search endpoint via REST
resp = requests.post(f"{HOST}/api/2.0/vector-search/endpoints",
    headers=HEADERS, json={"name": VS_ENDPOINT, "endpoint_type": "STANDARD"})
if resp.status_code in (200, 201):
    print(f"Creating Vector Search endpoint: {VS_ENDPOINT}")
elif "already exists" in resp.text.lower():
    print(f"Vector Search endpoint {VS_ENDPOINT} already exists")
else:
    print(f"Endpoint creation response {resp.status_code}: {resp.text[:300]}")

# COMMAND ----------

# Wait for endpoint to be ONLINE
for i in range(30):
    r = requests.get(f"{HOST}/api/2.0/vector-search/endpoints/{VS_ENDPOINT}", headers=HEADERS)
    state = r.json().get("endpoint_status", {}).get("state", "")
    print(f"Waiting for endpoint... state={state} ({i+1}/30)")
    if state == "ONLINE":
        print(f"Endpoint {VS_ENDPOINT} is ONLINE")
        break
    time.sleep(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Delta Sync Index

# COMMAND ----------

VS_INDEX = f"{catalog}.ai_ops.context_index_vs"

try:
    resp = requests.post(f"{HOST}/api/2.0/vector-search/indexes", headers=HEADERS, json={
        "name": VS_INDEX,
        "endpoint_name": VS_ENDPOINT,
        "primary_key": "asset_id",
        "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": f"{catalog}.ai_ops.context_index",
            "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [
                {"name": "text", "embedding_model_endpoint_name": "databricks-bge-large-en"}
            ],
        },
    })
    if resp.status_code in (200, 201):
        print(f"Creating Vector Search index: {VS_INDEX}")
    elif "already exists" in resp.text.lower():
        print(f"Index {VS_INDEX} already exists")
    else:
        raise Exception(f"Index creation failed {resp.status_code}: {resp.text[:300]}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Index {VS_INDEX} already exists")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Context Index Search UC Function

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.ai_ops.context_index_search(query STRING)
RETURNS TABLE(asset_type STRING, asset_id STRING, display_name STRING, text STRING, domain STRING, endorsement_level STRING, score DOUBLE)
COMMENT 'Semantic search over the Context Index — discovers Genie Spaces, metric views, tables, dashboards, and document indexes relevant to a user question.'
RETURN
  SELECT
    asset_type,
    asset_id,
    display_name,
    text,
    domain,
    endorsement_level,
    search_score AS score
  FROM VECTOR_SEARCH(
    index => '{VS_INDEX}',
    query => query,
    num_results => 10
  )
""")
print(f"Created UC function: {catalog}.ai_ops.context_index_search")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Also create Vector Search index for Policy Documents (RAG)

# COMMAND ----------

# Enable CDF on policy_documents
spark.sql(f"""
ALTER TABLE {catalog}.bronze.policy_documents
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# Create VS index for policy documents
DOC_INDEX = f"{catalog}.ai_ops.policy_docs_vs"

try:
    resp = requests.post(f"{HOST}/api/2.0/vector-search/indexes", headers=HEADERS, json={
        "name": DOC_INDEX,
        "endpoint_name": VS_ENDPOINT,
        "primary_key": "document_id",
        "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": f"{catalog}.bronze.policy_documents",
            "pipeline_type": "TRIGGERED",
            "embedding_source_columns": [
                {"name": "content", "embedding_model_endpoint_name": "databricks-bge-large-en"}
            ],
        },
    })
    if resp.status_code in (200, 201):
        print(f"Creating policy docs Vector Search index: {DOC_INDEX}")
    elif "already exists" in resp.text.lower():
        print(f"Index {DOC_INDEX} already exists")
    else:
        raise Exception(f"Doc index creation failed {resp.status_code}: {resp.text[:300]}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"Index {DOC_INDEX} already exists")
    else:
        raise e
