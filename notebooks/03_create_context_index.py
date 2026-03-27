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
# MAGIC Catalog of all discoverable assets: Genie Spaces and Document Indexes

# COMMAND ----------

from pyspark.sql import Row

DOC_VS_INDEX = f"{catalog}.ai_ops.policy_docs_vs"

WH_ID = "4b9b953939869799"

assets = [
    # ---------------------------------------------------------------------------
    # Genie Spaces — one per business domain.
    # asset_id is the Genie Space ID used by the worker agent.
    # ---------------------------------------------------------------------------
    Row(asset_type="genie_space", asset_id="01f1272d4ba6144ba75d868762f1925d",
        display_name="Claims Analytics Space",
        text="Genie Space for claims analytics. Covers claim counts, claim amounts, claim processing times, approval rates, fraud analysis, loss ratios, and suspicious claims. Ask questions about claims by region, product category, claim type, time period, and status.",
        domain="claims",
        endorsement_level="endorsed",
        metadata=f'{{"type": "genie_space", "space_id": "01f1272d4ba6144ba75d868762f1925d", "warehouse_id": "{WH_ID}"}}'),

    Row(asset_type="genie_space", asset_id="01f1272d4c6b1fb49223785ab841befd",
        display_name="Policy & Underwriting Space",
        text="Genie Space for policy and underwriting analytics. Covers premium volumes, policy counts, renewal rates, lapse rates, product mix, new business issuance, and underwriting performance. Ask questions about policies by region, product category, distribution channel, and time period.",
        domain="policies",
        endorsement_level="endorsed",
        metadata=f'{{"type": "genie_space", "space_id": "01f1272d4c6b1fb49223785ab841befd", "warehouse_id": "{WH_ID}"}}'),

    Row(asset_type="genie_space", asset_id="01f1272d4d271203ad122e9280470248",
        display_name="Distribution & Channels Space",
        text="Genie Space for distribution and channel analytics. Covers agent productivity, sales pipeline, channel contribution percentages, commission analysis, partner network performance, and agent rankings. Ask questions about agent performance, distribution channels, and partner metrics.",
        domain="distribution",
        endorsement_level="endorsed",
        metadata=f'{{"type": "genie_space", "space_id": "01f1272d4d271203ad122e9280470248", "warehouse_id": "{WH_ID}"}}'),

    Row(asset_type="genie_space", asset_id="01f1272d4de1188cac8feeb7e71bdb69",
        display_name="Customer Analytics Space",
        text="Genie Space for customer analytics. Covers customer segmentation, retention rates, claim frequency by segment, customer lifetime value, demographic analysis, and customer 360 views. Ask questions about customer segments, demographics, retention, and customer behavior.",
        domain="customers",
        endorsement_level="endorsed",
        metadata=f'{{"type": "genie_space", "space_id": "01f1272d4de1188cac8feeb7e71bdb69", "warehouse_id": "{WH_ID}"}}'),

    # ---------------------------------------------------------------------------
    # Document Indexes — metadata.vs_index tells the Multi-Tool agent which VS
    # index to query.
    # ---------------------------------------------------------------------------
    Row(asset_type="document_index", asset_id=f"{catalog}.bronze.policy_documents",
        display_name="Policy Documents Index",
        text="Collection of insurance policy documents including policy wordings, product disclosure sheets, benefit schedules, exclusion lists, claims procedure guides, FAQs, and underwriting guidelines. Use for answering questions about specific product coverage, benefits, exclusions, and procedures.",
        domain="documents",
        endorsement_level="endorsed",
        metadata=f'{{"type": "document_index", "vs_index": "{DOC_VS_INDEX}", "doc_types": ["Policy Wording", "Product Disclosure Sheet", "Benefit Schedule", "Exclusion List", "Claims Procedure Guide", "FAQ", "Underwriting Guidelines"]}}'),
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
COMMENT 'Semantic search over the Context Index — discovers Genie Spaces and Document Indexes relevant to a user question.'
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
