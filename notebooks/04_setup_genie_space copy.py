# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Setup Genie Spaces (Multi-Domain)
# MAGIC Creates multiple Genie Spaces, each scoped to a specific business domain.
# MAGIC The Supervisor Agent uses the Context Index to semantically route user
# MAGIC questions to the best-matching space at runtime.
# MAGIC
# MAGIC **Domains covered:**
# MAGIC - Claims Analytics (claims counts, amounts, processing, fraud)
# MAGIC - Policy & Underwriting (premiums, renewals, lapse rates, product mix)
# MAGIC - Distribution & Channels (agent performance, channel contributions, partner metrics)
# MAGIC - Customer Analytics (customer segments, retention, demographics)
# MAGIC
# MAGIC **Note:** Genie Spaces can be created via the UI or API. This notebook
# MAGIC provides the configuration for setup. Each space ID is stored in
# MAGIC `ai_ops.agent_config` and registered in the Context Index (notebook 03).

# COMMAND ----------

catalog = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Multi-Domain Genie Space Configurations

# COMMAND ----------

GENIE_SPACE_CONFIGS = [
    {
        "key": "genie_space_claims",
        "title": "Claims Analytics",
        "description": (
            "Ask questions about insurance claims across all AIA regions and products. "
            "Covers claim counts, amounts, processing times, approval rates, fraud analysis, "
            "customer segments, and policy premium trends."
        ),
        "tables": [
            f"{catalog}.gold.claims_summary",
            f"{catalog}.gold.fraud_analysis",
            f"{catalog}.silver.enriched_claims",
        ],
        "sample_questions": [
            "What is the total number of claims by region for the last 12 months?",
            "What is the average claim processing time by product category?",
            "Which regions have the highest fraud scores?",
            "Show me the monthly trend of hospitalization claims in Hong Kong",
            "What is the claim approval rate by claim type?",
            "How many suspicious claims were filed in Q4 2024?",
        ],
    },
    {
        "key": "genie_space_policies",
        "title": "Policy & Underwriting Analytics",
        "description": (
            "Ask questions about insurance policies, underwriting, and premium analytics. "
            "Covers premium volumes, policy counts, renewal rates, lapse rates, product mix, "
            "and underwriting performance across regions and product categories."
        ),
        "tables": [
            f"{catalog}.gold.policy_performance",
            f"{catalog}.silver.enriched_policies",
        ],
        "sample_questions": [
            "What is the total premium by distribution channel?",
            "Show me the policy renewal rate by region for this year",
            "Which product categories have the highest lapse rate?",
            "What is the month-over-month growth in new policy issuance?",
        ],
    },
    {
        "key": "genie_space_distribution",
        "title": "Distribution & Channels Analytics",
        "description": (
            "Ask questions about agent performance, distribution channels, and partner metrics. "
            "Covers agent productivity, sales pipeline, channel contribution percentages, "
            "commission analysis, and partner network performance."
        ),
        "tables": [
            f"{catalog}.gold.agent_performance",
        ],
        "sample_questions": [
            "Who are the top-performing agents by premium collected?",
            "What is the average policy count per agent by region?",
            "Show me channel contribution percentages for the last quarter",
        ],
    },
    {
        "key": "genie_space_customers",
        "title": "Customer Analytics",
        "description": (
            "Ask questions about customer segments, retention, demographics, and lifecycle. "
            "Covers customer segmentation, retention rates, claim frequency by segment, "
            "customer lifetime value, and demographic analysis."
        ),
        "tables": [
            f"{catalog}.silver.customer_360",
        ],
        "sample_questions": [
            "Which customer segments have the highest claim frequency?",
            "What is the retention rate by customer segment?",
            "Show me the demographic breakdown of our top-tier customers",
        ],
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Genie Spaces via API

# COMMAND ----------

# DBTITLE 1,Helper: create Genie space via REST API
import requests
import json
import uuid

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
host = w.config.host.rstrip("/")
_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
headers = {"Authorization": f"Bearer {_token}"}

# Use shared SQL warehouse (per workspace policy)
WAREHOUSE_ID = "862f1d757f0424f7"  # dbdemos-shared-endpoint

def create_genie_space(title: str, description: str, tables: list[str],
                       sample_questions: list[str] | None = None,
                       parent_path: str | None = None,
                       warehouse_id: str = WAREHOUSE_ID) -> str | None:
    """Create a Genie space via the REST API. Returns the space_id or None on failure."""
    sq = []
    for q in (sample_questions or []):
        sq.append({"id": uuid.uuid4().hex, "question": [q]})

    table_entries = [{"identifier": t} for t in tables]

    serialized = json.dumps({
        "version": 1,
        "config": {"sample_questions": sq} if sq else {},
        "data_sources": {"tables": table_entries},
        "instructions": {},
    })

    body = {
        "title": title,
        "description": description,
        "serialized_space": serialized,
        "warehouse_id": warehouse_id,
    }
    if parent_path:
        body["parent_path"] = parent_path

    resp = requests.post(f"{host}/api/2.0/genie/spaces", headers=headers, json=body)
    if resp.status_code == 200:
        return resp.json().get("space_id") or resp.json().get("id")
    else:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

print(f"Helper function ready (warehouse: {WAREHOUSE_ID}).")

# COMMAND ----------

# DBTITLE 1,Create all Genie spaces
created_spaces = {}

for config in GENIE_SPACE_CONFIGS:
    key = config["key"]
    title = config["title"]
    tables = config["tables"]
    description = config["description"]
    sample_questions = config.get("sample_questions", [])

    try:
        space_id = create_genie_space(
            title=title,
            description=description,
            tables=tables,
            sample_questions=sample_questions,
        )
        created_spaces[key] = space_id
        print(f"[OK] Created '{title}': {space_id}")
    except Exception as e:
        print(f"[WARN] '{title}' creation failed: {e}")
        print(f"       Please create manually via UI with tables: {', '.join(tables)}")
        created_spaces[key] = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store Genie Space IDs for Agent Use

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.agent_config (
    config_key STRING,
    config_value STRING,
    updated_at TIMESTAMP
)
""")

for key, space_id in created_spaces.items():
    if space_id:
        spark.sql(f"""
        MERGE INTO {catalog}.ai_ops.agent_config t
        USING (SELECT '{key}' AS config_key, '{space_id}' AS config_value, current_timestamp() AS updated_at) s
        ON t.config_key = s.config_key
        WHEN MATCHED THEN UPDATE SET t.config_value = s.config_value, t.updated_at = s.updated_at
        WHEN NOT MATCHED THEN INSERT *
        """)
        print(f"Stored {key} = {space_id}")

# Backward-compatible alias for the original claims space
claims_id = created_spaces.get("genie_space_claims")
if claims_id:
    spark.sql(f"""
    MERGE INTO {catalog}.ai_ops.agent_config t
    USING (SELECT 'bajaj_genie_space_id' AS config_key, '{claims_id}' AS config_value, current_timestamp() AS updated_at) s
    ON t.config_key = s.config_key
    WHEN MATCHED THEN UPDATE SET t.config_value = s.config_value, t.updated_at = s.updated_at
    WHEN NOT MATCHED THEN INSERT *
    """)
    print(f"Stored bajaj_genie_space_id (backward compat) = {claims_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Tables Are Accessible

# COMMAND ----------

all_tables = set()
for config in GENIE_SPACE_CONFIGS:
    all_tables.update(config["tables"])

for table in sorted(all_tables):
    try:
        count = spark.table(table).count()
        print(f"  {table}: {count} rows")
    except Exception as e:
        print(f"  {table}: NOT ACCESSIBLE — {str(e)[:80]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 60)
print("Genie Spaces Summary")
print("=" * 60)
for config in GENIE_SPACE_CONFIGS:
    key = config["key"]
    space_id = created_spaces.get(key, "NOT CREATED")
    print(f"  {config['title']}: {space_id}")
print()
print("Next step: Run notebook 03_create_context_index.py to register")
print("these spaces in the Context Index for semantic routing.")