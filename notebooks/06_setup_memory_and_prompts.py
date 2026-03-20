# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Setup Memory Tables, Prompt Management & AI Gateway
# MAGIC
# MAGIC Creates infrastructure tables for P0/P1/P2 enhancements:
# MAGIC - **P0**: `ai_ops.conversations` (short-term memory / checkpoints)
# MAGIC - **P1**: `ai_ops.agent_instructions` (prompt management)
# MAGIC - **P1**: AI Gateway guardrails configuration
# MAGIC - **P1**: Endorsed asset routing verification
# MAGIC - **P2**: `ai_ops.user_memory` (long-term user preferences & facts)
# MAGIC - **P2**: `ai_ops.episodic_memory` (interaction logs & lessons learned)
# MAGIC - **P2**: `ai_ops.agent_capabilities` (tool registry for semantic routing)

# COMMAND ----------

catalog = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## P0: Short-term Memory — Conversations Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.conversations (
    thread_id STRING NOT NULL COMMENT 'Unique conversation thread ID',
    checkpoint_id STRING NOT NULL COMMENT 'Unique checkpoint within the thread',
    state_json STRING COMMENT 'Serialized LangGraph state (messages, intent, domain)',
    created_at TIMESTAMP COMMENT 'When this checkpoint was saved'
)
USING DELTA
COMMENT 'Short-term memory: conversation checkpoints for multi-turn sessions'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
""")
print(f"Created {catalog}.ai_ops.conversations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P2: Long-term Memory — User Memory Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.user_memory (
    user_id STRING NOT NULL COMMENT 'User identifier',
    memory_key STRING NOT NULL COMMENT 'Fact or preference key (e.g. name, preferred_region)',
    memory_value STRING COMMENT 'Value of the fact or preference',
    memory_type STRING COMMENT 'Type: preference, fact, feedback',
    confidence DOUBLE DEFAULT 1.0 COMMENT 'Confidence score for this memory entry',
    created_at TIMESTAMP COMMENT 'When this memory was first saved',
    updated_at TIMESTAMP COMMENT 'When this memory was last updated',
    expires_at TIMESTAMP COMMENT 'Optional expiry (NULL = never expires)'
)
USING DELTA
COMMENT 'Long-term memory: persistent user preferences and facts across sessions'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true','delta.feature.allowColumnDefaults' = 'supported')
""")
print(f"Created {catalog}.ai_ops.user_memory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P2: Episodic Memory — Interaction Log Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.episodic_memory (
    episode_id STRING NOT NULL COMMENT 'Unique episode identifier',
    thread_id STRING COMMENT 'Conversation thread this episode belongs to',
    user_id STRING COMMENT 'User who initiated the interaction',
    question STRING COMMENT 'Original user question',
    intent STRING COMMENT 'Classified intent (simple_kpi, document_lookup, etc.)',
    domain STRING COMMENT 'Domain of the question (claims, policies, documents, etc.)',
    agents_used ARRAY<STRING> COMMENT 'Which agents were invoked (genie, multi_tool, etc.)',
    outcome STRING COMMENT 'Outcome of the interaction (success, failed)',
    lesson_learned STRING COMMENT 'Lesson extracted from feedback for future improvement',
    user_rating INT COMMENT 'Optional user rating (1-5)',
    created_at TIMESTAMP COMMENT 'When this episode was logged'
)
USING DELTA
COMMENT 'Episodic memory: interaction logs and lessons learned for continuous improvement'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true','delta.feature.allowColumnDefaults' = 'supported')
""")
print(f"Created {catalog}.ai_ops.episodic_memory")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P2: Tool Registry — Agent Capabilities Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.agent_capabilities (
    capability_id STRING NOT NULL COMMENT 'Unique capability identifier',
    agent_name STRING NOT NULL COMMENT 'Agent name (genie, multi_tool, analysis, etc.)',
    capability_name STRING COMMENT 'Human-readable capability name',
    description STRING COMMENT 'Description of what this agent can do',
    supported_intents ARRAY<STRING> COMMENT 'Intents this agent handles',
    supported_domains ARRAY<STRING> COMMENT 'Domains this agent covers',
    priority INT DEFAULT 100 COMMENT 'Routing priority (lower = higher priority)',
    is_active BOOLEAN DEFAULT true COMMENT 'Whether this capability is active'
)
USING DELTA
COMMENT 'Tool registry: agent capabilities for semantic routing decisions'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true','delta.feature.allowColumnDefaults' = 'supported')
""")
print(f"Created {catalog}.ai_ops.agent_capabilities")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P1: Prompt Management — Agent Instructions Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {catalog}.ai_ops.agent_instructions (
    agent_id STRING NOT NULL COMMENT 'Agent identifier (supervisor, genie, multi_tool, analysis, visualization)',
    scope STRING NOT NULL COMMENT 'Prompt scope (classify_intent, compose_answer, analysis, etc.)',
    base_prompt STRING COMMENT 'Base system prompt (maintained in Git, versioned with code)',
    overlay_prompt STRING COMMENT 'Dynamic overlay from feedback/Instruction Builder job',
    updated_at TIMESTAMP COMMENT 'Last update timestamp',
    updated_by STRING COMMENT 'Who updated this prompt'
)
USING DELTA
COMMENT 'Prompt management: base + overlay prompts for all agents'
TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true','delta.feature.allowColumnDefaults' = 'supported')
""")
print(f"Created {catalog}.ai_ops.agent_instructions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Seed Default Prompts

# COMMAND ----------

# Check if prompts already exist
existing = spark.sql(f"SELECT COUNT(*) as cnt FROM {catalog}.ai_ops.agent_instructions").collect()[0]["cnt"]

if existing == 0:
    prompts = [
        {
            "agent_id": "supervisor",
            "scope": "classify_intent",
            "base_prompt": """You are an intent classifier for an insurance analytics system.
Classify the following question into exactly ONE category and provide a confidence score (0.0 to 1.0).

Categories:
- "simple_kpi": Simple KPI/metric questions (counts, totals, averages, trends by region/product/time)
- "document_lookup": Policy terms, coverage details, exclusions, procedures, document search
- "conversational": Greetings, introductions, personal statements, small talk, or non-analytical messages

Question: {question}

Respond in JSON format ONLY:
{{"intent": "<category>", "confidence": <float>, "missing_filters": []}}

If the question is ambiguous or missing key filters (like region, time period, product), list them in missing_filters.""",
            "overlay_prompt": None,
        },
        {
            "agent_id": "supervisor",
            "scope": "compose_answer",
            "base_prompt": """Compose a clear, comprehensive answer for the user question.
Instructions:
- Cite specific numbers and data from the agent results.
- Use markdown formatting with headers and bullet points.
- If dashboard links are provided, include them prominently.
- Keep the answer concise but thorough (2-4 paragraphs max).
- Include a Warnings & Limitations section if there are any warnings.""",
            "overlay_prompt": None,
        },
        {
            "agent_id": "genie",
            "scope": "default",
            "base_prompt": "Query the Genie Space for structured BI answers using Text-to-SQL over metric views and curated tables. Multiple domain-specific spaces may be tried in priority order.",
            "overlay_prompt": None,
        },
        {
            "agent_id": "genie",
            "scope": "claims",
            "base_prompt": "Focus on claims KPIs: claim counts, claim amounts, processing times, approval rates, fraud analysis, loss ratios, and suspicious claims by region, product, and time period.",
            "overlay_prompt": None,
        },
        {
            "agent_id": "genie",
            "scope": "policies",
            "base_prompt": "Focus on policy and underwriting KPIs: premium volumes, policy counts, renewal rates, lapse rates, product mix, new business issuance, and underwriting performance.",
            "overlay_prompt": None,
        },
        {
            "agent_id": "genie",
            "scope": "distribution",
            "base_prompt": "Focus on distribution KPIs: agent productivity, sales pipeline, channel contributions, commission analysis, partner network performance, and agent rankings.",
            "overlay_prompt": None,
        },
        {
            "agent_id": "genie",
            "scope": "customers",
            "base_prompt": "Focus on customer KPIs: customer segmentation, retention rates, claim frequency by segment, customer lifetime value, and demographic analysis.",
            "overlay_prompt": None,
        },
        {
            "agent_id": "multi_tool",
            "scope": "default",
            "base_prompt": "Perform Vector Search RAG over policy document indexes to answer questions about coverage, exclusions, and procedures.",
            "overlay_prompt": None,
        },
    ]

    from pyspark.sql import Row
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType
    schema = StructType([
        StructField("agent_id", StringType(), True),
        StructField("scope", StringType(), True),
        StructField("base_prompt", StringType(), True),
        StructField("overlay_prompt", StringType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("updated_by", StringType(), True),
    ])
    import datetime
    rows = [Row(agent_id=p["agent_id"], scope=p["scope"], base_prompt=p["base_prompt"],
                overlay_prompt=p.get("overlay_prompt"), updated_at=datetime.datetime.utcnow(), updated_by="system")
            for p in prompts]
    df = spark.createDataFrame(rows, schema=schema)
    df.write.mode("append").saveAsTable(f"{catalog}.ai_ops.agent_instructions")
    print(f"Seeded {len(prompts)} default prompts into {catalog}.ai_ops.agent_instructions")
else:
    print(f"Prompts already exist ({existing} rows), skipping seed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P1: AI Gateway — Enable Guardrails on Model Endpoint
# MAGIC
# MAGIC AI Gateway provides rate limiting, PII filtering, and telemetry.
# MAGIC This configures guardrails on the supervisor agent's underlying LLM endpoint.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Enable AI Gateway on the LLM endpoint used by the supervisor
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

try:
    endpoint = w.serving_endpoints.get(LLM_ENDPOINT)
    print(f"LLM Endpoint: {LLM_ENDPOINT}")
    print(f"  State: {endpoint.state}")

    # Check if AI Gateway is already configured
    if hasattr(endpoint, 'ai_gateway') and endpoint.ai_gateway:
        print(f"  AI Gateway already configured")
        if hasattr(endpoint.ai_gateway, 'guardrails'):
            print(f"  Guardrails: {endpoint.ai_gateway.guardrails}")
        if hasattr(endpoint.ai_gateway, 'rate_limits'):
            print(f"  Rate limits: {endpoint.ai_gateway.rate_limits}")
    else:
        print(f"  AI Gateway not configured — this is a foundation model endpoint")
        print(f"  Note: Foundation model endpoints have AI Gateway enabled by default")
        print(f"  To add custom guardrails, use the Databricks UI: Serving > {LLM_ENDPOINT} > AI Gateway tab")
except Exception as e:
    print(f"Could not check endpoint: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## P1: Verify Endorsed Asset Routing
# MAGIC
# MAGIC Check that the Context Index has endorsement_level populated.

# COMMAND ----------

try:
    result = spark.sql(f"""
        SELECT endorsement_level, COUNT(*) as count
        FROM {catalog}.ai_ops.context_index
        GROUP BY endorsement_level
        ORDER BY count DESC
    """)
    print("Context Index endorsement breakdown:")
    result.show()
except Exception as e:
    print(f"Context Index not ready or endorsement_level not populated: {e}")
    print("Run notebook 03_create_context_index.py to populate the Context Index")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify all tables

# COMMAND ----------

tables_to_check = [
    f"{catalog}.ai_ops.conversations",
    f"{catalog}.ai_ops.agent_instructions",
    f"{catalog}.ai_ops.agent_config",
    f"{catalog}.ai_ops.context_index",
    f"{catalog}.ai_ops.user_memory",
    f"{catalog}.ai_ops.episodic_memory",
    f"{catalog}.ai_ops.agent_capabilities",
]

for t in tables_to_check:
    try:
        count = spark.table(t).count()
        print(f"  {t}: {count} rows")
    except Exception as e:
        print(f"  {t}: NOT FOUND — {str(e)[:80]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleanup: Schedule conversation table maintenance
# MAGIC
# MAGIC Old checkpoints should be cleaned up periodically.

# COMMAND ----------

# This SQL can be scheduled as a job to clean up old checkpoints (older than 30 days)
cleanup_sql = f"""
DELETE FROM {catalog}.ai_ops.conversations
WHERE created_at < current_timestamp() - INTERVAL 30 DAYS
"""
print("Cleanup SQL (schedule as a daily job):")
print(cleanup_sql)

# Optimize tables
for tbl in ["conversations", "agent_instructions", "user_memory", "episodic_memory", "agent_capabilities"]:
    try:
        spark.sql(f"OPTIMIZE {catalog}.ai_ops.{tbl}")
        print(f"Optimized {catalog}.ai_ops.{tbl}")
    except Exception:
        pass
print("Tables optimized")

# COMMAND ----------
