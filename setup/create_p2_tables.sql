-- =============================================================================
-- P2 Tables: Long-term Memory, Episodic Memory & Agent Capabilities
-- Catalog: aia_multi_agent_catalog | Schema: ai_ops
-- All statements are idempotent (IF NOT EXISTS / MERGE).
-- Run via Databricks SQL warehouse or notebook %sql cell.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. user_memory — long-term memory (persists user preferences/facts across sessions)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aia_multi_agent_catalog.ai_ops.user_memory (
  user_id       STRING NOT NULL       COMMENT 'User or session owner identifier',
  memory_key    STRING NOT NULL       COMMENT 'Unique key for this memory item (e.g. preferred_region)',
  memory_value  STRING               COMMENT 'Stored value (free-text or JSON)',
  memory_type   STRING               COMMENT 'Category: preference | fact | feedback',
  confidence    DOUBLE               COMMENT 'Confidence score 0.0-1.0 for inferred memories',
  created_at    TIMESTAMP            COMMENT 'When this memory was first created',
  updated_at    TIMESTAMP            COMMENT 'When this memory was last updated',
  expires_at    TIMESTAMP            COMMENT 'Optional TTL — NULL means never expires',
  CONSTRAINT pk_user_memory PRIMARY KEY (user_id, memory_key)
)
USING DELTA
COMMENT 'Long-term user memory: preferences, facts, and feedback that persist across sessions';

-- ---------------------------------------------------------------------------
-- 2. episodic_memory — stores notable interactions for learning
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aia_multi_agent_catalog.ai_ops.episodic_memory (
  episode_id      STRING NOT NULL     COMMENT 'Unique episode identifier (UUID)',
  thread_id       STRING             COMMENT 'Conversation thread this episode belongs to',
  user_id         STRING             COMMENT 'User who triggered this episode',
  question        STRING             COMMENT 'Original user question',
  intent          STRING             COMMENT 'Classified intent (simple_kpi, complex_analysis, etc.)',
  domain          STRING             COMMENT 'Domain routed to (claims, policies, products, distribution)',
  agents_used     ARRAY<STRING>      COMMENT 'List of agents invoked (genie, multi_tool, analysis, visualization)',
  outcome         STRING             COMMENT 'Outcome: success | partial | failed',
  user_rating     INT                COMMENT 'User rating 1-5 (NULL if not provided)',
  lesson_learned  STRING             COMMENT 'Auto-generated lesson from this interaction',
  created_at      TIMESTAMP          COMMENT 'When this episode was recorded',
  CONSTRAINT pk_episodic PRIMARY KEY (episode_id)
)
USING DELTA
COMMENT 'Episodic memory: notable interactions logged for continuous learning and review';

-- ---------------------------------------------------------------------------
-- 3. agent_capabilities — tool registry for semantic tool discovery
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aia_multi_agent_catalog.ai_ops.agent_capabilities (
  capability_id     STRING NOT NULL   COMMENT 'Unique capability identifier',
  agent_name        STRING NOT NULL   COMMENT 'Agent that owns this capability',
  capability_name   STRING NOT NULL   COMMENT 'Human-readable capability name',
  description       STRING           COMMENT 'Detailed description for semantic matching',
  supported_intents ARRAY<STRING>    COMMENT 'Intents this capability can handle',
  supported_domains ARRAY<STRING>    COMMENT 'Domains this capability covers',
  input_schema      STRING           COMMENT 'JSON Schema describing expected inputs',
  output_schema     STRING           COMMENT 'JSON Schema describing outputs',
  is_active         BOOLEAN          COMMENT 'Whether this capability is currently available',
  priority          INT              COMMENT 'Routing priority (lower = higher priority)',
  created_at        TIMESTAMP        COMMENT 'When this capability was registered',
  CONSTRAINT pk_capability PRIMARY KEY (capability_id)
)
USING DELTA
COMMENT 'Agent capability registry: enables semantic tool discovery and intent-based routing';

-- ---------------------------------------------------------------------------
-- 4. Seed data for agent_capabilities (idempotent via MERGE)
-- ---------------------------------------------------------------------------

-- Genie Agent: text-to-sql
MERGE INTO aia_multi_agent_catalog.ai_ops.agent_capabilities AS target
USING (
  SELECT
    'cap-genie-text2sql' AS capability_id,
    'genie' AS agent_name,
    'text-to-sql' AS capability_name,
    'Translates natural-language questions into SQL via Databricks Genie Space. Best for straightforward KPI queries over curated metric views and endorsed tables.' AS description,
    ARRAY('simple_kpi', 'complex_analysis') AS supported_intents,
    ARRAY('claims', 'policies', 'products', 'distribution') AS supported_domains,
    '{"type":"object","properties":{"question":{"type":"string"},"genie_space_id":{"type":"string"}}}' AS input_schema,
    '{"type":"object","properties":{"sql":{"type":"string"},"result":{"type":"array"},"description":{"type":"string"}}}' AS output_schema,
    true AS is_active,
    10 AS priority
) AS source
ON target.capability_id = source.capability_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

-- Multi-Tool Agent: sql + rag
MERGE INTO aia_multi_agent_catalog.ai_ops.agent_capabilities AS target
USING (
  SELECT
    'cap-multitool-sql-rag' AS capability_id,
    'multi_tool' AS agent_name,
    'sql+rag' AS capability_name,
    'Executes SQL queries over Unity Catalog tables and performs Vector Search RAG over document indexes. Handles questions that need both structured data and unstructured document context.' AS description,
    ARRAY('document_lookup', 'multi_domain') AS supported_intents,
    ARRAY('claims', 'policies', 'products', 'distribution') AS supported_domains,
    '{"type":"object","properties":{"question":{"type":"string"},"sql_query":{"type":"string"},"search_query":{"type":"string"}}}' AS input_schema,
    '{"type":"object","properties":{"sql_result":{"type":"array"},"rag_result":{"type":"string"},"sources":{"type":"array"}}}' AS output_schema,
    true AS is_active,
    20 AS priority
) AS source
ON target.capability_id = source.capability_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

-- Analysis Agent: statistical analysis
MERGE INTO aia_multi_agent_catalog.ai_ops.agent_capabilities AS target
USING (
  SELECT
    'cap-analysis-stats' AS capability_id,
    'analysis' AS agent_name,
    'statistical-analysis' AS capability_name,
    'Performs statistical analysis including anomaly detection (z-scores), trend decomposition, correlation analysis, and significance testing on insurance data.' AS description,
    ARRAY('anomaly_detection', 'complex_analysis') AS supported_intents,
    ARRAY('claims', 'products') AS supported_domains,
    '{"type":"object","properties":{"question":{"type":"string"},"analysis_type":{"type":"string","enum":["anomaly","trend","correlation","forecast"]}}}' AS input_schema,
    '{"type":"object","properties":{"findings":{"type":"array"},"statistics":{"type":"object"},"visualizations":{"type":"array"}}}' AS output_schema,
    true AS is_active,
    30 AS priority
) AS source
ON target.capability_id = source.capability_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

-- Visualization Agent: dashboard creation
MERGE INTO aia_multi_agent_catalog.ai_ops.agent_capabilities AS target
USING (
  SELECT
    'cap-viz-dashboard' AS capability_id,
    'visualization' AS agent_name,
    'dashboard-creation' AS capability_name,
    'Creates or finds AI/BI dashboards to visualize data. Prefers reusing existing endorsed dashboards when available. Generates charts, trend lines, and comparative visuals.' AS description,
    ARRAY('visualization_request') AS supported_intents,
    ARRAY('claims', 'policies', 'products', 'distribution') AS supported_domains,
    '{"type":"object","properties":{"question":{"type":"string"},"chart_type":{"type":"string"},"data_source":{"type":"string"}}}' AS input_schema,
    '{"type":"object","properties":{"dashboard_url":{"type":"string"},"chart_spec":{"type":"object"},"description":{"type":"string"}}}' AS output_schema,
    true AS is_active,
    40 AS priority
) AS source
ON target.capability_id = source.capability_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;
