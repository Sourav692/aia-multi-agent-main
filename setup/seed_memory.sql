-- =============================================================================
-- Seed Data: user_memory & episodic_memory
-- Catalog: aia_multi_agent_catalog | Schema: ai_ops
-- Idempotent via MERGE — safe to re-run.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Sample user_memory entries (long-term preferences & facts)
-- ---------------------------------------------------------------------------

MERGE INTO aia_multi_agent_catalog.ai_ops.user_memory AS target
USING (
  VALUES
    -- User 1: an analyst who focuses on claims in Central region
    ('analyst-001', 'preferred_region',   'Central',                   'preference', 1.0,  NULL),
    ('analyst-001', 'preferred_domain',   'claims',                    'preference', 1.0,  NULL),
    ('analyst-001', 'display_currency',   'USD',                       'preference', 1.0,  NULL),
    ('analyst-001', 'default_time_range', 'last_12_months',            'preference', 0.9,  NULL),
    ('analyst-001', 'role',               'Claims Analyst',            'fact',       1.0,  NULL),
    ('analyst-001', 'expertise_level',    'advanced',                  'fact',       0.8,  NULL),

    -- User 2: a manager who looks at distribution across all regions
    ('manager-002', 'preferred_domain',   'distribution',              'preference', 1.0,  NULL),
    ('manager-002', 'preferred_view',     'dashboard',                 'preference', 0.95, NULL),
    ('manager-002', 'default_time_range', 'current_quarter',           'preference', 1.0,  NULL),
    ('manager-002', 'role',               'Regional Sales Manager',    'fact',       1.0,  NULL),
    ('manager-002', 'team_size',          '12',                        'fact',       1.0,  NULL),
    ('manager-002', 'feedback_viz_style', 'Prefers bar charts over pie charts', 'feedback', 0.85, NULL),

    -- User 3: an executive who wants high-level summaries
    ('exec-003',    'preferred_view',     'summary',                   'preference', 1.0,  NULL),
    ('exec-003',    'preferred_domain',   'policies',                  'preference', 0.9,  NULL),
    ('exec-003',    'response_length',    'concise',                   'preference', 1.0,  NULL),
    ('exec-003',    'role',               'Chief Underwriting Officer','fact',       1.0,  NULL),
    ('exec-003',    'feedback_tone',      'Prefers formal business language', 'feedback', 0.9, NULL)

) AS source (user_id, memory_key, memory_value, memory_type, confidence, expires_at)
ON target.user_id = source.user_id AND target.memory_key = source.memory_key
WHEN MATCHED THEN UPDATE SET
  target.memory_value = source.memory_value,
  target.memory_type  = source.memory_type,
  target.confidence   = source.confidence,
  target.updated_at   = current_timestamp(),
  target.expires_at   = source.expires_at
WHEN NOT MATCHED THEN INSERT (
  user_id, memory_key, memory_value, memory_type, confidence, expires_at
) VALUES (
  source.user_id, source.memory_key, source.memory_value,
  source.memory_type, source.confidence, source.expires_at
);


-- ---------------------------------------------------------------------------
-- 2. Sample episodic_memory entries (notable past interactions)
-- ---------------------------------------------------------------------------

MERGE INTO aia_multi_agent_catalog.ai_ops.episodic_memory AS target
USING (
  VALUES
    -- Episode 1: successful simple KPI query
    (
      'ep-seed-001',
      'thread-demo-001',
      'analyst-001',
      'What is the total claims amount by region for Q4 2025?',
      'simple_kpi',
      'claims',
      ARRAY('genie'),
      'success',
      5,
      'Genie handled simple regional KPI accurately with no clarification needed.'
    ),

    -- Episode 2: complex multi-agent analysis
    (
      'ep-seed-002',
      'thread-demo-002',
      'analyst-001',
      'Are there any anomalies in claims processing times for health products this year?',
      'anomaly_detection',
      'claims',
      ARRAY('genie', 'analysis'),
      'success',
      4,
      'Genie provided raw data; Analysis agent detected 2 anomalous months. User found the z-score explanation helpful.'
    ),

    -- Episode 3: document lookup
    (
      'ep-seed-003',
      'thread-demo-003',
      'manager-002',
      'What are the exclusion clauses for the Premier Health Shield product?',
      'document_lookup',
      'products',
      ARRAY('multi_tool'),
      'success',
      5,
      'RAG retrieval returned the correct policy document section. No SQL needed.'
    ),

    -- Episode 4: visualization request
    (
      'ep-seed-004',
      'thread-demo-004',
      'manager-002',
      'Show me a dashboard of agent performance across all regions',
      'visualization_request',
      'distribution',
      ARRAY('visualization'),
      'success',
      4,
      'Existing endorsed dashboard was found and returned. User preferred bar charts — stored as preference.'
    ),

    -- Episode 5: partial success — needed clarification
    (
      'ep-seed-005',
      'thread-demo-005',
      'exec-003',
      'How are we doing on policies?',
      'simple_kpi',
      'policies',
      ARRAY('genie'),
      'partial',
      3,
      'Question was too vague — no region or time filter. Supervisor asked for clarification. After clarification, answer was good. Consider auto-applying user default filters.'
    ),

    -- Episode 6: failed interaction
    (
      'ep-seed-006',
      'thread-demo-006',
      'analyst-001',
      'Predict next quarter claims volume using ARIMA',
      'complex_analysis',
      'claims',
      ARRAY('analysis'),
      'failed',
      2,
      'Analysis agent does not yet support ARIMA forecasting. Returned a linear trend instead. User was unsatisfied. Forecasting capability should be added.'
    )

) AS source (episode_id, thread_id, user_id, question, intent, domain, agents_used, outcome, user_rating, lesson_learned)
ON target.episode_id = source.episode_id
WHEN MATCHED THEN UPDATE SET
  target.thread_id       = source.thread_id,
  target.user_id         = source.user_id,
  target.question        = source.question,
  target.intent          = source.intent,
  target.domain          = source.domain,
  target.agents_used     = source.agents_used,
  target.outcome         = source.outcome,
  target.user_rating     = source.user_rating,
  target.lesson_learned  = source.lesson_learned
WHEN NOT MATCHED THEN INSERT *;
