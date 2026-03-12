# Memory Architecture — AIA Multi-Agent System

This document describes all memory systems used in the AIA Multi-Agent project. Memory is organized into three priority tiers — **P0** (short-term, per-session), **P1** (prompts and semantic context), and **P2** (long-term learning and personalization). All persistent memory is backed by **Databricks Delta tables** and **Vector Search indexes** in the `aia_multi_agent_catalog` catalog.

---

## Memory Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Memory Architecture                         │
├──────────┬──────────────────────────────────────────────────────────┤
│          │                                                          │
│   P0     │  Short-Term Memory (conversations)                      │
│          │  UI Session Memory (ui_sessions)                        │
│          │                                                          │
├──────────┼──────────────────────────────────────────────────────────┤
│          │                                                          │
│   P1     │  Prompt Management (agent_instructions)                 │
│          │  Context Index — Vector Search (context_index_vs)       │
│          │  Policy Documents — Vector Search (policy_documents_vs) │
│          │                                                          │
├──────────┼──────────────────────────────────────────────────────────┤
│          │                                                          │
│   P2     │  Long-Term User Memory (user_memory)                    │
│          │  Episodic Memory (episodic_memory)                      │
│          │  Agent Capabilities Registry (agent_capabilities)       │
│          │  Asset Feedback (asset_feedback)                        │
│          │                                                          │
├──────────┼──────────────────────────────────────────────────────────┤
│          │                                                          │
│ Runtime  │  In-Process Caches (_prompt_cache, _memory_cache,       │
│          │                      _capabilities_cache)               │
│          │                                                          │
└──────────┴──────────────────────────────────────────────────────────┘
```

---

## P0 — Short-Term Memory

### 1. Conversation Checkpoints

| Property | Value |
|---|---|
| **Table** | `ai_ops.conversations` |
| **Purpose** | Persist conversation state across turns in multi-turn chat sessions |
| **Lifetime** | Per-session; auto-cleaned after 30 days |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `thread_id` | `STRING NOT NULL` | Unique conversation thread ID |
| `checkpoint_id` | `STRING NOT NULL` | Unique checkpoint (MD5 hash of `thread_id:timestamp`) |
| `state_json` | `STRING` | Serialized LangGraph state (messages, intent, domain) |
| `created_at` | `TIMESTAMP` | When the checkpoint was saved |

**How it works:**

- **Save** — After each turn, `_save_checkpoint(thread_id, state_data)` serializes the full agent state (messages, detected intent, resolved domain) as JSON and inserts a new row.
- **Load** — When a request arrives with a `thread_id` and ≤ 1 message, `_load_checkpoint(thread_id)` retrieves the latest checkpoint by `created_at`, restoring prior conversation context so the agent can understand follow-up questions.
- **Cleanup** — A scheduled SQL job deletes checkpoints older than 30 days.

**Used by:** `SupervisorResponsesAgent.predict()` in `agents/agent_code.py`

---

### 2. UI Session Memory

| Property | Value |
|---|---|
| **Table** | `ai_ops.ui_sessions` |
| **Purpose** | Persist chat UI state across browser reloads |
| **Lifetime** | Indefinite (single `default` session) |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `session_id` | `STRING NOT NULL` | Session identifier (currently `'default'`) |
| `state_json` | `STRING` | Serialized conversation list (base64-encoded) |
| `updated_at` | `TIMESTAMP` | Last update timestamp |

**How it works:**

- The Dash/Plotly chat UI saves all conversation state to this table via a `MERGE` statement on every message exchange.
- On page load, `_load_ui_session()` fetches the latest state for `session_id = 'default'` and restores the chat sidebar and conversation history.
- The table is auto-created by `_ensure_ui_sessions_table()` if it doesn't exist.

**Used by:** `app/app.py`

---

## P1 — Prompt & Semantic Context

### 3. Prompt Management (Agent Instructions)

| Property | Value |
|---|---|
| **Table** | `ai_ops.agent_instructions` |
| **Purpose** | Versioned, table-driven prompt management for all agents |
| **Cache TTL** | 5 minutes |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `agent_id` | `STRING NOT NULL` | Agent identifier (`supervisor`, `genie`, `multi_tool`, `analysis`, `visualization`) |
| `scope` | `STRING NOT NULL` | Prompt scope (`classify_intent`, `compose_answer`, `default`, etc.) |
| `base_prompt` | `STRING` | Base system prompt (maintained in Git, versioned with code) |
| `overlay_prompt` | `STRING` | Dynamic overlay from feedback or an Instruction Builder job |
| `updated_at` | `TIMESTAMP` | Last update timestamp |
| `updated_by` | `STRING` | Who updated this prompt |

**How it works:**

- `_load_prompts()` reads all rows from `agent_instructions` and caches them in a dict keyed by `agent_id:scope`. The cache refreshes every 5 minutes.
- `_get_prompt(agent_id, scope, fallback)` looks up the merged `base_prompt + overlay_prompt` for a given agent and scope. If the table is unavailable, it falls back to a hardcoded prompt.
- The overlay mechanism allows post-deployment prompt tuning without redeploying code.

**Seeded prompts:** `supervisor:classify_intent`, `supervisor:compose_answer`, `genie:default`, `multi_tool:default`

**Used by:** `classify_intent()` and `compose_answer()` in `agents/agent_code.py`; setup in `notebooks/06_setup_memory_and_prompts.py`

---

### 4. Context Index (Vector Search for Asset Discovery)

| Property | Value |
|---|---|
| **Delta Table** | `ai_ops.context_index` |
| **Vector Index** | `ai_ops.context_index_vs` (Delta Sync) |
| **VS Endpoint** | `aia_context_index_vs` |
| **Purpose** | Semantic search over all discoverable data assets (Genie Spaces, Metric Views, Tables, Dashboards) |

**Schema (Delta table):**

| Column | Type | Description |
|---|---|---|
| `asset_type` | `STRING` | `genie_space`, `metric_view`, `table`, `dashboard`, `document_index` |
| `asset_id` | `STRING` | Unique asset identifier (e.g., Genie Space ID, fully-qualified table name) |
| `display_name` | `STRING` | Human-readable name |
| `text` | `STRING` | Semantic description of the asset (embedded for vector search) |
| `domain` | `STRING` | Business domain (`claims`, `policies`, `products`, `distribution`) |
| `endorsement_level` | `STRING` | Governance status (`endorsed`, `standard`, `experimental`) |
| `metadata` | `STRING` | JSON blob with asset-specific config (space IDs, measures, dimensions, etc.) |

**How it works:**

- The Supervisor's `resolve_assets_with_context_index` node queries the vector index with the user question to discover relevant Genie Spaces, metric views, and tables.
- Worker agents use `_scoped_context_index_lookup()` for additional asset discovery within the domain the Supervisor already resolved. Workers cannot change or override the Supervisor's domain selection.
- Endorsed assets are prioritized over non-endorsed assets in the ranking.

**Used by:** `resolve_assets_with_context_index()` and `_scoped_context_index_lookup()` in `agents/agent_code.py`; setup in `notebooks/03_create_context_index.py`

---

### 5. Policy Documents (Vector Search for RAG)

| Property | Value |
|---|---|
| **Vector Index** | `bronze.policy_documents_vs` |
| **Purpose** | Retrieval-Augmented Generation (RAG) over unstructured policy documents |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `document_id` | `STRING` | Unique document identifier |
| `title` | `STRING` | Document title |
| `content` | `STRING` | Document text content (chunked for embedding) |
| `document_type` | `STRING` | Type of document (e.g., policy, procedure, coverage) |
| `category` | `STRING` | Document category |

**How it works:**

- The Multi-Tool Agent's `route_to_multi_tool` node runs a vector similarity search with the user question against the policy document index.
- The top 5 most relevant document chunks are retrieved and included in the LLM's compose prompt as grounding context.
- Handles questions about coverage details, exclusion clauses, procedures, and policy terms.

**Used by:** `route_to_multi_tool()` in `agents/agent_code.py`

---

## P2 — Long-Term Learning & Personalization

### 6. Long-Term User Memory

| Property | Value |
|---|---|
| **Table** | `ai_ops.user_memory` |
| **Purpose** | Store user preferences and facts across sessions for personalization |
| **Cache TTL** | 60 seconds |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `user_id` | `STRING NOT NULL` | User or session owner identifier |
| `memory_key` | `STRING NOT NULL` | Unique key for the memory item (e.g., `preferred_region`) |
| `memory_value` | `STRING` | Stored value (free-text or JSON) |
| `memory_type` | `STRING` | Category: `preference`, `fact`, or `feedback` |
| `confidence` | `DOUBLE` | Confidence score 0.0–1.0 for inferred memories |
| `created_at` | `TIMESTAMP` | When the memory was first created |
| `updated_at` | `TIMESTAMP` | When the memory was last updated |
| `expires_at` | `TIMESTAMP` | Optional TTL (NULL = never expires) |

**Primary Key:** `(user_id, memory_key)`

**How it works:**

- **Load** — `_load_user_memory(user_id)` queries non-expired rows for a user, ordered by confidence. Results are cached for 60 seconds.
- **Save** — `_save_user_memory()` performs a `MERGE` upsert so existing keys are updated rather than duplicated.
- **Auto-Extract** — `_extract_and_save_user_facts(user_id, question, answer)` uses the LLM to detect explicit user preferences and facts from conversation exchanges and saves them automatically. Only explicitly stated facts are stored (no inference or guessing).
- Skipped for anonymous users.

**Memory types and examples:**

| Type | Example Key | Example Value |
|---|---|---|
| `preference` | `preferred_region` | `Central` |
| `preference` | `preferred_view` | `dashboard` |
| `preference` | `response_length` | `concise` |
| `fact` | `role` | `Claims Analyst` |
| `fact` | `expertise_level` | `advanced` |
| `feedback` | `feedback_viz_style` | `Prefers bar charts over pie charts` |

**Where it influences behavior:**

- **Intent classification** — User preferences are appended to the classification prompt so the LLM can consider default filters.
- **Answer composition** — The user's name, role, preferred view, response style, and other preferences shape the final response tone and format.

**Used by:** `classify_intent()` and `compose_answer()` in `agents/agent_code.py`; seeded in `setup/seed_memory.sql`

---

### 7. Episodic Memory

| Property | Value |
|---|---|
| **Table** | `ai_ops.episodic_memory` |
| **Purpose** | Log notable interactions for continuous learning and improvement |
| **Lifetime** | Indefinite |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `episode_id` | `STRING NOT NULL` | Unique episode identifier (UUID) |
| `thread_id` | `STRING` | Conversation thread this episode belongs to |
| `user_id` | `STRING` | User who triggered this episode |
| `question` | `STRING` | Original user question |
| `intent` | `STRING` | Classified intent (`simple_kpi`, `complex_analysis`, etc.) |
| `domain` | `STRING` | Domain routed to (`claims`, `policies`, `products`, `distribution`) |
| `agents_used` | `ARRAY<STRING>` | List of agents invoked (`genie`, `multi_tool`, `analysis`, `visualization`) |
| `outcome` | `STRING` | Result: `success`, `partial`, or `failed` |
| `user_rating` | `INT` | User rating 1–5 (NULL if not provided) |
| `lesson_learned` | `STRING` | Auto-generated lesson from this interaction |
| `created_at` | `TIMESTAMP` | When this episode was recorded |

**How it works:**

- **Save** — `_save_episodic_memory()` is called at the end of each completed interaction in `SupervisorResponsesAgent.predict()`, logging the question, intent, domain, which agents were invoked, and the outcome.
- **Retrieve** — `_get_episodic_lessons(intent, domain, limit=3)` fetches recent lessons for similar intent/domain combinations. These lessons are injected into the `compose_answer` prompt as "past experience" so the LLM can avoid repeating known mistakes.
- **Human feedback** — The Review App (`setup/create_review_app.py`) inserts human feedback (thumbs up/down + reviewer comments) into episodic memory via `log_review_feedback()`, which also logs to MLflow for tracking.

**Used by:** `SupervisorResponsesAgent.predict()` and `compose_answer()` in `agents/agent_code.py`; `log_review_feedback()` in `setup/create_review_app.py`; seeded in `setup/seed_memory.sql`

---

### 8. Agent Capabilities Registry

| Property | Value |
|---|---|
| **Table** | `ai_ops.agent_capabilities` |
| **Purpose** | Semantic tool discovery and intent-based routing |
| **Cache TTL** | 5 minutes |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `capability_id` | `STRING NOT NULL` | Unique capability identifier |
| `agent_name` | `STRING NOT NULL` | Agent that owns this capability |
| `capability_name` | `STRING NOT NULL` | Human-readable capability name |
| `description` | `STRING` | Detailed description for semantic matching |
| `supported_intents` | `ARRAY<STRING>` | Intents this capability can handle |
| `supported_domains` | `ARRAY<STRING>` | Domains this capability covers |
| `input_schema` | `STRING` | JSON Schema describing expected inputs |
| `output_schema` | `STRING` | JSON Schema describing outputs |
| `is_active` | `BOOLEAN` | Whether this capability is currently available |
| `priority` | `INT` | Routing priority (lower = higher priority) |
| `created_at` | `TIMESTAMP` | When this capability was registered |

**Registered capabilities:**

| Agent | Capability | Priority | Supported Intents |
|---|---|---|---|
| `genie` | `text-to-sql` | 10 | `simple_kpi`, `complex_analysis` |
| `multi_tool` | `sql+rag` | 20 | `document_lookup`, `multi_domain` |
| `analysis` | `statistical-analysis` | 30 | `anomaly_detection`, `complex_analysis` |
| `visualization` | `dashboard-creation` | 40 | `visualization_request` |

**How it works:**

- `_load_agent_capabilities()` loads all active capabilities ordered by priority and caches the result for 5 minutes.
- The Supervisor's `route_by_intent()` consults the registry for capability-based routing before falling back to hardcoded rules.
- New agents can be added to the system by inserting rows into this table — no code changes required.

**Used by:** `route_by_intent()` in `agents/agent_code.py`; seeded in `setup/create_p2_tables.sql`

---

### 9. Asset Feedback

| Property | Value |
|---|---|
| **Table** | `ai_ops.asset_feedback` |
| **Purpose** | Record gaps discovered by worker agents for governance and data asset improvement |
| **Lifetime** | Indefinite |

**Schema:**

| Column | Type | Description |
|---|---|---|
| `agent_name` | `STRING` | Which agent discovered the gap (`genie`, `multi_tool`) |
| `domain` | `STRING` | Domain context |
| `feedback_type` | `STRING` | Type of gap (e.g., `genie_query_failed`, `missing_metric_view`, `suggested_dashboard`) |
| `details` | `STRING` | Freeform description of the gap |
| `user_question` | `STRING` | The question that triggered the gap discovery |
| `user_id` | `STRING` | User who triggered the interaction |
| `created_at` | `TIMESTAMP` | When the feedback was recorded |

**How it works:**

- `_record_asset_feedback()` is called when a worker agent encounters a failure or discovers a missing asset. For example, when Genie cannot answer a question, a `genie_query_failed` feedback record is created.
- This data is consumed by governance processes to improve Genie Spaces, add missing metric views, and curate the data ontology over time.
- The table is created on-demand; if it doesn't exist, the insert silently fails.

**Used by:** `_record_asset_feedback()` and `route_to_genie()` in `agents/agent_code.py`

---

## Runtime — In-Process Caches

These are Python-level in-memory caches that reduce SQL round-trips during a single agent session. They are not persisted and reset when the process restarts.

| Cache Variable | TTL | Contents | Trigger for Invalidation |
|---|---|---|---|
| `_prompt_cache` | 300s (5 min) | Agent prompts keyed by `agent_id:scope` | TTL expiry |
| `_memory_cache` | 60s | User memory key-value pairs keyed by `mem:{user_id}` | TTL expiry or explicit reset after save |
| `_capabilities_cache` | 300s (5 min) | Active agent capabilities ordered by priority | TTL expiry |

All caches use a simple timestamp-based expiry: if `time.time() - cache_ts > TTL`, the cache is refreshed from the Delta table on the next read.

---

## Memory Lifecycle Summary

```
User sends a message
       │
       ▼
  ┌─ Load Checkpoint (P0) ─── restore prior conversation context
  │
  ├─ Load User Memory (P2) ─── apply personalization preferences
  │
  ├─ Load Prompts (P1) ─────── get table-driven prompt for classification
  │
  ├─ Load Capabilities (P2) ── consult tool registry for routing
  │
  ├─ Query Context Index (P1) ─ discover relevant data assets
  │
  ├─ Query Policy Docs (P1) ── RAG retrieval for document questions
  │
  ├─ Get Episodic Lessons (P2) ── learn from similar past interactions
  │
  ├─ Compose Answer ──────────── incorporate all memory into response
  │
  ├─ Save Checkpoint (P0) ──── persist conversation state
  │
  ├─ Save Episodic Memory (P2) ── log this interaction for learning
  │
  ├─ Extract User Facts (P2) ── auto-detect preferences from conversation
  │
  └─ Record Asset Feedback (P2) ── log gaps for governance
```

---

## Setup & Dependencies

| Memory Component | Setup Script | Dependencies |
|---|---|---|
| Conversations (P0) | `notebooks/06_setup_memory_and_prompts.py` | Delta table |
| Agent Instructions (P1) | `notebooks/06_setup_memory_and_prompts.py` | Delta table |
| Context Index (P1) | `notebooks/03_create_context_index.py` | Delta table + Vector Search |
| Policy Documents (P1) | Pre-existing bronze layer | Vector Search |
| User Memory (P2) | `setup/create_p2_tables.sql` | Delta table |
| Episodic Memory (P2) | `setup/create_p2_tables.sql` | Delta table |
| Agent Capabilities (P2) | `setup/create_p2_tables.sql` | Delta table |
| UI Sessions | `app/app.py` (auto-created) | Delta table |
| Seed Data | `setup/seed_memory.sql` | Existing tables |
| Review App | `setup/create_review_app.py` | MLflow + `episodic_memory` table |
