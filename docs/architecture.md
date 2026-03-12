# AIA Multi-Agent System Architecture Design

> Full architecture document: [Google Docs](https://docs.google.com/document/d/1wqxDIXD98QaVNu_VVi7XC29K9huTbdp8bCoglpy8MHs/edit?tab=t.0)

## Purpose and Scope

This document describes the target architecture for AIA's Multi-Agent System on Databricks, covering both the high-level architecture of the overall system and the low-level architecture of key components.

The design follows Databricks Mosaic AI/Agent Bricks architecture reference patterns for Agents, Vector Search, AI Gateway, and governance.

**Project goal:** Deliver a governed, production-grade Multi-Agent system on Databricks that can answer natural language questions and orchestrate multiple agents and tools.

## AIA Initial MVP Requirements

Create an initial MVP to demonstrate a multi-agent system in which a data agent can:
- Understand questions
- Identify the appropriate data assets
- Return accurate, explainable answers
- Retain session state and learning
- Capture telemetry, evaluation signals, and governance data for continuous improvement

### Key Layers from AIA Reference Architecture

| Layer | Components |
|-------|------------|
| **Front End** | Q&A UI, Review UI |
| **Agent** | Supervisor Agent, Discovery Agent, Genie/SQL Agent, Insight Agent |
| **MCP/Tool** | Managed MCP, Customer MCP |
| **Memory** | Cache, Checkpoint, User Preference, Lesson Learn |
| **Observability** | AI Gateway, Review App |
| **Data** | Vector Store, Genie Spaces, Tables/Metric Views, Dashboard, Serverless Sandbox |

### Regional Constraints

Some features (e.g., Agent Bricks Supervisor Agent) are not yet available in certain Azure regions (SEA, East Asia). This design uses only GA features (Agent Framework, Model Serving, Genie, Vector Search, Metric Views, MLflow Tracing).

## Target Architecture — High-Level Flow

```
1. User asks question via Databricks App Chat UI
2. App sends request to Supervisor via Model Serving / AI Gateway
3. Supervisor loads context and memory (LangGraph + checkpoints)
4. Supervisor calls Context Index tool for semantic asset discovery
5. Supervisor routes to specialist/worker agents based on intent
6. Specialist agents execute tools and return structured results
7. Supervisor composes the final answer
8. Response returned to Databricks App with custom_outputs
```

### Detailed Flow: Context Index Usage

The Supervisor owns the Context Index. Results are shared with worker agents via the Supervisor's state:

```python
# Node: resolve_assets_with_context_index (Supervisor)
state["resolved_assets"] = {
    "domain": "claims",
    "genie_space": "claims_space",
    "metric_views": ["claims.metric_view_claim_count"],
    "tables": ["claims.fact_claim", "claims.dim_claim"],
    "dashboards": ["ai_bi.claim_anomaly_dashboard"],
    "endorsement_info": {...}
}
```

**Design choice:** Context Index at Supervisor layer so that:
- Asset selection and routing (including endorsed vs unendorsed) is performed once, centrally
- All worker agents see a consistent set of assets for a given question
- Clear responsibility separation: Supervisor resolves, Workers execute

## Component Architecture

### Databricks Apps and Chat UI

- Hosts the agent application (Python/Dash) wrapping the Supervisor Agent
- Exposes chat UI for natural language questions
- Handles user authentication & authorization
- Passes metadata (`user_id`, `thread_id`, `domain`) via `custom_inputs`

**Request schema:**
```json
{
  "input": [{"role": "user", "content": "..."}],
  "custom_inputs": {"thread_id": "...", "domain": "..."}
}
```

### Models and AI Gateway

- Serves agents via Databricks-hosted foundation models or external models
- AI Gateway provides: rate limiting, request/response filtering (PII), telemetry
- Model/agent configuration lives in MLflow and Unity Catalog

### Supervisor Agent (Planning, Routing, Composition)

**Role:** System's brain — interprets intent, selects assets and specialists, composes answers, maintains memory.

**Does NOT directly execute:** SQL, Vector Search retrieval, or analysis. Delegates to specialist agents.

**Responsibilities:**
1. Interpret user question and context (user, domain, thread)
2. Call Context Index for semantic asset discovery
3. Decide which specialists to call and with which assets
4. Compose final answer from specialist results
5. Maintain conversation state via checkpoints

**LangGraph StateGraph nodes:**
1. `classify_intent` — Categorize question intent with confidence score
2. `clarify_or_disambiguate` — Ask for clarification when confidence < 60%
3. `resolve_assets_with_context_index` — Discover relevant assets via Vector Search
4. `route_to_genie` — Delegate to Genie Agent for Text-to-SQL
5. `route_to_multi_tool` — Delegate to Multi-Tool Agent for SQL + RAG
6. `route_to_analysis` — Delegate to Data Analysis Agent for statistics
7. `route_to_visualization` — Delegate to Visualization Agent for dashboards
8. `compose_answer` — Synthesize final answer from all results

### Worker Agents

| Agent | Role | Tools | Returns |
|-------|------|-------|---------|
| **Genie Agent** | BI specialist for structured data | Genie Space API (Text-to-SQL) | SQL, result sets, summaries |
| **Multi-Tool Agent** | Generalist for SQL + RAG | UC tables (SQL), Vector Search (RAG) | Query results, retrieved docs |
| **Data Analysis Agent** | Statistical specialist | Z-score anomaly detection, summary stats | Key metrics, anomalies, narrative |
| **Visualization Agent** | Dashboard creator | Lakeview REST API | Dashboard URLs, metadata |

### Context Index (Tool)

- 16 indexed assets (Genie Spaces, Metric Views, Tables, Document Indexes)
- Vector Search for semantic matching
- Results include `endorsement_level` for prioritized routing
- Only the Supervisor calls Context Index; results shared via state

### Memory Architecture

**Short-term memory (P0):**
- Delta table: `ai_ops.conversations`
- Schema: `thread_id`, `checkpoint_id`, `state_json`, `created_at`
- Loaded at conversation start, saved after key nodes
- 30-day retention with scheduled cleanup

**Prompt management (P1):**
- Delta table: `ai_ops.agent_instructions`
- Schema: `agent_id`, `scope`, `base_prompt`, `overlay_prompt`
- 5-minute cache to avoid repeated reads
- Supports dynamic overlay prompts from feedback loops

### Observability

- **MLflow Tracing:** `@mlflow.trace` spans on every node with proper `SpanType`
- **AI Gateway:** Rate limiting, guardrails, PII filtering
- **Custom outputs:** `thread_id`, `checkpoint_id`, `nodes_executed`, `dashboard_urls`
- **UC Trace Tables:** (when enabled) MLflow writes OTEL-compatible trace tables

## Data Architecture

```
Unity Catalog: aia_multi_agent_catalog
├── bronze/                    # Raw data
│   ├── products, agents, customers, policies, claims
│   └── policy_documents, policy_documents_vs (Vector Search)
├── silver/                    # Enriched joins
│   ├── enriched_claims, enriched_policies
│   └── customer_360
├── gold/                      # Analytics-ready
│   ├── claims_summary, policy_performance
│   ├── agent_performance, fraud_analysis
│   └── mv_* (7 Metric Views)
└── ai_ops/                    # Agent infrastructure
    ├── context_index, context_index_vs
    ├── conversations (memory checkpoints)
    ├── agent_instructions (prompt management)
    └── agent_config
```

## Deployment Architecture

```
[User Browser]
      |
[Databricks App: aia-insurance-intelligence]
      |
[AI Gateway (rate limits, guardrails)]
      |
[Model Serving: aia-supervisor-agent]
      |
[LangGraph StateGraph]
      |
├── [Genie Space: 01f0d6ff25da1f229950bb97c1ec974c]
├── [Vector Search: aia_context_index_vs]
├── [SQL Warehouse: 2bdc1389949a9253]
├── [Foundation Model: databricks-meta-llama-3-3-70b-instruct]
└── [Lakeview Dashboard API]
```

## Response Flow

```
SupervisorResponsesAgent.predict()
  ├── Load checkpoint (if thread_id provided)
  ├── graph.invoke(state)
  │   ├── classify_intent
  │   ├── [clarify_or_disambiguate]  (if confidence < 60%)
  │   ├── resolve_assets_with_context_index
  │   ├── route_to_{genie|multi_tool|analysis|visualization}
  │   └── compose_answer
  ├── Save checkpoint
  └── Return ResponsesAgentResponse
        ├── output: [text_answer, metadata_json]
        └── custom_outputs: {intent, warnings, dashboard_urls, ...}
```
