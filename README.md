# AIA Agent 360 — Multi-Agent Insurance Intelligence System

**Built by Zulfikar Maulana, Resident Solutions Architect, Databricks**

A production-grade Multi-Agent System on Databricks that answers natural language questions about insurance data. The system orchestrates specialist agents — Genie, Multi-Tool, Data Analysis, and Visualization — through a Supervisor Agent built with LangGraph and the Databricks Agent Framework.

> **Live App:** [AIA Agent 360](https://aia-insurance-intelligence-7474656931931914.aws.databricksapps.com)

---

## Architecture

```
                         ┌──────────────────────────┐
                         │      Databricks App       │
                         │   (Dash Chat UI - AIA     │
                         │      Agent 360)           │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      AI Gateway           │
                         │  (guardrails, rate limits, │
                         │   PII filtering, telemetry)│
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │       Model Serving Endpoint       │
                    │      (aia-supervisor-agent)        │
                    └─────────────────┬─────────────────┘
                                      │
     ┌────────────────────────────────┴────────────────────────────────┐
     │                    Supervisor Agent (LangGraph)                  │
     │                                                                 │
     │  ┌───────────┐   ┌──────────────┐   ┌─────────────────────┐   │
     │  │ Classify   │──▶│  Clarify /   │──▶│  Resolve Assets     │   │
     │  │ Intent     │   │ Disambiguate │   │ (Context Index / VS)│   │
     │  └───────────┘   └──────────────┘   └──────────┬──────────┘   │
     │                                                 │              │
     │                    ┌────────────────────────────┴───┐          │
     │                    │         Route to Agents         │          │
     │                    └──┬──────┬──────┬──────┬────────┘          │
     │                       │      │      │      │                   │
     │              ┌────────┴┐ ┌───┴────┐ ┌┴─────┐ ┌┴───────────┐   │
     │              │ Genie   │ │ Multi- │ │ Data │ │Visualization│   │
     │              │ Agent   │ │ Tool   │ │Anal. │ │   Agent     │   │
     │              │(Txt2SQL)│ │(SQL+RAG│ │(Stats│ │ (Lakeview   │   │
     │              │         │ │)       │ │)     │ │  Dashboard) │   │
     │              └────┬────┘ └───┬────┘ └──┬───┘ └──────┬──────┘   │
     │                   │          │         │            │           │
     │                   └──────────┴────┬────┴────────────┘           │
     │                                   ▼                             │
     │                          ┌────────────────┐                     │
     │                          │ Compose Answer  │                     │
     │                          └────────────────┘                     │
     │                                                                 │
     │  Memory: Delta checkpoints (ai_ops.conversations)               │
     │  Tracing: MLflow spans on every node                            │
     │  Prompts: ai_ops.agent_instructions (base + overlay)            │
     └─────────────────────────────────────────────────────────────────┘
```

---

## Databricks Products Used

| Product | Purpose |
|---------|---------|
| **Unity Catalog** | Governance: `aia_multi_agent_catalog` with bronze/silver/gold/ai_ops schemas |
| **Agent Framework** | LangGraph + ResponsesAgent for supervisor + worker agents |
| **Model Serving** | Production endpoint with `custom_inputs`/`custom_outputs` |
| **Genie Spaces** | Text-to-SQL over curated claims/policy tables |
| **Vector Search** | Context Index (asset discovery) + Policy Documents (RAG) |
| **Metric Views** | Governed KPI definitions (claims, policies, agents, customers) |
| **MLflow Tracing** | End-to-end observability with `@mlflow.trace` on every node |
| **AI/BI Dashboards** | Lakeview dashboard creation via REST API |
| **Databricks Apps** | Managed Dash chat UI |
| **AI Gateway** | Rate limits, guardrails, PII filtering |
| **Foundation Model API** | LLaMA 3.3 70B for reasoning and composition |

---

## Agent Design

### Supervisor Agent (Orchestrator)
- Interprets user intent, discovers relevant assets via Context Index, routes to specialist agents, composes final answer
- **8 LangGraph nodes:** `classify_intent` → `clarify_or_disambiguate` → `resolve_assets` → `route_to_*` → `compose_answer`
- Supports `thread_id` for multi-turn conversations with Delta-based checkpoints
- Endorsed asset routing: assets with `endorsement_level = "endorsed"` are prioritized

### Genie Agent (BI Specialist)
- Calls Genie Space API for Text-to-SQL over curated tables
- Returns: SQL query, result summary, row count

### Multi-Tool Agent (Generalist)
- LLM-generated SQL over resolved Unity Catalog tables
- Vector Search RAG over policy document index
- Returns: query results, retrieved documents, data previews

### Data Analysis Agent (Statistical)
- Anomaly detection via z-score analysis
- Summary statistics and trend analysis
- Returns: key metrics, anomalies, narrative insights

### Visualization Agent (Dashboard Creator)
- Creates real AI/BI dashboards via Lakeview REST API
- Generates SQL queries for dashboard datasets
- Publishes dashboards and returns clickable links
- Falls back to LLM-suggested visualizations when needed

### Context Index (Asset Discovery)
- 16 indexed assets (Genie Spaces, Metric Views, Tables, Document Indexes)
- Vector Search for semantic matching of user questions to available assets
- Used by Supervisor to determine domain routing

---

## Data Pipeline

### Bronze Layer (Raw)
| Table | Records | Description |
|-------|---------|-------------|
| `products` | 16 | Insurance product catalog (Life, Health, Motor, etc.) |
| `agents` | 50 | Insurance agent/advisor profiles |
| `customers` | 2,000 | Customer demographics and segmentation |
| `policies` | 3,000 | Policy details with premium and coverage |
| `claims` | 5,000 | Insurance claims with fraud scores |
| `policy_documents` | 200 | Policy documents for RAG |

### Silver Layer (Enriched)
| Table | Description |
|-------|-------------|
| `enriched_claims` | Claims joined with customer, policy, and product data |
| `enriched_policies` | Policies with customer, product, and agent data |
| `customer_360` | Aggregated customer view with policy and claims metrics |

### Gold Layer (Analytics-Ready)
| Table | Description |
|-------|-------------|
| `claims_summary` | Monthly claims by region, product, type |
| `policy_performance` | Policy metrics by region, product, channel |
| `agent_performance` | Agent KPIs: premium sold, churn rate |
| `fraud_analysis` | Claims with elevated fraud risk scores |

### AI Ops Layer
| Table | Description |
|-------|-------------|
| `context_index` | Asset registry for semantic discovery |
| `conversations` | Short-term memory (Delta checkpoints) |
| `agent_instructions` | Prompt management (base + overlay prompts) |
| `agent_config` | Agent configuration and feature flags |

### Metric Views
| View | Description |
|------|-------------|
| `mv_claims_count` | Claim counts by month/region/product/type |
| `mv_claims_amount` | Claim amounts and processing times |
| `mv_fraud_summary` | Fraud scores and suspicious claim rates |
| `mv_policy_premium` | Premium by region/product/channel |
| `mv_policy_mix` | Premium distribution across products |
| `mv_agent_productivity` | Agent productivity and churn metrics |
| `mv_customer_segments` | Segment analysis with NPS and claims |

---

## Intent Classification

| Intent | Example Questions | Agent Route |
|--------|-------------------|-------------|
| `simple_kpi` | "Total claims by region?" | Genie Agent |
| `deep_analysis` | "Any anomalies in claims?" | Data Analysis Agent |
| `document_lookup` | "What does health plan cover?" | Multi-Tool Agent (RAG) |
| `multi_domain` | "Agent churn vs claim ratio?" | All agents (parallel) |
| `visualization` | "Create a dashboard for claims by country" | Visualization Agent |

---

## Example Flows

### Flow 1: Simple KPI Question
```
User: "What is the total number of claims by region?"

1. classify_intent → simple_kpi (95% confidence)
2. resolve_assets → domain: claims, tables: [claims_summary, enriched_claims]
3. route_to_genie → SQL: SELECT region, COUNT(*) FROM gold.claims_summary GROUP BY region
4. compose_answer → "Here are the claims by region: Central has 1,247 claims..."
```

### Flow 2: Document Lookup (RAG)
```
User: "What does the AIA Health Premium Plan cover?"

1. classify_intent → document_lookup (92% confidence)
2. resolve_assets → domain: documents, doc_index: policy_documents_vs
3. route_to_multi_tool → Vector Search RAG over policy documents
4. compose_answer → "The AIA Health Premium Plan covers hospitalization, surgical..."
```

### Flow 3: Deep Analysis
```
User: "Are there any anomalies in our claims data?"

1. classify_intent → deep_analysis (88% confidence)
2. resolve_assets → domain: claims, tables: [enriched_claims, claims_summary]
3. route_to_analysis → z-score anomaly detection on claims amounts
4. compose_answer → "I found 23 anomalous claims with z-scores above 2.0..."
```

### Flow 4: Visualization Request
```
User: "Create a dashboard for number of claims per country"

1. classify_intent → visualization (90% confidence)
2. resolve_assets → domain: claims, tables: [claims_summary]
3. route_to_visualization → generates SQL, creates Lakeview dashboard via API
4. compose_answer → "I've created a dashboard showing claims by country.
                     View it here: [Dashboard Link]"
```

### Flow 5: Multi-Domain Question
```
User: "Which agents have the highest churn rate vs claim ratio?"

1. classify_intent → multi_domain (85% confidence)
2. resolve_assets → domain: cross_domain, tables: [agent_performance, claims_summary]
3. run_all_agents → Phase 1: Genie + Multi-Tool (parallel)
                    Phase 2: Analysis (sequential)
                    Phase 3: Visualization (sequential)
4. compose_answer → synthesized answer from all agent results
```

### Flow 6: Ambiguous Question (Clarification)
```
User: "Show me the numbers"

1. classify_intent → simple_kpi (45% confidence, missing_filters: [domain, metric])
2. clarify_or_disambiguate → "Could you specify which numbers you'd like to see?
                               Claims, policies, agents, or customer data?"
```

---

## Project Structure

```
aia-multi-agent/
├── README.md                           # This file
├── docs/
│   └── architecture.md                 # Detailed architecture document
├── data/
│   ├── generate_insurance_data.py      # Synthetic data generator
│   ├── products.csv                    # 16 insurance products
│   ├── agents.csv                      # 50 agents
│   ├── customers.csv                   # 2,000 customers
│   ├── policies.csv                    # 3,000 policies
│   ├── claims.csv                      # 5,000 claims
│   └── policy_documents.csv           # 200 policy documents
├── notebooks/
│   ├── 01_setup_catalog_and_tables.py  # UC setup + Bronze/Silver/Gold tables
│   ├── 02_create_metric_views.py       # 7 metric views
│   ├── 03_create_context_index.py      # Vector Search + Context Index
│   ├── 04_setup_genie_space.py         # Genie Space for claims domain
│   ├── 05_evaluation.py               # MLflow Agent Evaluation
│   └── 06_setup_memory_and_prompts.py  # Memory tables + Prompt management
├── agents/
│   ├── agent_code.py                   # Standalone agent code (all nodes)
│   └── supervisor_agent.py             # MLflow model logging notebook
└── app/
    ├── app.py                          # Dash Chat UI (AIA Agent 360)
    ├── app.yaml                        # Databricks App config
    └── requirements.txt                # App dependencies
```

---

## Setup Guide

### Prerequisites
- Databricks workspace with Unity Catalog enabled
- Serverless SQL warehouse
- Vector Search endpoint
- Foundation Model API access (LLaMA 3.3 70B)

### Step-by-Step Deployment

**1. Generate and load data**
```bash
cd data/
python generate_insurance_data.py
```
Then run `notebooks/01_setup_catalog_and_tables.py` in Databricks to create the catalog, schemas, and load all tables.

**2. Create metric views**
Run `notebooks/02_create_metric_views.py` — creates 7 governed metric views.

**3. Create Context Index and Vector Search**
Run `notebooks/03_create_context_index.py` — creates the asset registry and VS index.

**4. Setup Genie Space**
Run `notebooks/04_setup_genie_space.py` — creates the Genie Space for claims domain.
> Note: You must manually add tables to the Genie Space via the Databricks UI.

**5. Setup memory and prompts**
Run `notebooks/06_setup_memory_and_prompts.py` — creates conversation checkpoints table and seeds default prompts.

**6. Register the agent model**
Run `agents/supervisor_agent.py` — logs the agent with MLflow and registers it in Unity Catalog.

**7. Deploy to Model Serving**
Create a serving endpoint named `aia-supervisor-agent` with:
```json
{
  "served_entities": [{
    "entity_name": "aia_multi_agent_catalog.ai_ops.supervisor_agent",
    "entity_version": "latest",
    "scale_to_zero_enabled": true
  }],
  "environment_vars": {
    "DATABRICKS_HOST": "https://<workspace-url>",
    "DATABRICKS_TOKEN": "{{secrets/<scope>/<key>}}"
  }
}
```

**8. Deploy the Chat App**
```bash
databricks apps deploy aia-insurance-intelligence \
  --source-code-path /Workspace/Users/<email>/aia-multi-agent/app \
  -p <profile>
```

**9. Run evaluation (optional)**
Run `notebooks/05_evaluation.py` — evaluates the agent with MLflow Agent Evaluation.

---

## P0/P1 Features

### P0 (High Impact — Implemented)
- **Short-term memory**: `thread_id` + Delta checkpoint table for multi-turn conversations
- **Visualization Agent**: Creates real AI/BI dashboards via Lakeview REST API
- **MLflow Tracing**: `@mlflow.trace` spans on every node with proper `SpanType`
- **Custom I/O**: `custom_inputs` (thread_id, user_id) and `custom_outputs` (dashboard_urls, nodes_executed)

### P1 (Production Readiness — Implemented)
- **Prompt management**: Dynamic prompts from `ai_ops.agent_instructions` with 5-minute cache
- **AI Gateway**: Guardrails, rate limiting, PII filtering on LLM endpoint
- **Endorsed asset routing**: Endorsed assets sorted first in Context Index results
- **Clarification node**: Low-confidence intents trigger clarification before routing

---

## Technologies

- **Languages:** Python, SQL
- **Frameworks:** LangGraph, LangChain, Dash, MLflow
- **AI Models:** Meta LLaMA 3.3 70B Instruct, BGE Large EN v1.5 (embeddings)
- **Infrastructure:** Databricks (Unity Catalog, Agent Framework, Model Serving, Genie, Vector Search, Apps, AI Gateway)

---

## Deployment Info

| Resource | Value |
|----------|-------|
| Workspace | `https://fevm-aia-multi-agent.cloud.databricks.com` |
| Catalog | `aia_multi_agent_catalog` |
| Serving Endpoint | `aia-supervisor-agent` |
| App | `aia-insurance-intelligence` |
| SQL Warehouse | `2bdc1389949a9253` |
| Genie Space | `01f0d6ff25da1f229950bb97c1ec974c` |
| VS Endpoint | `aia_context_index_vs` |
| Model | `aia_multi_agent_catalog.ai_ops.supervisor_agent` |

---

*Built for AIA Multi-Agent System PoC — Vibe-coded with Claude Code*
