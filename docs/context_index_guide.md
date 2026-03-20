# Context Index — Detailed Usage Guide

The Context Index is the semantic asset discovery layer of the AIA Multi-Agent System. It allows the Supervisor Agent to translate a natural-language user question into a set of concrete data assets (Genie Spaces, metric views, tables, document indexes) that worker agents then operate on.

This document explains what the Context Index is, how it is built, how it is consumed at runtime, and walks through three end-to-end examples showing exactly how user questions map to discovered assets and downstream agent execution.

---

## Table of Contents

1. [What is the Context Index?](#1-what-is-the-context-index)
2. [Architecture and Components](#2-architecture-and-components)
3. [Schema Reference](#3-schema-reference)
4. [How the Context Index is Built](#4-how-the-context-index-is-built)
5. [How the Context Index is Used at Runtime](#5-how-the-context-index-is-used-at-runtime)
6. [Access Patterns](#6-access-patterns)
7. [Endorsed Asset Routing](#7-endorsed-asset-routing)
8. [Adding New Assets](#8-adding-new-assets)
9. [Example 1 — Simple KPI Question (Claims by Region)](#9-example-1--simple-kpi-question-claims-by-region)
10. [Example 2 — Document Lookup (Policy Coverage)](#10-example-2--document-lookup-policy-coverage)
11. [Example 3 — Genie Failure with Scoped Lookup Fallback](#11-example-3--genie-failure-with-scoped-lookup-fallback)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What is the Context Index?

The Context Index is a **governed registry of all data assets** available to the multi-agent system, combined with a **Vector Search index** that enables semantic matching between user questions and those assets.

Think of it as a "data catalog for the AI" — when a user asks a question, the Context Index answers: *"Which Genie Spaces, metric views, tables, and document indexes are most relevant to this question?"*

**Key design principle:** The Supervisor Agent owns the Context Index. It performs a single centralized lookup, then shares the results with worker agents via the shared state. Worker agents never independently discover their own domains — they operate within the scope the Supervisor established.

### What Problem Does It Solve?

Without the Context Index, the system would need hardcoded rules mapping question keywords to specific tables and tools. The Context Index replaces this with semantic matching:

| Approach | How Routing Works | Maintainability |
|----------|-------------------|-----------------|
| Hardcoded rules | `if "claims" in question → use claims_summary` | Brittle, requires code changes |
| **Context Index** | Embed question → find nearest assets by meaning | Add a row to a table to register new assets |

---

## 2. Architecture and Components

```
┌────────────────────────────────────────────────────────────────────┐
│                        Context Index System                        │
│                                                                    │
│  ┌─────────────────────┐       ┌──────────────────────────────┐   │
│  │  Delta Table         │       │  Vector Search Index          │   │
│  │  ai_ops.context_index│──────▶│  ai_ops.context_index_vs     │   │
│  │                      │ Delta │                               │   │
│  │  19 asset rows       │ Sync  │  Embedding model:             │   │
│  │  (manually curated)  │       │  databricks-bge-large-en      │   │
│  └─────────────────────┘       └──────────────┬───────────────┘   │
│                                                │                   │
│                              ┌─────────────────┴────────────────┐ │
│                              │  Vector Search Endpoint           │ │
│                              │  aia_context_index_vs             │ │
│                              └──────────────────────────────────┘ │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  UC Function (optional SQL access)                           │  │
│  │  ai_ops.context_index_search(query STRING) → TABLE           │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Type | Name | Purpose |
|-----------|------|------|---------|
| Source Table | Delta Table | `aia_multi_agent_catalog.ai_ops.context_index` | Stores all asset metadata and semantic descriptions |
| Vector Index | Delta Sync VS Index | `aia_multi_agent_catalog.ai_ops.context_index_vs` | Enables semantic similarity search over the `text` column |
| VS Endpoint | Serverless Endpoint | `aia_context_index_vs` | Hosts the vector index for low-latency queries |
| Embedding Model | Foundation Model | `databricks-bge-large-en` | Converts text into vector embeddings (BGE Large EN v1.5) |
| UC Function | SQL Function | `ai_ops.context_index_search(query)` | SQL-accessible wrapper for the vector search |

---

## 3. Schema Reference

### Delta Table: `ai_ops.context_index`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `asset_type` | `STRING` | Category of the asset | `genie_space`, `metric_view`, `table`, `document_index`, `dashboard` |
| `asset_id` | `STRING` | Unique identifier (primary key for VS index) | `aia_multi_agent_catalog.gold.mv_claims_count` |
| `display_name` | `STRING` | Human-readable label | `Claims Count Metric View` |
| `text` | `STRING` | Rich semantic description (this column is embedded) | `"Total count of insurance claims by month, region, product category, claim type, and status..."` |
| `domain` | `STRING` | Business domain | `claims`, `policies`, `distribution`, `customers`, `documents` |
| `endorsement_level` | `STRING` | Governance status | `endorsed`, `standard`, `experimental` |
| `metadata` | `STRING` | JSON blob with asset-specific configuration | `{"type": "metric_view", "grain": "monthly", "dimensions": [...]}` |

### The `text` Column — Why It Matters

The `text` column is the most important field. It is the column that gets embedded by `databricks-bge-large-en` and searched against when the user asks a question. **The quality of your text descriptions directly determines the quality of asset discovery.**

Good descriptions should:
- State what the asset contains ("Total count of insurance claims")
- List the dimensions and measures available ("by month, region, product category")
- Describe when to use it ("Use for claims volume analysis and trend monitoring")
- Include domain-specific vocabulary users would naturally use

### Currently Indexed Assets (19 total)

| # | Asset Type | Display Name | Domain | Endorsement |
|---|-----------|--------------|--------|-------------|
| 1 | `genie_space` | Claims Analytics Space | claims | endorsed |
| 2 | `genie_space` | Policy & Underwriting Space | policies | endorsed |
| 3 | `genie_space` | Distribution & Channels Space | distribution | endorsed |
| 4 | `genie_space` | Customer Analytics Space | customers | endorsed |
| 5 | `metric_view` | Claims Count Metric View | claims | endorsed |
| 6 | `metric_view` | Claims Amount Metric View | claims | endorsed |
| 7 | `metric_view` | Fraud Summary Metric View | claims | endorsed |
| 8 | `metric_view` | Policy Premium Metric View | policies | endorsed |
| 9 | `metric_view` | Policy Mix Metric View | policies | endorsed |
| 10 | `metric_view` | Agent Productivity Metric View | distribution | endorsed |
| 11 | `metric_view` | Customer Segments Metric View | customers | endorsed |
| 12 | `table` | Claims Summary Table | claims | endorsed |
| 13 | `table` | Policy Performance Table | policies | endorsed |
| 14 | `table` | Agent Performance Table | distribution | endorsed |
| 15 | `table` | Fraud Analysis Table | claims | endorsed |
| 16 | `table` | Enriched Claims Table | claims | endorsed |
| 17 | `table` | Enriched Policies Table | policies | endorsed |
| 18 | `table` | Customer 360 Table | customers | endorsed |
| 19 | `document_index` | Policy Documents Index | documents | endorsed |

---

## 4. How the Context Index is Built

The Context Index is created by running `notebooks/03_create_context_index.py`. The process has four stages:

### Stage 1: Create the Delta Table

A list of asset `Row` objects is constructed in Python, each describing one asset with its type, ID, display name, semantic text, domain, endorsement level, and metadata JSON. The list is converted to a Spark DataFrame and written as a Delta table:

```python
context_df = spark.createDataFrame(assets)
context_df.write.mode("overwrite").saveAsTable(f"{catalog}.ai_ops.context_index")
```

### Stage 2: Enable Change Data Feed

Delta Sync Vector Search indexes require Change Data Feed (CDF) to be enabled on the source table:

```sql
ALTER TABLE aia_multi_agent_catalog.ai_ops.context_index
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
```

### Stage 3: Create the Vector Search Endpoint and Index

A Vector Search endpoint (`aia_context_index_vs`) is created via the Databricks REST API. Then a **Delta Sync index** is created on top of the source table:

```python
resp = requests.post(f"{HOST}/api/2.0/vector-search/indexes", headers=HEADERS, json={
    "name": f"{catalog}.ai_ops.context_index_vs",
    "endpoint_name": "aia_context_index_vs",
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
```

Key configuration:
- **Primary key:** `asset_id` — each asset has a unique identifier
- **Index type:** `DELTA_SYNC` — the index automatically syncs with the Delta table
- **Pipeline type:** `TRIGGERED` — sync happens on-demand (not continuous)
- **Embedding column:** `text` — the semantic description column is embedded
- **Embedding model:** `databricks-bge-large-en` (BGE Large EN v1.5)

### Stage 4: Create the UC SQL Function

A Unity Catalog function wraps the vector search for SQL-based access:

```sql
CREATE OR REPLACE FUNCTION aia_multi_agent_catalog.ai_ops.context_index_search(
    query STRING
)
RETURNS TABLE(
    asset_type STRING, asset_id STRING, display_name STRING,
    text STRING, domain STRING, endorsement_level STRING, score DOUBLE
)
RETURN
  SELECT asset_type, asset_id, display_name, text, domain,
         endorsement_level, search_score AS score
  FROM VECTOR_SEARCH(
    index => 'aia_multi_agent_catalog.ai_ops.context_index_vs',
    query => query,
    num_results => 10
  )
```

This allows ad-hoc queries like:
```sql
SELECT * FROM aia_multi_agent_catalog.ai_ops.context_index_search('fraud risk by region')
```

---

## 5. How the Context Index is Used at Runtime

The Context Index is queried during the **`resolve_assets_with_context_index`** node of the Supervisor Agent's LangGraph pipeline. Here is the step-by-step runtime flow:

### Step 1: Semantic Query

The user's question is sent directly as the `query_text` to the Vector Search index:

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

results = w.vector_search_indexes.query_index(
    index_name="aia_multi_agent_catalog.ai_ops.context_index_vs",
    columns=["asset_type", "asset_id", "display_name", "text", "domain", "endorsement_level"],
    query_text=user_question,
    num_results=10,
)
```

This returns the **top 10 assets** whose `text` descriptions are most semantically similar to the user's question.

### Step 2: Endorsed Asset Prioritization

Results are re-sorted so that `endorsed` assets appear first, with secondary sorting by similarity score:

```python
assets.sort(key=lambda a: (
    0 if a.get("endorsement_level") == "endorsed" else 1,
    -a.get("score", 0)
))
```

This means even if a `standard` asset has a slightly higher similarity score, an `endorsed` asset will be preferred — ensuring governed, curated data assets take precedence.

### Step 3: Domain Detection

The primary domain is determined by counting which domain appears most frequently among the **top 5** results:

```python
domain_counts = {}
for a in assets[:5]:
    d = a.get("domain", "unknown")
    domain_counts[d] = domain_counts.get(d, 0) + 1
primary_domain = max(domain_counts, key=domain_counts.get)
```

For example, if 4 of the top 5 assets belong to the `claims` domain and 1 belongs to `policies`, the primary domain is `claims`.

### Step 4: Asset Categorization

The results are grouped by asset type for downstream use:

```python
genie_spaces   = [a for a in assets if a["asset_type"] == "genie_space"]
metric_views   = [a for a in assets if a["asset_type"] == "metric_view"]
tables         = [a for a in assets if a["asset_type"] == "table"]
doc_indexes    = [a for a in assets if a["asset_type"] == "document_index"]
dashboards     = [a for a in assets if a["asset_type"] == "dashboard"]
```

### Step 5: State Population

The resolved assets are written to the shared state, which downstream nodes (routing, agent execution, composition) all read from:

```python
state["resolved_assets"] = {
    "domain": "claims",
    "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",  # best match (backward compat)
    "genie_spaces": [                                      # ranked list for multi-space routing
        {"space_id": "01f0d6ff...", "domain": "claims",
         "display_name": "Claims Analytics Space", "score": 0.87, "endorsement": "endorsed"},
        {"space_id": "abc123...", "domain": "policies",
         "display_name": "Policy & Underwriting Space", "score": 0.62, "endorsement": "endorsed"},
    ],
    "metric_views": ["gold.mv_claims_count", "gold.mv_claims_amount"],
    "tables": ["gold.claims_summary", "silver.enriched_claims"],
    "document_indexes": ["bronze.policy_documents"],
    "dashboards": [],
    "all_assets": [...],
    "endorsement_info": {"gold.mv_claims_count": "endorsed", ...},
}
```

The `genie_spaces` list is ordered by endorsement level (endorsed first) and semantic score. The Genie agent tries each space in order until one succeeds.

### Fallback Behavior

If the Vector Search index is unavailable (e.g., endpoint is offline, network error), the system falls back to **hardcoded default assets** based on the classified intent:

| Intent | Default Assets |
|--------|---------------|
| `simple_kpi` | Default `genie_space` + `doc_vs_index` |
| `document_lookup` | Default `genie_space` + `doc_vs_index` |
| `conversational` | Default `genie_space` + `doc_vs_index` |

A warning is added: *"Context Index not ready — using rule-based asset resolution"*

---

## 6. Access Patterns

The Context Index is accessed through two distinct patterns:

### Pattern 1: Supervisor Lookup (Global)

**Called by:** `resolve_assets_with_context_index()` in the Supervisor's LangGraph pipeline.

**Scope:** Unrestricted — searches all assets across all domains.

**Purpose:** Establish the domain, discover the primary Genie Space, and identify all relevant tables and indexes for the current question.

**Characteristics:**
- Returns top 10 results
- Results determine the primary domain for the entire interaction
- All matching `genie_space` assets are returned as a ranked list in `resolved_assets.genie_spaces`, enabling the Genie agent to try multiple spaces in priority order
- The first `genie_space` ID is also stored in `resolved_assets.genie_space` for backward compatibility with `route_by_intent()`
- The presence/absence of `document_indexes` influences routing to the Multi-Tool Agent

### Pattern 2: Worker Scoped Lookup (Domain-Restricted)

**Called by:** `_scoped_context_index_lookup()` from within worker agents (currently used by the Genie Agent).

**Scope:** Restricted to the domain already resolved by the Supervisor.

**Purpose:** Discover *additional* assets within the current domain when a worker agent needs enrichment context (e.g., when Genie fails and needs alternative metric views).

**Characteristics:**
- Returns top 5 results (default)
- Filters results to match the Supervisor's resolved domain (case-insensitive)
- Optionally filters by `asset_types` (e.g., only `["metric_view", "table"]`)
- Workers cannot change the domain or override the Supervisor's global selection

```python
extra = _scoped_context_index_lookup(
    query_text=question,
    domain="claims",
    asset_types=["metric_view", "genie_space", "table"],
    num_results=3,
)
```

### Pattern 3: SQL Function (Ad-Hoc / Debugging)

**Called by:** Any SQL client, notebook, or dashboard.

**Scope:** Unrestricted.

**Purpose:** Ad-hoc exploration, debugging, and validation of what the Context Index returns for a given query.

```sql
SELECT * FROM aia_multi_agent_catalog.ai_ops.context_index_search(
    'What is the average fraud score by region?'
)
ORDER BY score DESC
```

---

## 7. Endorsed Asset Routing

The `endorsement_level` field is central to the governance model. It controls which assets the system prefers when multiple options are available.

### Endorsement Levels

| Level | Meaning | Routing Behavior |
|-------|---------|------------------|
| `endorsed` | Curated, validated, and approved by data governance | **Preferred.** Sorted to the top of results regardless of similarity score. |
| `standard` | Available and functional but not officially curated | Used when no endorsed asset matches. |
| `experimental` | In development or testing | Lowest priority; only used when nothing else matches. |

### How It Influences the Pipeline

1. **Asset Resolution** — Endorsed assets float to the top of the sorted results. The first `genie_space` in this sorted list becomes the one used for routing.

2. **Routing Decision** — `route_by_intent()` checks `has_genie = bool(assets.get("genie_space"))`. If the top-ranked Genie Space is endorsed, it is used. If only a `standard` Genie Space is available, it is still used but ranked lower.

3. **Answer Composition** — The `endorsement_info` dictionary is available in the state, allowing future extensions to annotate answers with data provenance (e.g., "This answer is based on endorsed data sources").

4. **Asset Feedback** — When the Genie Agent fails, `_record_asset_feedback()` logs the failure. Over time, governance teams can review this feedback to endorse new assets or improve existing ones.

---

## 8. Adding New Assets

To register a new asset in the Context Index:

### Step 1: Insert a Row

Add a new row to the Delta table with a complete, high-quality semantic description:

```sql
INSERT INTO aia_multi_agent_catalog.ai_ops.context_index
VALUES (
    'metric_view',                                          -- asset_type
    'aia_multi_agent_catalog.gold.mv_renewal_rate',         -- asset_id
    'Policy Renewal Rate Metric View',                      -- display_name
    'Policy renewal rate by month, region, product category, and customer segment. Shows renewal percentage, lapsed policy count, and retention metrics. Use for retention analysis, churn prediction inputs, and customer lifecycle management.',
                                                            -- text (semantic description)
    'policies',                                             -- domain
    'endorsed',                                             -- endorsement_level
    '{"type": "metric_view", "grain": "monthly", "measures": ["renewal_rate_pct", "lapsed_count", "retained_count"], "dimensions": ["region", "product_category", "segment"]}'
                                                            -- metadata (JSON)
)
```

### Step 2: Trigger Index Sync

Since the Vector Search index uses `TRIGGERED` pipeline type, you need to trigger a sync after inserting new data:

```python
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.vector_search_indexes.sync_index(
    index_name="aia_multi_agent_catalog.ai_ops.context_index_vs"
)
```

Or via REST API:
```bash
curl -X POST "https://<workspace>/api/2.0/vector-search/indexes/aia_multi_agent_catalog.ai_ops.context_index_vs/sync" \
  -H "Authorization: Bearer <token>"
```

### Step 3: Validate

Query the Context Index to confirm the new asset is discoverable:

```sql
SELECT * FROM aia_multi_agent_catalog.ai_ops.context_index_search(
    'What is the policy renewal rate?'
)
```

The new metric view should appear near the top of the results.

### Tips for Writing Good `text` Descriptions

| Do | Don't |
|----|-------|
| Include measures and dimensions by name | Write vague descriptions like "Policy data" |
| Describe when to use it ("Use for...") | Assume the embedding model understands table schemas |
| Use business vocabulary users would naturally say | Use internal technical jargon only |
| Mention the grain (daily, monthly, per-agent) | Omit the aggregation level |
| List related concepts (e.g., "churn, retention, lapse") | Keep descriptions to a single sentence |

---

## 9. Example 1 — Simple KPI Question (Claims by Region)

> **User:** "What is the total number of claims by region?"

### Step-by-Step Walkthrough

#### 1. Intent Classification

The Supervisor's `classify_intent` node processes the question:

```json
{
  "intent": "simple_kpi",
  "confidence": 0.95,
  "missing_filters": []
}
```

High confidence, no missing filters → skip clarification, proceed directly to asset resolution.

#### 2. Context Index Query

The question `"What is the total number of claims by region?"` is sent to the Vector Search index. The embedding model converts it to a vector and finds the nearest neighbors.

**Top 5 results returned (sorted by endorsed-first, then score):**

| Rank | Asset Type | Display Name | Domain | Score | Endorsement |
|------|-----------|--------------|--------|-------|-------------|
| 1 | `metric_view` | Claims Count Metric View | claims | 0.87 | endorsed |
| 2 | `genie_space` | Claims Analytics Space | claims | 0.82 | endorsed |
| 3 | `table` | Claims Summary Table | claims | 0.79 | endorsed |
| 4 | `metric_view` | Claims Amount Metric View | claims | 0.71 | endorsed |
| 5 | `table` | Enriched Claims Table | claims | 0.65 | endorsed |

**Why these results?**
- The query mentions "claims" and "count by region" — the **Claims Count Metric View** matches closely because its `text` says: *"Total count of insurance claims by month, region, product category..."*
- The **Claims Analytics Space** description mentions *"claim counts, claim amounts, claim processing times, approval rates..."* — a strong domain match
- The **Claims Summary Table** is *"Monthly aggregated claims metrics by region..."* — strong match on "claims" and "region"

#### 3. Resolved Assets

```python
state["resolved_assets"] = {
    "domain": "claims",                           # 5/5 top results are "claims"
    "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",  # First genie_space found
    "metric_views": ["gold.mv_claims_count", "gold.mv_claims_amount"],
    "tables": ["gold.claims_summary", "silver.enriched_claims"],
    "document_indexes": [],
    "dashboards": [],
    "all_assets": [...],
    "endorsement_info": {
        "gold.mv_claims_count": "endorsed",
        "01f0d6ff25da1f229950bb97c1ec974c": "endorsed",
        ...
    },
}
```

#### 4. Routing Decision

`route_by_intent()` evaluates:
- Intent is `simple_kpi`
- `has_genie` is `True` (Genie Space was resolved)
- Decision: **route to `genie`**

#### 5. Agent Execution

The Genie Agent calls the Genie Space API with the question. Genie generates SQL:

```sql
SELECT region, COUNT(*) AS total_claims
FROM aia_multi_agent_catalog.gold.claims_summary
GROUP BY region
ORDER BY total_claims DESC
```

Returns result summary: *"Central: 1,247 | North: 1,103 | South: 998 | East: 876 | West: 776"*

#### 6. Answer Composition

The Supervisor composes a natural-language answer using the Genie results:

> "The Central region leads with **1,247 claims**, followed by North at **1,103** and South at **998**. East and West round out the distribution with 876 and 776 claims respectively. In total, there are approximately **5,000 claims** across all five regions."

#### Flow Diagram

```
"What is the total number of claims by region?"
    │
    ▼
classify_intent → simple_kpi (95%)
    │
    ▼
resolve_assets_with_context_index
    │  Query: "What is the total number of claims by region?"
    │  Top match: Claims Count Metric View (score: 0.87)
    │  Domain: claims (5/5 top results)
    │  Genie Space: found ✓
    │
    ▼
route_by_intent → genie (simple_kpi + has_genie)
    │
    ▼
route_to_genie → SQL generated, results returned
    │
    ▼
compose_answer → "The Central region leads with 1,247 claims..."
```

---

## 10. Example 2 — Document Lookup (Policy Coverage)

> **User:** "What does the AIA Health Premium Plan cover?"

### Step-by-Step Walkthrough

#### 1. Intent Classification

```json
{
  "intent": "document_lookup",
  "confidence": 0.92,
  "missing_filters": []
}
```

The question asks about policy terms/coverage — clearly a document lookup.

#### 2. Context Index Query

The question `"What does the AIA Health Premium Plan cover?"` is sent to the Vector Search index.

**Top 5 results returned:**

| Rank | Asset Type | Display Name | Domain | Score | Endorsement |
|------|-----------|--------------|--------|-------|-------------|
| 1 | `document_index` | Policy Documents Index | documents | 0.88 | endorsed |
| 2 | `metric_view` | Policy Premium Metric View | policies | 0.62 | endorsed |
| 3 | `metric_view` | Policy Mix Metric View | policies | 0.55 | endorsed |
| 4 | `table` | Policy Performance Table | policies | 0.48 | endorsed |
| 5 | `table` | Enriched Policies Table | policies | 0.44 | endorsed |

**Why these results?**
- The **Policy Documents Index** matches strongly because its `text` says: *"Collection of insurance policy documents including policy wordings, product disclosure sheets, benefit schedules, exclusion lists..."* — the words "policy", "coverage", "benefits" align closely with the user's question about what a plan "covers"
- Policy-related metric views and tables appear because they mention "policy" and "premium" but with lower scores since they deal with structured data, not coverage documentation

#### 3. Resolved Assets

```python
state["resolved_assets"] = {
    "domain": "documents",                 # document_index has domain "documents"
    "genie_space": None,                   # No Genie Space in top results for this domain
    "metric_views": ["gold.mv_policy_premium", "gold.mv_policy_mix"],
    "tables": ["gold.policy_performance", "silver.enriched_policies"],
    "document_indexes": ["bronze.policy_documents"],
    "dashboards": [],
    ...
}
```

Note: The domain is `"documents"` (from the top-ranked document_index asset), but the secondary assets are from `"policies"`. The primary domain is determined by majority vote of the top 5 — if `documents` appears only once and `policies` appears 4 times, `policies` would win. However, the **document_index** being present is what matters for routing.

#### 4. Routing Decision

`route_by_intent()` evaluates:
- Intent is `document_lookup`
- Decision: **route to `multi_tool`** (hardcoded rule: `document_lookup` always goes to Multi-Tool)

The presence of `document_indexes` in the resolved assets is not directly checked by the router for this intent, but the Multi-Tool agent uses them for RAG retrieval.

#### 5. Agent Execution

The Multi-Tool Agent performs a Vector Search query against the **Policy Documents index** (`bronze.policy_documents_vs`):

```python
vs_results = w.vector_search_indexes.query_index(
    index_name="aia_multi_agent_catalog.bronze.policy_documents_vs",
    columns=["document_id", "title", "content", "document_type", "category"],
    query_text="What does the AIA Health Premium Plan cover?",
    num_results=5,
)
```

Returns 5 document chunks with titles like:
- "AIA Health Premium Plan — Benefits Schedule"
- "AIA Health Premium Plan — Coverage Summary"
- "AIA Health Premium Plan — Exclusion List"

#### 6. Answer Composition

The Supervisor composes an answer grounded in the retrieved documents:

> "The **AIA Health Premium Plan** covers hospitalization including room and board charges, surgical procedures, and ICU stays. It also provides outpatient coverage for specialist consultations, diagnostic tests, and prescribed medications. Key benefits include..."

#### Flow Diagram

```
"What does the AIA Health Premium Plan cover?"
    │
    ▼
classify_intent → document_lookup (92%)
    │
    ▼
resolve_assets_with_context_index
    │  Query: "What does the AIA Health Premium Plan cover?"
    │  Top match: Policy Documents Index (score: 0.88)
    │  Domain: policies/documents
    │  Genie Space: not found ✗
    │  Document Index: found ✓
    │
    ▼
route_by_intent → multi_tool (document_lookup intent)
    │
    ▼
route_to_multi_tool → RAG retrieval over policy documents
    │  5 relevant document chunks retrieved
    │
    ▼
compose_answer → "The AIA Health Premium Plan covers hospitalization..."
```

---

## 11. Example 3 — Genie Failure with Scoped Lookup Fallback

> **User:** "What is the average claim settlement time by region for Q4?"

### Step-by-Step Walkthrough

This example demonstrates how the **scoped Context Index lookup** is used as a fallback when the Genie Agent fails to answer a `simple_kpi` question.

#### 1. Intent Classification

```json
{
  "intent": "simple_kpi",
  "confidence": 0.90,
  "missing_filters": []
}
```

The question is a straightforward KPI query about claim settlement times.

#### 2. Context Index Query

The question is sent to the Vector Search index.

**Top 5 results returned:**

| Rank | Asset Type | Display Name | Domain | Score | Endorsement |
|------|-----------|--------------|--------|-------|-------------|
| 1 | `genie_space` | Claims Analytics Space | claims | 0.82 | endorsed |
| 2 | `document_index` | Policy Documents Index | claims | 0.65 | endorsed |

**Why these results?**
- "Claim settlement time" and "region" match the **Genie Space** whose `text` includes claims-related KPIs
- The document index matches broadly on "claims" terms

#### 3. Resolved Assets

```python
state["resolved_assets"] = {
    "domain": "claims",
    "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",
    "doc_vs_index": "aia_multi_agent_catalog.ai_ops.policy_docs_vs",
    "document_indexes": ["aia_multi_agent_catalog.bronze.policy_documents"],
    "all_assets": [...],
    "endorsement_info": {...},
}
```

#### 4. Routing Decision

`route_by_intent()` evaluates:
- Intent is `simple_kpi`
- Genie Space is resolved
- Decision: **route to `genie`**

#### 5. Agent Execution — Genie with Fallback

**Genie Agent path:**
- Calls Genie Space API with the question
- Scenario: Genie returns `status: "failed"` — unable to generate valid SQL (e.g., the required column doesn't exist in the Genie Space's tables)

**When Genie fails, the scoped Context Index lookup kicks in:**

```python
genie_res = state.get("genie_results", {})
domain = state.get("resolved_assets", {}).get("domain", "claims")

if genie_res.get("status") != "success" or not genie_res.get("sql"):
    extra = _scoped_context_index_lookup(
        question, domain,
        asset_types=["genie_space"],
        num_results=3,
    )
```

This scoped lookup:
1. Queries the Context Index again with the user question
2. Filters results to **only** assets in the `claims` domain (the Supervisor's resolved domain)
3. Filters to only `genie_space` types
4. Returns up to 3 results as enrichment context

The enrichment results are attached to the Genie results:
```python
genie_res["ci_enrichment"] = [
    {"asset_id": "01f0d6ff25da1f229950bb97c1ec974c", "display_name": "Claims Analytics Space", "asset_type": "genie_space"},
]
```

Additionally, asset feedback is recorded for governance:
```python
_record_asset_feedback(
    "genie", "claims", "genie_query_failed",
    "Genie could not answer: What is the average claim settlement time by region for Q4?",
    state
)
```

#### 6. Answer Composition

The compose step receives:
- Genie results: `failed` with `ci_enrichment` metadata pointing to alternative assets
- No Multi-Tool results (Genie path only for `simple_kpi`)
- Episodic lessons from past `simple_kpi` + `claims` interactions

The LLM synthesizes a response acknowledging the limitation:

> "I wasn't able to retrieve the average claim settlement time by region for Q4 from the current data sources. This metric may require additional table configuration in the Genie Space. Please reach out to the data team to verify the availability of settlement time data."

#### Flow Diagram

```
"What is the average claim settlement time by region for Q4?"
    │
    ▼
classify_intent → simple_kpi (90%)
    │
    ▼
resolve_assets_with_context_index
    │  Query: "What is the average claim settlement time..."
    │  Top matches: Bajaj Demo Genie Space (0.82)
    │  Domain: claims
    │  Genie Space: found ✓
    │
    ▼
route_by_intent → genie (simple_kpi + has genie_space)
    │
    ▼
route_to_genie
    │
    │  Genie FAILS ✗
    │
    ▼
_scoped_context_index_lookup
    │  Scope: domain=claims, types=genie_space
    │  Returns: enrichment assets
    │
    ▼
_record_asset_feedback
    │  Type: genie_query_failed
    │
    ▼
compose_answer
    │
    ▼
    "I wasn't able to retrieve the average claim
     settlement time by region for Q4..."
```

### What the Governance Team Sees

After this interaction, the `ai_ops.asset_feedback` table contains a new row:

| agent_name | domain | feedback_type | details | user_question |
|------------|--------|---------------|---------|---------------|
| genie | claims | genie_query_failed | Genie could not answer: What is the average claim settlement time by region for Q4? | What is the average claim settlement time by region for Q4? |

This signals to the data governance team that:
- The Genie Space may need additional tables or columns for settlement time metrics
- A new dataset or view covering settlement time by region may need to be added to the Context Index

---

## 12. Troubleshooting

### "Context Index not ready — using rule-based asset resolution"

**Cause:** The Vector Search endpoint or index is offline, or the workspace client cannot authenticate.

**Fix:**
1. Verify the endpoint is online: `GET /api/2.0/vector-search/endpoints/aia_context_index_vs`
2. Verify the index exists and is synced: `GET /api/2.0/vector-search/indexes/aia_multi_agent_catalog.ai_ops.context_index_vs`
3. Ensure the Model Serving endpoint has proper credentials (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`)

### Wrong Domain Detected

**Cause:** The top 5 results are dominated by a domain that doesn't match the user's intent.

**Fix:**
1. Query the Context Index directly to see what it returns:
   ```sql
   SELECT * FROM aia_multi_agent_catalog.ai_ops.context_index_search('your question here')
   ```
2. Improve the `text` descriptions of the assets you expect to match
3. Add more specific assets with better semantic descriptions

### New Asset Not Appearing in Results

**Cause:** The Delta Sync index hasn't been triggered after inserting the new row.

**Fix:**
1. Trigger a sync:
   ```python
   w.vector_search_indexes.sync_index(
       index_name="aia_multi_agent_catalog.ai_ops.context_index_vs"
   )
   ```
2. Wait for the sync to complete (check index status)
3. Verify CDF is enabled on the source table:
   ```sql
   SHOW TBLPROPERTIES aia_multi_agent_catalog.ai_ops.context_index
   ```

### Genie Space Not Being Used

**Cause:** No `genie_space` asset in the Context Index results for the question, so `route_by_intent()` falls back to `multi_tool`.

**Fix:**
1. Verify a `genie_space` row exists in the Context Index table
2. Ensure its `text` description covers the vocabulary users are using
3. Check the `endorsement_level` — endorsed assets are prioritized
