# Supervisor Agent — Detailed Functionality Guide

## Enhanced Framework with Multi-Genie-Space Routing

This document provides a comprehensive, detailed walkthrough of the AIA Supervisor Agent — its architecture, six-node execution pipeline, multi-Genie-Space routing strategy, three-tier memory system, semantic asset discovery via the Context Index, and governance mechanisms. It reflects the fully enhanced framework with P0, P1, and P2 capabilities.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Multi-Genie-Space Strategy](#3-multi-genie-space-strategy)
4. [Execution Pipeline: Six Nodes](#4-execution-pipeline-six-nodes)
   - [Node 1: Intent Classification](#node-1-intent-classification)
   - [Node 2: Clarification & Disambiguation](#node-2-clarification--disambiguation)
   - [Node 3: Asset Resolution via Context Index](#node-3-asset-resolution-via-context-index)
   - [Node 4: Genie Agent (Multi-Space)](#node-4-genie-agent-multi-space)
   - [Node 5: Multi-Tool Agent (RAG)](#node-5-multi-tool-agent-rag)
   - [Node 6: Answer Composition](#node-6-answer-composition)
5. [Routing Engine](#5-routing-engine)
6. [Three-Tier Memory System](#6-three-tier-memory-system)
7. [Context Index Deep Dive](#7-context-index-deep-dive)
8. [Governance & Feedback Loops](#8-governance--feedback-loops)
9. [Prompt Management](#9-prompt-management)
10. [State Schema Reference](#10-state-schema-reference)
11. [Observability & MLflow Tracing](#11-observability--mlflow-tracing)
12. [End-to-End Walkthroughs](#12-end-to-end-walkthroughs)
13. [Multi-Space Failure Handling](#13-multi-space-failure-handling)
14. [Configuration Reference](#14-configuration-reference)
15. [Extending the System](#15-extending-the-system)

---

## 1. System Overview

The Supervisor Agent is the central orchestrator of the AIA Multi-Agent Insurance Intelligence System. It does not directly query databases or retrieve documents. Instead, it coordinates a pipeline of specialized operations that collectively transform a natural-language question into a data-grounded, personalized answer.

### Core Responsibilities

| Responsibility | How It Works |
|---|---|
| **Interpret** | Classifies user intent into `simple_kpi`, `document_lookup`, or `conversational` with confidence scoring |
| **Discover** | Queries a semantic Context Index (Vector Search) to find relevant Genie Spaces, document indexes, and data assets |
| **Route** | Directs the question to the best-matching worker agent(s) based on intent, available assets, and a dynamic tool registry |
| **Orchestrate** | Manages multi-Genie-Space fallback — tries ranked spaces sequentially until one succeeds |
| **Compose** | Synthesizes agent results, episodic lessons, and user preferences into a natural-language answer |
| **Learn** | Persists conversation state, logs interactions to episodic memory, and extracts user facts for personalization |

### Technology Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` (6 nodes, 2 conditional edges) |
| Model Serving | Databricks Model Serving via `mlflow.pyfunc.log_model` |
| API Standard | MLflow `ResponsesAgent` (`ResponsesAgentRequest` / `ResponsesAgentResponse`) |
| LLM | `databricks-meta-llama-3-3-70b-instruct` (temperature 0.1) |
| Asset Discovery | Databricks Vector Search (BGE Large EN embeddings) |
| Structured Queries | Databricks Genie Space API (Text-to-SQL) |
| Document Retrieval | Databricks Vector Search RAG over policy documents |
| Memory | Delta tables via Databricks SDK Statement Execution API |
| Observability | MLflow Tracing with explicit span types per node |
| SQL Execution | Databricks SDK Statement Execution API (no Spark dependency in serving) |

**Source file:** `agents/agent_code.py`

---

## 2. Architecture

### High-Level Flow

```
User Question (Chat UI / API)
        |
        v
Model Serving Endpoint (aia-supervisor-agent)
        |
        v
SupervisorResponsesAgent.predict()
        |
        +-- Load Short-term Memory (checkpoint from ai_ops.conversations)
        +-- Load User Memory (preferences from ai_ops.user_memory)
        |
        v
+------------------------- LangGraph StateGraph -------------------------+
|                                                                         |
|   +-------------------+                                                 |
|   | classify_intent   |  Node 1: Intent + confidence + follow-up        |
|   +--------+----------+         resolution                              |
|            |                                                            |
|     +------+------+                                                     |
|     | confidence  |                                                     |
|     |  < 60% ?    |                                                     |
|     +--+-------+--+                                                     |
|     yes|       |no                                                      |
|        v       |                                                        |
|   +------------------------+                                            |
|   |clarify_or_disambiguate |  Node 2: Refine intent or ask for          |
|   +-----------+------------+         clarification                      |
|               |                                                         |
|               v                                                         |
|   +----------------------------------+                                  |
|   | resolve_assets_with_context_index|  Node 3: Semantic Vector Search  |
|   +---------------+------------------+         over Context Index       |
|                   |                                                     |
|            +------+------+                                              |
|            | route_by_   |  Routing decision based on intent,           |
|            |  intent()   |  assets, and tool registry                   |
|            +--+----+--+--+                                              |
|               |    |  |                                                 |
|     +---------+    |  +----------+                                      |
|     v              v             v                                      |
|  +------+   +-----------+  +---------------+                            |
|  |genie |   |multi_tool  |  |compose_answer |                           |
|  |      |   |            |  |(conversational)|                          |
|  +--+---+   +-----+------+  +------+--------+                          |
|     |             |                |                                    |
|     +-------------+----------------+                                    |
|                   |                                                     |
|                   v                                                     |
|          +-----------------+                                            |
|          | compose_answer  |  Node 6: Synthesize final answer            |
|          +-----------------+                                            |
|                   |                                                     |
+---------+---------+-----------------------------------------------------+
          |
          v
    Post-Processing:
      +-- Save Checkpoint (short-term memory)
      +-- Log Episodic Memory (interaction history)
      +-- Extract User Facts (for conversational messages)
      +-- Build custom_outputs metadata
          |
          v
    Return ResponsesAgentResponse
```

### Key Design Principles

1. **The Supervisor never touches data directly.** It classifies, discovers, routes, and composes — all data access is delegated to worker agents (Genie, Multi-Tool).

2. **The Context Index is the single source of truth** for which assets exist and how they map to user questions. Adding a new Genie Space or document index is a data operation, not a code change.

3. **Multi-Genie-Space routing is sequential with ranked fallback.** The Context Index returns a ranked list of Genie Spaces; the agent tries each in order until one succeeds.

4. **Domain scope is established once by the Supervisor** and inherited by all workers. Workers can discover additional assets within the resolved domain but cannot change it.

5. **Every interaction generates learning signals** — episodic memory, asset feedback, and user fact extraction create a continuous improvement loop.

---

## 3. Multi-Genie-Space Strategy

The enhanced framework replaces the single-Genie-Space approach with a multi-space architecture where four domain-specific Genie Spaces are registered in the Context Index, and the Supervisor semantically selects and prioritizes among them at runtime.

### Registered Genie Spaces

| Space | Domain | Tables | Typical Questions |
|---|---|---|---|
| **Claims Analytics** | `claims` | `claims_summary`, `fraud_analysis`, `enriched_claims` | Claim counts by region, fraud scores, approval rates, processing times |
| **Policy & Underwriting** | `policies` | `policy_performance`, `enriched_policies` | Premium volumes, renewal rates, lapse rates, product mix |
| **Distribution & Channels** | `distribution` | `agent_performance` | Agent productivity, channel contributions, partner metrics |
| **Customer Analytics** | `customers` | `customer_360` | Customer segments, retention rates, demographics, lifetime value |

### How Multi-Space Routing Works

```
User: "What is the total premium by distribution channel?"
                    |
                    v
         Context Index Vector Search
         (embeds question, finds nearest assets)
                    |
                    v
          Ranked Genie Space List:
          1. Policy & Underwriting Space  (score: 0.89, endorsed)
          2. Distribution & Channels Space (score: 0.76, endorsed)
          3. Claims Analytics Space        (score: 0.42, endorsed)
                    |
                    v
          Genie Agent tries Space #1 (Policy & Underwriting)
                    |
              +-----+-----+
              |           |
           Success     Failure
              |           |
              v           v
         Use result   Try Space #2 (Distribution & Channels)
                          |
                    +-----+-----+
                    |           |
                 Success     Failure
                    |           |
                    v           v
               Use result   Try Space #3, then CI enrichment lookup
```

### Multi-Space Selection Criteria

The ranked list is determined by two factors applied in order:

1. **Endorsement level** — `endorsed` spaces appear before `standard` or `experimental` spaces, regardless of semantic score.
2. **Semantic similarity score** — Within the same endorsement level, spaces with higher scores (more semantically relevant to the question) appear first.

This means a `standard` space with a 0.95 score still ranks below an `endorsed` space with a 0.72 score — governance takes precedence over raw similarity.

### Why Multiple Spaces?

| Concern | Single Space | Multi-Space |
|---|---|---|
| Domain coverage | One space must hold all tables — gets complex | Each space is scoped to its domain — simpler, more accurate |
| Genie SQL generation | Single prompt context with many tables — higher failure rate | Smaller, focused table sets — Genie generates better SQL |
| Governance | All-or-nothing endorsement | Per-domain endorsement and ownership |
| Failure handling | No fallback | Automatic sequential fallback to next-ranked space |
| Extensibility | Requires modifying a single monolithic space | Add a new space, register it in Context Index |

---

## 4. Execution Pipeline: Six Nodes

The Supervisor's LangGraph StateGraph contains six nodes connected by two conditional edges. Each node is an MLflow-traced function that reads from and writes to the shared `AgentState`.

### Node 1: Intent Classification

**Function:** `classify_intent(state)` | **MLflow Span:** `classify_intent` (`CHAIN`)

This is the entry point of the pipeline. It determines what kind of question the user is asking and whether additional clarification is needed.

#### Step 1a: Follow-Up Resolution

For multi-turn conversations, short follow-ups (10 words or fewer) are resolved into standalone questions using conversation history.

**Mechanism:** When the conversation has prior messages and the current question is short, the LLM is given the recent context (up to 6 prior messages) and asked to rewrite the follow-up as a complete question.

| Before | After |
|---|---|
| "What about by product?" (after "Total claims by region?") | "What is the total number of claims by product?" |
| "Show me Q4" (after "Premium trends by region") | "Show me the premium trends by region for Q4" |
| "And fraud?" (after "Claims count by region") | "What is the fraud analysis by region?" |

If resolution fails (LLM error), the original short question proceeds as-is and a warning is logged.

#### Step 1b: User Memory Enrichment

User preferences from `ai_ops.user_memory` (e.g., `preferred_region=Central`, `role=Claims Analyst`) are loaded and appended to the classification prompt. This allows the LLM to consider user defaults when classifying ambiguous questions.

#### Step 1c: Prompt Retrieval

The classification prompt is fetched from `ai_ops.agent_instructions` using key `supervisor:classify_intent`. If the table is unavailable, a hardcoded fallback prompt is used. Prompts are cached for 5 minutes.

#### Step 1d: LLM Classification

The LLM classifies the question into one of three categories:

| Intent | Description | Example |
|---|---|---|
| `simple_kpi` | KPI/metric questions — counts, totals, averages, trends by dimensions | "Total claims by region?" |
| `document_lookup` | Policy terms, coverage details, exclusions, procedures, document search | "What does the health plan cover?" |
| `conversational` | Greetings, introductions, personal statements, small talk | "Hi, I'm Sarah from the claims team" |

The LLM returns:
```json
{
  "intent": "simple_kpi",
  "confidence": 0.92,
  "missing_filters": []
}
```

#### Step 1e: Clarification Trigger

`needs_clarification` is set to `True` if:
- Confidence < 60%, OR
- `missing_filters` is non-empty (e.g., `["region", "time_period"]`)

#### State Updates After Node 1

| Field | Value |
|---|---|
| `intent` | `simple_kpi` / `document_lookup` / `conversational` |
| `intent_confidence` | Float 0.0 to 1.0 |
| `needs_clarification` | Boolean |
| `_missing_filters` | List of missing filter names (internal) |
| `user_question` | Original or rewritten (if follow-up resolved) |

---

### Node 2: Clarification & Disambiguation

**Function:** `clarify_or_disambiguate(state)` | **MLflow Span:** `clarify_or_disambiguate` (`CHAIN`)

**Trigger condition:** Only executed when `needs_clarification` is `True` (routed via `should_clarify` conditional edge).

#### What It Does

1. **Gathers context** from the last 4 conversation messages.

2. **Asks the LLM** to either:
   - **Resolve** the ambiguity using conversation history (e.g., inferring that "Show me the numbers" refers to claims data from prior context), OR
   - **Generate a clarification question** if history is insufficient (e.g., "Could you specify which numbers — claims, policies, agents, or customer data?")

3. **LLM responds** with:
   ```json
   {
     "resolved": true,
     "refined_intent": "simple_kpi",
     "refined_confidence": 0.85,
     "inferred_filters": {"domain": "claims"},
     "clarification_question": ""
   }
   ```

4. **State updates:**
   - If resolved: `intent` and `intent_confidence` are updated with refined values.
   - If not resolved: `clarification_message` is set with the generated question, and a warning is appended.
   - `needs_clarification` is **always set to `False`** after this node, so the pipeline continues regardless.

**Important:** The pipeline never stops to wait for user input. If a clarification question is generated, it is woven into the final composed answer as a suggestion, while the agent proceeds with its best-guess processing.

---

### Node 3: Asset Resolution via Context Index

**Function:** `resolve_assets_with_context_index(state)` | **MLflow Span:** `resolve_assets_with_context_index` (`RETRIEVER`)

This is the semantic discovery layer — it translates the user's question into a concrete set of data assets that worker agents will operate on.

#### Step 3a: Vector Search Query

The user's question is embedded and searched against the Context Index:

```python
results = w.vector_search_indexes.query_index(
    index_name="aia_multi_agent_catalog.ai_ops.context_index_vs",
    columns=["asset_type", "asset_id", "display_name", "text",
             "domain", "endorsement_level", "metadata"],
    query_text=question,
    num_results=10,
)
```

The `metadata` column is included so worker-specific configuration (e.g., the VS index name for document indexes) can be extracted during resolution.

#### Step 3b: Endorsement-Prioritized Sorting

Results are sorted with a two-key strategy:
1. **Primary:** `endorsed` assets first (sort key `0`), then `standard` (sort key `1`)
2. **Secondary:** Within the same endorsement level, by descending semantic score

This ensures governed, curated assets always take precedence over experimental ones, even if the experimental asset has a marginally higher similarity score.

#### Step 3c: Domain Detection

The primary domain is determined by majority vote among the top 5 results:

```
Top 5 assets: [claims, claims, claims, policies, claims]
    -> domain_counts: {claims: 4, policies: 1}
    -> primary_domain: "claims"
```

#### Step 3d: Multi-Space Extraction

All `genie_space` assets from the results are collected into a ranked list:

```python
genie_spaces = [
    {
        "space_id": "01f12199fed5107a9d2ccac293b2c0b6",
        "domain": "claims",
        "display_name": "Claims Analytics Space",
        "score": 0.87,
        "endorsement": "endorsed"
    },
    {
        "space_id": "01f12199ff0a119d989b057bc2a491c3",
        "domain": "policies",
        "display_name": "Policy & Underwriting Space",
        "score": 0.62,
        "endorsement": "endorsed"
    },
]
```

The first space ID is also stored as `genie_space` for backward compatibility with the routing function.

#### Step 3e: Document Index Resolution

For `document_index` assets, the `metadata` JSON is parsed to extract the `vs_index` field. This tells the Multi-Tool agent which Vector Search index to query for RAG retrieval:

```python
meta = json.loads(doc_indexes[0].get("metadata") or "{}")
doc_vs_index = meta.get("vs_index")
# e.g., "aia_multi_agent_catalog.ai_ops.policy_docs_vs"
```

#### Step 3f: State Update

```python
state["resolved_assets"] = {
    "domain": "claims",
    "genie_space": "01f12199fed5107a9d2ccac293b2c0b6",
    "genie_spaces": [
        {"space_id": "01f12199...", "domain": "claims",
         "display_name": "Claims Analytics Space",
         "score": 0.87, "endorsement": "endorsed"},
        {"space_id": "01f12199...", "domain": "policies",
         "display_name": "Policy & Underwriting Space",
         "score": 0.62, "endorsement": "endorsed"},
    ],
    "document_indexes": ["aia_multi_agent_catalog.bronze.policy_documents"],
    "doc_vs_index": "aia_multi_agent_catalog.ai_ops.policy_docs_vs",
    "all_assets": [...],
    "endorsement_info": {
        "01f12199fed5107a9d2ccac293b2c0b6": "endorsed",
        ...
    },
}
```

#### Fallback

If the Context Index is unavailable (endpoint offline, auth failure), a hardcoded default is used:

```python
{
    "domain": "claims",
    "genie_space": "01f12199fed5107a9d2ccac293b2c0b6",
    "genie_spaces": [
        {"space_id": "01f12199...", "domain": "claims",
         "display_name": "Claims Analytics Space", "score": 1.0,
         "endorsement": "endorsed"}
    ],
    "document_indexes": ["aia_multi_agent_catalog.bronze.policy_documents"],
    "doc_vs_index": "aia_multi_agent_catalog.ai_ops.policy_docs_vs",
}
```

A warning `"Context Index not ready — using rule-based asset resolution"` is appended.

---

### Node 4: Genie Agent (Multi-Space)

**Function:** `route_to_genie(state)` | **MLflow Span:** `route_to_genie` (`TOOL`)

The Genie Agent translates natural-language questions into SQL by calling the Databricks Genie Space API. In the enhanced framework, it receives a **ranked list of spaces** from the Context Index and tries them sequentially.

#### Step 4a: Space List Retrieval

The agent reads the ranked list from `state["resolved_assets"]["genie_spaces"]`. If the list is empty but a single `genie_space` ID exists (backward compatibility), it wraps it into a single-element list.

#### Step 4b: Sequential Space Execution

For each space in the ranked list:

1. **Start a Genie conversation** with the user's question:
   ```python
   conversation = w.genie.start_conversation(
       space_id=space_id, content=question
   )
   ```

2. **Poll for completion** (up to 30 iterations, 2 seconds apart = 60 seconds max per space):
   ```python
   for _ in range(30):
       msg = w.genie.get_message(
           space_id=space_id,
           conversation_id=conversation.conversation_id,
           message_id=conversation.message_id,
       )
       if msg.status.value in ["COMPLETED", "FAILED"]:
           break
       time.sleep(2)
   ```

3. **Extract results** — If the status is `COMPLETED`, extract:
   - The **SQL query** Genie generated (from `msg.attachments[].query.query`)
   - The **result text** (from `msg.attachments[].text.content`)
   - The **space name** that answered (for transparency in the composed answer)

4. **Success check** — If the space returned a successful result **with SQL**, the agent stops iterating (best match found). If the space succeeded but without SQL, or failed entirely, the agent tries the next space.

5. **Record the attempt** — Every attempt (success or failure) is logged with space ID, domain, display name, and result details.

#### Step 4c: Best Result Selection

After all attempts:
- First preference: a result with `status: "success"` AND a non-null `sql`
- Second preference: a result with `status: "success"` (even without SQL)
- Last resort: the final attempt (even if failed)

#### Step 4d: Enrichment on Failure

If all spaces fail (or none produced SQL), two fallback mechanisms activate:

1. **Scoped Context Index lookup** — Queries the Context Index again, scoped to the resolved domain, looking for additional `genie_space` assets:
   ```python
   extra = _scoped_context_index_lookup(
       question, domain,
       asset_types=["genie_space"], num_results=3,
   )
   ```
   These enrichment results are attached to `genie_results["ci_enrichment"]` for diagnostic visibility.

2. **Asset feedback recording** — Logs a `genie_query_failed` entry to `ai_ops.asset_feedback` listing all spaces tried:
   ```python
   _record_asset_feedback(
       "genie", domain, "genie_query_failed",
       f"Genie could not answer on [{failed_spaces}]: {question}",
       state
   )
   ```

#### State Update After Node 4

```python
state["genie_results"] = {
    "space_id": "01f12199fed5107a9d2ccac293b2c0b6",
    "domain": "claims",
    "display_name": "Claims Analytics Space",
    "sql": "SELECT region, COUNT(*) AS total_claims FROM ...",
    "result_summary": "Central: 1,247 | North: 1,103 | ...",
    "status": "success",
    "attempts": [
        {"space_id": "01f12199...", "status": "success", "sql": "SELECT...", ...},
    ],
}
```

---

### Node 5: Multi-Tool Agent (RAG)

**Function:** `route_to_multi_tool(state)` | **MLflow Span:** `route_to_multi_tool` (`TOOL`)

The Multi-Tool Agent performs Vector Search RAG over policy documents using the VS index dynamically resolved from the Context Index.

#### How It Works

1. **Reads the document VS index** from `state["resolved_assets"]["doc_vs_index"]`. This was extracted from the `document_index` asset's `metadata.vs_index` field during Node 3. If no VS index was resolved, the node returns early with an error and a warning.

2. **Queries the Vector Search index** with the user's question:
   ```python
   vs_results = w.vector_search_indexes.query_index(
       index_name=doc_vs_index,
       columns=["document_id", "title", "content",
                "document_type", "category"],
       query_text=question,
       num_results=5,
   )
   ```

3. **Returns up to 5 document chunks**, each containing:
   - `document_id`: Unique document identifier
   - `title`: Document title (e.g., "AIA Health Premium Plan — Benefits Schedule")
   - `content`: Text chunk content
   - `document_type`: Type (Policy Wording, FAQ, etc.)
   - `category`: Document category

#### State Update After Node 5

```python
state["multi_tool_results"] = {
    "docs": [
        {"document_id": "doc_001", "title": "AIA Health Premium Plan — Benefits Schedule",
         "content": "Hospitalization coverage includes...", "document_type": "Benefit Schedule",
         "category": "Health"},
        ...
    ],
    "status": "success",
}
```

---

### Node 6: Answer Composition

**Function:** `compose_answer(state)` | **MLflow Span:** `compose_answer` (`CHAIN`)

This final processing node synthesizes all agent results, memory context, and episodic lessons into a natural-language answer.

#### Context Assembly

The node gathers context from up to five sources:

| Source | Condition | What's Included |
|---|---|---|
| **Genie results** | Genie was invoked and succeeded | Space name, SQL query, result summary |
| **Multi-Tool results** | Multi-Tool was invoked | Top 3 document titles and content previews (200 chars) |
| **Episodic lessons** | Similar past interactions exist | Up to 3 lessons with outcome indicators from `ai_ops.episodic_memory` |
| **User preferences** | User has stored preferences | Name, role, preferred view, response style, region, expertise level |
| **Clarification** | A clarification question was generated in Node 2 | The clarification text for the LLM to weave in |

#### Composition Guidelines

The LLM is instructed to:
- Answer like a knowledgeable insurance colleague — direct, natural, helpful
- Lead with the key insight or answer, not a preamble
- Include specific numbers and data naturally in sentences
- Scale detail to complexity: brief for simple KPIs, thorough for complex analysis
- Use markdown (tables, bold) when it helps readability
- Address the user by name if known from preferences
- Never include a "Warnings & Limitations" section (handled by the UI separately)

---

## 5. Routing Engine

### The `route_by_intent` Function

After asset resolution, the Supervisor decides which worker agent to invoke. The routing uses a two-phase approach:

```
Phase 1: Tool Registry (ai_ops.agent_capabilities)
    |
    +-- Load active capabilities (cached 5 min)
    +-- Filter: supported_intents contains current intent
    +-- Sort: by priority (ascending, lower = higher priority)
    +-- If match found: route to matched agent
    |
    v
Phase 2: Hardcoded Fallback
    |
    +-- conversational   -> compose_answer (no agent needed)
    +-- document_lookup  -> multi_tool (RAG)
    +-- simple_kpi + has genie_space -> genie
    +-- simple_kpi + no genie_space  -> multi_tool
    +-- default -> multi_tool
```

### Tool Registry Entries

The `ai_ops.agent_capabilities` table contains registrations for each agent:

| Agent | Capability | Priority | Supported Intents | Supported Domains |
|---|---|---|---|---|
| `genie` | text-to-sql | 10 (highest) | `simple_kpi`, `complex_analysis` | claims, policies, products, distribution, customers |
| `multi_tool` | sql+rag | 20 | `document_lookup` | claims, policies, products, distribution |
| `analysis` | statistical-analysis | 30 | `anomaly_detection`, `complex_analysis` | claims, products |
| `visualization` | dashboard-creation | 40 (lowest) | `visualization_request` | claims, policies, products, distribution |

The registry enables adding new agents without code changes — insert a row into `ai_ops.agent_capabilities` with the new agent's supported intents and priority.

### The `should_clarify` Conditional Edge

Before routing to asset resolution, the pipeline checks whether clarification is needed:

```
classify_intent
      |
  confidence < 60%
  OR missing_filters?
      |
   +--+--+
   |     |
  yes    no
   |     |
   v     v
clarify  resolve_assets
```

---

## 6. Three-Tier Memory System

The Supervisor uses a layered memory architecture spanning three priority tiers:

### P0: Short-Term Memory (Conversation Checkpoints)

| Component | Table | Purpose | Cache TTL |
|---|---|---|---|
| Conversation checkpoints | `ai_ops.conversations` | Multi-turn conversation context | None (read on demand) |

**How it works:**
- After each interaction, the Supervisor serializes the conversation state (all messages, current intent, resolved domain) to JSON and saves it as a checkpoint.
- On the next interaction with the same `thread_id`, the checkpoint is loaded to restore prior context.
- Only loaded when the incoming request has a single message (no conversation history in the payload).
- Checkpoints are cleaned up after 30 days via a scheduled SQL job.

**Checkpoint data:**
```python
{
    "messages": [
        {"role": "user", "content": "Total claims by region?"},
        {"role": "assistant", "content": "The Central region leads with..."},
        {"role": "user", "content": "What about by product?"}
    ],
    "intent": "simple_kpi",
    "domain": "claims"
}
```

### P1: Prompt Management & Semantic Context

| Component | Table/Index | Purpose | Cache TTL |
|---|---|---|---|
| Prompt management | `ai_ops.agent_instructions` | Table-driven prompts with base + overlay support | 5 min |
| Context Index | `ai_ops.context_index_vs` | Semantic asset discovery via Vector Search | N/A |
| Policy Documents | Resolved from Context Index `metadata.vs_index` | RAG retrieval for document questions | N/A |

**Prompt structure:**
Each prompt has a `base_prompt` (maintained in Git, versioned with code) and an optional `overlay_prompt` (dynamic, from feedback/Instruction Builder job). They are concatenated at load time:
```
final_prompt = base_prompt + "\n" + overlay_prompt
```

Domain-specific prompts exist for each Genie scope (`genie:claims`, `genie:policies`, `genie:distribution`, `genie:customers`).

### P2: Long-Term Learning

| Component | Table | Purpose | Cache TTL |
|---|---|---|---|
| User Memory | `ai_ops.user_memory` | Personalization — name, role, preferences | 60 sec |
| Episodic Memory | `ai_ops.episodic_memory` | Interaction logs for continuous learning | N/A |
| Agent Capabilities | `ai_ops.agent_capabilities` | Semantic tool registry for routing | 5 min |
| Asset Feedback | `ai_ops.asset_feedback` | Gap discovery for governance improvement | N/A |

#### User Memory

Stores key-value pairs per user:
```
user_id: "sarah_claims"
  name: "Sarah"
  role: "Claims Analyst"
  preferred_region: "Central"
  preferred_view: "summary"
  response_length: "detailed"
```

Memories have optional expiry (`expires_at`) and confidence scores. They are written via MERGE (upsert) so repeated extractions update rather than duplicate.

**Extraction process:** For `conversational` messages (greetings, introductions), the LLM analyzes the exchange and extracts any explicitly stated personal facts. Only facts the user directly stated are saved — no inferences.

#### Episodic Memory

Every completed interaction is logged:
```python
{
    "episode_id": "a1b2c3d4e5f6g7h8i9j0",
    "thread_id": "thread_abc123",
    "user_id": "sarah_claims",
    "question": "What is the total number of claims by region?",
    "intent": "simple_kpi",
    "domain": "claims",
    "agents_used": ["genie"],
    "outcome": "success",
    "lesson_learned": null,  # populated later via feedback
    "user_rating": null,     # populated when user rates the answer
}
```

Past lessons (interactions where `lesson_learned` is populated) are retrieved during answer composition to improve future responses for similar questions.

---

## 7. Context Index Deep Dive

### What It Contains

The Context Index is a governed registry of all data assets available to the multi-agent system. It currently indexes 5 assets across 2 asset types:

| # | Asset Type | Display Name | Domain | Space ID / Asset ID |
|---|---|---|---|---|
| 1 | `genie_space` | Claims Analytics Space | claims | `01f12199fed5107a9d2ccac293b2c0b6` |
| 2 | `genie_space` | Policy & Underwriting Space | policies | `01f12199ff0a119d989b057bc2a491c3` |
| 3 | `genie_space` | Distribution & Channels Space | distribution | `01f12199ff2b1aef96fc954dc1de1a06` |
| 4 | `genie_space` | Customer Analytics Space | customers | `01f12199ff561a40817162d95a240597` |
| 5 | `document_index` | Policy Documents Index | documents | `aia_multi_agent_catalog.bronze.policy_documents` |

### How It Enables Multi-Space Routing

The `text` column of each Genie Space contains a rich semantic description of what the space covers. When a user asks a question, the Context Index embeds the question and finds the nearest asset descriptions. Because each space has a different focus, the semantic similarity scores naturally rank the most relevant space highest.

**Example — "What is the fraud score by region?":**

| Space | Score | Why |
|---|---|---|
| Claims Analytics | 0.88 | Description includes "fraud analysis, loss ratios, suspicious claims" |
| Customer Analytics | 0.41 | Description includes "claim frequency by segment" (tangential) |
| Policy & Underwriting | 0.32 | "premium volumes, renewal rates" — low relevance to fraud |
| Distribution & Channels | 0.28 | "agent productivity, sales pipeline" — lowest relevance |

The Genie Agent receives `[Claims, Customer, Policy, Distribution]` as its ranked list and tries Claims first, which succeeds.

### Two Access Patterns

| Pattern | Used By | Scope | Purpose |
|---|---|---|---|
| **Supervisor Lookup** (global) | `resolve_assets_with_context_index()` | Unrestricted, all domains | Establish domain, discover primary Genie Spaces, identify all relevant assets |
| **Worker Scoped Lookup** (domain-restricted) | `_scoped_context_index_lookup()` | Restricted to resolved domain | Discover additional assets within the current domain (used as fallback) |

Workers cannot change domains or override the Supervisor's global selection — they can only discover additional assets within the established scope.

### Context Index Infrastructure

| Component | Name | Purpose |
|---|---|---|
| Delta Table | `ai_ops.context_index` | Source table with asset metadata and semantic descriptions |
| Vector Search Index | `ai_ops.context_index_vs` | Delta Sync VS index over `text` column |
| VS Endpoint | `aia_context_index_vs` | Serverless endpoint hosting the vector index |
| Embedding Model | `databricks-bge-large-en` | BGE Large EN v1.5 for text embeddings |
| UC Function | `ai_ops.context_index_search(query)` | SQL-accessible wrapper for ad-hoc queries |

---

## 8. Governance & Feedback Loops

### Endorsement Levels

Every asset in the Context Index has an `endorsement_level` that controls routing priority:

| Level | Meaning | Routing Behavior |
|---|---|---|
| `endorsed` | Curated, validated, approved by data governance | Sorted to the top regardless of similarity score |
| `standard` | Available and functional but not officially curated | Used when no endorsed asset matches |
| `experimental` | In development or testing | Lowest priority; last resort |

### Asset Feedback Loop

When the Genie Agent fails to answer on all tried spaces, a feedback record is written:

```sql
INSERT INTO ai_ops.asset_feedback
(agent_name, domain, feedback_type, details, user_question, user_id, created_at)
VALUES ('genie', 'claims', 'genie_query_failed',
        'Genie could not answer on [Claims Analytics Space, Policy & Underwriting Space]: ...',
        'What is the average claim settlement time by region?',
        'user_123', current_timestamp())
```

This creates a continuous improvement signal:
1. **Governance teams** query `asset_feedback` to find common failure patterns
2. They improve Genie Spaces (add tables, update instructions) or add new assets to the Context Index
3. The next time a similar question is asked, the improved space succeeds

### Episodic Learning Loop

1. Every interaction is logged to `episodic_memory`
2. When a lesson is later extracted (via feedback or manual review), `lesson_learned` is populated
3. Future interactions with the same intent + domain retrieve these lessons during answer composition
4. The LLM uses lessons as additional context to improve response quality

---

## 9. Prompt Management

### Architecture

All prompts are stored in `ai_ops.agent_instructions` with a two-layer structure:

| Layer | Field | Source | Purpose |
|---|---|---|---|
| Base | `base_prompt` | Maintained in Git, seeded by notebook 06 | Stable, version-controlled core prompt |
| Overlay | `overlay_prompt` | Updated by feedback jobs or Instruction Builder | Dynamic refinements without code changes |

At load time: `effective_prompt = base_prompt + "\n" + overlay_prompt`

### Registered Prompts

| Agent | Scope | Purpose |
|---|---|---|
| `supervisor` | `classify_intent` | Intent classification instructions and category definitions |
| `supervisor` | `compose_answer` | Answer composition guidelines and formatting rules |
| `genie` | `default` | General Genie Space usage instructions |
| `genie` | `claims` | Claims-specific KPI focus areas |
| `genie` | `policies` | Policy & underwriting KPI focus areas |
| `genie` | `distribution` | Distribution & channel KPI focus areas |
| `genie` | `customers` | Customer analytics KPI focus areas |
| `multi_tool` | `default` | RAG retrieval and document answering instructions |

### Caching

Prompts are cached in-memory for 5 minutes (`_prompt_cache_ts`). After the TTL expires, the next call to `_get_prompt()` refreshes the cache from the database.

---

## 10. State Schema Reference

The `AgentState` TypedDict defines the shared state that flows through all nodes:

| Field | Type | Set By | Purpose |
|---|---|---|---|
| `messages` | `list` | Step 0 | Full conversation history (all prior messages + current) |
| `user_question` | `str` | Step 0/Node 1 | Current question (may be rewritten by follow-up resolution) |
| `intent` | `str` | Node 1/2 | Classified intent: `simple_kpi`, `document_lookup`, `conversational` |
| `intent_confidence` | `float` | Node 1/2 | Classification confidence (0.0 to 1.0) |
| `clarification_message` | `Optional[str]` | Node 2 | Generated clarification question (if ambiguity unresolved) |
| `needs_clarification` | `bool` | Node 1/2 | Controls routing to clarification node |
| `resolved_assets` | `Optional[dict]` | Node 3 | Discovered data assets (domain, genie_spaces, doc indexes) |
| `genie_results` | `Optional[dict]` | Node 4 | Genie agent output (SQL, results, attempts) |
| `multi_tool_results` | `Optional[dict]` | Node 5 | Multi-Tool agent output (documents, status) |
| `final_answer` | `Optional[str]` | Node 6 | Composed natural-language response |
| `warnings` | `list` | All nodes | Accumulated operational warnings |
| `thread_id` | `Optional[str]` | Step 0 | Conversation thread identifier |
| `user_id` | `Optional[str]` | Step 0 | User identifier for memory and personalization |

---

## 11. Observability & MLflow Tracing

Every node is instrumented with explicit MLflow Tracing spans:

| Node | Span Name | Span Type | Key Attributes |
|---|---|---|---|
| `classify_intent` | `classify_intent` | `CHAIN` | intent, confidence, follow-up resolved |
| `clarify_or_disambiguate` | `clarify_or_disambiguate` | `CHAIN` | resolved, refined_intent, clarification |
| `resolve_assets_with_context_index` | `resolve_assets_with_context_index` | `RETRIEVER` | domain, genie_spaces count, assets count |
| `route_to_genie` | `route_to_genie` | `TOOL` | space_id, sql, status, attempts count |
| `route_to_multi_tool` | `route_to_multi_tool` | `TOOL` | docs_found, doc_vs_index |
| `compose_answer` | `compose_answer` | `CHAIN` | answer length, sources used |

The `predict()` method is traced with `SpanType.AGENT`, creating a parent span that contains all node spans as children.

Additionally, `mlflow.langchain.autolog()` is enabled to capture LangGraph execution details automatically.

### Custom Outputs (Response Metadata)

Every response includes a metadata payload for UI rendering and debugging:

```json
{
  "intent": "simple_kpi",
  "intent_confidence": 0.92,
  "domain": "claims",
  "genie_space": "01f12199fed5107a9d2ccac293b2c0b6",
  "doc_vs_index": null,
  "warnings": [],
  "clarification": null,
  "nodes_executed": ["classify_intent", "resolve_assets", "genie", "compose_answer"],
  "agent_details": {
    "genie": {
      "status": "success",
      "space_id": "01f12199fed5107a9d2ccac293b2c0b6",
      "display_name": "Claims Analytics Space",
      "sql": "SELECT region, COUNT(*) ...",
      "row_count": 5,
      "spaces_tried": 1
    }
  },
  "thread_id": "abc123",
  "checkpoint_id": "a1b2c3d4"
}
```

---

## 12. End-to-End Walkthroughs

### Walkthrough 1: Simple KPI (Multi-Space Selection)

> **User:** "What is the total number of claims by region?"

| Step | Component | Action | Result |
|---|---|---|---|
| 0 | `predict()` | Initialize state, no prior checkpoint | Fresh state created |
| 1 | `classify_intent` | Classify question | `simple_kpi` (95% confidence) |
| -- | `should_clarify` | 95% > 60%, no missing filters | Skip clarification |
| 3 | `resolve_assets` | Query Context Index | Domain: `claims`, 2 Genie Spaces matched |
| -- | `route_by_intent` | `simple_kpi` + has genie_space | Route to `genie` |
| 4 | `route_to_genie` | Try Claims Analytics Space | SQL generated, success on first space |
| 6 | `compose_answer` | Synthesize with Genie results | "Central leads with 1,247 claims..." |
| 7 | `predict()` | Save checkpoint, log episode | Checkpoint ID returned |

**Spaces tried:** 1 (Claims Analytics Space succeeded immediately)

---

### Walkthrough 2: Cross-Domain KPI (Space Fallback)

> **User:** "What is the average premium per agent by region?"

| Step | Component | Action | Result |
|---|---|---|---|
| 0 | `predict()` | Initialize state | Fresh state |
| 1 | `classify_intent` | Classify question | `simple_kpi` (88% confidence) |
| 3 | `resolve_assets` | Query Context Index | Domain: `distribution`, 3 spaces ranked |

**Ranked spaces from Context Index:**
1. Distribution & Channels Space (score: 0.83, endorsed) — "agent performance, channel contributions"
2. Policy & Underwriting Space (score: 0.71, endorsed) — "premium volumes"
3. Customer Analytics Space (score: 0.35, endorsed) — low relevance

| Step | Component | Action | Result |
|---|---|---|---|
| 4a | `route_to_genie` | Try Distribution & Channels Space | Failed (no premium data in agent_performance) |
| 4b | `route_to_genie` | Try Policy & Underwriting Space | SQL generated, success |
| 6 | `compose_answer` | Synthesize from Space #2 results | "The average premium per agent is..." |

**Spaces tried:** 2 (Distribution failed, Policy & Underwriting succeeded)

---

### Walkthrough 3: Document Lookup (RAG)

> **User:** "What does the AIA Health Premium Plan cover?"

| Step | Component | Action | Result |
|---|---|---|---|
| 0 | `predict()` | Initialize state | Fresh state |
| 1 | `classify_intent` | Classify question | `document_lookup` (92% confidence) |
| 3 | `resolve_assets` | Query Context Index | Domain: `documents`, Policy Documents Index found |
| -- | `route_by_intent` | `document_lookup` always | Route to `multi_tool` |
| 5 | `route_to_multi_tool` | RAG over policy_docs_vs | 5 document chunks retrieved |
| 6 | `compose_answer` | Synthesize from documents | "The AIA Health Premium Plan covers hospitalization..." |

---

### Walkthrough 4: Conversational (User Fact Extraction)

> **User:** "Hi, I'm Sarah from the claims team in the Central region"

| Step | Component | Action | Result |
|---|---|---|---|
| 0 | `predict()` | Initialize state | Fresh state |
| 1 | `classify_intent` | Classify question | `conversational` (95% confidence) |
| 3 | `resolve_assets` | Query Context Index | Minimal resolution (low relevance) |
| -- | `route_by_intent` | `conversational` | Route directly to `compose_answer` |
| 6 | `compose_answer` | Generate friendly response | "Hello Sarah! Welcome to the insurance analytics system..." |
| 7a | `predict()` | Save checkpoint | Conversation state saved |
| 7b | `predict()` | Extract user facts | Extracted: `name=Sarah`, `role=claims team`, `preferred_region=Central` |
| 7c | `predict()` | Save to user_memory | 3 facts saved to `ai_ops.user_memory` |

**Next interaction:** When Sarah asks "Total claims by region?", the LLM classification prompt includes her preferences, and the composed answer addresses her by name and may default to the Central region.

---

### Walkthrough 5: Follow-Up with Short-Term Memory

> **Prior:** "What is the total number of claims by region?"
> **Follow-up:** "What about by product?"

| Step | Component | Action | Result |
|---|---|---|---|
| 0 | `predict()` | Load checkpoint from `thread_id` | Prior conversation restored (1 prior exchange) |
| 1a | `classify_intent` | Detect short follow-up (4 words) | Rewritten: "What is the total number of claims by product?" |
| 1b | `classify_intent` | Classify rewritten question | `simple_kpi` (93% confidence) |
| 3 | `resolve_assets` | Query Context Index with rewritten question | Domain: `claims`, Claims Analytics Space matched |
| 4 | `route_to_genie` | Try Claims Analytics Space | SQL: `SELECT product_category, COUNT(*) FROM ...` |
| 6 | `compose_answer` | Synthesize results | "Breaking down claims by product category..." |

---

### Walkthrough 6: Ambiguous Question with Clarification

> **User:** "Show me the numbers"

| Step | Component | Action | Result |
|---|---|---|---|
| 0 | `predict()` | Initialize state | Fresh state, no prior context |
| 1 | `classify_intent` | Classify question | `simple_kpi` (45% confidence), missing: `[domain, metric]` |
| -- | `should_clarify` | 45% < 60% AND missing filters | Route to clarification |
| 2 | `clarify_or_disambiguate` | No prior context, cannot resolve | Generates: "Could you specify which numbers — claims, policies, agent performance, or customer data?" |
| 3 | `resolve_assets` | Query Context Index (best-guess) | Domain: `claims` (default), Claims Space matched |
| 4 | `route_to_genie` | Try Claims Analytics Space | SQL generated (best guess) |
| 6 | `compose_answer` | Synthesize with clarification note | "Here are the latest claims metrics... You might also want to specify which area you're interested in — claims, policies, agent performance, or customer data." |

---

## 13. Multi-Space Failure Handling

### Full Failure Scenario

When all Genie Spaces fail to answer a question, the system activates multiple fallback and feedback mechanisms:

```
User: "What is the average claim settlement time by region for Q4?"
    |
    v
Genie Agent: Try Space #1 (Claims Analytics) -> FAILED
    |
    v
Genie Agent: Try Space #2 (Policy & Underwriting) -> FAILED
    |
    v
All spaces exhausted
    |
    +----> Scoped Context Index Lookup
    |      (query CI again within domain=claims for genie_space assets)
    |      Result: ci_enrichment metadata attached
    |
    +----> Record Asset Feedback
    |      Type: genie_query_failed
    |      Details: "Genie could not answer on [Claims Analytics Space,
    |                Policy & Underwriting Space]: ..."
    |
    v
compose_answer
    |
    v
"I wasn't able to retrieve the average claim settlement time
 by region for Q4 from the current data sources..."
```

### What Governance Teams See

After a failure, `ai_ops.asset_feedback` contains:

| Column | Value |
|---|---|
| `agent_name` | `genie` |
| `domain` | `claims` |
| `feedback_type` | `genie_query_failed` |
| `details` | `Genie could not answer on [Claims Analytics Space, Policy & Underwriting Space]: What is the average claim settlement time by region for Q4?` |
| `user_question` | `What is the average claim settlement time by region for Q4?` |
| `user_id` | `sarah_claims` |

This signals:
- The Genie Space may need additional tables/columns for settlement time metrics
- A new dataset covering settlement time may need to be added to the Context Index
- Sample questions in the Genie Space configuration may need updating

---

## 14. Configuration Reference

### Core Configuration

| Parameter | Value | Location |
|---|---|---|
| LLM Model | `databricks-meta-llama-3-3-70b-instruct` | `agent_code.py` |
| Temperature | `0.1` | `agent_code.py` |
| Max Tokens | `2000` | `agent_code.py` |
| Catalog | `aia_multi_agent_catalog` | `agent_code.py` |
| SQL Warehouse ID | `4b9b953939869799` | `agent_code.py` |

### Vector Search Configuration

| Parameter | Value | Location |
|---|---|---|
| Context Index VS | `aia_multi_agent_catalog.ai_ops.context_index_vs` | `agent_code.py` |
| VS Endpoint | `aia_context_index_vs` | `agent_code.py` |
| Document VS Index | Resolved dynamically from Context Index `metadata.vs_index` | `resolve_assets_with_context_index()` |
| Embedding Model | `databricks-bge-large-en` | Context Index setup |
| Context Index Results | Top 10 | `resolve_assets_with_context_index()` |
| RAG Results | Top 5 documents | `route_to_multi_tool()` |
| Scoped Lookup Results | Top 5 (default), Top 3 for fallback | `_scoped_context_index_lookup()` |

### Genie Configuration

| Parameter | Value | Location |
|---|---|---|
| Polling iterations | 30 per space | `_query_genie_space()` |
| Polling interval | 2 seconds | `_query_genie_space()` |
| Max wait per space | 60 seconds | Derived from 30 x 2s |
| Spaces in ranked list | All matched (typically 2-4) | `resolve_assets_with_context_index()` |

### Cache TTLs

| Cache | TTL | Location |
|---|---|---|
| Prompt cache | 5 minutes | `_load_prompts()` |
| User memory cache | 60 seconds | `_load_user_memory()` |
| Agent capabilities cache | 5 minutes | `_load_agent_capabilities()` |
| Checkpoint retention | 30 days | Scheduled SQL cleanup |
| Episodic lessons retrieved | Top 3 | `_get_episodic_lessons()` |

### Genie Space IDs

| Space | Space ID | Domain |
|---|---|---|
| Claims Analytics | `01f12199fed5107a9d2ccac293b2c0b6` | claims |
| Policy & Underwriting | `01f12199ff0a119d989b057bc2a491c3` | policies |
| Distribution & Channels | `01f12199ff2b1aef96fc954dc1de1a06` | distribution |
| Customer Analytics | `01f12199ff561a40817162d95a240597` | customers |

---

## 15. Extending the System

### Adding a New Genie Space

1. **Create the Genie Space** (via UI or API):
   ```python
   space = w.genie.create_space(
       title="Reinsurance Analytics",
       description="Ask questions about reinsurance treaties, cession rates...",
       table_identifiers=["catalog.gold.reinsurance_summary"],
   )
   ```

2. **Register in Context Index** — Insert a row into `ai_ops.context_index`:
   ```sql
   INSERT INTO aia_multi_agent_catalog.ai_ops.context_index
   VALUES (
       'genie_space',
       '<new_space_id>',
       'Reinsurance Analytics Space',
       'Genie Space for reinsurance analytics. Covers treaty performance, cession rates, retention analysis, facultative placements, and catastrophe exposure...',
       'reinsurance',
       'endorsed',
       '{"type": "genie_space", "space_id": "<new_space_id>", "warehouse_id": "..."}'
   )
   ```

3. **Trigger VS index sync:**
   ```python
   w.vector_search_indexes.sync_index(
       index_name="aia_multi_agent_catalog.ai_ops.context_index_vs"
   )
   ```

4. **Add domain-specific prompt** (optional):
   ```sql
   INSERT INTO aia_multi_agent_catalog.ai_ops.agent_instructions
   VALUES ('genie', 'reinsurance', 'Focus on reinsurance KPIs: treaty performance...', NULL, current_timestamp(), 'admin')
   ```

No code changes required. The Supervisor will automatically discover the new space via the Context Index on the next query.

### Adding a New Worker Agent

1. **Implement the agent** as a new node function in `agent_code.py`
2. **Register in the tool registry:**
   ```sql
   INSERT INTO aia_multi_agent_catalog.ai_ops.agent_capabilities
   VALUES ('cap-new-agent', 'new_agent', 'capability-name', 'Description...',
           ARRAY('target_intent'), ARRAY('target_domain'), '{}', '{}', true, 25, current_timestamp())
   ```
3. **Add the node** to the LangGraph workflow and wire edges
4. **Add routing** — either via the tool registry (priority-based) or by extending `route_by_intent()`

### Adding a New Document Index

1. **Create the document table** and VS index
2. **Register in Context Index** with `asset_type = "document_index"` and `metadata.vs_index = "<new_vs_index_name>"`
3. **Trigger sync** — the Multi-Tool agent will dynamically discover the new VS index at runtime

---

## Summary

The enhanced Supervisor Agent orchestrates a six-node pipeline that transforms user questions into data-grounded answers through semantic asset discovery and multi-Genie-Space routing:

1. **Classify** the user's intent with confidence scoring and follow-up resolution
2. **Clarify** ambiguous questions using conversation context (when confidence < 60%)
3. **Resolve** relevant data assets by querying the Context Index — returns a ranked list of Genie Spaces, document indexes, and domain metadata
4. **Route** to the best agent(s) using a two-phase engine (tool registry + hardcoded fallback)
5. **Execute** the worker agent — Genie tries ranked spaces sequentially with automatic fallback; Multi-Tool performs RAG over dynamically resolved document indexes
6. **Compose** a natural-language answer enriched with agent results, episodic lessons from past interactions, and user preferences for personalization

The multi-Genie-Space architecture enables domain-scoped Text-to-SQL, endorsement-based governance, sequential failure recovery, and no-code extensibility through the Context Index. The three-tier memory system (short-term checkpoints, prompt management, and long-term learning) ensures the agent improves over time while maintaining conversation continuity across sessions.
