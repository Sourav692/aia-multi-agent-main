# Supervisor Agent — Step-by-Step Functionality Guide

This document provides a detailed, step-by-step walkthrough of how the Supervisor Agent in the AIA Multi-Agent Insurance Intelligence System processes a user question from start to finish.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Step-by-Step Execution Flow](#3-step-by-step-execution-flow)
   - [Step 0: Request Reception and State Initialization](#step-0-request-reception-and-state-initialization)
   - [Step 1: Intent Classification](#step-1-intent-classification)
   - [Step 2: Clarification and Disambiguation (Conditional)](#step-2-clarification-and-disambiguation-conditional)
   - [Step 3: Asset Resolution via Context Index](#step-3-asset-resolution-via-context-index)
   - [Step 4: Routing Decision](#step-4-routing-decision)
   - [Step 5: Agent Execution](#step-5-agent-execution)
   - [Step 6: Answer Composition](#step-6-answer-composition)
   - [Step 7: Post-Processing and Response](#step-7-post-processing-and-response)
4. [State Schema](#4-state-schema)
5. [Memory Systems](#5-memory-systems)
6. [Routing Logic Deep Dive](#6-routing-logic-deep-dive)
7. [Worker Agents](#7-worker-agents)
8. [Observability and Tracing](#8-observability-and-tracing)
9. [Example Walkthroughs](#9-example-walkthroughs)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Overview

The Supervisor Agent is the orchestrator (the "brain") of the AIA Multi-Agent System. It does **not** directly query databases or retrieve documents. Instead, it:

- **Interprets** the user's intent and confidence level
- **Discovers** relevant data assets via a semantic Context Index
- **Routes** the question to specialist worker agents (Genie, Multi-Tool, or both)
- **Composes** a final natural-language answer from the agent results
- **Maintains** conversation memory across turns

The Supervisor is implemented as a **LangGraph StateGraph** wrapped in an **MLflow ResponsesAgent**, deployed on **Databricks Model Serving**.

**Source file:** `agents/agent_code.py`

---

## 2. Architecture at a Glance

```
User Question (Dash Chat UI)
        │
        ▼
Model Serving Endpoint (aia-supervisor-agent)
        │
        ▼
SupervisorResponsesAgent.predict()
        │
        ├── Load Checkpoint (short-term memory)
        │
        ▼
┌─────────────────── LangGraph StateGraph ───────────────────┐
│                                                             │
│   ┌──────────────────┐                                      │
│   │  classify_intent  │  ← Node 1                           │
│   └────────┬─────────┘                                      │
│            │                                                │
│     ┌──────┴──────┐                                         │
│     │ Confidence   │                                        │
│     │   < 60% ?    │                                        │
│     └──┬───────┬──┘                                         │
│     yes│       │no                                          │
│        ▼       │                                            │
│   ┌────────────────────────┐                                │
│   │clarify_or_disambiguate │  ← Node 2 (conditional)       │
│   └────────────┬───────────┘                                │
│                │                                            │
│                ▼                                            │
│   ┌──────────────────────────────────┐                      │
│   │ resolve_assets_with_context_index │  ← Node 3           │
│   └──────────────┬───────────────────┘                      │
│                  │                                          │
│           ┌──────┴──────┐                                   │
│           │ route_by_   │                                   │
│           │  intent()   │                                   │
│           └──┬────┬──┬──┘                                   │
│              │    │  │                                      │
│    ┌─────────┘    │  └──────────┐                           │
│    ▼              ▼             ▼                           │
│ ┌──────┐   ┌───────────┐  ┌───────────────┐                │
│ │genie │   │multi_tool  │  │compose_answer │                │
│ │      │   │            │  │(conversational)│               │
│ └──┬───┘   └─────┬──────┘  └──────┬────────┘               │
│    │             │                │                         │
│    └─────────────┴────────────────┘                         │
│                  │                                          │
│                  ▼                                          │
│         ┌────────────────┐                                  │
│         │ compose_answer  │  ← Node 6                       │
│         └────────────────┘                                  │
│                  │                                          │
└──────────────────┼──────────────────────────────────────────┘
                   │
                   ▼
        Save Checkpoint + Episodic Memory
                   │
                   ▼
        Return ResponsesAgentResponse
```

---

## 3. Step-by-Step Execution Flow

### Step 0: Request Reception and State Initialization

**Function:** `SupervisorResponsesAgent.predict(request)`

When a user sends a message through the Dash Chat UI, the following happens:

1. **The Chat UI** sends a POST request to the Model Serving endpoint:
   ```json
   {
     "input": [
       {"role": "user", "content": "What is the total number of claims by region?"}
     ],
     "custom_inputs": {
       "thread_id": "abc123",
       "user_id": "default"
     }
   }
   ```

2. **Message extraction** — The agent iterates through `request.input` to extract all messages and identify the latest user message. It handles text parts within multi-part content structures.

3. **Short-term memory check** — If a `thread_id` is provided and the request contains only one message (no conversation history in the payload), the agent loads the **latest checkpoint** from the `ai_ops.conversations` Delta table. This restores prior conversation context (previous messages, previously detected intent, and domain) so the agent can understand follow-up questions.

4. **State initialization** — A fresh `AgentState` dictionary is created:
   ```python
   initial_state = {
       "messages": all_messages,       # Full conversation history
       "user_question": user_message,  # Latest question text
       "intent": "",                   # To be filled by Node 1
       "intent_confidence": 0.0,       # To be filled by Node 1
       "clarification_message": None,  # To be filled by Node 2 (if needed)
       "needs_clarification": False,   # To be set by Node 1
       "resolved_assets": None,        # To be filled by Node 3
       "genie_results": None,          # To be filled by Node 4
       "multi_tool_results": None,     # To be filled by Node 5
       "final_answer": None,           # To be filled by Node 6
       "warnings": [],                 # Accumulated warnings
       "thread_id": thread_id,         # Conversation thread ID
       "user_id": user_id,             # User identifier
   }
   ```

5. **Graph invocation** — The LangGraph `graph.invoke(initial_state)` is called, which triggers the node pipeline.

---

### Step 1: Intent Classification

**Node:** `classify_intent` | **Span Type:** `CHAIN`

This is the first node in the pipeline. It determines *what kind of question* the user is asking.

#### 1a. Follow-Up Resolution

If the conversation has more than one message and the current question is short (10 words or fewer), the LLM is asked to rewrite it as a standalone question using conversation context.

**Example:**
- Prior context: "What is the total number of claims by region?"
- Follow-up: "What about by product?"
- Resolved: "What is the total number of claims by product?"

#### 1b. User Memory Enrichment (P2)

The agent loads any stored user preferences from `ai_ops.user_memory` (e.g., `preferred_region=Central`, `role=Claims Analyst`). These preferences are appended to the classification prompt so the LLM can consider default filters.

#### 1c. Prompt Retrieval

The classification prompt is loaded from the `ai_ops.agent_instructions` table using key `supervisor:classify_intent`. If the table is unavailable, a hardcoded fallback prompt is used.

#### 1d. LLM Classification

The prompt asks the LLM to classify the question into exactly one of three categories:

| Intent | Description | Example |
|--------|-------------|---------|
| `simple_kpi` | Simple KPI/metric questions — counts, totals, averages, trends | "Total claims by region?" |
| `document_lookup` | Policy terms, coverage details, exclusions, procedures | "What does the health plan cover?" |
| `conversational` | Greetings, introductions, personal statements, small talk | "Hi, I'm Sarah from the claims team" |

The LLM returns a JSON response:
```json
{
  "intent": "simple_kpi",
  "confidence": 0.92,
  "missing_filters": []
}
```

#### 1e. Clarification Trigger

The node sets `needs_clarification = True` if:
- **Confidence is below 60%**, OR
- **Missing filters** are detected (e.g., no region, time period, or product specified)

#### State Updates

| Field | Updated To |
|-------|-----------|
| `intent` | One of: `simple_kpi`, `document_lookup`, `conversational` |
| `intent_confidence` | Float 0.0–1.0 |
| `needs_clarification` | Boolean |
| `_missing_filters` | List of missing filter names (internal) |
| `user_question` | May be rewritten if follow-up resolution occurred |

---

### Step 2: Clarification and Disambiguation (Conditional)

**Node:** `clarify_or_disambiguate` | **Span Type:** `CHAIN`

This node is **only triggered** when `needs_clarification` is `True` (confidence < 60% or missing filters detected).

#### What it does:

1. **Gathers context** — Collects the last 4 messages from conversation history.

2. **Asks the LLM** to either:
   - **Resolve the ambiguity** using conversation history (e.g., inferring that "Show me the numbers" refers to claims data based on prior questions), OR
   - **Generate a clarification question** if the history doesn't provide enough context.

3. **Updates state:**
   - If resolved: updates `intent` and `intent_confidence` with refined values
   - If not resolved: stores a `clarification_message` (e.g., "Could you specify which numbers you'd like to see — claims, policies, agents, or customer data?") and appends it to warnings

4. **Always sets** `needs_clarification = False` after processing, so the flow continues regardless.

**Important:** Even when a clarification message is generated, the pipeline continues to execute (it does not stop and wait for user input). The clarification message is included in the final composed answer.

---

### Step 3: Asset Resolution via Context Index

**Node:** `resolve_assets_with_context_index` | **Span Type:** `RETRIEVER`

This is where the Supervisor discovers **which data assets** are relevant to the user's question.

#### How it works:

1. **Vector Search query** — The user's question is sent as a semantic query to the Context Index Vector Search endpoint (`aia_context_index_vs`), retrieving the top 10 matching assets. The query includes the `metadata` column so that worker-specific configuration (e.g., the VS index name for document indexes) can be extracted at resolution time.

2. **Asset types** in the Context Index:
   - **Genie Spaces** — For Text-to-SQL queries. The `asset_id` is the Genie Space ID used directly by the Genie worker agent.
   - **Document Indexes** — For RAG over policy documents. The `metadata` JSON contains a `vs_index` field that tells the Multi-Tool agent which Vector Search index to query.

3. **Endorsed asset prioritization** — Results are sorted so that assets with `endorsement_level = "endorsed"` appear first, followed by score-based ranking. This ensures governed, curated assets are preferred over experimental ones.

4. **Domain detection** — The primary domain (e.g., `claims`, `policies`, `documents`) is determined by counting the most frequent domain among the top 5 results.

5. **VS index extraction** — For the first matched `document_index` asset, the `metadata` JSON is parsed to extract the `vs_index` field. This allows the Multi-Tool agent to dynamically discover which Vector Search index to query, rather than relying on a hardcoded constant.

6. **State update** — The `resolved_assets` dictionary is populated:
   ```python
   state["resolved_assets"] = {
       "domain": "claims",
       "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",
       "document_indexes": ["bronze.policy_documents"],
       "doc_vs_index": "aia_multi_agent_catalog.ai_ops.policy_docs_vs",
       "all_assets": [...],          # Full asset list
       "endorsement_info": {...},     # asset_id → endorsement_level
   }
   ```

#### Fallback

If the Context Index is unavailable (e.g., during initial setup), a **rule-based default** is used with a hardcoded Genie Space ID and document VS index:
```python
{
    "domain": "claims",
    "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",
    "document_indexes": ["aia_multi_agent_catalog.bronze.policy_documents"],
    "doc_vs_index": "aia_multi_agent_catalog.ai_ops.policy_docs_vs",
    ...
}
```

---

### Step 4: Routing Decision

**Function:** `route_by_intent(state)` (conditional edge function)

After assets are resolved, the Supervisor decides **which worker agent(s)** to invoke.

#### Decision Logic (in priority order):

**Phase 1 — Tool Registry (P2):**
If the `ai_ops.agent_capabilities` table has active entries, the router checks for agents whose `supported_intents` match the classified intent. The agent with the lowest `priority` number wins.

**Phase 2 — Hardcoded Fallback:**
If the registry has no matches or isn't available:

| Intent | Condition | Routes To |
|--------|-----------|-----------|
| `conversational` | Always | `compose_answer` (direct response, no agent) |
| `document_lookup` | Always | `multi_tool` (RAG over policy documents) |
| `simple_kpi` | Genie Space resolved | `genie` (Text-to-SQL via Genie API) |
| `simple_kpi` | No Genie Space | `multi_tool` (fallback) |
| Default | — | `multi_tool` |

---

### Step 5: Agent Execution

Based on the routing decision, one of three paths is taken:

#### Path A: Genie Agent

**Node:** `route_to_genie` | **Span Type:** `TOOL`

1. Reads the Genie Space ID from `state["resolved_assets"]["genie_space"]` — this ID was dynamically resolved from the Context Index in Step 3.
2. Calls the **Databricks Genie API** to start a conversation with the user's question.
3. Polls the Genie API (up to 30 iterations, 2 seconds apart) until the query completes.
4. Extracts:
   - The **SQL query** Genie generated
   - The **result summary** (natural language)
5. If Genie fails, performs a **scoped Context Index lookup** for additional Genie Spaces in the same domain.
6. Records **asset feedback** when Genie cannot answer a query, enabling governance teams to improve Genie Spaces over time.

**State update:** `state["genie_results"]` with `space_id`, `sql`, `result_summary`, `status`

#### Path B: Multi-Tool Agent (RAG)

**Node:** `route_to_multi_tool` | **Span Type:** `TOOL`

1. Reads the document VS index name from `state["resolved_assets"]["doc_vs_index"]` — this was dynamically resolved from the `document_index` asset's `metadata.vs_index` field in Step 3.
2. Queries the resolved **Policy Documents Vector Search index** with the user's question.
3. Retrieves the **top 5 matching document chunks**, each containing:
   - `document_id`, `title`, `content`, `document_type`, `category`
4. Returns the retrieved documents for use in answer composition.

**State update:** `state["multi_tool_results"]` with `docs`, `status`

---

### Step 6: Answer Composition

**Node:** `compose_answer` | **Span Type:** `CHAIN`

This is where all agent results are synthesized into a final, natural-language answer.

#### Input Assembly

The node gathers context from multiple sources:

1. **Genie results** — SQL query and result summary (if Genie was invoked and succeeded)
2. **Multi-Tool results** — Top 3 retrieved documents with titles and content previews (if Multi-Tool was invoked)
3. **Episodic lessons (P2)** — Up to 3 lessons learned from similar past interactions (same intent + domain), retrieved from `ai_ops.episodic_memory`
4. **User preferences (P2)** — Personalization data such as the user's name, role, preferred response style, preferred region, etc.
5. **Clarification message** — If generated in Step 2, included for the LLM to weave into the response

#### Composition Prompt

The LLM is instructed to:
- Answer like a knowledgeable insurance colleague — direct, natural, and helpful
- Lead with the key insight, not an introduction or preamble
- Include specific numbers and data naturally
- Scale detail to the question: brief for simple KPIs, thorough for complex analysis
- Use markdown for readability (tables, bold key figures)
- Address the user by name if known from preferences
- Never include a "Warnings & Limitations" section (these are handled separately by the UI)

**State update:** `state["final_answer"]` with the composed response text

---

### Step 7: Post-Processing and Response

After `graph.invoke()` returns, the `predict()` method handles several post-processing tasks:

#### 7a. Save Conversation Checkpoint (P0)

If a `thread_id` was provided, the conversation state (all messages, current intent, resolved domain) is serialized to JSON and saved to `ai_ops.conversations`. This enables multi-turn conversations.

#### 7b. Log Episodic Memory (P2)

Every completed interaction is logged to `ai_ops.episodic_memory` with:
- `question`, `intent`, `domain`
- `agents_used` (which agents were invoked)
- `outcome` (`success` or `failed`)

This data powers the episodic lessons used in future answer composition.

#### 7c. Extract User Facts (P2)

For conversational messages (greetings, introductions), the LLM analyzes the exchange to detect any personal facts or preferences the user explicitly shared (e.g., "I'm the Claims Manager for the Central region"). Extracted facts are saved to `ai_ops.user_memory` for future personalization.

#### 7d. Build Response

The final response includes:

**Output items:**
1. `msg_answer` — The composed natural-language answer
2. `msg_metadata` — JSON blob with execution metadata

**Custom outputs:**
```json
{
  "intent": "simple_kpi",
  "intent_confidence": 0.92,
  "domain": "claims",
  "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",
  "doc_vs_index": "aia_multi_agent_catalog.ai_ops.policy_docs_vs",
  "warnings": [],
  "clarification": null,
  "nodes_executed": ["classify_intent", "resolve_assets", "genie", "compose_answer"],
  "agent_details": {
    "genie": {"status": "success", "space_id": "01f0d6ff25da1f229950bb97c1ec974c", "sql": "SELECT region, COUNT(*)...", "row_count": 5}
  },
  "thread_id": "abc123",
  "checkpoint_id": "a1b2c3d4"
}
```

---

## 4. State Schema

The `AgentState` TypedDict defines the shared state that flows through all nodes:

| Field | Type | Set By | Purpose |
|-------|------|--------|---------|
| `messages` | `list` | Step 0 | Full conversation history |
| `user_question` | `str` | Step 0/1 | Current question (may be rewritten) |
| `intent` | `str` | Step 1 | Classified intent category |
| `intent_confidence` | `float` | Step 1 | Classification confidence (0.0–1.0) |
| `clarification_message` | `Optional[str]` | Step 2 | Generated clarification question |
| `needs_clarification` | `bool` | Step 1/2 | Whether clarification is needed |
| `resolved_assets` | `Optional[dict]` | Step 3 | Discovered data assets |
| `genie_results` | `Optional[dict]` | Step 5 | Genie agent output |
| `multi_tool_results` | `Optional[dict]` | Step 5 | Multi-Tool agent output |
| `final_answer` | `Optional[str]` | Step 6 | Composed response text |
| `warnings` | `list` | All steps | Accumulated operational warnings |
| `thread_id` | `Optional[str]` | Step 0 | Conversation thread identifier |
| `user_id` | `Optional[str]` | Step 0 | User identifier |

---

## 5. Memory Systems

The Supervisor uses a layered memory architecture:

### P0 — Short-Term Memory

| Component | Table | TTL | Purpose |
|-----------|-------|-----|---------|
| Conversation Checkpoints | `ai_ops.conversations` | 30 days | Multi-turn conversation context |
| UI Sessions | `ai_ops.ui_sessions` | Indefinite | Chat UI state across browser reloads |

### P1 — Prompt and Semantic Context

| Component | Table/Index | Cache TTL | Purpose |
|-----------|-------------|-----------|---------|
| Prompt Management | `ai_ops.agent_instructions` | 5 min | Table-driven prompts with overlay support |
| Context Index | `ai_ops.context_index_vs` | — | Semantic asset discovery via Vector Search (Genie Spaces + Document Indexes) |
| Policy Documents | Resolved from Context Index `metadata.vs_index` | — | RAG retrieval for document questions |

### P2 — Long-Term Learning

| Component | Table | Cache TTL | Purpose |
|-----------|-------|-----------|---------|
| User Memory | `ai_ops.user_memory` | 60 sec | Personalization (name, role, preferences) |
| Episodic Memory | `ai_ops.episodic_memory` | — | Interaction logs for learning |
| Agent Capabilities | `ai_ops.agent_capabilities` | 5 min | Semantic tool registry for routing |
| Asset Feedback | `ai_ops.asset_feedback` | — | Gap discovery for governance improvement |

---

## 6. Routing Logic Deep Dive

The routing function `route_by_intent()` uses a two-phase approach:

```
Phase 1: Tool Registry (P2, if available)
    │
    ├── Load active capabilities from ai_ops.agent_capabilities
    ├── Filter by: supported_intents contains current intent
    ├── Sort by: priority (ascending, lower = higher priority)
    └── If match found → route to matched agent
    │
    ▼
Phase 2: Hardcoded Fallback
    │
    ├── conversational → compose_answer (direct response)
    ├── document_lookup → multi_tool (RAG)
    ├── simple_kpi + has genie_space → genie
    ├── simple_kpi + no genie_space → multi_tool
    └── default → multi_tool
```

The tool registry enables adding new agents without code changes — simply insert a row into `ai_ops.agent_capabilities` with the new agent's supported intents and priority.

---

## 7. Worker Agents

### Genie Agent (BI Specialist)

| Aspect | Detail |
|--------|--------|
| **API** | Databricks Genie Space API (`w.genie.start_conversation`, `w.genie.get_message`) |
| **Space ID source** | Dynamically resolved from Context Index (`resolved_assets.genie_space`) |
| **Function** | Translates natural language into SQL via curated Genie Spaces |
| **Returns** | Space ID, generated SQL, result summary, row count |
| **Fallback** | Scoped Context Index lookup for alternative Genie Spaces in the same domain |
| **Feedback** | Records `genie_query_failed` to `ai_ops.asset_feedback` on failure |

### Multi-Tool Agent (RAG)

| Aspect | Detail |
|--------|--------|
| **API** | Databricks Vector Search (`w.vector_search_indexes.query_index`) |
| **VS index source** | Dynamically resolved from Context Index (`resolved_assets.doc_vs_index`, extracted from `document_index` asset metadata) |
| **Function** | Retrieves relevant policy document chunks via semantic similarity |
| **Returns** | Up to 5 document chunks with title, content, type, and category |
| **Use case** | Policy coverage, terms, exclusions, procedures |

### Scoped Context Index Lookup (Worker Helper)

Worker agents can perform **additional asset discovery** within the domain already resolved by the Supervisor, using `_scoped_context_index_lookup()`. Workers cannot change the Supervisor's domain selection — they can only discover additional assets within the established scope.

---

## 8. Observability and Tracing

Every node in the pipeline is instrumented with **MLflow Tracing**:

```python
@mlflow.trace(name="classify_intent", span_type=SpanType.CHAIN)
@mlflow.trace(name="clarify_or_disambiguate", span_type=SpanType.CHAIN)
@mlflow.trace(name="resolve_assets_with_context_index", span_type=SpanType.RETRIEVER)
@mlflow.trace(name="route_to_genie", span_type=SpanType.TOOL)
@mlflow.trace(name="route_to_multi_tool", span_type=SpanType.TOOL)
@mlflow.trace(name="compose_answer", span_type=SpanType.CHAIN)
```

The `predict()` method itself is traced with `SpanType.AGENT`, creating a parent span for the entire request.

The Chat UI renders a "Thinking..." panel showing each node that executed, its status (success/failed), and relevant details (intent confidence, resolved domain, agent row counts).

---

## 9. Example Walkthroughs

### Example 1: Simple KPI Question

> **User:** "What is the total number of claims by region?"

| Step | Node | Action | Result |
|------|------|--------|--------|
| 0 | `predict()` | Initialize state, no prior checkpoint | Fresh state |
| 1 | `classify_intent` | Classify question | `simple_kpi` (95% confidence) |
| — | `should_clarify` | Check confidence | 95% > 60% → skip clarification |
| 3 | `resolve_assets` | Query Context Index | domain: `claims`, genie_space: `01f0d6...` |
| 4 | `route_by_intent` | Route decision | Has genie_space → route to `genie` |
| 5 | `route_to_genie` | Call Genie API | SQL: `SELECT region, COUNT(*) FROM claims...` |
| 6 | `compose_answer` | Synthesize answer | "Here are the claims by region: Central has 1,247 claims..." |
| 7 | `predict()` | Save checkpoint, log episode | Checkpoint saved, episode logged |

### Example 2: Document Lookup (RAG)

> **User:** "What does the AIA Health Premium Plan cover?"

| Step | Node | Action | Result |
|------|------|--------|--------|
| 0 | `predict()` | Initialize state | Fresh state |
| 1 | `classify_intent` | Classify question | `document_lookup` (92% confidence) |
| — | `should_clarify` | Check confidence | 92% > 60% → skip clarification |
| 3 | `resolve_assets` | Query Context Index | domain: `documents`, doc_indexes found |
| 4 | `route_by_intent` | Route decision | `document_lookup` → route to `multi_tool` |
| 5 | `route_to_multi_tool` | Vector Search RAG | 5 matching document chunks retrieved |
| 6 | `compose_answer` | Synthesize answer | "The AIA Health Premium Plan covers hospitalization, surgical..." |
| 7 | `predict()` | Save checkpoint, log episode | Checkpoint saved |

### Example 3: Conversational Question

> **User:** "Hi, I'm Sarah from the claims team"

| Step | Node | Action | Result |
|------|------|--------|--------|
| 0 | `predict()` | Initialize state | Fresh state |
| 1 | `classify_intent` | Classify question | `conversational` (95% confidence) |
| — | `should_clarify` | Check confidence | 95% > 60% → skip clarification |
| 3 | `resolve_assets` | Query Context Index | Minimal resolution (conversational intent) |
| 4 | `route_by_intent` | Route decision | `conversational` → route directly to `compose_answer` |
| 5 | `compose_answer` | Generate greeting | "Hello Sarah! I'm your insurance analytics assistant..." |
| 6 | `predict()` | Save checkpoint, extract user memory | Name preference saved |

### Example 4: Ambiguous Question with Clarification

> **User:** "Show me the numbers"

| Step | Node | Action | Result |
|------|------|--------|--------|
| 0 | `predict()` | Initialize state | Fresh state |
| 1 | `classify_intent` | Classify question | `simple_kpi` (45% confidence), missing: `[domain, metric]` |
| — | `should_clarify` | Check confidence | 45% < 60% → route to clarification |
| 2 | `clarify_or_disambiguate` | Attempt resolution | Not resolved → generates clarification question |
| 3 | `resolve_assets` | Query Context Index | Best-guess assets based on refined intent |
| 4–6 | Normal flow | Process as usual | Answer includes clarification prompt |

### Example 5: Follow-Up Question

> **Prior:** "What is the total number of claims by region?"
> **User:** "What about by product?"

| Step | Node | Action | Result |
|------|------|--------|--------|
| 0 | `predict()` | Load checkpoint from `thread_id` | Prior conversation restored |
| 1 | `classify_intent` | Detect short follow-up (4 words) | Rewrites to: "What is the total number of claims by product?" |
| 1 | `classify_intent` | Classify rewritten question | `simple_kpi` (93% confidence) |
| 3–7 | Normal flow | Process rewritten question | Full answer about claims by product |

---

## 10. Configuration Reference

| Parameter | Value | Location |
|-----------|-------|----------|
| **LLM Model** | `databricks-meta-llama-3-3-70b-instruct` | `agent_code.py` |
| **Temperature** | `0.1` | `agent_code.py` |
| **Max Tokens** | `2000` | `agent_code.py` |
| **Catalog** | `aia_multi_agent_catalog` | `agent_code.py` |
| **Context Index VS** | `aia_multi_agent_catalog.ai_ops.context_index_vs` | `agent_code.py` |
| **VS Endpoint** | `aia_context_index_vs` | `agent_code.py` |
| **Document VS Index** | Resolved dynamically from Context Index `document_index` asset `metadata.vs_index` | `resolve_assets_with_context_index()` |
| **SQL Warehouse ID** | `4b9b953939869799` | `agent_code.py` |
| **Serving Endpoint** | `aia-supervisor-agent` | `databricks.yml` |
| **Prompt Cache TTL** | 5 minutes | `_load_prompts()` |
| **User Memory Cache TTL** | 60 seconds | `_load_user_memory()` |
| **Capabilities Cache TTL** | 5 minutes | `_load_agent_capabilities()` |
| **Checkpoint Retention** | 30 days | Scheduled SQL cleanup |
| **Genie Polling** | 30 iterations, 2s interval | `route_to_genie()` |
| **Context Index Results** | Top 10 | `resolve_assets_with_context_index()` |
| **RAG Results** | Top 5 documents | `route_to_multi_tool()` |
| **Episodic Lessons** | Top 3 | `_get_episodic_lessons()` |

---

## Summary

The Supervisor Agent orchestrates the entire question-answering pipeline through six core nodes:

1. **Classify** the user's intent with confidence scoring
2. **Clarify** ambiguous questions using conversation context (when needed)
3. **Resolve** relevant data assets via semantic search over the Context Index
4. **Route** to the appropriate specialist agent(s) based on intent and available assets
5. **Execute** the specialist agent(s) — Genie for structured data, Multi-Tool for document retrieval, or both in parallel
6. **Compose** a natural-language answer enriched with agent results, episodic lessons, and user personalization

All steps are traced with MLflow, state is persisted for multi-turn conversations, and the system continuously learns through episodic memory and user fact extraction.
