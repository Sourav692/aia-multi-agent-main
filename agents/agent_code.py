"""
AIA Supervisor Agent — Standalone code file for MLflow code-based logging.
This file is loaded by mlflow.pyfunc.log_model(python_model="agent_code.py").

Implements 6 nodes:
1. classify_intent
2. clarify_or_disambiguate (optional, when confidence is low)
3. resolve_assets_with_context_index
4. route_to_genie
5. route_to_multi_tool (Vector Search RAG only)
6. compose_answer

P0/P1 enhancements:
- Short-term memory via Delta checkpoint table (ai_ops.conversations)
- Explicit MLflow Tracing spans on every node
- custom_inputs (thread_id, user_id, domain) / custom_outputs
- Prompt management from ai_ops.agent_instructions table
- Endorsed asset routing preference

P2 enhancements:
- Long-term user memory (ai_ops.user_memory) for personalized responses
- Episodic memory (ai_ops.episodic_memory) for continuous learning
- Tool registry (ai_ops.agent_capabilities) for semantic agent routing
"""

import mlflow
import json
import time
import os
import hashlib
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from databricks_langchain import ChatDatabricks
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.entities.span import SpanType
from mlflow.models import set_model

mlflow.langchain.autolog()

# --- Configuration ---
CATALOG = "aia_multi_agent_catalog"
MODEL_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
VS_INDEX = f"{CATALOG}.ai_ops.context_index_vs"
VS_ENDPOINT = "aia_context_index_vs"
SQL_WAREHOUSE_ID = "4b9b953939869799"


# --- SQL Helper (works in Model Serving, no Spark needed) ---
def _run_sql(sql_statement, max_rows=50):
    """Execute SQL via Databricks SDK Statement Execution API."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    response = w.statement_execution.execute_statement(
        warehouse_id=SQL_WAREHOUSE_ID,
        statement=sql_statement,
        wait_timeout="50s",
    )
    if response.status and response.status.state and response.status.state.value == "SUCCEEDED":
        columns = []
        column_meta = []  # list of {"name": ..., "type_name": ...}
        try:
            manifest = response.manifest
            if manifest:
                cols_obj = getattr(manifest, 'columns', None)
                if not cols_obj:
                    schema = getattr(manifest, 'schema', None)
                    if schema:
                        cols_obj = getattr(schema, 'columns', None)
                if cols_obj:
                    columns = [c.name for c in cols_obj]
                    column_meta = [{"name": c.name, "type_name": (getattr(c, 'type_text', None) or getattr(c, 'type_name', None) or 'STRING').upper()} for c in cols_obj]
        except (AttributeError, TypeError):
            pass
        # Fallback: parse column names from SELECT statement
        if not columns:
            import re
            m = re.match(r'\s*SELECT\s+(.+?)\s+FROM\s', sql_statement, re.IGNORECASE | re.DOTALL)
            if m:
                col_str = m.group(1)
                columns = [c.strip().split('.')[-1].split(' ')[-1] for c in col_str.split(',')]
                column_meta = [{"name": c, "type_name": "STRING"} for c in columns]
        rows = []
        if response.result and response.result.data_array:
            for row in response.result.data_array[:max_rows]:
                if columns and len(columns) == len(row):
                    rows.append(dict(zip(columns, row)))
                else:
                    rows.append({f"col_{i}": v for i, v in enumerate(row)})
        return {"columns": columns, "column_meta": column_meta, "rows": rows, "row_count": len(rows)}
    else:
        error_msg = ""
        if response.status and response.status.error:
            error_msg = response.status.error.message
        raise Exception(f"SQL failed: {error_msg}")


# --- Prompt Management (P1) ---
_prompt_cache = {}
_prompt_cache_ts = 0


def _load_prompts():
    """Load prompts from ai_ops.agent_instructions table. Cached for 5 minutes."""
    global _prompt_cache, _prompt_cache_ts
    if time.time() - _prompt_cache_ts < 300 and _prompt_cache:
        return _prompt_cache
    try:
        result = _run_sql(
            f"SELECT agent_id, scope, base_prompt, overlay_prompt "
            f"FROM {CATALOG}.ai_ops.agent_instructions"
        )
        prompts = {}
        for row in result["rows"]:
            key = f"{row['agent_id']}:{row['scope']}"
            base = row.get("base_prompt", "") or ""
            overlay = row.get("overlay_prompt", "") or ""
            prompts[key] = (base + "\n" + overlay).strip()
        _prompt_cache = prompts
        _prompt_cache_ts = time.time()
    except Exception:
        pass
    return _prompt_cache


def _get_prompt(agent_id, scope, fallback=""):
    """Get a prompt for a given agent and scope. Falls back to hardcoded if table not ready."""
    prompts = _load_prompts()
    key = f"{agent_id}:{scope}"
    return prompts.get(key, fallback)


# --- Short-term Memory (P0): Delta-based checkpoint ---
def _save_checkpoint(thread_id, state_data):
    """Save conversation checkpoint to Delta table."""
    try:
        checkpoint_id = hashlib.md5(f"{thread_id}:{time.time()}".encode()).hexdigest()[:16]
        state_json = json.dumps(state_data, default=str).replace("'", "''")
        # Escape any single quotes in state_json for SQL safety
        safe_json = state_json.replace("\\", "\\\\")
        _run_sql(f"""
            INSERT INTO {CATALOG}.ai_ops.conversations
            (thread_id, checkpoint_id, state_json, created_at)
            VALUES ('{thread_id}', '{checkpoint_id}', '{safe_json}', current_timestamp())
        """)
        return checkpoint_id
    except Exception:
        return None


def _load_checkpoint(thread_id):
    """Load the latest checkpoint for a thread."""
    try:
        result = _run_sql(f"""
            SELECT state_json FROM {CATALOG}.ai_ops.conversations
            WHERE thread_id = '{thread_id}'
            ORDER BY created_at DESC LIMIT 1
        """)
        if result["rows"]:
            return json.loads(result["rows"][0]["state_json"])
    except Exception:
        pass
    return None


# --- P2: Long-term Memory ---
_memory_cache = {}
_memory_cache_ts = 0


def _load_user_memory(user_id):
    """Load user preferences/facts from long-term memory table. Cached 60s."""
    global _memory_cache, _memory_cache_ts
    cache_key = f"mem:{user_id}"
    if time.time() - _memory_cache_ts < 60 and cache_key in _memory_cache:
        return _memory_cache[cache_key]
    try:
        result = _run_sql(f"""
            SELECT memory_key, memory_value, memory_type, confidence
            FROM {CATALOG}.ai_ops.user_memory
            WHERE user_id = '{user_id}'
              AND (expires_at IS NULL OR expires_at > current_timestamp())
            ORDER BY confidence DESC
        """)
        memories = {r["memory_key"]: r["memory_value"] for r in result["rows"]}
        _memory_cache[cache_key] = memories
        _memory_cache_ts = time.time()
        return memories
    except Exception:
        return {}


def _save_user_memory(user_id, memory_key, memory_value, memory_type="preference", confidence=1.0):
    """Save a user preference or fact to long-term memory."""
    if not user_id or user_id == "anonymous":
        return
    try:
        key_esc = memory_key.replace("'", "''")
        val_esc = memory_value.replace("'", "''")
        _run_sql(f"""
            MERGE INTO {CATALOG}.ai_ops.user_memory AS t
            USING (SELECT '{user_id}' AS user_id, '{key_esc}' AS memory_key,
                   '{val_esc}' AS memory_value, '{memory_type}' AS memory_type,
                   {confidence} AS confidence,
                   current_timestamp() AS created_at, current_timestamp() AS updated_at,
                   NULL AS expires_at) AS s
            ON t.user_id = s.user_id AND t.memory_key = s.memory_key
            WHEN MATCHED THEN UPDATE SET t.memory_value = s.memory_value,
                t.memory_type = s.memory_type, t.confidence = s.confidence, t.updated_at = s.updated_at
            WHEN NOT MATCHED THEN INSERT *
        """)
        # Invalidate cache
        global _memory_cache, _memory_cache_ts
        _memory_cache_ts = 0
    except Exception:
        pass


def _extract_and_save_user_facts(user_id, question, answer):
    """Use LLM to detect user preferences/facts from conversation and save them."""
    if not user_id or user_id == "anonymous":
        return
    try:
        llm = ChatDatabricks(endpoint=MODEL_ENDPOINT, temperature=0)
        prompt = f"""Analyze this conversation exchange and extract any personal facts or preferences the user shared about themselves.

User said: {question}
Assistant responded: {answer}

Extract facts like: name, role, preferred region, preferred domain, language preference, etc.
Only extract facts the user EXPLICITLY stated. Do not infer or guess.

Respond in JSON format ONLY:
{{"facts": [{{"key": "name", "value": "John", "type": "fact"}}, {{"key": "preferred_region", "value": "Central", "type": "preference"}}]}}

If no personal facts were shared, respond: {{"facts": []}}"""
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        parsed = json.loads(raw)
        for fact in parsed.get("facts", []):
            key = fact.get("key", "").strip()
            value = fact.get("value", "").strip()
            mem_type = fact.get("type", "fact")
            if key and value:
                _save_user_memory(user_id, key, value, mem_type)
    except Exception:
        pass


def _save_episodic_memory(thread_id, user_id, question, intent, domain, agents_used, outcome="success"):
    """Log this interaction to episodic_memory for continuous learning."""
    try:
        episode_id = hashlib.md5(f"{thread_id}:{question}:{time.time()}".encode()).hexdigest()[:20]
        agents_sql = ", ".join([f"'{a}'" for a in agents_used])
        q_esc = question.replace("'", "''")
        _run_sql(f"""
            INSERT INTO {CATALOG}.ai_ops.episodic_memory
            (episode_id, thread_id, user_id, question, intent, domain, agents_used, outcome, created_at)
            VALUES ('{episode_id}', '{thread_id}', '{user_id}', '{q_esc}', '{intent}', '{domain}',
                    ARRAY({agents_sql}), '{outcome}', current_timestamp())
        """)
    except Exception:
        pass


def _get_episodic_lessons(intent, domain, limit=3):
    """Retrieve recent lessons learned for similar intents/domains."""
    try:
        result = _run_sql(f"""
            SELECT question, lesson_learned, outcome, user_rating
            FROM {CATALOG}.ai_ops.episodic_memory
            WHERE intent = '{intent}' AND domain = '{domain}'
              AND lesson_learned IS NOT NULL
            ORDER BY created_at DESC LIMIT {limit}
        """)
        return result["rows"]
    except Exception:
        return []


# --- P2: Tool Registry ---
_capabilities_cache = []
_capabilities_cache_ts = 0


def _load_agent_capabilities():
    """Load agent capabilities from registry for semantic routing. Cached 5min."""
    global _capabilities_cache, _capabilities_cache_ts
    if time.time() - _capabilities_cache_ts < 300 and _capabilities_cache:
        return _capabilities_cache
    try:
        result = _run_sql(f"""
            SELECT capability_id, agent_name, capability_name, description,
                   supported_intents, supported_domains, priority
            FROM {CATALOG}.ai_ops.agent_capabilities
            WHERE is_active = true
            ORDER BY priority ASC
        """)
        _capabilities_cache = result["rows"]
        _capabilities_cache_ts = time.time()
        return _capabilities_cache
    except Exception:
        return []


# --- State ---
class AgentState(TypedDict):
    messages: list
    user_question: str
    intent: str
    intent_confidence: float
    clarification_message: Optional[str]
    needs_clarification: bool
    resolved_assets: Optional[dict]
    genie_results: Optional[dict]
    multi_tool_results: Optional[dict]
    final_answer: Optional[str]
    warnings: list
    thread_id: Optional[str]
    user_id: Optional[str]


# --- LLM ---
llm = ChatDatabricks(endpoint=MODEL_ENDPOINT, temperature=0.1, max_tokens=2000)


# --- Node 1: Classify Intent ---
@mlflow.trace(name="classify_intent", span_type=SpanType.CHAIN)
def classify_intent(state):
    question = state["user_question"]

    # Resolve short follow-ups into full questions using conversation context
    messages = state.get("messages", [])
    state.setdefault("warnings", [])

    if len(messages) > 1 and len(question.split()) <= 10:
        # Get recent conversation excluding the current user message
        recent = messages[-(min(len(messages), 6)):-1]
        if recent:
            conv_lines = [f"{m.get('role','user')}: {m.get('content','')[:300]}" for m in recent]
            try:
                conv_text = "\n".join(conv_lines)
                resolve_resp = llm.invoke([HumanMessage(content=f"Given this conversation:\n{conv_text}\n\nThe user now says: \"{question}\"\n\nThis is a follow-up. Rewrite this as a complete, standalone question that captures what the user actually wants.\nReturn ONLY the rewritten question, nothing else.")])
                resolved = resolve_resp.content.strip().strip('"')
                if len(resolved) > len(question):
                    question = resolved
                    state["user_question"] = question
                    state["warnings"].append(f"Follow-up resolved: {resolved[:150]}")
            except Exception as e:
                state["warnings"].append(f"Follow-up resolution failed: {str(e)[:100]}")

    fallback_prompt = """You are an intent classifier for an insurance analytics system.
Classify the following question into exactly ONE category and provide a confidence score (0.0 to 1.0).

Categories:
- "simple_kpi": Simple KPI/metric questions (counts, totals, averages, trends by region/product/time)
- "document_lookup": Policy terms, coverage details, exclusions, procedures, document search
- "conversational": Greetings, introductions, personal statements, small talk, or non-analytical messages

Question: {question}

Respond in JSON format ONLY:
{{"intent": "<category>", "confidence": <float>, "missing_filters": []}}

If the question is ambiguous or missing key filters (like region, time period, product), list them in missing_filters."""

    # P2: Enrich with user memory context if available
    user_id = state.get("user_id") or "default"
    memory_context = ""
    user_mem = _load_user_memory(user_id)
    if user_mem:
        prefs = "; ".join([f"{k}={v}" for k, v in user_mem.items()])
        memory_context = f"\nUser preferences: {prefs}"

    prompt_template = _get_prompt("supervisor", "classify_intent", fallback_prompt)
    try:
        prompt = prompt_template.format(question=question)
    except (KeyError, IndexError):
        prompt = fallback_prompt.replace("{question}", question)
    prompt += memory_context

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "simple_kpi")
        confidence = float(parsed.get("confidence", 0.8))
        missing_filters = parsed.get("missing_filters", [])
    except (json.JSONDecodeError, ValueError):
        intent = "simple_kpi"
        confidence = 0.5
        missing_filters = []

    valid = ["simple_kpi", "document_lookup", "conversational"]
    if intent not in valid:
        intent = "simple_kpi"

    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["needs_clarification"] = confidence < 0.6 or len(missing_filters) > 0
    state["warnings"] = state.get("warnings", [])

    if missing_filters:
        state["_missing_filters"] = missing_filters

    return state


# --- Node 2: Clarify or Disambiguate (optional) ---
@mlflow.trace(name="clarify_or_disambiguate", span_type=SpanType.CHAIN)
def clarify_or_disambiguate(state):
    """When intent confidence is low or required filters are missing,
    attempt to infer from context or generate a clarification message."""
    question = state["user_question"]
    messages = state.get("messages", [])
    missing_filters = state.get("_missing_filters", [])

    history_context = ""
    if len(messages) > 1:
        prior = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")]
        if prior:
            history_context = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in prior[-4:]])

    prompt = f"""You are helping disambiguate an insurance analytics question.

Question: {question}
Detected intent: {state.get('intent', 'unknown')} (confidence: {state.get('intent_confidence', 0):.2f})
Missing filters: {', '.join(missing_filters) if missing_filters else 'none'}
Conversation history:
{history_context if history_context else 'No prior context'}

Tasks:
1. If the conversation history provides enough context to resolve ambiguity, infer the missing information and set "resolved": true.
2. If not, generate a brief clarification question and set "resolved": false.
3. If the intent seems wrong based on context, suggest the correct intent.

Respond in JSON:
{{"resolved": true/false, "refined_intent": "<intent>", "refined_confidence": <float>, "inferred_filters": {{}}, "clarification_question": "<question if not resolved>"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        parsed = json.loads(raw)
        resolved = parsed.get("resolved", True)
        refined_intent = parsed.get("refined_intent", state["intent"])
        refined_confidence = float(parsed.get("refined_confidence", state["intent_confidence"]))
        clarification = parsed.get("clarification_question", "")
    except (json.JSONDecodeError, ValueError):
        resolved = True
        refined_intent = state["intent"]
        refined_confidence = state["intent_confidence"]
        clarification = ""

    valid = ["simple_kpi", "document_lookup", "conversational"]
    if refined_intent in valid:
        state["intent"] = refined_intent
        state["intent_confidence"] = refined_confidence

    if not resolved and clarification:
        state["clarification_message"] = clarification
        state["warnings"].append(f"Note: The question may benefit from clarification — {clarification}")

    state["needs_clarification"] = False
    return state


# --- Default assets fallback ---
def _get_default_assets(intent="simple_kpi"):
    return {
        "domain": "claims",
        "genie_space": "01f0d6ff25da1f229950bb97c1ec974c",
        "document_indexes": [f"{CATALOG}.bronze.policy_documents"],
        "doc_vs_index": f"{CATALOG}.ai_ops.policy_docs_vs",
        "all_assets": [],
        "endorsement_info": {},
    }


# --- Node 3: Resolve Assets via Context Index (Vector Search) ---
@mlflow.trace(name="resolve_assets_with_context_index", span_type=SpanType.RETRIEVER)
def resolve_assets_with_context_index(state):
    from databricks.sdk import WorkspaceClient
    question = state["user_question"]
    intent = state.get("intent", "simple_kpi")
    try:
        w = WorkspaceClient()
        results = w.vector_search_indexes.query_index(
            index_name=VS_INDEX,
            columns=["asset_type", "asset_id", "display_name", "text", "domain", "endorsement_level", "metadata"],
            query_text=question, num_results=10,
        )
        assets = []
        for row in results.result.data_array:
            assets.append({
                "asset_type": row[0], "asset_id": row[1], "display_name": row[2],
                "text": row[3], "domain": row[4], "endorsement_level": row[5],
                "metadata": row[6] if len(row) > 6 else "{}",
                "score": float(row[7]) if len(row) > 7 else 0.0,
            })

        assets.sort(key=lambda a: (
            0 if a.get("endorsement_level") == "endorsed" else 1,
            -a.get("score", 0)
        ))

        domain_counts = {}
        for a in assets[:5]:
            d = a.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        primary_domain = max(domain_counts, key=domain_counts.get) if domain_counts else "claims"
        genie_spaces = [a for a in assets if a["asset_type"] == "genie_space"]
        doc_indexes = [a for a in assets if a["asset_type"] == "document_index"]

        doc_vs_index = None
        if doc_indexes:
            try:
                meta = json.loads(doc_indexes[0].get("metadata") or "{}")
                doc_vs_index = meta.get("vs_index")
            except (json.JSONDecodeError, TypeError):
                pass

        state["resolved_assets"] = {
            "domain": primary_domain,
            "genie_space": genie_spaces[0]["asset_id"] if genie_spaces else None,
            "document_indexes": [d["asset_id"] for d in doc_indexes],
            "doc_vs_index": doc_vs_index,
            "all_assets": assets,
            "endorsement_info": {a["asset_id"]: a["endorsement_level"] for a in assets},
        }
    except Exception as e:
        state["resolved_assets"] = _get_default_assets(intent)
        state["warnings"].append("Context Index not ready — using rule-based asset resolution")
    return state


# --- Scoped Context Index Lookup (for Worker Agents) ---
def _scoped_context_index_lookup(query_text, domain, asset_types=None, num_results=5):
    """Worker-scoped Context Index lookup.

    Worker agents call this to discover additional assets *within* the domain
    the Supervisor already resolved.  They cannot change domains or override
    the Supervisor's global asset selection.

    Args:
        query_text: semantic search query (e.g. "product hierarchy table")
        domain: the domain resolved by the Supervisor (e.g. "claims")
        asset_types: optional list to filter (e.g. ["genie_space", "document_index"])
        num_results: max results to return (default 5)

    Returns:
        list of asset dicts, or empty list on failure.
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        results = w.vector_search_indexes.query_index(
            index_name=VS_INDEX,
            columns=["asset_type", "asset_id", "display_name", "text", "domain", "endorsement_level"],
            query_text=query_text, num_results=num_results,
        )
        assets = []
        for row in results.result.data_array:
            asset = {
                "asset_type": row[0], "asset_id": row[1], "display_name": row[2],
                "text": row[3], "domain": row[4], "endorsement_level": row[5],
                "score": float(row[6]) if len(row) > 6 else 0.0,
            }
            # Scope restriction: only return assets in the Supervisor's resolved domain
            if asset.get("domain", "").lower() == domain.lower():
                if asset_types is None or asset.get("asset_type") in asset_types:
                    assets.append(asset)
        # Prefer endorsed assets
        assets.sort(key=lambda a: (0 if a.get("endorsement_level") == "endorsed" else 1, -a.get("score", 0)))
        return assets
    except Exception:
        return []


def _record_asset_feedback(agent_name, domain, feedback_type, details, state):
    """Record feedback when a worker agent discovers missing or useful assets.

    This writes to the feedback table so that the ontology and Genie Spaces
    can be improved over time via governance — not ad-hoc at runtime.

    Args:
        agent_name: which agent discovered the gap (e.g. "genie", "multi_tool")
        domain: the domain context
        feedback_type: e.g. "genie_query_failed", "missing_document_index"
        details: freeform description of the gap
        state: the supervisor state (to attach user_id, question context)
    """
    try:
        user_id = state.get("user_id", "default")
        question = state.get("user_question", "")[:200].replace("'", "''")
        details_esc = str(details)[:500].replace("'", "''")
        _run_sql(f"""
            INSERT INTO {CATALOG}.ai_ops.asset_feedback
            (agent_name, domain, feedback_type, details, user_question, user_id, created_at)
            VALUES ('{agent_name}', '{domain}', '{feedback_type}', '{details_esc}',
                    '{question}', '{user_id}', current_timestamp())
        """)
    except Exception:
        # Table may not exist yet — silently skip; governance can create it later
        pass


# --- Node 4: Genie Agent ---
@mlflow.trace(name="route_to_genie", span_type=SpanType.TOOL)
def route_to_genie(state):
    from databricks.sdk import WorkspaceClient
    question = state["user_question"]
    w = WorkspaceClient()

    space_id = state.get("resolved_assets", {}).get("genie_space")

    if not space_id:
        state["genie_results"] = {"error": "No Genie Space resolved from Context Index"}
        state["warnings"].append("Genie Space not found in resolved assets — check Context Index")
        return state

    try:
        conversation = w.genie.start_conversation(space_id=space_id, content=question)
        for _ in range(30):
            msg = w.genie.get_message(
                space_id=space_id,
                conversation_id=conversation.conversation_id,
                message_id=conversation.message_id,
            )
            if msg.status and msg.status.value in ["COMPLETED", "FAILED"]:
                break
            time.sleep(2)

        if msg.status and msg.status.value == "COMPLETED":
            sql_query, result_data = None, None
            if msg.attachments:
                for att in msg.attachments:
                    if att.query and att.query.query:
                        sql_query = att.query.query
                    if att.text and att.text.content:
                        result_data = att.text.content
            state["genie_results"] = {
                "space_id": space_id, "sql": sql_query,
                "result_summary": result_data or "No result text", "status": "success",
            }
        else:
            state["genie_results"] = {"error": f"Genie status: {msg.status}", "status": "failed"}
            state["warnings"].append("Genie query did not complete")
    except Exception as e:
        state["genie_results"] = {"error": str(e)[:200], "status": "failed"}
        state["warnings"].append(f"Genie Agent error: {str(e)[:100]}")

    genie_res = state.get("genie_results", {})
    domain = state.get("resolved_assets", {}).get("domain", "claims")
    if genie_res.get("status") != "success" or not genie_res.get("sql"):
        extra = _scoped_context_index_lookup(
            question, domain, asset_types=["genie_space"], num_results=3,
        )
        if extra:
            genie_res["ci_enrichment"] = [{"asset_id": a["asset_id"], "display_name": a["display_name"],
                                            "asset_type": a["asset_type"]} for a in extra]
        # Feedback: record failure so governance can improve the space
        if genie_res.get("status") != "success":
            _record_asset_feedback("genie", domain, "genie_query_failed",
                                   f"Genie could not answer: {question[:150]}", state)

    return state


# --- Node 5: Multi-Tool Agent (RAG) ---
@mlflow.trace(name="route_to_multi_tool", span_type=SpanType.TOOL)
def route_to_multi_tool(state):
    """Vector Search RAG over policy documents using the VS index resolved from Context Index."""
    question = state["user_question"]
    results = {}

    doc_vs_index = state.get("resolved_assets", {}).get("doc_vs_index")
    if not doc_vs_index:
        results["docs"] = []
        results["error"] = "No document VS index resolved from Context Index"
        results["status"] = "failed"
        state["warnings"].append("Document VS index not found in resolved assets — check Context Index")
        state["multi_tool_results"] = results
        return state

    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        vs_results = w.vector_search_indexes.query_index(
            index_name=doc_vs_index,
            columns=["document_id", "title", "content", "document_type", "category"],
            query_text=question, num_results=5,
        )
        docs = [
            {"document_id": r[0], "title": r[1], "content": r[2], "document_type": r[3], "category": r[4]}
            for r in vs_results.result.data_array
        ]
        results["docs"] = docs
        results["status"] = "success"
    except Exception as e:
        results["docs"] = []
        results["error"] = str(e)[:200]
        state["warnings"].append(f"Document retrieval failed: {str(e)[:100]}")

    state["multi_tool_results"] = results
    return state


# --- Node 8: Compose Final Answer ---
@mlflow.trace(name="compose_answer", span_type=SpanType.CHAIN)
def compose_answer(state):
    question = state["user_question"]
    intent = state.get("intent", "unknown")
    assets = state.get("resolved_assets", {})
    genie = state.get("genie_results", {})
    multi_tool = state.get("multi_tool_results", {})
    clarification = state.get("clarification_message", "")

    context_parts = []

    # Genie results
    if genie and genie.get("status") == "success":
        context_parts.append(f"**Genie Agent:** SQL: {genie.get('sql', 'N/A')}\nResult: {genie.get('result_summary', 'N/A')}")

    # Multi-tool RAG results
    if multi_tool and multi_tool.get("docs"):
        doc_text = "\n".join([f"- {d['title']}: {d['content'][:200]}..." for d in multi_tool["docs"][:3]])
        context_parts.append(f"**Documents:**\n{doc_text}")

    # P2: Episodic memory — retrieve lessons from similar past interactions
    domain = assets.get("domain", "unknown") if assets else "unknown"
    lessons = _get_episodic_lessons(intent, domain)
    if lessons:
        lesson_text = "\n".join([f"- [{l.get('outcome','?')}] {l.get('lesson_learned','')}" for l in lessons if l.get("lesson_learned")])
        if lesson_text:
            context_parts.append(f"**Past experience (internal):**\n{lesson_text}")

    # P2: User memory context for personalization
    user_id = state.get("user_id") or "default"
    user_mem = _load_user_memory(user_id)
    if user_mem:
        pref_parts = []
        if user_mem.get("name"):
            pref_parts.append(f"user's name: {user_mem['name']}")
        if user_mem.get("preferred_view"):
            pref_parts.append(f"preferred view: {user_mem['preferred_view']}")
        if user_mem.get("response_length"):
            pref_parts.append(f"response style: {user_mem['response_length']}")
        if user_mem.get("role"):
            pref_parts.append(f"user role: {user_mem['role']}")
        # Include any other stored preferences
        for k, v in user_mem.items():
            if k not in ("name", "preferred_view", "response_length", "role",
                         "preferred_region", "preferred_domain", "display_currency",
                         "default_time_range", "expertise_level", "team_size",
                         "feedback_viz_style", "feedback_tone"):
                continue
            if k not in [p.split(":")[0].strip() for p in pref_parts]:
                pref_parts.append(f"{k}: {v}")
        if pref_parts:
            context_parts.append(f"**User preferences:** {', '.join(pref_parts)}")

    context = "\n\n".join(context_parts) if context_parts else "No agent results available."

    warning_text = "\n\nCRITICAL RULE: NEVER include a 'Warnings & Limitations' section, warnings, caveats, disclaimers, or notes about data limitations in your answer. These are handled separately by the UI. Your answer must end after the main content — do not append any warning/limitation section."

    clarification_text = ""
    if clarification:
        clarification_text = f"\n\nAlso note this clarification for the user: {clarification}"

    compose_prompt = f"""Answer this question naturally and conversationally: {question}

Agent Results:
{context}

Instructions:
- Answer like a knowledgeable insurance colleague — direct, natural, and helpful.
- Lead with the key insight or answer, not an introduction or preamble.
- Include specific numbers and data from the results naturally in your sentences.
- When the answer warrants detail (trends, analysis, comparisons), provide thorough explanations.
- For simple KPI questions, keep it brief. For complex analysis, be detailed and insightful.
- Use markdown when it helps readability (e.g., tables for comparisons, bold for key figures).
- If you know the user's name from the User preferences section, address them by name naturally.{warning_text}{clarification_text}"""

    response = llm.invoke([HumanMessage(content=compose_prompt)])
    state["final_answer"] = response.content
    return state


# --- Routing Logic ---
def should_clarify(state):
    """Route to clarification if confidence is low."""
    if state.get("needs_clarification", False):
        return "clarify"
    return "resolve"


def route_by_intent(state):
    """Route to the appropriate agent based on intent, resolved assets, and tool registry (P2)."""
    intent = state.get("intent", "simple_kpi")
    assets = state.get("resolved_assets", {})
    has_genie = bool(assets.get("genie_space"))

    capabilities = _load_agent_capabilities()
    if capabilities:
        matching_agents = []
        for cap in capabilities:
            cap_intents = cap.get("supported_intents", "[]")
            if isinstance(cap_intents, str):
                try:
                    cap_intents = json.loads(cap_intents.replace("'", '"'))
                except Exception:
                    cap_intents = []
            if intent in cap_intents:
                matching_agents.append((cap["agent_name"], int(cap.get("priority", 100))))
        if matching_agents:
            matching_agents.sort(key=lambda x: x[1])
            best_agent = matching_agents[0][0]
            agent_map = {"genie": "genie", "multi_tool": "multi_tool"}
            if best_agent in agent_map:
                return agent_map[best_agent]

    if intent == "conversational":
        return "compose_answer"
    elif intent == "document_lookup":
        return "multi_tool"
    elif intent == "simple_kpi":
        return "genie" if has_genie else "multi_tool"
    return "multi_tool"


# --- Build LangGraph ---
workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent)
workflow.add_node("clarify_or_disambiguate", clarify_or_disambiguate)
workflow.add_node("resolve_assets", resolve_assets_with_context_index)
workflow.add_node("genie", route_to_genie)
workflow.add_node("multi_tool", route_to_multi_tool)
workflow.add_node("compose_answer", compose_answer)

workflow.add_edge(START, "classify_intent")

workflow.add_conditional_edges(
    "classify_intent", should_clarify,
    {"clarify": "clarify_or_disambiguate", "resolve": "resolve_assets"},
)

workflow.add_edge("clarify_or_disambiguate", "resolve_assets")

workflow.add_conditional_edges(
    "resolve_assets", route_by_intent,
    {"genie": "genie", "multi_tool": "multi_tool", "compose_answer": "compose_answer"},
)

workflow.add_edge("genie", "compose_answer")
workflow.add_edge("multi_tool", "compose_answer")

workflow.add_edge("compose_answer", END)

graph = workflow.compile()


# --- ResponsesAgent Wrapper ---
class SupervisorResponsesAgent(ResponsesAgent):

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        user_message = None
        all_messages = []
        for item in request.input:
            role = getattr(item, "role", "user")
            content = getattr(item, "content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if hasattr(part, "text"):
                        text_parts.append(part.text)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts)
            elif not isinstance(content, str):
                content = str(content) if content else ""
            all_messages.append({"role": role, "content": content})
            if role == "user":
                user_message = content

        if not user_message:
            return ResponsesAgentResponse(
                output=[self.create_text_output_item(text="Please ask a question.", id="msg_empty")]
            )

        # Extract custom_inputs (P0: thread_id, user_id, domain)
        custom_inputs = {}
        if hasattr(request, "custom_inputs") and request.custom_inputs:
            custom_inputs = request.custom_inputs if isinstance(request.custom_inputs, dict) else {}
        thread_id = custom_inputs.get("thread_id")
        user_id = custom_inputs.get("user_id")

        # Load previous conversation state if thread_id provided (P0: short-term memory)
        # Only use checkpoint if the input doesn't already contain conversation history
        prior_state = None
        if thread_id and len(all_messages) <= 1:
            prior_state = _load_checkpoint(thread_id)
            if prior_state and prior_state.get("messages"):
                prior_msgs = prior_state["messages"]
                all_messages = prior_msgs + all_messages

        initial_state = {
            "messages": all_messages,
            "user_question": user_message,
            "intent": "",
            "intent_confidence": 0.0,
            "clarification_message": None,
            "needs_clarification": False,
            "resolved_assets": None,
            "genie_results": None,
            "multi_tool_results": None,
            "final_answer": None,
            "warnings": [],
            "thread_id": thread_id,
            "user_id": user_id,
        }

        result = graph.invoke(initial_state)
        answer = result.get("final_answer", "I was unable to process your question.")

        # Save checkpoint (P0: short-term memory)
        checkpoint_id = None
        if thread_id:
            checkpoint_data = {
                "messages": all_messages + [{"role": "assistant", "content": answer}],
                "intent": result.get("intent"),
                "domain": result.get("resolved_assets", {}).get("domain") if result.get("resolved_assets") else None,
            }
            checkpoint_id = _save_checkpoint(thread_id, checkpoint_data)

        # P2: Log episodic memory for learning
        agents_used = []
        if result.get("genie_results"):
            agents_used.append("genie")
        if result.get("multi_tool_results"):
            agents_used.append("multi_tool")
        ep_domain = result.get("resolved_assets", {}).get("domain", "unknown") if result.get("resolved_assets") else "unknown"
        has_errors = any(r.get("status") == "failed" for r in [
            result.get("genie_results", {}), result.get("multi_tool_results", {}),
        ] if isinstance(r, dict))
        ep_outcome = "failed" if has_errors and not answer else "success"
        _save_episodic_memory(
            thread_id=thread_id or "anonymous",
            user_id=user_id or "anonymous",
            question=user_message,
            intent=result.get("intent", "unknown"),
            domain=ep_domain,
            agents_used=agents_used,
            outcome=ep_outcome,
        )

        # P2: Extract user facts only for conversational messages (not data queries)
        intent = result.get("intent", "")
        if intent in ("conversational", "unknown", "greeting"):
            _extract_and_save_user_facts(user_id or "default", user_message, answer)

        # Build nodes_executed list
        nodes_executed = [n for n in [
            "classify_intent",
            "clarify_or_disambiguate" if result.get("clarification_message") else None,
            "resolve_assets",
            "genie" if result.get("genie_results") else None,
            "multi_tool" if result.get("multi_tool_results") else None,
            "compose_answer",
        ] if n is not None]

        # Build custom_outputs (P0)
        resolved = result.get("resolved_assets", {})
        domain = resolved.get("domain", "unknown") if resolved else "unknown"

        agent_details = {}
        if result.get("genie_results"):
            gr = result["genie_results"]
            agent_details["genie"] = {
                "status": gr.get("status", "unknown"),
                "space_id": gr.get("space_id"),
                "sql": gr.get("sql", "")[:120] if gr.get("sql") else None,
                "row_count": gr.get("row_count"),
            }
        if result.get("multi_tool_results"):
            mt = result["multi_tool_results"]
            agent_details["multi_tool"] = {
                "docs_found": len(mt.get("docs", [])) if mt.get("docs") else 0,
                "doc_vs_index": resolved.get("doc_vs_index") if resolved else None,
            }
        custom_outputs = {
            "intent": result.get("intent", "unknown"),
            "intent_confidence": result.get("intent_confidence", 0.0),
            "domain": domain,
            "genie_space": resolved.get("genie_space") if resolved else None,
            "doc_vs_index": resolved.get("doc_vs_index") if resolved else None,
            "warnings": result.get("warnings", []),
            "clarification": result.get("clarification_message"),
            "nodes_executed": nodes_executed,
            "agent_details": agent_details,
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }

        # Return metadata as second output item (for backwards compatibility with app.py)
        metadata_json = json.dumps(custom_outputs)

        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(text=answer, id="msg_answer"),
                self.create_text_output_item(text=metadata_json, id="msg_metadata"),
            ],
            custom_outputs=custom_outputs,
        )


agent = SupervisorResponsesAgent()
set_model(agent)
