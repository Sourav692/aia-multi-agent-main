# COMMAND ---------- [markdown]
# # AIA Supervisor Agent — Interactive Notebook
#
# This notebook is the interactive version of `agent_code.py`. It implements the full Supervisor Agent with:
#
# **6 LangGraph Nodes:**
# 1. `classify_intent` — Interprets user question intent
# 2. `clarify_or_disambiguate` — Handles ambiguous queries (optional)
# 3. `resolve_assets_with_context_index` — Discovers data assets via Vector Search
# 4. `route_to_genie` — Text-to-SQL via Genie Space
# 5. `route_to_multi_tool` — Vector Search RAG over policy documents
# 6. `compose_answer` — Synthesizes final response
#
# **Enhancements:**
# - P0: Short-term memory (Delta checkpoints), MLflow Tracing, custom I/O
# - P1: Prompt management, endorsed asset routing
# - P2: Long-term user memory, episodic memory, tool registry

# COMMAND ---------- [markdown]
# ## Install Dependencies

# COMMAND ----------

# !pip install "mlflow>=3.1" "databricks-agents>=1.0.0" "pydantic>=2" "langgraph>=0.2" langchain-core databricks-langchain databricks-vectorsearch databricks-sdk databricks-ai-bridge rich --upgrade

# COMMAND ----------

# dbutils.library.restartPython()  # Uncomment on Databricks

# COMMAND ---------- [markdown]
# ## Imports & Configuration

# COMMAND ----------

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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

mlflow.langchain.autolog()

# COMMAND ----------

CATALOG = "aia_multi_agent_catalog"
MODEL_ENDPOINT = "databricks-claude-opus-4-6"
VS_INDEX = f"{CATALOG}.ai_ops.context_index_vs"
VS_ENDPOINT = "aia_context_index_vs"
SQL_WAREHOUSE_ID = "4b9b953939869799"

print(f"Catalog: {CATALOG}")
print(f"LLM Endpoint: {MODEL_ENDPOINT}")
print(f"VS Index: {VS_INDEX}")
MAX_MESSAGES = 7

# COMMAND ----------

console = Console()
_flow_step = 0

NODE_STYLES = {
    "classify_intent": ("bold cyan", "Classify Intent"),
    "clarify_or_disambiguate": ("bold yellow", "Clarify / Disambiguate"),
    "resolve_assets": ("bold green", "Resolve Assets (Context Index)"),
    "genie": ("bold blue", "Genie Agent (Text-to-SQL)"),
    "multi_tool": ("bold magenta", "Multi-Tool Agent (RAG)"),
    "compose_answer": ("bold white", "Compose Final Answer"),
}


def _reset_steps():
    global _flow_step
    _flow_step = 0


def _next_step():
    global _flow_step
    _flow_step += 1
    return _flow_step


def log_flow_start(question, thread_id=None, user_id=None):
    _reset_steps()
    parts = Text()
    parts.append("AIA Supervisor Agent — Flow Started\n\n", style="bold bright_green")
    parts.append("Question: ", style="dim")
    parts.append(f"{question[:150]}\n", style="bright_white")
    if thread_id:
        parts.append("Thread:   ", style="dim")
        parts.append(f"{thread_id}\n", style="bright_white")
    if user_id:
        parts.append("User:     ", style="dim")
        parts.append(f"{user_id}", style="bright_white")
    console.print()
    console.print(Panel(parts, border_style="bright_green", box=box.DOUBLE_EDGE))


def log_node_start(node_name, step):
    style, label = NODE_STYLES.get(node_name, ("bold white", node_name))
    console.print()
    console.rule(f"[{style}] Step {step} | {label} [/]", style=style)


def log_node_detail(key, value, style="bright_white"):
    console.print(f"  [dim]{key + ':':<22}[/] [{style}]{value}[/]")


def log_node_complete(node_name, duration_ms=None):
    style, label = NODE_STYLES.get(node_name, ("bold white", node_name))
    dur = f" ({duration_ms:.0f}ms)" if duration_ms else ""
    console.print(f"  [{style}]>> {label} complete{dur}[/]")


def _log_node_details(node_name, state):
    if node_name == "classify_intent":
        log_node_detail("User Question", state.get("user_question", "?")[:100])
        intent = state.get("intent", "?")
        conf = state.get("intent_confidence", 0)
        log_node_detail("Detected Intent", intent, style="bold cyan")
        log_node_detail("Confidence", f"{conf:.0%}", style="bold cyan" if conf >= 0.6 else "bold red")
        missing = state.get("_missing_filters", [])
        if missing:
            log_node_detail("Missing Filters", ", ".join(missing), style="yellow")
        log_node_detail("Needs Clarification", str(state.get("needs_clarification", False)))

    elif node_name == "clarify_or_disambiguate":
        log_node_detail("Intent (refined)", state.get("intent", "?"), style="bold yellow")
        log_node_detail("Confidence (refined)", f"{state.get('intent_confidence', 0):.0%}")
        if state.get("clarification_message"):
            log_node_detail("Clarification", state["clarification_message"][:120], style="yellow")

    elif node_name == "resolve_assets":
        assets = state.get("resolved_assets") or {}
        log_node_detail("Domain", assets.get("domain", "N/A"), style="bold green")
        gs = assets.get("genie_space")
        log_node_detail("Genie Space", gs[:36] + "..." if gs and len(gs) > 36 else (gs or "None"))
        log_node_detail("Doc VS Index", assets.get("doc_vs_index") or "None")
        all_assets = assets.get("all_assets", [])
        log_node_detail("Total Assets", str(len(all_assets)))
        endorsed = [a for a in all_assets if a.get("endorsement_level") == "endorsed"]
        if endorsed:
            log_node_detail("Endorsed", str(len(endorsed)), style="bold green")

    elif node_name == "genie":
        gr = state.get("genie_results") or {}
        status = gr.get("status", "N/A")
        log_node_detail("Status", status, style="bold blue" if status == "success" else "bold red")
        if gr.get("sql"):
            log_node_detail("SQL Query", gr["sql"][:120])
        if gr.get("result_summary") and status == "success":
            log_node_detail("Result Preview", str(gr["result_summary"])[:120])
        if gr.get("error"):
            log_node_detail("Error", str(gr["error"])[:120], style="red")

    elif node_name == "multi_tool":
        mt = state.get("multi_tool_results") or {}
        status = mt.get("status", "N/A")
        log_node_detail("Status", status, style="bold magenta" if status == "success" else "bold red")
        docs = mt.get("docs", [])
        log_node_detail("Docs Retrieved", str(len(docs)))
        for i, doc in enumerate(docs[:3]):
            log_node_detail(f"  Doc {i+1}", doc.get("title", "untitled")[:60])
        if mt.get("error"):
            log_node_detail("Error", str(mt["error"])[:120], style="red")

    elif node_name == "compose_answer":
        answer = state.get("final_answer", "")
        log_node_detail("Answer Length", f"{len(answer)} chars")
        preview = answer[:150].replace("\n", " ")
        log_node_detail("Preview", preview + ("..." if len(answer) > 150 else ""))


def _with_logging(node_name, fn):
    def wrapper(state):
        step = _next_step()
        log_node_start(node_name, step)
        _start = time.time()
        result = fn(state)
        _dur = (time.time() - _start) * 1000
        _log_node_details(node_name, result)
        log_node_complete(node_name, _dur)
        return result
    return wrapper


def log_flow_end(result, duration_s=None):
    table = Table(
        title="Flow Complete",
        box=box.ROUNDED,
        border_style="bright_green",
        title_style="bold bright_green",
        show_lines=False,
    )
    table.add_column("Field", style="dim", width=22)
    table.add_column("Value", style="bright_white")

    intent = result.get("intent", "?")
    conf = result.get("intent_confidence", 0)
    table.add_row("Intent", f"{intent} ({conf:.0%})")

    assets = result.get("resolved_assets") or {}
    table.add_row("Domain", assets.get("domain", "N/A"))

    agents = []
    if result.get("genie_results"):
        agents.append("genie")
    if result.get("multi_tool_results"):
        agents.append("multi_tool")
    table.add_row("Agents Used", ", ".join(agents) if agents else "none (direct answer)")

    warnings = result.get("warnings", [])
    table.add_row("Warnings", str(len(warnings)))

    if duration_s is not None:
        table.add_row("Total Duration", f"{duration_s:.1f}s")

    answer = result.get("final_answer", "")
    table.add_row("Answer Length", f"{len(answer)} chars")

    console.print()
    console.print(table)
    console.print()


print("Rich flow logging utilities initialized")

# COMMAND ---------- [markdown]
# ## SQL Helper
#
# Executes SQL via the Databricks SDK Statement Execution API — works in both notebooks and Model Serving (no Spark required).

# COMMAND ----------

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
        column_meta = []
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

# COMMAND ---------- [markdown]
# ## Prompt Management (P1)
#
# Loads prompts from `ai_ops.agent_instructions` with a 5-minute cache. Falls back to hardcoded prompts if the table isn't ready.

# COMMAND ----------

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

# COMMAND ---------- [markdown]
# ## Short-term Memory (P0)
#
# Delta-based conversation checkpoints for multi-turn sessions. Saves/loads state from `ai_ops.conversations`.

# COMMAND ----------

def _save_checkpoint(thread_id, state_data):
    """Save conversation checkpoint to Delta table."""
    try:
        checkpoint_id = hashlib.md5(f"{thread_id}:{time.time()}".encode()).hexdigest()[:16]
        state_json = json.dumps(state_data, default=str).replace("'", "''")
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

# COMMAND ---------- [markdown]
# ## Long-term Memory (P2)
#
# User preferences and facts stored in `ai_ops.user_memory` for personalized responses across sessions.

# COMMAND ----------

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
            WHEN MATCHED AND s.confidence >= t.confidence THEN UPDATE SET t.memory_value = s.memory_value,
                t.memory_type = s.memory_type, t.confidence = s.confidence, t.updated_at = s.updated_at
            WHEN NOT MATCHED THEN INSERT *
        """)
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


def _extract_implicit_signals(user_id, question, intent, resolved_assets=None):
    """Infer user preferences from query patterns (implicit signals).

    Runs on non-conversational intents to pick up repeated filter patterns
    (region, product_line, domain) that reveal the user's focus areas.
    Saved at confidence=0.6 so explicit facts (1.0) are never downgraded.
    """
    if not user_id or user_id == "anonymous":
        return
    try:
        _llm = ChatDatabricks(endpoint=MODEL_ENDPOINT, temperature=0)
        domain = (resolved_assets or {}).get("domain", "unknown")
        prompt = f"""Analyze this analytics query and extract implicit user preferences.

Query: {question}
Intent: {intent}
Domain: {domain}

Extract ONLY signals clearly present in the query:
- region/location filter  → key: preferred_region
- product line focus      → key: preferred_product_lines
- time period filter      → key: preferred_time_period

Rules:
- Do NOT extract name, role, or anything requiring an explicit personal statement
- Skip generic filters that carry no personal preference signal

Respond in JSON ONLY:
{{"signals": [{{"key": "preferred_region", "value": "Singapore", "type": "preference"}}]}}
If no clear signals: {{"signals": []}}"""

        response = _llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        parsed = json.loads(raw)
        for sig in parsed.get("signals", []):
            key = sig.get("key", "").strip()
            value = sig.get("value", "").strip()
            mem_type = sig.get("type", "preference")
            if key and value:
                # Lower confidence than explicit facts — MERGE will not overwrite
                # higher-confidence entries already in user_memory
                _save_user_memory(user_id, key, value, mem_type, confidence=0.6)
    except Exception:
        pass

# COMMAND ---------- [markdown]
# ## Episodic Memory (P2)
#
# Logs interactions to `ai_ops.episodic_memory` and retrieves lessons from similar past queries for continuous improvement.

# COMMAND ----------

def _generate_lesson_learned(question, intent, result, outcome):
    """Use the LLM to distill a 1-2 sentence actionable lesson from an interaction.

    Only generates for non-conversational intents. Returns None when there is
    nothing meaningful to record (e.g. pure greetings, or the LLM signals NULL).
    """
    if intent == "conversational":
        return None

    context_parts = []
    genie = (result.get("genie_results") or {})
    multi_tool = (result.get("multi_tool_results") or {})

    if genie.get("sql"):
        context_parts.append(f"Genie SQL used: {genie['sql'][:300]}")
    if genie.get("error"):
        context_parts.append(f"Genie error: {genie['error'][:200]}")
    if genie.get("status") == "failed":
        failed_spaces = ", ".join(
            a.get("display_name", a.get("space_id", "?"))
            for a in genie.get("attempts", [])
        )
        if failed_spaces:
            context_parts.append(f"Genie spaces tried (all failed): {failed_spaces}")
    if multi_tool.get("docs"):
        context_parts.append(f"Documents retrieved: {len(multi_tool['docs'])} doc(s)")
    if multi_tool.get("error"):
        context_parts.append(f"Document retrieval error: {multi_tool['error'][:200]}")
    if result.get("warnings"):
        context_parts.append(f"Warnings: {'; '.join(result['warnings'][:3])}")

    if not context_parts:
        return None

    context = "\n".join(context_parts)

    prompt = f"""You are a learning system for an AI analytics agent. Analyze this interaction and write ONE concise, actionable lesson (1-2 sentences max) that would help the agent handle similar questions better in the future.

Question: {question}
Outcome: {outcome}
What happened:
{context}

Rules:
- Focus on data schema insights (column names, table structure), routing decisions, or query patterns
- If outcome is "failed": explain what went wrong and what to try instead
- If outcome is "success": note what worked (especially useful SQL patterns or schema facts)
- Be specific and actionable — not generic advice like "check the schema"
- If there is genuinely no meaningful lesson, respond with exactly: NULL

Lesson:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        lesson = response.content.strip()
        if not lesson or lesson.upper() == "NULL":
            return None
        # Strip any accidental "Lesson:" prefix the LLM may echo back
        if lesson.lower().startswith("lesson:"):
            lesson = lesson[7:].strip()
        return lesson[:500]
    except Exception:
        return None


def _save_episodic_memory(thread_id, user_id, question, intent, domain, agents_used,
                          outcome="success", lesson_learned=None):
    """Log this interaction to episodic_memory for continuous learning."""
    try:
        episode_id = hashlib.md5(f"{thread_id}:{question}:{time.time()}".encode()).hexdigest()[:20]
        agents_sql = ", ".join([f"'{a}'" for a in agents_used])
        q_esc = question.replace("'", "''")
        lesson_sql = (
            f"'{lesson_learned.replace(chr(39), chr(39)*2).replace(chr(10), " ").replace(chr(13), "")}'"
            if lesson_learned else "NULL"
        )
        _run_sql(f"""
            INSERT INTO {CATALOG}.ai_ops.episodic_memory
            (episode_id, thread_id, user_id, question, intent, domain,
             agents_used, outcome, lesson_learned, created_at)
            VALUES ('{episode_id}', '{thread_id}', '{user_id}', '{q_esc}',
                    '{intent}', '{domain}', ARRAY({agents_sql}),
                    '{outcome}', {lesson_sql}, current_timestamp())
        """)
    except Exception as _e:
        print(f"  [episodic] save failed: {_e}")


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

# COMMAND ---------- [markdown]
# ## Tool Registry (P2)
#
# Loads agent capabilities from `ai_ops.agent_capabilities` for semantic routing decisions.

# COMMAND ----------

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

# COMMAND ---------- [markdown]
# ## Agent State & LLM
#
# The `AgentState` TypedDict defines all fields that flow through the LangGraph nodes.

# COMMAND ----------

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
    episodic_lessons: Optional[list]


llm = ChatDatabricks(endpoint=MODEL_ENDPOINT, temperature=0.1, max_tokens=2000)
print(f"LLM initialized: {MODEL_ENDPOINT}")

# COMMAND ---------- [markdown]
# ## Node 1: Classify Intent
#
# Classifies user questions into: `simple_kpi`, `document_lookup`, or `conversational`.
# Also resolves short follow-up questions using conversation history.

# COMMAND ----------

# @mlflow.trace(name="classify_intent", span_type=SpanType.CHAIN)
def classify_intent(state):
    question = state["user_question"]

    messages = state.get("messages", [])
    state.setdefault("warnings", [])

    if len(messages) > 1 and len(question.split()) <= 10:
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

    # ── Context-synthesis heuristic ──────────────────────────────────────
    # When the conversation already contains data-bearing assistant answers
    # AND the current question asks for comparisons / insights / synthesis
    # from those prior results, classify as conversational immediately so
    # the LLM answers from conversation history (no agent round-trip).
    if len(messages) > 2:
        _ctx_ref_signals = [
            "based on those", "based on the above", "from those results",
            "from our previous", "from our conversation", "compare those",
            "compare the above", "looking at both", "considering both",
            "given those", "given the above", "from what we discussed",
            "those two results", "previous results", "earlier results",
            "what we\'ve seen", "what can we conclude", "summarize our",
            "insights from", "combine those", "cross-reference",
            "based on those two", "from the data above", "earlier data",
            "from all of our previous", "from our earlier", "deduce from",
        ]
        _has_prior_data = any(
            m.get("role") == "assistant" and len(m.get("content", "")) > 100
            for m in messages[:-1]
        )
        if _has_prior_data and any(sig in question.lower() for sig in _ctx_ref_signals):
            state["intent"] = "conversational"
            state["intent_confidence"] = 0.95
            state["needs_clarification"] = False
            state["warnings"].append("Context synthesis: answering from conversation history")
            return state
    # ────────────────────────────────────────────────────────────────────

    fallback_prompt = """You are an intent classifier for an insurance analytics system.
Classify the following question into exactly ONE category and provide a confidence score (0.0 to 1.0).

Categories:
- "simple_kpi": Simple KPI/metric questions that require NEW data retrieval (counts, totals, averages, trends by region/product/time)
- "document_lookup": Policy terms, coverage details, exclusions, procedures, document search
- "conversational": Greetings, introductions, personal statements, small talk, non-analytical messages, OR questions that ask for comparisons, insights, cross-analysis, or synthesis based on information already provided in the conversation history (e.g., "based on those results", "compare the above", "what can we conclude from our discussion", "looking at both sets of data")

IMPORTANT: If the user is referencing data or results from earlier in the conversation and asking for analysis, comparison, or insights from that existing information, classify as "conversational" — the system can answer from conversation context without querying new data.

Conversation history:
{history}

Question: {question}

Respond in JSON format ONLY:
{{"intent": "<category>", "confidence": <float>, "missing_filters": []}}

If the question is ambiguous or missing key filters (like region, time period, product), list them in missing_filters."""

    user_id = state.get("user_id") or "default"
    memory_context = ""
    user_mem = _load_user_memory(user_id)
    if user_mem:
        prefs = "; ".join([f"{k}={v}" for k, v in user_mem.items()])
        memory_context = f"\nUser preferences: {prefs}"

    # Build conversation history summary for the classifier
    _history_lines = []
    for m in messages[:-1]:
        _role = m.get("role", "user")
        _content = m.get("content", "")[:300]
        if _role in ("user", "assistant") and _content:
            _history_lines.append(f"{_role}: {_content}")
    _history_text = "\n".join(_history_lines[-6:]) if _history_lines else "No prior conversation."

    prompt_template = _get_prompt("supervisor", "classify_intent", fallback_prompt)
    try:
        prompt = prompt_template.format(question=question, history=_history_text)
    except (KeyError, IndexError):
        prompt = fallback_prompt.replace("{question}", question).replace("{history}", _history_text)
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

    # Heuristic override: catch greetings the LLM misclassified with low confidence
    if intent == "simple_kpi" and confidence < 0.7:
        greet_signals = ["hi ", "hello", "hey ", "i'm ", "i am ", "my name", "nice to meet"]
        if any(sig in question.lower() for sig in greet_signals):
            intent = "conversational"

    state["intent"] = intent
    state["intent_confidence"] = confidence
    state["needs_clarification"] = confidence < 0.6 or len(missing_filters) > 0
    state["warnings"] = state.get("warnings", [])

    if missing_filters:
        state["_missing_filters"] = missing_filters

    return state


# COMMAND ---------- [markdown]
# ## Node 2: Clarify or Disambiguate
#
# Activated when intent confidence is low or required filters are missing. Attempts to infer from conversation context before asking the user.

# COMMAND ----------

# @mlflow.trace(name="clarify_or_disambiguate", span_type=SpanType.CHAIN)
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
        state["warnings"].append(f"Note: The question may benefit from clarification \u2014 {clarification}")

    state["needs_clarification"] = False
    return state

# COMMAND ---------- [markdown]
# ## Default Assets Fallback
#
# Rule-based asset resolution used when the Vector Search Context Index is not available.
# Only resolves the two asset types supported by workers: `genie_space` and `document_index`.

# COMMAND ----------

def _get_default_assets(intent="simple_kpi"):
    default_space_id = "01f1272d4ba6144ba75d868762f1925d"
    return {
        "domain": "claims",
        "genie_space": default_space_id,
        "genie_spaces": [
            {"space_id": default_space_id, "domain": "claims",
             "display_name": "Claims Analytics Space", "score": 1.0,
             "endorsement": "endorsed"},
        ],
        "document_indexes": [f"{CATALOG}.bronze.policy_documents"],
        "doc_vs_index": f"{CATALOG}.ai_ops.policy_docs_vs",
        "all_assets": [],
        "endorsement_info": {},
    }

# COMMAND ---------- [markdown]
# ## Node 3: Resolve Assets via Context Index
#
# Uses Vector Search to semantically match the user question to available data assets (Genie Spaces, Document Indexes). Endorsed assets are prioritized. The `metadata` column carries worker-specific config (e.g. `vs_index` for document indexes).

# COMMAND ----------

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
            "genie_spaces": [
                {"space_id": g["asset_id"], "domain": g["domain"],
                 "display_name": g["display_name"], "score": g.get("score", 0),
                 "endorsement": g.get("endorsement_level", "standard")}
                for g in genie_spaces
            ],
            "document_indexes": [d["asset_id"] for d in doc_indexes],
            "doc_vs_index": doc_vs_index,
            "all_assets": assets,
            "endorsement_info": {a["asset_id"]: a["endorsement_level"] for a in assets},
        }
    except Exception as e:
        state["resolved_assets"] = _get_default_assets(intent)
        state["warnings"].append("Context Index not ready \u2014 using rule-based asset resolution")

    # Fetch episodic lessons now that intent + domain are known.
    # Stored in state so route_by_intent and route_to_genie can use them
    # without re-querying the DB.
    _ep_intent = state.get("intent", "unknown")
    _ep_domain = (state.get("resolved_assets") or {}).get("domain", "unknown")
    state["episodic_lessons"] = _get_episodic_lessons(_ep_intent, _ep_domain)
    return state

# COMMAND ---------- [markdown]
# ## Scoped Context Index Lookup & Asset Feedback
#
# Worker agents use scoped lookups to discover additional assets within their domain. Feedback is recorded for governance improvement.

# COMMAND ----------

def _scoped_context_index_lookup(query_text, domain, asset_types=None, num_results=5):
    """Worker-scoped Context Index lookup within a single domain."""
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
            if asset.get("domain", "").lower() == domain.lower():
                if asset_types is None or asset.get("asset_type") in asset_types:
                    assets.append(asset)
        assets.sort(key=lambda a: (0 if a.get("endorsement_level") == "endorsed" else 1, -a.get("score", 0)))
        return assets
    except Exception:
        return []


def _record_asset_feedback(agent_name, domain, feedback_type, details, state):
    """Record feedback when a worker agent discovers missing or useful assets."""
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
        pass

# COMMAND ---------- [markdown]
# ## Node 4: Genie Agent (Multi-Space)
#
# Iterates through ranked Genie Spaces resolved from the Context Index (`resolved_assets.genie_spaces`). Tries the best-matching space first and falls back to the next space if it fails. The `_query_genie_space` helper encapsulates a single Genie API call for reuse.

# COMMAND ----------

def _get_genie_client():
    """Return a WorkspaceClient configured for Genie API calls.

    In Model Serving, uses on-behalf-of (OBO) user credentials so that
    Genie queries execute under the calling user's identity — inheriting
    their UC table permissions automatically. No manual SP grants needed.
    Falls back to default credentials (user PAT) in notebooks.
    """
    from databricks.sdk import WorkspaceClient
    try:
        from databricks_ai_bridge import ModelServingUserCredentials
        return WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
    except Exception:
        return WorkspaceClient()


def _query_genie_space(w, space_id, question):
    """Call the Genie API for a single space and return a result dict."""
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
            return {
                "sql": sql_query,
                "result_summary": result_data or "No result text",
                "status": "success",
            }
        else:
            return {"error": f"Genie status: {msg.status}", "status": "failed"}
    except Exception as e:
        return {"error": str(e)[:200], "status": "failed"}


@mlflow.trace(name="route_to_genie", span_type=SpanType.TOOL)
def route_to_genie(state):
    question = state["user_question"]

    # Inject schema hints from successful past queries into the Genie question
    _lessons = state.get("episodic_lessons") or []
    _hints = [
        l.get("lesson_learned") for l in _lessons
        if l.get("outcome") == "success" and l.get("lesson_learned")
    ]
    if _hints:
        question = question + "\n[Schema hints from past queries: " + " | ".join(_hints[:2]) + "]"

    # OBO auth: in Model Serving, Genie calls run as the calling user
    # (inherits their UC table permissions). No manual SP grants needed.
    w = _get_genie_client()

    genie_spaces = state.get("resolved_assets", {}).get("genie_spaces", [])
    if not genie_spaces:
        single = state.get("resolved_assets", {}).get("genie_space")
        if single:
            genie_spaces = [{"space_id": single, "domain": "unknown",
                             "display_name": "Default"}]

    if not genie_spaces:
        state["genie_results"] = {"error": "No Genie Spaces resolved from Context Index",
                                  "status": "failed"}
        state["warnings"].append("Genie Space not found in resolved assets \u2014 check Context Index")
        return state

    attempts = []
    for space_info in genie_spaces:
        space_id = space_info["space_id"]
        result = _query_genie_space(w, space_id, question)
        attempt = {
            **result,
            "space_id": space_id,
            "domain": space_info.get("domain"),
            "display_name": space_info.get("display_name"),
        }
        attempts.append(attempt)
        if result["status"] == "success" and result.get("sql"):
            break

    best = next((a for a in attempts if a["status"] == "success" and a.get("sql")), None)
    if best is None:
        best = next((a for a in attempts if a["status"] == "success"), attempts[-1])

    state["genie_results"] = {**best, "attempts": attempts}

    if best.get("status") != "success":
        state["warnings"].append("Genie query did not complete on any resolved space")

    domain = state.get("resolved_assets", {}).get("domain", "claims")
    if best.get("status") != "success" or not best.get("sql"):
        extra = _scoped_context_index_lookup(
            question, domain, asset_types=["genie_space"], num_results=3,
        )
        if extra:
            state["genie_results"]["ci_enrichment"] = [
                {"asset_id": a["asset_id"], "display_name": a["display_name"],
                 "asset_type": a["asset_type"]} for a in extra
            ]
        if best.get("status") != "success":
            failed_spaces = ", ".join(a.get("display_name", a["space_id"]) for a in attempts)
            _record_asset_feedback("genie", domain, "genie_query_failed",
                                   f"Genie could not answer on [{failed_spaces}]: {question[:150]}", state)

    return state

# COMMAND ---------- [markdown]
# ## Node 5: Multi-Tool Agent (RAG)
#
# Performs Vector Search RAG over policy documents for document lookup queries. The VS index name is dynamically resolved from the Context Index (`resolved_assets.doc_vs_index`).

# COMMAND ----------

@mlflow.trace(name="route_to_multi_tool", span_type=SpanType.TOOL)
def route_to_multi_tool(state):
    """Vector Search RAG over policy documents using the VS index resolved from Context Index."""
    question = state["user_question"]

    # Inject document-retrieval hints from successful past queries
    _lessons = state.get("episodic_lessons") or []
    _hints = [
        l.get("lesson_learned") for l in _lessons
        if l.get("outcome") == "success" and l.get("lesson_learned")
    ]
    if _hints:
        question = question + " " + " ".join(_hints[:2])

    results = {}

    doc_vs_index = state.get("resolved_assets", {}).get("doc_vs_index")
    if not doc_vs_index:
        results["docs"] = []
        results["error"] = "No document VS index resolved from Context Index"
        results["status"] = "failed"
        state["warnings"].append("Document VS index not found in resolved assets \u2014 check Context Index")
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

# COMMAND ---------- [markdown]
# ## Node 6: Compose Final Answer
#
# Synthesizes results from all agents into a natural, conversational response. Incorporates user memory for personalization and episodic lessons for improvement.

# COMMAND ----------

@mlflow.trace(name="compose_answer", span_type=SpanType.CHAIN)
def compose_answer(state):
    question = state["user_question"]
    intent = state.get("intent", "unknown")
    assets = state.get("resolved_assets", {})
    genie = state.get("genie_results", {})
    multi_tool = state.get("multi_tool_results", {})
    clarification = state.get("clarification_message", "")

    context_parts = []

    if genie and genie.get("status") == "success":
        space_name = genie.get("display_name", "Genie Agent")
        context_parts.append(f"**{space_name}:** SQL: {genie.get('sql', 'N/A')}\nResult: {genie.get('result_summary', 'N/A')}")

    if multi_tool and multi_tool.get("docs"):
        doc_text = "\n".join([f"- {d['title']}: {d['content'][:200]}..." for d in multi_tool["docs"][:3]])
        context_parts.append(f"**Documents:**\n{doc_text}")

    # ── Conversation history injection ───────────────────────────────
    # When no agent results are available (e.g. context-synthesis questions
    # classified as conversational), inject prior conversation Q&A so the
    # LLM can answer from history rather than saying "no data available".
    if not context_parts:
        _msgs = state.get("messages", [])
        _history_pairs = []
        for m in _msgs[:-1]:  # exclude the current question
            _role = m.get("role", "")
            _content = m.get("content", "")
            if _role in ("user", "assistant") and _content:
                _history_pairs.append(f"**{_role.title()}:** {_content[:800]}")
        if _history_pairs:
            context_parts.append(
                "**Conversation History (use this data to answer):**\n"
                + "\n\n".join(_history_pairs)
            )
    # ────────────────────────────────────────────────────────────────────

    domain = assets.get("domain", "unknown") if assets else "unknown"
    lessons = _get_episodic_lessons(intent, domain)
    if lessons:
        lesson_text = "\n".join([f"- [{l.get('outcome','?')}] {l.get('lesson_learned','')}" for l in lessons if l.get("lesson_learned")])
        if lesson_text:
            context_parts.append(f"**Past experience (internal):**\n{lesson_text}")

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

    warning_text = "\n\nCRITICAL RULE: NEVER include a 'Warnings & Limitations' section, warnings, caveats, disclaimers, or notes about data limitations in your answer. These are handled separately by the UI. Your answer must end after the main content \u2014 do not append any warning/limitation section."

    clarification_text = ""
    if clarification:
        clarification_text = f"\n\nAlso note this clarification for the user: {clarification}"

    compose_prompt = f"""Answer this question naturally and conversationally: {question}

Agent Results:
{context}

Instructions:
- Answer like a knowledgeable insurance colleague \u2014 direct, natural, and helpful.
- Lead with the key insight or answer, not an introduction or preamble.
- Include specific numbers and data from the results naturally in your sentences.
- When the answer warrants detail (trends, analysis, comparisons), provide thorough explanations.
- For simple KPI questions, keep it brief. For complex analysis, be detailed and insightful.
- Use markdown when it helps readability (e.g., tables for comparisons, bold for key figures).
- If you know the user's name from the User preferences section, address them by name naturally.
- If the context includes Conversation History, use the data and numbers from prior answers to synthesize your response. Do not say you lack data \u2014 the prior answers ARE your data source.{warning_text}{clarification_text}"""

    response = llm.invoke([HumanMessage(content=compose_prompt)])
    state["final_answer"] = response.content
    return state


# COMMAND ---------- [markdown]
# ## Routing Logic
#
# Conditional edge functions that determine the LangGraph traversal path.

# COMMAND ----------

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


    # Lesson-driven routing: use past failures to override default routing
    lessons = state.get("episodic_lessons") or []
    if lessons and intent == "simple_kpi" and has_genie:
        failed_genie = any(
            l.get("outcome") == "failed" and
            "genie" in (l.get("lesson_learned") or "").lower()
            for l in lessons
        )
        if failed_genie:
            return "multi_tool"  # past genie failure on this domain — fall back

    if intent == "conversational":
        return "compose_answer"
    elif intent == "document_lookup":
        return "multi_tool"
    elif intent == "simple_kpi":
        return "genie" if has_genie else "multi_tool"
    return "multi_tool"

# COMMAND ---------- [markdown]
# ## Build LangGraph
#
# Compiles the full state graph:
#
# ```
# START -> classify_intent -> [clarify] -> resolve_assets -> [genie | multi_tool | compose_answer] -> compose_answer -> END
# ```

# COMMAND ----------

workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", _with_logging("classify_intent", classify_intent))
workflow.add_node("clarify_or_disambiguate", _with_logging("clarify_or_disambiguate", clarify_or_disambiguate))
workflow.add_node("resolve_assets", _with_logging("resolve_assets", resolve_assets_with_context_index))
workflow.add_node("genie", _with_logging("genie", route_to_genie))
workflow.add_node("multi_tool", _with_logging("multi_tool", route_to_multi_tool))
workflow.add_node("compose_answer", _with_logging("compose_answer", compose_answer))

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

print(f"Graph compiled with nodes: {list(graph.nodes)}")

# # COMMAND ----------

# from IPython.display import Image, display

# display(Image(graph.get_graph().draw_mermaid_png()))

# COMMAND ---------- [markdown]
# ## ResponsesAgent Wrapper
#
# The `SupervisorResponsesAgent` wraps the LangGraph in an MLflow `ResponsesAgent` for Model Serving deployment. It handles message parsing, memory management, and custom I/O.

# COMMAND ----------

class SupervisorResponsesAgent(ResponsesAgent):
    def __init__(self):
        # Explicit per-thread conversation memory keyed by thread_id.
        # Provides fast in-memory history; Delta checkpoint serves as durable fallback
        # (e.g. after a kernel restart). Mirrors the pattern in Genie_deepresearch.ipynb.
        self._conversation_history: dict[str, list[dict]] = {}

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        user_message = None
        new_msgs = []
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
            new_msgs.append({"role": role, "content": content})
            if role == "user":
                user_message = content

        if not user_message:
            return ResponsesAgentResponse(
                output=[self.create_text_output_item(text="Please ask a question.", id="msg_empty")]
            )

        custom_inputs = {}
        if hasattr(request, "custom_inputs") and request.custom_inputs:
            custom_inputs = request.custom_inputs if isinstance(request.custom_inputs, dict) else {}
        thread_id = custom_inputs.get("thread_id")
        user_id = custom_inputs.get("user_id")

        # Tag MLflow trace with session ID so all turns in a conversation are grouped
        # under one session in the MLflow UI (same pattern as Genie_deepresearch.ipynb).
        if thread_id:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": thread_id})

        # ---- Build full message list: in-memory history + new messages ----
        history = self._conversation_history.get(thread_id, []) if thread_id else []

        # Fall back to Delta checkpoint when in-memory history is empty (kernel restart, etc.)
        if not history and thread_id:
            prior_state = _load_checkpoint(thread_id)
            if prior_state and prior_state.get("messages"):
                history = prior_state["messages"]

        all_messages = history + new_msgs

        # Apply context window limit to keep prompts manageable
        if len(all_messages) > MAX_MESSAGES:
            all_messages = all_messages[-MAX_MESSAGES:]

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

        log_flow_start(user_message, thread_id, user_id)
        _flow_start = time.time()
        result = graph.invoke(initial_state)
        _flow_duration = time.time() - _flow_start
        answer = result.get("final_answer", "I was unable to process your question.")

        # ---- Update in-memory conversation history for this thread ----
        if thread_id:
            updated_history = history + new_msgs + [{"role": "assistant", "content": answer}]
            self._conversation_history[thread_id] = updated_history

        checkpoint_id = None
        if thread_id:
            checkpoint_data = {
                "messages": all_messages + [{"role": "assistant", "content": answer}],
                "intent": result.get("intent"),
                "domain": result.get("resolved_assets", {}).get("domain") if result.get("resolved_assets") else None,
            }
            checkpoint_id = _save_checkpoint(thread_id, checkpoint_data)

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

        # Auto-generate lesson_learned so episodic memory drives real learning.
        # Skips conversational intents (no schema/routing lesson to extract).
        lesson = _generate_lesson_learned(
            question=user_message,
            intent=result.get("intent", "unknown"),
            result=result,
            outcome=ep_outcome,
        )

        _save_episodic_memory(
            thread_id=thread_id or "anonymous",
            user_id=user_id or "anonymous",
            question=user_message,
            intent=result.get("intent", "unknown"),
            domain=ep_domain,
            agents_used=agents_used,
            outcome=ep_outcome,
            lesson_learned=lesson,
        )

        intent = result.get("intent", "")
        if intent in ("conversational", "unknown", "greeting"):
            # Explicit fact extraction: user stated personal info directly
            _extract_and_save_user_facts(user_id or "default", user_message, answer)
        elif intent in ("simple_kpi", "document_lookup"):
            # Implicit signal extraction: infer preferences from query filters
            _extract_implicit_signals(
                user_id or "default", user_message, intent,
                resolved_assets=result.get("resolved_assets"),
            )

        nodes_executed = [n for n in [
            "classify_intent",
            "clarify_or_disambiguate" if result.get("clarification_message") else None,
            "resolve_assets",
            "genie" if result.get("genie_results") else None,
            "multi_tool" if result.get("multi_tool_results") else None,
            "compose_answer",
        ] if n is not None]

        resolved = result.get("resolved_assets", {})
        domain = resolved.get("domain", "unknown") if resolved else "unknown"

        agent_details = {}
        if result.get("genie_results"):
            gr = result["genie_results"]
            agent_details["genie"] = {
                "status": gr.get("status", "unknown"),
                "space_id": gr.get("space_id"),
                "display_name": gr.get("display_name"),
                "sql": gr.get("sql", "")[:120] if gr.get("sql") else None,
                "row_count": gr.get("row_count"),
                "spaces_tried": len(gr.get("attempts", [])),
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

        metadata_json = json.dumps(custom_outputs)

        log_flow_end(result, duration_s=_flow_duration)

        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(text=answer, id="msg_answer"),
                self.create_text_output_item(text=metadata_json, id="msg_metadata"),
            ],
            custom_outputs=custom_outputs,
        )


agent = SupervisorResponsesAgent()
set_model(agent)
print("SupervisorResponsesAgent created and registered with set_model()")
