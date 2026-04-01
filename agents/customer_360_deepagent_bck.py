# Databricks notebook source
# COMMAND ---------- [markdown]
# # AIA Customer 360 — DeepAgent Implementation
#
# This module reimplements the Customer 360 Supervisor Agent using **LangChain Deep Agents**
# (`deepagents` package). Instead of a hand-wired LangGraph `StateGraph`, it exposes the
# agent's capabilities as **tools** that the DeepAgent orchestrates autonomously via its
# built-in planning/reflection loop.
#
# **Architecture:**
# - **Supervisor DeepAgent** with planning (`write_todos`), `FilesystemBackend`, and subagents
# - **Genie Subagent** — isolated text-to-SQL specialist (delegates Genie Space queries)
# - **Document RAG Subagent** — isolated policy document retrieval specialist
# - **Custom tools**: `classify_intent`, `resolve_assets`, `query_genie_space`,
#   `search_policy_documents`, `run_sql`, `load_user_memory`, `get_episodic_lessons`
#
# **Key differences from the LangGraph version (`customer_360.py`):**
# - No manual graph wiring — the DeepAgent decides tool ordering via planning
# - Subagents provide isolated context for Genie and Document RAG workloads
# - Built-in planning middleware replaces the fixed classify→resolve→route→compose pipeline
# - Same Databricks integrations: Unity Catalog, Vector Search, Genie, SQL Warehouse, MLflow

# COMMAND ---------- [markdown]
# ## Install Dependencies

# COMMAND ----------

# %pip install deepagents "mlflow>=3.1" "databricks-agents>=1.0.0" "pydantic>=2" \
#     langchain-core langchain-anthropic databricks-langchain databricks-vectorsearch \
#     databricks-sdk databricks-ai-bridge rich --upgrade

# COMMAND ----------

# dbutils.library.restartPython()  # Uncomment on Databricks

# COMMAND ---------- [markdown]
# ## Imports & Configuration

# COMMAND ----------

import mlflow
import json
import time
import hashlib
import os
from typing import Optional
from langchain_core.tools import tool
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
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
MAX_MESSAGES = 7

print(f"Catalog: {CATALOG}")
print(f"LLM Endpoint: {MODEL_ENDPOINT}")
print(f"VS Index: {VS_INDEX}")

# COMMAND ---------- [markdown]
# ## Rich Logging Utilities

# COMMAND ----------

console = Console()


def log_deep_agent_start(question, thread_id=None, user_id=None):
    parts = Text()
    parts.append("AIA DeepAgent — Flow Started\n\n", style="bold bright_green")
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


def log_tool_call(tool_name, details=None):
    console.print(f"  [bold cyan]>> Tool: {tool_name}[/]", end="")
    if details:
        console.print(f"  [dim]{details}[/]")
    else:
        console.print()


def log_deep_agent_end(answer, duration_s=None):
    table = Table(
        title="DeepAgent Flow Complete",
        box=box.ROUNDED,
        border_style="bright_green",
        title_style="bold bright_green",
        show_lines=False,
    )
    table.add_column("Field", style="dim", width=22)
    table.add_column("Value", style="bright_white")
    if duration_s is not None:
        table.add_row("Total Duration", f"{duration_s:.1f}s")
    table.add_row("Answer Length", f"{len(answer)} chars")
    console.print()
    console.print(table)
    console.print()


# COMMAND ---------- [markdown]
# ## SQL Helper

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
    if (
        response.status
        and response.status.state
        and response.status.state.value == "SUCCEEDED"
    ):
        columns = []
        column_meta = []
        try:
            manifest = response.manifest
            if manifest:
                cols_obj = getattr(manifest, "columns", None)
                if not cols_obj:
                    schema = getattr(manifest, "schema", None)
                    if schema:
                        cols_obj = getattr(schema, "columns", None)
                if cols_obj:
                    columns = [c.name for c in cols_obj]
                    column_meta = [
                        {
                            "name": c.name,
                            "type_name": (
                                getattr(c, "type_text", None)
                                or getattr(c, "type_name", None)
                                or "STRING"
                            ).upper(),
                        }
                        for c in cols_obj
                    ]
        except (AttributeError, TypeError):
            pass

        if not columns:
            import re

            m = re.match(
                r"\s*SELECT\s+(.+?)\s+FROM\s",
                sql_statement,
                re.IGNORECASE | re.DOTALL,
            )
            if m:
                col_str = m.group(1)
                columns = [
                    c.strip().split(".")[-1].split(" ")[-1]
                    for c in col_str.split(",")
                ]
                column_meta = [{"name": c, "type_name": "STRING"} for c in columns]

        rows = []
        if response.result and response.result.data_array:
            for row in response.result.data_array[:max_rows]:
                if columns and len(columns) == len(row):
                    rows.append(dict(zip(columns, row)))
                else:
                    rows.append({f"col_{i}": v for i, v in enumerate(row)})
        return {
            "columns": columns,
            "column_meta": column_meta,
            "rows": rows,
            "row_count": len(rows),
        }
    else:
        error_msg = ""
        if response.status and response.status.error:
            error_msg = response.status.error.message
        raise Exception(f"SQL failed: {error_msg}")


# COMMAND ---------- [markdown]
# ## Memory & Episodic Helpers (same as customer_360.py)

# COMMAND ----------

_prompt_cache = {}
_prompt_cache_ts = 0


def _load_prompts():
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
    prompts = _load_prompts()
    key = f"{agent_id}:{scope}"
    return prompts.get(key, fallback)


def _save_checkpoint(thread_id, state_data):
    try:
        checkpoint_id = hashlib.md5(
            f"{thread_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        state_json = json.dumps(state_data, default=str).replace("'", "''")
        safe_json = state_json.replace("\\", "\\\\")
        _run_sql(
            f"""
            INSERT INTO {CATALOG}.ai_ops.conversations
            (thread_id, checkpoint_id, state_json, created_at)
            VALUES ('{thread_id}', '{checkpoint_id}', '{safe_json}', current_timestamp())
        """
        )
        return checkpoint_id
    except Exception:
        return None


def _load_checkpoint(thread_id):
    try:
        result = _run_sql(
            f"""
            SELECT state_json FROM {CATALOG}.ai_ops.conversations
            WHERE thread_id = '{thread_id}'
            ORDER BY created_at DESC LIMIT 1
        """
        )
        if result["rows"]:
            return json.loads(result["rows"][0]["state_json"])
    except Exception:
        pass
    return None


_memory_cache = {}
_memory_cache_ts = 0


def _load_user_memory_internal(user_id):
    global _memory_cache, _memory_cache_ts
    cache_key = f"mem:{user_id}"
    if time.time() - _memory_cache_ts < 60 and cache_key in _memory_cache:
        return _memory_cache[cache_key]
    try:
        result = _run_sql(
            f"""
            SELECT memory_key, memory_value, memory_type, confidence
            FROM {CATALOG}.ai_ops.user_memory
            WHERE user_id = '{user_id}'
              AND (expires_at IS NULL OR expires_at > current_timestamp())
            ORDER BY confidence DESC
        """
        )
        memories = {r["memory_key"]: r["memory_value"] for r in result["rows"]}
        _memory_cache[cache_key] = memories
        _memory_cache_ts = time.time()
        return memories
    except Exception:
        return {}


def _save_user_memory(user_id, memory_key, memory_value, memory_type="preference", confidence=1.0):
    if not user_id or user_id == "anonymous":
        return
    try:
        key_esc = memory_key.replace("'", "''")
        val_esc = memory_value.replace("'", "''")
        _run_sql(
            f"""
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
        """
        )
        global _memory_cache, _memory_cache_ts
        _memory_cache_ts = 0
    except Exception:
        pass


def _save_episodic_memory(
    thread_id, user_id, question, intent, domain, agents_used,
    outcome="success", lesson_learned=None,
):
    try:
        episode_id = hashlib.md5(
            f"{thread_id}:{question}:{time.time()}".encode()
        ).hexdigest()[:20]
        agents_sql = ", ".join([f"'{a}'" for a in agents_used])
        q_esc = question.replace("'", "''")
        lesson_sql = (
            f"'{lesson_learned.replace(chr(39), chr(39)*2).replace(chr(10), ' ').replace(chr(13), '')}'"
            if lesson_learned
            else "NULL"
        )
        _run_sql(
            f"""
            INSERT INTO {CATALOG}.ai_ops.episodic_memory
            (episode_id, thread_id, user_id, question, intent, domain,
             agents_used, outcome, lesson_learned, created_at)
            VALUES ('{episode_id}', '{thread_id}', '{user_id}', '{q_esc}',
                    '{intent}', '{domain}', ARRAY({agents_sql}),
                    '{outcome}', {lesson_sql}, current_timestamp())
        """
        )
    except Exception as _e:
        print(f"  [episodic] save failed: {_e}")


def _get_episodic_lessons_internal(intent, domain, limit=3):
    try:
        result = _run_sql(
            f"""
            SELECT question, lesson_learned, outcome, user_rating
            FROM {CATALOG}.ai_ops.episodic_memory
            WHERE intent = '{intent}' AND domain = '{domain}'
              AND lesson_learned IS NOT NULL
            ORDER BY created_at DESC LIMIT {limit}
        """
        )
        return result["rows"]
    except Exception:
        return []


# COMMAND ---------- [markdown]
# ## DeepAgent Tool Definitions
#
# Each capability of the original LangGraph supervisor is exposed as a `@tool`-decorated
# function that the DeepAgent can call via its built-in tool-calling loop.

# COMMAND ----------

llm = ChatDatabricks(endpoint=MODEL_ENDPOINT, temperature=0.1, max_tokens=2000)


@tool
def classify_user_intent(question: str, conversation_history: str = "") -> str:
    """Classify an insurance analytics question into one of three intent categories.

    Use this tool FIRST for every user question to determine the right approach.

    Returns JSON with:
    - intent: "simple_kpi" (data/metrics queries), "document_lookup" (policy docs),
              or "conversational" (greetings, synthesis from prior context)
    - confidence: float 0.0-1.0
    - missing_filters: list of filters that could improve the query

    Args:
        question: The user's question to classify
        conversation_history: Prior conversation turns formatted as "role: content" lines
    """
    log_tool_call("classify_user_intent", f"question={question[:80]}...")

    prompt = f"""You are an intent classifier for an insurance analytics system.
Classify the following question into exactly ONE category and provide a confidence score (0.0 to 1.0).

Categories:
- "simple_kpi": Simple KPI/metric questions that require NEW data retrieval (counts, totals, averages, trends by region/product/time)
- "document_lookup": Policy terms, coverage details, exclusions, procedures, document search
- "conversational": Greetings, introductions, personal statements, small talk, non-analytical messages, OR questions that ask for comparisons, insights, cross-analysis, or synthesis based on information already provided in the conversation history

IMPORTANT: If the user is referencing data or results from earlier in the conversation and asking for analysis, comparison, or insights from that existing information, classify as "conversational".

Conversation history:
{conversation_history if conversation_history else 'No prior conversation.'}

Question: {question}

Respond in JSON format ONLY:
{{"intent": "<category>", "confidence": <float>, "missing_filters": []}}

If the question is ambiguous or missing key filters (like region, time period, product), list them in missing_filters."""

    from langchain_core.messages import HumanMessage

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        parsed = json.loads(raw)
        intent = parsed.get("intent", "simple_kpi")
        valid = ["simple_kpi", "document_lookup", "conversational"]
        if intent not in valid:
            intent = "simple_kpi"
        return json.dumps(parsed)
    except (json.JSONDecodeError, ValueError):
        return json.dumps(
            {"intent": "simple_kpi", "confidence": 0.5, "missing_filters": []}
        )


@tool
def resolve_data_assets(question: str) -> str:
    """Discover relevant data assets (Genie Spaces, Document Indexes) for a question.

    Uses Vector Search on the Context Index to semantically match the question
    to available data assets. Returns the best Genie Space IDs and document
    VS index names for downstream tools.

    Args:
        question: The user's question to match against available data assets
    """
    log_tool_call("resolve_data_assets", f"query={question[:80]}...")

    from databricks.sdk import WorkspaceClient

    try:
        w = WorkspaceClient()
        results = w.vector_search_indexes.query_index(
            index_name=VS_INDEX,
            columns=[
                "asset_type", "asset_id", "display_name", "text",
                "domain", "endorsement_level", "metadata",
            ],
            query_text=question,
            num_results=10,
        )

        assets = []
        for row in results.result.data_array:
            assets.append({
                "asset_type": row[0],
                "asset_id": row[1],
                "display_name": row[2],
                "text": row[3],
                "domain": row[4],
                "endorsement_level": row[5],
                "metadata": row[6] if len(row) > 6 else "{}",
                "score": float(row[7]) if len(row) > 7 else 0.0,
            })

        assets.sort(
            key=lambda a: (
                0 if a.get("endorsement_level") == "endorsed" else 1,
                -a.get("score", 0),
            )
        )

        domain_counts = {}
        for a in assets[:5]:
            d = a.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        primary_domain = (
            max(domain_counts, key=domain_counts.get) if domain_counts else "claims"
        )

        genie_spaces = [a for a in assets if a["asset_type"] == "genie_space"]
        doc_indexes = [a for a in assets if a["asset_type"] == "document_index"]

        doc_vs_index = None
        if doc_indexes:
            try:
                meta = json.loads(doc_indexes[0].get("metadata") or "{}")
                doc_vs_index = meta.get("vs_index")
            except (json.JSONDecodeError, TypeError):
                pass

        result = {
            "domain": primary_domain,
            "genie_spaces": [
                {
                    "space_id": g["asset_id"],
                    "domain": g["domain"],
                    "display_name": g["display_name"],
                    "score": g.get("score", 0),
                    "endorsement": g.get("endorsement_level", "standard"),
                }
                for g in genie_spaces
            ],
            "doc_vs_index": doc_vs_index,
            "total_assets_found": len(assets),
        }

        return json.dumps(result)

    except Exception as e:
        default_space_id = "01f1272d4ba6144ba75d868762f1925d"
        fallback = {
            "domain": "claims",
            "genie_spaces": [
                {
                    "space_id": default_space_id,
                    "domain": "claims",
                    "display_name": "Claims Analytics Space",
                    "score": 1.0,
                    "endorsement": "endorsed",
                }
            ],
            "doc_vs_index": f"{CATALOG}.ai_ops.policy_docs_vs",
            "total_assets_found": 0,
            "warning": f"Context Index unavailable, using defaults: {str(e)[:100]}",
        }
        return json.dumps(fallback)


@tool
def query_genie_space(space_id: str, question: str) -> str:
    """Execute a text-to-SQL query against a Databricks Genie Space.

    Use this for simple_kpi intent questions that need data retrieval (counts,
    totals, averages, trends). The Genie Space translates natural language to
    SQL and returns query results.

    Args:
        space_id: The Genie Space ID to query (from resolve_data_assets)
        question: The natural language analytics question
    """
    log_tool_call("query_genie_space", f"space={space_id[:20]}... q={question[:60]}...")

    try:
        from databricks.sdk import WorkspaceClient

        try:
            from databricks_ai_bridge import ModelServingUserCredentials
            client = WorkspaceClient(
                credentials_strategy=ModelServingUserCredentials()
            )
        except Exception:
            client = WorkspaceClient()

        agent = GenieAgent(
            genie_space_id=space_id,
            genie_agent_name="Genie",
            description="Genie Agent for text-to-SQL",
            include_context=True,
            client=client,
        )

        agent_result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        msgs = agent_result.get("messages", [])
        sql_query = next(
            (m.content for m in msgs if getattr(m, "name", "") == "query_sql"), None
        )
        result_text = next(
            (m.content for m in msgs if getattr(m, "name", "") == "query_result"), None
        )

        return json.dumps({
            "status": "success",
            "sql": sql_query,
            "result_summary": result_text or "No result text",
            "space_id": space_id,
        })
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "error": str(e)[:300],
            "space_id": space_id,
        })


@tool
def search_policy_documents(question: str, vs_index_name: str = "") -> str:
    """Search policy documents using Vector Search RAG.

    Use this for document_lookup intent questions about policy terms, coverage
    details, exclusions, procedures, or any document-based queries.

    Args:
        question: The question to search for in policy documents
        vs_index_name: The Vector Search index name (from resolve_data_assets).
                       Defaults to the standard policy docs index if empty.
    """
    if not vs_index_name:
        vs_index_name = f"{CATALOG}.ai_ops.policy_docs_vs"

    log_tool_call("search_policy_documents", f"index={vs_index_name}")

    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        vs_results = w.vector_search_indexes.query_index(
            index_name=vs_index_name,
            columns=["document_id", "title", "content", "document_type", "category"],
            query_text=question,
            num_results=5,
        )

        docs = [
            {
                "document_id": r[0],
                "title": r[1],
                "content": r[2][:500],
                "document_type": r[3],
                "category": r[4],
            }
            for r in vs_results.result.data_array
        ]

        return json.dumps({
            "status": "success",
            "docs": docs,
            "doc_count": len(docs),
        })
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "error": str(e)[:200],
            "docs": [],
        })


@tool
def load_user_memory(user_id: str) -> str:
    """Load stored user preferences and facts for personalized responses.

    Returns known preferences like name, preferred region, role, etc.

    Args:
        user_id: The user identifier to load memory for
    """
    log_tool_call("load_user_memory", f"user={user_id}")
    memories = _load_user_memory_internal(user_id)
    return json.dumps(memories) if memories else json.dumps({})


@tool
def get_episodic_lessons(intent: str, domain: str) -> str:
    """Retrieve lessons learned from past similar interactions.

    Returns actionable insights from previous queries with the same intent
    and domain — useful for improving Genie SQL queries or document retrieval.

    Args:
        intent: The classified intent (simple_kpi, document_lookup, conversational)
        domain: The data domain (e.g., claims, policies)
    """
    log_tool_call("get_episodic_lessons", f"intent={intent}, domain={domain}")
    lessons = _get_episodic_lessons_internal(intent, domain)
    return json.dumps(lessons) if lessons else json.dumps([])


@tool
def execute_sql_query(sql_statement: str) -> str:
    """Execute a SQL query against the Databricks SQL Warehouse.

    Use sparingly — prefer query_genie_space for analytics questions.
    This tool is for ad-hoc lookups like checking agent_capabilities or
    agent_instructions tables.

    Args:
        sql_statement: The SQL statement to execute
    """
    log_tool_call("execute_sql_query", f"sql={sql_statement[:80]}...")
    try:
        result = _run_sql(sql_statement, max_rows=20)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})


# COMMAND ---------- [markdown]
# ## Subagent Definitions
#
# Isolated specialists that the supervisor DeepAgent can delegate to.

# COMMAND ----------

GENIE_SUBAGENT = {
    "name": "genie-analyst",
    "description": (
        "Delegate data analytics questions that need SQL-based answers. "
        "This sub-agent will resolve the right Genie Space, execute the query, "
        "and return structured results with the SQL used. "
        "Give it the user's question and any resolved asset info."
    ),
    "system_prompt": """You are a data analytics specialist for an insurance company.

Your job is to answer KPI and metrics questions by:
1. First calling resolve_data_assets to find the right Genie Space
2. Then calling query_genie_space with the best space_id
3. If the first space fails, try the next one from the resolved assets
4. Optionally call get_episodic_lessons to get hints from past queries
5. Return the complete results including SQL, data, and any warnings

Always return your findings clearly with the SQL query used and the data results.""",
    "tools": [resolve_data_assets, query_genie_space, get_episodic_lessons],
}

DOCUMENT_RAG_SUBAGENT = {
    "name": "document-researcher",
    "description": (
        "Delegate policy document lookups, coverage questions, exclusion details, "
        "or procedure inquiries. This sub-agent searches the policy document "
        "knowledge base and returns relevant document excerpts."
    ),
    "system_prompt": """You are a policy document research specialist for an insurance company.

Your job is to find relevant policy documents by:
1. First calling resolve_data_assets to find the right document VS index
2. Then calling search_policy_documents with the index name
3. Optionally call get_episodic_lessons for retrieval hints
4. Return comprehensive document excerpts with titles and categories

Present findings clearly, citing document titles and relevant content sections.""",
    "tools": [resolve_data_assets, search_policy_documents, get_episodic_lessons],
}

# COMMAND ---------- [markdown]
# ## Supervisor System Prompt

# COMMAND ----------

SUPERVISOR_SYSTEM_PROMPT = """You are the AIA Customer 360 Supervisor Agent — an intelligent insurance analytics assistant built on Databricks.

You help insurance professionals with:
- **KPI & Metrics queries** (claims counts, premium trends, loss ratios by region/product/time)
- **Policy document lookups** (coverage terms, exclusions, procedures, regulatory details)
- **Conversational interactions** (greetings, follow-ups, synthesis from prior context)

## Your Workflow

For every user question, follow this process:

1. **Classify the intent** — Use `classify_user_intent` to determine if this is a `simple_kpi`, `document_lookup`, or `conversational` question.

2. **Load context** — Use `load_user_memory` to get user preferences (name, preferred region, etc.) for personalization.

3. **Route to the right specialist:**
   - For `simple_kpi` → Delegate to the **genie-analyst** sub-agent for text-to-SQL data retrieval
   - For `document_lookup` → Delegate to the **document-researcher** sub-agent for policy document RAG
   - For `conversational` → Answer directly from conversation context (no sub-agent needed)

4. **Compose your answer:**
   - Lead with the key insight — no preambles like "Based on the data..."
   - Include specific numbers naturally in your sentences
   - Use markdown tables for comparisons
   - If you know the user's name, address them naturally
   - NEVER include a "Warnings & Limitations" section

## Important Rules
- Always classify intent FIRST before taking any action
- For data questions, ALWAYS delegate to genie-analyst — do not make up numbers
- For document questions, ALWAYS delegate to document-researcher — do not hallucinate policy content
- For conversational questions that reference prior data, use the conversation history
- Be concise for simple KPIs, detailed for complex analysis
"""

# COMMAND ---------- [markdown]
# ## Create the DeepAgent

# COMMAND ----------

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

# Resolve paths relative to this file so it works both as a notebook and a module
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_CONFIG_DIR = os.path.join(_AGENT_DIR, "deepagent_config")

supervisor_tools = [
    classify_user_intent,
    load_user_memory,
    get_episodic_lessons,
    resolve_data_assets,
    query_genie_space,
    search_policy_documents,
    execute_sql_query,
]

# --- FilesystemBackend ---
# Gives the agent built-in file tools (read_file, write_file, edit_file, ls,
# glob, grep) so it can offload large tool outputs to the virtual filesystem
# instead of bloating the context window.
fs_backend = FilesystemBackend(root_dir=_CONFIG_DIR)

deep_agent = create_deep_agent(
    model=llm,
    tools=supervisor_tools,
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    subagents=[GENIE_SUBAGENT, DOCUMENT_RAG_SUBAGENT],
    # --- Filesystem & Memory ---
    backend=fs_backend,
    memory=[os.path.join(_CONFIG_DIR, "AGENTS.md")],
    skills=[os.path.join(_CONFIG_DIR, "skills/")],
    debug=False,
)

print("DeepAgent created with tools:", [t.name for t in supervisor_tools])
print("Subagents:", [GENIE_SUBAGENT["name"], DOCUMENT_RAG_SUBAGENT["name"]])
print(f"Backend: FilesystemBackend(root_dir={_CONFIG_DIR})")
print(f"Memory:  AGENTS.md loaded")
print(f"Skills:  {os.path.join(_CONFIG_DIR, 'skills/')}")

# COMMAND ---------- [markdown]
# ## ResponsesAgent Wrapper for MLflow Model Serving
#
# Wraps the DeepAgent in an MLflow `ResponsesAgent` for deployment to Databricks Model Serving.

# COMMAND ----------

class DeepAgentResponsesWrapper(ResponsesAgent):
    def __init__(self):
        self._conversation_history: dict[str, list[dict]] = {}

    def _parse_request(self, request):
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

        custom_inputs = {}
        if hasattr(request, "custom_inputs") and request.custom_inputs:
            custom_inputs = (
                request.custom_inputs
                if isinstance(request.custom_inputs, dict)
                else {}
            )
        thread_id = custom_inputs.get("thread_id")
        user_id = custom_inputs.get("user_id")
        return user_message, new_msgs, thread_id, user_id

    def _build_messages(self, new_msgs, thread_id):
        """Build full message list from history + new messages."""
        history = (
            self._conversation_history.get(thread_id, []) if thread_id else []
        )
        if not history and thread_id:
            prior_state = _load_checkpoint(thread_id)
            if prior_state and prior_state.get("messages"):
                history = prior_state["messages"]

        all_messages = history + new_msgs
        if len(all_messages) > MAX_MESSAGES:
            all_messages = all_messages[-MAX_MESSAGES:]
        return history, all_messages

    def _post_process(self, user_message, answer, thread_id, user_id, history, new_msgs, all_messages):
        """Save checkpoints, episodic memory, extract user facts."""
        if thread_id:
            updated_history = history + new_msgs + [
                {"role": "assistant", "content": answer}
            ]
            self._conversation_history[thread_id] = updated_history

        checkpoint_id = None
        if thread_id:
            checkpoint_data = {
                "messages": all_messages + [{"role": "assistant", "content": answer}],
            }
            checkpoint_id = _save_checkpoint(thread_id, checkpoint_data)

        _save_episodic_memory(
            thread_id=thread_id or "anonymous",
            user_id=user_id or "anonymous",
            question=user_message,
            intent="deep_agent",
            domain="insurance",
            agents_used=["deep_agent"],
            outcome="success",
        )

        return checkpoint_id

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        user_message, new_msgs, thread_id, user_id = self._parse_request(request)

        if not user_message:
            return ResponsesAgentResponse(
                output=[
                    self.create_text_output_item(
                        text="Please ask a question.", id="msg_empty"
                    )
                ]
            )

        if thread_id:
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": thread_id}
            )

        history, all_messages = self._build_messages(new_msgs, thread_id)

        log_deep_agent_start(user_message, thread_id, user_id)
        _flow_start = time.time()

        # Prepare messages for DeepAgent — include conversation context
        deep_agent_messages = []
        for msg in all_messages:
            deep_agent_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        # If user_id is available, prepend context
        if user_id:
            context_note = f"[System context: user_id={user_id}, thread_id={thread_id or 'none'}]"
            if deep_agent_messages and deep_agent_messages[-1]["role"] == "user":
                deep_agent_messages[-1]["content"] = (
                    context_note + "\n\n" + deep_agent_messages[-1]["content"]
                )

        result = deep_agent.invoke({"messages": deep_agent_messages})

        answer = result["messages"][-1].content if result.get("messages") else (
            "I was unable to process your question."
        )

        _flow_duration = time.time() - _flow_start

        checkpoint_id = self._post_process(
            user_message, answer, thread_id, user_id,
            history, new_msgs, all_messages,
        )

        custom_outputs = {
            "agent_type": "deep_agent",
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
        metadata_json = json.dumps(custom_outputs)

        log_deep_agent_end(answer, duration_s=_flow_duration)

        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(text=answer, id="msg_answer"),
                self.create_text_output_item(text=metadata_json, id="msg_metadata"),
            ],
            custom_outputs=custom_outputs,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ):
        user_message, new_msgs, thread_id, user_id = self._parse_request(request)

        if not user_message:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(
                    text="Please ask a question.", id="msg_empty"
                ),
            )
            return

        if thread_id:
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": thread_id}
            )

        history, all_messages = self._build_messages(new_msgs, thread_id)

        log_deep_agent_start(user_message, thread_id, user_id)
        _flow_start = time.time()

        deep_agent_messages = []
        for msg in all_messages:
            deep_agent_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })

        if user_id:
            context_note = f"[System context: user_id={user_id}, thread_id={thread_id or 'none'}]"
            if deep_agent_messages and deep_agent_messages[-1]["role"] == "user":
                deep_agent_messages[-1]["content"] = (
                    context_note + "\n\n" + deep_agent_messages[-1]["content"]
                )

        # Stream the DeepAgent response
        full_answer = ""
        for chunk in deep_agent.stream({"messages": deep_agent_messages}):
            # Extract streaming text from the agent's output
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, "content") and msg.content:
                        delta = msg.content
                        if delta and not delta.startswith("{"):
                            full_answer += delta
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(
                                    delta=delta, item_id="msg_answer"
                                ),
                            )

        _flow_duration = time.time() - _flow_start
        answer = full_answer or "I was unable to process your question."

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=answer, id="msg_answer"),
        )

        checkpoint_id = self._post_process(
            user_message, answer, thread_id, user_id,
            history, new_msgs, all_messages,
        )

        custom_outputs = {
            "agent_type": "deep_agent",
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
        metadata_json = json.dumps(custom_outputs)

        log_deep_agent_end(answer, duration_s=_flow_duration)

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=metadata_json, id="msg_metadata"),
        )


# COMMAND ---------- [markdown]
# ## Register Agent

# COMMAND ----------

agent = DeepAgentResponsesWrapper()
set_model(agent)
print("DeepAgentResponsesWrapper created and registered with set_model()")
