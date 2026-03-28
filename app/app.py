"""
AIA Agent 360 — Multi-Agent Insurance System Chat UI
Instant thinking animation via assets/thinking.js (raw JS outside Dash/React).
"""

import os, json, uuid, time, requests
import dash
from dash import html, dcc, Input, Output, State, callback, ALL, ctx, no_update
import dash_bootstrap_components as dbc

AGENT_ENDPOINT = os.getenv("SERVING_ENDPOINT_NAME", "agents_aia_multi_agent_catalog-ai_ops-supervisor_agent")
SQL_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "4b9b953939869799")
CATALOG = "aia_multi_agent_catalog"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="AIA Agent 360",
    suppress_callback_exceptions=True,
)
server = app.server

# --- Design Tokens (Stitch "Obsidian Architect" dark theme) ---
C_BG = "#060e20"; C_SIDEBAR = "#131b2e"; C_HEADER = "#0b1326"
C_PRIMARY = "#c0392b"; C_PRIMARY_LIGHT = "rgba(192,57,43,0.15)"
C_SURFACE = "#2d3449"; C_BORDER = "rgba(89,65,61,0.15)"; C_TEXT = "#dae2fd"
C_MUTED = "#94a3b8"; C_SUCCESS = "#68dba9"; C_WARNING = "#d97706"
C_DANGER = "#dc2626"; C_PURPLE = "#c0c1ff"
C_THINKING_BG = "#222a3d"; C_THINKING_BORDER = "rgba(89,65,61,0.15)"
C_SURFACE_MID = "#171f33"; C_SURFACE_HIGH = "#222a3d"; C_SURFACE_TOP = "#2d3449"

USER_BUBBLE = {
    "backgroundColor": C_PRIMARY, "color": "white",
    "padding": "12px 20px", "borderRadius": "16px 16px 4px 16px",
    "maxWidth": "80%", "marginLeft": "auto", "marginBottom": "10px",
    "fontSize": "0.9em", "lineHeight": "1.6", "whiteSpace": "pre-wrap", "wordBreak": "break-word",
    "boxShadow": "0 4px 20px rgba(192,57,43,0.25)",
}
AI_BUBBLE = {
    "backgroundColor": C_SURFACE_TOP, "color": C_TEXT,
    "padding": "16px 20px", "borderRadius": "16px 16px 16px 4px",
    "maxWidth": "85%", "marginBottom": "10px",
    "border": f"1px solid {C_BORDER}",
    "boxShadow": "0 4px 20px rgba(0,0,0,0.3)",
    "lineHeight": "1.6", "fontSize": "0.9em",
}
THINKING_STYLE = {
    "backgroundColor": C_SURFACE_HIGH, "border": f"1px solid {C_THINKING_BORDER}",
    "borderRadius": "12px", "padding": "12px 16px", "marginBottom": "8px",
    "maxWidth": "85%", "fontSize": "0.8em", "color": C_MUTED,
    "backdropFilter": "blur(12px)",
}
HISTORY_ITEM = {
    "padding": "10px 14px", "borderRadius": "8px", "cursor": "pointer",
    "marginBottom": "4px", "fontSize": "0.82em",
    "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
    "transition": "all 0.2s ease",
}

SAMPLE_QUESTIONS = [
    "What is the total number of claims by region?",
    "Which product categories have the highest fraud scores?",
    "What does the AIA Health Premium Plan cover?",
    "Are there any anomalies in our claims data?",
    "Show me a dashboard of claims trends by region",
    "Show me the top 5 agents by premium sold",
]
AGENT_COLORS = {"Genie": "#007954", "Multi-Tool": "#3131c0", "Analysis": "#d97706", "Visualization": "#c0392b"}


def _make_ai_bubble(answer, warnings, clarification, dashboard_urls, msg_offset=0, timestamp=None):
    """Build AI answer elements: bubble with warning icon, clarification, dashboards."""
    elements = []
    ai_content = [dcc.Markdown(answer, style={"margin": 0})]
    if warnings:
        tooltip_text = " | ".join(warnings)
        ai_content.append(html.Div([
            html.Span("\u26a0", id={"type": "warn-icon", "index": msg_offset},
                       style={"cursor": "pointer", "fontSize": "0.85em", "color": C_WARNING,
                              "marginLeft": "6px", "position": "relative"}),
            html.Div(tooltip_text, style={
                "display": "none", "position": "absolute", "bottom": "100%", "left": "0",
                "backgroundColor": "#1f2937", "color": "white", "padding": "6px 10px",
                "borderRadius": "6px", "fontSize": "0.75em", "maxWidth": "320px",
                "whiteSpace": "normal", "zIndex": "1000", "marginBottom": "4px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.2)", "lineHeight": "1.4",
            }, className="aia-warn-tooltip"),
        ], style={"display": "inline-block", "position": "relative"}))
    elements.append(html.Div(ai_content, style=AI_BUBBLE))
    if clarification:
        elements.append(html.Div([
            html.I(className="bi bi-chat-dots", style={"marginRight": "8px", "color": "#ffb4a9", "fontSize": "0.85em"}),
            html.Span(clarification, style={"fontSize": "0.88em", "color": C_TEXT, "lineHeight": "1.5"}),
        ], style={"padding": "12px 16px", "backgroundColor": C_PRIMARY_LIGHT, "borderRadius": "12px",
                  "marginBottom": "8px", "border": f"1px solid {C_BORDER}", "maxWidth": "85%"}))
    for url in (dashboard_urls or []):
        elements.append(html.Div([
            html.I(className="bi bi-bar-chart-line", style={"marginRight": "8px", "color": "#ffb4a9"}),
            html.A("Open AI/BI Dashboard", href=url, target="_blank",
                   style={"fontWeight": "600", "fontSize": "0.85em", "color": "#ffb4a9"}),
        ], style={"padding": "10px 16px", "backgroundColor": C_PRIMARY_LIGHT, "borderRadius": "8px",
                  "marginBottom": "8px", "border": f"1px solid rgba(192,57,43,0.3)", "maxWidth": "85%"}))
    if timestamp:
        elements.append(html.Div(timestamp, style={
            "fontSize": "0.65em", "color": C_MUTED, "marginTop": "2px", "marginBottom": "8px"}))
    return elements


def _badge(name, color):
    return html.Span(name, style={
        "display": "inline-block", "padding": "2px 8px", "borderRadius": "20px",
        "backgroundColor": color, "color": "white", "fontSize": "0.6em",
        "fontWeight": "700", "marginLeft": "4px", "letterSpacing": "0.05em",
        "textTransform": "uppercase"})


def make_welcome():
    return [html.Div([
        html.P("Welcome to AIA Agent 360",
               style={"fontSize": "1.4em", "fontWeight": "800", "margin": "0 0 8px",
                      "color": "white", "letterSpacing": "-0.02em"}),
        html.P("Ask me anything about claims, policies, agents, or coverage.",
               style={"margin": "0", "fontSize": "0.9em", "color": C_MUTED}),
    ], style={"textAlign": "center", "padding": "40px 20px", "maxWidth": "100%"})]


def make_thinking_block(trace_steps):
    items = [html.Div([
        html.Span("", style={"width": "6px", "height": "6px", "borderRadius": "50%",
                              "backgroundColor": C_SUCCESS if s.get("ok", True) else C_DANGER,
                              "display": "inline-block", "marginRight": "8px", "flexShrink": "0", "marginTop": "5px",
                              "boxShadow": f"0 0 6px {C_SUCCESS if s.get('ok', True) else C_DANGER}"}),
        html.Span(s["step"], style={"fontWeight": "600", "marginRight": "8px", "color": "white", "fontSize": "0.85em"}),
        html.Span(s.get("detail", ""), style={"color": C_MUTED, "fontSize": "0.8em"}),
    ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "4px"}) for s in trace_steps]
    return html.Div([
        html.Div([
            html.Span("", style={"width": "6px", "height": "6px", "borderRadius": "50%",
                                  "backgroundColor": C_SUCCESS, "display": "inline-block",
                                  "marginRight": "8px", "animation": "pulse 2s ease-in-out infinite"}),
            html.Span("AGENT INTELLIGENCE WORKFLOW", style={"fontWeight": "700", "color": C_SUCCESS,
                                                             "fontSize": "0.65em", "letterSpacing": "0.12em"}),
        ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center",
                  "paddingBottom": "8px", "borderBottom": f"1px solid {C_BORDER}"}),
        *items,
    ], style=THINKING_STYLE)


def build_sidebar_list(all_convs, active_id=None):
    children = []
    for cid in reversed(list(all_convs.keys())):
        c = all_convs[cid]
        is_active = cid == active_id
        style = {**HISTORY_ITEM,
                 "backgroundColor": C_SURFACE_TOP if is_active else "transparent",
                 "color": "#ffb4a9" if is_active else C_MUTED,
                 "fontWeight": "500" if is_active else "400",
                 "borderRight": f"2px solid {C_PRIMARY}" if is_active else "2px solid transparent"}
        children.append(html.Div([
            html.Div(c.get("title", "Untitled"),
                     style={"overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            html.Span(c.get("created_at", ""), style={"fontSize": "0.68em", "color": C_MUTED}),
        ], id={"type": "conv-item", "index": cid}, style=style, n_clicks=0))
    if not children:
        children = [html.P("No conversations yet",
                           style={"color": C_MUTED, "fontSize": "0.82em", "textAlign": "center", "marginTop": "16px"})]
    return children


# -- Layout --
app.layout = html.Div([
    html.Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"),
    html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"),

    # Header
    html.Div([
        html.Div([
            html.Div(html.Span("AIA", style={"fontWeight": "900", "fontSize": "0.65em", "color": "white",
                                               "letterSpacing": "0.08em"}),
                     style={"display": "inline-flex", "alignItems": "center", "justifyContent": "center",
                            "width": "32px", "height": "32px", "borderRadius": "4px",
                            "backgroundColor": C_PRIMARY, "marginRight": "12px"}),
            html.Span("AIA", style={"fontWeight": "800", "fontSize": "1em", "color": "white",
                                     "marginRight": "6px", "letterSpacing": "-0.02em", "textTransform": "uppercase"}),
            html.Span("AGENT 360", style={"fontWeight": "400", "fontSize": "1em", "color": "rgba(255,255,255,0.5)",
                                           "letterSpacing": "-0.02em"}),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Div([
                html.Span("", style={"width": "6px", "height": "6px", "borderRadius": "50%",
                                      "backgroundColor": c, "display": "inline-block", "marginRight": "6px",
                                      "boxShadow": f"0 0 6px {c}"}),
                html.Span(n, style={"fontSize": "0.6em", "fontWeight": "700", "color": c,
                                     "textTransform": "uppercase", "letterSpacing": "0.1em"}),
            ], style={"display": "flex", "alignItems": "center",
                      "padding": "3px 10px", "borderRadius": "20px",
                      "backgroundColor": f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.12)",
                      "border": f"1px solid rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.25)",
                      "marginLeft": "6px"})
            for n, c in AGENT_COLORS.items()
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "padding": "10px 24px", "backgroundColor": C_HEADER, "height": "48px"}),

    dbc.Row([
        # Sidebar
        dbc.Col(html.Div([
            html.Div([
                html.Span("Conversations", style={"fontSize": "0.65em", "fontWeight": "700", "color": C_MUTED,
                                                    "textTransform": "uppercase", "letterSpacing": "0.12em"}),
                dbc.Button("+", id="new-chat-btn", size="sm", n_clicks=0,
                           style={"borderRadius": "4px", "padding": "1px 8px", "fontSize": "0.85em",
                                  "lineHeight": "1", "backgroundColor": C_PRIMARY, "border": "none",
                                  "color": "white"}),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"}),
            html.Div(id="conversation-list", children=[
                html.P("No conversations yet",
                       style={"color": C_MUTED, "fontSize": "0.82em", "textAlign": "center", "marginTop": "16px"})],
                     style={"flex": "1", "overflowY": "auto"}),
            html.Div([
                html.Div("Quick start", style={"fontSize": "0.62em", "fontWeight": "700", "color": C_MUTED,
                                                "textTransform": "uppercase", "letterSpacing": "0.12em", "marginBottom": "8px"}),
                *[html.Div(q, id={"type": "sample-q", "index": i}, n_clicks=0,
                           style={"fontSize": "0.75em", "color": "#ffb4a9", "cursor": "pointer",
                                  "padding": "4px 0", "lineHeight": "1.4",
                                  "transition": "color 0.2s ease"})
                  for i, q in enumerate(SAMPLE_QUESTIONS)],
            ], style={"borderTop": f"1px solid {C_BORDER}", "paddingTop": "12px"}),
        ], style={"height": "calc(100vh - 48px)", "display": "flex", "flexDirection": "column",
                  "padding": "16px 12px", "backgroundColor": C_SIDEBAR}),
                 width=2, className="p-0"),

        # Chat
        dbc.Col(html.Div([
            html.Div(id="chat-messages", children=make_welcome(),
                     style={"flex": "1", "overflowY": "auto", "padding": "20px 32px", "backgroundColor": C_BG}),
            html.Div([
                dbc.InputGroup([
                    dbc.Input(id="user-input", placeholder="Ask a question...", type="text",
                              n_submit=0, debounce=False,
                              style={"borderRadius": "12px 0 0 12px", "border": f"1px solid {C_BORDER}",
                                     "padding": "12px 16px", "fontSize": "0.9em", "boxShadow": "none",
                                     "backgroundColor": C_SURFACE_TOP, "color": C_TEXT}),
                    dbc.Button(html.I(className="bi bi-arrow-up"), id="send-btn", n_clicks=0,
                               style={"borderRadius": "0 12px 12px 0", "padding": "12px 16px",
                                      "backgroundColor": C_PRIMARY, "border": "none", "color": "white"}),
                ]),
            ], style={"padding": "12px 32px 16px", "backgroundColor": C_SURFACE_MID,
                      "borderTop": f"1px solid {C_BORDER}"}),
        ], style={"position": "relative", "height": "calc(100vh - 48px)", "display": "flex", "flexDirection": "column"}),
                 width=7, className="p-0"),

        # Right panel
        dbc.Col(html.Div([
            html.Div("Warnings", style={"fontSize": "0.65em", "fontWeight": "700", "color": C_MUTED,
                                          "textTransform": "uppercase", "letterSpacing": "0.12em", "marginBottom": "12px"}),
            html.Div(id="warnings-panel", children=[
                html.Span("No warnings", style={"color": C_MUTED, "fontSize": "0.82em"})]),
        ], style={"height": "calc(100vh - 48px)", "display": "flex", "flexDirection": "column",
                  "padding": "16px 14px",
                  "backgroundColor": C_SIDEBAR, "overflowY": "auto"}), width=3, className="p-0"),
    ], className="g-0"),

    # Stores
    dcc.Store(id="all-conversations", data={}),
    dcc.Store(id="active-conversation-id", data=None),
    dcc.Store(id="session-loaded", data=False),
    dcc.Store(id="chat-history", data=[]),
    dcc.Store(id="pending-question", data=None),
    dcc.Store(id="pending-new-chat", data=None),
    dcc.Interval(id="session-loader", interval=1000, max_intervals=1),
], style={"height": "100vh", "overflow": "hidden",
          "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          "backgroundColor": C_HEADER})


# -- Session persistence: save/load conversations to Delta --
def _run_sql(sql_statement, max_rows=50):
    """Execute SQL via Databricks SDK Statement Execution API."""
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()  # Uses Config() auto-detection internally
        response = w.statement_execution.execute_statement(
            warehouse_id=SQL_WAREHOUSE_ID, statement=sql_statement, wait_timeout="30s",
        )
        state_val = None
        try:
            state_val = response.status.state.value if response.status and response.status.state else None
        except Exception:
            pass
        if state_val == "SUCCEEDED":
            columns = []
            try:
                manifest = response.manifest
                if manifest:
                    # Try different attribute paths for column names
                    cols_obj = getattr(manifest, 'columns', None)
                    if not cols_obj:
                        schema = getattr(manifest, 'schema', None)
                        if schema:
                            cols_obj = getattr(schema, 'columns', None)
                    if cols_obj:
                        columns = [c.name for c in cols_obj]
            except (AttributeError, TypeError):
                pass
            rows = []
            if response.result and response.result.data_array:
                for row in response.result.data_array[:max_rows]:
                    if columns:
                        rows.append(dict(zip(columns, row)))
                    else:
                        # Fallback: use positional keys
                        rows.append({f"col_{i}": v for i, v in enumerate(row)})
            return rows
    except Exception:
        pass
    return []


def _save_ui_session(all_convs):
    """Save UI conversation state to Delta for persistence across refreshes."""
    import sys
    try:
        import base64
        session_json = json.dumps(all_convs, default=str)
        b64 = base64.b64encode(session_json.encode()).decode()
        result = _run_sql(f"""
            MERGE INTO {CATALOG}.ai_ops.ui_sessions AS t
            USING (SELECT 'default' AS session_id,
                   CAST(unbase64('{b64}') AS STRING) AS state_json,
                   current_timestamp() AS updated_at) AS s
            ON t.session_id = s.session_id
            WHEN MATCHED THEN UPDATE SET t.state_json = s.state_json, t.updated_at = s.updated_at
            WHEN NOT MATCHED THEN INSERT *
        """)
    except Exception:
        pass


def _load_ui_session():
    """Load saved UI conversation state from Delta."""
    try:
        rows = _run_sql(f"""
            SELECT state_json FROM {CATALOG}.ai_ops.ui_sessions
            WHERE session_id = 'default'
            ORDER BY updated_at DESC LIMIT 1
        """)
        if rows:
            sj = rows[0].get("state_json") or rows[0].get("col_0", "")
            return json.loads(sj)
    except Exception:
        pass
    return {}


def _ensure_ui_sessions_table():
    """Create ui_sessions table if it doesn't exist."""
    try:
        _run_sql(f"""
            CREATE TABLE IF NOT EXISTS {CATALOG}.ai_ops.ui_sessions (
                session_id STRING NOT NULL,
                state_json STRING,
                updated_at TIMESTAMP,
                CONSTRAINT pk_ui_sessions PRIMARY KEY (session_id)
            ) USING DELTA
        """)
    except Exception:
        pass


# Session loading is deferred — done via callback, not at import time.


# -- Clientside: clear input + set pending-question --
app.clientside_callback(
    """function(sendClicks, nSubmit, sampleClicks, userInput) {
        var triggered = dash_clientside.callback_context.triggered;
        if (!triggered || triggered.length === 0) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }
        var propId = triggered[0].prop_id;
        var question = "";
        if (propId.indexOf("sample-q") >= 0) {
            try {
                var idx = JSON.parse(propId.split(".")[0]).index;
                var qs = """ + json.dumps(SAMPLE_QUESTIONS) + """;
                question = qs[idx];
            } catch(e) { return [dash_clientside.no_update, dash_clientside.no_update]; }
        } else if (userInput && userInput.trim()) {
            question = userInput.trim();
        }
        if (!question) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }
        return ["", {"question": question, "ts": Date.now()}];
    }""",
    Output("user-input", "value", allow_duplicate=True),
    Output("pending-question", "data"),
    Input("send-btn", "n_clicks"),
    Input("user-input", "n_submit"),
    Input({"type": "sample-q", "index": ALL}, "n_clicks"),
    State("user-input", "value"),
    prevent_initial_call=True,
)


# -- Clientside: new chat button sets a store (no output conflicts) --
app.clientside_callback(
    """function(n) {
        if (!n) return dash_clientside.no_update;
        return {"ts": Date.now()};
    }""",
    Output("pending-new-chat", "data"),
    Input("new-chat-btn", "n_clicks"),
    prevent_initial_call=True,
)


# -- Agent Call --
def call_agent(question, history, thread_id=None):
    messages = [{"role": h["role"], "content": h["content"]} for h in (history or [])[-6:]]
    messages.append({"role": "user", "content": question})
    try:
        from databricks.sdk.core import Config
        cfg = Config()
        host = cfg.host.rstrip("/")
        auth_headers = cfg.authenticate()
        payload = {"input": messages}
        custom_inputs = {}
        if thread_id:
            custom_inputs["thread_id"] = thread_id
        custom_inputs["user_id"] = "default"
        payload["custom_inputs"] = custom_inputs
        response = requests.post(
            f"{host}/serving-endpoints/{AGENT_ENDPOINT}/invocations",
            headers={**auth_headers, "Content-Type": "application/json"},
            json=payload, timeout=300)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        answer, meta = "", {}
        for item in data.get("output", []):
            if item.get("type") == "message" and item.get("role") == "assistant":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        text = c.get("text", "")
                        try:
                            parsed = json.loads(text)
                            if "intent" in parsed and "warnings" in parsed:
                                meta = parsed
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                        if not answer:
                            answer = text
        custom_out = data.get("custom_outputs", {})
        if custom_out and isinstance(custom_out, dict) and not meta:
            meta = custom_out
        if not answer and "choices" in data:
            answer = data["choices"][0].get("message", {}).get("content", "")
        if not answer:
            answer = json.dumps(data, indent=2, default=str)[:2000]

        intent = meta.get("intent", "unknown")
        confidence = meta.get("intent_confidence", 0.0)
        warnings = meta.get("warnings", [])
        nodes = meta.get("nodes_executed", [])
        dashboard_urls = meta.get("dashboard_urls", [])
        domain = meta.get("domain", "unknown")
        clarification = meta.get("clarification")
        genie_space = meta.get("genie_space")
        doc_vs_index = meta.get("doc_vs_index")
        agent_details = meta.get("agent_details", {})

        trace = [{"step": "Classify Intent", "detail": f"{intent} ({confidence:.0%})", "ok": True}]
        if "clarify_or_disambiguate" in nodes:
            trace.append({"step": "Clarify", "detail": clarification or "Resolved", "ok": True})

        asset_detail = f"domain: {domain}"
        if genie_space:
            asset_detail += f" · genie: {genie_space[:12]}..."
        if doc_vs_index:
            asset_detail += f" · docs: {doc_vs_index.split('.')[-1]}"
        trace.append({"step": "Resolve Assets", "detail": asset_detail, "ok": True})

        # Show agent-specific details
        for n, label in [("genie", "Genie Agent"), ("multi_tool", "Multi-Tool"),
                         ("analysis", "Analysis"), ("visualization", "Visualization")]:
            if n in nodes:
                ad = agent_details.get(n, {})
                detail_parts = []
                status = ad.get("status", "completed")
                if status == "success":
                    if n == "genie" and ad.get("row_count") is not None:
                        detail_parts.append(f"{ad['row_count']} rows")
                    if n == "multi_tool":
                        if ad.get("docs_found"):
                            detail_parts.append(f"{ad['docs_found']} docs")
                        if ad.get("sql_rows") is not None:
                            detail_parts.append(f"{ad['sql_rows']} rows")
                    if n == "analysis" and ad.get("row_count") is not None:
                        detail_parts.append(f"{ad['row_count']} rows analyzed")
                    if n == "visualization":
                        dc = ad.get("dashboards_count", 0)
                        detail_parts.append(f"{dc} dashboard{'s' if dc != 1 else ''}")
                elif status == "failed":
                    detail_parts.append("failed")
                detail = " · ".join(detail_parts) if detail_parts else "completed"
                trace.append({"step": label, "detail": detail, "ok": status != "failed"})
        trace.append({"step": "Compose Answer", "detail": "Done", "ok": True})
        return {"answer": answer, "status": "success", "warnings": warnings,
                "intent": intent, "confidence": confidence, "trace": trace,
                "dashboard_urls": dashboard_urls, "clarification": clarification}
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)[:300]}",
                "status": "error", "warnings": [str(e)[:200]],
                "intent": "error", "confidence": 0.0,
                "trace": [{"step": "Agent Call", "detail": str(e)[:80], "ok": False}],
                "dashboard_urls": []}


# -- Load saved sessions on page load (triggered by Interval) --
@callback(
    Output("all-conversations", "data", allow_duplicate=True),
    Output("conversation-list", "children", allow_duplicate=True),
    Output("session-loaded", "data"),
    Input("session-loader", "n_intervals"),
    State("session-loaded", "data"),
    prevent_initial_call=True,
)
def load_session(n_intervals, already_loaded):
    if already_loaded:
        return (no_update, no_update, True)
    saved = _load_ui_session()
    if saved:
        return (saved, build_sidebar_list(saved, None), True)
    return (no_update, no_update, True)


# -- Single unified callback: new chat, switch conversation, and agent processing --
# Avoids allow_duplicate issues in Databricks Apps environment.
@callback(
    Output("chat-messages", "children"),
    Output("chat-history", "data"),
    Output("warnings-panel", "children"),
    Output("all-conversations", "data"),
    Output("active-conversation-id", "data"),
    Output("conversation-list", "children"),
    Input("pending-question", "data"),
    Input("pending-new-chat", "data"),
    Input({"type": "conv-item", "index": ALL}, "n_clicks"),
    State("chat-messages", "children"),
    State("chat-history", "data"),
    State("all-conversations", "data"),
    State("active-conversation-id", "data"),
    prevent_initial_call=True,
)
def unified_callback(pending, new_chat, conv_clicks, current_messages, history, all_convs, active_conv_id):
    triggered = ctx.triggered_id
    all_convs = all_convs or {}
    no_w = [html.Span("No warnings", style={"color": C_MUTED, "fontSize": "0.82em"})]

    # -- New Chat --
    if triggered == "pending-new-chat":
        return (make_welcome(), [], no_w, all_convs, None, build_sidebar_list(all_convs, None))

    # -- Switch Conversation --
    if isinstance(triggered, dict) and triggered.get("type") == "conv-item":
        clicked_id = triggered.get("index")
        if not clicked_id or clicked_id == active_conv_id:
            return (no_update,) * 6
        conv = all_convs.get(clicked_id)
        if not conv:
            return (no_update,) * 6
        messages = conv.get("messages", [])
        msgs = make_welcome()
        exchanges = conv.get("exchanges", [])
        for i in range(0, len(messages), 2):
            user_msg = messages[i] if i < len(messages) else None
            ai_msg = messages[i + 1] if i + 1 < len(messages) else None
            ex_idx = i // 2
            ex = exchanges[ex_idx] if ex_idx < len(exchanges) else {}
            if user_msg:
                user_ts = ex.get("user_ts", "")
                msgs.append(html.Div([
                    html.Div(user_msg["content"], style=USER_BUBBLE),
                    html.Div(user_ts, style={"fontSize": "0.65em", "color": C_MUTED, "marginTop": "2px",
                                              "textAlign": "right", "marginBottom": "8px"}) if user_ts else None,
                ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-end"}))
            if ai_msg:
                if ex.get("trace"):
                    msgs.append(make_thinking_block(ex["trace"]))
                reply_ts = ex.get("reply_ts", "")
                msgs.extend(_make_ai_bubble(ai_msg["content"], ex.get("warnings", []),
                                            ex.get("clarification"), ex.get("dashboard_urls", []),
                                            len(msgs), timestamp=reply_ts))
        w_list = _warn_list(conv.get("all_warnings", [])) or no_w
        return msgs, messages, w_list, all_convs, clicked_id, build_sidebar_list(all_convs, clicked_id)

    # -- Process Agent Question --
    if triggered == "pending-question":
        if not pending or not pending.get("question"):
            return (no_update,) * 6

        question = pending["question"]

        if not active_conv_id:
            active_conv_id = str(uuid.uuid4())[:8]
            all_convs[active_conv_id] = {
                "title": question[:50], "messages": [], "all_warnings": [],
                "created_at": time.strftime("%H:%M"), "thread_id": str(uuid.uuid4()),
            }
            current_messages = make_welcome()

        current_messages = current_messages or []
        now_ts = time.strftime("%H:%M:%S")
        current_messages.append(html.Div([
            html.Div(question, style=USER_BUBBLE),
            html.Div(now_ts, style={"fontSize": "0.65em", "color": C_MUTED, "marginTop": "2px",
                                     "textAlign": "right", "marginBottom": "8px"}),
        ], style={"display": "flex", "flexDirection": "column", "alignItems": "flex-end"}))

        conv_thread_id = all_convs.get(active_conv_id, {}).get("thread_id")
        result = call_agent(question, history or [], thread_id=conv_thread_id)
        reply_ts = time.strftime("%H:%M:%S")

        if result.get("trace"):
            current_messages.append(make_thinking_block(result["trace"]))

        current_messages.extend(_make_ai_bubble(
            result["answer"], result.get("warnings", []),
            result.get("clarification"), result.get("dashboard_urls", []),
            len(current_messages), timestamp=reply_ts))

        history = history or []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})

        conv = all_convs.get(active_conv_id, {})
        conv["messages"] = history
        conv.setdefault("exchanges", []).append({
            "trace": result.get("trace", []),
            "warnings": result.get("warnings", []),
            "clarification": result.get("clarification"),
            "dashboard_urls": result.get("dashboard_urls", []),
            "user_ts": now_ts,
            "reply_ts": reply_ts,
        })
        if result.get("warnings"):
            warn_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            conv.setdefault("all_warnings", []).extend(
                [{"text": w, "ts": warn_ts} for w in result["warnings"]]
            )
        all_convs[active_conv_id] = conv
        w_list = _warn_list(conv.get("all_warnings", [])) or no_w

        # Persist sessions to Delta for page refresh survival
        _save_ui_session(all_convs)

        return (current_messages, history, w_list,
                all_convs, active_conv_id, build_sidebar_list(all_convs, active_conv_id))

    return (no_update,) * 6


def _warn_list(warnings):
    if not warnings:
        return None
    items = []
    for w in warnings:
        # warnings can be plain strings or dicts with "text" and "ts"
        if isinstance(w, dict):
            text = w.get("text", str(w))
            ts = w.get("ts", "")
        else:
            text = w
            ts = ""
        warning_els = [
            html.Div(text, style={
                "padding": "8px 12px", "backgroundColor": "rgba(217,119,6,0.1)",
                "border": "1px solid rgba(217,119,6,0.3)", "borderRadius": "8px",
                "fontSize": "0.75em", "color": C_TEXT,
                "lineHeight": "1.4", "maxHeight": "80px", "overflowY": "auto",
            }),
        ]
        if ts:
            warning_els.append(html.Div(ts, style={
                "fontSize": "0.6em", "color": C_MUTED, "marginTop": "1px", "marginLeft": "2px",
            }))
        items.append(html.Div(warning_els, style={"marginBottom": "6px"}))
    return items


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
