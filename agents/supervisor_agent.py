# Databricks notebook source
# MAGIC %md
# MAGIC # Supervisor Agent — AIA Multi-Agent System
# MAGIC
# MAGIC The Supervisor is the brain of the system. It:
# MAGIC 1. Interprets user intent
# MAGIC 2. Calls Context Index for semantic asset discovery
# MAGIC 3. Routes to specialist agents (Genie, Multi-Tool, Data Analysis, Visualization)
# MAGIC 4. Optionally clarifies ambiguous questions
# MAGIC 5. Composes the final answer
# MAGIC
# MAGIC **Architecture:** LangGraph StateGraph + MLflow ResponsesAgent
# MAGIC **Deployment:** Model Serving endpoint via code-based logging

# COMMAND ----------

# MAGIC %pip install mlflow>=3.1 databricks-agents>=1.0.0 pydantic>=2 langgraph>=0.2 langchain-core databricks-langchain databricks-vectorsearch databricks-ai-bridge --upgrade
# MAGIC %restart_python

# COMMAND ----------

import mlflow

CATALOG = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write the standalone agent code file
# MAGIC MLflow ResponsesAgent requires code-based logging — the agent must be in a standalone .py file.

# COMMAND ----------

import shutil, os

notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
notebook_dir = os.path.dirname(notebook_path)  # e.g. /Users/.../aia-multi-agent/agents
agent_source = f"/Workspace{notebook_dir}/agent_code.py"

agent_file_path = "/tmp/agent_code.py"
shutil.copy(agent_source, agent_file_path)
print(f"Agent code copied from {agent_source} to {agent_file_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate the agent code loads correctly

# COMMAND ----------

# Quick validation - import and check
import importlib.util
spec = importlib.util.spec_from_file_location("agent_code", agent_file_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print(f"Agent class: {mod.SupervisorResponsesAgent}")
print(f"Graph nodes: {list(mod.graph.nodes)}")
print("Agent code validated successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the Agent with MLflow

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/aia_supervisor_agent")

with mlflow.start_run(run_name="supervisor_agent_v8_p0p1"):
    model_info = mlflow.pyfunc.log_model(
        name="supervisor_agent",
        python_model=agent_file_path,
        pip_requirements=[
            "mlflow>=3.1",
            "databricks-agents>=1.0.0",
            "pydantic>=2",
            "langgraph>=0.2",
            "langchain-core",
            "databricks-langchain",
            "databricks-vectorsearch",
            "databricks-sdk",
            "databricks-ai-bridge",
        ],
        registered_model_name=f"{CATALOG}.ai_ops.supervisor_agent",
    )
    print(f"Logged model: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the LangGraph directly

# COMMAND ----------

# Quick test using the graph directly (not through ResponsesAgent)
# NOTE: Skipping live tests to avoid rate limits during registration
print("Skipping live graph tests — model registered successfully. Run tests manually after deployment.")
test_questions = [] if True else [
    "What is the total number of claims by region?",
    "What does the AIA Health plan cover?",
    "Are there any anomalies in our claims data?",
    "Which agents have the highest churn rate?",
]

for q in test_questions:
    print(f"\n{'='*70}")
    print(f"Q: {q}")
    print(f"{'='*70}")

    state = {
        "messages": [{"role": "user", "content": q}],
        "user_question": q,
        "intent": "",
        "intent_confidence": 0.0,
        "clarification_message": None,
        "needs_clarification": False,
        "resolved_assets": None,
        "genie_results": None,
        "multi_tool_results": None,
        "analysis_results": None,
        "viz_results": None,
        "final_answer": None,
        "warnings": [],
        "thread_id": None,
        "user_id": None,
        "dashboard_urls": [],
    }

    result = mod.graph.invoke(state)
    print(f"\nIntent: {result['intent']} (confidence: {result.get('intent_confidence', 0):.0%})")
    print(f"Agents used: genie={'yes' if result.get('genie_results') else 'no'}, multi_tool={'yes' if result.get('multi_tool_results') else 'no'}, analysis={'yes' if result.get('analysis_results') else 'no'}, viz={'yes' if result.get('viz_results') else 'no'}")
    print(f"Answer:\n{result['final_answer'][:500]}")
    if result.get("warnings"):
        print(f"Warnings: {result['warnings']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create or Update Model Serving Endpoint
# MAGIC Creates the `aia-supervisor-agent` endpoint if it doesn't exist, otherwise updates it to serve
# MAGIC the latest registered model version. Uses the Databricks SDK directly — avoids the
# MAGIC `agents.deploy()` local-store limitation.

# COMMAND ----------

from databricks import agents
import mlflow

# Get the latest registered version
client = mlflow.tracking.MlflowClient()
model_name = f"{CATALOG}.ai_ops.supervisor_agent"
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max(int(v.version) for v in versions)
print(f"Latest model version: {latest_version}")

# Deploy using agents.deploy() — this creates:
#   1. Model serving endpoint
#   2. Review App for human evaluation
#   3. Inference tables for logging
#   4. Secure authentication for Databricks resources
deployment = agents.deploy(
    model_name,
    latest_version,
    scale_to_zero_enabled=True,
)

print(f"\nEndpoint name: {deployment.endpoint_name}")
print(f"Query endpoint: {deployment.query_endpoint}")
print(f"Review App URL: {deployment.review_app_url}")
