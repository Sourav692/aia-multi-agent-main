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
agent_source = f"/Workspace{notebook_dir}/customer_360.py"

agent_file_path = "/tmp/customer_360.py"
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
# MAGIC ## Register the Agent with MLflow (On-Behalf-Of Authorization)

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksSQLWarehouse, DatabricksVectorSearchIndex, DatabricksGenieSpace
from mlflow.models.auth_policy import AuthPolicy, SystemAuthPolicy, UserAuthPolicy

# System-managed resources — the serving endpoint will auto-authenticate to these
resources = [
    DatabricksServingEndpoint(endpoint_name="databricks-claude-opus-4-6"),
    DatabricksSQLWarehouse(warehouse_id="4b9b953939869799"),
    DatabricksVectorSearchIndex(index_name=f"{CATALOG}.ai_ops.context_index_vs"),
    # TODO: Re-add once the policy_docs_vs index is created
    # DatabricksVectorSearchIndex(index_name=f"{CATALOG}.ai_ops.policy_docs_vs"),
    DatabricksGenieSpace(genie_space_id="01f1272d4ba6144ba75d868762f1925d"),
    DatabricksGenieSpace(genie_space_id="01f1272d4c6b1fb49223785ab841befd"),
    DatabricksGenieSpace(genie_space_id="01f1272d4d271203ad122e9280470248"),
    DatabricksGenieSpace(genie_space_id="01f1272d4de1188cac8feeb7e71bdb69"),
]
system_auth_policy = SystemAuthPolicy(resources=resources)

# User-level auth — callers must have these API scopes
user_auth_policy = UserAuthPolicy(api_scopes=[
    "sql.warehouses",
    "sql.statement-execution",
    "serving.serving-endpoints",
    "vectorsearch.vector-search-indexes",
    "genie",
    "dashboards.genie",
    "iam.current-user:read",
])

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/aia_supervisor_agent")

with mlflow.start_run(run_name="supervisor_agent_v12_conv_context_routing"):
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
        auth_policy=AuthPolicy(
            system_auth_policy=system_auth_policy,
            user_auth_policy=user_auth_policy,
        ),
    )
    print(f"Logged model: {model_info.model_uri}")

# COMMAND ----------

# DBTITLE 1,Test logged model — Load and Q1
import mlflow
import json

# Load the latest registered model
model_name = f"{CATALOG}.ai_ops.supervisor_agent"
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = max(int(v.version) for v in versions)
print(f"Loading model: {model_name} v{latest_version}")

loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
print("Model loaded successfully!\n")

# Helper to extract answer text from pyfunc response
def extract_answer(response):
    """Extract msg_answer text from pyfunc predict response."""
    if hasattr(response, 'output'):
        for item in response.output:
            item_id = getattr(item, 'id', '')
            content = getattr(item, 'content', [])
            if item_id == 'msg_answer' and content and isinstance(content[0], dict):
                return content[0].get('text', '')
    # Fallback: try dict-like access
    if isinstance(response, dict):
        for item in response.get('output', []):
            if item.get('id') == 'msg_answer':
                content = item.get('content', [])
                if content and isinstance(content[0], dict):
                    return content[0].get('text', '')
    return str(response)

THREAD_ID = "model-validation-thread1"
USER_ID = "model-validation-user1"

questions = [
    "What is the total number of claims submitted by region for the last three calendar months include current month?",
    "Show me the total premium collected by product type across all regions.",
    "Based on those two results \u2014 the claims by region and premium by product \u2014 which regions are generating the most premium but also have the highest claim volumes? Are there any regions where we might be underpriced?",
]

conversation_history = []
for i, q in enumerate(questions, 1):
    print(f"{'='*80}")
    print(f"Q{i}: {q}")
    print(f"{'='*80}")

    conversation_history.append({"role": "user", "content": q})

    request = {
        "input": list(conversation_history),
        "custom_inputs": {"thread_id": THREAD_ID, "user_id": USER_ID},
    }

    resp = loaded_model.predict(request)
    answer = extract_answer(resp)
    print(f"\n{answer}\n")

    # Add assistant response to history for multi-turn context
    conversation_history.append({"role": "assistant", "content": answer})
    print(f"\u2705 Q{i} completed\n")

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

# COMMAND ----------

from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI()

# Query the endpoint — same as what the playground does
response = client.responses.create(
    model="agents_aia_multi_agent_catalog-ai_ops-supervisor_agent",
    input=[{"role": "user", "content": "What is the total number of claims submitted by region for the last three calendar months include current month?"}],
    extra_body={
        "custom_inputs": {"thread_id": "playground-test", "user_id": "playground-test-user"},
    },
)

print("Response type:", type(response))
print()
if hasattr(response, 'output'):
    for item in response.output:
        print(f"Item: {item}")
        print()
else:
    print(response)

# COMMAND ----------

# DBTITLE 1,Test v11 — Multi-turn endpoint conversation (OBO Genie)
from databricks_openai import DatabricksOpenAI
import json, time

client = DatabricksOpenAI()
ENDPOINT = "agents_aia_multi_agent_catalog-ai_ops-supervisor_agent"
THREAD = f"multi-turn-v11-{int(time.time())}"

questions = [
    "What is the total number of claims submitted by region for the last three calendar months include current month?",
    "Now show me the total premium collected by product type across all regions.",
    "Based on those two results — the claims by region and premium by product — which regions are generating the most premium but also have the highest claim volumes? Are there any regions where we might be underpriced?",
]

conversation = []
for i, q in enumerate(questions, 1):
    print(f"{'='*80}")
    print(f"Q{i}: {q}")
    print(f"{'='*80}")

    conversation.append({"role": "user", "content": q})

    response = client.responses.create(
        model=ENDPOINT,
        input=list(conversation),
        extra_body={"custom_inputs": {"thread_id": THREAD, "user_id": "multi-turn-test"}},
    )

    answer_text, metadata = "", {}
    for item in response.output:
        item_id = getattr(item, 'id', '')
        content = getattr(item, 'content', [])
        if content and hasattr(content[0], 'text'):
            text = content[0].text
            if item_id == 'msg_answer':
                answer_text = text
            elif item_id == 'msg_metadata':
                try:
                    metadata = json.loads(text)
                except:
                    pass

    # Show status
    genie = metadata.get("agent_details", {}).get("genie", {})
    intent = metadata.get("intent", "?")
    conf = metadata.get("intent_confidence", 0)
    domain = metadata.get("domain", "?")
    warnings = metadata.get("warnings", [])

    print(f"  Intent:  {intent} ({conf:.0%})  |  Domain: {domain}")
    if genie:
        print(f"  Genie:   {genie.get('status','?')}  |  Space: {genie.get('display_name','?')}  |  Spaces tried: {genie.get('spaces_tried','?')}")
        if genie.get('sql'):
            print(f"  SQL:     {genie['sql'][:120]}...")
    if warnings:
        print(f"  ⚠ Warnings: {warnings}")

    print(f"\nANSWER ({len(answer_text)} chars):")
    print(answer_text[:600])
    if len(answer_text) > 600:
        print(f"  ... [{len(answer_text) - 600} more chars]")

    status = "✅" if not warnings and (not genie or genie.get('status') == 'success') else "⚠️"
    print(f"\n{status} Q{i} completed\n")

    conversation.append({"role": "assistant", "content": answer_text})

print(f"{'='*80}")
print(f"Multi-turn test complete — thread: {THREAD}")
print(f"All {len(questions)} questions processed via OBO Genie auth (v11)")

# COMMAND ----------

