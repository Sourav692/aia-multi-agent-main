# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Agent Evaluation with MLflow
# MAGIC Evaluates the Supervisor Agent using MLflow Agent Evaluation with LLM judges.

# COMMAND ----------

# MAGIC %pip install mlflow>=2.18 databricks-agents
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import pandas as pd

CATALOG = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Evaluation Dataset

# COMMAND ----------

eval_data = pd.DataFrame([
    {
        "request": "What is the total number of claims by region?",
        "expected_response": "The response should include claim counts broken down by region (Hong Kong, Singapore, Thailand, etc.)",
        "expected_facts": ["claim count", "region", "Hong Kong", "Singapore"],
    },
    {
        "request": "What is the average claim amount for hospitalization claims?",
        "expected_response": "The response should include the average claim amount specifically for hospitalization claims in USD.",
        "expected_facts": ["average", "hospitalization", "USD"],
    },
    {
        "request": "Which regions have the highest fraud scores?",
        "expected_response": "The response should identify regions with elevated fraud scores and mention specific numbers.",
        "expected_facts": ["fraud score", "region", "highest"],
    },
    {
        "request": "What does the AIA Health plan cover?",
        "expected_response": "The response should describe health insurance coverage including hospitalization, surgery, outpatient benefits.",
        "expected_facts": ["coverage", "hospitalization", "benefits"],
    },
    {
        "request": "Are there any anomalies in our claims data?",
        "expected_response": "The response should mention statistical analysis, z-scores, and identify any unusual patterns in claims.",
        "expected_facts": ["anomaly", "analysis", "z-score"],
    },
    {
        "request": "What is the claim approval rate by claim type?",
        "expected_response": "The response should show approval rates broken down by claim type (Hospitalization, Surgery, Outpatient, etc.)",
        "expected_facts": ["approval rate", "claim type"],
    },
    {
        "request": "Show me the top 5 agents by total premium sold",
        "expected_response": "The response should list the top 5 insurance agents ranked by total premium sold in USD.",
        "expected_facts": ["agent", "premium", "top"],
    },
    {
        "request": "How many suspicious claims were filed in 2024?",
        "expected_response": "The response should provide the count of suspicious claims for 2024 with relevant context.",
        "expected_facts": ["suspicious", "claims", "2024", "count"],
    },
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Evaluation

# COMMAND ----------

mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/aia_agent_evaluation")

# Load the registered model
model_uri = f"models:/{CATALOG}.ai_ops.supervisor_agent/1"

with mlflow.start_run(run_name="agent_eval_v1"):
    results = mlflow.evaluate(
        model=model_uri,
        data=eval_data,
        model_type="databricks-agent",
    )

    print(f"\nEvaluation Metrics:")
    for metric, value in results.metrics.items():
        print(f"  {metric}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review Results

# COMMAND ----------

eval_table = results.tables["eval_results"]
display(eval_table[["request", "response", "retrieval/llm_judged/relevance/rating", "response/llm_judged/correctness/rating"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Evaluation Results to UC

# COMMAND ----------

eval_df = spark.createDataFrame(eval_table)
eval_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.ai_ops.agent_evaluation_results")
print(f"Saved evaluation results to {CATALOG}.ai_ops.agent_evaluation_results")
