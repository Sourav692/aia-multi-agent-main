# Databricks notebook source
# MAGIC %md
# MAGIC # Review App Integration — AIA Multi-Agent System
# MAGIC
# MAGIC Sets up MLflow Review App integration for collecting human feedback
# MAGIC on agent responses. Feedback (thumbs up/down + comments) is logged
# MAGIC to the `episodic_memory` table for continuous learning.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Run `create_p2_tables.sql` first to create the `episodic_memory` table.
# MAGIC - Databricks profile `aia-multi-agent` configured in `~/.databrickscfg`.

# COMMAND ----------

# MAGIC %pip install mlflow>=3.1 databricks-sdk>=0.40.0 --upgrade --quiet
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import uuid
from datetime import datetime
from databricks.sdk import WorkspaceClient

CATALOG = "aia_multi_agent_catalog"
SCHEMA = "ai_ops"
EXPERIMENT_NAME = "/Shared/aia-multi-agent/review-app"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create or set the MLflow Experiment for reviews

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f"MLflow Experiment: {experiment.name}")
print(f"  Experiment ID : {experiment.experiment_id}")
print(f"  Artifact URI  : {experiment.artifact_location}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the review logging function

# COMMAND ----------

def log_review_feedback(
    thread_id: str,
    user_id: str,
    question: str,
    intent: str,
    domain: str,
    agents_used: list,
    thumbs_up: bool,
    comment: str = None,
):
    """
    Logs human review feedback to both MLflow (for tracking) and the
    episodic_memory table (for agent learning).

    Parameters
    ----------
    thread_id : str
        The conversation thread ID being reviewed.
    user_id : str
        The reviewer's user ID.
    question : str
        The original user question.
    intent : str
        Classified intent of the question.
    domain : str
        Domain the question was routed to.
    agents_used : list[str]
        List of agents that handled the question.
    thumbs_up : bool
        True for positive feedback, False for negative.
    comment : str, optional
        Free-text reviewer comment.
    """
    episode_id = str(uuid.uuid4())
    rating = 5 if thumbs_up else 1
    outcome = "success" if thumbs_up else "failed"
    lesson = comment if comment else ("Positive feedback" if thumbs_up else "Negative feedback — needs investigation")

    # --- Log to MLflow ---
    with mlflow.start_run(run_name=f"review-{episode_id[:8]}"):
        mlflow.log_params({
            "episode_id": episode_id,
            "thread_id": thread_id,
            "user_id": user_id,
            "intent": intent,
            "domain": domain,
            "outcome": outcome,
        })
        mlflow.log_metrics({
            "user_rating": rating,
            "thumbs_up": int(thumbs_up),
        })
        if comment:
            mlflow.log_text(comment, "reviewer_comment.txt")
        mlflow.log_text(question, "original_question.txt")

    # --- Log to episodic_memory table ---
    agents_sql = ", ".join([f"'{a}'" for a in agents_used])
    escaped_question = question.replace("'", "''")
    escaped_lesson = lesson.replace("'", "''")

    insert_sql = f"""
    INSERT INTO {CATALOG}.{SCHEMA}.episodic_memory
    (episode_id, thread_id, user_id, question, intent, domain, agents_used, outcome, user_rating, lesson_learned)
    VALUES (
        '{episode_id}',
        '{thread_id}',
        '{user_id}',
        '{escaped_question}',
        '{intent}',
        '{domain}',
        ARRAY({agents_sql}),
        '{outcome}',
        {rating},
        '{escaped_lesson}'
    )
    """
    spark.sql(insert_sql)
    print(f"Review logged: episode_id={episode_id}, rating={rating}, outcome={outcome}")
    return episode_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register the review function as an MLflow model (optional)
# MAGIC
# MAGIC This wraps `log_review_feedback` in a pyfunc model so it can be called
# MAGIC from the Databricks Review App UI or via the model serving endpoint.

# COMMAND ----------

class ReviewAppModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper for the review feedback logger.
    Accepts a DataFrame with review columns and logs each row.
    """

    def predict(self, context, model_input):
        """
        Parameters
        ----------
        model_input : pd.DataFrame
            Expected columns: thread_id, user_id, question, intent, domain,
                              agents_used, thumbs_up, comment
        """
        import pandas as pd

        results = []
        for _, row in model_input.iterrows():
            agents = row.get("agents_used", "[]")
            if isinstance(agents, str):
                import json
                agents = json.loads(agents)

            episode_id = log_review_feedback(
                thread_id=str(row["thread_id"]),
                user_id=str(row["user_id"]),
                question=str(row["question"]),
                intent=str(row.get("intent", "unknown")),
                domain=str(row.get("domain", "unknown")),
                agents_used=agents,
                thumbs_up=bool(row["thumbs_up"]),
                comment=str(row.get("comment", "")),
            )
            results.append({"episode_id": episode_id, "status": "logged"})

        return pd.DataFrame(results)

# COMMAND ----------

# Log the review model to MLflow
with mlflow.start_run(run_name="review-app-model") as run:
    mlflow.pyfunc.log_model(
        artifact_path="review_app",
        python_model=ReviewAppModel(),
        pip_requirements=[
            "mlflow>=3.1",
            "databricks-sdk>=0.40.0",
            "pandas",
        ],
    )
    model_uri = f"runs:/{run.info.run_id}/review_app"
    print(f"Review App model logged: {model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quick test — log a sample review

# COMMAND ----------

test_episode = log_review_feedback(
    thread_id="test-thread-001",
    user_id="reviewer@example.com",
    question="What is the total claims amount by region for Q4 2025?",
    intent="simple_kpi",
    domain="claims",
    agents_used=["genie"],
    thumbs_up=True,
    comment="Accurate answer with correct regional breakdown.",
)
print(f"Test episode created: {test_episode}")

# COMMAND ----------

# Verify the record landed in episodic_memory
display(
    spark.sql(f"""
        SELECT episode_id, thread_id, user_id, question, outcome, user_rating, lesson_learned
        FROM {CATALOG}.{SCHEMA}.episodic_memory
        ORDER BY created_at DESC
        LIMIT 5
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary
# MAGIC
# MAGIC | Component | Status |
# MAGIC |-----------|--------|
# MAGIC | MLflow Experiment | Created at `{EXPERIMENT_NAME}` |
# MAGIC | `log_review_feedback()` | Available — logs to both MLflow and `episodic_memory` |
# MAGIC | Review App pyfunc model | Logged to MLflow — can be registered and served |
# MAGIC
# MAGIC **Next steps:**
# MAGIC 1. Register the model to Unity Catalog: `mlflow.register_model(model_uri, f"{CATALOG}.ai_ops.review_app")`
# MAGIC 2. Deploy as a serving endpoint to receive feedback from the Streamlit app.
# MAGIC 3. Wire the Streamlit thumbs-up/down buttons to call this endpoint.
