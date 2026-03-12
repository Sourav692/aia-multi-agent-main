"""
AIA Multi-Agent System — MLflow Agent Evaluation Pipeline

Evaluates the aia-supervisor-agent serving endpoint across:
  - Intent classification accuracy
  - Domain resolution accuracy
  - Agent routing accuracy
  - Answer relevance (keyword coverage)
  - Latency

Usage:
    python evaluation/run_eval.py

Requires:
    pip install mlflow>=2.18 databricks-sdk pandas tabulate
"""

import json
import time
import os
import sys
import pandas as pd
import mlflow
from mlflow.metrics import make_metric
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EVAL_DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")
SERVING_ENDPOINT_NAME = "aia-supervisor-agent"
DATABRICKS_PROFILE = "e2-demo_latest"
EXPERIMENT_NAME = "aia-agent-evaluation"


# ---------------------------------------------------------------------------
# Databricks client setup
# ---------------------------------------------------------------------------
def _get_workspace_client():
    """Return a Databricks WorkspaceClient using the configured profile."""
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient(profile=DATABRICKS_PROFILE)


def _get_endpoint_url(w) -> str:
    """Build the serving endpoint invocation URL."""
    host = w.config.host.rstrip("/")
    return f"{host}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations"


# ---------------------------------------------------------------------------
# Call the serving endpoint
# ---------------------------------------------------------------------------
def call_agent(question: str, w=None) -> dict:
    """
    Send a question to the aia-supervisor-agent endpoint and return parsed
    response including custom_outputs, latency, and the raw answer text.
    """
    if w is None:
        w = _get_workspace_client()

    url = _get_endpoint_url(w)

    payload = {
        "input": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "custom_inputs": {},
    }

    headers = {
        "Content-Type": "application/json",
    }

    import requests

    token = w.config.token
    headers["Authorization"] = f"Bearer {token}"

    start = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        latency = time.time() - start
        body = resp.json()
    except Exception as exc:
        latency = time.time() - start
        return {
            "answer": f"ERROR: {exc}",
            "custom_outputs": {},
            "latency": latency,
            "raw_response": None,
            "error": str(exc),
        }

    # Parse the response — the endpoint returns ResponsesAgentResponse
    # with output items and custom_outputs at the top level.
    custom_outputs = body.get("custom_outputs", {})

    # Extract answer text from output items
    output_items = body.get("output", [])
    answer_text = ""
    for item in output_items:
        if isinstance(item, dict):
            text = item.get("text", "")
            # Skip the metadata JSON blob (second output item)
            if text and not text.startswith("{"):
                answer_text += text
            elif text and text.startswith("{"):
                # This is the metadata JSON — parse as fallback for custom_outputs
                try:
                    meta = json.loads(text)
                    if not custom_outputs:
                        custom_outputs = meta
                except json.JSONDecodeError:
                    pass

    if not answer_text and output_items:
        # Fallback: use first output item text regardless
        answer_text = output_items[0].get("text", "") if isinstance(output_items[0], dict) else str(output_items[0])

    return {
        "answer": answer_text,
        "custom_outputs": custom_outputs,
        "latency": latency,
        "raw_response": body,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Custom metric functions for mlflow.evaluate()
# ---------------------------------------------------------------------------
def _intent_accuracy_fn(predictions, targets, metrics):
    """Compute intent classification accuracy."""
    scores = []
    for pred, tgt in zip(predictions["predicted_intent"], targets["expected_intent"]):
        # Normalize: the codebase uses 'deep_analysis' internally but the
        # eval dataset may use alternate names for readability.
        intent_map = {
            "complex_analysis": "deep_analysis",
            "anomaly_detection": "deep_analysis",
            "visualization_request": "visualization",
        }
        expected = intent_map.get(tgt, tgt)
        actual = intent_map.get(str(pred), str(pred))
        scores.append(1.0 if actual == expected else 0.0)
    return np.mean(scores) if scores else 0.0


def _domain_accuracy_fn(predictions, targets, metrics):
    """Compute domain resolution accuracy."""
    scores = []
    for pred, tgt in zip(predictions["predicted_domain"], targets["expected_domain"]):
        if tgt == "unknown":
            # For unknown expected domain, any non-empty domain is acceptable
            scores.append(1.0)
        else:
            scores.append(1.0 if str(pred).lower() == str(tgt).lower() else 0.0)
    return np.mean(scores) if scores else 0.0


def _agent_routing_accuracy_fn(predictions, targets, metrics):
    """Compute agent routing accuracy (Jaccard similarity of agent sets)."""
    scores = []
    for pred, tgt in zip(predictions["predicted_agents"], targets["expected_agents"]):
        try:
            pred_set = set(json.loads(pred)) if isinstance(pred, str) else set(pred)
        except (json.JSONDecodeError, TypeError):
            pred_set = set()
        try:
            tgt_set = set(json.loads(tgt)) if isinstance(tgt, str) else set(tgt)
        except (json.JSONDecodeError, TypeError):
            tgt_set = set()

        if not tgt_set and not pred_set:
            scores.append(1.0)
        elif not tgt_set or not pred_set:
            scores.append(0.0)
        else:
            jaccard = len(pred_set & tgt_set) / len(pred_set | tgt_set)
            scores.append(jaccard)
    return np.mean(scores) if scores else 0.0


def _answer_relevance_fn(predictions, targets, metrics):
    """Compute answer relevance as fraction of expected keywords found."""
    scores = []
    for answer, expected_kw in zip(predictions["answer"], targets["expected_answer_contains"]):
        try:
            keywords = json.loads(expected_kw) if isinstance(expected_kw, str) else expected_kw
        except (json.JSONDecodeError, TypeError):
            keywords = []
        if not keywords:
            scores.append(1.0)
            continue
        answer_lower = str(answer).lower()
        hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
        scores.append(hits / len(keywords))
    return np.mean(scores) if scores else 0.0


def _mean_latency_fn(predictions, targets, metrics):
    """Compute mean latency in seconds."""
    latencies = predictions["latency"]
    return np.mean([float(x) for x in latencies]) if len(latencies) > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------
def load_eval_dataset(path: str = EVAL_DATASET_PATH) -> list[dict]:
    """Load the evaluation dataset from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def run_all_questions(dataset: list[dict]) -> pd.DataFrame:
    """
    Call the agent endpoint for every question and return a DataFrame
    with predictions alongside expected values.
    """
    w = _get_workspace_client()
    rows = []

    for i, example in enumerate(dataset):
        question = example["question"]
        print(f"[{i+1}/{len(dataset)}] {question[:80]}...")

        result = call_agent(question, w=w)
        co = result["custom_outputs"]

        # Extract nodes_executed and map to agent names
        nodes = co.get("nodes_executed", [])
        agent_nodes = [n for n in nodes if n in ("genie", "multi_tool", "analysis", "visualization")]

        rows.append({
            # Identifiers
            "question": question,
            # Predictions (from endpoint)
            "answer": result["answer"],
            "predicted_intent": co.get("intent", "unknown"),
            "predicted_domain": co.get("domain", "unknown"),
            "predicted_agents": json.dumps(agent_nodes),
            "latency": result["latency"],
            "warnings": json.dumps(co.get("warnings", [])),
            "error": result["error"],
            # Expected values (from eval dataset)
            "expected_intent": example["expected_intent"],
            "expected_domain": example["expected_domain"],
            "expected_agents": json.dumps(example["expected_agents"]),
            "expected_answer_contains": json.dumps(example["expected_answer_contains"]),
        })

        # Brief status
        status = "OK" if result["error"] is None else "ERR"
        print(f"  -> {status} | intent={co.get('intent','?')} | domain={co.get('domain','?')} | "
              f"agents={agent_nodes} | {result['latency']:.1f}s")

    return pd.DataFrame(rows)


def compute_per_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row metric columns used for the summary table."""
    intent_map = {
        "complex_analysis": "deep_analysis",
        "anomaly_detection": "deep_analysis",
        "visualization_request": "visualization",
    }

    intent_correct = []
    domain_correct = []
    routing_scores = []
    relevance_scores = []

    for _, row in df.iterrows():
        # Intent accuracy
        expected = intent_map.get(row["expected_intent"], row["expected_intent"])
        actual = intent_map.get(row["predicted_intent"], row["predicted_intent"])
        intent_correct.append(1 if actual == expected else 0)

        # Domain accuracy
        if row["expected_domain"] == "unknown":
            domain_correct.append(1)
        else:
            domain_correct.append(1 if str(row["predicted_domain"]).lower() == str(row["expected_domain"]).lower() else 0)

        # Routing accuracy (Jaccard)
        try:
            pred_set = set(json.loads(row["predicted_agents"]))
            tgt_set = set(json.loads(row["expected_agents"]))
            if not tgt_set and not pred_set:
                routing_scores.append(1.0)
            elif not tgt_set or not pred_set:
                routing_scores.append(0.0)
            else:
                routing_scores.append(len(pred_set & tgt_set) / len(pred_set | tgt_set))
        except Exception:
            routing_scores.append(0.0)

        # Answer relevance
        try:
            keywords = json.loads(row["expected_answer_contains"])
            answer_lower = str(row["answer"]).lower()
            hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
            relevance_scores.append(hits / len(keywords) if keywords else 1.0)
        except Exception:
            relevance_scores.append(0.0)

    df["intent_correct"] = intent_correct
    df["domain_correct"] = domain_correct
    df["routing_score"] = routing_scores
    df["relevance_score"] = relevance_scores
    return df


def print_summary(df: pd.DataFrame):
    """Print a formatted summary table of evaluation results."""
    separator = "=" * 90
    print(f"\n{separator}")
    print("AIA MULTI-AGENT EVALUATION SUMMARY")
    print(separator)

    # Aggregate metrics
    n = len(df)
    errors = df["error"].notna().sum()
    intent_acc = df["intent_correct"].mean()
    domain_acc = df["domain_correct"].mean()
    routing_acc = df["routing_score"].mean()
    relevance = df["relevance_score"].mean()
    mean_lat = df["latency"].mean()
    p95_lat = df["latency"].quantile(0.95)

    print(f"\nTotal examples:           {n}")
    print(f"Errors:                   {errors}")
    print(f"\n--- Accuracy Metrics ---")
    print(f"Intent accuracy:          {intent_acc:.1%}")
    print(f"Domain accuracy:          {domain_acc:.1%}")
    print(f"Agent routing accuracy:   {routing_acc:.1%}")
    print(f"Answer relevance:         {relevance:.1%}")
    print(f"\n--- Latency ---")
    print(f"Mean latency:             {mean_lat:.2f}s")
    print(f"P95 latency:              {p95_lat:.2f}s")

    # Per-intent breakdown
    print(f"\n--- Per-Intent Breakdown ---")
    intent_groups = df.groupby("expected_intent").agg(
        count=("question", "count"),
        intent_acc=("intent_correct", "mean"),
        routing_acc=("routing_score", "mean"),
        relevance=("relevance_score", "mean"),
        avg_latency=("latency", "mean"),
    ).reset_index()

    try:
        from tabulate import tabulate
        print(tabulate(intent_groups, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False))
    except ImportError:
        print(intent_groups.to_string(index=False))

    # Per-row detail
    print(f"\n--- Per-Question Detail ---")
    detail = df[["question", "predicted_intent", "expected_intent", "intent_correct",
                  "routing_score", "relevance_score", "latency"]].copy()
    detail["question"] = detail["question"].str[:60]
    try:
        from tabulate import tabulate
        print(tabulate(detail, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False))
    except ImportError:
        print(detail.to_string(index=False))

    print(separator)


def run_evaluation():
    """
    Main entry point: load dataset, call endpoint, compute metrics,
    log to MLflow, and print summary.
    """
    print("Loading evaluation dataset...")
    dataset = load_eval_dataset()
    print(f"Loaded {len(dataset)} evaluation examples.\n")

    print("Calling aia-supervisor-agent endpoint for each question...\n")
    results_df = run_all_questions(dataset)

    print("\nComputing per-row metrics...")
    results_df = compute_per_row_metrics(results_df)

    # ------------------------------------------------------------------
    # Log to MLflow
    # ------------------------------------------------------------------
    print(f"\nLogging results to MLflow experiment: {EXPERIMENT_NAME}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"agent_eval_{time.strftime('%Y%m%d_%H%M%S')}") as run:
        # Log aggregate metrics
        mlflow.log_metric("intent_accuracy", results_df["intent_correct"].mean())
        mlflow.log_metric("domain_accuracy", results_df["domain_correct"].mean())
        mlflow.log_metric("agent_routing_accuracy", results_df["routing_score"].mean())
        mlflow.log_metric("answer_relevance", results_df["relevance_score"].mean())
        mlflow.log_metric("mean_latency_seconds", results_df["latency"].mean())
        mlflow.log_metric("p95_latency_seconds", results_df["latency"].quantile(0.95))
        mlflow.log_metric("error_count", int(results_df["error"].notna().sum()))
        mlflow.log_metric("total_examples", len(results_df))

        # Log per-intent metrics
        for intent, group in results_df.groupby("expected_intent"):
            mlflow.log_metric(f"intent_accuracy__{intent}", group["intent_correct"].mean())
            mlflow.log_metric(f"routing_accuracy__{intent}", group["routing_score"].mean())
            mlflow.log_metric(f"relevance__{intent}", group["relevance_score"].mean())
            mlflow.log_metric(f"mean_latency__{intent}", group["latency"].mean())

        # Log the eval dataset and full results as artifacts
        mlflow.log_artifact(EVAL_DATASET_PATH, "eval_dataset")
        results_path = os.path.join(os.path.dirname(__file__), "eval_results.csv")
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, "eval_results")

        # Log params
        mlflow.log_param("endpoint", SERVING_ENDPOINT_NAME)
        mlflow.log_param("num_examples", len(dataset))
        mlflow.log_param("eval_dataset", EVAL_DATASET_PATH)

        # ---------------------------------------------------------------
        # Use mlflow.evaluate() with custom metrics for formal evaluation
        # ---------------------------------------------------------------
        # Prepare the eval DataFrame in the format mlflow.evaluate() expects
        eval_df = results_df[["question", "answer"]].copy()
        eval_df.rename(columns={"question": "inputs", "answer": "predictions"}, inplace=True)

        # Define custom metrics
        intent_accuracy_metric = make_metric(
            eval_fn=lambda predictions, targets, metrics: _intent_accuracy_fn(results_df, results_df, metrics),
            greater_is_better=True,
            name="intent_accuracy",
        )
        domain_accuracy_metric = make_metric(
            eval_fn=lambda predictions, targets, metrics: _domain_accuracy_fn(results_df, results_df, metrics),
            greater_is_better=True,
            name="domain_accuracy",
        )
        routing_accuracy_metric = make_metric(
            eval_fn=lambda predictions, targets, metrics: _agent_routing_accuracy_fn(results_df, results_df, metrics),
            greater_is_better=True,
            name="agent_routing_accuracy",
        )
        relevance_metric = make_metric(
            eval_fn=lambda predictions, targets, metrics: _answer_relevance_fn(results_df, results_df, metrics),
            greater_is_better=True,
            name="answer_relevance",
        )
        latency_metric = make_metric(
            eval_fn=lambda predictions, targets, metrics: _mean_latency_fn(results_df, results_df, metrics),
            greater_is_better=False,
            name="mean_latency",
        )

        eval_results = mlflow.evaluate(
            data=eval_df,
            predictions="predictions",
            extra_metrics=[
                intent_accuracy_metric,
                domain_accuracy_metric,
                routing_accuracy_metric,
                relevance_metric,
                latency_metric,
            ],
            model_type="text",
        )

        print(f"\nMLflow evaluate() metrics:")
        for k, v in eval_results.metrics.items():
            print(f"  {k}: {v}")

        print(f"\nMLflow Run ID: {run.info.run_id}")

    # Print the human-readable summary
    print_summary(results_df)

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_evaluation()
