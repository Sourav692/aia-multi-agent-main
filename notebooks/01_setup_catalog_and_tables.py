# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Setup Unity Catalog & Upload Insurance Data
# MAGIC Creates schemas and tables for the AIA Multi-Agent Insurance demo.
# MAGIC
# MAGIC **Catalog:** `aia_multi_agent_catalog` (pre-created by FEVM)
# MAGIC **Schemas:** `bronze`, `silver`, `gold`, `ai_ops`, `ai_insights`

# COMMAND ----------

catalog = "aia_multi_agent_catalog"

schemas = ["bronze", "silver", "gold", "ai_ops", "ai_insights"]
for schema in schemas:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    print(f"Created schema: {catalog}.{schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upload CSV Data to Volumes

# COMMAND ----------

volume_path = f"/Volumes/{catalog}/bronze/raw_data"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.bronze.raw_data")

# Upload CSVs — run this after uploading files to the volume via UI or CLI:
# databricks fs cp data/*.csv dbfs:/Volumes/aia_multi_agent_catalog/bronze/raw_data/

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Bronze Tables

# COMMAND ----------

csv_files = ["products", "agents", "customers", "policies", "claims", "policy_documents"]

for table_name in csv_files:
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .csv(f"{volume_path}/{table_name}.csv"))

    df.write.mode("overwrite").saveAsTable(f"{catalog}.bronze.{table_name}")
    count = spark.table(f"{catalog}.bronze.{table_name}").count()
    print(f"Created {catalog}.bronze.{table_name} ({count} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Silver Tables (Enriched)

# COMMAND ----------

# Silver: enriched_claims — claims joined with customer, policy, and product info
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.silver.enriched_claims AS
SELECT
    cl.claim_id,
    cl.policy_id,
    cl.customer_id,
    cl.product_id,
    cl.claim_type,
    cl.claim_date,
    cl.claim_amount_usd,
    cl.approved_amount_usd,
    cl.claim_status,
    cl.region,
    cl.hospital_provider,
    cl.diagnosis_code,
    cl.processing_days,
    cl.fraud_score,
    cl.is_suspicious,
    cl.submitted_via,
    cl.settlement_date,
    cu.first_name,
    cu.last_name,
    cu.gender,
    cu.age,
    cu.segment AS customer_segment,
    cu.annual_income_usd,
    cu.occupation,
    cu.nps_score,
    po.policy_status,
    po.annual_premium_usd,
    po.sum_assured_usd,
    po.start_date AS policy_start_date,
    po.channel,
    pr.product_name,
    pr.category AS product_category,
    pr.base_annual_premium_usd
FROM {catalog}.bronze.claims cl
JOIN {catalog}.bronze.customers cu ON cl.customer_id = cu.customer_id
JOIN {catalog}.bronze.policies po ON cl.policy_id = po.policy_id
JOIN {catalog}.bronze.products pr ON cl.product_id = pr.product_id
""")
print(f"Created {catalog}.silver.enriched_claims")

# COMMAND ----------

# Silver: enriched_policies
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.silver.enriched_policies AS
SELECT
    po.policy_id,
    po.customer_id,
    po.product_id,
    po.agent_id,
    po.policy_status,
    po.start_date,
    po.end_date,
    po.annual_premium_usd,
    po.sum_assured_usd,
    po.payment_frequency,
    po.payment_method,
    po.underwriting_class,
    po.riders,
    po.channel,
    po.region,
    cu.first_name,
    cu.last_name,
    cu.segment AS customer_segment,
    cu.annual_income_usd,
    cu.age,
    cu.gender,
    pr.product_name,
    pr.category AS product_category,
    pr.base_annual_premium_usd,
    ag.agent_name,
    ag.certification_level AS agent_certification
FROM {catalog}.bronze.policies po
JOIN {catalog}.bronze.customers cu ON po.customer_id = cu.customer_id
JOIN {catalog}.bronze.products pr ON po.product_id = pr.product_id
JOIN {catalog}.bronze.agents ag ON po.agent_id = ag.agent_id
""")
print(f"Created {catalog}.silver.enriched_policies")

# COMMAND ----------

# Silver: customer_360 — aggregated customer view
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.silver.customer_360 AS
SELECT
    cu.customer_id,
    cu.first_name,
    cu.last_name,
    cu.gender,
    cu.age,
    cu.region,
    cu.segment,
    cu.annual_income_usd,
    cu.occupation,
    cu.customer_since,
    cu.nps_score,
    cu.risk_profile,
    COUNT(DISTINCT po.policy_id) AS total_policies,
    SUM(po.annual_premium_usd) AS total_annual_premium_usd,
    SUM(po.sum_assured_usd) AS total_sum_assured_usd,
    COUNT(DISTINCT CASE WHEN po.policy_status = 'Active' THEN po.policy_id END) AS active_policies,
    COUNT(DISTINCT cl.claim_id) AS total_claims,
    COALESCE(SUM(cl.claim_amount_usd), 0) AS total_claim_amount_usd,
    COALESCE(SUM(cl.approved_amount_usd), 0) AS total_approved_amount_usd,
    COALESCE(AVG(cl.fraud_score), 0) AS avg_fraud_score,
    COALESCE(MAX(cl.claim_date), NULL) AS last_claim_date
FROM {catalog}.bronze.customers cu
LEFT JOIN {catalog}.bronze.policies po ON cu.customer_id = po.customer_id
LEFT JOIN {catalog}.bronze.claims cl ON cu.customer_id = cl.customer_id
GROUP BY ALL
""")
print(f"Created {catalog}.silver.customer_360")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Gold Tables (Aggregated / Analytics-Ready)

# COMMAND ----------

# Gold: claims_summary — monthly aggregated claims by region, product, type
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.gold.claims_summary AS
SELECT
    DATE_TRUNC('MONTH', TO_DATE(claim_date)) AS claim_month,
    region,
    product_category,
    claim_type,
    claim_status,
    COUNT(*) AS claim_count,
    SUM(claim_amount_usd) AS total_claim_amount_usd,
    SUM(approved_amount_usd) AS total_approved_amount_usd,
    AVG(claim_amount_usd) AS avg_claim_amount_usd,
    AVG(processing_days) AS avg_processing_days,
    SUM(CASE WHEN is_suspicious THEN 1 ELSE 0 END) AS suspicious_claims_count,
    AVG(fraud_score) AS avg_fraud_score
FROM {catalog}.silver.enriched_claims
GROUP BY ALL
ORDER BY claim_month DESC
""")
print(f"Created {catalog}.gold.claims_summary")

# COMMAND ----------

# Gold: policy_performance — policy metrics by region, product, channel
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.gold.policy_performance AS
SELECT
    region,
    product_category,
    channel,
    policy_status,
    COUNT(*) AS policy_count,
    SUM(annual_premium_usd) AS total_premium_usd,
    AVG(annual_premium_usd) AS avg_premium_usd,
    SUM(sum_assured_usd) AS total_sum_assured_usd,
    COUNT(DISTINCT customer_id) AS unique_customers
FROM {catalog}.silver.enriched_policies
GROUP BY ALL
""")
print(f"Created {catalog}.gold.policy_performance")

# COMMAND ----------

# Gold: agent_performance — agent KPIs
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.gold.agent_performance AS
SELECT
    ag.agent_id,
    ag.agent_name,
    ag.region,
    ag.channel,
    ag.years_experience,
    ag.certification_level,
    COUNT(DISTINCT po.policy_id) AS policies_sold,
    SUM(po.annual_premium_usd) AS total_premium_sold_usd,
    AVG(po.annual_premium_usd) AS avg_premium_per_policy,
    COUNT(DISTINCT po.customer_id) AS unique_customers,
    COUNT(DISTINCT CASE WHEN po.policy_status = 'Active' THEN po.policy_id END) AS active_policies,
    COUNT(DISTINCT CASE WHEN po.policy_status IN ('Lapsed', 'Surrendered', 'Cancelled') THEN po.policy_id END) AS churned_policies,
    ROUND(COUNT(DISTINCT CASE WHEN po.policy_status IN ('Lapsed', 'Surrendered', 'Cancelled') THEN po.policy_id END) * 100.0 / NULLIF(COUNT(DISTINCT po.policy_id), 0), 2) AS churn_rate_pct
FROM {catalog}.bronze.agents ag
LEFT JOIN {catalog}.bronze.policies po ON ag.agent_id = po.agent_id
GROUP BY ALL
""")
print(f"Created {catalog}.gold.agent_performance")

# COMMAND ----------

# Gold: fraud_analysis — claims with high fraud scores
spark.sql(f"""
CREATE OR REPLACE TABLE {catalog}.gold.fraud_analysis AS
SELECT
    claim_id,
    policy_id,
    customer_id,
    product_id,
    claim_type,
    claim_date,
    claim_amount_usd,
    approved_amount_usd,
    claim_status,
    region,
    hospital_provider,
    diagnosis_code,
    fraud_score,
    is_suspicious,
    first_name,
    last_name,
    customer_segment,
    annual_income_usd,
    product_name,
    product_category,
    CASE
        WHEN fraud_score >= 0.8 THEN 'High Risk'
        WHEN fraud_score >= 0.5 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS fraud_risk_tier,
    claim_amount_usd / NULLIF(annual_income_usd, 0) AS claim_to_income_ratio
FROM {catalog}.silver.enriched_claims
WHERE fraud_score > 0.3 OR is_suspicious = true
ORDER BY fraud_score DESC
""")
print(f"Created {catalog}.gold.fraud_analysis")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Table Comments for Discovery

# COMMAND ----------

comments = {
    f"{catalog}.gold.claims_summary": "Monthly aggregated claims metrics by region, product category, claim type, and status. Use for claims trend analysis, anomaly detection, and KPI reporting.",
    f"{catalog}.gold.policy_performance": "Policy performance metrics by region, product category, channel, and status. Use for premium analysis, distribution channel effectiveness, and policy mix reporting.",
    f"{catalog}.gold.agent_performance": "Agent/advisor performance KPIs including policies sold, premium generated, and churn rates. Use for agent productivity analysis and incentive planning.",
    f"{catalog}.gold.fraud_analysis": "Claims flagged with elevated fraud risk scores. Use for fraud investigation, anomaly detection, and risk management.",
    f"{catalog}.silver.enriched_claims": "Claims enriched with customer demographics, policy details, and product information. Detailed claim-level data for deep analysis.",
    f"{catalog}.silver.enriched_policies": "Policies enriched with customer, product, and agent details. Detailed policy-level data for underwriting and portfolio analysis.",
    f"{catalog}.silver.customer_360": "360-degree customer view with aggregated policy and claims metrics. Use for customer segmentation, lifetime value, and retention analysis.",
}

for table, comment in comments.items():
    spark.sql(f"COMMENT ON TABLE {table} IS '{comment}'")
    print(f"Added comment to {table}")

# COMMAND ----------

# Summary
for schema in ["bronze", "silver", "gold"]:
    tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
    print(f"\n{schema.upper()} ({len(tables)} tables):")
    for t in tables:
        count = spark.table(f"{catalog}.{schema}.{t.tableName}").count()
        print(f"  {t.tableName}: {count} rows")
