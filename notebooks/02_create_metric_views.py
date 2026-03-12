# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Create Metric Views
# MAGIC Creates Unity Catalog Metric Views for governed KPI definitions.
# MAGIC These metric views are queryable by Genie Spaces and agents.

# COMMAND ----------

catalog = "aia_multi_agent_catalog"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Claims Metric Views

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_claims_count
COMMENT 'Total count of insurance claims by month, region, product category, claim type, and status'
AS
SELECT
    claim_month,
    region,
    product_category,
    claim_type,
    claim_status,
    claim_count
FROM {catalog}.gold.claims_summary
""")
print("Created mv_claims_count")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_claims_amount
COMMENT 'Total and average claim amounts in USD by month, region, product category, and claim type'
AS
SELECT
    claim_month,
    region,
    product_category,
    claim_type,
    claim_status,
    total_claim_amount_usd,
    total_approved_amount_usd,
    avg_claim_amount_usd,
    avg_processing_days
FROM {catalog}.gold.claims_summary
""")
print("Created mv_claims_amount")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_fraud_summary
COMMENT 'Fraud risk analysis showing suspicious claims count and average fraud scores by month, region, and product category'
AS
SELECT
    claim_month,
    region,
    product_category,
    suspicious_claims_count,
    avg_fraud_score,
    claim_count AS total_claims,
    ROUND(suspicious_claims_count * 100.0 / NULLIF(claim_count, 0), 2) AS suspicious_pct
FROM {catalog}.gold.claims_summary
WHERE suspicious_claims_count > 0
""")
print("Created mv_fraud_summary")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Policy Metric Views

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_policy_premium
COMMENT 'Total and average premium by region, product category, channel, and policy status'
AS
SELECT
    region,
    product_category,
    channel,
    policy_status,
    policy_count,
    total_premium_usd,
    avg_premium_usd,
    total_sum_assured_usd,
    unique_customers
FROM {catalog}.gold.policy_performance
""")
print("Created mv_policy_premium")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_policy_mix
COMMENT 'Policy distribution and premium mix by product category and distribution channel'
AS
SELECT
    product_category,
    channel,
    SUM(policy_count) AS total_policies,
    SUM(total_premium_usd) AS total_premium_usd,
    ROUND(SUM(total_premium_usd) / NULLIF(SUM(SUM(total_premium_usd)) OVER (), 0) * 100, 2) AS premium_share_pct,
    SUM(unique_customers) AS total_customers
FROM {catalog}.gold.policy_performance
GROUP BY product_category, channel
""")
print("Created mv_policy_mix")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent Metric Views

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_agent_productivity
COMMENT 'Agent productivity metrics including policies sold, premium generated, and churn rate by agent, region, and channel'
AS
SELECT
    agent_id,
    agent_name,
    region,
    channel,
    years_experience,
    certification_level,
    policies_sold,
    total_premium_sold_usd,
    avg_premium_per_policy,
    unique_customers,
    active_policies,
    churned_policies,
    churn_rate_pct
FROM {catalog}.gold.agent_performance
""")
print("Created mv_agent_productivity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customer Metric Views

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {catalog}.gold.mv_customer_segments
COMMENT 'Customer segment analysis showing policy count, premium, claims, and NPS by region and customer segment'
AS
SELECT
    region,
    segment,
    COUNT(*) AS customer_count,
    AVG(total_policies) AS avg_policies_per_customer,
    SUM(total_annual_premium_usd) AS total_premium_usd,
    AVG(total_annual_premium_usd) AS avg_premium_per_customer,
    SUM(total_claims) AS total_claims,
    SUM(total_claim_amount_usd) AS total_claim_amount_usd,
    AVG(nps_score) AS avg_nps_score,
    AVG(avg_fraud_score) AS avg_fraud_score
FROM {catalog}.silver.customer_360
GROUP BY region, segment
""")
print("Created mv_customer_segments")

# COMMAND ----------

# List all metric views
print("\nAll Metric Views created:")
views = spark.sql(f"""
    SELECT table_name, comment
    FROM {catalog}.information_schema.tables
    WHERE table_schema = 'gold'
    AND table_type = 'VIEW'
    AND table_name LIKE 'mv_%'
""").collect()

for v in views:
    print(f"  {v.table_name}: {v.comment}")
