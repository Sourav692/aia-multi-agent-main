# AIA Customer 360 DeepAgent — Persistent Memory

## Agent Identity
- Name: AIA Customer 360 Supervisor
- Domain: Insurance Analytics (AIA Group)
- Platform: Databricks (Unity Catalog, Genie, Vector Search, MLflow)

## Data Assets
- Catalog: `aia_multi_agent_catalog`
- Genie default space: `01f1272d4ba6144ba75d868762f1925d` (Claims Analytics)
- Document VS index: `aia_multi_agent_catalog.ai_ops.policy_docs_vs`
- Context index: `aia_multi_agent_catalog.ai_ops.context_index_vs`

## Schema Knowledge
- Claims table has columns: claim_id, policy_id, customer_id, claim_date, claim_amount, claim_status, region, product_line
- Policies table has columns: policy_id, customer_id, product_type, premium_amount, start_date, end_date, region, status
- Common regions: Central, North, South, East, West
- Common product lines: Motor, Health, Life, Property, Travel

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profiles
- Sourav: Data engineer, focuses on insurance claims analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profiles
- Sourav: Data engineer, focuses on insurance claims analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profiles
- Sourav: Data engineer, focuses on insurance claims analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profiles
- Sourav: Data engineer, focuses on insurance claims analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## User Profile
- Name: Sourav
- Role: Data Engineer
- Focus Area: Insurance Claims Analytics

## Learned Patterns
- When Genie fails on a space, try the next resolved space before giving up
- For "loss ratio" queries, Genie needs both claims and premium data — frame the question clearly
- Document lookups for exclusions work best when including the product type in the query
- Users from Singapore office often focus on motor and property insurance
