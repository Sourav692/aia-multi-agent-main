# Skill: KPI Analysis

Use this skill when the user asks about claims counts, premium totals, loss ratios,
trends over time, or comparisons across regions/products.

## Steps
1. Call `classify_user_intent` to confirm this is a `simple_kpi` query
2. Call `resolve_data_assets` to find the best Genie Space
3. Call `get_episodic_lessons` for schema hints from past queries
4. Call `query_genie_space` with the top-ranked space_id
5. If the query fails, try the next space from resolved assets
6. Synthesize the SQL result into a natural language answer with specific numbers

## Tips
- Always include the SQL query in your internal reasoning so it can be traced
- If the user asks about "loss ratio", they mean claims_amount / premium_amount
- Region filters are case-sensitive in the Genie schema
- Time period filters: use "last quarter", "YTD", "last 12 months" as Genie understands these
