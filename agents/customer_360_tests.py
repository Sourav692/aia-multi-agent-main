# Databricks notebook source
# MAGIC %md
# MAGIC # AIA Supervisor Agent — Test Notebook
# MAGIC
# MAGIC This notebook contains all test and demo cells for the **AIA Supervisor Agent**.
# MAGIC It loads the full agent implementation from `agent_code.py` via `%run`, so all
# MAGIC functions, state, and the compiled LangGraph are available here without duplication.
# MAGIC
# MAGIC **Sections:**
# MAGIC 1. Setup — load agent code
# MAGIC 2. Helper: `inspect_memory_tables`
# MAGIC 3. Memory Lifecycle Demo (short-term · long-term · episodic)
# MAGIC 4. Multi-Domain Genie Space Routing (Claims · Policies · Distribution · Customers)
# MAGIC 5. Multi-Turn Conversation Test
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup — Load Agent Code
# MAGIC
# MAGIC Runs `agent_code.py` so every function, the compiled LangGraph, and `agent` are
# MAGIC available in this session without re-importing anything manually.
# MAGIC

# COMMAND ----------

# MAGIC %run ./customer_360

# COMMAND ----------

# Verify the agent is loaded
print(f'Agent: {type(agent).__name__}')
print(f'Graph nodes: {list(graph.nodes)}')
print(f'LLM endpoint: {MODEL_ENDPOINT}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Helper: `inspect_memory_tables`
# MAGIC
# MAGIC Queries all three memory tables for a given `thread_id` / `user_id` and prints a
# MAGIC structured summary. Reused across every test section below.
# MAGIC

# COMMAND ----------

def inspect_memory_tables(thread_id, user_id, label=''):
    """Query and display all three memory tables for a thread / user."""
    sep = '=' * 70
    print(f'\n{sep}')
    print(f"  MEMORY STATE{f' — {label}' if label else ''}")
    print(sep)

    # 1. Short-term memory
    print('\n[1] SHORT-TERM MEMORY (ai_ops.conversations)')
    try:
        conv = _run_sql(
            f"SELECT thread_id, checkpoint_id, created_at, LEFT(state_json, 200) AS state_preview "
            f"FROM {CATALOG}.ai_ops.conversations "
            f"WHERE thread_id = '{thread_id}' ORDER BY created_at DESC LIMIT 5"
        )
        if conv['rows']:
            for r in conv['rows']:
                print(f"  checkpoint={r['checkpoint_id']}  created={r['created_at']}")
                print(f"    preview: {r['state_preview']}...")
            print(f"  -> {len(conv['rows'])} checkpoint(s) found")
        else:
            print('  -> No checkpoints found (empty)')
    except Exception as e:
        print(f'  -> Table not available: {str(e)[:100]}')

    # 2. Long-term memory
    print('\n[2] LONG-TERM MEMORY (ai_ops.user_memory)')
    try:
        mem = _run_sql(
            f"SELECT memory_key, memory_value, memory_type, confidence, updated_at "
            f"FROM {CATALOG}.ai_ops.user_memory "
            f"WHERE user_id = '{user_id}' ORDER BY updated_at DESC"
        )
        if mem['rows']:
            for r in mem['rows']:
                print(f"  {r['memory_key']:25s} = {r['memory_value']:30s}  "
                      f"(type={r['memory_type']}, conf={r['confidence']})")
            print(f"  -> {len(mem['rows'])} memory entries found")
        else:
            print('  -> No user memories found (empty)')
    except Exception as e:
        print(f'  -> Table not available: {str(e)[:100]}')

    # 3. Episodic memory
    print('\n[3] EPISODIC MEMORY (ai_ops.episodic_memory)')
    try:
        ep = _run_sql(
            f"SELECT episode_id, question, intent, domain, agents_used, outcome, lesson_learned, created_at "
            f"FROM {CATALOG}.ai_ops.episodic_memory "
            f"WHERE thread_id = '{thread_id}' ORDER BY created_at DESC LIMIT 5"
        )
        if ep['rows']:
            for r in ep['rows']:
                print(f"  episode={r['episode_id']}  intent={r['intent']}  "
                      f"domain={r['domain']}  outcome={r['outcome']}")
                print(f"    Q: {r['question'][:80]}")
                print(f"    agents: {r['agents_used']}  lesson: {r.get('lesson_learned', 'None')}")
            print(f"  -> {len(ep['rows'])} episode(s) found")
        else:
            print('  -> No episodes found (empty)')
    except Exception as e:
        print(f'  -> Table not available: {str(e)[:100]}')

    print(f'\n{sep}\n')

print('inspect_memory_tables() defined')


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 3. Memory Lifecycle Demo
# MAGIC
# MAGIC Demonstrates **end-to-end memory population and retrieval** across the three layers:
# MAGIC
# MAGIC | Layer | Table | Purpose |
# MAGIC |-------|-------|---------|
# MAGIC | **Short-term** | `ai_ops.conversations` | Delta checkpoints for multi-turn context |
# MAGIC | **Long-term** | `ai_ops.user_memory` | Persistent user preferences & facts |
# MAGIC | **Episodic** | `ai_ops.episodic_memory` | Interaction logs & lessons learned |
# MAGIC
# MAGIC **Flow:**
# MAGIC 1. Clean prior demo data → check baseline
# MAGIC 2. Conversational request with personal facts → writes to all 3 tables
# MAGIC 3. Inspect tables
# MAGIC 4. Follow-up document lookup → reads short-term + long-term, writes new checkpoint & episode
# MAGIC 5. Inspect tables
# MAGIC 6. Third KPI request → all 3 layers read + written
# MAGIC 7. Final inspection & summary
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 — Prepare: Clean Prior Demo Data & Check Baseline
# MAGIC

# COMMAND ----------

DEMO_THREAD_ID = 'memory-demo-thread-001'
DEMO_USER_ID = 'demo-user-sarah'

print('Cleaning up prior demo data...')
_cleanup = [
    (f"DELETE FROM {CATALOG}.ai_ops.conversations WHERE thread_id = '{DEMO_THREAD_ID}'", 'conversations'),
    (f"DELETE FROM {CATALOG}.ai_ops.user_memory WHERE user_id = '{DEMO_USER_ID}'", 'user_memory'),
    (f"DELETE FROM {CATALOG}.ai_ops.episodic_memory WHERE thread_id = '{DEMO_THREAD_ID}'", 'episodic_memory'),
]
for _sql, _label in _cleanup:
    try:
        _run_sql(_sql)
        print(f'  Cleared {_label}')
    except Exception as _e:
        print(f'  Warning — could not clear {_label}: {_e}')

inspect_memory_tables(DEMO_THREAD_ID, DEMO_USER_ID, label='BASELINE (before any requests)')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 — First Request: Conversational Message with Personal Facts
# MAGIC
# MAGIC Intent: `conversational` → agent **writes** a checkpoint, **extracts & saves** user facts
# MAGIC (name, role, region, product preference), and **logs** an episode.
# MAGIC

# COMMAND ----------

request_1 = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        "Hi, I'm Sarah, a regional claims manager based in Singapore. "
        'I prefer concise responses and usually focus on the Health and '
        'Critical Illness product lines.'
    )}],
    custom_inputs={'thread_id': DEMO_THREAD_ID, 'user_id': DEMO_USER_ID}
)

print('Sending Request 1 (conversational with personal facts)...')
print(f'  thread_id: {DEMO_THREAD_ID}  |  user_id: {DEMO_USER_ID}')
print(f"  message: {getattr(request_1.input[0], 'content', '')}\n")

response_1 = agent.predict(request_1)

for item in response_1.output:
    item_id = getattr(item, 'id', '')
    text = getattr(item, 'text', '')
    if item_id == 'msg_answer':
        print('=== AGENT RESPONSE ===')
        print(text)
    elif item_id == 'msg_metadata' and text.strip():
        try:
            meta = json.loads(text)
            print('\n=== METADATA ===')
            print(f"  Intent:     {meta.get('intent')}")
            print(f"  Nodes:      {meta.get('nodes_executed')}")
            print(f"  Checkpoint: {meta.get('checkpoint_id')}")
        except json.JSONDecodeError:
            print(f'\n=== METADATA (raw) ===\n  {text[:300]}')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 — Inspect Memory Tables After Request 1
# MAGIC
# MAGIC Expected: **1 checkpoint**, extracted user facts (name, role, region, product), **1 episode** (`intent=conversational`).
# MAGIC

# COMMAND ----------

import time
time.sleep(3)  # allow async writes to settle
inspect_memory_tables(DEMO_THREAD_ID, DEMO_USER_ID, label='AFTER REQUEST 1 (conversational intro)')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 — Second Request: Document Lookup on the Same Thread
# MAGIC
# MAGIC Uses **short-term memory** (checkpoint) for conversation context + **long-term memory**
# MAGIC (Sarah's preferences) for personalisation. Writes a new checkpoint and a new episode.
# MAGIC

# COMMAND ----------

request_2 = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': 'What are the exclusions for the Critical Illness plans?'}],
    custom_inputs={'thread_id': DEMO_THREAD_ID, 'user_id': DEMO_USER_ID}
)

print('Sending Request 2 (document lookup — same thread)...')
print(f'  thread_id: {DEMO_THREAD_ID}  (loads prior checkpoint)')
print(f'  user_id:   {DEMO_USER_ID}    (loads saved preferences)')
print(f"  message: {getattr(request_2.input[0], 'content', '')}\n")

_memory_cache_ts = 0  # force fresh load from tables

response_2 = agent.predict(request_2)

for item in response_2.output:
    item_id = getattr(item, 'id', '')
    text = getattr(item, 'text', '')
    if item_id == 'msg_answer':
        print('=== AGENT RESPONSE ===')
        print(text)
    elif item_id == 'msg_metadata' and text.strip():
        try:
            meta = json.loads(text)
            print('\n=== METADATA ===')
            print(f"  Intent:     {meta.get('intent')}")
            print(f"  Nodes:      {meta.get('nodes_executed')}")
            print(f"  Checkpoint: {meta.get('checkpoint_id')}")
        except json.JSONDecodeError:
            print(f'\n=== METADATA (raw) ===\n  {text[:300]}')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 — Inspect Memory Tables After Request 2
# MAGIC
# MAGIC Expected: **2 checkpoints** (latest has 2-turn history), user memory unchanged,
# MAGIC **2 episodes** (second has `intent=document_lookup`).
# MAGIC

# COMMAND ----------

time.sleep(3)
inspect_memory_tables(DEMO_THREAD_ID, DEMO_USER_ID, label='AFTER REQUEST 2 (document lookup)')

print('--- Verifying Short-Term Memory Contains Prior Conversation ---')
checkpoint = _load_checkpoint(DEMO_THREAD_ID)
if checkpoint and checkpoint.get('messages'):
    print(f"  Checkpoint has {len(checkpoint['messages'])} messages:")
    for i, msg in enumerate(checkpoint['messages']):
        print(f"    [{i}] {msg.get('role', '?')}: {msg.get('content', '')[:100]}...")
    print(f"  Stored intent: {checkpoint.get('intent')}")
    print(f"  Stored domain: {checkpoint.get('domain')}")
else:
    print('  No checkpoint found')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 — Third Request: All Memory Layers Working Together
# MAGIC
# MAGIC **Short-term** loads 4-message history · **Long-term** personalises with Sarah's prefs ·
# MAGIC **Episodic** injects prior lessons from the `document_lookup` episode.
# MAGIC

# COMMAND ----------

request_3 = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'How do I file a claim under the Critical Illness plans? '
        'Also, what is the waiting period?'
    )}],
    custom_inputs={'thread_id': DEMO_THREAD_ID, 'user_id': DEMO_USER_ID}
)

print('Sending Request 3 (follow-up using all memory layers)...')
print(f'  thread_id: {DEMO_THREAD_ID}')
print(f'  user_id:   {DEMO_USER_ID}')
print(f"  message: {getattr(request_3.input[0], 'content', '')}\n")

_memory_cache_ts = 0

response_3 = agent.predict(request_3)

for item in response_3.output:
    item_id = getattr(item, 'id', '')
    text = getattr(item, 'text', '')
    if item_id == 'msg_answer':
        print('=== AGENT RESPONSE ===')
        print(text)
        print('\n--- Personalization Checks ---')
        lower = text.lower()
        checks = {
            "Addressed by name ('Sarah')": 'sarah' in lower,
            'Concise reply (< 500 chars)': len(text) < 500,
            'Mentions Critical Illness': 'critical illness' in lower,
        }
        for check, passed in checks.items():
            print(f"  [{'PASS' if passed else 'CHECK'}] {check}")
    elif item_id == 'msg_metadata' and text.strip():
        try:
            meta = json.loads(text)
            print('\n=== METADATA ===')
            print(f"  Intent:     {meta.get('intent')}")
            print(f"  Nodes:      {meta.get('nodes_executed')}")
            print(f"  Checkpoint: {meta.get('checkpoint_id')}")
        except json.JSONDecodeError:
            print(f'\n=== METADATA (raw) ===\n  {text[:300]}')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 — Final Inspection & Summary
# MAGIC
# MAGIC | Table | Expected |
# MAGIC |-------|----------|
# MAGIC | `ai_ops.conversations` | 3 checkpoints |
# MAGIC | `ai_ops.user_memory` | Facts extracted from Request 1 (name, role, region, product pref) |
# MAGIC | `ai_ops.episodic_memory` | 3 episodes: `conversational` → `document_lookup` → `document_lookup` |
# MAGIC

# COMMAND ----------

time.sleep(3)
inspect_memory_tables(DEMO_THREAD_ID, DEMO_USER_ID, label='FINAL STATE (after 3 requests)')

print('\n' + '=' * 70)
print('  MEMORY LIFECYCLE SUMMARY')
print('=' * 70)

try:
    c = _run_sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.ai_ops.conversations WHERE thread_id = '{DEMO_THREAD_ID}'")['rows'][0]['cnt']
    m = _run_sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.ai_ops.user_memory WHERE user_id = '{DEMO_USER_ID}'")['rows'][0]['cnt']
    e = _run_sql(f"SELECT COUNT(*) AS cnt FROM {CATALOG}.ai_ops.episodic_memory WHERE thread_id = '{DEMO_THREAD_ID}'")['rows'][0]['cnt']
    print(f"""
  Requests sent:                3
  ─────────────────────────────────────────────
  Conversation checkpoints:     {c}  (short-term memory)
  User memory entries:          {m}  (long-term memory)
  Episodic memory episodes:     {e}  (episodic memory)
  ─────────────────────────────────────────────

  Request 1 — conversational intro:
    WRITE: checkpoint, user facts, episode
    READ:  none (first interaction)

  Request 2 — document lookup:
    WRITE: new checkpoint, new episode
    READ:  checkpoint (prior context), user_memory (preferences)

  Request 3 — follow-up CI query:
    WRITE: new checkpoint, new episode
    READ:  checkpoint (4-message history), user_memory (name + prefs), episodic lessons
""")
except Exception as e:
    print(f'  Could not generate summary: {str(e)[:200]}')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Cleanup (Optional) — Reset Demo Data
# MAGIC
# MAGIC Uncomment and run to wipe all demo data so the test can be re-run from scratch.
# MAGIC

# COMMAND ----------

# _run_sql(f"DELETE FROM {CATALOG}.ai_ops.conversations WHERE thread_id = '{DEMO_THREAD_ID}'")
# _run_sql(f"DELETE FROM {CATALOG}.ai_ops.user_memory WHERE user_id = '{DEMO_USER_ID}'")
# _run_sql(f"DELETE FROM {CATALOG}.ai_ops.episodic_memory WHERE thread_id = '{DEMO_THREAD_ID}'")
# _memory_cache_ts = 0
# print('Demo data cleaned up. Ready to re-run.')


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 4. Multi-Domain Genie Space Routing
# MAGIC
# MAGIC Each query targets a different Genie Space registered in the Context Index.
# MAGIC The supervisor classifies intent as `simple_kpi`, resolves the best-matching
# MAGIC space via Vector Search, and routes to `route_to_genie`.
# MAGIC
# MAGIC | # | Target Space | Domain |
# MAGIC |---|---|---|
# MAGIC | 1 | Claims Analytics | `claims` |
# MAGIC | 2 | Policy & Underwriting | `policies` |
# MAGIC | 3 | Distribution & Channels | `distribution` |
# MAGIC | 4 | Customer Analytics | `customers` |
# MAGIC

# COMMAND ----------

GENIE_THREAD_ID = 'genie-routing-demo-thread'
GENIE_USER_ID = 'demo-user-genie'

def _print_routing_response(response, checks: dict):
    """Print agent response + routing checks from a ResponsesAgentResponse."""
    for item in response.output:
        item_id = getattr(item, 'id', '')
        text = getattr(item, 'text', '')
        if item_id == 'msg_answer':
            print('=== AGENT RESPONSE ===')
            print(text)
            print('\n--- Routing Checks ---')
            lower = text.lower()
            for label, condition in checks.items():
                passed = condition(lower, text)
                print(f"  [{'PASS' if passed else 'CHECK'}] {label}")
        elif item_id == 'msg_metadata' and text.strip():
            try:
                meta = json.loads(text)
                genie = meta.get('agent_details', {}).get('genie', {})
                print('\n=== METADATA ===')
                print(f"  Intent: {meta.get('intent')}  |  Domain: {meta.get('domain')}")
                print(f"  Nodes:  {meta.get('nodes_executed')}")
                print(f"  Genie status:     {genie.get('status', 'N/A')}")
                print(f"  Genie space used: {genie.get('space_id', 'N/A')}")
                print(f"  Genie display:    {genie.get('display_name', 'N/A')}")
                print(f"  SQL preview:      {str(genie.get('sql', 'N/A'))[:120]}")
            except json.JSONDecodeError:
                print(f'\n=== METADATA (raw) ===\n  {text[:300]}')

print('Routing helper _print_routing_response() defined')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 1 — Claims Analytics Space
# MAGIC
# MAGIC Target: Claims Analytics (`01f1272d4ba6144ba75d868762f1925d`).
# MAGIC Semantic match: *claims*, *region*, *count*.
# MAGIC

# COMMAND ----------

_memory_cache_ts = 0

request_claims = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'What is the total number of claims by region for the last 12 months?'
    )}],
    custom_inputs={'thread_id': GENIE_THREAD_ID, 'user_id': GENIE_USER_ID}
)

print('Sending Claims Analytics query...')
print(f"  message: {getattr(request_claims.input[0], 'content', '')}\n")

response_claims = agent.predict(request_claims)

_print_routing_response(response_claims, checks={
    'Mentions claims': lambda lower, _: 'claim' in lower,
    'Mentions region': lambda lower, _: 'region' in lower,
    'Contains numeric data': lambda _, text: any(c.isdigit() for c in text),
})


# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 2 — Policy & Underwriting Space
# MAGIC
# MAGIC Target: Policy & Underwriting (`01f1272d4c6b1fb49223785ab841befd`).
# MAGIC Semantic match: *premium*, *distribution channel*.
# MAGIC

# COMMAND ----------

_memory_cache_ts = 0

request_policies = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'What is the total premium volume by distribution channel?'
    )}],
    custom_inputs={'thread_id': GENIE_THREAD_ID, 'user_id': GENIE_USER_ID}
)

print('Sending Policy & Underwriting query...')
print(f"  message: {getattr(request_policies.input[0], 'content', '')}\n")

response_policies = agent.predict(request_policies)

_print_routing_response(response_policies, checks={
    'Mentions premium': lambda lower, _: 'premium' in lower,
    'Mentions policy/policies': lambda lower, _: 'policy' in lower or 'policies' in lower,
    'Contains numeric data': lambda _, text: any(c.isdigit() for c in text),
})


# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 3 — Distribution & Channels Space
# MAGIC
# MAGIC Target: Distribution & Channels (`01f1272d4d271203ad122e9280470248`).
# MAGIC Semantic match: *top-performing agents*, *channel contribution*, *commission*.
# MAGIC

# COMMAND ----------

_memory_cache_ts = 0

request_distribution = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'Who are the top 10 performing agents by premium collected? '
        'Also show me channel contribution percentages and commission breakdown for last quarter.'
    )}],
    custom_inputs={'thread_id': GENIE_THREAD_ID, 'user_id': GENIE_USER_ID}
)

print('Sending Distribution & Channels query...')
print(f"  message: {getattr(request_distribution.input[0], 'content', '')}\n")

response_distribution = agent.predict(request_distribution)

_print_routing_response(response_distribution, checks={
    'Mentions agent(s)': lambda lower, _: 'agent' in lower,
    'Mentions channel': lambda lower, _: 'channel' in lower,
    'Mentions premium or commission': lambda lower, _: 'premium' in lower or 'commission' in lower,
})


# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 4 — Customer Analytics Space
# MAGIC
# MAGIC Target: Customer Analytics (`01f1272d4de1188cac8feeb7e71bdb69`).
# MAGIC Semantic match: *customer segments*, *retention rate*, *demographics*.
# MAGIC

# COMMAND ----------

_memory_cache_ts = 0

request_customers = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'Which customer segments have the highest claim frequency? '
        'What is the retention rate by segment and show the demographic breakdown '
        'of our top-tier customers.'
    )}],
    custom_inputs={'thread_id': GENIE_THREAD_ID, 'user_id': GENIE_USER_ID}
)

print('Sending Customer Analytics query...')
print(f"  message: {getattr(request_customers.input[0], 'content', '')}\n")

response_customers = agent.predict(request_customers)

_print_routing_response(response_customers, checks={
    'Mentions customer': lambda lower, _: 'customer' in lower,
    'Mentions segment': lambda lower, _: 'segment' in lower,
    'Mentions retention': lambda lower, _: 'retention' in lower or 'retain' in lower,
})


# COMMAND ----------

# MAGIC %md
# MAGIC ### Post-Routing Memory Inspection
# MAGIC
# MAGIC Inspect memory state after the four Genie routing queries.
# MAGIC

# COMMAND ----------

time.sleep(3)
inspect_memory_tables(GENIE_THREAD_ID, GENIE_USER_ID, label='AFTER GENIE ROUTING QUERIES')

print('--- Latest checkpoint ---')
checkpoint = _load_checkpoint(GENIE_THREAD_ID)
if checkpoint and checkpoint.get('messages'):
    print(f"  {len(checkpoint['messages'])} messages in latest checkpoint")
    for i, msg in enumerate(checkpoint['messages']):
        print(f"    [{i}] {msg.get('role', '?')}: {msg.get('content', '')[:100]}...")
    print(f"  intent={checkpoint.get('intent')}  domain={checkpoint.get('domain')}")
else:
    print('  No checkpoint found')


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 5. Multi-Turn Conversation Test
# MAGIC
# MAGIC Three back-to-back queries on the **same thread** to verify:
# MAGIC - Context carry-over (Turn 3 cross-references results from Turns 1 & 2)
# MAGIC - Short-term memory checkpoint growth
# MAGIC - Correct intent resolution of follow-up shorthand questions
# MAGIC

# COMMAND ----------

CONV_THREAD_ID = 'conv-test-thread-001'
CONV_USER_ID = 'conv-test-user-001'
_memory_cache_ts = 0

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 1 — Claims by Region (last 3 months)
# MAGIC

# COMMAND ----------

request_turn1 = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'What is the total number of claims submitted by region for the last 3 months?'
    )}],
    custom_inputs={'thread_id': CONV_THREAD_ID, 'user_id': CONV_USER_ID}
)

print('[Turn 1] Claims by region...')
response_turn1 = agent.predict(request_turn1)

for item in response_turn1.output:
    item_id = getattr(item, 'id', '')
    text = getattr(item, 'text', '')
    if item_id == 'msg_answer':
        print('=== AGENT RESPONSE ===')
        print(text)
    elif item_id == 'msg_metadata' and text.strip():
        try:
            meta = json.loads(text)
            print(f"  Intent: {meta.get('intent')}  Domain: {meta.get('domain')}")
            print(f"  Nodes:  {meta.get('nodes_executed')}")
        except json.JSONDecodeError:
            pass


# COMMAND ----------

response_turn1.output[0].content[0]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 2 — Premium by Product Type (same thread)
# MAGIC

# COMMAND ----------

_memory_cache_ts = 0

request_turn2 = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'Show me the total premium collected by product type across all regions.'
    )}],
    custom_inputs={'thread_id': CONV_THREAD_ID, 'user_id': CONV_USER_ID}
)

print('[Turn 2] Premium by product type...')
response_turn2 = agent.predict(request_turn2)

for item in response_turn2.output:
    item_id = getattr(item, 'id', '')
    text = getattr(item, 'text', '')
    if item_id == 'msg_answer':
        print('=== AGENT RESPONSE ===')
        print(text)
    elif item_id == 'msg_metadata' and text.strip():
        try:
            meta = json.loads(text)
            print(f"  Intent: {meta.get('intent')}  Domain: {meta.get('domain')}")
            print(f"  Nodes:  {meta.get('nodes_executed')}")
        except json.JSONDecodeError:
            pass


# COMMAND ----------

response_turn2.output[0].content[0]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Turn 3 — Cross-Reference Follow-Up (tests context carry-over)
# MAGIC
# MAGIC This follow-up question references both previous results. The agent must load the
# MAGIC short-term checkpoint to resolve the context correctly.
# MAGIC

# COMMAND ----------

_memory_cache_ts = 0

request_turn3 = ResponsesAgentRequest(
    input=[{'role': 'user', 'content': (
        'Based on those two results — the claims by region and premium by product — '
        'which regions are generating the most premium but also have the highest claim volumes? '
        'Are there any regions where we might be underpriced?'
    )}],
    custom_inputs={'thread_id': CONV_THREAD_ID, 'user_id': CONV_USER_ID}
)

print('[Turn 3] Cross-reference follow-up...')
response_turn3 = agent.predict(request_turn3)

for item in response_turn3.output:
    item_id = getattr(item, 'id', '')
    text = getattr(item, 'text', '')
    if item_id == 'msg_answer':
        print('=== AGENT RESPONSE ===')
        print(text)
        print('\n--- Context Carry-Over Checks ---')
        lower = text.lower()
        checks = {
            'References region data': 'region' in lower,
            'References premium data': 'premium' in lower,
            'References claims data': 'claim' in lower,
            'Addresses underpricing': any(w in lower for w in ['underpric', 'priced', 'pricing']),
        }
        for check, passed in checks.items():
            print(f"  [{'PASS' if passed else 'CHECK'}] {check}")
    elif item_id == 'msg_metadata' and text.strip():
        try:
            meta = json.loads(text)
            print(f"  Intent: {meta.get('intent')}  Domain: {meta.get('domain')}")
            print(f"  Nodes:  {meta.get('nodes_executed')}")
        except json.JSONDecodeError:
            pass


# COMMAND ----------

response_turn3.output[0].content[0]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Post-Conversation Memory State
# MAGIC

# COMMAND ----------

time.sleep(3)
inspect_memory_tables(CONV_THREAD_ID, CONV_USER_ID, label='AFTER 3-TURN CONVERSATION')

print('--- Checkpoint message count ---')
cp = _load_checkpoint(CONV_THREAD_ID)
if cp and cp.get('messages'):
    print(f"  {len(cp['messages'])} messages stored in latest checkpoint")
    for i, msg in enumerate(cp['messages']):
        print(f"    [{i}] {msg.get('role', '?')}: {msg.get('content', '')[:100]}...")
else:
    print('  No checkpoint found')


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ## 6. 5-Turn Conversational Demo Sequences
# MAGIC
# MAGIC Three end-to-end conversation flows that verify:
# MAGIC - **Multi-domain routing** across Claims, Policy, Distribution, and Customer Genie Spaces
# MAGIC - **Context carry-forward** via short-term memory checkpoints
# MAGIC - **Comparative synthesis** — Turn 5 always cross-references prior results
# MAGIC
# MAGIC | Sequence | Persona | Genie Spaces touched |
# MAGIC |----------|---------|----------------------|
# MAGIC | 6a — CFO Loss Ratio Review | CFO | Claims → Policy → Claims → Policy → Synthesis |
# MAGIC | 6b — Distribution Head Strategy | Head of Distribution | Distribution → Policy → Distribution → Customer → Synthesis |
# MAGIC | 6c — Actuarial Motor Risk | Chief Risk Officer | Claims → Claims → Policy → Customer → Synthesis |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shared: Response printer with context-carry checks

# COMMAND ----------

def _print_turn(label, response, checks=None):
    """Print one conversation turn: answer + optional checks + metadata."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print('='*70)
    for item in response.output:
        item_id  = getattr(item, 'id',   '')
        text     = getattr(item, 'text', '')
        if item_id == 'msg_answer':
            print(text)
            if checks:
                print("\n--- Checks ---")
                lower = text.lower()
                for lbl, fn in checks.items():
                    passed = fn(lower, text)
                    print(f"  [{'PASS' if passed else 'FAIL'}] {lbl}")
        elif item_id == 'msg_metadata' and text.strip():
            try:
                meta  = json.loads(text)
                genie = meta.get('agent_details', {}).get('genie', {})
                print(f"\n  Intent: {meta.get('intent')}  Domain: {meta.get('domain')}  "
                      f"Nodes: {meta.get('nodes_executed')}")
                if genie.get('space_id'):
                    print(f"  Genie: {genie.get('display_name','?')} ({genie.get('space_id','')})")
                    print(f"  SQL:   {str(genie.get('sql',''))[:120]}")
            except json.JSONDecodeError:
                pass

print("_print_turn() helper defined")


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ### 6a — CFO Loss Ratio Review
# MAGIC
# MAGIC **Persona:** CFO reviewing which Asian markets are at repricing risk.  
# MAGIC **Route:** Claims Analytics → Policy Underwriting → Claims (drill-down) → Policy (drill-down) → Synthesis
# MAGIC
# MAGIC | Turn | Query | Expected Genie Space |
# MAGIC |------|-------|----------------------|
# MAGIC | 1 | Claims by region | AIA Claims Analytics |
# MAGIC | 2 | Premium by product type | AIA Policy Underwriting |
# MAGIC | 3 | HK claims by product category | AIA Claims Analytics |
# MAGIC | 4 | HK premium by product category | AIA Policy Underwriting |
# MAGIC | 5 | Compare both — loss ratio + repricing risk | Synthesis (memory) |
# MAGIC

# COMMAND ----------

SEQ1_THREAD = 'demo-cfr-loss-ratio-001'
SEQ1_USER   = 'demo-cfr-user-001'
_memory_cache_ts = 0

# Clean prior runs
for tbl, col in [('conversations','thread_id'), ('episodic_memory','thread_id'), ('user_memory','user_id')]:
    _val = SEQ1_THREAD if col == 'thread_id' else SEQ1_USER
    try:
        _run_sql(f"DELETE FROM {CATALOG}.ai_ops.{tbl} WHERE {col} = '{_val}'")
    except Exception:
        pass

print("Sequence 6a — CFO Loss Ratio Review")
print(f"  thread={SEQ1_THREAD}  user={SEQ1_USER}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 1 — Claims by Region (Claims Analytics Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s1t1 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'What is the total number of claims submitted by region?'}],
    custom_inputs={'thread_id': SEQ1_THREAD, 'user_id': SEQ1_USER}
))
_print_turn("[Seq1 Turn1] Claims by region", resp_s1t1, checks={
    'Mentions region':         lambda l,_: 'region' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
    'Routes to Claims Genie':  lambda l,_: 'claim' in l,
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 2 — Premium by Product Type (Policy Underwriting Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s1t2 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'Show me the total premium collected by product type across all regions.'}],
    custom_inputs={'thread_id': SEQ1_THREAD, 'user_id': SEQ1_USER}
))
_print_turn("[Seq1 Turn2] Premium by product type", resp_s1t2, checks={
    'Mentions premium':        lambda l,_: 'premium' in l,
    'Mentions product':        lambda l,_: 'product' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 3 — HK Claims Drill-Down (Claims Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s1t3 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'For Hong Kong specifically, break down total claims by product category.'}],
    custom_inputs={'thread_id': SEQ1_THREAD, 'user_id': SEQ1_USER}
))
_print_turn("[Seq1 Turn3] HK claims by product", resp_s1t3, checks={
    'Mentions Hong Kong':      lambda l,_: 'hong kong' in l,
    'Mentions product':        lambda l,_: 'product' in l or 'categor' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 4 — HK Premium Drill-Down (Policy Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s1t4 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'Now show me Hong Kong premium broken down by product category as well.'}],
    custom_inputs={'thread_id': SEQ1_THREAD, 'user_id': SEQ1_USER}
))
_print_turn("[Seq1 Turn4] HK premium by product", resp_s1t4, checks={
    'Mentions Hong Kong':      lambda l,_: 'hong kong' in l,
    'Mentions premium':        lambda l,_: 'premium' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 5 — Comparative Synthesis (references Turns 1–4)

# COMMAND ----------

_memory_cache_ts = 0
resp_s1t5 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':(
        'Based on the claims volumes we saw for Hong Kong and Singapore (Turn 1), '
        'and the premium data from Turns 2 and 4 — which of these two markets has the '
        'worse loss ratio? And which product category in Hong Kong looks most at risk for repricing?'
    )}],
    custom_inputs={'thread_id': SEQ1_THREAD, 'user_id': SEQ1_USER}
))
_print_turn("[Seq1 Turn5] Loss ratio synthesis", resp_s1t5, checks={
    'References claims data':  lambda l,_: 'claim' in l,
    'References premium data': lambda l,_: 'premium' in l,
    'Mentions HK or Singapore':lambda l,_: 'hong kong' in l or 'singapore' in l,
    'Addresses repricing':     lambda l,_: any(w in l for w in ['repric','underpric','pric']),
})

import time; time.sleep(2)
inspect_memory_tables(SEQ1_THREAD, SEQ1_USER, label='SEQ1 FINAL — 5 turns complete')


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ### 6b — Distribution Head: Agent & Channel Strategy
# MAGIC
# MAGIC **Persona:** Head of Distribution reviewing agent productivity and cross-sell opportunity.  
# MAGIC **Route:** Distribution → Policy → Distribution → Customer → Synthesis
# MAGIC
# MAGIC | Turn | Query | Expected Genie Space |
# MAGIC |------|-------|----------------------|
# MAGIC | 1 | Top 5 agents by premium + churn rate | AIA Distribution Channels |
# MAGIC | 2 | Investment-Linked premium by channel | AIA Policy Underwriting |
# MAGIC | 3 | Hui Garcia vs Malaysia average | AIA Distribution Channels |
# MAGIC | 4 | Customer segment breakdown | AIA Customer Analytics |
# MAGIC | 5 | Malaysia cross-sell opportunity synthesis | Synthesis (memory) |
# MAGIC

# COMMAND ----------

SEQ2_THREAD = 'demo-dist-strategy-001'
SEQ2_USER   = 'demo-dist-user-001'
_memory_cache_ts = 0

for tbl, col in [('conversations','thread_id'), ('episodic_memory','thread_id'), ('user_memory','user_id')]:
    _val = SEQ2_THREAD if col == 'thread_id' else SEQ2_USER
    try:
        _run_sql(f"DELETE FROM {CATALOG}.ai_ops.{tbl} WHERE {col} = '{_val}'")
    except Exception:
        pass

print("Sequence 6b — Distribution Head Strategy")
print(f"  thread={SEQ2_THREAD}  user={SEQ2_USER}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 1 — Top Agents by Premium (Distribution Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s2t1 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'Who are the top 5 agents by total premium sold? Include their region and churn rate.'}],
    custom_inputs={'thread_id': SEQ2_THREAD, 'user_id': SEQ2_USER}
))
_print_turn("[Seq2 Turn1] Top 5 agents", resp_s2t1, checks={
    'Mentions agent(s)':       lambda l,_: 'agent' in l,
    'Mentions premium':        lambda l,_: 'premium' in l,
    'Mentions churn':          lambda l,_: 'churn' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 2 — Investment-Linked Premium by Channel (Policy Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s2t2 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'For Investment-Linked products, which distribution channel generates the highest total premium, and across which regions?'}],
    custom_inputs={'thread_id': SEQ2_THREAD, 'user_id': SEQ2_USER}
))
_print_turn("[Seq2 Turn2] Investment-Linked by channel", resp_s2t2, checks={
    'Mentions Investment-Linked': lambda l,_: 'investment' in l,
    'Mentions channel':           lambda l,_: 'channel' in l,
    'Contains numeric data':      lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 3 — Hui Garcia vs Malaysia Peer Average (Distribution Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s2t3 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'How does the top agent from Malaysia compare to the average agent premium sold in Malaysia?'}],
    custom_inputs={'thread_id': SEQ2_THREAD, 'user_id': SEQ2_USER}
))
_print_turn("[Seq2 Turn3] Malaysia agent comparison", resp_s2t3, checks={
    'Mentions Malaysia':       lambda l,_: 'malaysia' in l,
    'Mentions average':        lambda l,_: 'average' in l or 'avg' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 4 — Customer Segment Breakdown (Customer Analytics Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s2t4 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'What is the customer segment breakdown and average number of active policies per segment?'}],
    custom_inputs={'thread_id': SEQ2_THREAD, 'user_id': SEQ2_USER}
))
_print_turn("[Seq2 Turn4] Customer segments", resp_s2t4, checks={
    'Mentions segment':        lambda l,_: 'segment' in l,
    'Mentions policies':       lambda l,_: 'polic' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 5 — Malaysia Cross-Sell Synthesis (references Turns 1–4)

# COMMAND ----------

_memory_cache_ts = 0
resp_s2t5 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':(
        'From Turn 1 we saw the top agent in Malaysia sold ~$270K premium. '
        'Turn 2 showed Bancassurance is the dominant channel for Investment-Linked products. '
        'But Turn 4 shows Mass Affluent customers hold only 1.62 policies on average. '
        'Are we leaving revenue on the table in Malaysia — should we push Bancassurance '
        'to cross-sell more Investment-Linked to the Mass Affluent segment?'
    )}],
    custom_inputs={'thread_id': SEQ2_THREAD, 'user_id': SEQ2_USER}
))
_print_turn("[Seq2 Turn5] Malaysia cross-sell synthesis", resp_s2t5, checks={
    'References agent/premium': lambda l,_: 'premium' in l or 'agent' in l,
    'References channel':       lambda l,_: 'channel' in l or 'bancassurance' in l,
    'References segment':       lambda l,_: 'segment' in l or 'affluent' in l,
    'Gives recommendation':     lambda l,_: any(w in l for w in ['recommend','opportunit','cross-sell','potential','suggest']),
})

import time; time.sleep(2)
inspect_memory_tables(SEQ2_THREAD, SEQ2_USER, label='SEQ2 FINAL — 5 turns complete')


# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC ### 6c — Actuarial Risk: Motor Claims Investigation
# MAGIC
# MAGIC **Persona:** Chief Risk Officer investigating Motor insurance pricing adequacy.  
# MAGIC **Route:** Claims → Claims (fraud) → Policy → Customer → Synthesis
# MAGIC
# MAGIC | Turn | Query | Expected Genie Space |
# MAGIC |------|-------|----------------------|
# MAGIC | 1 | Motor claim distribution by region | AIA Claims Analytics |
# MAGIC | 2 | Fraud scores for Motor by region | AIA Claims Analytics |
# MAGIC | 3 | Motor premium by region and channel | AIA Policy Underwriting |
# MAGIC | 4 | Customer segment and policy count | AIA Customer Analytics |
# MAGIC | 5 | Identify most underpriced Motor market | Synthesis (memory) |
# MAGIC

# COMMAND ----------

SEQ3_THREAD = 'demo-motor-risk-001'
SEQ3_USER   = 'demo-motor-user-001'
_memory_cache_ts = 0

for tbl, col in [('conversations','thread_id'), ('episodic_memory','thread_id'), ('user_memory','user_id')]:
    _val = SEQ3_THREAD if col == 'thread_id' else SEQ3_USER
    try:
        _run_sql(f"DELETE FROM {CATALOG}.ai_ops.{tbl} WHERE {col} = '{_val}'")
    except Exception:
        pass

print("Sequence 6c — Actuarial Motor Risk")
print(f"  thread={SEQ3_THREAD}  user={SEQ3_USER}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 1 — Motor Claims by Region (Claims Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s3t1 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'How are Motor insurance claims distributed across regions? Show total claim count per region for Motor products.'}],
    custom_inputs={'thread_id': SEQ3_THREAD, 'user_id': SEQ3_USER}
))
_print_turn("[Seq3 Turn1] Motor claims by region", resp_s3t1, checks={
    'Mentions Motor':          lambda l,_: 'motor' in l,
    'Mentions region':         lambda l,_: 'region' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 2 — Motor Fraud Analysis (Claims Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s3t2 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'For Motor insurance specifically, which regions have the highest average fraud scores or suspicious claim rates?'}],
    custom_inputs={'thread_id': SEQ3_THREAD, 'user_id': SEQ3_USER}
))
_print_turn("[Seq3 Turn2] Motor fraud by region", resp_s3t2, checks={
    'Mentions fraud':          lambda l,_: 'fraud' in l,
    'Mentions Motor':          lambda l,_: 'motor' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 3 — Motor Premium by Region & Channel (Policy Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s3t3 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'What is the total Motor insurance premium collected by region and distribution channel?'}],
    custom_inputs={'thread_id': SEQ3_THREAD, 'user_id': SEQ3_USER}
))
_print_turn("[Seq3 Turn3] Motor premium by region/channel", resp_s3t3, checks={
    'Mentions Motor':          lambda l,_: 'motor' in l,
    'Mentions premium':        lambda l,_: 'premium' in l,
    'Mentions channel':        lambda l,_: 'channel' in l,
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 4 — Customer Segment & Policy Count (Customer Genie)

# COMMAND ----------

_memory_cache_ts = 0
resp_s3t4 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':'What is the customer segment breakdown? How many customers are in each segment and what is the average policy count?'}],
    custom_inputs={'thread_id': SEQ3_THREAD, 'user_id': SEQ3_USER}
))
_print_turn("[Seq3 Turn4] Customer segments", resp_s3t4, checks={
    'Mentions segment':        lambda l,_: 'segment' in l,
    'Contains numeric data':   lambda _,t: any(c.isdigit() for c in t),
})


# COMMAND ----------

# MAGIC %md
# MAGIC #### Turn 5 — Motor Underpricing Synthesis (references Turns 1–4)

# COMMAND ----------

_memory_cache_ts = 0
resp_s3t5 = agent.predict(ResponsesAgentRequest(
    input=[{'role':'user','content':(
        'Thailand and Malaysia showed high Motor claim volumes in Turn 1, '
        'with elevated fraud risk in Turn 2. But Motor premium via Agency is only '
        'around $12K for Thailand and $7.6K for Malaysia (Turn 3). '
        'Given the Mass segment holds the majority of customers with 1.39 avg policies (Turn 4), '
        'which region is most underpriced on Motor insurance, and what should we do about it?'
    )}],
    custom_inputs={'thread_id': SEQ3_THREAD, 'user_id': SEQ3_USER}
))
_print_turn("[Seq3 Turn5] Motor underpricing synthesis", resp_s3t5, checks={
    'References Motor claims':  lambda l,_: 'motor' in l and 'claim' in l,
    'References fraud':         lambda l,_: 'fraud' in l,
    'References premium':       lambda l,_: 'premium' in l,
    'Identifies underpricing':  lambda l,_: any(w in l for w in ['underpric','pric','risk','adjust']),
    'Names a region':           lambda l,_: any(r in l for r in ['thailand','malaysia','singapore','hong kong']),
})

import time; time.sleep(2)
inspect_memory_tables(SEQ3_THREAD, SEQ3_USER, label='SEQ3 FINAL — 5 turns complete')
