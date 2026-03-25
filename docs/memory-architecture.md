# AIA Supervisor Agent — Memory Architecture

The agent maintains three complementary memory layers that work together to make every conversation contextually aware, personalized, and progressively smarter over time.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MEMORY ARCHITECTURE                          │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  SHORT-TERM      │  │  LONG-TERM       │  │  EPISODIC        │  │
│  │  MEMORY          │  │  MEMORY          │  │  MEMORY          │  │
│  │                  │  │                  │  │                  │  │
│  │  Conversation    │  │  User Profile    │  │  Interaction     │  │
│  │  history within  │  │  & preferences   │  │  log + lessons   │  │
│  │  a session       │  │  across sessions │  │  learned         │  │
│  │                  │  │                  │  │                  │  │
│  │  Scope: thread   │  │  Scope: user     │  │  Scope: global   │  │
│  │  TTL: session    │  │  TTL: permanent  │  │  TTL: permanent  │  │
│  │                  │  │                  │  │                  │  │
│  │  ai_ops.         │  │  ai_ops.         │  │  ai_ops.         │  │
│  │  conversations   │  │  user_memory     │  │  episodic_memory │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Short-term Memory

**What it stores:** The full message history of a conversation thread (rolling window of the last 7 turns).

**Where it lives:**
- **In-memory dict** (`_conversation_history`) — primary, zero-latency lookup within the same kernel session.
- **Delta table** `ai_ops.conversations` — durable fallback; survives kernel restarts and cross-process calls.

**Key behaviours:**
- Every response is appended to the in-memory history under `thread_id`.
- At the start of each turn the agent checks in-memory first; if empty (e.g. after a restart) it falls back to the latest Delta checkpoint.
- The context window is capped at `MAX_MESSAGES = 7` to keep prompt sizes predictable.
- Follow-up questions shorter than 10 words are automatically rewritten into self-contained queries using conversation history (see `classify_intent` → follow-up resolution).

### Conversation walkthrough

> **Thread:** `claims-session-001` | **User:** `analyst-raj`

---

**Turn 1** — first question, no prior history

```
User:  "What is the total number of claims filed in the Northern region
        for Q3 2025?"

[Memory read]
  short-term  → history = []           ← empty, first turn
  long-term   → user_memory = {}       ← new user, nothing stored yet
  episodic    → lessons = []           ← no prior episodes for this intent/domain

[Graph executes]
  classify_intent → simple_kpi (0.95)
  resolve_assets  → genie_space: "Claims Analytics Space"
  genie           → SQL query runs, returns 4,712 claims

[Memory write]
  short-term  → conversations: checkpoint saved
                _conversation_history["claims-session-001"] = [
                  {role: "user",      content: "What is the total..."},
                  {role: "assistant", content: "The Northern region recorded..."}
                ]
  episodic    → episode logged: intent=simple_kpi, domain=claims,
                agents_used=["genie"], outcome=success

Agent: "The Northern region recorded 4,712 claims in Q3 2025,
        representing a 12% increase over Q2."
```

---

**Turn 2** — short follow-up; short-term memory resolves context

```
User:  "What about the Southern?"

[Memory read]
  short-term  → history = [turn 1 exchange]   ← loaded from _conversation_history
  long-term   → user_memory = {}
  episodic    → no lessons yet

[classify_intent — follow-up resolution]
  Detects: len("What about the Southern?".split()) = 4  ← under 10 words
  Rewrites using last 6 messages from history:
    → "What is the total number of claims filed in the Southern region
       for Q3 2025?"

[Graph executes with rewritten query]
  genie → SQL query for Southern region

[Memory write]
  short-term  → checkpoint updated; history now has 4 messages
  episodic    → second episode logged

Agent: "The Southern region had 3,891 claims in Q3 2025 — about
        17% fewer than the Northern region over the same period."
```

---

**Turn 3** — context window in action (MAX_MESSAGES = 7)

After 4+ turns the context window trims to the last 7 messages, ensuring the
prompt never grows unbounded while retaining the most relevant recent context.

```
[Memory read at turn 5]
  _conversation_history["claims-session-001"] has 10 messages
  → all_messages = history[-7:]   ← only the last 7 are passed to the graph
```

---

## 2. Long-term Memory

**What it stores:** User-specific facts and preferences extracted from conversations — name, role, region focus, preferred response style, preferred domains, etc.

**Where it lives:** Delta table `ai_ops.user_memory`, keyed by `user_id`. Entries are loaded with a 60-second in-process cache.

### Why is long-term memory needed if short-term memory already exists?

This is the critical question. Short-term memory is keyed by **`thread_id`**. Long-term memory is keyed by **`user_id`**. They solve completely different problems.

```python
# Short-term — dies when the thread changes
history = self._conversation_history.get(thread_id, [])   # ← thread_id

# Long-term — survives forever across all threads for the same user
user_mem = _load_user_memory(user_id)                      # ← user_id
```

There are three gaps that short-term memory **cannot** fill:

```
GAP 1 — New thread, returning user
─────────────────────────────────────────────────────────────
  Sarah talked to the agent yesterday on thread-A.
  Today she opens a new chat → thread-B.

  short-term read:  history = []    ← thread-B has no history
  long-term read:   { name: "Sarah", role: "...", ... }
                                    ← persisted from thread-A

  Without long-term: agent treats Sarah as a brand new user
  every single conversation.

GAP 2 — MAX_MESSAGES = 7 trims the introduction
─────────────────────────────────────────────────────────────
  Sarah introduced herself on turn 1 of a long thread.
  By turn 8, that introduction has been trimmed from the
  rolling context window.

  short-term at turn 8:  last 7 messages only — turn 1 is gone
  long-term at turn 8:   { name: "Sarah", ... }  ← still there

  Without long-term: agent forgets who the user is mid-thread.

GAP 3 — Raw messages vs structured facts
─────────────────────────────────────────────────────────────
  Short-term stores raw conversation text that must be re-parsed
  on every turn to extract meaning.

  Long-term stores structured, directly usable key-value pairs:
    { response_length: "concise", preferred_domain: "CI" }

  These are injected directly into classify_intent and
  compose_answer prompts — no re-parsing needed, no risk
  of the LLM missing a subtle preference buried in old text.
```

**Key behaviours:**
- When the agent detects a `conversational` intent (greetings, introductions, small talk), the LLM scans the exchange and persists any explicitly stated facts via `_extract_and_save_user_facts`.
- At every subsequent turn the agent loads these facts and passes them to both `classify_intent` (for smarter routing) and `compose_answer` (for personalisation — addressing the user by name, matching response style, etc.).
- Facts are stored with a `confidence` score and an optional `expires_at`.
- Updates use `MERGE` semantics — re-stating a preference overwrites the old value.

### Conversation walkthrough

> **Thread A:** `onboarding-thread-001` | **User:** `demo-user-sarah`

---

**Turn 1 (Thread A)** — user introduces herself; long-term memory is written

```
User:  "Hi, I'm Sarah, a regional claims manager based in Singapore.
        I prefer concise responses and usually focus on the Health
        and Critical Illness product lines."

[Memory read]
  short-term  → []   (thread_id = "onboarding-thread-001", first turn)
  long-term   → {}   ← new user, nothing stored yet
  episodic    → []

[Graph executes]
  classify_intent → conversational (0.97)
  compose_answer  → friendly greeting, no data lookup

[Memory write — long-term]
  _extract_and_save_user_facts triggered (intent = conversational)
  LLM extracts facts from the exchange:
    name             = "Sarah"                   (fact,       conf: 1.0)
    role             = "regional claims manager" (fact,       conf: 1.0)
    preferred_region = "Singapore"               (preference, conf: 1.0)
    preferred_domain = "Health, Critical Illness"(preference, conf: 1.0)
    response_length  = "concise"                 (preference, conf: 1.0)

  Saved to ai_ops.user_memory → user_id = "demo-user-sarah"

Agent: "Hello Sarah! Great to meet you. I'm here to help with your
        Health and Critical Illness analytics. What would you like
        to explore today?"
```

---

**Turn 1 (Thread B, next day)** — new thread, short-term is empty; long-term bridges the gap

```
User:  "What are the exclusions for the Critical Illness plans?"

  ┌──────────────────────────────────────────────────────┐
  │  thread_id = "analysis-thread-002"  (NEW thread)     │
  │  user_id   = "demo-user-sarah"      (SAME user)      │
  └──────────────────────────────────────────────────────┘

[Memory read]
  short-term  → []   ← new thread; _conversation_history has no
                        entry for "analysis-thread-002"
                        Delta checkpoint also returns nothing
                        ∴ all_messages = [current message only]

  long-term   → {    ← loaded from ai_ops.user_memory;
                  name: "Sarah",            written during Thread A,
                  role: "regional claims manager",  days ago
                  preferred_region: "Singapore",
                  preferred_domain: "Health, Critical Illness",
                  response_length: "concise"
                }

[classify_intent]
  Prompt appends: "User preferences: name=Sarah;
                  preferred_domain=Health, Critical Illness;
                  response_length=concise"
  → intent = document_lookup (0.93)
  → no clarification needed (preferences fill in the context)

[compose_answer]
  Builds response knowing: name=Sarah, style=concise, domain=CI

Agent: "Sarah, the Critical Illness plans share these core exclusions:
        pre-existing conditions diagnosed within 12 months of
        inception, self-inflicted injuries, and war-related illnesses.
        The Elite plan additionally excludes cosmetic procedures."
         ↑ name used   ↑ concise   ↑ CI-focused — all from long-term
                                     memory, not from thread history
```

Without long-term memory this response would be:
`"The Critical Illness plans have the following exclusions: [generic list]."` — no name, potentially verbose, no domain focus.

---

**Turn 8 (Thread B)** — showing MAX_MESSAGES trim; long-term survives where short-term doesn't

```
Turn 1 of Thread B: Sarah's introduction (via prior thread data) is NOT in
                    this thread's short-term history at all.
Turn 8 context window: last 7 messages = turns 2-8 only.

  short-term at turn 8:  [msg-2, msg-3, msg-4, msg-5, msg-6, msg-7, msg-8]
                          ← user's name/prefs are not in any of these

  long-term  at turn 8:  { name: "Sarah", response_length: "concise", ... }
                          ← always available, never trimmed

Agent at turn 8 still addresses Sarah by name and matches her style.
```

---

**Long-term memory state** — persists forever, grows richer over time

| `memory_key`       | `memory_value`              | `memory_type` | `confidence` |
|--------------------|-----------------------------|---------------|:------------:|
| `name`             | Sarah                       | fact          | 1.0          |
| `role`             | regional claims manager     | fact          | 1.0          |
| `preferred_region` | Singapore                   | preference    | 1.0          |
| `preferred_domain` | Health, Critical Illness    | preference    | 1.0          |
| `response_length`  | concise                     | preference    | 1.0          |

---

## 3. Episodic Memory

**What it stores:** A log of every interaction — the question asked, intent, domain, which agents ran, whether the outcome was a success or failure, and (when available) a `lesson_learned` to improve future answers.

**Where it lives:** Delta table `ai_ops.episodic_memory`.

**Key behaviours:**
- Every `predict()` call writes one episode regardless of outcome.
- Before generating the final answer, `compose_answer` calls `_get_episodic_lessons(intent, domain)` to fetch the 3 most recent lessons for the same intent + domain combination.
- Lessons are injected into the compose prompt as **"Past experience (internal)"**, allowing the agent to avoid past mistakes or replicate past successes.
- Failed queries (Genie returning an error, document retrieval failures) can eventually generate lessons if a human or automated review annotates the `lesson_learned` field.

### Conversation walkthrough

> **Scenario:** The agent learns from a failed Genie query and avoids the same mistake.

---

**Episode 1** — initial query, Genie fails

```
User (thread: kpi-thread-A):
  "What is the average settlement time for Critical Illness claims
   in the Singapore region, segmented by agent tier?"

[Graph executes]
  classify_intent → simple_kpi (0.88)
  resolve_assets  → genie_space: "Claims Analytics Space"
  genie           → FAILED (column "agent_tier" not found in schema)

[Memory write — episodic]
  episode saved:
    episode_id  = "a3f9e1b2c4d5"
    intent      = "simple_kpi"
    domain      = "claims"
    agents_used = ["genie"]
    outcome     = "failed"
    lesson_learned = NULL   ← not yet annotated

Agent: "I wasn't able to retrieve the settlement time breakdown by
        agent tier — the data schema may not include that dimension.
        I can show overall average settlement times instead."
```

---

**Episode 2 (same day, different user)** — lesson prevents repeat failure

> Assume an admin or automated pipeline annotated episode 1 with:
> `lesson_learned = "Claims data does not have an 'agent_tier' column. Use 'channel_type' instead."`

```
User (thread: kpi-thread-B):
  "Show me settlement time for CI claims in Singapore by agent tier."

[compose_answer — episodic lessons lookup]
  _get_episodic_lessons(intent="simple_kpi", domain="claims")
  → returns 1 lesson:
      outcome: "failed"
      lesson:  "Claims data does not have an 'agent_tier' column.
                Use 'channel_type' instead."

[compose prompt now includes]
  **Past experience (internal):**
  - [failed] Claims data does not have an 'agent_tier' column.
             Use 'channel_type' instead.

[Genie query — automatically corrects column name]
  SQL uses "channel_type" instead of "agent_tier" → succeeds

[Memory write — episodic]
  episode saved: outcome = "success"

Agent: "The average CI claim settlement time in Singapore, broken down
        by channel type (the equivalent of agent tier in our data):
        Tied Agents — 8.2 days, Bancassurance — 11.4 days,
        Direct — 6.1 days."
```

---

**Episodic memory table** after both episodes

| `episode_id` | `intent`   | `domain` | `agents_used` | `outcome` | `lesson_learned`                                             |
|--------------|------------|----------|---------------|-----------|--------------------------------------------------------------|
| `a3f9e1b2`   | simple_kpi | claims   | [genie]       | failed    | Claims data has no 'agent_tier'; use 'channel_type' instead  |
| `b7c2d4e8`   | simple_kpi | claims   | [genie]       | success   | NULL                                                         |

---

## How All Three Memories Work Together

The following example shows a single turn where all three memory layers contribute to the final response.

> **User (Sarah, returning after 3 days, new thread):**
> `"How do I file a claim? Also, what's the waiting period?"`

```
┌─────────────────────────────────────────────────────────────────────┐
│ TURN EXECUTION TRACE                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  REQUEST ARRIVES                                                    │
│    thread_id = "new-thread-xyz"                                     │
│    user_id   = "demo-user-sarah"                                    │
│                                                                     │
│  ① SHORT-TERM MEMORY READ                                          │
│    in-memory history = []   (new thread)                           │
│    Delta checkpoint  = none (new thread)                           │
│    → all_messages = [current user message only]                    │
│                                                                     │
│  ② LONG-TERM MEMORY READ                          ← personalize    │
│    _load_user_memory("demo-user-sarah")                             │
│    → { name: "Sarah", role: "regional claims manager",             │
│        preferred_domain: "Health, Critical Illness",               │
│        response_length: "concise" }                                 │
│    classify_intent sees user preferences → faster, accurate intent │
│                                                                     │
│  GRAPH NODES                                                        │
│    classify_intent  → document_lookup (0.91)                       │
│    resolve_assets   → doc_vs_index: policy_docs_vs                  │
│    multi_tool (RAG) → 3 policy docs retrieved                      │
│                                                                     │
│  ③ EPISODIC MEMORY READ                           ← learn          │
│    _get_episodic_lessons("document_lookup", "claims")               │
│    → 1 lesson: "For CI claims, always mention the 90-day waiting   │
│                 period as users frequently miss this detail."       │
│                                                                     │
│  compose_answer BUILDS PROMPT WITH:                                 │
│    • RAG results from 3 policy documents                            │
│    • User preference: name=Sarah, response style=concise           │
│    • Episodic lesson: highlight the 90-day waiting period          │
│                                                                     │
│  FINAL RESPONSE                                                     │
│    "Sarah, to file a CI claim: (1) notify us within 30 days of     │
│     diagnosis, (2) complete the claim form at any AIA branch or    │
│     via the app, (3) attach the specialist's report and hospital    │
│     discharge summary. Important: the Critical Illness plans have  │
│     a 90-day waiting period from inception — claims arising         │
│     before this window are not covered."                            │
│                              ↑ name used    ↑ episodic lesson      │
│                                                                     │
│  MEMORY WRITES                                                      │
│    short-term → history updated for "new-thread-xyz"               │
│    episodic   → episode logged: document_lookup / success          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Reference

| Memory layer | Read function | Write function | Delta table | Scope |
|---|---|---|---|---|
| Short-term | `_conversation_history.get(thread_id)` | `_conversation_history[thread_id] = …` | `ai_ops.conversations` (fallback) | Per thread |
| Long-term | `_load_user_memory(user_id)` | `_save_user_memory(user_id, key, value)` | `ai_ops.user_memory` | Per user |
| Episodic | `_get_episodic_lessons(intent, domain)` | `_save_episodic_memory(…)` | `ai_ops.episodic_memory` | Global (intent + domain) |

### When each layer is read and written

```
predict() called
│
├── mlflow.update_current_trace(session=thread_id)
│
├── READ short-term  (_conversation_history / Delta checkpoint)
│     └── build all_messages = history + new_msgs
│         trim to MAX_MESSAGES = 7
│
└── graph.invoke(initial_state)
      │
      ├── classify_intent
      │     └── READ long-term  (_load_user_memory)
      │           append preferences to classify prompt
      │
      ├── resolve_assets
      │
      ├── genie / multi_tool
      │
      └── compose_answer
            ├── READ episodic  (_get_episodic_lessons)
            │     inject lessons into compose prompt
            └── READ long-term  (_load_user_memory)
                  inject name / style / domain prefs

predict() returns
│
├── WRITE short-term  (_conversation_history[thread_id] updated)
├── WRITE short-term  (Delta checkpoint saved)
└── WRITE episodic   (_save_episodic_memory)
      (long-term written only when intent = conversational,
       via _extract_and_save_user_facts)
```
