# Memory System — Architecture Redesign

**Status:** Proposed · **Author:** Architecture review · **Date:** 2026-08-07
**Supersedes:** the memory sections of `AUDIT_REPORT.md`
**Scope:** `app/memory/`, memory-adjacent parts of `app/agents/`, `app/routes/agent_routes.py`, `app/services/{qdrant,cohere,chunking}_service.py`

---

## 0. Executive summary

The current memory system is organised around **storage engines**. Modules are named
after databases (`short_term_memory.py` = Postgres, `long_term_memory_qdrant.py` =
Qdrant, `smart_memory.py` = mem0). Every architectural problem in this document
descends from that single choice.

The redesign is organised around the **domain**: one typed `MemoryRecord` entity, a
pluggable storage substrate behind ports, an asynchronous extraction pipeline, and a
hybrid retrieval engine with an explicit token budget.

Five changes carry most of the value:

| # | Change | Fixes |
|---|---|---|
| **D1** | One unified `MemoryRecord` with a `kind` discriminator, replacing three storage-shaped modules | God-class, duplicated retrieval code, no shared ranking |
| **D2** | Separate *memory* from *application data* (attendance, timetable, drafts…) | 1,145-line `ShortTermMemory`, 30 proxy methods on `MemoryManager` |
| **D3** | Retrieval = hybrid candidates → fusion → rerank → **budget allocation** | Blind concatenate-then-truncate at 20k chars |
| **D4** | Writes are **extracted, deduped, scored, versioned** — asynchronously | Raw transcript logging that degrades as it grows |
| **D5** | `visibility` as a first-class field, not ownership alone | Recruiter demo retrieves an empty partition |

---

## 1. Audit: what is actually wrong

The previous pass documented behaviour. This section states the *architectural*
faults, ordered by consequence.

### 1.1 Module boundaries follow the database, not the domain

`short_term_memory.py` is 1,145 lines and owns twelve unrelated entity types:
chat history, attendance, timetable, job bookmarks, email drafts, email templates,
exams, plans, user profile, episodic memory, agent playbooks, tool memory.

It is named "short-term" but nothing in it ever expires. It is the system of record
for permanent data. The name is actively misleading to anyone reading the codebase.

`MemoryManager` then proxies ~30 methods straight through to it
(`store_attendance`, `save_email_draft`, `retrieve_exams`, `mark_plan_done`…),
which makes the memory facade a dumping ground for anything that touches Postgres.

**Consequence:** there is no place to put memory logic. Ranking, dedup, importance
and lifecycle have no home, so they were never built.

### 1.2 Attendance records are not memories

Attendance, timetable, exams, plans, job bookmarks, email drafts and templates are
**application records**. They have their own schemas, their own CRUD, their own
routes, and no semantic-retrieval role. Filing them under `app/memory/` conflates
"data the app stores" with "things the assistant remembers about you".

### 1.3 Write path is a raw transcript log, not memory

`on_user_input()` and `on_agent_response()` embed **every utterance verbatim** into
`smart_memory_chunks`. There is no extraction, no importance, no dedup, no decay.

This is a noise amplifier. Semantic search over an unfiltered transcript gets
*worse* as it grows: after a year, "what are my goals?" competes against thousands
of embedded fragments of small talk. The system cannot become a digital
representation of the user by construction — it only accumulates.

### 1.4 mem0 is dead code

`SmartMemory.__init__` builds a `mem0.Memory` with a Groq LLM and a HuggingFace
MiniLM embedder. **No code path ever calls `self.memory.*`** (verified: zero grep
hits). The class bypasses mem0 entirely and talks to Qdrant + Cohere directly —
with a *different* embedder (Cohere 1024-dim) than the one it configured
(MiniLM 384-dim).

`mem0ai==0.1.37` is a dependency, an import cost, and a source of reader confusion,
for zero runtime behaviour.

### 1.5 Dual-write with no transaction — inconsistency is designed in

`on_agent_response()` writes Postgres, then writes Qdrant. There is no shared
transaction. The code knows this and logs it:

> `"Vector memory write failed … chat history was saved but this turn is absent from semantic memory."`

Documenting a data-integrity hole is not the same as closing it. Two systems of
record for one fact will drift.

### 1.6 Retrieval is concatenation, not ranking

`retrieve_context()` fetches five sources and `format_context_for_prompt()`
concatenates them in a hard-coded priority order. `inject_memory_context()` then
truncates at 20,000 characters.

There is **no relevance ranking across sources**. A stale resume line and a
critical goal compete only by their position in a fixed list. The priority ordering
is a thoughtful mitigation, but it mitigates the wrong thing: the fix for context
overflow is *allocation*, not *truncation*.

### 1.7 Conversations do not survive a refresh

`ChatShell.tsx` mints `session_id = crypto.randomUUID()` on every mount and never
persists it. `get_recent_context()` filters chat history by exactly that
`session_id`. A page refresh therefore retrieves **zero** prior turns while the
entire conversation sits in Postgres under the previous id.

The frontend also never populates `conversation_history` on the request (verified:
zero references in `frontend/components`, `frontend/lib`), so the API's own
multi-turn parameter is dead on the text path.

Three uncoordinated representations of "recent turns" exist:
Postgres `chat_history` (by `session_id`), the request-body `conversation_history`
array (never sent), and `ParticipantState.history` in the voice worker
(in-process, capped at 24, lost on worker exit).

### 1.8 Ownership is the only access model — the recruiter demo is structurally broken

Memory is partitioned solely by `user_id`. `new_guest_user_id()` correctly isolates
guests as `guest-<uuid>`. `settings.owner_user_id` is referenced **only at login** —
there is no fallback anywhere in the agent or memory path. `output_mode="recruiter"`
changes only the response tone in `response_agent.py`.

Therefore a recruiter session retrieves `user_id = "guest-abc123"`: empty profile
facts, empty episodes, and Qdrant searches that match nothing. **The public-facing
demo cannot discuss the resume, skills, or projects it exists to present.**

This is not a bug to patch. It says the access model needs a *visibility* dimension
independent of ownership.

### 1.9 Guest partitions accumulate forever

Every recruiter visit creates a permanent partition that writes `chat_history`,
`episodic_memory` and `smart_memory_chunks` rows, and is never revisited or cleaned
up. Over time, orphaned guest data becomes the dominant share of the store.

### 1.10 Smaller faults

| Fault | Detail |
|---|---|
| `agent_playbooks` is write-only | Full CRUD + routes exist; **no agent ever calls `get_active_playbook()`**. Versioned prompts that cannot take effect. |
| Cache key ignores `session_id` and is never invalidated on write | `MemoryCache` keys on `user_id`(+query hash), 5-min TTL, no invalidation. A write inside the window serves stale preferences/episodes. |
| Sentinel strings as return values | `retrieve_skills()` returns `List[…]` **or** the string `"NO_DATA"`. Callers must `isinstance`-guard, and `format_context_for_prompt` does exactly that. Type-unsafe by design. |
| Resume parsing is heading-keyword matching | `_extract_semantic_resume_chunks` depends on a hard-coded `heading_map`. Unrecognised layouts silently fall through to token windows with no confidence signal. |
| **`_detect_name` mistakes a section heading for the user's name** | No heading guard: a resume whose name line is absent — common, since PDF header text often extracts as an image — stores `"SKILLS"` or `"EXPERIENCE"` as an `identity` chunk at `high` importance, which is then injected into every prompt. Pinned as a strict `xfail` in `tests/test_memory_characterisation.py`; Phase 1 identity extraction must fix it. |
| Voice/text write asymmetry | The streaming path skips the reasoning loop, so `ToolMemory` is **read** on that path but never **written** there. |
| No pruning anywhere | `chat_history` and `episodic_memory` grow unbounded. Per-call cost is capped by `last_n`; storage is not capped at all. |
| Process-local cache | Single-instance only; blocks horizontal scaling alongside the already-flagged `active_workers` registry. |

---

## 2. Keep / redesign / remove

### Keep (genuinely good work)

- **Auth and identity resolution.** `resolve_user_id`, scope filtering in
  `BaseAgent._filter_tools_by_scope`, HttpOnly cookies + CSRF. This is solid and
  the redesign builds on it unchanged.
- **Sensitive-value filtering.** `_is_sensitive()` — carry forward into the
  ingestion pipeline as a governance stage.
- **Upsert-before-delete replacement** in `store_resume` — the correct pattern;
  generalise it into document versioning.
- **Exhaustive scroll semantics** (`scroll_collection` follows cursors to
  exhaustion) — the correctness instinct here is right.
- **Parallel retrieval + `parallel_init_node`** — keep the concurrency; replace
  what runs inside it.
- **Confidence gating on inferred facts** (`>= 0.8`) — promote to a first-class
  field on every record.

### Redesign

| Component | Becomes |
|---|---|
| `memory_manager.py` (facade + 30 proxies) | `MemoryService` — a narrow interface: `remember()`, `recall()`, `assemble_context()`, `forget()` |
| `short_term_memory.py` | Split: `memory/` repositories + `domain/` repositories (attendance, timetable, …) |
| `long_term_memory_qdrant.py` | `DocumentIngestor` + a `VectorStore` port |
| `memory_cache.py` | `RetrievalCache` — correct key, explicit invalidation, pluggable backend |
| `retrieve_context` + `format_context_for_prompt` | `RetrievalEngine` + `ContextAssembler` (rank → budget → render) |
| `ChatHistory` + `session_id` | `Conversation` + `Turn`, persistent and resumable |

### Remove entirely

| Removal | Justification |
|---|---|
| `mem0ai` dependency and `smart_memory.py` | Zero call sites. Replaced by the extraction pipeline. |
| `agent_playbooks` table + routes + 3 manager methods | Write-only feature, no reader, ~120 lines. Prompts live in code/config until there is a real need. |
| `"NO_DATA"` / `"FALLBACK"` sentinel strings | Replaced by typed `RetrievalResult` with an explicit `status` enum. |
| `chroma_persist_dir`, `mem0_api_key` settings | Dead config for a removed backend. |
| `MemoryManager` domain proxies (~30 methods) | Callers use domain repositories directly. |

---

## 3. Target architecture

### 3.1 Layering

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L4  CONTROL PLANE                                                        │
│     Memory API · editing · provenance ("why do you know this?")          │
│     export · user-initiated deletion · consent & visibility management   │
├──────────────────────────────────────────────────────────────────────────┤
│ L3  COGNITION            (asynchronous, off the request path)            │
│     Extractor · Deduplicator · ImportanceScorer · Consolidator           │
│     ConflictResolver · DecayEngine · Archiver                            │
├──────────────────────────────────────────────────────────────────────────┤
│ L2  ACCESS               (synchronous, in the request path)              │
│     RetrievalEngine (hybrid → fusion → rerank)                           │
│     ContextAssembler (tier allocation → budget → render)                 │
│     WorkingMemory (conversation window + running summary)                │
├──────────────────────────────────────────────────────────────────────────┤
│ L1  DOMAIN MODEL                                                         │
│     MemoryRecord(kind) · Conversation/Turn · Document/Chunk              │
│     Entity · Relation · Source                                           │
├──────────────────────────────────────────────────────────────────────────┤
│ L0  SUBSTRATE            (ports + adapters)                             │
│     RecordStore · VectorStore · LexicalIndex · BlobStore                 │
│     Cache · EventQueue · Clock                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

Dependencies point **downward only**. L3 and L2 never import a database client;
they depend on L0 ports. This is what makes the vector-store decision reversible
and the system testable without a live Postgres or Qdrant.

### 3.2 The memory taxonomy

Every remembered thing is a `MemoryRecord` with a `kind`. This is one table with
one retrieval path, not nine subsystems.

| Kind | Holds | Half-life | Retrieval |
|---|---|---|---|
| `identity` | Name, role, education, location, GitHub/LinkedIn/LeetCode handles | ∞ | **Always injected** |
| `preference` | How the assistant should behave: tone, format, language, verbosity | ∞ | **Always injected** |
| `goal` | Intentions with state + target date: "SDE internship by June 2027" | until closed | **Always injected while active** |
| `task` | Actionable items derived from goals or conversation | 7 d | Active only |
| `semantic` | Durable facts: skills, habits, opinions, relationships | 365 d | Hybrid search |
| `episodic` | Timestamped events and conversation summaries | 90 d | Recency + search |
| `procedural` | Tool strategies that worked (today's `ToolMemory`) | 180 d | Agent/tool scoped |
| `document` | Chunks of ingested artifacts, with provenance to a `Document` | ∞ (versioned) | Hybrid search |
| `relation` | Typed edges: `(Vansh)-[built]->(My_Agent)-[uses]->(Qdrant)` | derived | Graph expansion |

**Working memory is not a stored kind.** It is assembled at request time from the
conversation window plus a running summary. Storing it would duplicate `Turn`.

The user's requested categories map cleanly:
who I am → `identity`; resume/projects/GitHub/LeetCode/LinkedIn → `document` +
`relation`; skills/education → `semantic` + `identity`; goals → `goal`;
habits/preferences → `semantic` + `preference`; conversations → `episodic` + `Turn`;
important events → `episodic` (high importance, pinned); uploaded documents →
`document`; future notes → `semantic`/`task`; future integrations → new `Source`
adapters feeding the same pipeline.

### 3.3 The core record

```python
@dataclass
class MemoryRecord:
    # ── identity ────────────────────────────────────────────────────────
    id: UUID
    owner_id: str                     # partition key; multi-user from day one
    kind: MemoryKind

    # ── content: dual representation ────────────────────────────────────
    content: str                      # self-contained NL statement — embedded AND injected
    structured: dict                  # machine-readable payload, kind-specific
    embedding: Vector                 # committed in the same transaction as the row

    # ── salience ────────────────────────────────────────────────────────
    importance: float                 # 0–1, assigned at extraction, decays
    confidence: float                 # 0–1, how sure we are this is true
    pinned: bool                      # user override: never decay, never drop

    # ── temporal ────────────────────────────────────────────────────────
    occurred_at: datetime | None      # when the fact was true / event happened
    valid_from: datetime
    valid_to: datetime | None         # None = still true (bitemporal)
    created_at: datetime
    last_accessed_at: datetime
    access_count: int

    # ── lineage ─────────────────────────────────────────────────────────
    source_type: SourceType           # chat | upload | github | gmail | calendar | …
    source_ref: str                   # conversation/turn id, document id, external id
    derived_from: list[UUID]          # provenance for consolidated memories
    supersedes: UUID | None           # versioning chain
    version: int

    # ── governance ──────────────────────────────────────────────────────
    visibility: Visibility            # private | shared | public
    sensitivity: Sensitivity          # normal | sensitive | secret
    status: Status                    # active | superseded | archived | deleted

    # ── dedup ───────────────────────────────────────────────────────────
    content_hash: str                 # exact-duplicate short circuit
    dedup_key: str | None             # normalised semantic key for conflict detection
```

Two properties earn their complexity:

**Dual representation (`content` + `structured`).** `content` is always a complete
natural-language sentence, because that is simultaneously what gets embedded and
what gets injected into the prompt. `structured` carries the machine-readable
version for filters, UI and integrations. One table then serves both semantic
search and structured query without a second schema.

**Bitemporal validity (`valid_from` / `valid_to`).** "Vansh is a student" was true
until it wasn't. Superseding a fact sets `valid_to` and links `supersedes` rather
than deleting — which is what makes versioning, "what did I believe last March",
and honest conflict handling possible.

### 3.4 Visibility — and the recruiter fix

`visibility` is orthogonal to `owner_id`:

- `private` — default. Only the owner retrieves it.
- `shared` — reachable by explicitly granted principals (future multi-user).
- `public` — reachable by any principal holding `PROFILE_READ`.

The owner marks resume, projects, skills and public GitHub work as `public`. A
recruiter's `RetrievalScope` becomes `owner_id = <owner>, visibility = public`
instead of `owner_id = <their own guest id>`. The demo works, the private data
stays private, and the same mechanism serves real multi-user sharing later.

This replaces `output_mode="recruiter"` as a cosmetic tone switch with a real
access-control boundary.

### 3.5 Ingestion — asynchronous, extracted, deduped

```
turn completes
     │
     ▼  (same transaction as the Turn write — outbox pattern)
┌──────────────────┐
│ memory_events    │   at-least-once, no broker required
└────────┬─────────┘
         │  worker polls  FOR UPDATE SKIP LOCKED
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. EXTRACT      LLM → candidate memories, typed             │
│                 {kind, content, structured, importance,      │
│                  confidence, occurred_at}                    │
│ 2. GOVERN       sensitive-value filter, consent, visibility  │
│ 3. DEDUP        exact (content_hash) → semantic (cosine>0.92)│
│                 → merge provenance instead of inserting      │
│ 4. RECONCILE    contradicts an active record?                │
│                 → close old (valid_to), link supersedes      │
│ 5. EMBED+WRITE  embedding and row commit together            │
└─────────────────────────────────────────────────────────────┘
```

Non-negotiable: **turn latency must never include extraction.** Voice makes this
existential — a 600 ms extraction call inside a spoken turn is audible dead air.

The outbox pattern (`memory_events` table written in the same transaction as the
turn) gives durable at-least-once delivery with no message broker. It swaps for
Redis/Celery later behind the `EventQueue` port without touching L3.

### 3.6 Retrieval — hybrid, fused, ranked, budgeted

```
query + conversation context
     │
     ▼
┌─ ANALYSE ────────────────────────────────────────────────────┐
│  intent · which kinds matter · temporal scope · entities      │
└────────┬──────────────────────────────────────────────────────┘
         │
    ┌────┴──────────────┬──────────────────┬──────────────────┐
    ▼                   ▼                  ▼                  ▼
 VECTOR              LEXICAL           STRUCTURED         GRAPH
 (cosine,            (BM25/trigram —   (kind, recency,    (relation
  top-k per kind)     exact names,      pinned, active     expansion
                      IDs, acronyms)    goals)             1 hop)
    └────────┬──────────┴──────────────────┴──────────────────┘
             ▼
    ┌─ FUSE ─────────────────────────────────────────┐
    │  Reciprocal Rank Fusion  (1 / (k + rank))      │
    └────────┬────────────────────────────────────────┘
             ▼
    ┌─ RERANK ───────────────────────────────────────┐
    │  score = w_sim·similarity                       │
    │        + w_imp·importance                       │
    │        + w_rec·decay(age, half_life[kind])      │
    │        + w_use·log1p(access_count)              │
    │        + w_kind·prior(kind | intent)            │
    │        − w_unc·(1 − confidence)                 │
    └────────┬────────────────────────────────────────┘
             ▼
    ┌─ BUDGET ───────────────────────────────────────┐
    │  allocate BEFORE rendering, not truncate after  │
    └────────┬────────────────────────────────────────┘
             ▼
        context block  +  RetrievalTrace
```

**Why lexical alongside vector.** Embeddings are poor at exact tokens — "LeetCode
1462", "T-2 hostel", a specific repo name. Postgres trigram/`tsvector` costs almost
nothing and rescues precisely the queries where a personal assistant most needs to
be exact. Vector-only retrieval is the single most common RAG mistake.

**Why RRF.** It fuses rankings without needing calibrated scores across channels,
which cosine similarity and BM25 do not share.

### 3.7 Token budget

A fixed memory budget (default 6,000 tokens) allocated in tiers:

| Tier | Contents | Budget | Droppable |
|---|---|---|---|
| 0 | identity, preferences, active goals | ~800 | **Never** |
| 1 | conversation window + running summary | ~2,000 | Compressed, not dropped |
| 2 | ranked semantic / episodic / document | ~2,500 | Greedy fill by score |
| 3 | relations, procedural hints | remainder | First to go |

Allocation happens *before* rendering. Anything dropped for budget is recorded in
the `RetrievalTrace`. Compare with today: render everything, then cut at character
20,000 and hope the ordering saved you.

### 3.8 Working memory & conversation persistence

`Conversation` becomes a first-class entity:

```
Conversation(id, owner_id, title, started_at, last_active_at,
             running_summary, summary_through_turn, status)
Turn(id, conversation_id, role, content, modality,  # text | voice
     agent, intent, tokens, created_at)
```

- The browser persists `conversation_id` in `localStorage`; a refresh **resumes**.
- `GET /conversations` and `GET /conversations/{id}/turns` rehydrate the UI.
- Voice and text write to the **same** `Conversation` — one continuous thread
  across modalities, which is the unified-memory goal.
- Working memory = last *N* turns verbatim + `running_summary` for everything
  older. The summary is recomputed asynchronously every *K* turns, so the
  conversation window stays bounded no matter how long the thread runs.
- `ParticipantState.history` in the voice worker becomes a cache of the
  `Conversation`, not a separate source of truth.

### 3.9 Consolidation, decay, forgetting

Scheduled background jobs, all idempotent:

| Job | Cadence | Action |
|---|---|---|
| **Rollup** | nightly | Cluster the day's `episodic` → distilled `semantic` with `derived_from` provenance |
| **Dedup sweep** | nightly | Merge near-identical `semantic` (cosine > 0.92), union provenance, keep highest confidence |
| **Decay** | nightly | Recompute effective importance; below threshold → `archived` |
| **Conflict scan** | weekly | Detect contradictory active facts → close the older, link `supersedes` |
| **Guest GC** | daily | Purge `guest-*` partitions idle > 30 days — closes §1.9 |

**Forgetting is a state machine, never a silent delete:**

```
active ──decay──> archived ──user action──> deleted (hard, cascading)
   │                  │
   └──superseded──────┘   (retained for versioning/audit)
```

`archived` records are excluded from default retrieval but remain searchable on
explicit deep recall. Hard delete happens only on user request or erasure
obligation, and must cascade to embeddings *and* to memories `derived_from` the
deleted record — otherwise a distillation quietly outlives the fact it came from.

### 3.10 Sources — the MCP extensibility seam

```python
class Source(Protocol):
    source_type: SourceType
    async def sync(self, cursor: str | None) -> AsyncIterator[RawArtifact]: ...
```

Every connector — chat, upload, GitHub, LeetCode, LinkedIn, Gmail, Calendar,
Notion, Drive — yields `RawArtifact` into the **same** extraction pipeline. Adding
an integration means writing one adapter, not touching memory internals.

`sync_cursor` per connector gives incremental sync. MCP servers slot in as
`Source` implementations. This is the difference between an architecture that
absorbs the user's "future integrations" requirement and one that needs surgery for
each.

### 3.11 Observability

Every retrieval emits a structured `RetrievalTrace`: query, per-channel candidates,
fused ranking, final selection, items dropped for budget, per-stage latency.

Metrics worth alerting on:

- extraction lag (queue depth, p95 event age) — the async pipeline's health
- memory count by kind and status; growth rate
- dedup rate and consolidation compression ratio
- retrieval: cache hit rate, budget utilisation, tier-2 drop rate
- **grounding rate** — how often the answer actually used injected memory

That last one is the only metric that measures whether the memory system is
*working* rather than merely running.

---

## 4. The contentious decision: pgvector vs Qdrant

**Recommendation: introduce a `VectorStore` port now; target pgvector; keep a
Qdrant adapter; migrate only after the port is proven.**

**For pgvector:**

- **Atomicity.** The record and its embedding commit in one transaction. This
  structurally eliminates the divergence the code currently logs about (§1.5). No
  amount of retry logic in a dual-write achieves this.
- **Hybrid retrieval in one query.** Vector + trigram + structured filters +
  recency, planned by one query planner. Today that needs app-level fan-out across
  two systems and manual fusion.
- **Scale reality.** Single user → tens of thousands of memories over years. Ten
  users × 100k = 1M vectors. pgvector with HNSW is comfortable there. Qdrant's
  advantages appear at 10M+ vectors and sustained high QPS — orders of magnitude
  from this workload.
- **Ops surface.** Removes an external stateful dependency, an API key, a timeout
  config, a client singleton and a health check. Render's managed Postgres
  supports `CREATE EXTENSION vector`.

**Against — stated honestly:**

- Qdrant works today and is deployed; migration is real, unglamorous work.
- pgvector HNSW index builds are memory-hungry; on a Render **starter** plan that
  is a genuine constraint and must be load-tested before committing.
- If this ever becomes a large multi-tenant product, Qdrant scales further.

The port is the actual architectural move: it makes the choice **reversible** and
lets the migration be a Phase-5 decision made with real numbers rather than a
Phase-1 bet. Ship behind the interface; decide with data.

---

## 5. Why this is better

| Dimension | Today | Redesigned |
|---|---|---|
| **Growth behaviour** | Degrades — raw transcript noise accumulates | Improves — extraction distils, consolidation compresses, decay prunes |
| **Relevance** | Fixed source order; no cross-source ranking | Hybrid retrieval, RRF fusion, multi-signal rerank |
| **Context safety** | Truncate at 20k chars, hope ordering saves you | Tier allocation with guaranteed floor; drops are traced |
| **Consistency** | Dual-write, no transaction, documented drift | Record + embedding commit atomically |
| **Continuity** | Refresh loses the thread | Persistent conversations across refresh, restart, device, modality |
| **Recruiter demo** | Retrieves an empty partition | `visibility=public` scope against the owner's memory |
| **Latency** | Embedding write inside the turn | Extraction fully async; turn path is retrieval-only |
| **Extensibility** | New source ⇒ new bespoke code path | New source ⇒ one `Source` adapter |
| **Governance** | Delete-only, no versioning, no provenance | Versioned, archived, provenance-tracked, user-editable |
| **Testability** | Requires live Postgres + Qdrant + Cohere | L0 ports fake out; L2/L3 unit-testable |

### Trade-offs I am accepting

1. **More moving parts.** An async pipeline and background jobs are strictly more
   complex than synchronous writes. Justified because the alternative — extraction
   in the request path — is incompatible with voice latency.
2. **LLM cost per turn.** Extraction adds a call per turn. Mitigated by batching
   several turns per extraction, using a small fast model, and skipping
   low-information turns. Budget it explicitly rather than discovering it.
3. **Extraction can be wrong.** A wrong memory is worse than a missing one. Hence
   confidence scoring, provenance display, and user editing as *first-class*
   features rather than nice-to-haves.
4. **Eventual consistency.** A fact stated this turn may not be retrievable for a
   few seconds. Accepted: the current turn already has it verbatim in working
   memory, so the user never perceives the gap.
5. **Migration cost.** Real. Phased below, with dual-write and shadow-read so no
   phase is a cliff.

---

## 6. Implementation roadmap

Every phase ships independently, keeps the app working, and is reversible.

### Phase 0 — Foundations & subtraction ✅ **COMPLETE** *(no behaviour change)*

Delivered:

- **Characterisation tests first** (`tests/test_memory_characterisation.py`, 33
  tests) pinning `format_context_for_prompt`, `_format_chat_history_for_prompt`
  and `_extract_semantic_resume_chunks` before anything moved.
- **mem0 removed.** Dead `Memory.from_config`, the `mem0ai` dependency, and the
  `chroma_persist_dir` / `mem0_api_key` settings are gone.
- **`agent_playbooks` deleted** — model, three repository methods, three manager
  proxies, three routes, and the two `playbooks:*` scopes.
- **Typed retrieval results** (`app/memory/retrieval_result.py`) replacing the
  `List | "NO_DATA"` union, *and* fixing the correctness bug it concealed
  (see below).
- **God-class split.** `ShortTermMemory` 1,145 → 468 lines. Application records
  moved to `app/domain/{academic,email,jobs}.py`; shared engine and session
  factory extracted to `app/db/`.
- **~30 domain proxies removed** from `MemoryManager`; 8 call sites repointed.
- **Structural guards** (`tests/test_memory_domain_separation.py`) so an
  attendance or email method cannot drift back onto the memory objects.

Net: **+212 / −1,212 lines**, 179 → 254 tests passing.

**Correctness bug found and fixed while typing the results.** `retrieve_skills()`
swallowed its exception and returned `[]`; `search_all()` then labelled that
`status="OK"`, making a failed Qdrant lookup indistinguishable from a user with
nothing stored. The prompt consequently carried neither the "no skills data"
hint nor the refusal policy — precisely the state in which a model invents an
answer. `RetrievalStatus.ERROR` now distinguishes *"we could not find out"* from
*"there is nothing to find"*, and the prompt says "treat skills as unknown, not
absent" rather than asserting a falsehood.

**Known defect recorded, not fixed:** `_detect_name` has no section-heading
guard, so a resume whose name line is absent stores `"SKILLS"` as an `identity`
chunk at high importance. Pinned as a strict `xfail`; Phase 1 must fix it.

**Migration note:** `create_all` never drops, so deployed databases retain an
orphaned `agent_playbooks` table. Left in place deliberately — "never
destructive on the first pass". A later migration drops it.

### Phase 1 — Unified memory core ✅ **COMPLETE**

Delivered:

- **Taxonomy and record** (`app/memory/kinds.py`, `record.py`): nine kinds with
  per-kind decay half-lives, dual content representation, bitemporal validity,
  lineage, governance, and a version chain via `superseding()`.
- **`memory_records` table** with a *partial* unique index on
  `(owner_id, kind, content_hash) WHERE status='active'`, making exact dedup a
  database invariant rather than an application check-then-insert race.
- **Two L0 ports** — `RecordStore`, `VectorStore` — with a Postgres adapter, a
  Qdrant adapter, and an in-memory fake held to the same contract.
- **Dual-write** for profile facts, episodes, and tool outcomes, behind
  `MEMORY_V2_DUAL_WRITE`. Reads remain entirely on the legacy path.
- **Backfill** (`scripts/migrate_memory_v2.py`): dry-run by default,
  idempotent, resumable, non-destructive, preserving original timestamps.
- **Tests:** 131 new (record invariants, port contract, writer mapping),
  254 → 358 passing.

**Scope deviation, deliberate.** This phase was planned to introduce five ports.
It introduces two. `LexicalIndex` and `Cache` have no consumer until Phase 2 and
`EventQueue` none until Phase 3; defining them now would be interface design
with no implementation to validate it. Each arrives with its first caller.

**Embedding is deliberately not on the request path.** Records are written
`embedding_status='pending'` and vectorised by a background pass. A profile fact
saved mid-conversation must not pay for a Cohere round trip, and a voice turn
cannot afford one at all.

**Only structured writes are mirrored.** Raw conversation turns are excluded on
purpose: mirroring every utterance is the noise-accumulation behaviour Phase 3
exists to remove, and Phase 4 gives turns a proper home in `Conversation`/`Turn`.
Document chunks are backfilled as a point-in-time snapshot; new uploads begin
dual-writing in Phase 2, when retrieval starts reading from the new store.

**Bug found and fixed by its own test.** The profile-fact conflict path built
the replacement as a *fresh* record, so every version landed as `v1` with a null
`supersedes_id`. The history looked correct — one active row, N superseded rows —
while the links between them were missing, which is the part that makes it
reconstructable. Replacements now derive from the previous record via
`superseding()`, covered by a test that walks a three-version chain back to `v1`.

### Phase 2 — Retrieval engine ✅ **COMPLETE**

Delivered:

- **`LexicalIndex` port** with a Postgres `tsvector` adapter (`ts_rank_cd`,
  functional GIN index) and an in-memory fake. `plainto_tsquery` is used so a
  raw user utterance can never become a tsquery syntax error.
- **Scoring** (`retrieval/scoring.py`): per-kind exponential recency decay,
  saturating frequency, kind priors, a weighted rank score whose terms sum to
  1.0, and reciprocal rank fusion.
- **`RetrievalEngine`**: vector, lexical and structured channels run
  concurrently, fuse by RRF, hydrate in one bulk query, then rerank.
- **`ContextAssembler`**: tier allocation *before* rendering, with tier 0
  guaranteed.
- **`RetrievalTrace`**: per-channel candidates and latency, fused and ranked
  counts, selection by kind, budget utilisation, and every budget eviction.
- **Shadow mode** behind `MEMORY_V2_SHADOW_READ`, plus consolidation of the two
  duplicated `retrieve_context` + `format_context_for_prompt` call sites into
  `MemoryManager.build_memory_prompt`.
- **Tests:** 60 new (scoring, fusion, engine, budget, shadow safety),
  358 → 418 passing.

**Why RRF rather than score averaging.** Cosine similarity and `ts_rank_cd`
share neither scale nor distribution. Averaging them silently lets whichever
channel emits larger magnitudes dominate; RRF reads only *position*, which every
channel agrees on.

**Why max-scaling rather than min-max normalisation.** When every candidate
scores similarly, min-max stretches trivial differences across the full range
and manufactures confidence the underlying signal does not support.

**Graceful degradation is a tested property, not a hope.** Every channel is
independent and individually fallible: Qdrant unreachable, no embeddings
computed yet, FTS index missing. Any subset can fail and retrieval returns what
remains, flagged `degraded` in the trace. A user whose vector store is down gets
slightly worse memory, not a broken assistant.

**Tier 0 is guaranteed under any budget.** Tested at a 10-token budget:
identity, preferences and goals still render. Allocation-before-render also
means records can only be dropped whole — the old truncate-at-20,000-characters
cut mid-record, producing fragments the model reads as complete statements.

**Shadow defaults OFF**, unlike dual-write. Comparing against a store that has
not been backfilled yet produces noise, not signal. It is also fully detached:
tests assert that a shadow which raises, or hangs for five seconds, changes
neither the served prompt nor the turn's latency.

**Not yet done in this phase, deliberately:** the vector channel is wired but
inert until embeddings exist (records are written `pending`), and document
ingestion still writes only to the legacy collections. Both land with the
Phase 3 background worker, which is the natural owner of batch embedding.

### Phase 3 — Async extraction ✅ **COMPLETE**

Delivered:

- **`EventQueue` port** and the `memory_events` outbox, with a Postgres adapter
  using `SELECT … FOR UPDATE SKIP LOCKED` inside the claiming transaction so
  concurrent workers can never process the same conversation twice.
- **Batched LLM extraction** over a window of turns → typed candidates with
  kind, self-contained content, importance, confidence, and an optional
  `dedup_key`.
- **Ingestion pipeline**: credential governance, exact de-duplication, and
  `dedup_key` conflict reconciliation that supersedes rather than duplicates.
- **Embedding pass** — the piece that finally activates the Phase 2 vector
  channel, which had been wired but inert because nothing produced vectors.
- **Worker** with per-group and per-stage failure isolation, runnable
  in-process from the app lifespan or standalone via
  `scripts/run_memory_worker.py`.
- **Tests:** 51 new (queue lifecycle, extraction parsing, governance, dedup,
  embedding failure modes, worker resilience), 418 → 469 passing.

**Retry, not discard.** A failed event returns to `PENDING` until the attempt
ceiling, then parks as `FAILED`. Dropping on first failure would mean a
transient Groq outage silently loses everything the user said while it was down.
`SKIPPED` is deliberately distinct from `FAILED`: a conversation that yielded no
durable fact is the *normal* outcome for small talk and must never be retried.

**Failures are isolated at every level.** One malformed conversation cannot stop
other users being extracted; a broken embedding pass cannot fail the extraction
cycle; a completely unreachable queue cannot kill the loop. An unattended worker
that dies on the first bad payload is worse than no worker, because nothing
reports it — memory simply stops improving and nobody notices for a month.

**Embedding failures leave records `PENDING`, not `FAILED`.** Marking them
failed would exclude them from semantic search permanently over what is almost
always a transient outage. A partial embedding batch is refused outright:
pairing records with the wrong vectors is worse than not embedding, because the
resulting search results look plausible.

**Bug found by its own test.** `MemoryIngestor` inferred the write outcome by
comparing ids — but a duplicate and a supersession *both* return a record whose
id differs from the one submitted, so every supersession was miscounted as a
duplicate. Fixed by having the writer report a `WriteOutcome` rather than
leaving the caller to reverse-engineer it.

**Scope note — raw-utterance embedding is gated, not yet removed.** The
mechanism ships behind `MEMORY_V2_REPLACE_RAW_EMBEDDING`, defaulting **off**,
because `retrieve_context` still reads that data for the "User Preferences &
Interests" section of the *served* prompt. Flipping it before the Phase 6 read
cutover would remove a live signal while its replacement is still only running
in shadow. The flip belongs with the cutover, not here.

**Also deferred deliberately:** semantic near-duplicate merging (cosine > 0.92)
stays in Phase 5's nightly sweep, where it can reuse vectors that already exist
rather than paying for an extra embedding per candidate on the ingest path.

### Phase 4 — Conversations & working memory ✅ **COMPLETE**

Delivered:

- **`conversations` + `conversation_turns`** with a `ConversationRepository`.
  The conversation's primary key *is* the existing `session_id` string, so no
  mapping layer exists and the browser's `session_<uuid>` keys coexist with the
  voice worker's `lk_<room>_<identity>` keys untouched.
- **Working memory** (`retrieval/working.py`): recent turns verbatim plus a
  running summary of everything older, wired in as tier 1 of the context
  budget.
- **Turn dual-write** from `MemoryManager`, tagged by modality — voice and text
  land in the same thread, and a thread used both ways is marked `mixed`.
- **Async summarisation** in the worker, folding aged-out turns into the
  running summary.
- **REST endpoints** — list, fetch-with-turns, archive — all owner-scoped.
- **Frontend persistence**: `conversation_id` in `localStorage`, rehydrating
  the transcript on mount.
- **Backfill** of existing `chat_history`, grouped by `(user_id, session_id)`.
- **Tests:** 32 new, 469 → 501 passing.

**Sequence numbers are assigned atomically**, via `UPDATE … RETURNING` on the
conversation row rather than `SELECT max(sequence)`. The latter races: a voice
turn and a text turn landing together would read the same maximum and collide.

**Only aged-out turns are summarised.** The summary marker trails the verbatim
window by design — folding in turns still shown in full would put the same
exchange in the prompt twice, once condensed and once verbatim.

**Clearing the chat starts a new thread rather than emptying the view.** The
old conversation is archived server-side and its extracted memories stay valid;
without this, the next reload would cheerfully resume the conversation the user
just asked to clear.

**Storage access is treated as fallible.** Private browsing and some embedded
webviews deny `localStorage`. A non-resumable session is a degraded experience,
not a broken one, so every access is guarded.

**Test-hygiene bug found and fixed.** Adding the summariser to the worker made
the Phase 3 worker tests fall through to the *real* summariser, which opened a
live Postgres connection on every run. They still passed — the worker swallows
stage failures — and only a `RuntimeWarning` about an un-awaited asyncpg
coroutine revealed it. The tests now inject a stub explicitly. Worth noting as
a pattern: exception-swallowing code turns a leaked dependency into a silent
one, so tests around it must inject rather than default.

### Phase 5 — Lifecycle ✅ **COMPLETE** · vector-store decision ⏸ **still open**

Delivered:

- **Lifecycle operations on `RecordStore`** — `set_status`, `hard_delete`,
  `find_derived_from`, `iter_active`, `duplicate_dedup_keys`, `owner_activity`
  — on both adapters.
- **Decay engine**: archives records whose `importance × recency` has fallen
  below threshold, exempting pinned records, the always-injected kinds, and
  anything younger than the minimum age.
- **Conflict reconciliation**: closes duplicate `dedup_key` slots that raced
  past the writer's check.
- **Semantic de-duplication**: merges near-identical same-kind memories using
  the vector store, unioning provenance into the survivor.
- **Guest collection**: purges anonymous partitions after a retention window.
- **Cascading erasure**: `forget_record` follows `derived_from` chains to any
  depth and deletes the vectors alongside the rows.
- **Worker integration** on a separate, much slower clock, plus
  `run_memory_worker.py --maintain`.
- **Tests:** 33 new, 501 → 534 passing.

**Weighted toward what must *not* be removed.** These are the only jobs in the
system that take things away, so most of the tests assert survival: identity at
ten years old, pinned records at any score, important-but-old records, and
anything too young to have proven itself. A decay engine that is slightly too
lazy wastes rows; one that is slightly too eager loses the user's name.

**Episodes are excluded from de-duplication.** Two similar-sounding events are
usually two genuinely different events, and merging them would fabricate a
history that never happened. Only `semantic` and `document` are dedupable.

**De-duplication is here rather than at ingest** because it needs a vector, and
paying for an embedding per candidate on the write path would put provider
latency back into a turn — precisely the cost Phase 3 moved out.

**Erasure completes even when the vector store is unreachable.** A stale vector
is a problem; a record the user asked to erase surviving is a worse one.

#### The vector-store decision — deliberately not made

The roadmap called for choosing pgvector or Qdrant here. **That decision needs
data this deployment has not yet produced**, so making it now would be a guess
dressed as a milestone. What is missing:

- the backfill has not been run, so the real record count is unknown;
- no embeddings exist yet, so no index has ever been built;
- shadow mode has never been enabled, so there are no retrieval latency
  measurements from either engine.

The `VectorStore` port already makes this reversible, which was the actual
architectural work — the choice is now a configuration decision rather than a
rewrite, and it costs nothing to defer.

**What would close it.** Run the backfill and the worker, then measure:

1. `SELECT count(*) FROM memory_records` and the pending-embedding count.
2. p50/p95 latency of the vector channel from `RetrievalTrace`.
3. pgvector HNSW index build time and memory on the actual Render plan — the
   one genuine risk, since index builds are memory-hungry and the starter tier
   is small.

If record count stays under roughly a million and index build fits the plan's
memory, pgvector wins on atomicity and one fewer stateful dependency. If the
build does not fit, staying on Qdrant is the correct answer and the port means
nothing was wasted either way.

#### Consolidation rollup — deferred with reason

Episodic → semantic rollup (distilling a week of episodes into durable facts)
is **not** implemented. Decay and de-duplication already bound growth, which was
the urgent property; rollup is a *quality* improvement whose prompt and cadence
need real extraction output to tune, and it adds a recurring LLM cost. Building
it against imagined data would mean rebuilding it against real data.

### Phase 6 — Control plane & visibility 🟡 **PARTIAL** (read cutover held)

Delivered:

- **`RetrievalScope`** (`app/memory/scope.py`) resolving a `Principal` into
  *whose* memory to read and *which* visibilities are permitted.
- **Scope threaded** through both workflows and the shadow path.
- **Control plane API**: browse and filter records, inspect one with its
  provenance chain, edit content/importance/pin/visibility, erase with cascade,
  export everything as JSON.
- **Tests:** 18 new, 534 → 552 passing.

**Read scope and write identity are separate — and the first attempt got this
wrong.** Passing the scoped owner as the workflow's `user_id` fixed reads and
simultaneously made every guest turn write into the *owner's* memory. Recruiter
chatter polluting the owner's history would have been a worse bug than the one
being fixed. The caller now always writes under their own identity;
`memory_owner_id` redirects reads alone, and the conversation window stays the
caller's own thread.

**Editing creates a version rather than mutating.** A correction is auditable
and the original recoverable — consistent with how conflicts are handled
everywhere else in the system.

**Records outside the caller's visibility return 404, not 403.** A 403 confirms
that the id exists, which is itself a disclosure.

**Provenance is the point of the control plane.** A system that forms memories
automatically is only trustworthy if the user can see what it concluded and
overrule it. Extraction is an LLM; a wrong memory is worse than a missing one,
so "why do you know this?" and an edit path are core features rather than
polish.

**Ordering bug caught by its own test.** The patch route looked the record up
before validating the payload, so an empty patch cost a database query — and
surfaced as a 500 when the table was unreachable rather than the 400 it always
was. Input validation now precedes I/O.

#### Not done in this phase, deliberately

**The read cutover is held.** Flipping retrieval from legacy to v2 is the one
step in this rollout that shadow mode exists to de-risk, and shadow mode has
never been enabled against real data. Cutting over now would mean asserting the
v2 path is at least as good as v1 with no evidence for it. The flag exists; the
flip is a decision for after the traces.

**Consequence, stated plainly: the recruiter demo is not yet fixed in
production.** The mechanism is complete and tested, but it only takes effect
once v2 serves reads — the legacy path has no visibility concept at all, since
`user_profile` and `episodic_memory` have no such column. Claiming the demo
works today would be false.

**No frontend memory UI.** The API is complete and exercised by tests; the
management screen is UI work with no architectural risk, and it is a poor use
of the remaining rollout risk budget compared with getting the cutover right.

### Phase 6.5 — Source-aware retrieval ✅ **COMPLETE**

Phases 0–6 built storage and retrieval. This phase supplies the thing that was
missing between them: **a decision about where to look.**

The evidence that it was missing came from real conversations, not from a
metric. Asked "What is my current CPI?", the assistant answered "I don't have
real-time access to your current CPI" — while the number sat in an `education`
chunk that `retrieve_section` returns correctly and always had. Nothing was
broken in the store. Every source was flattened into one prompt block, and the
model was left to infer both relevance and trust from an undifferentiated wall
of text.

Five defects, one shape:

| Symptom | Cause |
|---|---|
| "No real-time access to your CPI" | `CPI`/`SPI` absent from the CGPA vocabulary; query matched no subject and fell through to the planner |
| "Remember the name Devasi" changed the user's name | No distinction between canonical identity and an explicitly remembered value |
| "I don't have real-time access to the date" | No clock anywhere in the system |
| Three scoping questions before answering "how do I build an AI agent" | Clarification unbudgeted, and how-to questions read as questions about the user's own projects |
| "Okay." stored as long-term memory | Every utterance embedded verbatim |

**New modules**

- `app/memory/sources.py` — `QueryCategory`, `MemorySource`, `SOURCE_PRECEDENCE`,
  `SOURCE_TRUST`, `RetrievedMemory`. Precedence says where to look; trust says
  whom to believe when two sources disagree. They are deliberately different
  orderings — a passing remark is the freshest thing in the store and the least
  authoritative.
- `app/agents/query_intent.py` — deterministic categorisation by grammatical
  shape. Runs before the planner on every turn including spoken ones, so it
  must cost microseconds; the LLM keeps only the genuinely ambiguous residue.
- `app/memory/answerability.py` — `ANSWERABLE` / `PARTIALLY_ANSWERABLE` /
  `NO_DATA` / `AMBIGUOUS` / `ACTION_MISSING_PARAMETER` / `TOOL_ERROR`.
  `NO_DATA ≠ TOOL_ERROR` is the load-bearing distinction: telling a model there
  is no data after a Qdrant timeout asserts something false, and is precisely
  the state in which it invents an answer.
- `app/memory/identity.py` — canonical identity, the explicit-memory namespace,
  and a deterministic conflict resolver. Nothing in it deletes; every outcome
  records the losing value with its provenance.
- `app/memory/write_policy.py` — the gate deciding what earns a place in
  long-term memory. Conservative towards *not* storing: a fact missed today is
  usually restated, a store full of "okay" cannot be cleaned up later.
- `app/tools/time_tool.py` — the clock, timezone from configuration rather than
  the host. Injected into every agent prompt, and answerable without a model
  call or a planner (so a rate-limited LLM cannot take the date down with it).
- `app/agents/clarification_policy.py` — one question per conversation, and
  none at all once the user has asked to be questioned less.

**Wiring**

- `workflow.decide_route` is now the single routing decision, shared by the
  graph and the streaming path. The spoken path previously read
  `needs_clarification` straight off the planner, so every policy the graph
  applied was silently absent from voice turns.
- A `temporal` node answers date/time deterministically and bypasses reflect.
- `format_context_for_prompt(context, category=...)` renders only the sources
  that answer that kind of question. `category=None` renders everything, so no
  existing caller is narrowed silently, and an unmapped category degrades to
  showing *more* rather than nothing.
- `retrieve_resume` finally returns the `name` it has always stored. The chunk
  existed with `semantic_type="name"`; nothing read it back, so every caller
  doing `resume_data["name"]` — including the profile summary — got `None` from
  a store holding the answer.

**Preserved unchanged:** Qdrant, Postgres, `memory_records`, the event queue and
worker, embeddings, decay, dedup, erasure, `RetrievalScope`, visibility, dual
writes, shadow reads, resume ingestion, project entity IDs, and every legacy
fallback. No read cutover was performed.

**Tests:** `tests/test_memory_intelligence.py` — 131 cases covering all thirty
named regressions. Each asserts the decision, not only the answer: selected
category, selected source, whether retrieval occurred, whether clarification
occurred and why, and provenance.

### Phase 6.6 — Integrity audit ✅ **COMPLETE**

An audit of the lifecycle, store and caching layers, looking for defects rather
than missing features. Six were found; all six were places where the system
*reported* one thing and *did* another.

| Defect | Consequence |
|---|---|
| `forget_owner` erased 1 store of 9 | A user who erased their memory kept being answered from it. Legacy stores are authoritative until cutover, and none were touched. Guest collection was a no-op for the same reason. |
| `memory_cache` was never invalidated | Every write — résumé re-upload, name correction, memory deletion — kept serving the stale value for the full 5-minute TTL. On the deletion path the deletion silently did not take effect. |
| `memory_cache.get` returned its own storage | `retrieve_context` overwrote chat history *inside the cache* on every hit, corrupting the entry for the next reader and racing with concurrent turns on the same user. |
| `identity.resolve_conflict` was dead code | Written and tested last phase, called from nowhere. The live writer still superseded on any content difference, so an inferred remark permanently overwrote résumé- and user-stated facts. |
| `canonical_name` classified as `semantic` | Moving identity to its own key silently cost it always-injected status, the identity kind prior, and decay exemption — a name on a 365-day half-life. |
| `MemoryRecord.pinned` was never set | Honoured by `is_decay_exempt` and by ranking, assigned by nothing. Explicit vs inferred existed in the schema and nowhere in behaviour. |

**New:** `app/memory/erasure.py` — cross-store deletion with a per-store report.
Stores are enumerated in one table; partial failure is named, never swallowed
(`complete` is false and `failed_stores` lists the survivors); a store that
cannot report a count returns `None` rather than a misleading `0`. Two drift
guards in the tests fail the moment a Qdrant collection or an owner-keyed table
is added without being registered.

**New:** `DELETE /memory/all?confirm=erase` — there was no right-to-erasure
endpoint at all. Returns 207 on partial erasure, because telling a user their
data is gone while some of it survives is the same false statement as reporting
NO_DATA after a failed lookup.

**New:** the `degraded` workflow node. Found by a live run during a real Groq
outage: every question became an error page, including ones whose answer is a
database row. Identity and explicit-memory questions are now answered from
stored records with no model call — they cannot hallucinate, because they
compose no prose beyond a fixed template — and the reply distinguishes "I have
no record of this" from "I couldn't think right now". Categories needing
synthesis still fail honestly rather than guessing from fragments.

**Changed:** `resolve_conflict` is wired into `MemoryWriter.upsert_with_outcome`,
which gained `WriteOutcome.REJECTED`. Its equal-trust tie-break was also wrong —
it kept the stored value, so a user correcting their own explicit fact was
silently refused; only a demonstrably *older* claim now loses.

**Preserved:** every existing store, flag, contract and fallback. No read
cutover. 941 pre-existing tests still pass unchanged, plus 52 new ones.

### Phase 7 — Connectors

- `Source` protocol + `sync_cursor`.
- GitHub, LeetCode, LinkedIn first (highest signal for this user's goals).
- MCP adapters for Gmail, Calendar, Drive, Notion.
- **Tests:** adapter contract suite; incremental-sync idempotency.

---

## 7. Decisions

### Resolved (2026-08-07)

| Decision | Choice | Rationale |
|---|---|---|
| **Starting phase** | Phase 0 first | Subtraction before construction. Every later phase gets smaller, and the characterisation tests added here are what let subsequent phases *prove* they preserved behaviour rather than assert it. |
| **`agent_playbooks`** | **Delete** | Write-only feature with no reader. Rebuild against the new architecture if prompt versioning becomes a real requirement. |
| **Extraction cadence** | **Batched every *N* turns** (default `N = 5`) | ~5× cheaper than per-turn, and a multi-turn window produces *better* memories — a single turn frequently lacks the context to yield a self-contained fact. Freshness is not compromised in practice: the current turn is already verbatim in working memory, so the user never perceives the extraction lag. |

Implications of batched extraction for Phase 3:

- The outbox worker accumulates `memory_events` and triggers extraction when
  either *N* turns are pending **or** a conversation goes idle past a threshold
  (so a 2-turn conversation is not stranded unextracted).
- Conversation end / session close forces a flush.
- `MEMORY_EXTRACTION_BATCH_SIZE` and `MEMORY_EXTRACTION_IDLE_FLUSH_SECONDS`
  become tunable settings rather than constants.

### First real run — 2026-08-07

Migration and worker executed against the live local Postgres, Qdrant Cloud,
Cohere and Groq.

**Migrated:** 9 episodes + 1 tool outcome → 10 `memory_records`; 62 chat rows →
4 conversations / 62 turns; FTS index created. Zero failures. A second
`--apply` created nothing, confirming idempotency in practice rather than only
in tests.

**Two real bugs, both found only by running it:**

1. **The `memory_records` Qdrant collection was never created.**
   `QdrantVectorStore.initialize()` existed but nothing called it. The
   embedding pass failed with a 404 — and degraded exactly as designed: records
   stayed `pending` rather than being marked failed, summarisation and
   maintenance still completed, nothing was lost. Fixed by ensuring the
   collection at startup *and* lazily on first write, since the standalone
   worker never goes through the FastAPI lifespan.

2. **The collection was created with the wrong payload indexes.**
   `ensure_collection` hardcoded the legacy field names (`user_id`, `type`),
   while the new store filters on `owner_id` and `kind`. Qdrant rejects a filter
   on an unindexed field with a **400, not a slow scan**, so the vector channel
   returned nothing while reporting itself healthy. `ensure_collection` now
   takes the index set, and `QdrantVectorStore` declares its own.

Neither was reachable by unit tests: both live in the gap between a fake that
accepts anything and a real service with its own schema requirements.

**Measured latency** (single user, 10 records, Qdrant in `eu-central-1`):

| Stage | Cold | Warm |
|---|---|---|
| Cohere query embed | ~715 ms | ~0 ms (60 s cache) |
| Qdrant search | ~715 ms | — |
| **Full v2 retrieve + assemble** | ~1 300 ms | **~168 ms** |

The cold path is dominated by two cross-continent round trips, not by anything
in the ranking code. This is the strongest evidence so far *for* pgvector: it
would remove one of those hops entirely by putting vectors in the database the
request is already talking to.

**v1 vs v2 on the same query:** v1 produced 1 687 chars across 3 sections; v2
produced 3 260 chars / 706 tokens (12 % of budget) across 3 sections, drawing on
all three channels with `degraded: false`. v2 surfaces strictly more, correctly
typed — but with 10 records total this is not yet evidence about *ranking*
quality, only that the pipeline works end to end.

**Also observed:** the Qdrant `resume_chunks` / `skills_chunks` /
`projects_chunks` collections are **empty** — no résumé has ever been ingested.
That explains the `profile agent: "I don't have information about your
projects"` replies visible in the migrated episodes, and it means the document
memory path has never actually carried data.

### Still open

1. **Vector store target** — pgvector (atomicity, simplicity) vs stay on Qdrant.
   The first latency numbers now favour pgvector (see above), but the deciding
   measurement — HNSW index build time and memory on the Render starter plan —
   still has not been taken.
2. **Guest retention window** — 30 days proposed for GC.
3. **Extraction model** — small/fast (cost) vs the main model (quality). Decide
   in Phase 3 against the extraction golden set.

---

## 8. Migration safety rules

Applied to every phase:

- **Dual-write before dual-read; shadow-read before cutover.** No phase flips
  behaviour and storage at once.
- **Feature-flag each cutover** (`MEMORY_V2_*`), reversible without a deploy.
- **Migrations are idempotent, resumable, `--dry-run` first** — following the
  existing `scripts/migrate_add_constraints.py` convention.
- **Never destructive on the first pass.** Old tables are dropped a phase *after*
  the new path is serving traffic, not in the same one.
- **Characterisation tests first** (Phase 0), so equivalence is provable rather
  than asserted.
