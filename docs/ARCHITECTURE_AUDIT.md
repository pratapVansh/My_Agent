# My_Agent → Personal Jarvis: Architecture Audit

> **Scope** `app/` · `frontend/` · `tests/` — 150 Python modules, 1,116 passing tests
> **Branch** `main` · **Commit** `6e14b05` · **Date** 2026-08-10
> **Status** Audit only. No source files were modified in producing this report.

A read of the system as it actually stands after the memory work, and a phased route to a
dependable personal assistant. Every claim below is traced to a file in this repository.

---

## Contents

- [Verdict](#verdict)
- [1 · How the current agent actually works](#1--how-the-current-agent-actually-works)
- [2 · Architectural debt register](#2--architectural-debt-register)
- [3 · Target architecture](#3--target-architecture)
- [4 · Where MCP fits — and where it does not](#4--where-mcp-fits--and-where-it-does-not)
- [5 · Job agent, rebuilt around evidence](#5--job-agent-rebuilt-around-evidence)
- [6 · The action and confirmation layer](#6--the-action-and-confirmation-layer)
- [7 · What makes it feel like one assistant](#7--what-makes-it-feel-like-one-assistant)
- [8 · Phased roadmap](#8--phased-roadmap)
- [9 · Recommended Phase 1 — the first eight tasks](#9--recommended-phase-1--the-first-eight-tasks)

**Legend** — `HAVE` already built · `PARTIAL` exists but incomplete · `BUILD` does not exist

---

## Verdict

### A strong memory system wearing a thin agent

The memory layer is the most carefully designed part of this codebase and, in places, genuinely
novel — deterministic source precedence, provenance you can interrogate, an identity model that
separates *who you are* from *a name you asked me to keep*, and a NO_DATA/TOOL_ERROR distinction
most systems never make. It is also where essentially all the test weight sits.

The agent layer around it has not had the same treatment. It is a planner LLM, four specialists
holding closures over ad-hoc dicts, and a reflect loop that retries by re-running the same
specialist. It has **no confirmation layer**, **no typed tool contract**, and **almost no tests**.
That asymmetry — not a shortage of agents — is what stands between this and a Jarvis.

> ### ⚠ Single highest risk
>
> `send_email` is an ordinary tool in the email agent's registry. Nothing deterministic stands
> between the model deciding to call it and SMTP delivering it. The only guards are a scope check
> (`email:send`, owner-only), address validation, and a 300-second in-process duplicate window.
> The instruction to confirm before sending lives in the system prompt as English prose
> (`email_agent.py:49`) — exactly the prompt-only correctness the memory layer was deliberately
> built to avoid.

The good news is that the hard part is done. The routing spine (`query_intent` → `sources` →
`answerability` → `provenance`) is the right substrate for an action layer, and the confirmation
system described in §6 is a natural extension of it rather than a rewrite.

---

## 1 · How the current agent actually works

### 1.1 Text request path

```
POST /api/v1/agents/query
  → RateLimitMiddleware
  → JWT → Principal (scopes, role)
  → run_workflow()                         [app/agents/workflow.py]
  → parallel_init_node                     [memory retrieval ∥ planner]   LLM
  → decide_route()                         [deterministic]
      ├── temporal      → END              no model call
      ├── provenance    → END              no model call
      ├── degraded      → END              no model call
      ├── clarification → END
      └── profile | job | email | academic  LLM, tool loop ≤3 iterations
              → reflect  (retry | next_step | done)
              → response → END
```

`parallel_workflow_enabled` defaults to true, so memory retrieval and the planner run concurrently,
then `route_after_init` calls `decide_route`. Routing itself is deterministic:
`query_intent.classify` assigns a `QueryCategory` by grammatical shape, `sources_for` gives the
ordered stores, and `agent_for` can override the planner outright. Four categories terminate
without any specialist at all.

### 1.2 Voice path, and how it differs

`livekit_worker.py` holds a `ParticipantState` per speaker: Deepgram bridge for STT, an agent audio
track for TTS, one `turn_task` at a time under a lock, and a stall watchdog. Interim transcripts
drive barge-in; final transcripts start a turn. Then comes the divergence that matters:

| Concern | Text | Voice |
|---|---|---|
| Routers | One — `decide_route` | **Two** — `hybrid_router` then `decide_route` |
| Tool access | Always available | Only on the `tool_required` branch |
| Engine | `run_workflow` (graph) | `run_streaming_workflow` *or* `run_workflow` |
| History | Postgres conversation store | In-process `state.history`, 12 turns |
| Interruption | None | Stop-words + speech barge-in |
| Output | JSON response | Chunked TTS + data-channel events |

The consequence is sharp: `hybrid_router` classifies a spoken turn as conversational or
tool-requiring *before* the deterministic classifier ever sees it, and the conversational branch has
**no tools at all**. A misroute there does not degrade the answer — it removes the assistant's
ability to look anything up.

### 1.3 Inventory

| Layer | Components | Notes |
|---|---|---|
| Agents | planner, profile, job, email, academic, response | All extend `BaseAgent`; profile is the largest (~38 kB) |
| Graph nodes | temporal, degraded, provenance, clarification, reflect | First three answer with no model call |
| Tools | job_search, email_draft, timetable, attendance, time, timetable_pdf_parser | ~23 tool callables across 4 agents |
| Memory | qdrant LTM, manager, short/smart, retrieval v2, cognition worker, stores, identity, provenance, answerability | The mature half of the system |
| External | Groq, Cohere, Qdrant, Postgres, Tavily, Deepgram, Cartesia, LiveKit, SMTP, LangSmith | Tavily reachable *only* from inside job search |
| Auth | JWT, cookies, CSRF, 14 scopes, owner/guest | Enforced at route, resolver, and tool registry |
| Frontend | Next.js 14, `ChatShell.tsx` | Single 54 kB component |

**Per-agent tool counts** — profile 13+, academic 10, email 6, job 4.

### 1.4 Tool execution, errors, retries, cancellation

- **Execution** — `base_agent.execute_reasoning_loop`: JSON tool-call decisions, 3–5 iterations,
  15 s per tool, 30 s per LLM call, registry filtered by the caller's scopes.
- **Errors** — every tool is individually wrapped; failures and timeouts become observation strings
  and now also feed a deterministic answerability verdict.
- **Retries** — two independent mechanisms: `call_groq` retries 3× with backoff, and the reflect
  loop re-runs the whole specialist up to `MAX_ITERATIONS = 3` with failure context injected.
- **Cancellation** — exists only at the voice turn boundary. Cancelling the asyncio task stops the
  loop, but a tool that has already issued a side effect is not compensated.
- **Confirmation** — none. See the verdict above.

> ### ⚠ Retry × side effects
>
> The reflect loop can re-run a specialist that already sent an email. The only thing preventing
> double delivery is `EmailSenderService._recent_sends`, an in-process dict with a 300-second
> window — which does not survive a restart and does not exist across replicas.

### 1.5 Where clarification still happens

Mostly solved. `clarification_policy` permits questions only from `AMBIGUOUS_ACTION` and
`ACTION_REQUEST`, budgets one per conversation, and takes a sticky opt-out. The residue is
`_is_underspecified_action`: a short imperative with no digits and few content words is called
under-specified, so *"schedule a meeting"* asks a question. For a Jarvis with a calendar, that
should usually be resolved from context and proposed as a draft, not bounced back.

---

## 2 · Architectural debt register

Ordered by what blocks the Jarvis goal, not by size.

### D1 — No action/confirmation layer · **critical**

Side-effecting tools (`send_email`, bookmark writes, attendance marking, memory deletion) are
indistinguishable from read-only ones in the registry. Nothing classifies effects; nothing can pause
for approval.

*Blocks:* every irreversible capability — sending, applying, booking, paying, deleting.

### D2 — Agent layer is effectively untested · **critical**

Of 1,116 tests, roughly 500 cover memory. There is **no** dedicated test file for the job, email,
academic or planner agents, for the reflect loop, or for the graph end-to-end.
`test_email_sender_safety.py` covers the SMTP wrapper, not the agent that calls it.

*Consequence:* the layer about to gain irreversible powers is the layer with no safety net.

### D3 — Conditional-edge state writes are discarded · **critical**

`decide_route` is a LangGraph conditional-edge function, so its writes to `state` never reach the
nodes. `error`, `memory_sources`, `followup_subject` and `profile_intent` are all set there.
Confirmed live: terminal nodes ran with a stale error flag and empty sources.

*Fix shape:* promote routing to a real node that returns state; keep the edge function a pure
selector.

### D4 — Intent vocabulary duplicated three ways · **medium**

`hybrid_router._TOOL_PATTERNS`, `query_intent`'s category rules, and the planner's prompt each
independently describe what counts as a job/email/timetable request. They already disagree —
`_CHAT_PATTERNS` matches `stop|wait`, which the new `interruption` module also owns.

### D5 — Streaming path re-implements graph routing · **medium**

`streaming_workflow` hand-handles `temporal`, `clarification` and now `provenance`, duplicating
logic that already exists as graph nodes. Each new terminal category must be added in two places or
silently regresses on voice.

### D6 — Untyped tool results · **medium**

Tools return free-form dicts. Conventions (`success`, `found`, `count`) are real but informal —
`base_agent` now has to *sniff* those shapes to derive answerability. A typed result is the natural
place to also carry effect class, preview, and idempotency key.

### D7 — Planner runs even when routing is already decided · **medium**

Every text turn pays for a 70B planner call. For deterministic categories its answer is discarded by
`agent_for`. A live run showed planner rate-limiting killing turns whose answers required no model
at all — the degraded path now rescues identity, clock and provenance, but not the rest.

### D8 — Per-process ephemeral state · **low**

Clarification budget, provenance records, email dedupe and voice history are all in-process dicts.
Correct for one instance; silently wrong behind a load balancer or across a restart.

### D9 — Job search is web search in a trench coat · **low**

`job_search_tool` queries Tavily with `site:linkedin.com OR site:indeed.com`, then recovers company
names with regex over page titles. There is no structured job schema, no real posting metadata, and
"apply" only drafts an email. See §5.

### D10 — Frontend monolith · **low**

`ChatShell.tsx` is 54 kB in one component and owns transport, audio, rendering and state. Action
previews and confirmation UI (§6) will need surfaces it has nowhere to put.

---

## 3 · Target architecture

The shape that carries all the capabilities you listed is not more agents. It is four new horizontal
layers under the agents you already have.

```
Surfaces          text · voice · notifications
      ↓
Router            one deterministic classifier, both modes
      ↓
Orchestrator      plan · execute · reflect
      ↓
Action Gateway    effect class · preview · confirm          ← new
      ↓
Capability Registry   internal tools + MCP tools            ← unified
      ↓
Connectors        GitHub · Gmail · Calendar · web           ← new

        ↑ reads ↑                    ↓ writes ↓
Memory (existing, unchanged) · Provenance · Answerability · Audit log
```

| Component | Status | What changes |
|---|---|---|
| Memory & retrieval | `HAVE` | Reuse as-is. Do not rebuild. |
| Deterministic router | `PARTIAL` | Exists for text; make it the only router for voice too |
| Provenance | `HAVE` | Extend to record actions, not just answers |
| Answerability | `HAVE` | Now wired into the tool loop |
| Scoped authorization | `HAVE` | Add per-connector scopes |
| Capability registry | `PARTIAL` | Per-agent dicts today → one typed registry |
| Typed tool contract | `BUILD` | `ToolResult` with effect, preview, idempotency key |
| Action gateway | `BUILD` | The core of §6 |
| Audit log | `BUILD` | Append-only record of every executed effect |
| MCP host | `BUILD` | §4 |
| Task/job runner | `PARTIAL` | Memory worker exists; generalize for long-running actions |
| Proactive engine | `BUILD` | §7 — last, deliberately |

---

## 4 · Where MCP fits — and where it does not

MCP earns its place for **third-party services with real auth surfaces and evolving APIs**. It is
the wrong tool for anything reading your own memory, because that data never leaves the process and
the protocol only adds a hop and a serialization boundary.

**The dividing line:**

- **Stays internal** — anything touching Qdrant/Postgres memory, identity, provenance, resume
  parsing, timetable/attendance (your own DB), job *matching* logic, the action gateway itself.
- **Becomes MCP** — GitHub, Gmail, Google Calendar, web search, and later cloud/file services. All
  are external, OAuth-shaped, and independently versioned.

Architecturally the MCP host sits *behind* the capability registry and *under* the action gateway —
never beside it. MCP tools are registered like any other capability, carry the same effect
classification, and pass through the same confirmation gate. **An MCP server must never be able to
perform an irreversible action without traversing §6.**

| Integration | Benefit | Auth | Effect | Failure handling |
|---|---|---|---|---|
| **GitHub** | Real project evidence for résumé and job matching: languages, recency, README, commit cadence | Fine-grained PAT, read-only, repo metadata + contents | read | Cache last good snapshot; return NO_DATA rather than "no projects" |
| **Web search** | General lookup as a first-class capability — today Tavily is trapped inside job search | API key, server-side only | read | Degrade to model knowledge, labelled as such |
| **Gmail** | *Reading* mail — currently impossible. Enables triage, thread context, reply drafting | OAuth 2.0; `gmail.readonly` + `gmail.compose`; withhold `gmail.send` | read + draft | Never silently retry a send; surface auth expiry as re-consent prompt |
| **Google Calendar** | Merges with your timetable to answer "am I free Thursday" truthfully | OAuth 2.0; `calendar.readonly` first, `calendar.events` only once §6 ships | write | Writes idempotent by client-side event id |
| **Job boards** | Structured postings instead of scraped titles | Per-provider key; ToS-bound | read | Fall back to Tavily, marked lower-confidence |

### Security model

Three non-negotiables:

1. **Process isolation** — MCP servers run as separate processes with their own credentials. The
   agent never holds a third-party token, and a compromised server cannot read your memory.
2. **Scope mapping** — every MCP tool maps to an existing `Scope`, so guest sessions cannot reach
   any of them.
3. **Untrusted descriptions** — tool descriptions returned by a remote server are untrusted text and
   must never be treated as instructions. This is the same discipline already applied to retrieved
   memory.

### Deliberately *not* MCP

A "memory MCP server". Your memory system is the crown jewel and is already well-factored behind
`ports.py`. Exposing it over a protocol would add latency, weaken the provenance chain, and create
an exfiltration path, in exchange for nothing.

---

## 5 · Job agent, rebuilt around evidence

Today's agent ranks with a weighted blend — `0.5·source + 0.25·overlap + 0.25·skill_ratio` — over
regex-extracted titles. It can produce a number but cannot justify it, and `skill_match_ratio`
divides by *all* your skills, so having a broad résumé mechanically lowers every score.

The redesign keeps the semantic matching, which is good, and adds a spine that can explain itself:

| Stage | Input → output | Status |
|---|---|---|
| 1 · Profile | Résumé sections + GitHub → typed `CandidateProfile` (skills with evidence, projects, experience spans) | `PARTIAL` |
| 2 · Sourcing | Query + preferences → normalized `JobPosting[]` | `PARTIAL` |
| 3 · Requirements | Posting text → `Requirement[]` (skill, level, must-have vs nice-to-have) | `BUILD` |
| 4 · Matching | Requirement × profile → `Evidence \| GAP` per requirement | `BUILD` |
| 5 · Ranking | Must-have coverage first, then nice-to-have, then freshness | `PARTIAL` |
| 6 · Explanation | Rendered from the evidence table, not generated freely | `BUILD` |
| 7 · Application prep | Draft + tailored bullets, held pending confirmation | `PARTIAL` |

### Never inventing qualifications

This is a structural guarantee, not a prompt rule. Every matched requirement must carry an
`Evidence` record pointing at a real memory item — a résumé section id, a project, a GitHub repo. A
requirement with no evidence is a `GAP` and renders as one. The explanation is composed from that
table, so the model never gets the opportunity to *assert* a skill; it only gets to phrase what the
matcher found. This mirrors exactly what `answerability` already does for facts.

**Worked shape:**

> Strong match (5/6 must-haves). Python — 4 projects incl. My_Agent. FastAPI — this repo.
> Qdrant/vector search — memory subsystem. Missing: 3+ years professional experience; your résumé
> shows internship experience only. I have not claimed otherwise in the draft.

---

## 6 · The action and confirmation layer

One gateway every capability passes through. Its job is to classify what an action *does*, and to
hold the irreversible ones until you say yes.

### Effect classes

| Class | Meaning | Gate | Examples |
|---|---|---|---|
| `READ` | No state change anywhere | Execute | search jobs, read timetable, recall memory, web search |
| `LOCAL_WRITE` | Changes your data, reversible by you | Execute, report | draft email, bookmark job, save preference |
| `EXTERNAL_WRITE` | Visible outside the system, hard to retract | **Confirm** | send email, create calendar event, submit application |
| `DESTRUCTIVE` | Removes data or is irreversible | **Confirm + echo** | delete memory, erase all, cancel booking |

The class is declared **on the tool**, alongside `scope` — a natural extension of a registry entry
that already carries one. It is *not* inferred from the request text, because the request text is
exactly what an injection controls.

### The flow for "Apply to this job"

1. Router recognises an application intent; orchestrator plans READ + LOCAL_WRITE steps.
2. Job is fetched, matched, explained. Résumé variant selected. Cover email drafted.
   *All of this runs without asking.*
3. The submission step is `EXTERNAL_WRITE`. The gateway suspends it and returns an `ActionPreview`:
   destination, subject, full body, attachments, and the match explanation.
4. You confirm. The gateway executes with an idempotency key and writes an audit record.
5. Anything else — "change the subject first" — mutates the pending action and re-previews.

So you say four words and get asked one question, at the only moment where a question is actually
load-bearing.

> ### ⚠ Invariants
>
> - The confirmation token binds to a **content hash** of the exact previewed action — you cannot
>   approve a draft and have a different email sent.
> - Tokens **expire**.
> - Confirmations are **single-use**, so a reflect-loop retry cannot re-spend one.
> - On voice, confirmation must be an **explicit affirmative**: silence, a barge-in, or an ambiguous
>   transcript is never consent.

---

## 7 · What makes it feel like one assistant

| Property | Status | Gap |
|---|---|---|
| Persistent context | `HAVE` | Strong — reuse it |
| Voice ⇄ text continuity | `PARTIAL` | Voice keeps its own 12-turn history; unify on the conversation store |
| Barge-in | `HAVE` | Stop-words now cancel LLM/tool work, not just TTS |
| Concise answers | `PARTIAL` | Response agent trims; length should follow modality |
| Tool transparency | `PARTIAL` | Provenance can explain — surface it in the UI, not only on request |
| Action previews | `BUILD` | §6 |
| Graceful failure | `HAVE` | NO_DATA vs TOOL_ERROR is exemplary |
| Long-running tasks | `BUILD` | Everything is request-scoped today |
| Notifications | `BUILD` | Data channel exists for voice; no general event bus |
| Proactivity | `BUILD` | Deliberately last |

**On proactivity:** the failure mode is an assistant that interrupts. Gate it on a *triggering fact*
(attendance crossed 75%, an exam is in 48 hours, a saved job closes tomorrow), a quiet-hours window,
and a frequency budget in the same spirit as the clarification budget — one unsolicited nudge per
period, and a sticky opt-out per category. Build it only after §6, because a proactive assistant
that can act without confirmation is the worst possible combination.

---

## 8 · Phased roadmap

### P0 · Reliability & safety — *make the agent trustworthy*

Nothing external gets built until an action cannot fire unreviewed.

- Action gateway + effect classes (D1)
- Typed `ToolResult` contract (D6)
- Agent-layer test harness (D2)
- Routing as a real node (D3)
- Audit log for every effect

### P1 · Personal capability — *make it useful daily*

Depth on what you already have, before breadth.

- Structured `CandidateProfile` from résumé
- Evidence-based job matching (§5)
- Web search as a first-class tool
- Unify voice/text routing + history (D4, D5)
- Skip planner on decided routes (D7)

### P2 · MCP integrations — *reach outside*

Each behind the gateway, each with its own scope.

- MCP host + registry bridge
- GitHub (read-only) → feeds §5
- Calendar read → merges with timetable
- Gmail read + compose
- Calendar/Gmail writes, gated

### P3 · Proactive Jarvis — *anticipate, sparingly*

Only meaningful once actions are safe and context is unified.

- Event bus + notification surface
- Long-running task runner
- Trigger rules with quiet hours + budget
- Daily brief (timetable + jobs + mail)
- Durable per-conversation state (D8)

---

## 9 · Recommended Phase 1 — the first eight tasks

Ordered by dependency. T1–T3 are the minimum that makes everything after it safe to build. None of
these require touching the memory system.

### T1 · Typed tool contract

Introduce `ToolResult` — `status` (ok / no_data / error), `effect`, `data`, `preview`,
`idempotency_key`. Adopt it in one agent first; keep dict-sniffing as a fallback so nothing breaks
mid-migration.

- **Files** — new `app/tools/contract.py` · `app/agents/base_agent.py`
- **Done when** — `_tool_yielded_evidence` reads a field instead of guessing a shape, and the
  fallback path is covered by tests.

### T2 · Action gateway with effect classes

Declare `effect` on every registry entry beside `scope`. READ and LOCAL_WRITE execute;
EXTERNAL_WRITE and DESTRUCTIVE return a suspended `PendingAction` with a content-hash-bound token.

- **Files** — new `app/agents/actions.py` · `base_agent.py` · all four agents
- **Done when** — a tool with no declared effect cannot be registered, and the default for an
  unknown effect is to confirm.

### T3 · Route `send_email` through the gateway

The first real consumer, and the one that closes D1. Sending must produce a preview containing final
recipient, subject and full body, then require an explicit confirmation. Delete the prompt sentence
that currently substitutes for this.

- **Files** — `app/agents/email_agent.py` · `app/services/email_sender_service.py`
- **Done when** — a test proves no sequence of model outputs can deliver mail without a valid,
  unexpired, single-use token matching the previewed content hash.

### T4 · Agent-layer test harness

A fake LLM that replays scripted tool-call sequences plus fake tools that record invocations. This
is the missing infrastructure behind D2 — without it, T2 and T3 cannot be verified adversarially.

- **Files** — new `tests/support/fake_llm.py` · `tests/test_agent_tool_loop.py`
- **Done when** — job, email and academic agents each have a happy path, a tool-failure path, and a
  retry path under test.

### T5 · Idempotency across the reflect loop

Retry currently re-runs a specialist that may already have acted. Carry an idempotency key on every
non-READ result and make the gateway refuse a repeat within the conversation, replacing the
in-process SMTP dedupe with something the whole system honours.

- **Files** — `app/agents/workflow.py` (reflect) · `actions.py`
- **Done when** — a forced reflect retry after a successful send delivers exactly one email in test.

### T6 · Promote routing to a real node

Fixes D3 at the source. A `routing_node` returns state; the conditional edge becomes a pure selector
reading what the node wrote. Then remove the source-derivation workaround in `_record_provenance`.

- **Files** — `app/agents/workflow.py`
- **Done when** — `memory_sources` and `error` are observably correct inside `temporal`, `degraded`
  and `provenance`, and the workaround is deleted with tests still green.

### T7 · Single routing vocabulary

Make `hybrid_router` consume `query_intent` rather than its own regex table: a category that owns an
agent implies `tool_required`. Removes D4 and stops voice losing tool access on a misroute.

- **Files** — `app/agents/hybrid_router.py` · `query_intent.py`
- **Done when** — `_TOOL_PATTERNS` is gone and voice/text agree on a shared routing corpus,
  including the stop-words now owned by `interruption`.

### T8 · Action audit log

Append-only Postgres table: actor, effect, tool, arguments hash, preview hash, confirmation token,
outcome, timestamp. The evidentiary base for everything in P2–P3, and the thing that makes an
unexpected action explainable after the fact.

- **Files** — new `app/domain/audit.py` · scripts migration
- **Done when** — every EXTERNAL_WRITE and DESTRUCTIVE execution writes exactly one row, and a test
  asserts no path executes without one.

### Sequencing note

T1 → T2 → T3 is the critical path and should land together. T4 ideally lands **before** T3 so the
confirmation gate is verified adversarially rather than by inspection. T5–T8 are independent of each
other and can be interleaved.

---

*Audit performed against working tree at commit `6e14b05`, main branch. No source files were
modified in producing this report.*
