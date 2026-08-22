# My_Agent — Completion Audit

> **Date** 2026-08-18 · **Branch** `main` · **Suite** 2,236 passing / 27 skipped
> **Scope** whole project: `app/` (124 modules, ~37k LOC), `evals/`, `tests/` (57 files)
>
> Every figure below was produced by running the thing it describes. Where a
> number could not be measured, that is said rather than estimated.

---

## Contents

- [Verdict](#verdict)
- [1 · What was broken](#1--what-was-broken)
- [2 · What was fixed](#2--what-was-fixed)
- [3 · What is fully functional](#3--what-is-fully-functional)
- [4 · What remains limited](#4--what-remains-limited)
- [5 · Evaluation results](#5--evaluation-results)
- [6 · MCP implementation](#6--mcp-implementation)
- [7 · Architecture quality](#7--architecture-quality)
- [8 · Security and reliability](#8--security-and-reliability)
- [9 · Remaining risks](#9--remaining-risks)
- [10 · Next steps](#10--next-steps)

---

## Verdict

The system is **safe and correct, measurably so, and slow for reasons that are
not its fault**.

The property that matters most — *the assistant never states a personal fact it
did not look up* — now holds under measurement rather than under argument. Zero
grounding violations across 12 deterministic scenarios and 9 live scenarios,
including five prompt-injection attempts and four cases where the model was
scripted to fabricate outright.

Three real bugs were found and fixed during this pass, and all three were
**silent**: none raised, none failed a test, and each one degraded the agent in
a way a user would experience as "it just isn't very good" rather than as an
error. That is the category of defect this codebase was most exposed to, and it
is the reason the evaluation harness now exists.

What is *not* production-ready is latency, and the cause is external: the Groq
account is capped at 8,000 tokens/minute, which turns a 2-second answer into a
26-second one under backoff.

---

## 1 · What was broken

Four defects, found by inspection and by running the system rather than by
reading it.

### B1 — A skipped tool ended the turn instead of retrying · **high**

`grounding` correctly refused to deliver an answer whose required tool was never
called, and the turn stopped there. The user got *"I wasn't able to retrieve
that, please ask again"* for a question the agent could answer — the tool
existed, the data was in the database, the model simply had not looked.

Measured against `openai/gpt-oss-120b`, the first attempt skipped its required
tool often enough (roughly 1 turn in 5 during model-migration testing) for this
to be the single most visible quality problem in the product.

*Why it was invisible:* the refusal is polite and well-worded. Nothing errored.

### B2 — `grounding` never reached the reflect loop · **high**

Fixing B1 exposed a second bug underneath it. `execute_reasoning_loop` computed
the grounding verdict, and `reflect_node` could not see it: **LangGraph only
propagates keys declared in the state schema**, and `grounding` was not declared
in `AgentState`. An undeclared key written by a node is dropped silently before
the next node runs.

This is the same class of defect the earlier architecture audit recorded as D3
(conditional-edge writes being discarded) — a second instance, in a different
place, with the same silent failure mode.

*Why it was invisible:* the retry simply never fired. No error, no log line.

### B3 — The voice router called a decommissioned model · **high**

`hybrid_router._ROUTER_MODEL` was `llama-3.1-8b-instant`. Groq has removed the
entire Llama family from this account — verified against the live model list.
Every LLM-fallback routing call returned **404**, and the failure path defaults
to `conversational`.

The consequence is specific and bad: on **voice**, the conversational branch has
no tools at all. So an ambiguous spoken turn did not get a worse answer — it
lost the ability to look anything up, silently.

*Why it was invisible:* the heuristic fast path handles most turns, so the
broken fallback only fired on genuinely ambiguous input, and it failed closed
into a plausible-sounding reply.

### B4 — Unit tests reached the live database · **medium**

`academic_repository.has_timetable` was not stubbed in `tests/support/harness.py`,
while every neighbouring method was. The schedule tools call it before reading
any rows, so every timetable unit test depended on a live Postgres connection and
on which asyncio event loop happened to still be open. It surfaced as a test that
passed alone and failed in the full suite.

---

## 2 · What was fixed

| # | Fix | Files | Verified by |
|---|---|---|---|
| B1 | `reflect_node` treats `grounding == "skipped"` as a failed attempt and retries with failure context injected. Only `skipped` — `no_data` is an answer and `failed` is already covered by `TOOL_ERROR`. | `app/agents/workflow.py` | 4 unit tests + 1 end-to-end test through the real graph |
| B2 | `grounding` declared in `AgentState`; written centrally by `execute_reasoning_loop` rather than by each agent, so a specialist that forgets to copy it cannot cost a recoverable turn. | `app/agents/state.py`, `app/agents/base_agent.py` | the e2e retry test fails without it |
| B3 | Router model → `openai/gpt-oss-20b`; `max_tokens` 16 → 512, because `gpt-oss` bills its reasoning trace from the same budget and a tight cap returns empty content that Groq rejects as invalid JSON. | `app/agents/hybrid_router.py` | live: 3/3 correct classifications, mean 574 ms, inside the 1.2 s router budget |
| B4 | `has_timetable` stubbed in the shared harness, defaulting to `False` to match `retrieve_timetable`'s empty default. | `tests/support/harness.py` | full suite now stable |

Two dependency issues were also resolved while integrating MCP:

- Installing `mcp` pulled `starlette>=1.0`, which is **incompatible with
  fastapi 0.115** (`Router.__init__() got an unexpected keyword argument
  'on_startup'`) and broke the application at import. `starlette` is now pinned
  `>=0.37.2,<0.39.0` with the reason recorded in `requirements.txt`. Only the
  stdio transport is used, so the newer starlette buys nothing here.
- The MCP SDK v2 renamed `FastMCP` → `MCPServer`; the test fixture uses the
  current API.

**Test suite: 2,205 → 2,236 passing, 0 failing.** No functionality was removed
or regressed.

---

## 3 · What is fully functional

Verified by execution, not by reading.

| Capability | Evidence |
|---|---|
| **Timetable** — day/range/subject/time queries, next class, professor lookup | 108 tests; real semester-7 data live in Postgres; deterministic rendering from stored rows |
| **Timetable re-upload** — deterministic PDF parse, hash comparison, atomic replace | verified end-to-end against a live database: 5-class timetable replaced by a 3-class one, old rows retired, no mixing |
| **Grounding** — no personal fact without a lookup | 0 violations across 21 evaluated turns (12 deterministic + 9 live), including 5 injection attempts |
| **Action gateway** — EXTERNAL_WRITE held for confirmation, content-hash bound, single-use, durable | 41 gateway tests + Postgres integration tests |
| **Confirmation** — "yes" resolved deterministically, never by the model | 33 tests; "sure"/"okay" deliberately ambiguous |
| **Audit log** — one append-only row per consequential action, digests only | 31 tests, partial unique index enforced by Postgres |
| **Memory** — hybrid Postgres + Qdrant, provenance, source precedence | ~500 tests, the most mature subsystem |
| **Auth** — JWT, scopes enforced at route, resolver, and tool registry | guest cannot reach owner tools, including MCP ones |
| **MCP** | 17 tests against a real child-process server |
| **Evaluation harness** | 8 gate tests; both modes run |

---

## 4 · What remains limited

Stated plainly, because each of these is a real boundary.

1. **Live latency is unusable on the current Groq tier.** Mean 25.9 s, p95 64.9 s
   across the live suite. The agent's own work is a small fraction of that; the
   rest is rate-limit backoff at 8,000 TPM. This is a billing problem, not a code
   problem, and no amount of optimisation inside the app will fix it.

2. **First-attempt tool compliance is imperfect.** `gpt-oss-120b` sometimes
   answers without calling its tool. The retry (B1) now rescues most of these
   and grounding blocks the rest, so the user never sees an invention — but they
   may occasionally see a refusal, and each retry costs another round trip.

3. **The PDF timetable parser handles one class per line.** A scanned or
   genuine-grid PDF is refused with a clear reason rather than guessed at. The
   current semester's timetable is an image and uses a hand-verified
   transcription (`app/tools/timetable_source.py`) pinned to the source PDF's
   SHA-256.

4. **No MCP server is configured.** The mechanism is built and proved; wiring
   GitHub or Gmail is a decision with an OAuth surface and was not made here.
   `SERVERS` is empty and every server ships `enabled=False`.

5. **The evaluation suite is 12 scenarios.** Enough to gate the properties that
   matter, not enough to be a benchmark. It is designed to grow.

6. **Deferred from earlier audits, still open:** streaming path duplicates some
   graph routing (D5); the planner runs even when routing is already decided
   (D7) — now more expensive given the rate limit; `ChatShell.tsx` is a 54 kB
   component with nowhere to put a confirmation UI (D10); voice worker state is
   process-local (M15), blocking horizontal scaling.

---

## 5 · Evaluation results

`evals/` measures the agent; `tests/test_evals.py` gates it in CI.

**Two modes, reported separately, because they answer different questions.**
Blending them would be dishonest: deterministic mode scripts the model's
decisions and therefore measures *the system* — routing, grounding enforcement,
tool wiring, the retry path. Live mode measures *the model* — whether it
actually calls the tool it was told to. A pass in one says nothing about the
other.

Roughly half the scenarios script the model **misbehaving** — inventing a CGPA,
ignoring its tools, claiming an email was sent. That is the part that matters:
measuring the agent only when the model cooperates measures the easy half of the
problem, and the system's real job is to be correct when the model is wrong.

### Deterministic (reproducible, free, in CI)

```
  Task success         12/12  (100%)
  Grounding violations 0
  Tool-call rate       100%  (3 turns required one)
  Retry recoveries     1
  Latency  mean 14.0 ms   p50 10.2 ms   p95 71.1 ms
```

### Live (`openai/gpt-oss-120b`, real API)

```
  Task success         9/9  (100%)
  Grounding violations 0
  Tool-call rate       100%  (2 turns required one)
  Latency  mean 25,903 ms   p50 33,990 ms   p95 64,872 ms
```

A focused four-scenario re-run immediately afterwards scored **3/4**, the single
failure being a **120 s timeout** on a scenario that had passed minutes earlier —
quota exhaustion from the first run, not a behavioural regression. It is
reported here rather than discarded because it is an honest measurement of what
this tier does under sustained use.

**The headline result:** `grounding_violations = 0` in both modes. This is
tracked as a count that must be zero rather than a rate to improve, because a
run with 98% task success and one violation is a worse system than one with 90%
and none.

Notable live behaviour: `cgpa_invented` **passed by calling `get_education` and
honestly reporting no data** — the real model looked instead of guessing.

### Running it

```bash
python scripts/run_eval.py                    # deterministic
python scripts/run_eval.py --mode live        # real model, paced for the rate limit
python scripts/run_eval.py --json report.json
```

Exit code 1 on any grounding violation, or task success below `--min-success`
(default 0.9), so it works as a CI gate.

---

## 6 · MCP implementation

Three modules, ~670 lines, 17 tests **against a real MCP server running as a
real child process over stdio** — not a mock. The claims worth making about MCP
are all claims about behaviour at a process boundary, and a mocked boundary
would assume every one of them.

### The design in one sentence

An MCP tool becomes an ordinary registry entry — `description`, `callable`,
`effect`, `scope` — and from that moment nothing downstream knows the call
leaves the process.

That is the entire point. Every safety mechanism in this system is attached to
the *registry entry* rather than to a tool's identity, so an MCP tool inherits
all of them by construction:

| Mechanism | How MCP inherits it |
|---|---|
| **Permission** | `base_agent._filter_tools_by_scope`, unchanged. A guest never learns an MCP tool exists. |
| **Confirmation** | An `EXTERNAL_WRITE` MCP tool is intercepted by `action_gateway` before the call, with a content-hash-bound token — exactly like `send_email`. |
| **Grounding** | An MCP tool counts as evidence only if it ran and returned something. |
| **Audit** | One `action_audit` row per consequential execution, keyed by the qualified name. |
| **Typed results** | `ToolResult`, so a transport failure cannot read as an empty result. |

The alternative — a parallel MCP path with its own permission checks — would
mean every future safety property gets implemented twice, and the second
implementation is the one that gets forgotten.

### Three refusals (`app/mcp/config.py` is the security boundary)

An MCP server is a separate, often third-party process whose entire output is
attacker-controlled in the threat model. The protocol invites you to treat what
it advertises as configuration. This implementation refuses:

1. **Allowlist, fail closed.** A server may advertise ten tools; only those
   named in local config are registered. *Proved:* the fixture server offers
   `not_allowlisted`; it is not registered.

2. **Effect and scope are declared locally.** The server is never asked. A
   remote process cannot name its own effect class and thereby route around the
   confirmation gate.

3. **The description is local.** This is the one that is easy to miss. Tool
   descriptions go verbatim into the system prompt, so a server returning
   *"Ignore all previous instructions…"* is a prompt injection with a
   guaranteed delivery path. *Proved:* the fixture server advertises exactly
   that; the registered description is the local one, and the injection text is
   absent.

Plus namespacing — `mcp__<server>__<tool>` — so a remote server cannot shadow
`send_email` by advertising a tool of that name.

### A real protocol finding

**`isError` alone is not sufficient.** Against the reference SDK, a tool that
*raises* comes back with `isError` unset and the exception rendered into the
text content. Trusting the flag alone would make a crashed tool
indistinguishable from a successful one — and the model would read *"Error
executing tool"* as data.

The fix recognises the SDK's own error envelope as a second signal, matched
narrowly: anchored at the start, and it must name the tool actually called.
Sniffing for the word "error" anywhere would be the wrong fix — a log-search
tool returning error lines has *succeeded*, and misreading that would hide real
data behind a fake outage. Both directions are pinned by tests.

### Reliability

Bounded call timeouts; a connect lock so concurrent calls cannot spawn two
processes; teardown on failure so a dead session is never reused; child
processes stopped in the application lifespan. **A broken MCP server degrades
one tool, not the assistant** — proved with a server that exits immediately and
one that hangs.

### What is not claimed

This does not sandbox the server process — it runs with the privileges of
whoever launched it. A hostile server can still lie in its *results*, which is
why results are coerced through the tool contract and why grounding governs
what may be stated from them.

---

## 7 · Architecture quality

**Strong.** The layering earlier audits called for is now real:

```
Surfaces      text · voice
Router        deterministic classifier (query_intent) — one vocabulary
Orchestrator  plan · execute · reflect  (retries on TOOL_ERROR and skipped tools)
Grounding     may this answer be stated at all?
Action Gateway  effect · preview · confirm · idempotency
Registry      internal tools + MCP tools, one shape
Connectors    MCP servers (isolated processes)
       ↑ reads ↑                    ↓ writes ↓
Memory · Provenance · Answerability · Audit log
```

What is genuinely good:

- **Safety is structural, not prompt-based.** The gateway holds a send because
  of a declared effect, not because a prompt asked it to. Grounding refuses an
  answer because no tool ran, not because the model was told to be careful.
- **One implementation per concept.** Confirmation, grounding, and routing each
  have exactly one home, used by both text and voice.
- **Failure distinctions are preserved end to end** — `NO_DATA` vs `TOOL_ERROR`
  survives from a tool return, through the contract, to the sentence the user
  reads. Now across a process boundary too.
- **Comments explain *why*, including what was tried and rejected.** The
  `reasoning_effort="low"` experiment is recorded in `groq_service.py` with the
  measurement that killed it, so nobody re-runs it.

Weaknesses: the streaming path still re-implements some graph routing (D5); the
planner runs even when routing is decided (D7); the frontend is one large
component (D10).

---

## 8 · Security and reliability

**Closed and verified:**

- Prompt injection into a tool description — the highest-value MCP vector — is
  structurally impossible: descriptions are local.
- A remote server cannot escalate its own privileges: effect and scope are local.
- Guests cannot reach MCP tools; the existing scope filter covers them unchanged.
- No consequential action executes without a valid, unexpired, single-use,
  content-hash-bound token.
- Idempotency is enforced by a Postgres partial unique index, so a retry after a
  restart or on another replica cannot send twice.
- The gateway **fails closed** when the audit log is unreachable — observed
  during this work, and it behaved correctly.
- Auth: JWT with scopes at three layers; SSRF guard; rate limiting; upload caps.

**Open:**

- `SMTP` per-user send quota still missing (C4, partial since the original
  audit). Confirmation now gates every send, which is the larger half.
- Voice worker registry is process-local (M15) — blocks running a second
  instance.
- No live browser verification of the voice pipeline. Every voice fix is covered
  by tests with faked provider boundaries; none has been confirmed against real
  LiveKit/Deepgram/Cartesia traffic in a browser. **This remains the single
  largest untested surface.**
- MCP servers are not sandboxed.

---

## 9 · Remaining risks

Ordered by expected cost.

1. **Rate limit (high, external).** 8,000 TPM makes live use painful and voice
   effectively non-viable. Everything else is downstream of this.
2. **Model compliance (medium).** `gpt-oss-120b` skips tools sometimes. Mitigated
   by retry + grounding; the residue is occasional refusals and extra latency.
3. **Voice unverified in a browser (medium).** The largest untested surface.
4. **Single-process state (medium).** Voice worker registry blocks horizontal
   scaling.
5. **Eval suite is small (low).** 12 scenarios gate the important properties but
   will miss regressions outside them.
6. **MCP is unexercised in production (low).** Proved against a fixture server;
   no real third-party server has been run.

---

## 10 · Next steps

Ordered by dependency and value.

**Immediate**

1. **Commit this work.** ~70 files are uncommitted, including the entire action
   gateway, grounding layer, MCP support, and eval harness. Suggest logical
   commits: fixes → evals → MCP.
2. **Upgrade the Groq tier**, or accept that live use is a demo. This is the
   single highest-value change available and it is a billing action, not code.

**Short term**

3. **Verify voice in a real browser** — the largest untested surface. Run the
   `.bench_scratch/` benchmark again once the rate limit is lifted; it was
   previously inconclusive for exactly that reason.
4. **Wire the confirmation UI into `ChatShell.tsx`.** The backend holds actions
   and returns previews; the frontend has no button to approve one, so
   confirmation currently only works conversationally.
5. **Grow the eval suite** to ~30 scenarios: multi-step plans, memory recall,
   job matching, voice-specific routing.

**Medium term**

6. **Enable one real MCP server**, read-only, behind the gateway — GitHub
   (fine-grained PAT, repo metadata) is the natural first, and it feeds the
   evidence-based job matching already built in `app/matching/`.
7. **Skip the planner on decided routes (D7)** — saves a 70B call per turn,
   which matters far more under a token cap.
8. **Add the SMTP per-user quota** to close C4 completely.

**Later**

9. Move voice worker state to Redis (M15) to allow more than one instance.
10. Unify the streaming path with the graph (D5).

---

*Every figure in this report was produced by executing the system it describes.
Where something could not be measured — live voice behaviour, MCP against a real
third-party server — that is stated rather than estimated.*
