# My_Agent — Working Plan

> Scope: **local development only** (`uvicorn` + `npm run dev`). Deployment,
> CORS/`ALLOWED_ORIGINS`, `render.yaml`, CI and production build issues are out
> of scope by decision, not by oversight.
>
> Status vocabulary: `todo` · `in progress` · `done` · `blocked`

---

## Task 1 — Tavily job search works end to end

| # | Item | Files | Status |
|---|---|---|---|
| 1.1 | Missing/placeholder `TAVILY_API_KEY` fails with one clear message — no silent empty result | `app/config.py`, `app/tools/job_search_tool.py` | **done** |
| 1.2 | Timeouts, rate limits and auth failures return a typed `TOOL_ERROR`, not `[]` | `app/tools/job_search_tool.py` | **done** |
| 1.3 | Empty result stays `NO_DATA` — distinct from a failed lookup | `app/tools/job_search_tool.py` | **done** |
| 1.4 | Live run through the real router + agent | — | **done — verified live 2026-09-21** with a real key. Tool layer: real results. Agent turn: intermittent, see 1.7 |
| 1.5 | Offline test with a mocked Tavily client | `tests/test_agents_job.py` | **done** — 9 tests |
| 1.6 | *Found live:* the clear message never reached the user — grounding replaced it with a generic refusal | `app/agents/job_agent.py` | **done** |
| 1.7 | *Found live 2026-09-21:* the model often never calls `job_search`; grounding then refuses | `app/agents/prefetch.py` | **done** — `job_search` is prefetched; **5/5 live** |

Route already works: `query_intent` → `JOB_SEARCH` → `job` agent → `job_search`
tool. No routing change needed.

**1.6 — what running it actually found.** Routing and the tool were both
correct, and the user still saw nothing useful. `grounding` discards the
model's answer on any JOB_SEARCH turn whose required tool produced no
evidence, and substitutes a generic line, because it knows only *that* nothing
was retrieved, never *why*:

| observed live | substituted | |
|---|---|---|
| model called nothing | "Let me check job listings again — I don't want to guess at it." | `grounding=skipped` |
| tool errored on the missing key | "Couldn't get to job listings just now. Try me again in a second." | `grounding=failed` |

Both invite a retry that can never work. The job agent now replaces them with
the reason when — and only when — the key is the cause, the same way a
`match_job` report replaces the model's prose. Verified live:

    Job search is unavailable: Set TAVILY_API_KEY in .env
    (get a key at https://tavily.com). No search was performed.

---

## Task 2 — Email sending works end to end

| # | Item | Files | Status |
|---|---|---|---|
| 2.1 | SMTP config validated at first use; placeholder email/password named explicitly, Gmail app-password mentioned | `app/services/email_sender_service.py` | **done** |
| 2.2 | ~~Same check at preview time~~ → moved to **startup**. A preview that reads `.env` makes the offline suite env-dependent (it broke 10 tests). The operator gets the signal at boot instead. | `app/main.py` | **done** |
| 2.3 | Per-user send-rate cap (audit item **C4**), next to the existing 300 s dedupe — 10/hour, charged to the confirmed action's owner | `app/services/email_sender_service.py`, `app/agents/confirmable_tools.py` | **done** |
| 2.4 | `SMTP_*` block documented in `.env.example` (was absent entirely) | `.env.example`, `README.md` | **done** |
| 2.5 | Live run of draft → preview → hold → `yes` → token → SMTP | — | **done — real message delivered 2026-09-21** to `vanshprataps2004@gmail.com` |
| 2.6 | Offline tests with SMTP mocked | `tests/test_email_sender_safety.py` | **done** — 7 tests |

Confirmation stays **chat-only**: draft → preview → `yes` → token → SMTP.
No frontend confirmation UI is built here.

**What the live run showed.** The chain holds end to end. A drafted send is
intercepted, previewed with the real recipient and body, and stored durably;
`yes` resolves it, the gateway executes it once against a content-bound token,
and SMTP refuses with the configuration message reaching the user verbatim:

    I couldn't complete that: Email sending is not configured. Set SMTP_EMAIL
    and SMTP_PASSWORD in .env. SMTP_PASSWORD must be a Gmail App Password ...

The refused send did **not** consume the rate cap, which is the intended
ordering. Two things worth recording from the same run, neither caused by this
work and neither a safety failure:

- The model called `send_email` twice in one turn, against its own prompt.
  Two actions were held, and `confirmation.resolve` refused to guess between
  them rather than sending either — the correct outcome.
- Several turns degenerated under the 8,000 TPM Groq cap: 429s, and one
  `400 tool_use_failed` ("Tool choice is none, but model called a tool").
  This is the latency/tier problem `FINAL_AUDIT` §4.1 already records.

---

## Task 3 — Quick local fixes

| # | Item | Files | Status |
|---|---|---|---|
| 3.1 | Log the real exception (repr + traceback) when memory init fails | `app/main.py` | **done** |
| 3.2 | Unreachable Qdrant: app still starts, but the warning is loud and unmissable | `app/main.py` | **done** |
| 3.3 | Document `playwright install chromium` in README setup | `README.md` | **done** |

Explicitly skipped: `/login` Suspense build error (dev mode works), `/health/deep`
auth, MCP wiring, eval expansion, horizontal scaling.

---

## Live verification run — 2026-09-21

Run against the real providers with real keys, driving `run_workflow` exactly
as `/api/v1/agents/query` does (owner scopes, owner `user_id`).

### Tavily job search — works at the tool layer, unreliable through the agent

`JobSearchTool.search_jobs` was called live and returned real listings:

    success: True — 10 raw Tavily results, 5 returned after ranking
    "600 Entry Level Backend Python Job Vacancies"
      https://in.indeed.com/q-entry-level-backend-python-jobs.html
      skills_matched=['Python']
    "100 Python Programming Job Vacancies in Kochi, Kerala"
      skills_matched=['Programming', 'Python', 'Linux']

Skill injection, Qdrant skill lookup, URL dedupe and per-result skill matching
all ran. **The tool itself is verified working.** `TAVILY_API_KEY` is live, so
1.4's blocker is gone.

Through the full router + agent the same query was run **5 times**. The route
is always correct (`JOB_SEARCH` → `job`, planner correctly skipped), but only
**1 of 5** turns delivered real listings. The other 4 ended on:

    Let me check job listings again — I don't want to guess at it.

The cause is visible in the logs and is **not** a tool failure — Tavily never
errored once:

    Agent 'job' answered a JOB_SEARCH question with grounding=skipped
    (used=[] errored=[])

The model returns prose **without emitting the tool call at all**, `grounding`
correctly discards it, `reflect` retries once, the second pass does the same,
and the turn ends on the refusal. On degraded turns the tool is never reached,
so no Tavily request appears in the log.

This is the `FINAL_AUDIT` §4.1 latency/tier problem showing up as a *correctness*
symptom rather than a latency one. Each degraded turn costs 6 logical LLM calls
and ~20k estimated tokens against an 8,000 TPM budget, so the limiter spends the
whole turn admitting over budget:

    Groq token budget still short after 20.0s (want ~4000 tokens, budget 8000/min)
    — admitting anyway to avoid stalling the caller.

Recorded as **1.7**. Two observations worth keeping:

- Cooling down 95 s between turns did **not** fix it, so TPM back-pressure is
  not the whole story — attempts 2 and 3 skipped the tool with no 429 on the
  first call at all.
- Every degraded turn writes its own refusal into `smart_memory_chunks`. The
  next turn retrieves it as context. Worth checking whether the agent is being
  taught, from its own episode history, that this question is answered with a
  refusal rather than a tool call.

### Email send — verified end to end, real message delivered

A real email was sent to `vanshprataps2004@gmail.com` through the full gate:

    Action held for confirmation: {'tool': 'send_email', 'effect': 'EXTERNAL_WRITE',
      'token': '0ATcQOe9', 'content_hash': '0ef25774f09a'}
    → user replies "yes"
    Action confirmed: {'tool': 'send_email', 'content_hash': '0ef25774f09a'}
    Email sent to=vanshprataps2004@gmail.com
    Confirmed action executed: {'status': 'ok', 'idempotency_key': '303b876c50a6'}
    → "Sent to vanshprataps2004@gmail.com."

Draft → preview → durable hold → `yes` → content-bound token → SMTP → pending
cleared. **2.5 is done.** Three things the run surfaced:

- **The router misreads a fully-specified email request.** "Send an email to
  `<addr>`. Subject: … Body: …" classifies as `PROFILE_GENERAL`, routes to
  `profile`, and answers *"I don't have the ability to send emails."* The
  address and the imperative are both right there. Plain phrasing ("send an
  email to `<addr>` saying …") classifies correctly as `ACTION_REQUEST`. The
  `Subject:`/`Body:` keywords appear to pull it into the profile branch — the
  more precise the user is, the worse the routing.
- **`send_email` called twice in one turn, reproduced.** Two actions held with
  an *identical* content hash (`2e844a55cecc`). `confirmation.resolve` then
  refuses to guess between them, and because it has no way to pick, the flow
  **deadlocks** — neither can ever be confirmed. Already listed as a follow-up;
  the deadlock half is new. Identical content hashes could safely collapse.
- **Groq `400 tool_use_failed` on the confirmation turn**, reproduced exactly as
  recorded. The planner crashed and fell back to `profile`, but the turn still
  succeeded: `confirm_action` is reached before the planner's opinion matters.
  The gate held under a provider error, which is the right shape.

### Attendance / ERP scraper — parked

Not touched, not run, not fixed. The source page carries no attendance data at
present, so there is nothing to verify against and any change would be written
blind. `scripts/scrape_erp_attendance.py` (721 lines) and
`app/tools/attendance_tool.py` are left exactly as they are.

**Un-park when:** the ERP page serves attendance rows again. Until then, treat
any attendance failure as expected and do not debug it.

---

---

## Fix pass — 2026-09-21

### 1.7 — JOB_SEARCH now runs its lookup before the first completion

`job_search` was excluded from `prefetch.ZERO_ARGUMENT_LOOKUPS` on the grounds
that its argument comes out of the utterance. That reading was wrong:
`job_agent.tool_job_search` already defaults `query` to the user's own
utterance, so the zero-argument call is not a guess at the question — it *is*
the sentence the model would have passed, enriched with the user's skills by
the tool itself. It qualifies on exactly the property the module already
states.

One line in the prefetchable set. No new abstraction, no new config, no change
to `grounding`, which still owns the question of what a category requires.

**Live result, same query, five consecutive turns:**

| | before | after |
|---|---|---|
| turns delivering listings | **1 / 5** | **5 / 5** |
| logical LLM calls per turn | 6 | **1** |
| estimated tokens per turn | ~20,400 | **~4,400** |
| wall time | 94–115 s | **5–14 s** |

The failure is now unreachable by construction rather than by persuasion: the
model is handed the observation instead of the choice, so it cannot skip the
call, and `grounding` is satisfied before the first completion. Run 5 absorbed
a 429 mid-turn and still answered — with one call in flight instead of six,
the turn fits inside the tier.

The memory-poisoning theory in the previous section is now moot for this path
(no refusals are being written any more), and is left unpursued.

### Router — a dictated email is no longer read as a question about the user

The earlier note blamed `Subject:`/`Body:` phrasing. That was wrong, and the
real cause is worse: **a possessive anywhere in the utterance wins**, including
inside the *content of the email being dictated*. `profile_intent` reads the
whole sentence, so "Subject: **My** project update" made the turn a question
about the user's projects. `Subject: Project update` routed correctly.

The consequence was backwards: the more precisely the user dictated the
message, the more likely it was to be answered *"I don't have the ability to
send emails."*

Fixed with one check ahead of the profile branch, in the position and for the
reason 7b-bis already establishes — both claim it and only one can act on it.
An explicit verb plus a **destination** address is not a question about the
user. Narrow on purpose:

| utterance | before | after |
|---|---|---|
| `send an email to a@b.com. Subject: My project update. …` | PROFILE_PROJECTS | **ACTION_REQUEST** |
| `send an email to a@b.com saying my build is green` | PROFILE_PROJECTS | **ACTION_REQUEST** |
| `forward my resume to recruiter@corp.com` | PROFILE_GENERAL | **ACTION_REQUEST** |
| `email my supervisor` | PROFILE_GENERAL | PROFILE_GENERAL *(who that is is a lookup)* |
| `my email address is a@b.com` | PROFILE_GENERAL | PROFILE_GENERAL *(not a destination)* |
| `send this to him` | AMBIGUOUS_ACTION | AMBIGUOUS_ACTION |

### Duplicate holds collapse instead of deadlocking

Two holds with an **identical** `content_hash` are one request held twice, not
a choice. The disambiguation prompt was a dead end rather than a safeguard: the
reply that resolves it is a token the user has no way to express, so "yes"
re-listed the same two actions forever and neither could be sent *or*
cancelled.

Identical holds now collapse to one and the extras are **cancelled**, not left
to expire, so nothing can be replayed later. Two holds with *different* content
still refuse to guess — that test is untouched and still passes.

This does not remove the underlying defect (the model emitting `send_email`
twice in one turn); it stops that defect from stranding the turn.

### Test suite: the offline guarantee had a hole

`test_a_token_is_consumed_even_when_execution_fails` passed a `failing_send`
callable and expected it to run. It never did — the gateway rebuilds the
callable from the tool *name* (`actions.py`, "Reconstruct, never deserialise"),
so the **real** `send_email` executed. With placeholder SMTP that returned an
error and the test passed for the wrong reason; with real credentials it
succeeded, the test failed, and the run sent live mail (`Email sent to=a@b.com`).

The gateway is right. The test now registers its double through
`register_confirmable`, whose own docstring warns about this exact hazard, so
the resolution path is still fully exercised and SMTP is never reached.

**The class is now closed too.** `tests/conftest.py` installs a session-scoped
autouse guard that replaces `smtplib.SMTP` and `smtplib.SMTP_SSL` for the whole
run. Any test that reaches a mail server fails immediately and is told which
stub to use instead.

It sits one layer *below* every existing double — `_send_sync`, the tool
callable, `register_confirmable` — so it cannot mask a stub that is working. It
only fires on a path that would otherwise have opened a socket, and nothing in
the suite legitimately constructs `smtplib.SMTP`.

`SMTPContactedInTests` inherits from **`BaseException`**, not `Exception`, and
that is load-bearing: `EmailSenderService.send_email` ends in a bare
`except Exception`, which would have caught the guard and turned a test that
reached a mail server into a quiet `{"success": False}`. Contained, but
invisible — which is exactly how the original defect survived a full run.

Verified by reverting the one-line fix to
`test_a_token_is_consumed_even_when_execution_fails` and re-running it:

    SMTPContactedInTests: This test tried to open a real SMTP connection
    (SMTP to 'smtp.gmail.com'). The suite runs offline.

No `Email sent` line — the message was stopped at the socket, where the same
test previously logged `Email sent to=a@b.com`. The fix was then restored.

Two tests assert the guard itself, so it cannot be removed silently: one checks
both constructors are blocked, the other drives a fully configured
`EmailSenderService` with no stub on the send path and asserts it still cannot
deliver.


## Found by the live run — new follow-ups

- **The offline suite is not offline once SMTP is configured.**
  `tests/test_action_gateway.py::test_a_token_is_consumed_even_when_execution_fails`
  passes a `failing_send` callable and expects it to run. It never does: the
  gateway deliberately rebuilds the callable from the tool *name* at
  confirmation time (`actions.py:807`, "Reconstruct, never deserialise"), so
  the **real** `send_email` executes. With placeholder SMTP that returned an
  error and the test passed by accident. With real credentials it succeeds, the
  test fails, and the run sends live mail — observed: `Email sent to=a@b.com`.
  The gateway is right; the test's premise is wrong. Baseline is therefore
  **2469 passed, 1 failed, 27 skipped** on this machine, and the failure is
  environmental, not a regression.
- **Router misclassifies precisely-worded email requests** — see the live-run
  section. `Subject:`/`Body:` phrasing lands in `PROFILE_GENERAL`.
- **Duplicate held actions deadlock the confirmation flow** rather than merely
  prompting — two holds with the same content hash can never be resolved.
- ~~**`scripts/upload_timetable_pdf.py` cannot work as the README documents
  it.**~~ **Fixed.** It is an HTTP client needing a running API and defaults to
  `http://localhost:8000`, while the README told you to run uvicorn on
  **10000**. README step 5 now points at `scripts/upload_new_timetable.py`,
  which does the same job directly against the database, and explains when to
  reach for the HTTP one instead.


---

## Repo structure pass — 2026-09-21

No behaviour changed. `pytest` was run after every batch and stayed green.

**Audit outcome first, because it is the useful part: `app/` has no dead
modules.** Every module under `app/` is reachable from production code. There
was no duplicated logic worth extracting and nothing provably unused to delete
except build caches and one dead frontend helper. The deletion list is
deliberately short because the codebase did not warrant a long one.

### What moved

| Batch | Change |
|---|---|
| 1 | Deleted 21 `__pycache__` directories, `.pytest_cache`, `frontend/tsconfig.tsbuildinfo`. All were already gitignored — on-disk only. Stale `cpython-314` bytecode sat beside `cpython-312`. |
| 2 | Moved all 69 test files from a flat `tests/` into 13 packages mirroring `app/`. |
| 3 | Split `ChatShell.tsx` 1229 → 999 lines: `MessageList.tsx` (148), `MemoryPanel.tsx` (109), `lib/conversation.ts` (53). |
| 4 | README setup, layout and test counts corrected. |

### Two things worth knowing about batch 2

- `tests/voice/` has no `app/` counterpart. The voice path spans
  `livekit_worker.py`, `agents/` and `services/`; splitting its four test files
  across three packages would hide it.
- Two hidden couplings surfaced only once the files moved, and both are now
  fixed rather than worked around:
  - `test_job_matching` and `test_voice_tool_escalation` import fixtures from
    `test_candidate_profile` — a cross-test dependency invisible while
    everything shared one directory.
  - `test_tool_contract` located `app/agents/` by counting parent directories
    from `__file__`. It now anchors on the repo marker, so moving the file
    again cannot silently point the scan at a directory that does not exist.

### Batch 3 — what was deliberately *not* split

`ChatShell.tsx` came down by extracting presentation and pure helpers. The
LiveKit block (`connectLiveKit` and the turn lifecycle, ~210 lines) was left
where it is on purpose. Pulling it into a `useLiveKitVoice` hook means passing
roughly fifteen callbacks back in — `addMessage`, `beginTurn`, `endTurn`, six
setters and the caption refs — which is precisely the wrapper layer this pass
was told not to build, and it would put a behaviour change on the barge-in path
to buy a smaller file. Recorded here so the decision is visible rather than
looking like an oversight.

`workflow.py` (1657) and `query_intent.py` (1626) were left unsplit for the
same reason: both are cohesive ordered state machines whose correctness depends
on the order of their checks, and that order is the design.

One dead frontend helper was found and removed: `clearStoredConversationId`
was defined and never called.

### Still oversized, deliberately

| File | Lines | Why it stays |
|---|---|---|
| `app/memory/long_term_memory_qdrant.py` | 1685 | One store, one collection layout |
| `app/agents/workflow.py` | 1657 | Ordered state machine |
| `app/agents/query_intent.py` | 1626 | Ordered classifier; order is the design |
| `app/routes/agent_routes.py` | 1449 | One surface, one auth story |
| `frontend/components/ChatShell.tsx` | 999 | See above |

## Roadmap (still open, not in this pass)

- **MCP** — mechanism built and tested; no server configured, `SERVERS` empty.
- **Eval suite** — 12 scenarios; grow to ~30 (multi-step, memory recall, matching, voice routing).
- **PDF timetable parser** — one class per line only; grid/scanned PDFs refused.
- **Attendance / ERP scraper** — **parked.** Source page serves no attendance
  data; nothing to verify against. Not fixed, not tested, left as is.
- **Process-local state** — voice worker registry (M15) blocks a second instance.
- **Unwired memory endpoints** — exposed but not reached by any surface.
- **Frontend confirmation UI** — `ChatShell.tsx` (D10); confirmation is chat-only.
- **`useLiveKitVoice` hook** — `ChatShell.tsx` is still 999 lines, and the
  LiveKit block is the remaining bulk. Worth doing when the callback surface
  can be narrowed first, not before.
- **`send_email` emitted twice in one turn** — duplicate holds now collapse
  instead of deadlocking, but the model still makes the duplicate call.
- **Voice testing** — never verified against real LiveKit/Deepgram/Cartesia in a browser.

---

## New follow-ups found during this pass

Each was observed while doing the work above, and none is in scope for it.

- **`qdrant_service` logs empty errors.** Observed live at startup:
  `Failed to upsert points to 'memory_records': ` — nothing after the colon.
  Same defect as `main.py:79` (a `%s` on an exception that stringifies to
  `""`); the fix is the same `%r` + `exc_info=True`. One line, but in a file
  these tasks do not touch, so it is listed rather than changed.
- **The send dedupe fingerprint is not scoped by user.** `_fingerprint` hashes
  recipient + subject + body only, so two users sending byte-identical mail
  inside 300 s would have the second suppressed. Harmless in a single-owner
  build; wrong the moment it is not. The new rate cap *is* per user.
- **The send cap and the dedupe are both process-local.** Lost on restart, and
  not shared across replicas. The durable half of the guarantee already exists
  (the gateway's Postgres idempotency index); making these durable needs a
  table and a migration, which is more than "keep it small" allows.
- **`/tools/job-search` now returns HTTP 200 with `success: false`** for a
  provider failure, where it previously raised a 500. Better for the agent
  path (the reason reaches the model); worth a look if anything external
  depends on the status code.
- **`grounding`'s refusals invite retries that cannot succeed.** Fixed for the
  Tavily case in `job_agent` (1.6), but the shape is general: any required
  tool that fails for a *configuration* reason gets "try again in a second".
  A `ToolResult` that could mark an error as permanently unfixable would let
  `grounding` say so once, everywhere, instead of per agent.
- **`send_email` can be called twice in one turn.** Observed live. Safe — both
  are held and `confirmation.resolve` refuses to guess — but it costs the user
  a disambiguation prompt for what they asked once. The agent could refuse a
  second `send_email` within a turn, the way it already caps iterations.
- **Groq `400 tool_use_failed`.** Observed twice on the email turn: the model
  emitted a tool call while `tool_choice` was none. Treated as a permanent
  error and not retried, which is right, but the turn is then lost. Worth
  checking whether the loop sets `tool_choice` correctly on follow-up passes.
