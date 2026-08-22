# My_Agent — Production Readiness Audit

**Original audit:** 2026-08-06 · **Last updated:** 2026-08-07 · **Scope:** `app/`, `frontend/`, `scripts/`

**Method:** manual trace of the FastAPI/LangGraph backend, hybrid Postgres+Qdrant+mem0 memory system, and LiveKit/Deepgram/Cartesia voice pipeline, plus four parallel deep-dives into the memory/DB layer, services/tools layer, voice pipeline, and frontend/API security surface.

**Original audit totals: 5 Critical · 13 High · 20 Medium · 16 Low — 54 findings**
**Voice pipeline follow-up: 23 additional defects (see [Voice pipeline remediation](#voice))**

> This document started as a read-only audit. It is now also the remediation
> record: the finding sections below are preserved as the historical description
> of each defect, and each one carries a **Status** line stating where it stands
> today. Where the current state contradicts a finding's original text, the
> Status line is authoritative.

---

## Current state at a glance

| | Count |
|---|---|
| Original findings fixed | **49 of 54** |
| Original findings still open | 5 (1 partial, 4 deferred by decision) |
| Voice pipeline defects found and fixed since | **23** |
| Automated tests | **179 passing** (`pytest`) |

**Open items, in priority order:** [C4](#c4) (email quota/confirmation — partial),
[M15](#m15) (single-process worker registry — blocks horizontal scaling),
[H5](#h5), [M16](#m16), [L3](#l3), [L15](#l15).

---

## Remediation status

### Original 54 findings

| Finding | Status |
|---|---|
| C1 — No authentication | ✅ **Fixed** — JWT access/refresh, HTTP-only cookies, CSRF, scope-checked routes (`app/auth/`, `app/routes/auth_routes.py`, `frontend/lib/auth.ts`) |
| C2 — Recruiter/User share one identity | ✅ **Fixed** — separate `owner` and `guest` roles; `Scope` capabilities enforced at the route, the identity resolver, *and* the agent tool registry (`app/auth/models.py`, `app/agents/base_agent.py:130-139`) |
| C3 — SSRF via ERP scraper | ✅ Fixed — SSRF guard (`app/services/url_guard.py`) |
| C4 — Open SMTP relay | 🟡 **Partial** — address validation + content-hash dedupe shipped; per-user quota and human confirmation still missing |
| C5 — Voice worker disposes shared DB pool | ✅ Fixed |
| H1–H4, H6–H13 | ✅ Fixed |
| H5 — Reflect loop never retries on low confidence | ⏸ **Deferred** — retry policy is a workflow decision |
| M1–M14, M17–M20 | ✅ Fixed |
| M15 — `active_workers` single-process | ⏸ **Deferred** — needs Redis/queue. Hardened in the meantime: the worker deregisters synchronously at the start of teardown, closing a race where a reconnecting client joined a room whose worker was already leaving |
| M16 — One failed step aborts whole plan | ⏸ **Deferred** — workflow behaviour decision |
| L1, L2, L4–L14, L16 | ✅ Fixed |
| L3 — Markdown rendered as literal text | ⏸ **Deferred** — needs a new frontend dependency |
| L15 — `conversation_history` unused on chat path | ⏸ **Deferred** — API contract decision |

**Migration required for existing databases:** run
`python scripts/migrate_add_constraints.py` (supports `--dry-run`). New deployments
pick the changes up automatically from the models.

---

## Fix first

The original top six are all closed. What remains, in order:

1. **[C4](#c4)** — Add a per-user send quota and an explicit confirmation step before an email leaves the system. Validation and dedupe are in; the spend/abuse ceiling is not.
2. **[M15](#m15)** — Move worker-assignment state to a shared store before running more than one instance. Today a second process would route a caller to an instance with no voice worker.
3. **Echo cancellation** ([V2](#v2)) — the assistant's own voice reaching the microphone is the one remaining path to a self-talk loop, and it currently depends entirely on the browser's AEC.
4. **Live voice verification** — every voice fix below is covered by tests with faked provider boundaries. None has been confirmed against real LiveKit/Deepgram/Cartesia traffic in a browser.
5. **[H5](#h5)** / **[M16](#m16)** — decide the workflow retry and partial-plan policies.
6. **[L3](#l3)** — render markdown, so replies stop showing literal `**asterisks**`.

---

## Table of contents

- [Voice pipeline remediation (23)](#voice)
- [Critical (5)](#critical)
- [High (13)](#high)
- [Medium (20)](#medium)
- [Low (16)](#low)

---

<a id="voice"></a>
## Voice pipeline remediation

*Found and fixed after the original audit, while making a spoken conversation
actually work end to end. The original audit caught individual voice defects
(H11–H13, M13, M14, L8, L9); this pass traced the whole path — browser mic →
LiveKit → Deepgram → routing → LangGraph/Groq → chunker → Cartesia → LiveKit →
playback — and found the reasons a conversation failed as a whole.*

V-numbers follow discovery order; the sections below are grouped by theme, so they
do not appear in numeric order. V21–V23 were found in a second pass, after the
first round of fixes was in place — and V23 was *caused* by one of them.

**Verification caveat, stated plainly:** these fixes are covered by 37 tests in
`tests/test_voice_pipeline.py` exercising the real chunker, streamer, bridge, and
turn-liveness code with faked network boundaries, plus `tsc --noEmit` and API
signature checks against the installed `livekit 1.1.13` / `deepgram-sdk 4.1.0`.
**No live provider call has been made and no browser has been driven.** The
latency and sync figures below come from simulations with realistic provider
delays, not from measured playback.

### Why the assistant appeared not to reply

<a id="v1"></a>
**V1 — Every reply cancelled itself via interim transcripts.** `_on_transcript`
called the barge-in callback on *every* interim result as a "fallback VAD", and
that callback cancelled the in-flight turn. A breath, a keystroke, or the
assistant's own voice through the speakers killed the reply within ~300 ms,
before a single word was audible. This was the primary cause.
→ Barge-in moved out of the bridge entirely. It is now transcript-confirmed and
gated on the assistant *actually speaking*, requiring `voice_bargein_min_words`
real words.

<a id="v2"></a>
**V2 — Raw VAD also cancelled replies.** `SpeechStarted` fires on any sound, not
on speech. → Demoted to a logging hint. **Note:** the microphone hearing the
assistant's own voice is still guarded only by the browser's echo cancellation
(now requested explicitly via `audioCaptureDefaults`). Headphones remain the
reliable configuration.

<a id="v3"></a>
**V3 — `TextChunker` yielded from a `finally` block.** Closing the generator early
— exactly what a barge-in does — raised
`RuntimeError: async generator ignored GeneratorExit`, turning a clean
interruption into a crashed turn. → Rewritten so every `yield` is in the normal
body; `finally` only cancels the in-flight token fetch.

**V4 — UI deadlock after the first tool-route turn.** `utterance_end` set
`isSending`; only `token_batch` cleared it, and the tool route never emits
`token_batch`. With the mic button disabled on `isSending`, one tool turn killed
voice permanently. → Every terminal path funnels through one `endTurn()`, backed
by a client watchdog.

<a id="v5"></a>
**V5 — Zombie worker after any LiveKit disconnect.** `room.on("disconnected")`
only logged; the worker sat on `asyncio.Event().wait()` forever holding a dead
room while `active_workers` still pointed at a live task. Since
`livekit_routes` skips starting a worker when the existing task isn't done, a
reconnecting user joined a room **nobody was listening to** — permanent silence,
unrecoverable without a server restart. → Disconnect sets a shutdown event, an
idle watchdog exits after 120 s empty, and teardown deregisters synchronously.

**V6 — Audio track publish race.** `publish_track()` was fire-and-forget; frames
handed to an unpublished `AudioSource` are discarded, eating the start of the
first reply. → A `track_ready` event is awaited before speaking.

<a id="v7"></a>
**V7 — Concurrent turns.** A new utterance overwrote `state.turn_task` without
cancelling the predecessor: two TTS streams into one audio source, and the first
task reference lost so barge-in could never reach it. → One `turn_lock` per
participant; the predecessor is cancelled **and awaited**, then the playout queue
cleared.

**V8 — Cartesia failures swallowed.** Every exception was caught and nothing
yielded, so a bad key was indistinguishable from having nothing to say. → Raises
`TTSUnavailable`; the user is told the reply is on screen but the voice broke.

<a id="v9"></a>
**V9 — No timeout anywhere on a voice turn.** One hung dependency meant permanent
silence and a permanently stuck UI. → See [V23](#v23) for the design this landed on.

**V10 — Participants already in the room were ignored.** `participant_connected`
does not fire for them, and the browser normally joins *first* since its token
request is what starts the worker. → Explicit post-connect sweep of
`room.remote_participants`.

**V11 — Frontend playback.** Autoplay blocking was never detected; `track.attach()`
elements were appended to `document.body`, never removed on unmount, and leaked a
duplicate per reconnect. → `room.startAudio()` + `AudioPlaybackStatusChanged` with
a clickable banner; elements tracked and removed.

**V12 — `run_workflow` result read a `success` key that does not exist.** Failures
always reported success. → Derived from `error is None`.

**V13 — Mic disabled while speaking**, removing the only UI affordance for
interruption. → The mic is a session toggle and stays live across turns.

<a id="v21"></a>
**V21 — Clarifying questions were displayed but never spoken.**
`run_streaming_workflow` short-circuits for a clarification: `metadata`, then
`complete`, and returns — no `token` events. The chunker therefore received an
empty stream, Cartesia was never called, and the question appeared on screen in
silence. The same hole affected the "agent not found" and LLM-error replies.
→ `speech_token_stream()` yields the completed reply when a turn finishes without
having streamed anything.

### Text/audio synchronisation

<a id="v22"></a>
**V22 — On-screen text raced ahead of the voice.** Text was paced by the **LLM
clock**: `token_batch` published the accumulated reply as Groq produced it. Groq
emits a 50-word answer in ~2 s; speaking it takes ~20 s. `final_response` then
committed the finished bubble the instant generation ended — up to a minute
early on a long reply.
→ Text is now driven by the **audio clock**. `capture_frame` self-paces to
playback, so `AudioSource.queued_duration` says exactly how far ahead of the
listener's ear we are. `TTSStreamer` buffers each chunk to learn its exact
duration and emits
`SpokenChunk(text, starts_at = now + queued_duration, duration)`; a caption pump
reveals one word at a time against that schedule and re-anchors on every chunk,
so drift cannot accumulate. The permanent message bubble is committed at
`turn_end`, after `wait_for_playout()`. Simulated worst-case drift across a
54-word / 20.5 s reply: **177 ms** (the intended 150 ms offset plus 27 ms jitter).

Two consequences worth recording:
- `queue_size_ms` raised 1000 → 2000 ms, for margin against an audible seam
  between per-chunk Cartesia requests. Barge-in clears the queue explicitly, so
  the extra depth costs no interrupt latency.
- `endTurn` originally called `addMessage` from inside a `setState` updater.
  React may invoke updaters twice, which would have duplicated every message; the
  caption text is now mirrored in a ref and read synchronously.

### Turn liveness

<a id="v23"></a>
**V23 — A 25 s wall-clock deadline killed long replies.** Introduced by the
[V9](#v9) fix and wrong: a turn's elapsed time is dominated by TTS, and TTS is a
*real-time medium* — because `capture_frame` paces to playback, a 60-second
answer occupies the turn for 60 seconds **by construction**. A fixed deadline
cannot tell "long" from "stuck", so every reply longer than ~25 s of speech was
cancelled mid-sentence (`Turn for <user> exceeded 25s and was abandoned`).
Measured: 20.5 s of audio → 21.3 s wall clock.
→ A turn is now bounded by **progress**, not elapsed time. `TurnProgress.touch()`
is called by every token, every audio frame (~50/s), and every spoken word;
`watch_for_stall()` cancels only when that gap exceeds
`voice_turn_stall_seconds` (20 s), with `voice_turn_max_seconds` (600 s) as a
last-resort ceiling. The watchdog cancels the turn task itself — not a child, so
nothing can outlive the turn and keep talking — sets `abort_reason` *before*
cancelling so the turn can distinguish its own deadline from a barge-in, and the
turn calls `task.uncancel()` to absorb that cancellation. Without the absorb, the
task would be marked cancelled and `turn_end` would never reach the browser,
reproducing V4's stuck UI.

### Latency

The pipeline was taking roughly 3–6 s to first audio. Each of these was on the
critical path of every single turn:

<a id="v17"></a>

| | Defect | Fix |
|---|---|---|
| **V14** | Router made an unconditional LLM call — against `llama3-8b-8192`, **decommissioned by Groq**. Every turn paid for a request that could only fail, then defaulted to `conversational` anyway. | Heuristic-first regex (~0 ms for common turns); LLM only when ambiguous, on `llama-3.1-8b-instant`, capped at 1.2 s |
| **V15** | `memory_node` and `planner_node` ran serially despite being independent | Concurrent via `parallel_init_node` (~1300 ms → ~800 ms) |
| **V16** | `utterance_end_ms=1000` with no `endpointing` — the API floors that value, so every turn carried a full second of dead air | `endpointing=500`; turns start on `speech_final`, `UtteranceEnd` as backstop |
| **V17** | Chunker waited for a full sentence or 15 words before the first TTS request | First chunk released at ~4 words |
| **V18** | `aiter_bytes(chunk_size=8192)` held back ~170 ms of audio at 24 kHz | Unbuffered `aiter_bytes()` |
| **V19** | A manual `AudioResampler` ran per frame although `rtc.AudioStream` resamples internally | Stage removed; `AudioStream(track, sample_rate=16000, num_channels=1)` |
| **V20** | Duplicate keepalive — the SDK already runs one under `options={"keepalive":"true"}` | Bridge's own loop removed |

Simulated token-stream → first audio frame after these changes: **716 ms**.

### Files carrying the voice work

`app/livekit_worker.py` (rewritten) · `app/services/deepgram_bridge.py` (rewritten) ·
`app/services/text_chunker.py` (rewritten) · `app/services/tts_streamer.py` (rewritten) ·
`app/services/voice_service.py` · `app/agents/hybrid_router.py` (rewritten) ·
`app/agents/streaming_workflow.py` · `app/routes/livekit_routes.py` · `app/config.py` ·
`frontend/components/ChatShell.tsx` · `.env.example` · `tests/test_voice_pipeline.py` (new)

### Data-channel contract

The browser and worker now agree on this event set. Exactly one terminal event
(`turn_end` or `interrupted`) is published per turn, from a `finally` block, so
the UI can never be left disabled:

`interim_transcript` · `utterance_end` · `caption` (one word, audio-timed) ·
`final_response` (authoritative text + metadata, held not rendered) · `error` ·
**`turn_end`** · **`interrupted`**

---

## Critical

*Exploitable today, or breaks the app for everyone.*

### C1 — No authentication anywhere — `user_id` is a self-asserted client string
`Security` `API design`

> **Status: ✅ Fixed.** Authentication is implemented in `app/auth/` (JWT access +
> refresh with separate secrets, bcrypt owner password, HTTP-only cookies, CSRF
> protection) and enforced by `require_scope(...)` dependencies. `user_id` is
> derived from the verified principal and never read from client input; the voice
> room name is derived the same way (`room_name_for(principal)`), and a request
> naming someone else's room is rejected outright. `frontend/lib/auth.ts` replaces
> the localStorage identity with a real session, with a `/login` page for the owner.
> Covered by `tests/test_auth_endpoints.py`, `tests/test_auth_tokens.py`, and
> `scripts/verify_auth_coverage.py`.

**Why it's a problem** *(as originally found)*
Every endpoint that touches personal data — chat, profile facts, resume/skills upload, attendance, email drafts, LiveKit voice tokens — trusts a `user_id` supplied directly in the request body or URL, with no session, cookie, or API key behind it. `_assert_safe_user_id` checks character set and length, never ownership. Any caller can read, overwrite, or delete another user's resume, skills, attendance history, saved emails, and preferences simply by sending a different string. The frontend makes this trivial to demonstrate: it hardcodes a single stable identity and pulls it straight out of `localStorage`, which any visitor can edit.

**Files**
- `app/routes/agent_routes.py:29-80` (`_assert_safe_user_id` — format only, not identity), every handler using `request.user_id`
- `app/routes/livekit_routes.py:22-71` (`/token` mints a voice session for any identity)
- `frontend/components/ChatShell.tsx:24-55` (`STABLE_USER_ID` hardcoded, localStorage-editable)

**Fix**
Add real authentication (session cookie or JWT) issued server-side, and derive `user_id` from the verified principal — never from client input. Gate every read/write/delete endpoint behind an auth dependency before this is exposed publicly.

---

### C2 — "Recruiter Mode" and "User Mode" are the same account with the same private tool access
`AI/LLM workflow` `Product design`

> **Status: ✅ Fixed.** The two personas are now distinct principals: an `owner`
> role and an anonymous read-only `guest` role, each carrying an explicit `Scope`
> set (`app/auth/models.py`). Capabilities are checked at three layers — the route,
> the identity resolver, and the agent tool registry
> (`app/agents/base_agent.py:130-139`), so a guest cannot reach email-send,
> profile-mutation, or attendance tools even by talking the model into trying.
> Scopes travel into the voice worker as well, so a spoken turn enforces exactly
> the same permissions as an HTTP request. The frontend hides owner-only controls
> via `hasScope`, but that is cosmetic — the server is the boundary.

**Why it's a problem** *(as originally found)*
The product ships a public-facing "Recruiter" chat meant for strangers to ask about the owner's profile. `getOrCreateUserId()` ignores the `mode` prop entirely and always returns the same stable identity, so a recruiter visitor shares the owner's exact memory store and full agentic toolset. None of the four specialist agents read `state["output_mode"]` — only `response_agent.py` does, and only to decide whether to speak the reply aloud. Nothing stops a "recruiter" from asking the assistant to send an email as the owner, edit or forget profile facts, or read private attendance and saved drafts — `output_mode` authorizes nothing, it only controls whether TTS fires.

**Files**
- `frontend/components/ChatShell.tsx:24-55` (identity ignores `mode`)
- `app/agents/job_agent.py`, `email_agent.py`, `academic_agent.py`, `profile_agent.py` (no `output_mode` check)
- `app/agents/response_agent.py` (the only place `output_mode` is read — for speech formatting only)

**Fix**
Give the recruiter persona its own restricted identity and a read-only tool allowlist (profile summary, skills, projects). Enforce the restriction at the agent-routing layer — reject email-send, profile-mutation, and attendance tools when `output_mode == "recruiter"`, not just in the frontend UI.

---

### C5 — Ending one voice call disposes the shared Postgres connection pool for the whole process
`Async issue` `Database` `Voice pipeline`

> **Status: ✅ Fixed.** The `memory_manager.cleanup()` call is gone from the worker
> teardown path, with a comment at the site recording why it must not come back.
> Engine and pool lifecycle belongs solely to the `lifespan` handler in
> `app/main.py`. The worker now *does* exit on room disconnect or after 120 s idle
> (see [V5](#v5)), which makes that boundary load-bearing rather than academic.

**Why it's a problem** *(as originally found)*
`livekit_worker.main()`'s shutdown path calls `memory_manager.cleanup()` when a single LiveKit room's worker exits — e.g. one user hanging up. That call chain ends at `engine.dispose()` on `short_term_memory`, which is a module-level singleton shared by the *entire FastAPI process*: every REST request and every other concurrent voice room. Disposing it forcibly drops the whole pooled-connection set out from under anyone mid-query, then pays a reconnect-storm penalty right after. This fires every time any one voice session ends — not at application shutdown, where it belongs.

**Files**
- `app/livekit_worker.py` (worker shutdown `finally` block)
- `app/memory/memory_manager.py:524-526` (`cleanup()`)
- `app/memory/short_term_memory.py:40-42` (`close()` → `engine.dispose()`)

**Fix**
Remove the `memory_manager.cleanup()` call from the per-room worker shutdown path entirely. Engine and pool lifecycle belongs solely to `app/main.py`'s `lifespan` shutdown handler, which already does this correctly.

---

### C3 — SSRF and blind data exfiltration via the attendance ERP scraper
`Security` `SSRF`

> **Status: ✅ Fixed.** `app/services/url_guard.py` resolves and validates the target
> before any navigation: private, loopback, and link-local ranges (including cloud
> metadata endpoints) are rejected unless `allow_private_network_scraping` is
> explicitly enabled. The endpoint also now requires an authenticated principal with
> the attendance-write scope. Covered by `tests/test_url_guard.py`.

**Why it's a problem** *(as originally found)*
`POST /api/v1/agents/tools/attendance/scrape` accepts a caller-supplied `erp_url` and a free-form CSS `selectors` dict with no allowlist and no authentication (see C1). The server's Playwright instance navigates to that URL and extracts whatever the attacker-chosen selectors point at, returning the scraped text back through the "attendance records" it stores. That means any caller can point the server at an internal service or arbitrary external site, and use the selectors to pull page content back out — while also submitting attacker-chosen `username`/`password` into attacker-chosen form fields on an attacker-chosen page.

**Files**
- `app/tools/attendance_tool.py` (scrape loop, browser lifecycle)
- `app/routes/agent_routes.py` — `AttendanceScrapeRequest` / `scrape_attendance`

**Fix**
Require auth; store a per-user allowlist of ERP hostnames server-side instead of accepting `erp_url` from the client; reject private/link-local IP ranges; replace the free-form selector dict with a small vetted preset per known ERP layout.

---

<a id="c4"></a>
### C4 — Unauthenticated, unvalidated SMTP relay via the email agent
`Security`

> **Status: 🟡 Partially fixed — the highest-priority remaining item.**
> Shipped: the endpoint requires authentication with an email scope (so it is no
> longer reachable anonymously, and a guest cannot reach it at all); `to_email`,
> `cc`, and `bcc` are format-validated (`_validate_address`); and an identical
> resend inside a dedupe window is suppressed by content fingerprint, which is what
> closed [H4](#h4).
> **Still missing: a per-user send quota and an explicit human confirmation step.**
> An authenticated owner session can still drive an unbounded number of distinct
> emails to arbitrary addresses. Drafting should stay automatic; sending should not.
> Covered so far by `tests/test_email_sender_safety.py`.

**Why it's a problem** *(as originally found)*
`email_sender_service.send_email` performs no validation on `to_email` and is reachable from `email_agent`'s `send_email` tool, itself reachable from the fully unauthenticated `/api/v1/agents/query` endpoint. Any caller can drive the agent — through ordinary conversation or a crafted prompt — into sending arbitrary LLM-generated content from the operator's real Gmail account to any address, with no rate limit or confirmation step anywhere in the path. That's an open spam/phishing relay sitting behind a chat endpoint, and it risks the sending account being suspended by Google.

**Files**
- `app/services/email_sender_service.py` (no recipient validation)
- `app/agents/email_agent.py:138-179` (`tool_send_email` forwards LLM-chosen fields verbatim)

**Fix**
Validate `to_email` format before sending; require auth plus a per-user send quota; add an explicit human-confirmation step before an email actually leaves the system (drafting can stay fully automatic).

---

## High

*Real bugs, races, or open cost/DoS exposure.*

### H1 — No rate limiting anywhere in the API
`Security` `Scalability`

**Why it's a problem**
`/agents/query` (a multi-LLM-call LangGraph run), `/stream`, `/memory/upload-pdf`, `/tools/job-search`, `/tools/attendance/scrape` (a headless browser per call), and `/voice/token` (spawns a background worker and burns Groq/Deepgram/Cartesia credits) are all completely unthrottled. Combined with C1's missing auth, this is an open door to cost-exhaustion attacks against Groq/Cohere/Tavily/Qdrant billing, and to straightforward resource exhaustion on the server itself.

**Files**
- `app/main.py` (no rate-limit middleware)
- all of `app/routes/agent_routes.py` and `app/routes/livekit_routes.py`

**Fix**
Add a rate-limiting layer (e.g. `slowapi` or reverse-proxy limits) keyed by IP and/or authenticated user; minimum coverage: `/agents/query`, `/stream`, `/memory/upload-pdf`, `/tools/attendance/scrape`, `/voice/token`.

---

### H2 — Unbounded PDF/text upload size — memory-exhaustion DoS
`Security` `Missing error handling`

**Why it's a problem**
`upload_pdf_document`, `upload_timetable_pdf`, and `upload_text_document` call `await file.read()` / accept a raw form field with no size cap, and Starlette enforces no body-size limit by default. A single large file, or repeated smaller ones, can exhaust worker memory before any validation runs.

**Files**
- `app/routes/agent_routes.py:380-530`

**Fix**
Enforce a max upload size (reject on `Content-Length` or stream-read with a cutoff, e.g. 10MB) before reading the full file into memory; cap `text_content` with a Pydantic `max_length`.

---

### H3 — `temperature or self.temperature` silently discards `temperature=0.0`
`Bug` `AI/LLM workflow`

**Why it's a problem**
`0.0` is falsy in Python, so `chat_completion`/`stream_chat_completion` building their request as `temperature=temperature or self.temperature` means any caller explicitly requesting deterministic output with `temperature=0.0` silently gets the default `0.7` instead. The LiveKit voice router calls exactly this with `temperature=0.0` to make "conversational vs. tool_required" classification deterministic — it never actually gets determinism, making voice intent routing noisier and less reproducible than the code believes it is.

**Files**
- `app/services/groq_service.py:58,105`
- caller `app/agents/hybrid_router.py:43`

**Fix**
Use `temperature if temperature is not None else self.temperature` (same pattern for `max_tokens`).

---

<a id="h4"></a>
### H4 — Automatic retry can re-send an email that already sent successfully
`LangGraph design` `AI/LLM workflow`

**Why it's a problem**
When a specialist's overall status is `"failed"` (only reachable via an uncaught exception — see H5), `reflect_node` re-runs the *entire* specialist from a fresh reasoning loop, with no memory of which tool calls already succeeded in the failed attempt. If `send_email` already succeeded but a later step — say, the `mark_email_sent` database write — throws, the retry can have the LLM call `send_email` a second time: a duplicate email to a real recipient. The same non-idempotency risk applies, at lower stakes, to job bookmarking and plan creation.

**Files**
- `app/agents/workflow.py` (`reflect_node` retry path)
- `app/agents/email_agent.py` (`tool_send_email`)
- `app/agents/base_agent.py` (`execute_reasoning_loop`)

**Fix**
Make side-effecting tools idempotent (dedupe by content hash or `draft_id`), and/or carry forward which tool calls already executed across a retry so they aren't blindly repeated.

---

<a id="h5"></a>
### H5 — The reflect loop's "learn from failure" retry is nearly dead code
`Logic error` `LangGraph design`

> **Status: ⏸ Still open.** `reflect_node` continues to gate its retry on
> `status == "failed"` (`app/agents/workflow.py:223`), which remains reachable only
> via an uncaught exception, and `_compute_confidence` still feeds nothing into that
> decision. Confidence *is* now used earlier — the planner's 0.6 bar routes to a
> clarification node before a specialist runs — but a confidently-wrong specialist
> answer still goes straight to the user. Deferred because "retry when confidence
> is low" trades latency and cost for accuracy, which is a product call.

**Why it's a problem** *(as originally found)*
`execute_reasoning_loop` always produces a non-empty `final_answer` — it falls back to an apology string rather than returning nothing — so every specialist's `status` is effectively always `"success"`. The only way to reach `"failed"` is an outright uncaught exception. A confidently-wrong or low-confidence answer (confidence 0.5, say) still reports `"success"` and goes straight to the user; the confidence score `_compute_confidence` works hard to calculate is never fed back into the retry decision. The planner's own 0.6 confidence bar for routing is never re-applied once a specialist has run.

**Files**
- `app/agents/workflow.py` (`reflect_node` — the "Fix 2" block)
- `app/agents/base_agent.py` (`_compute_confidence`)

**Fix**
Feed the computed confidence into the reflect decision — retry (or ask for clarification) when confidence falls below a threshold, not only on exception.

---

### H6 — The planner never sees conversation history
`Logic error` `AI/LLM workflow`

**Why it's a problem**
Every specialist's reasoning loop includes the last several turns of conversation history — but `planner_agent.execute()` sends only `[system_prompt, user_input]` to the LLM. The one component responsible for deciding *which* agent handles a turn is the one component that can't see what was just discussed. Follow-ups like "email him about that" or "when's my next one" get classified in a vacuum, which will misroute or trigger unnecessary clarification requests on ordinary multi-turn conversations.

**Files**
- `app/agents/planner_agent.py` (`execute()`)

**Fix**
Inject the last few turns (or at minimum the `memory_prompt` summary) into the planner's message list before routing.

---

### H7 — Job bookmarks can be duplicated by a check-then-insert race
`Race condition` `Database`

**Why it's a problem**
`save_job_bookmark` checks `is_job_bookmarked()` in one session, then inserts unconditionally in another. Two concurrent calls for the same URL — an agent retry, a double-tap — can both pass the check before either commits, producing duplicate rows. Nothing at the schema level prevents it either: `JobBookmark` has no unique constraint on `(user_id, url)`.

**Files**
- `app/memory/short_term_memory.py:322-359`
- `app/memory/models.py:57-70`

**Fix**
Add `UniqueConstraint("user_id", "url")` and use an `ON CONFLICT DO NOTHING` upsert instead of check-then-insert.

---

### H8 — "Latest resume" selection is non-deterministic — no timestamp anywhere
`Bug` `Database`

**Why it's a problem**
`retrieve_resume()` groups scrolled Qdrant chunks by `parent_id` and picks the last key in dict order as "latest." Qdrant scroll order reflects internal storage order, not upload recency, and no timestamp field is ever written into the resume payload. Normally the old version's chunks are deleted right after the new upsert, so only one group survives — but if that delete step throws partway (a network blip is enough), two versions can coexist, and the code then has roughly even odds of silently serving the stale resume/skills/projects indefinitely, with no error surfaced anywhere.

**Files**
- `app/memory/long_term_memory_qdrant.py:495-558` (upsert/delete sequence)
- `app/memory/long_term_memory_qdrant.py:759-761` (arbitrary "latest" pick)

**Fix**
Write an explicit `uploaded_at`/version field into the payload at write time, and select the parent group with the max value instead of relying on scroll order.

---

### H9 — Unbounded Qdrant scroll means "delete all my data" doesn't always delete all your data
`Bug` `Compliance`

**Why it's a problem**
`scroll_collection` takes a single `limit` and never paginates via Qdrant's scroll cursor — the offset the client returns is discarded. `reset_user_memories()` and `delete_memory()` call it with `limit=1000`; a user with more than 1000 stored memory points will have some of them *survive an explicit erasure request*. That's a data-deletion correctness problem, not just a scaling one. The same pattern silently truncates large multi-version resumes on retrieval.

**Files**
- `app/services/qdrant_service.py:333-386`
- `app/memory/smart_memory.py` (reset/delete, `limit=1000`)
- `app/memory/long_term_memory_qdrant.py:745` (`limit=100`)

**Fix**
Implement real cursor-based pagination in `scroll_collection`, or use the already-implemented `delete_by_filter` for erasure instead of scroll-then-delete-by-id.

---

### H10 — Re-scraping attendance duplicates rows and skews the risk math the whole academic agent relies on
`Bug` `AI/LLM workflow`

**Why it's a problem**
`scrape_and_store` inserts a fresh row per scraped record on every run, with no uniqueness check on `(user_id, date, subject)`. Re-running a scrape — including the scheduled `scripts/refresh_attendance.bat` — duplicates every previously-seen day, inflating the "total" denominator and skewing exactly the <75%-attendance risk flags the academic agent's advice is built on. The core feature degrades in accuracy every time it's refreshed.

**Files**
- `app/tools/attendance_tool.py:32-52`

**Fix**
Upsert on `(user_id, date, subject)` instead of a blind insert per row.

---

### H11 — Concurrent Deepgram reconnects can leak connections and duplicate transcripts
`Race condition` `Voice pipeline`

**Why it's a problem**
`ensure_connected()` has no lock. A keepalive failure and an error event arriving close together — a common pattern during a network blip — can both see the bridge disconnected and both start independent reconnects. The second `start()` silently overwrites the first connection and its keepalive task, which is never cancelled — a task leak — and briefly two live Deepgram sockets can stream the same audio, double-firing transcript callbacks into the workflow.

**Files**
- `app/services/deepgram_bridge.py` (`ensure_connected`, `_on_error`, `_keepalive_loop`)

**Fix**
Guard `ensure_connected()` with an `asyncio.Lock`; concurrent callers should await the in-flight reconnect rather than starting their own.

---

### H12 — Barge-in cancels synthesis but not the audio already queued for playback
`Bug` `Voice pipeline`

> **Status: ✅ Fixed, then superseded.** The queue is cleared via
> `AudioSource.clear_queue()` on cancellation. The follow-up pass found that fixing
> the flush was necessary but not sufficient: barge-in was firing on interim
> transcripts and raw VAD, so it was cancelling replies that should never have been
> interrupted at all — see [V1](#v1) and [V2](#v2). Cancellation also now awaits the
> previous turn before starting the next ([V7](#v7)), because returning while the old
> turn sat inside `capture_frame` left its speech playing over the new one.

**Why it's a problem** *(as originally found)*
Cancelling `agent_task` on barge-in stops further `capture_frame()` calls, but frames already handed to LiveKit's `AudioSource` are not flushed anywhere in the codebase. The user hears a tail of the assistant's previous sentence keep playing after they've started talking over it — which defeats the point of the barge-in feature the code comments describe as implemented.

**Files**
- `app/services/tts_streamer.py`
- `app/livekit_worker.py` (`_speech_started_callback`)

**Fix**
On cancellation, clear the `AudioSource`'s queued frames (or recreate it) before returning, so buffered-but-unplayed audio drops immediately.

---

### H13 — A Deepgram outage triggers a reconnect attempt on every single audio frame
`Performance` `Voice pipeline`

**Why it's a problem**
`audio_pipeline()` checks `bridge.connected` and calls `ensure_connected()` for every resampled frame — roughly 50 times a second at 20ms frames. During a provider outage, every one of those triggers a fresh reconnect attempt with no backoff, hammering Deepgram's API, flooding logs, and burning CPU for the duration of the outage.

**Files**
- `app/livekit_worker.py:257-266`

**Fix**
Add exponential backoff with a retry ceiling, either in the audio pipeline's reconnect call or inside `DeepgramBridge` itself (e.g. a "don't retry before T" timestamp).

---

## Medium

*Real costs to reliability, maintainability, or correctness at scale.*

### M1 — Internal exception strings leak straight to API clients
`Security` `API design`

15+ handlers return `detail=f"...{str(e)}"` directly from a caught exception, which can surface stack-trace fragments, hostnames, or file paths to any unauthenticated caller.

**Files:** `app/routes/agent_routes.py` — every `HTTPException(..., detail=f"...{str(e)}")` site

**Fix:** Log `str(e)` server-side via `logger.exception`; return a generic message, with detail only when `settings.environment == "development"`.

---

### M2 — CORS has no production origin configured, and the WebSocket has no origin check at all
`Security` `API design`

`allowed_origins` defaults to localhost; `render.yaml` never sets `ALLOWED_ORIGINS` for the backend service, so production either silently blocks the deployed frontend or gets loosened ad hoc. Separately, `/stream` performs no Origin check at all — CORS middleware doesn't apply to WebSocket upgrades in Starlette.

**Files:** `app/config.py:87` · `render.yaml` (no `ALLOWED_ORIGINS` key) · `app/routes/agent_routes.py:717-762`

**Fix:** Set `ALLOWED_ORIGINS` explicitly in `render.yaml`; add explicit Origin validation before accepting the WebSocket handshake.

---

### M3 — List endpoints have no pagination or upper bound
`API design` `Scalability`

`get_episodes`, `list_profile_facts`, and `list_playbooks` can return unbounded result sets as data grows, and caller-supplied `limit` values aren't clamped.

**Files:** `app/routes/agent_routes.py:597-605, 652-660, 695-702`

**Fix:** Add `Query(..., le=100)` caps and offset pagination to every list endpoint.

---

### M4 — No connection pool sizing on the async Postgres engine
`Database` `Scalability`

SQLAlchemy defaults apply (5 + 10 overflow = 15 max connections), while every chat turn fires several concurrent writes (chat message, smart-memory extraction, episode, profile facts) across possibly many concurrent voice and REST sessions sharing this one engine. 15 connections exhausts fast, especially against a starter-tier Postgres connection cap.

**Files:** `app/memory/short_term_memory.py:22-33`

**Fix:** Set explicit `pool_size`/`max_overflow`/`pool_timeout` sized to expected concurrency, so exhaustion fails fast instead of hanging requests.

---

### M5 — The only guard against secrets landing in profile facts — and then in every LLM prompt — is a bypassable heuristic
`Security`

`_is_sensitive()` matches on key-name keywords and a "looks like a 20+ char opaque token" regex. Store a password under the key `info`, or a short token, and it sails through — straight into `inject_memory_context`, which puts every stored profile fact into every agent's system prompt.

**Files:** `app/memory/short_term_memory.py:655-716`

**Fix:** Treat this as a real security boundary: require an allowlist of permitted profile-fact keys rather than accepting arbitrary LLM-chosen keys filtered by a value-shape heuristic.

---

### M6 — Postgres and Qdrant/mem0 writes aren't transactional, and the vector-write failure is invisible
`Database` `Missing error handling`

`on_agent_response()` writes chat history to Postgres, then the same content to Qdrant, as two independent calls. If the vector write fails, `smart_memory.store_memory` swallows the exception internally (prints, returns `None`) — that turn is permanently absent from long-term/semantic memory with no operator-visible signal.

**Files:** `app/memory/memory_manager.py:77-110` · `app/memory/smart_memory.py` (`store_memory`)

**Fix:** Log at error level (not `print`) on silent vector-write failure; consider a lightweight outbox/retry if cross-store consistency matters.

---

### M7 — Every attendance summary pulls a user's entire history and aggregates it in Python
`Performance` `Database`

`retrieve_attendance()` has no `LIMIT` and no date range, and it's called unbounded from both the attendance-summary and study-schedule tools. Fine at small scale, degrades linearly forever.

**Files:** `app/memory/short_term_memory.py:174-217` · `app/agents/academic_agent.py`

**Fix:** Add a default date window (e.g. current semester), or aggregate with SQL (`GROUP BY subject, status`) instead of pulling every row into Python.

---

### M8 — A failed `new_page()` leaks the Playwright browser process
`Memory leak`

`browser = await p.chromium.launch(...)` and `page = await browser.new_page()` both run before the `try/finally` that closes the browser. If `new_page()` throws — plausible under load, see H1/M4 — the browser process is never closed.

**Files:** `app/tools/attendance_tool.py:83-112`

**Fix:** Move `launch()` inside the `try` block, or wrap the whole launch-to-close sequence in `try/finally` from the start.

---

### M9 — The chunker's own comment says it splits oversized paragraphs — it doesn't
`Bug`

When a single paragraph already exceeds `chunk_size`, the code comment reads "paragraph itself is too long, need to split it," but the implementation just assigns `current_chunk = paragraph` unchanged. Any document without blank-line breaks — common in PDF-extracted resume text — produces oversized chunks that blow past the intended token budget.

**Files:** `app/services/chunking_service.py:129-132`

**Fix:** Recursively split oversized paragraphs by sentence or fixed token window using the existing `tiktoken` encoder instead of passing them through.

---

### M10 — Two chunking implementations exist, and they disagree
`Code duplication` `Architecture`

`chunking_service.py` implements a generic token-based chunker; separately, `long_term_memory_qdrant.py` implements its own bespoke semantic resume-chunker (name/skills/projects detection) and only references the generic one as an unused attribute. Two different splitting strategies, no shared contract, unclear which is canonical.

**Files:** `app/services/chunking_service.py` vs `app/memory/long_term_memory_qdrant.py:27-31,102+`

**Fix:** Pick one canonical strategy; either delete the unused one or have the semantic chunker call into it for its own oversized-section fallback (fixing M9 in one place).

---

### M11 — Raw user queries and memory payloads are printed to stdout, unconditionally, on every turn
`Security` `Maintainability`

`debug_logger.log_step` is a bare `print()` with no environment gate, called from the workflow, memory manager, and Qdrant service on effectively every step of every request — `"USER QUERY"`, `"MEMORY UPSERT"`, `"QDRANT RESULT"`. In production this writes every user's raw query text and retrieved memory to stdout on every turn, un-redacted, with no way to filter it by log level. `agent_routes.py` and `smart_memory.py` add their own scattered `print()` calls on top.

**Files:** `app/services/debug_logger.py` · call sites in `app/agents/workflow.py`, `app/memory/memory_manager.py`, `app/services/qdrant_service.py`

**Fix:** Gate behind `settings.environment == "development"` or a dedicated flag, and route through `logging` at `DEBUG` level instead of raw `print`, so it can be silenced and redacted in production.

---

### M12 — Every service is a singleton built at import time — nothing here can be unit-tested without live API keys
`Maintainability`

`groq_service`, `cohere_service`, and `qdrant_service` are instantiated at module import, and their constructors read mandatory (no-default) settings fields. Importing almost any module in this codebase transitively imports `app.config.settings`, which raises at import time without `GROQ_API_KEY`/`COHERE_API_KEY`/`QDRANT_URL`/`QDRANT_API_KEY` set — there's no way to unit-test even an unrelated module in isolation.

**Files:** `app/services/groq_service.py`, `cohere_service.py`, `qdrant_service.py` (import-time construction) · `app/config.py:13,42,47-48` (mandatory fields)

**Fix:** Make external API keys `Optional[str] = None`; validate presence lazily at call time; construct singletons behind a factory so tests can inject fakes.

---

### M13 — A dead Deepgram connection fails silently from the user's perspective
`Missing error handling` `Voice pipeline`

When `ensure_connected()` fails, the frame loop just breaks and drops that audio — no message is ever published on the data channel telling the frontend "voice recognition is down." The user experiences a mic that appears to do nothing, indefinitely, with no error surfaced in the UI.

**Files:** `app/livekit_worker.py:257-266`

**Fix:** After N consecutive reconnect failures, publish a `{"type":"error", ...}` data-channel message so the frontend can surface it.

---

### M14 — Bridge teardown is fire-and-forget, unlike the clean-shutdown path a few lines away
`Async issue` `Voice pipeline`

`on_participant_disconnected` and `on_track_unsubscribed` call `asyncio.create_task(state.bridge.close())` without storing or awaiting it — any exception becomes an unretrieved-exception log line instead of a structured error, and a re-subscribing participant can get a new bridge created before the old one has actually finished tearing down. The main shutdown path a few lines below does this correctly with `await state.bridge.close()`.

**Files:** `app/livekit_worker.py:317-321, 345-348, 392-396` vs the correct pattern at `:415-421`

**Fix:** Track cleanup tasks and await them (or attach a done-callback that logs exceptions through the module logger) before allowing a new bridge for the same identity.

---

<a id="m15"></a>
### M15 — `active_workers` is a plain in-process dict — fine today, breaks the moment this scales past one process
`Scalability` `Architecture`

> **Status: ⏸ Still open, but hardened.** Still a process-local dict, so this remains
> a hard blocker on running a second uvicorn worker or Render instance — a caller
> would be routed to an instance with no voice worker and would get silence.
> What did change: the worker now exits on room disconnect and after 120 s idle
> (previously it never exited, which was worse), and it invokes an `on_shutdown`
> callback that removes its `active_workers` entry **synchronously, before any
> teardown await**. That closes a race where a token request arriving during
> shutdown reused a worker already on its way out. The remaining work is unchanged:
> a shared store with a distributed lock, or a real job queue.

No persistence, no cross-instance coordination. Safe under the current single-uvicorn-worker deployment (no `await` between the check and the set), but two processes — a second uvicorn worker, a second Render instance — could each spawn a LiveKit worker for the same room. A process crash also abandons the room server-side with no explicit cleanup.

**Files:** `app/routes/livekit_routes.py:20, 50-63`

**Fix:** Before scaling horizontally, move worker-assignment state to a shared store (Redis) with a distributed lock, or dispatch via a real job queue.

---

<a id="m16"></a>
### M16 — One exhausted step aborts an entire multi-step plan, even when later steps don't depend on it
`LangGraph design`

> **Status: ⏸ Still open.** The plan-advance block still requires `status != "failed"`
> (`app/agents/workflow.py:258`). Deferred: doing this properly needs a dependency
> declaration between plan steps, otherwise "continue anyway" produces a cover letter
> about jobs that were never found.

The plan-advance block in `reflect_node` requires `status != "failed"` to move to the next step. So for "search for jobs and draft a cover letter," if the job-search step fails three times, the email-draft step — which doesn't actually require search results to exist — never runs at all. The whole plan aborts instead of proceeding with what it can.

**Files:** `app/agents/workflow.py` (`reflect_node` plan-advance block)

**Fix:** Let independent later steps execute regardless, or at minimum tell the user which step failed versus which was skipped as a result.

---

### M17 — Each specialist agent's personality is hand-written twice
`Code duplication` `Maintainability`

The voice/streaming conversational path (`streaming_workflow.py`) hardcodes its own `agent_prompts` dict describing job/email/academic/profile personas, entirely separate from each agent class's own `base_system_prompt`. Any tone or behavior change now has to be made in two places, and the two are already slightly different.

**Files:** `app/agents/streaming_workflow.py` (`_get_agent_system_prompt`) vs `job_agent.py`, `email_agent.py`, `academic_agent.py`, `profile_agent.py`

**Fix:** Derive the streaming prompt from the same source as the tool-calling agent (a shared short description constant per agent) instead of a second hand-written copy.

---

### M18 — Malformed planner output fails silently, with zero logging
`Missing error handling`

`planner_agent._parse_response()`'s except-block defaults to `profile`/no-clarification/confidence 0.5 with no warning logged — unlike `base_agent._parse_reasoning_decision`, which does log a warning on non-JSON output. If the routing model's output format ever drifts, every misrouted query is invisible in production.

**Files:** `app/agents/planner_agent.py` (`_parse_response` except block)

**Fix:** Log a warning with the raw response on parse failure, matching the pattern already used in `base_agent.py`.

---

### M19 — No overall deadline on a workflow run — worst case stacks to several minutes
`Performance` `Scalability`

The reflect loop (up to 3 retries) times a specialist's own internal reasoning loop (up to 5 iterations for the email agent) times a 30s per-LLM-call timeout plus a 15s per-tool timeout — with no ceiling wrapping any of it. A client or proxy timeout can disconnect while the workflow keeps running server-side, still writing to the database after nobody's listening.

**Files:** `app/agents/workflow.py` (`run_workflow`) · `app/routes/agent_routes.py` (`/query`)

**Fix:** Wrap `run_workflow` in an overall `asyncio.wait_for` with a sane ceiling (45-60s) and return a clear timeout response instead of an open-ended hang.

---

### M20 — A fire-and-forget task with no retained reference can be garbage-collected mid-write
`Async issue`

`execute_reasoning_loop` calls `asyncio.create_task(_mm.save_tool_outcome(...))` without holding a reference anywhere. Per the well-documented asyncio pitfall, a task with no strong reference can be collected before it finishes, silently dropping the write and swallowing whatever it raised.

**Files:** `app/agents/base_agent.py` (`execute_reasoning_loop`)

**Fix:** Hold references in a module-level set and discard on completion (the standard fire-and-forget pattern), or just await it — it's off the response's critical path either way.

---

## Low

*Cleanup, polish, and small correctness gaps.*

### L1 — A full legacy ChromaDB memory implementation still sits in the tree, unused
`Dead code`

260 lines, never imported anywhere, and it imports `chromadb` — which isn't even in `requirements.txt`. Pure confusion for the next person who has to figure out which of two "long-term memory" files is canonical.

**Files:** `app/memory/long_term_memory.py` (entire file)

**Fix:** Delete it.

---

### L2 — Cohere retry backoff has no jitter
`Performance`

Fixed exponential delay shared across all concurrent callers — under a burst hitting a rate limit, everyone backs off in lockstep and retries at the same instant.

**Files:** `app/services/cohere_service.py:38-71`

**Fix:** Add random jitter to the backoff delay.

---

<a id="l3"></a>
### L3 — The assistant's markdown is shown to users as literal asterisks
`UX`

> **Status: ⏸ Still open.** `ChatShell.tsx:854` still renders message text with
> `whitespace-pre-wrap`. Safe, but users see raw `**bold**`. Note this now also
> affects the live caption bubble, which renders the same way. Deferred: needs a new
> frontend dependency, and it must be one that does not enable raw HTML.

`response_agent.py` formats replies with markdown (bold headers, email structure), but `ChatShell.tsx` renders it as plain `whitespace-pre-wrap` text — safe from XSS, but the user sees raw `**bold**` syntax instead of formatted text.

**Files:** `frontend/components/ChatShell.tsx:854` (and the partial-caption bubble)

**Fix:** Render with a markdown library that defaults to no raw HTML.

---

### L4 — React Strict Mode is off for the whole app to paper over one dev-only bug
`Maintainability`

`reactStrictMode: false` applies everywhere, not just to the LiveKit double-connect symptom it was set to work around — permanently disabling a safety net that would otherwise catch missing effect-cleanup bugs, of which this codebase has several.

**Files:** `frontend/next.config.js`

**Fix:** Re-enable it; guard the LiveKit connect effect against double-invocation directly instead.

---

### L5 — The frontend points at a WebSocket route that doesn't exist on the backend
`Dead code`

`getVoiceWebSocketUrl()` builds a URL for `/api/v1/voice/stream_v2`, which is defined nowhere in the backend (only `/agents/stream` and `/voice/token` exist), and the function is never even called elsewhere in the component.

**Files:** `frontend/components/ChatShell.tsx:61-65`

**Fix:** Delete it, unless a non-LiveKit WebSocket fallback is genuinely intended — in which case implement and wire it.

---

### L6 — Two config flags exist that nothing ever reads
`Dead code`

`cartesia_streaming_enabled` and `streaming_tts_enabled` suggest a toggle between streaming and non-streaming TTS, but `tts_streamer.py` unconditionally calls the streaming path regardless of either flag.

**Files:** `app/config.py:72, 81`

**Fix:** Wire the flags into `tts_streamer.py`, or remove them.

---

### L7 — A documented confidence threshold and the actual code disagree
`Dead code` `Logic error`

`CONFIDENCE_THRESHOLD = 0.6` is defined and never referenced anywhere; the real safety-net check in `_parse_response` hardcodes `confidence < 0.35` instead — a different number than the one the module documents.

**Files:** `app/agents/planner_agent.py`

**Fix:** Use the named constant in the safety-net check, or delete it and document the real threshold where it's applied.

---

### L8 — The TTS text chunker's cleanup path is fragile
`Async issue`

> **Status: ✅ Fixed, then rewritten.** Both original problems are gone. The follow-up
> pass found a worse one in the same `finally` block: it *yielded* from there, so any
> early consumer exit — every barge-in — raised
> `RuntimeError: async generator ignored GeneratorExit`. See [V3](#v3). The chunker
> now also releases its first chunk at ~4 words ([V17](#v17)) and guards abbreviations
> and decimals against being read as sentence ends.

`pending` is only assigned after `token_stream.__aiter__()` succeeds — if that call itself raises, the `finally` block's reference to `pending` throws `UnboundLocalError` and replaces the real error. Cancelled tasks in that block also aren't awaited, so their completion (and any exception) is never observed.

**Files:** `app/services/text_chunker.py:36-105`

**Fix:** Initialize `pending = set()` before the `try`; await cancelled tasks with `return_exceptions=True`.

---

### L9 — The last few milliseconds of every spoken reply get dropped
`Bug` `Voice pipeline`

A sub-frame remainder of synthesized audio carried in `buffer` is discarded in `finally: buffer.clear()` at the end of a TTS stream rather than zero-padded and flushed as one final frame.

**Files:** `app/services/tts_streamer.py:38-75`

**Fix:** On clean completion, zero-pad any remaining buffer to a full frame and `capture_frame()` it before clearing.

---

### L10 — A few broad `except Exception` blocks turn real failures into quiet, degraded results
`Missing error handling`

An expired Tavily key, a Cartesia outage, or a failed auto-save all currently look identical to "nothing found" — `except Exception: return []` style handling with no logging.

**Files:** `app/tools/job_search_tool.py:144-145` · `app/services/voice_service.py:249-250` · `app/agents/email_agent.py:97-98`

**Fix:** Log via `logger.error`/`exception` before returning the degraded result, so failures are visible even though the user-facing behavior stays graceful.

---

### L11 — Rows that fail to parse are dropped without a count or reason
`Missing error handling`

Unlike `timetable_pdf_parser.py`, which returns `parse_notes`, the attendance and timetable date/time parsers silently `continue` past unrecognized formats — a scrape that drops most of its rows looks identical to a clean, small result.

**Files:** `app/tools/attendance_tool.py:33-36,116-124` · `app/tools/timetable_tool.py:36-39,115-123`

**Fix:** Return a `skipped_count`/`skip_reasons` summary alongside `stored_count`, matching the PDF parser's existing pattern.

---

### L12 — Two tools reach past the memory facade into its internals
`Architecture`

`job_search_tool.py` and `email_draft_tool.py` call `memory_manager.long_term.search_all(...)` directly instead of through a dedicated `memory_manager` method — coupling them to the internal shape of a class that's supposed to be the single interface to the memory subsystem.

**Files:** `app/tools/job_search_tool.py:87-91` · `app/tools/email_draft_tool.py:90-94`

**Fix:** Add thin pass-through methods on `memory_manager`, consistent with every other subsystem it wraps.

---

### L13 — Email drafts never see the user's saved name or preferred tone
`AI/LLM workflow`

`_retrieve_rag_context` hardcodes `chat_history` and `preferences` to empty when building the drafting prompt — only Qdrant skills/projects/resume content reaches it. A saved `preferred_tone` or the user's real name via `remember_preference` never makes it into a drafted email.

**Files:** `app/tools/email_draft_tool.py:96-102`

**Fix:** Pass through `get_profile_facts(user_id)` alongside the long-term context so drafts can use the user's name and tone preference.

---

### L14 — The Cohere query cache sorts its entire contents just to evict the oldest ones
`Performance`

Every time the 512-entry cap is exceeded, eviction does a full `sorted()` over every entry to find the oldest 128 — negligible today, but the wrong data structure for what should be O(1).

**Files:** `app/services/cohere_service.py:98-102`

**Fix:** Use an `OrderedDict`-based LRU (move-to-end + `popitem(last=False)`).

---

<a id="l15"></a>
### L15 — The `conversation_history` parameter is dead weight on the app's own primary chat path
`Architecture`

> **Status: ⏸ Still open for the HTTP chat path.** `askWithVoice()` still sends no
> `conversation_history`. The voice path is now the exception: the worker keeps
> `voice_history_turns` of in-process history per participant and passes it to both
> the router and the workflow, so the parameter is live there. The inconsistency is
> the thing to resolve — two continuity mechanisms, each wired on a different path.

The frontend's text-chat call, `askWithVoice()`, never sends `conversation_history` — so the history-trimming logic every specialist agent carries in `execute_reasoning_loop` is unreachable from the app's own UI. Multi-turn continuity for that path runs entirely through server-side memory retrieval instead: two mechanisms for the same concern, only one of which is actually wired end-to-end.

**Files:** `frontend/lib/api.ts` (`askWithVoice`) · `app/agents/base_agent.py` (`execute_reasoning_loop`)

**Fix:** Either have the frontend send recent turns explicitly, or delete the unused parameter path and document `memory_prompt` as the sole continuity mechanism.

---

### L16 — Chat history has no composite index, despite always being queried by both columns together
`Performance` `Database`

`ChatHistory` indexes `user_id` and `session_id` separately, but every read path (`get_recent_context`, `retrieve_chat_history`) filters on both plus an `ORDER BY created_at`.

**Files:** `app/memory/models.py` (`ChatHistory`)

**Fix:** Add a composite index on `(user_id, session_id, created_at)` matching the actual query shape.

---

## Known limitations (not defects — accepted trade-offs)

Recorded so they are not rediscovered as bugs:

1. **No live voice verification.** Every voice fix is covered by tests with faked
   provider boundaries. Nothing has been confirmed against real LiveKit, Deepgram,
   or Cartesia traffic in a browser. This is the single biggest gap.
2. **Echo cancellation is the browser's job.** With speakers loud and AEC imperfect,
   the assistant's voice can still be transcribed and answered as user input. There
   is no server-side AEC and deliberately no "mute STT while speaking" gate — that
   would disable barge-in. Headphones are the reliable configuration.
3. **`voice_caption_offset_seconds` (150 ms) is an estimate** of the browser's jitter
   buffer, not a measurement. It is the one value most likely to need tuning by ear.
4. **Aggressive endpointing splits long pauses.** A mid-sentence pause over 500 ms
   starts a reply that the continuation then interrupts. Recoverable, but can feel
   jumpy; tune `DEEPGRAM_ENDPOINTING_MS`.
5. **The tool route does not stream**, so it waits for the whole LangGraph run before
   speaking — seconds, not sub-second.
6. **Interrupted turns fragment history** as `user` / `assistant …[interrupted]` / `user`.
7. **Cartesia is called over HTTP, not WebSocket**, so each chunk pays request
   overhead. Their WS API would cut more latency.
8. **`refactor_worker.py` is dead cruft** — a stale codemod targeting worker code that
   no longer exists. Safe to delete; left alone to keep the voice work scoped.

---

## Document history

| Date | Change |
|---|---|
| 2026-08-06 | Original read-only audit — 54 findings, no code changed |
| 2026-08-06 | First remediation pass — 47 of 54 fixed |
| 2026-08-07 | Auth landed (closes C1, C2); voice pipeline traced end to end — 23 further defects found and fixed; totals now 49 of 54 original findings closed |

*Finding sections describe the code as originally read. Each **Status** line records
where that finding stands as of the last update above and is authoritative where the
two disagree.*
