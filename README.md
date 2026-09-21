# My_Agent

A personal AI assistant with voice and text interfaces, built on FastAPI and a
hybrid memory system. It answers from what it actually knows about you —
résumé, timetable, attendance, saved facts, past conversations — and refuses
rather than guessing when a lookup fails.

Voice runs over LiveKit WebRTC with Deepgram speech-to-text and Cartesia
text-to-speech. Text runs over a REST endpoint and a token-streaming WebSocket.
Both share one deterministic router, one memory layer, and one confirmation
gate for actions that touch the outside world.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running it](#running-it)
- [Tests](#tests)
- [Configuration](#configuration)
- [Operations](#operations)
- [Project layout](#project-layout)
- [Further reading](#further-reading)

---

## What it does

| Capability | Notes |
|---|---|
| Answers questions about you | Résumé, skills, projects, education, saved profile facts |
| Academic assistant | Timetable, attendance, exams, daily plans |
| Job search and matching | Evidence-backed matching; never invents a qualification |
| Email drafting | Sending is held for explicit confirmation, never automatic |
| Conversational memory | Cross-session recall with provenance you can interrogate |
| Voice conversation | Barge-in, streaming captions, stall detection |
| Guest / recruiter mode | Read-only, scoped to public records |

Two properties are enforced structurally rather than by prompt instruction:

- **Grounding.** A question whose category requires a tool cannot be answered
  without that tool running. If it did not run, the model's answer is discarded
  and replaced with an honest one.
- **Confirmation.** Irreversible actions (`send_email`, deletions) are
  classified on the tool, intercepted before execution, and released only
  against a single-use token bound to a hash of the exact previewed action.

---

## Architecture

```
             ┌──────────── text ────────────┐   ┌──────── voice ────────┐
             │  POST /api/v1/agents/query   │   │  LiveKit room + WebRTC │
             │  WS   /api/v1/agents/stream  │   │  Deepgram STT          │
             └──────────────┬───────────────┘   └───────────┬────────────┘
                            │                               │
                     JWT → Principal → scopes        hybrid_router
                            │                               │
                            └───────────┬───────────────────┘
                                        │
                         deterministic router (query_intent)
                    categorise → sources → required tools → agent
                     (the planner is skipped when nothing it produces
                      can change the outcome of this turn)
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
     temporal      provenance       clarification    confirm_action   specialist
     (no model)    (no model)        (no model)      (gateway)     job│email│academic│profile
                                                                        │
                                                     required lookup runs first
                                                          (no model call)
                                                                        │
                                                            reasoning loop ≤2 iterations
                                                                        │
                                                            reflect → response
                                        │
   ┌────────────────────────────────────┴────────────────────────────────────┐
   │  Memory: Postgres (history, facts, records) · Qdrant (vectors)          │
   │          Cohere (embeddings) · category-scoped retrieval                │
   └─────────────────────────────────────────────────────────────────────────┘
```

Four categories terminate without any model call at all, and a question whose
route and required lookup are both already settled — "what is my name", "what
is my CGPA" — skips the planner and runs its lookup before the first
completion. Those turns cost **one** model call rather than three.

Every Groq request passes through one shared limiter (bounded concurrency plus
a token-per-minute budget) so a fan-out queues instead of arriving as a burst.
The limiter reserves each request's worst case up front and refunds the
difference once the provider reports what it actually cost, and every wait
underneath a turn — token budget, `Retry-After` — is capped by that turn's
remaining deadline rather than by its own private ceiling.

**External services:** Groq (LLM), Cohere (embeddings), Qdrant (vectors),
PostgreSQL (records), Deepgram (STT), Cartesia (TTS), LiveKit (WebRTC),
Tavily (web search), LangSmith (tracing, optional).

---

## Prerequisites

- **Python 3.11+** (3.12 recommended)
- **PostgreSQL 14+** reachable and empty, or an existing database you own
- **Node.js 18+** — only if you want the web frontend
- API keys: **Groq** and **Cohere** and a **Qdrant** endpoint are required.
  Deepgram, Cartesia and LiveKit are required for voice only. Tavily is
  required for job search only. LangSmith is optional.

---

## Setup

```bash
git clone <your-repo-url>
cd My_Agent

python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt -r requirements-dev.txt

# Browser for the ERP attendance scraper. pip installs Playwright itself but
# not the browser binary it drives, so the scraper fails at runtime without it.
# Skip only if you will not use attendance scraping.
playwright install chromium
```

### 1. Configure

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Open `.env` and fill in at minimum: `GROQ_API_KEY`, `COHERE_API_KEY`,
`QDRANT_URL`, `QDRANT_API_KEY`, and the `POSTGRES_*` block.

Two optional capabilities are off until their keys are filled in, and each says
so at startup rather than failing mid-turn:

| Capability | Needs | Without it |
|---|---|---|
| Job search | `TAVILY_API_KEY` | Job queries refuse and name the variable |
| Email sending | `SMTP_EMAIL`, `SMTP_PASSWORD` | Drafting still works; sending refuses |

`SMTP_PASSWORD` must be a Gmail **App Password** (Google Account → Security →
2-Step Verification → App passwords), not your account password — Google
rejects account passwords over SMTP.

`.env` is gitignored and must stay that way — it holds live credentials.

### 2. Generate authentication secrets

The app refuses to start without these, and it will tell you so.

```bash
python scripts/create_owner_password.py
```

Copy the printed `JWT_ACCESS_SECRET`, `JWT_REFRESH_SECRET` and
`OWNER_PASSWORD_HASH` into `.env`. The plaintext password is never stored.

### 3. Create the database

```bash
python scripts/init_db.py
python scripts/migrate_memory_v2.py --apply     # indexes + backfill
```

### 4. Set your Groq token budget

`GROQ_TOKENS_PER_MINUTE` **must match the TPM your Groq account actually
grants**. Setting it above your real limit disables the back-pressure gate in
practice — every request passes and is then rejected by the provider. Find your
limit in a 429 response body (`"Limit 8000, Used ..., Requested ..."`) or on the
Groq console. The default is the measured free-tier limit.

### 5. Load your data (optional)

```bash
python scripts/upload_pdf.py path/to/resume.pdf --user-id you
python scripts/upload_new_timetable.py path/to/timetable.pdf --user you
```

Both talk to the database directly, so neither needs the API running.
`scripts/upload_timetable_pdf.py` does the same job *through* the HTTP API and
therefore needs a server already up — and it defaults to `http://localhost:8000`,
not the port under [Running it](#running-it). Use it only when you want the
upload to go through the endpoint.

The parser is deterministic and needs a text layer with one class per line. A
scanned or grid-style timetable is refused rather than guessed at; see
`scripts/load_timetable.py` for the transcription path.

---

## Running it

**API**

```bash
uvicorn app.main:app --reload --port 10000
```

Interactive docs at `http://localhost:10000/docs`.

**Frontend** (optional)

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

**Voice worker** — starts on demand when a client requests a LiveKit token; no
separate process to run.

**Memory worker** — runs in-process by default (`MEMORY_WORKER_ENABLED=true`).
Behind more than one API instance, set it to `false` and run one worker
separately so replicas do not each drain the same queue:

```bash
python scripts/run_memory_worker.py
```

**Docker**

```bash
docker build -t my-agent .
docker run --env-file .env -p 10000:10000 my-agent
```

> The image does not install Playwright browsers, so the ERP attendance scraper
> is a local-only tool. Everything else works in the container.

---

## Tests

```bash
pytest                              # full suite
pytest tests/memory/test_memory_scope.py -v  # one file

# Postgres integration tests are opt-in and destructive against their own
# test database. Set POSTGRES_INTEGRATION_TESTS=1 in the environment first:
#   macOS/Linux:  POSTGRES_INTEGRATION_TESTS=1 pytest -m postgres
#   Windows:      set POSTGRES_INTEGRATION_TESTS=1 && pytest -m postgres
```

The suite runs fully offline — no API keys, no network, no database. Postgres
integration tests are opt-in and skipped by default. 2,507 tests; 2,480 run by
default and 27 skip.

```bash
pytest tests/memory -q            # one package
pytest tests/agents/test_persona.py -v
```

---

## Configuration

Every setting lives in `app/config.py` and is overridable from `.env`.
`.env.example` documents all of them. The ones most worth knowing:

| Setting | Default | Why it matters |
|---|---|---|
| `GROQ_TOKENS_PER_MINUTE` | `8000` | Must match your account's real TPM |
| `GROQ_MAX_CONCURRENCY` | `4` | Simultaneous in-flight Groq requests |
| `WORKFLOW_TIMEOUT_SECONDS` | `120` | Ceiling on one typed turn |
| `VOICE_WORKFLOW_TIMEOUT_SECONDS` | `35` | Shorter: the caller is waiting on audio |
| `VOICE_TURN_STALL_SECONDS` | `20` | Cancels a turn that stops making progress |
| `MEMORY_WORKER_MODEL` | unset | Set to a smaller model to keep background work off the user-facing budget |
| `MEMORY_V2_SHADOW_READ` | `false` | Costs a full extra retrieval per turn; for comparison data only |
| `RATE_LIMIT_ENABLED` | `true` | Per-IP HTTP limits; leave on outside local dev |
| `ALLOWED_ORIGINS` | localhost | **Must** be your real frontend URL in production |

---

## Operations

**Health**

| Endpoint | Cost | Use |
|---|---|---|
| `GET /health` | free — calls nothing | Liveness probe. Safe to poll continuously. |
| `GET /health/deep` | one probe per provider, memoized 60s | Is Groq/Cohere/Qdrant reachable? |

Point your platform's health check at `/health`. Pointing it at `/health/deep`
spends provider quota on every probe.

**Per-turn cost.** Every turn logs a `TURN_COST` line with logical LLM calls,
actual HTTP attempts, retries, 429s, embeddings, cache hits, coalesced
embeddings, Qdrant operations, and phase timings. Logical calls and HTTP
attempts are counted separately on purpose — that ratio is how retry
amplification becomes visible.

**Provider latency check**

```bash
python scripts/probe_providers.py
```

Issues a handful of small live requests and reports round-trip latency per
provider plus whether the voice router fits inside its budget.

---

## Project layout

```
app/
  agents/        planner, specialists, routing, grounding, action gateway
  memory/        hybrid memory: stores, retrieval, cognition worker, scope
  services/      Groq, Cohere, Qdrant, Deepgram, Cartesia, limiter, metrics
  tools/         timetable, attendance, job search, email draft, typed contract
  routes/        agents, auth, livekit
  auth/          JWT, cookies, CSRF, scopes
  domain/        repositories: jobs, email, schedule, audit, pending actions
  db/            engine and session
  matching/      evidence-based job matching
  candidate/     structured candidate profile built from the resume
  middleware/    per-IP rate limiting
  mcp/           MCP host (stdio transport)
  config.py           every setting, overridable from .env
  livekit_worker.py   voice turn loop
  main.py             FastAPI entry point
frontend/        Next.js 14 client
  components/    ChatShell (composition root), MessageList, MemoryPanel, modals
  lib/           api client, auth, conversation helpers
scripts/         setup, migrations, data loading, diagnostics
tests/           2,507 tests, offline by default — mirrors app/
  agents/ memory/ services/ auth/ tools/ routes/ matching/
  candidate/ domain/ mcp/ middleware/ voice/ evals/
  support/       fakes and harnesses shared across the suite
evals/           scenario-based agent evaluation
docs/            architecture and memory design notes
```

`tests/` mirrors `app/`, one package per subpackage, so the tests for a module
are where you would look for them. `tests/voice/` is the one grouping that has
no `app/` counterpart: the voice path spans `livekit_worker.py`, `agents/` and
`services/`, and splitting it across three packages would hide it.

---

## Further reading

- [`docs/MEMORY_ARCHITECTURE.md`](docs/MEMORY_ARCHITECTURE.md) — the memory
  system's design, and why it is shaped this way
- [`docs/ARCHITECTURE_AUDIT.md`](docs/ARCHITECTURE_AUDIT.md) — architectural
  review and roadmap
- [`docs/FINAL_AUDIT.md`](docs/FINAL_AUDIT.md) — latest findings
- [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) — historical security and
  correctness audit; source comments reference its finding IDs
