"""
Application configuration using Pydantic Settings.
Loads environment variables for Groq API and application settings.
"""
import re
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Groq API Configuration (REQUIRED at runtime — see validate_required_keys()).
    # Declared Optional so importing any module does not raise before the app
    # starts; startup calls validate_required_keys() to fail fast with a clear
    # message, and tests can import modules without a populated .env.
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.7
    groq_max_tokens: int = 2048

    # Application Configuration
    app_name: str = "My_Agent"
    app_version: str = "1.0.0"
    environment: str = "development"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 10000

    # Memory Configuration
    # PostgreSQL — relational memory and application records
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "my_agent_db"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # Cohere Configuration (REQUIRED at runtime — see validate_required_keys())
    cohere_api_key: Optional[str] = None
    cohere_model: str = "embed-english-v3.0"
    cohere_embedding_dimension: int = 1024

    # Qdrant Configuration (REQUIRED at runtime — see validate_required_keys())
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_timeout: int = 30

    # PostgreSQL connection pool sizing.
    # Every chat turn issues several concurrent writes and each LiveKit
    # participant holds its own workflow, so the SQLAlchemy default of 5+10
    # is exhausted quickly. pool_timeout makes exhaustion fail fast and
    # loudly instead of hanging the request indefinitely.
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_pool_timeout: int = 30
    postgres_pool_recycle: int = 1800

    # ── Memory redesign rollout ──────────────────────────────────────────
    # Phase 1 mirrors structured writes into the unified `memory_records`
    # table while the legacy tables continue to serve every read. Mirroring is
    # additive and failure-tolerant, so it defaults on; the flag exists to kill
    # it instantly without a deploy if it ever misbehaves in production.
    # See docs/MEMORY_ARCHITECTURE.md §6.
    memory_v2_dual_write: bool = True

    # Phase 2 runs the new retrieval engine alongside the legacy path, logging
    # a comparison while the legacy result is always what gets served. It runs
    # in the background so it adds no turn latency.
    #
    # Defaults OFF, unlike dual-write: comparing against a store that has not
    # been backfilled yet produces noise, not signal. Enable after running
    # scripts/migrate_memory_v2.py --apply.
    memory_v2_shadow_read: bool = False

    # Token budget for the assembled memory block. Allocated across tiers
    # before rendering — see docs/MEMORY_ARCHITECTURE.md §3.7.
    memory_v2_budget_tokens: int = 6000

    # ── Phase 3: asynchronous extraction ─────────────────────────────────
    # Turns are extracted in batches rather than one at a time: roughly 5x
    # cheaper, and a multi-turn window yields better memories because a single
    # turn frequently lacks the context to state a self-contained fact.
    memory_extraction_batch_size: int = 5
    # A short conversation must not sit unextracted forever waiting to reach
    # the batch size, so a group also flushes once it has been idle this long.
    memory_extraction_idle_flush_seconds: float = 180.0

    # The background worker drains the outbox and embeds pending records.
    # In-process is correct for a single-instance deployment; run
    # scripts/run_memory_worker.py instead when scaling out.
    memory_worker_enabled: bool = True
    memory_worker_poll_seconds: float = 30.0
    memory_embedding_batch_size: int = 32

    # Turns off the legacy per-utterance vector write. Kept ON by default
    # because the *legacy* prompt still reads that data — flipping this before
    # the Phase 6 read cutover would remove a served signal without its
    # replacement being served yet. See docs/MEMORY_ARCHITECTURE.md §6.
    memory_v2_replace_raw_embedding: bool = False

    # ── Phase 5: lifecycle ───────────────────────────────────────────────
    # Maintenance runs far less often than extraction — these are sweeps over
    # the whole store, not per-turn work.
    memory_maintenance_enabled: bool = True
    memory_maintenance_interval_seconds: float = 3600.0

    # A record is archived when importance × recency falls below this. Archived
    # records leave default retrieval but remain searchable on demand and are
    # never deleted, so the threshold errs generous.
    memory_decay_threshold: float = 0.15
    # Nothing is archived before this age regardless of score: a brand-new
    # low-importance memory has not had a chance to prove useful yet.
    memory_decay_min_age_days: float = 30.0

    # Cosine similarity above which two same-kind memories are treated as
    # near-duplicates and merged. Deliberately high — wrongly merging two
    # distinct facts destroys information, while missing a duplicate only
    # wastes a row.
    memory_dedup_similarity_threshold: float = 0.92

    # Anonymous recruiter sessions accumulate a permanent partition each. They
    # are purged after this long with no activity.
    memory_guest_retention_days: int = 30

    # Text Chunking Configuration
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Job Search Tool (Tavily)
    tavily_api_key: Optional[str] = None

    # Voice: Deepgram STT
    deepgram_api_key: Optional[str] = None
    deepgram_model: str = "nova-2"
    deepgram_streaming_enabled: bool = False
    deepgram_utterance_end_ms: int = 1000  # Deepgram API strictly requires a minimum of 1000ms
    deepgram_vad_events: bool = True
    # Trailing silence before Deepgram marks a final result `speech_final`.
    # This is what actually decides how fast the assistant starts answering:
    # utterance_end_ms cannot go below 1000 ms, so without endpointing every
    # reply carried a full second of dead air. 300 ms is responsive without
    # cutting off people who pause mid-sentence.
    deepgram_endpointing_ms: int = 500

    # ── Voice turn behaviour ─────────────────────────────────────────────
    # A spoken turn is bounded by *progress*, not by elapsed time. Speech is a
    # real-time medium: pushing frames self-paces to playback, so a 90-second
    # answer legitimately occupies the turn for 90 seconds. A wall-clock
    # deadline therefore cannot distinguish "long" from "stuck" and kills long
    # replies mid-sentence. This is the gap since the last token, audio frame,
    # or spoken word — exceed it and the turn really is wedged.
    voice_turn_stall_seconds: float = 20.0
    # Last-resort ceiling so a pathological loop cannot hold a turn forever.
    # Sized well beyond any believable spoken answer (max_tokens=1024 is roughly
    # five minutes of speech), so reaching it means something is wrong.
    voice_turn_max_seconds: float = 600.0
    # Captions are scheduled from LiveKit's playout queue, which does not know
    # about the browser's own jitter buffer. This nudges text later so it lands
    # with the voice rather than a moment ahead of it.
    voice_caption_offset_seconds: float = 0.15
    # Minimum words a transcript must contain before it interrupts the
    # assistant mid-sentence. Raw VAD alone barges in on coughs, keyboard
    # noise, and the assistant's own voice leaking back through the speakers.
    voice_bargein_min_words: int = 2
    # Turns of spoken context kept in the worker's in-process history.
    voice_history_turns: int = 12

    # Voice: Cartesia TTS
    cartesia_api_key: Optional[str] = None
    cartesia_model_id: str = "sonic-2"
    cartesia_voice_id: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    cartesia_version: str = "2024-06-10"
    cartesia_sample_rate: int = 24000

    # LiveKit WebRTC (Phase 0 — settings only, not active yet)
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None

    # Voice Agent Optimization Feature Flags
    streaming_stt_enabled: bool = True
    parallel_workflow_enabled: bool = True  # Run memory + planner concurrently (~200-300ms gain)
    binary_audio_enabled: bool = False
    vad_enabled: bool = True

    # CORS — restrict in production (comma-separated origins in .env)
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    # ── Request limits ───────────────────────────────────────────────────
    # Starlette enforces no body-size cap by default, so an unbounded upload
    # is read fully into memory before any validation runs.
    max_upload_bytes: int = 10 * 1024 * 1024   # 10 MB
    max_text_upload_chars: int = 500_000
    max_query_chars: int = 8_000

    # ── Rate limiting (in-process, per client IP) ────────────────────────
    # Protects LLM/embedding/browser-automation endpoints from cost-exhaustion
    # and resource-exhaustion abuse. Limits are deliberately generous so normal
    # interactive use is never throttled. See app/middleware/rate_limit.py.
    rate_limit_enabled: bool = True
    rate_limit_default_per_minute: int = 120     # ordinary reads
    rate_limit_llm_per_minute: int = 30          # /agents/query, /stream
    rate_limit_upload_per_minute: int = 10       # PDF/text ingestion
    rate_limit_expensive_per_minute: int = 5     # scraping, token minting
    rate_limit_auth_per_minute: int = 10         # login/guest — brute-force guard

    # ── Workflow execution ───────────────────────────────────────────────
    # Ceiling on a single end-to-end agent run. Without it, reflect retries ×
    # reasoning iterations × per-call timeouts can stack into several minutes
    # while the client has long since disconnected.
    workflow_timeout_seconds: float = 120.0

    # ── Outbound network safety ──────────────────────────────────────────
    # Server-side fetches driven by user-supplied URLs (ERP scraping) must not
    # be able to reach internal infrastructure or cloud metadata endpoints.
    allow_private_network_scraping: bool = False

    # ── Authentication ───────────────────────────────────────────────────
    # Separate secrets for access and refresh tokens so a leaked access token
    # can never be replayed as a refresh token (and vice versa).
    # MUST be set in production — startup refuses to run with the dev default.
    jwt_access_secret: Optional[str] = None
    jwt_refresh_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "my-agent"

    access_token_ttl_minutes: int = 15
    owner_refresh_token_ttl_days: int = 30
    guest_refresh_token_ttl_days: int = 7

    # Owner credentials. The password is stored only as a bcrypt hash;
    # generate one with: python scripts/create_owner_password.py
    owner_username: Optional[str] = None
    owner_password_hash: Optional[str] = None
    # Stable identity the owner's data is filed under. Changing this orphans
    # existing memory, so it defaults to the historical hardcoded value.
    owner_user_id: str = "vansh"

    # Guest (recruiter/demo) sessions: anonymous, auto-issued, read-only.
    guest_sessions_enabled: bool = True

    # Cookie transport. secure=True requires HTTPS; samesite="none" is needed
    # when the frontend is served from a different site than the API.
    auth_cookie_secure: Optional[bool] = None      # defaults to (not development)
    auth_cookie_samesite: Optional[str] = None     # defaults to lax/none by env
    auth_cookie_domain: Optional[str] = None
    csrf_protection_enabled: bool = True

    # ── Debug tracing ────────────────────────────────────────────────────
    # When true, log_step() emits full step payloads (including raw user text).
    # Defaults to on only in development so production logs stay clean and
    # free of unredacted personal data.
    debug_step_logging: Optional[bool] = None

    # SMTP Email Sending (Gmail)
    smtp_email: Optional[str] = None        # your Gmail address
    smtp_password: Optional[str] = None     # Gmail App Password (not normal password)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

    # Observability: LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "my-agent"
    langsmith_endpoint: Optional[str] = None
    langsmith_tracing_enabled: bool = False

    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def is_streaming_stt_available(self) -> bool:
        """Check if streaming STT should be enabled based on API key availability."""
        return bool(self.deepgram_api_key and self.deepgram_api_key.strip())

    @property
    def is_livekit_configured(self) -> bool:
        """Check if LiveKit credentials are fully configured."""
        return bool(
            self.livekit_url
            and self.livekit_api_key
            and self.livekit_api_secret
        )

    @property
    def is_development(self) -> bool:
        return self.environment.strip().lower() == "development"

    @property
    def step_logging_enabled(self) -> bool:
        """Verbose per-step tracing: explicit flag wins, else dev-only."""
        if self.debug_step_logging is not None:
            return self.debug_step_logging
        return self.is_development

    @property
    def cookie_secure(self) -> bool:
        """Cookies are Secure everywhere except local development."""
        if self.auth_cookie_secure is not None:
            return self.auth_cookie_secure
        return not self.is_development

    @property
    def cookie_samesite(self) -> str:
        """
        SameSite policy for auth cookies.

        Production commonly serves the frontend from a different site than the
        API (Vercel + Render), which requires SameSite=None — and browsers only
        accept SameSite=None together with Secure. Localhost is same-site
        regardless of port, so development can use the stricter Lax.
        """
        if self.auth_cookie_samesite:
            return self.auth_cookie_samesite.lower()
        return "lax" if self.is_development else "none"

    @property
    def is_owner_login_configured(self) -> bool:
        return bool(
            (self.owner_username or "").strip()
            and (self.owner_password_hash or "").strip()
        )

    def validate_auth_config(self) -> List[str]:
        """
        Return fatal auth misconfigurations.

        Auth secrets have no safe default: a predictable signing key means
        anyone can mint a valid owner token, so a missing secret is fatal in
        every environment except local development.
        """
        problems: List[str] = []

        # 32 bytes is the minimum key length RFC 7518 recommends for HS256.
        _MIN_SECRET_LENGTH = 32
        for name, secret in (
            ("JWT_ACCESS_SECRET", self.jwt_access_secret),
            ("JWT_REFRESH_SECRET", self.jwt_refresh_secret),
        ):
            value = (secret or "").strip()
            if not value:
                problems.append(f"{name} is not set")
            elif len(value) < _MIN_SECRET_LENGTH:
                problems.append(
                    f"{name} is too short ({len(value)} chars); "
                    f"use at least {_MIN_SECRET_LENGTH}"
                )
        if (
            self.jwt_access_secret
            and self.jwt_access_secret == self.jwt_refresh_secret
        ):
            problems.append(
                "JWT_ACCESS_SECRET and JWT_REFRESH_SECRET must be different values"
            )
        if not self.is_owner_login_configured:
            problems.append("OWNER_USERNAME / OWNER_PASSWORD_HASH are not set")

        # The owner's user_id becomes a storage key and a Qdrant filter value,
        # so an operator typo containing path or quote characters must not
        # reach the persistence layer.
        owner_id = (self.owner_user_id or "").strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_\-\.]{1,126}[a-z0-9]", owner_id):
            problems.append(
                "OWNER_USER_ID must be 3-128 chars of lowercase letters, digits, "
                "hyphens, underscores, or dots"
            )
        if self.cookie_samesite == "none" and not self.cookie_secure:
            problems.append(
                "AUTH_COOKIE_SAMESITE=none requires AUTH_COOKIE_SECURE=true "
                "(browsers reject SameSite=None without Secure)"
            )

        return problems

    def validate_required_keys(self) -> List[str]:
        """
        Return the names of required third-party credentials that are missing.

        Called once at startup so the app fails fast with an actionable message
        rather than raising at import time (which would make the modules
        untestable) or failing deep inside a request.
        """
        required = {
            "GROQ_API_KEY": self.groq_api_key,
            "COHERE_API_KEY": self.cohere_api_key,
            "QDRANT_URL": self.qdrant_url,
            "QDRANT_API_KEY": self.qdrant_api_key,
        }
        return [name for name, value in required.items() if not (value or "").strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
