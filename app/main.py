"""
FastAPI main application entry point.
Configures async FastAPI server with Groq integration and hybrid memory system.
"""
import asyncio
import logging
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings

logger = logging.getLogger(__name__)
from app.services.groq_service import groq_service
from app.services.cohere_service import cohere_service
from app.services.qdrant_service import qdrant_service
from app.services.voice_service import voice_service
from app.services.langsmith_service import configure_langsmith
from app.memory.memory_manager import memory_manager
from app.middleware.rate_limit import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes memory systems and verifies API connectivity.
    """
    # Startup
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Environment: %s | Groq model: %s", settings.environment, settings.groq_model)

    # Fail fast on missing credentials. Services build their clients lazily so
    # modules stay importable without a .env; this is where a real deployment
    # is told, once and clearly, that it cannot function.
    missing = settings.validate_required_keys()
    if missing:
        raise RuntimeError(
            "Missing required configuration: "
            + ", ".join(missing)
            + ". Set these in your .env file (see .env.example)."
        )

    # Auth misconfiguration is fatal: a missing or shared signing secret means
    # anyone could mint a valid owner token.
    auth_problems = settings.validate_auth_config()
    if auth_problems:
        raise RuntimeError(
            "Authentication is misconfigured:\n  - "
            + "\n  - ".join(auth_problems)
            + "\n\nRun: python scripts/create_owner_password.py"
        )
    logger.info(
        "Auth ready — owner=%s, guest sessions %s, cookies (secure=%s, samesite=%s), CSRF %s",
        settings.owner_username,
        "enabled" if settings.guest_sessions_enabled else "disabled",
        settings.cookie_secure,
        settings.cookie_samesite,
        "on" if settings.csrf_protection_enabled else "OFF",
    )

    # Optional observability setup
    if configure_langsmith():
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    else:
        logger.info("LangSmith tracing disabled")

    # Initialize memory system
    logger.info("Initializing hybrid memory system...")
    try:
        await memory_manager.initialize()
        logger.info(
            "Memory system initialized — Qdrant=%s, Cohere=%s, PG=%s:%s",
            settings.qdrant_url, settings.cohere_model,
            settings.postgres_host, settings.postgres_port,
        )
    except Exception as e:
        logger.warning("Memory initialization failed: %s", e)

    # Verify API connections
    is_healthy = await groq_service.health_check()
    logger.info("Groq API: %s", "OK" if is_healthy else "DEGRADED")

    cohere_healthy = await cohere_service.health_check()
    logger.info("Cohere API: %s", "OK" if cohere_healthy else "DEGRADED")

    qdrant_healthy = await qdrant_service.health_check()
    logger.info("Qdrant: %s", "OK" if qdrant_healthy else "DEGRADED")

    if settings.is_streaming_stt_available:
        logger.info("Deepgram streaming STT enabled (linear16, 16000 Hz, WebSocket)")
    else:
        logger.warning("Deepgram streaming STT disabled — no API key")

    if settings.cartesia_api_key and settings.cartesia_api_key != "your_cartesia_api_key_here":
        logger.info("Cartesia TTS enabled")
    else:
        logger.info("Cartesia TTS disabled — no API key")

    if settings.rate_limit_enabled:
        logger.info(
            "Rate limiting enabled (llm=%d/min, upload=%d/min, expensive=%d/min, default=%d/min)",
            settings.rate_limit_llm_per_minute, settings.rate_limit_upload_per_minute,
            settings.rate_limit_expensive_per_minute, settings.rate_limit_default_per_minute,
        )
    else:
        logger.warning("Rate limiting is DISABLED — not recommended outside local development")

    if not settings.is_development and settings.allowed_origins == [
        "http://localhost:3000", "http://localhost:3001"
    ]:
        logger.warning(
            "ALLOWED_ORIGINS is still the localhost default in a non-development "
            "environment — browser requests from your deployed frontend will be blocked."
        )

    # LiveKit WebRTC readiness (Phase 0 — informational only)
    if settings.is_livekit_configured:
        logger.info("LiveKit WebRTC configured (url=%s)", settings.livekit_url)
    else:
        logger.info("LiveKit WebRTC not configured — WebSocket transport active")

    # Background memory worker: extracts durable memories from queued turns and
    # embeds pending records. In-process is correct for a single instance; when
    # running more than one, disable it here and run
    # scripts/run_memory_worker.py as a separate process instead, so several
    # replicas do not each poll the same queue.
    worker_stop: Optional[asyncio.Event] = None
    worker_task: Optional[asyncio.Task] = None
    if settings.memory_worker_enabled:
        from app.memory.cognition.worker import memory_worker

        worker_stop = asyncio.Event()
        worker_task = asyncio.create_task(
            memory_worker.run_forever(worker_stop), name="memory-worker"
        )
        logger.info(
            "Memory worker enabled (batch=%d turns, idle flush=%.0fs, poll=%.0fs)",
            settings.memory_extraction_batch_size,
            settings.memory_extraction_idle_flush_seconds,
            settings.memory_worker_poll_seconds,
        )
    else:
        logger.info("Memory worker disabled — run scripts/run_memory_worker.py separately")

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.app_name)

    # Stop the worker before disposing the engine it is using, otherwise its
    # next query runs against a closed pool.
    if worker_task is not None and worker_stop is not None:
        worker_stop.set()
        try:
            await asyncio.wait_for(worker_task, timeout=10.0)
        except asyncio.TimeoutError:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        except Exception as e:
            logger.warning("Memory worker shutdown error: %s", e)

    # MCP servers are child processes this application spawned, so they are
    # this application's to stop. Done before the memory cleanup below because
    # a hung server must not be able to hold up closing the database pool.
    try:
        from app.mcp.client import close_all as close_mcp_servers

        await asyncio.wait_for(close_mcp_servers(), timeout=10.0)
    except Exception as e:
        logger.warning("MCP shutdown error: %s", e)

    try:
        await memory_manager.cleanup()
        await voice_service.close()
        logger.info("Memory system cleaned up")
    except Exception as e:
        logger.warning("Memory cleanup error: %s", e)


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready FastAPI backend with Groq LLM integration",
    lifespan=lifespan
)

# Rate limiting sits outermost so throttled requests are rejected before any
# handler, database session, or third-party API call is touched.
app.add_middleware(RateLimitMiddleware)

# CORS — origins controlled via settings.allowed_origins (set ALLOWED_ORIGINS in .env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online",
        "model": settings.groq_model
    }


@app.get("/health")
async def health_check():
    """
    Liveness probe. Answers from this process only — no provider is called.

    This used to issue a real `chat_completion` against `settings.groq_model`.
    `render.yaml` points `healthCheckPath` here, so every platform probe spent
    a request from the same Groq RPM/TPM budget the users are competing for,
    forever, with no user attached to it — and the route is unauthenticated,
    so anyone who found it could drive that spend at will.

    A liveness probe answers one question: is this process able to serve? That
    is answerable without leaving the process. Provider reachability is a
    different question with a different cost, and it lives at /health/deep.
    """
    return {
        "status": "healthy",
        "api": "online",
        "model": settings.groq_model,
    }


# Provider probes are cached so that a platform health check pointed at the
# deep endpoint — or a dashboard polling it — cannot turn into a continuous
# drain on the Groq and Cohere quotas. One probe per minute is enough to
# notice an outage; more than that only costs money.
_DEEP_HEALTH_TTL_SECONDS = 60.0
_deep_health_cache: dict[str, object] = {"at": 0.0, "result": None}
_deep_health_lock = asyncio.Lock()


async def _probe_providers() -> dict:
    """Ask each provider whether it is reachable. Never raises."""
    groq_healthy, cohere_healthy, qdrant_healthy = await asyncio.gather(
        groq_service.health_check(),
        cohere_service.health_check(),
        qdrant_service.health_check(),
        return_exceptions=True,
    )

    def _ok(value) -> bool:
        return value is True

    providers = {
        "groq": "connected" if _ok(groq_healthy) else "disconnected",
        "cohere": "connected" if _ok(cohere_healthy) else "disconnected",
        "qdrant": "connected" if _ok(qdrant_healthy) else "disconnected",
    }
    all_ok = all(v == "connected" for v in providers.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "api": "online",
        "model": settings.groq_model,
        **providers,
    }


@app.get("/health/deep")
async def deep_health_check():
    """
    Provider reachability, memoized for ~60s.

    The lock is what makes the cache worth having: several probes arriving
    together would otherwise all miss and all issue their own round of provider
    calls, which is the exact amplification this endpoint exists to avoid.
    """
    now = time.monotonic()
    cached = _deep_health_cache.get("result")
    if cached is not None and now - float(_deep_health_cache["at"]) < _DEEP_HEALTH_TTL_SECONDS:
        return {**cached, "cached": True}

    async with _deep_health_lock:
        # Re-check under the lock: whoever held it may have just refreshed.
        now = time.monotonic()
        cached = _deep_health_cache.get("result")
        if cached is not None and now - float(_deep_health_cache["at"]) < _DEEP_HEALTH_TTL_SECONDS:
            return {**cached, "cached": True}

        result = await _probe_providers()
        _deep_health_cache["result"] = result
        _deep_health_cache["at"] = time.monotonic()

    return {**result, "cached": False}


# Include routers
from app.routes import agent_routes
from app.routes import auth_routes
from app.routes import livekit_routes

app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(agent_routes.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(livekit_routes.router, prefix="/api/v1/voice", tags=["voice"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development"
    )
