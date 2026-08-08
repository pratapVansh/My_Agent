"""
FastAPI main application entry point.
Configures async FastAPI server with Groq integration and hybrid memory system.
"""
import asyncio
import logging
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
    Health check endpoint.
    Verifies FastAPI and Groq API connectivity.
    """
    groq_healthy = await groq_service.health_check()

    return {
        "status": "healthy" if groq_healthy else "degraded",
        "api": "online",
        "groq": "connected" if groq_healthy else "disconnected",
        "model": settings.groq_model
    }


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
