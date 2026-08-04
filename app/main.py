"""
FastAPI main application entry point.
Configures async FastAPI server with Groq integration and hybrid memory system.
"""
import logging

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes memory systems and verifies API connectivity.
    """
    # Startup
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("Environment: %s | Groq model: %s", settings.environment, settings.groq_model)

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

    # LiveKit WebRTC readiness (Phase 0 — informational only)
    if settings.is_livekit_configured:
        logger.info("LiveKit WebRTC configured (url=%s)", settings.livekit_url)
    else:
        logger.info("LiveKit WebRTC not configured — WebSocket transport active")

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.app_name)
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
from app.routes import livekit_routes

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
