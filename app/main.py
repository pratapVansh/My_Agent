"""
FastAPI main application entry point.
Configures async FastAPI server with Groq integration and hybrid memory system.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
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
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Groq Model: {settings.groq_model}")

    # Optional observability setup
    if configure_langsmith():
        print(f"✓ LangSmith tracing enabled ({settings.langsmith_project})")
    else:
        print("- LangSmith tracing disabled")

    # Initialize memory system
    print("Initializing hybrid memory system...")
    try:
        await memory_manager.initialize()
        print("✓ Memory system initialized")
        print(f"  - Long-term: Qdrant ({settings.qdrant_url})")
        print(f"  - Embeddings: Cohere ({settings.cohere_model})")
        print(f"  - Short-term: PostgreSQL ({settings.postgres_host}:{settings.postgres_port})")
        print(f"  - Smart: mem0 (Cohere embeddings)")
    except Exception as e:
        print(f"✗ Warning: Memory initialization failed: {str(e)}")

    # Verify API connections
    print("\nVerifying API connections...")

    # Groq LLM
    is_healthy = await groq_service.health_check()
    if is_healthy:
        print("✓ Groq API connection verified")
    else:
        print("✗ Warning: Groq API health check failed")

    # Cohere embeddings
    cohere_healthy = await cohere_service.health_check()
    if cohere_healthy:
        print("✓ Cohere API connection verified")
    else:
        print("✗ Warning: Cohere API health check failed")

    # Qdrant vector database
    qdrant_healthy = await qdrant_service.health_check()
    if qdrant_healthy:
        print("✓ Qdrant connection verified")
    else:
        print("✗ Warning: Qdrant connection failed")

    yield

    # Shutdown
    print(f"Shutting down {settings.app_name}")
    try:
        await memory_manager.cleanup()
        await voice_service.close()
        print("✓ Memory system cleaned up")
    except Exception as e:
        print(f"Warning: Memory cleanup error: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready FastAPI backend with Groq LLM integration",
    lifespan=lifespan
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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
from app.routes import voice_routes

app.include_router(agent_routes.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(voice_routes.router, prefix="/api/v1/voice", tags=["voice"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development"
    )
