"""
Application configuration using Pydantic Settings.
Loads environment variables for Groq API and application settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Groq API Configuration (MANDATORY)
    groq_api_key: str
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
    # ChromaDB - Long-term memory
    chroma_persist_dir: str = "./data/chroma"

    # PostgreSQL - Short-term memory
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "my_agent_db"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # mem0 - Smart memory
    mem0_api_key: Optional[str] = None

    # Cohere Configuration
    cohere_api_key: str
    cohere_model: str = "embed-english-v3.0"
    cohere_embedding_dimension: int = 1024

    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: str
    qdrant_timeout: int = 30

    # Text Chunking Configuration
    chunk_size: int = 400
    chunk_overlap: int = 50

    # Job Search Tool (Tavily)
    tavily_api_key: Optional[str] = None

    # Voice: Deepgram STT
    deepgram_api_key: Optional[str] = None
    deepgram_model: str = "nova-2"
    deepgram_streaming_enabled: bool = False
    deepgram_interim_results: bool = True
    deepgram_utterance_end_ms: int = 1000  # Deepgram API strictly requires a minimum of 1000ms
    deepgram_vad_events: bool = True

    # Voice: Cartesia TTS
    cartesia_api_key: Optional[str] = None
    cartesia_model_id: str = "sonic-2"
    cartesia_voice_id: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    cartesia_version: str = "2024-06-10"
    cartesia_sample_rate: int = 24000
    cartesia_streaming_enabled: bool = False

    # LiveKit WebRTC (Phase 0 — settings only, not active yet)
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None

    # Voice Agent Optimization Feature Flags
    streaming_stt_enabled: bool = True
    streaming_tts_enabled: bool = False
    parallel_workflow_enabled: bool = True  # Run memory + planner concurrently (~200-300ms gain)
    binary_audio_enabled: bool = False
    vad_enabled: bool = True

    # CORS — restrict in production (comma-separated origins in .env)
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
