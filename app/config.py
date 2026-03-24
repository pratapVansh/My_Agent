"""
Application configuration using Pydantic Settings.
Loads environment variables for Groq API and application settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


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
    deepgram_utterance_end_ms: int = 1000
    deepgram_vad_events: bool = True

    # Voice: Cartesia TTS
    cartesia_api_key: Optional[str] = None
    cartesia_model_id: str = "sonic-2"
    cartesia_voice_id: str = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    cartesia_version: str = "2024-06-10"
    cartesia_sample_rate: int = 24000
    cartesia_streaming_enabled: bool = False

    # Voice Agent Optimization Feature Flags
    streaming_stt_enabled: bool = True
    streaming_tts_enabled: bool = False
    parallel_workflow_enabled: bool = False
    binary_audio_enabled: bool = False
    vad_enabled: bool = True

    # Observability: LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "my-agent"
    langsmith_endpoint: Optional[str] = None
    langsmith_tracing_enabled: bool = False

    @property
    def postgres_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
settings = Settings()
