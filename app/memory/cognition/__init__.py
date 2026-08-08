"""
Cognition (L3): asynchronous memory formation.

Everything here runs *off* the request path, in a background worker. Extraction
calls an LLM and embedding calls a provider; neither cost may be paid inside a
turn, least of all a spoken one.

Consolidation, decay and archiving join this layer in Phase 5.
"""
from app.memory.cognition.embedder import EmbeddingPass, EmbedStats, embedding_pass
from app.memory.cognition.extractor import (
    Candidate,
    MemoryExtractor,
    memory_extractor,
    parse_extraction,
)
from app.memory.cognition.ingest import (
    IngestResult,
    MemoryIngestor,
    memory_ingestor,
)
from app.memory.cognition.maintenance import (
    MaintenanceStats,
    MemoryMaintenance,
    memory_maintenance,
)
from app.memory.cognition.summarizer import (
    ConversationSummarizer,
    SummaryStats,
    conversation_summarizer,
)
from app.memory.cognition.worker import MemoryWorker, WorkerStats, memory_worker

__all__ = [
    "Candidate",
    "ConversationSummarizer",
    "EmbedStats",
    "EmbeddingPass",
    "IngestResult",
    "MaintenanceStats",
    "MemoryExtractor",
    "MemoryIngestor",
    "MemoryMaintenance",
    "MemoryWorker",
    "SummaryStats",
    "WorkerStats",
    "conversation_summarizer",
    "embedding_pass",
    "memory_extractor",
    "memory_ingestor",
    "memory_maintenance",
    "memory_worker",
    "parse_extraction",
]
