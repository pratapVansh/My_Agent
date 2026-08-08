"""
Retrieval (L2): hybrid candidate generation, ranking, and context assembly.

Synchronous and on the request path, unlike the cognition layer (L3). Nothing
here writes; nothing here calls an LLM.
"""
from app.memory.retrieval.assembler import (
    AssembledContext,
    ContextAssembler,
    context_assembler,
)
from app.memory.retrieval.engine import (
    RetrievalEngine,
    RetrievalResultSet,
    ScoredRecord,
)
from app.memory.retrieval.working import (
    WorkingMemory,
    WorkingMemoryBuilder,
    working_memory_builder,
)
from app.memory.retrieval.trace import (
    ChannelResult,
    DroppedItem,
    RetrievalTrace,
    SelectedItem,
)

__all__ = [
    "AssembledContext",
    "ChannelResult",
    "ContextAssembler",
    "DroppedItem",
    "RetrievalEngine",
    "RetrievalResultSet",
    "RetrievalTrace",
    "ScoredRecord",
    "SelectedItem",
    "WorkingMemory",
    "WorkingMemoryBuilder",
    "context_assembler",
    "working_memory_builder",
]
