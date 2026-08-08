"""
LLM extraction of durable memories from conversation.

This is the component that changes the system's character. The replaced design
embedded every utterance verbatim, which made semantic search *worse* as
history grew: after a year, "what are my goals?" competes against thousands of
embedded fragments of small talk. Extraction distils instead of accumulating.

Batched over several turns on purpose. A single turn frequently lacks the
context to state a self-contained fact — "yeah, that one" means nothing alone —
and batching is also roughly five times cheaper.

See docs/MEMORY_ARCHITECTURE.md §3.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import json
import logging
import re

from app.memory.kinds import MemoryKind
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

# Kinds the extractor may produce. `document`, `procedural` and `relation` come
# from other sources (uploads, tool outcomes, graph derivation), so allowing
# them here would let the model invent provenance it does not have.
EXTRACTABLE_KINDS = frozenset({
    MemoryKind.IDENTITY,
    MemoryKind.PREFERENCE,
    MemoryKind.SEMANTIC,
    MemoryKind.GOAL,
    MemoryKind.TASK,
    MemoryKind.EPISODIC,
})

# Guard against a runaway response filling the store from one conversation.
MAX_CANDIDATES_PER_BATCH = 12

_SYSTEM_PROMPT = """You extract durable facts about a user from conversation transcripts.

Return ONLY a JSON object of this exact shape, with no preamble and no code fences:
{"memories": [{"kind": "...", "content": "...", "importance": 0.0-1.0, "confidence": 0.0-1.0, "dedup_key": null}]}

kind must be one of: identity, preference, semantic, goal, task, episodic
  identity   - stable facts about who they are: name, role, school, location, profile links
  preference - how they want the assistant to behave: tone, format, language, detail level
  semantic   - durable facts about them: skills, habits, interests, relationships
  goal       - something they are working towards, often with a deadline
  task       - a specific actionable item they intend to do
  episodic   - a notable thing that happened, worth recalling later

RULES — these matter more than completeness:
1. content MUST be one self-contained sentence in the third person about the user.
   It will be stored alone and read months later with no surrounding context.
   Good: "The user is targeting a backend SDE internship for summer 2027."
   Bad:  "wants that one" / "yes" / "the internship"
2. Extract ONLY durable facts. Skip small talk, greetings, one-off questions,
   and anything the assistant said about itself. Returning an empty list is a
   correct and common answer.
3. Do NOT infer beyond what was actually said. No guessing.
4. NEVER extract passwords, tokens, keys, card numbers, or any credential.
5. dedup_key: set it ONLY for facts where the user can have exactly one current
   value, formatted as "profile:<attribute>" — for example "profile:name",
   "profile:role", "profile:timezone". A newer value then replaces the old one.
   Leave it null for everything else, including goals and episodes.
6. importance: 0.9 identity, 0.85 preference, 0.8 goal, 0.6 semantic,
   0.5 task, 0.4 episodic. Adjust modestly for how central it seems.
7. confidence: how certain you are the user actually stated this. Use below 0.8
   when you are inferring rather than quoting.
"""


@dataclass
class Candidate:
    """One proposed memory, before governance and de-duplication."""

    kind: MemoryKind
    content: str
    importance: float = 0.5
    confidence: float = 0.8
    dedup_key: Optional[str] = None
    structured: Dict[str, Any] = field(default_factory=dict)


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def parse_extraction(raw: str) -> List[Candidate]:
    """
    Parse the model's response into candidates, discarding anything malformed.

    Tolerant by necessity: this runs unattended in a worker, so one badly
    formatted response must cost the batch rather than crash the pipeline. Each
    candidate is validated independently, so a single bad entry does not
    discard its well-formed siblings.
    """
    text = (raw or "").strip()
    if not text:
        return []

    payload: Any = None
    for attempt in (text, _first_json_object(text)):
        if not attempt:
            continue
        try:
            payload = json.loads(attempt)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(payload, dict):
        logger.warning("Extractor returned non-JSON output: %.160s", text)
        return []

    entries = payload.get("memories")
    if not isinstance(entries, list):
        return []

    candidates: List[Candidate] = []
    for entry in entries[:MAX_CANDIDATES_PER_BATCH]:
        if not isinstance(entry, dict):
            continue

        content = str(entry.get("content") or "").strip()
        if len(content) < 10:
            # Too short to be a self-contained statement; almost always a
            # fragment the model failed to expand.
            continue

        try:
            kind = MemoryKind(str(entry.get("kind", "")).strip().lower())
        except ValueError:
            continue
        if kind not in EXTRACTABLE_KINDS:
            continue

        dedup_key = entry.get("dedup_key")
        dedup_key = str(dedup_key).strip().lower() if dedup_key else None
        if dedup_key and not dedup_key.startswith("profile:"):
            # The model occasionally invents its own namespace; an unnamespaced
            # key would collide across unrelated attributes.
            dedup_key = None

        candidates.append(Candidate(
            kind=kind,
            content=content,
            importance=_clamp(entry.get("importance"), 0.5),
            confidence=_clamp(entry.get("confidence"), 0.8),
            dedup_key=dedup_key,
        ))

    return candidates


def _first_json_object(text: str) -> Optional[str]:
    """Extract the first {...} block, for responses wrapped in prose or fences."""
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def render_transcript(turns: Sequence[Dict[str, Any]], *, max_chars: int = 400) -> str:
    """Format turns for the extraction prompt."""
    lines: List[str] = []
    for turn in turns:
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:max_chars]}")
    return "\n".join(lines)


class MemoryExtractor:
    """Turns a batch of conversation turns into candidate memories."""

    def __init__(self, llm=None):
        # Injected so extraction is testable without a Groq key.
        self.llm = llm or groq_service

    async def extract(
        self, owner_id: str, turns: Sequence[Dict[str, Any]]
    ) -> List[Candidate]:
        transcript = render_transcript(turns)
        if not transcript.strip():
            return []

        response = await self.llm.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Transcript:\n{transcript}"},
            ],
            # Extraction is a reading task, not a creative one.
            temperature=0.1,
            max_tokens=900,
        )
        candidates = parse_extraction(response.get("content", ""))

        logger.info(
            "Extracted %d candidate memories for owner=%s from %d turns",
            len(candidates), owner_id, len(turns),
        )
        return candidates


# Singleton instance
memory_extractor = MemoryExtractor()
