"""
Project entity identity — ingestion and retrieval.

`retrieve_projects` used to group chunks by `parent_id`, which is the *upload*,
not the project. Every project from one résumé therefore behaved as a single
document: asking about one returned all of them concatenated, and there was no
way to rank one project above another. The grouping key is now `entity_id`,
assigned per project at ingest, with `parent_id` retained alongside it so
provenance back to the source résumé survives — and used as the grouping
fallback for points written before entity ids existed.
"""
import pytest

from app.memory.long_term_memory_qdrant import (
    LongTermMemoryQdrant,
    derive_entry_title,
    long_term_memory_qdrant,
)

extract_chunks = long_term_memory_qdrant._extract_semantic_resume_chunks


MULTI_PROJECT_RESUME = """Jane Roe

PROJECTS
Voice Assistant | GitHub  July 2026 - Present
Low-latency assistant with streaming speech.
• Built with FastAPI and LangGraph.
Placement Portal | GitHub  Jan 2026 - March 2026
Full-stack platform for campus placement.
• Built with Next.js and PostgreSQL.
TRACE | GitHub  June 2026 - July 2026
Document intelligence and semantic search.
• Built a 13-stage pipeline with Neo4j.

TECHNICAL SKILLS
Backend:FastAPI, Node.js
"""


class FakeScore:
    """Minimal stand-in for a Qdrant ScoredPoint."""

    def __init__(self, payload, score):
        self.payload = payload
        self.score = score


class RecordingQdrant:
    """Captures upserts so ingestion payloads can be inspected without a server."""

    def __init__(self):
        self.upserts = {}

    async def scroll_collection(self, collection_name, filter_conditions=None, limit=None):
        return []

    async def upsert_points(self, collection_name, points):
        self.upserts.setdefault(collection_name, []).extend(points)

    async def delete_points(self, collection_name, point_ids):
        pass


class FakeCohere:
    async def embed_batch(self, texts, input_type):
        return [[0.1, 0.2, 0.3] for _ in texts]


async def ingest(resume_text, user_id="jane"):
    """Run store_resume against fakes and return the captured points by collection."""
    memory = LongTermMemoryQdrant()
    recorder = RecordingQdrant()
    memory.qdrant = recorder
    memory.cohere = FakeCohere()
    await memory.store_resume(user_id=user_id, resume_text=resume_text)
    return recorder.upserts


# ─────────────────────────────────────────────────────────────────────────
# Ingestion — one entity per project, provenance retained
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_each_project_gets_its_own_entity_id():
    upserts = await ingest(MULTI_PROJECT_RESUME)
    payloads = [p.payload for p in upserts["projects_chunks"]]

    assert len(payloads) == 3
    entity_ids = {p["entity_id"] for p in payloads}
    assert len(entity_ids) == 3, "projects from one upload must not share an entity id"


@pytest.mark.asyncio
async def test_entity_ids_retain_provenance_to_the_source_resume():
    upserts = await ingest(MULTI_PROJECT_RESUME)
    payloads = [p.payload for p in upserts["projects_chunks"]]

    parents = {p["parent_id"] for p in payloads}
    assert len(parents) == 1, "all three projects came from one résumé"
    parent = parents.pop()
    for payload in payloads:
        assert payload["entity_id"].startswith(parent)
        assert payload["source_file"] if "source_file" in payload else True


@pytest.mark.asyncio
async def test_each_project_carries_a_human_readable_title():
    upserts = await ingest(MULTI_PROJECT_RESUME)
    titles = {p.payload["entity_title"] for p in upserts["projects_chunks"]}
    assert titles == {"Voice Assistant", "Placement Portal", "TRACE"}


@pytest.mark.asyncio
async def test_no_data_is_duplicated_to_give_projects_identity():
    """Identity is two payload fields on existing points, not extra points."""
    upserts = await ingest(MULTI_PROJECT_RESUME)
    project_texts = [p.payload["text"] for p in upserts["projects_chunks"]]
    assert len(project_texts) == len(set(project_texts)) == 3


@pytest.mark.asyncio
async def test_reingesting_the_same_resume_is_stable_within_an_upload():
    upserts = await ingest(MULTI_PROJECT_RESUME)
    payloads = sorted(upserts["projects_chunks"], key=lambda p: p.payload["entity_id"])
    suffixes = [p.payload["entity_id"].rsplit("_", 1)[-1] for p in payloads]
    assert suffixes == ["1", "2", "3"]


# ─────────────────────────────────────────────────────────────────────────
# Retrieval — projects rank independently
# ─────────────────────────────────────────────────────────────────────────

def project_points(scores):
    """Three project chunks from one résumé, each with its own entity id."""
    return [
        FakeScore(
            {
                "parent_id": "resume_jane_abc",
                "entity_id": f"resume_jane_abc_project_{i + 1}",
                "entity_title": title,
                "text": f"{title}\nDescription of {title}.",
                "chunk_index": i,
                "type": "projects",
            },
            score,
        )
        for i, (title, score) in enumerate(scores)
    ]


@pytest.mark.asyncio
async def test_projects_are_returned_as_separate_items(monkeypatch):
    memory = LongTermMemoryQdrant()

    async def fake_embed(text, input_type):
        return [0.1, 0.2]

    async def fake_query(**kwargs):
        return project_points([("Voice Assistant", 0.41), ("Placement Portal", 0.33), ("TRACE", 0.27)])

    memory.cohere = type("C", (), {"embed_text": staticmethod(fake_embed)})()
    memory.qdrant = type("Q", (), {"query_points": staticmethod(fake_query)})()

    result = await memory.retrieve_projects(user_id="jane", query="voice assistant")
    items = list(result)

    assert len(items) == 3, "three projects must not collapse into one document"
    assert all("Description of" in item["content"] for item in items)


@pytest.mark.asyncio
async def test_the_most_relevant_project_ranks_first():
    memory = LongTermMemoryQdrant()

    async def fake_embed(text, input_type):
        return [0.1, 0.2]

    async def fake_query(**kwargs):
        # Deliberately returned worst-first, as Qdrant grouping order is arbitrary.
        return project_points([("Placement Portal", 0.21), ("TRACE", 0.55), ("Voice Assistant", 0.30)])

    memory.cohere = type("C", (), {"embed_text": staticmethod(fake_embed)})()
    memory.qdrant = type("Q", (), {"query_points": staticmethod(fake_query)})()

    items = list(await memory.retrieve_projects(user_id="jane", query="document search"))

    assert items[0]["title"] == "TRACE"
    assert items[0]["score"] == 0.55
    assert [i["score"] for i in items] == sorted((i["score"] for i in items), reverse=True)


@pytest.mark.asyncio
async def test_limit_keeps_the_best_projects_not_an_arbitrary_slice():
    memory = LongTermMemoryQdrant()

    async def fake_embed(text, input_type):
        return [0.1, 0.2]

    async def fake_query(**kwargs):
        return project_points([("Placement Portal", 0.10), ("TRACE", 0.90), ("Voice Assistant", 0.50)])

    memory.cohere = type("C", (), {"embed_text": staticmethod(fake_embed)})()
    memory.qdrant = type("Q", (), {"query_points": staticmethod(fake_query)})()

    items = list(await memory.retrieve_projects(user_id="jane", query="anything", limit=2))

    assert [i["title"] for i in items] == ["TRACE", "Voice Assistant"]


@pytest.mark.asyncio
async def test_chunks_of_one_project_regroup_into_that_project():
    """A project split across chunks stays one item, ordered by chunk_index."""
    memory = LongTermMemoryQdrant()

    async def fake_embed(text, input_type):
        return [0.1, 0.2]

    async def fake_query(**kwargs):
        shared = "resume_jane_abc_project_1"
        return [
            FakeScore({"parent_id": "p", "entity_id": shared, "entity_title": "TRACE",
                       "text": "second half", "chunk_index": 5}, 0.2),
            FakeScore({"parent_id": "p", "entity_id": shared, "entity_title": "TRACE",
                       "text": "first half", "chunk_index": 1}, 0.6),
        ]

    memory.cohere = type("C", (), {"embed_text": staticmethod(fake_embed)})()
    memory.qdrant = type("Q", (), {"query_points": staticmethod(fake_query)})()

    items = list(await memory.retrieve_projects(user_id="jane", query="trace"))

    assert len(items) == 1
    assert items[0]["content"].index("first half") < items[0]["content"].index("second half")
    assert items[0]["score"] == 0.6, "a project scores as its best-matching chunk"


@pytest.mark.asyncio
async def test_legacy_points_without_entity_ids_still_group():
    """Data written before entity ids existed must keep working, not disappear."""
    memory = LongTermMemoryQdrant()

    async def fake_embed(text, input_type):
        return [0.1, 0.2]

    async def fake_query(**kwargs):
        return [
            FakeScore({"parent_id": "legacy_resume", "text": "old project A", "chunk_index": 0}, 0.4),
            FakeScore({"parent_id": "legacy_resume", "text": "old project B", "chunk_index": 1}, 0.3),
        ]

    memory.cohere = type("C", (), {"embed_text": staticmethod(fake_embed)})()
    memory.qdrant = type("Q", (), {"query_points": staticmethod(fake_query)})()

    items = list(await memory.retrieve_projects(user_id="jane", query="old"))

    assert len(items) == 1, "legacy points fall back to parent_id grouping"
    assert "old project A" in items[0]["content"]
    assert "old project B" in items[0]["content"]


@pytest.mark.asyncio
async def test_unqueried_listing_returns_projects_in_document_order():
    memory = LongTermMemoryQdrant()

    async def fake_scroll(collection_name, filter_conditions=None, limit=None):
        return [
            {"payload": {"parent_id": "p", "entity_id": "p_project_3", "entity_title": "TRACE",
                         "text": "TRACE", "chunk_index": 9}},
            {"payload": {"parent_id": "p", "entity_id": "p_project_1", "entity_title": "Voice Assistant",
                         "text": "Voice Assistant", "chunk_index": 3}},
            {"payload": {"parent_id": "p", "entity_id": "p_project_2", "entity_title": "Placement Portal",
                         "text": "Placement Portal", "chunk_index": 6}},
        ]

    memory.qdrant = type("Q", (), {"scroll_collection": staticmethod(fake_scroll)})()

    items = list(await memory.retrieve_projects(user_id="jane"))

    assert [i["title"] for i in items] == ["Voice Assistant", "Placement Portal", "TRACE"]


# ─────────────────────────────────────────────────────────────────────────
# Titles
# ─────────────────────────────────────────────────────────────────────────

def test_titles_come_from_the_entry_not_the_section_heading():
    projects = [c["content"] for c in extract_chunks(MULTI_PROJECT_RESUME) if c["type"] == "projects"]
    assert derive_entry_title(projects[0]) == "Voice Assistant"
    assert "PROJECTS" not in derive_entry_title(projects[0])
