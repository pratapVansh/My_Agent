"""
Typed résumé section retrieval — experience, education, achievements.

These sections were parsed and stored correctly but had no way out of memory.
The profile agent could reach skills and projects through their own collections,
and everything else only through the first 1500 characters of `get_resume`. So
"What company did I intern at?" was answered "I don't have information about
that" while the internship sat in an `experience` chunk in Qdrant.

Retrieval scrolls and filters in Python rather than issuing a filtered vector
search: `semantic_type` carries no payload index, and Qdrant rejects a filter on
an unindexed field with a 400 rather than falling back to a scan. A résumé is a
couple of dozen chunks, so this is exact and cheap — and a question like "what
is my CGPA" wants the whole education section, not a fuzzy top-k.
"""
import pytest

from app.memory.long_term_memory_qdrant import LongTermMemoryQdrant
from app.memory.retrieval_result import RetrievalStatus


def point(semantic_type, text, *, chunk_index=0, tags=None, parent="resume_v2",
          uploaded_at="2026-08-09T00:00:00"):
    return {
        "id": f"{parent}_{chunk_index}",
        "payload": {
            "user_id": "vansh",
            "parent_id": parent,
            "uploaded_at": uploaded_at,
            "semantic_type": semantic_type,
            "text": text,
            "chunk_index": chunk_index,
            "tags": tags or [],
        },
    }


RESUME_POINTS = [
    point("name", "Vansh Pratap Singh", chunk_index=0),
    point("education",
          "Education\nRajiv Gandhi Institute of Petroleum Technology\n"
          "B.Tech. in Information Technology (CGPA: 8.80 / 10) Aug 2023-Present",
          chunk_index=1),
    point("experience",
          "Experience\nDrUpsc (EdTech) May 2026 - June 2026\nFull Stack Developer Intern",
          chunk_index=2),
    point("experience",
          "Department of Computer Science, BHU June 2025 - July 2025\nResearch Intern",
          chunk_index=3),
    point("projects", "Projects\nTRACE | GitHub", chunk_index=4),
    point("other", "Achievements\nMerit-cum-Means Scholarship", chunk_index=5,
          tags=["achievements"]),
    point("other", "contact details", chunk_index=6, tags=["general"]),
]


def memory_with(points, *, fail=False):
    class FakeQdrant:
        async def scroll_collection(self, collection_name, filter_conditions=None, limit=None):
            if fail:
                raise RuntimeError("qdrant unreachable")
            return points

    memory = LongTermMemoryQdrant()
    memory.qdrant = FakeQdrant()
    return memory


# ─────────────────────────────────────────────────────────────────────────
# The sections that had no tool
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_internship_details_are_retrievable():
    """The reported failure: "What company did I intern at?" → "I don't have it"."""
    result = await memory_with(RESUME_POINTS).retrieve_section("vansh", "experience")
    joined = "\n".join(item["content"] for item in result)

    assert result.status == RetrievalStatus.OK
    assert "DrUpsc" in joined
    assert "BHU" in joined


@pytest.mark.asyncio
async def test_cgpa_and_college_are_retrievable():
    result = await memory_with(RESUME_POINTS).retrieve_section("vansh", "education")
    joined = "\n".join(item["content"] for item in result)

    assert "CGPA: 8.80" in joined
    assert "Rajiv Gandhi Institute" in joined
    assert "Information Technology" in joined


@pytest.mark.asyncio
async def test_achievements_are_identified_by_tag_not_type():
    """Achievements are stored as `other`; the tag is what distinguishes them."""
    result = await memory_with(RESUME_POINTS).retrieve_section("vansh", "achievements")
    joined = "\n".join(item["content"] for item in result)

    assert "Merit-cum-Means" in joined
    assert "contact details" not in joined


@pytest.mark.asyncio
async def test_sections_come_back_in_document_order():
    result = await memory_with(RESUME_POINTS).retrieve_section("vansh", "experience")
    assert "DrUpsc" in list(result)[0]["content"]
    assert "BHU" in list(result)[1]["content"]


@pytest.mark.asyncio
async def test_each_section_returns_only_its_own_content():
    memory = memory_with(RESUME_POINTS)
    education = "\n".join(i["content"] for i in await memory.retrieve_section("vansh", "education"))
    assert "DrUpsc" not in education
    assert "TRACE" not in education


# ─────────────────────────────────────────────────────────────────────────
# Empty and failed retrieval stay distinguishable
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_empty_resume_reports_no_data_rather_than_erroring():
    result = await memory_with([]).retrieve_section("vansh", "experience")
    assert result.status == RetrievalStatus.NO_DATA
    assert not list(result)


@pytest.mark.asyncio
async def test_a_resume_without_that_section_reports_no_data():
    only_skills = [point("skills", "Python, FastAPI", chunk_index=0)]
    result = await memory_with(only_skills).retrieve_section("vansh", "experience")
    assert result.status == RetrievalStatus.NO_DATA


@pytest.mark.asyncio
async def test_a_lookup_failure_is_an_error_not_an_absence():
    """
    "We could not find out" must never be reported as "there is nothing" — that
    is the state in which a model invents a plausible internship.
    """
    result = await memory_with(RESUME_POINTS, fail=True).retrieve_section("vansh", "education")
    assert result.status == RetrievalStatus.ERROR


@pytest.mark.asyncio
async def test_an_unknown_section_is_refused_quietly():
    result = await memory_with(RESUME_POINTS).retrieve_section("vansh", "hobbies")
    assert result.status == RetrievalStatus.NO_DATA


# ─────────────────────────────────────────────────────────────────────────
# Only the current résumé answers
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_superseded_resume_version_does_not_answer():
    mixed = [
        point("experience", "OldCorp Intern", chunk_index=0,
              parent="resume_v1", uploaded_at="2026-01-01T00:00:00"),
        point("experience", "DrUpsc Intern", chunk_index=0,
              parent="resume_v2", uploaded_at="2026-08-09T00:00:00"),
    ]
    joined = "\n".join(
        i["content"] for i in await memory_with(mixed).retrieve_section("vansh", "experience")
    )
    assert "DrUpsc" in joined
    assert "OldCorp" not in joined


# ─────────────────────────────────────────────────────────────────────────
# The agent exposes them
# ─────────────────────────────────────────────────────────────────────────

def test_the_profile_agent_declares_the_new_retrieval_tools():
    import inspect

    from app.agents.profile_agent import ProfileAgent

    source = inspect.getsource(ProfileAgent)
    for tool in ("get_experience", "get_education", "get_achievements"):
        assert f'"{tool}"' in source, f"{tool} is not declared as a tool"
        assert "Scope.PROFILE_READ.value" in source


def test_the_agent_is_told_not_to_clarify_instead_of_reporting_an_empty_result():
    import inspect

    from app.agents.profile_agent import ProfileAgent

    prompt = inspect.getsource(ProfileAgent).lower()
    assert "answer first, clarify last" in prompt
    assert "does not mean the question was unclear" in prompt
