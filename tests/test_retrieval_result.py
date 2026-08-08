"""
Tests for typed long-term retrieval results (Phase 0).

The behaviour under test is the distinction the old `List | "NO_DATA"` union
could not express: a lookup that *failed* is not the same as a user who has
nothing stored. Conflating them let a Qdrant timeout reach the prompt as
"status: OK, zero results", which suppressed both the no-data hint and the
refusal policy — the exact conditions under which a model invents an answer.

See docs/MEMORY_ARCHITECTURE.md §1.10.
"""
import pytest

from app.memory.memory_manager import memory_manager
from app.memory.retrieval_result import RetrievalResult, RetrievalStatus


format_context = memory_manager.format_context_for_prompt


# ─────────────────────────────────────────────────────────────────────────
# RetrievalResult behaves as a sequence
# ─────────────────────────────────────────────────────────────────────────

def test_ok_result_is_truthy_and_iterable():
    result = RetrievalResult.ok([{"content": "Python"}, {"content": "Rust"}])
    assert bool(result) is True
    assert len(result) == 2
    assert [item["content"] for item in result] == ["Python", "Rust"]
    assert result[0]["content"] == "Python"


def test_empty_results_are_falsy():
    assert not RetrievalResult.no_data()
    assert not RetrievalResult.error()
    assert not RetrievalResult.ok([])


def test_list_conversion_yields_plain_dicts():
    result = RetrievalResult.fallback([{"content": "from resume"}])
    assert list(result) == [{"content": "from resume"}]


# ─────────────────────────────────────────────────────────────────────────
# Status semantics
# ─────────────────────────────────────────────────────────────────────────

def test_status_string_values_are_unchanged_for_backward_compatibility():
    """
    Cached context dictionaries and existing comparisons rely on these exact
    strings. Changing them is a breaking change, not a rename.
    """
    assert RetrievalStatus.OK.value == "OK"
    assert RetrievalStatus.FALLBACK.value == "FALLBACK"
    assert RetrievalStatus.NO_DATA.value == "NO_DATA"
    assert RetrievalStatus.ERROR.value == "ERROR"


def test_status_compares_equal_to_its_string():
    assert RetrievalStatus.NO_DATA == "NO_DATA"


def test_only_no_data_is_a_trustworthy_absence():
    assert RetrievalResult.no_data().is_trustworthy_absence is True
    # A failed lookup tells us nothing about whether data exists.
    assert RetrievalResult.error().is_trustworthy_absence is False
    assert RetrievalResult.ok([{"content": "x"}]).is_trustworthy_absence is False


def test_error_with_items_is_never_usable():
    assert RetrievalResult.ok([{"content": "x"}]).is_usable is True
    assert RetrievalResult.no_data().is_usable is False
    assert RetrievalResult(items=[{"content": "x"}],
                           status=RetrievalStatus.ERROR).is_usable is False


# ─────────────────────────────────────────────────────────────────────────
# Prompt construction distinguishes "absent" from "unknown"
# ─────────────────────────────────────────────────────────────────────────

def test_error_status_does_not_claim_data_is_absent():
    """Asserting 'no skills found' after a failed lookup states a falsehood."""
    out = format_context({"long_term": {
        "skills": [], "projects": [],
        "skills_status": "ERROR", "projects_status": "OK",
    }})
    assert "No skills data found" not in out
    assert "treat skills as unknown, not absent" in out


def test_error_status_arms_the_refusal_policy():
    """
    The regression this guards: a failed search previously reached the prompt
    with no status at all, so the model had neither data nor instruction.
    """
    out = format_context({"long_term": {
        "skills_status": "ERROR", "projects_status": "OK",
    }})
    assert "I don't have information about that." in out


def test_projects_error_reported_independently():
    out = format_context({"long_term": {
        "skills_status": "OK", "projects_status": "ERROR",
    }})
    assert "treat projects as unknown, not absent" in out
    assert "treat skills as unknown" not in out


def test_no_data_still_states_absence_plainly():
    out = format_context({"long_term": {
        "skills_status": "NO_DATA", "projects_status": "NO_DATA",
    }})
    assert "No skills data found in vector memory." in out
    assert "unknown, not absent" not in out


def test_ok_statuses_emit_no_retrieval_hints():
    out = format_context({"long_term": {
        "skills": [{"content": "Python"}],
        "projects": [{"content": "My_Agent"}],
        "skills_status": "OK", "projects_status": "OK",
    }})
    assert "Retrieval Status:" not in out
    assert "I don't have information about that." not in out


# ─────────────────────────────────────────────────────────────────────────
# search_all boundary contract
# ─────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_all_reports_error_status_when_retrieval_raises(monkeypatch):
    """
    A total search failure must surface as ERROR on both keys. Previously the
    handler returned a dict with the status keys missing entirely.
    """
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt

    async def boom(*args, **kwargs):
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(lt, "retrieve_skills", boom)
    monkeypatch.setattr(lt, "retrieve_projects", boom)
    monkeypatch.setattr(lt, "retrieve_resume", boom)

    result = await lt.search_all(user_id="vansh", query="skills")

    assert result["skills_status"] == "ERROR"
    assert result["projects_status"] == "ERROR"
    assert result["skills"] == []
    assert result["projects"] == []


@pytest.mark.asyncio
async def test_search_all_propagates_error_from_a_single_lookup(monkeypatch):
    """
    The original bug: retrieve_skills() swallowed its exception and returned
    [], which search_all then labelled "OK" — a failed lookup masquerading as
    a user with no skills.
    """
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt

    async def failing_skills(*args, **kwargs):
        return RetrievalResult.error()

    async def ok_projects(*args, **kwargs):
        return RetrievalResult.ok([{"content": "My_Agent"}])

    async def no_resume(*args, **kwargs):
        return None

    monkeypatch.setattr(lt, "retrieve_skills", failing_skills)
    monkeypatch.setattr(lt, "retrieve_projects", ok_projects)
    monkeypatch.setattr(lt, "retrieve_resume", no_resume)

    result = await lt.search_all(user_id="vansh", query="skills")

    assert result["skills_status"] == "ERROR"
    assert result["projects_status"] == "OK"
    assert result["projects"] == [{"content": "My_Agent"}]


@pytest.mark.asyncio
async def test_search_all_does_not_run_fallback_for_a_failed_lookup(monkeypatch):
    """
    Fallback means "the dedicated collection is empty". Running it after an
    ERROR would report FALLBACK, asserting something we do not know.
    """
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt

    called = []

    async def failing(*args, **kwargs):
        return RetrievalResult.error()

    async def no_resume(*args, **kwargs):
        return None

    async def track_fallback(*args, **kwargs):
        called.append(args)
        return [{"content": "recovered"}]

    monkeypatch.setattr(lt, "retrieve_skills", failing)
    monkeypatch.setattr(lt, "retrieve_projects", failing)
    monkeypatch.setattr(lt, "retrieve_resume", no_resume)
    monkeypatch.setattr(lt, "_fallback_resume_search", track_fallback)

    result = await lt.search_all(user_id="vansh", query="skills")

    assert called == []
    assert result["skills_status"] == "ERROR"


@pytest.mark.asyncio
async def test_search_all_uses_fallback_when_collection_is_genuinely_empty(monkeypatch):
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt

    async def empty(*args, **kwargs):
        return RetrievalResult.no_data()

    async def no_resume(*args, **kwargs):
        return None

    async def recovered(*args, **kwargs):
        return [{"content": "Python, FastAPI"}]

    monkeypatch.setattr(lt, "retrieve_skills", empty)
    monkeypatch.setattr(lt, "retrieve_projects", empty)
    monkeypatch.setattr(lt, "retrieve_resume", no_resume)
    monkeypatch.setattr(lt, "_fallback_resume_search", recovered)

    result = await lt.search_all(user_id="vansh", query="skills")

    assert result["skills_status"] == "FALLBACK"
    assert result["projects_status"] == "FALLBACK"
    assert result["skills"] == [{"content": "Python, FastAPI"}]


@pytest.mark.asyncio
async def test_search_all_returns_no_data_when_fallback_also_finds_nothing(monkeypatch):
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt

    async def empty(*args, **kwargs):
        return RetrievalResult.no_data()

    async def no_resume(*args, **kwargs):
        return None

    async def nothing(*args, **kwargs):
        return []

    monkeypatch.setattr(lt, "retrieve_skills", empty)
    monkeypatch.setattr(lt, "retrieve_projects", empty)
    monkeypatch.setattr(lt, "retrieve_resume", no_resume)
    monkeypatch.setattr(lt, "_fallback_resume_search", nothing)

    result = await lt.search_all(user_id="vansh", query="skills")

    assert result["skills_status"] == "NO_DATA"
    assert result["projects_status"] == "NO_DATA"


@pytest.mark.asyncio
async def test_search_all_returns_plain_lists_not_result_objects(monkeypatch):
    """
    format_context_for_prompt isinstance-checks for `list`. Leaking a
    RetrievalResult into this slot would silently drop the skills section.
    """
    from app.memory.long_term_memory_qdrant import long_term_memory_qdrant as lt

    async def skills(*args, **kwargs):
        return RetrievalResult.ok([{"content": "Python"}])

    async def projects(*args, **kwargs):
        return RetrievalResult.ok([{"content": "My_Agent"}])

    async def no_resume(*args, **kwargs):
        return None

    monkeypatch.setattr(lt, "retrieve_skills", skills)
    monkeypatch.setattr(lt, "retrieve_projects", projects)
    monkeypatch.setattr(lt, "retrieve_resume", no_resume)

    result = await lt.search_all(user_id="vansh", query="skills")

    assert type(result["skills"]) is list
    assert type(result["projects"]) is list

    # And the formatter must actually render them.
    out = format_context({"long_term": result})
    assert "User Skills:" in out
    assert "User Projects:" in out
