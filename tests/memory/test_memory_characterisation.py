"""
Characterisation tests for the memory system — Phase 0 refactoring safety net.

These pin the *current* behaviour of the pure functions that the memory redesign
moves or rewrites. They are deliberately descriptive rather than aspirational:
where today's behaviour is odd (sentinel strings leaking into the prompt,
silent de-duplication of a name against a body chunk), the test records the
oddity so a refactor cannot change it by accident.

When a later phase intentionally changes one of these behaviours, the
corresponding test should be updated in the same commit — that diff is the
record of what changed and why.

See docs/MEMORY_ARCHITECTURE.md.
"""
import pytest

from app.memory.memory_manager import memory_manager
from app.memory.long_term_memory_qdrant import long_term_memory_qdrant


format_context = memory_manager.format_context_for_prompt
format_history = memory_manager._format_chat_history_for_prompt
extract_chunks = long_term_memory_qdrant._extract_semantic_resume_chunks


# ─────────────────────────────────────────────────────────────────────────
# format_context_for_prompt — section ordering and truncation safety
# ─────────────────────────────────────────────────────────────────────────

FULL_CONTEXT = {
    "profile_facts": [
        {"key": "name", "value": "Vansh"},
        {"key": "preferred_tone", "value": "concise"},
    ],
    "episodes": [
        {"agent_used": "job", "user_summary": "asked for ML jobs",
         "agent_summary": "returned 5 listings"},
    ],
    "chat_history": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ],
    "preferences": [{"memory": "interested in distributed systems"}],
    "long_term": {
        "skills": [{"content": "Python, FastAPI"}],
        "projects": [{"content": "My_Agent — a personal assistant"}],
        "resume": {"content": "Vansh Pratap Singh, backend engineer."},
        "skills_status": "OK",
        "projects_status": "OK",
    },
}


def test_empty_context_produces_empty_string():
    assert format_context({}) == ""


def test_all_sections_render_when_present():
    out = format_context(FULL_CONTEXT)

    assert "User Profile Facts:" in out
    assert "- name: Vansh" in out
    assert "Recent Activity:" in out
    assert "[job] asked for ML jobs → returned 5 listings" in out
    assert "Recent Conversation:" in out
    assert "User Preferences & Interests:" in out
    assert "User Skills:" in out
    assert "User Projects:" in out
    assert "User Resume:" in out


def test_section_priority_order_is_stable():
    """
    The whole truncation-safety design rests on this ordering: sections are
    written highest-priority first so a downstream character cap only ever
    drops the tail. If this order changes, inject_memory_context's guarantee
    that profile facts survive truncation silently breaks.
    """
    out = format_context(FULL_CONTEXT)
    positions = [
        out.index("User Profile Facts:"),
        out.index("Recent Activity:"),
        out.index("Recent Conversation:"),
        out.index("User Preferences & Interests:"),
        out.index("User Skills:"),
        out.index("User Projects:"),
        out.index("User Resume:"),
    ]
    assert positions == sorted(positions)


def test_profile_facts_come_first_and_resume_last():
    out = format_context(FULL_CONTEXT)
    assert out.startswith("User Profile Facts:")
    assert out.index("User Resume:") == max(
        out.index(marker)
        for marker in ("User Profile Facts:", "Recent Activity:", "User Resume:")
    )


def test_sections_are_separated_by_blank_lines():
    out = format_context(FULL_CONTEXT)
    assert "\n\nRecent Activity:" in out


def test_profile_facts_without_value_are_skipped():
    out = format_context({"profile_facts": [
        {"key": "name", "value": "Vansh"},
        {"key": "empty", "value": ""},
        {"key": "missing"},
    ]})
    assert "- name: Vansh" in out
    assert "empty" not in out
    assert "missing" not in out


def test_episodes_without_any_summary_are_skipped():
    out = format_context({"episodes": [{"agent_used": "job"}]})
    assert out == ""


def test_resume_is_capped_at_300_characters():
    long_resume = "x" * 1000
    out = format_context({"long_term": {"resume": {"content": long_resume}}})
    body = out.split("User Resume:\n", 1)[1]
    assert len(body) == 300


def test_projects_are_capped_at_200_characters_each():
    out = format_context({"long_term": {"projects": [{"content": "y" * 500}]}})
    assert "y" * 200 in out
    assert "y" * 201 not in out


def test_only_first_five_skills_and_three_projects_render():
    out = format_context({"long_term": {
        "skills": [{"content": f"skill{i}"} for i in range(10)],
        "projects": [{"content": f"project{i}"} for i in range(10)],
    }})
    assert "skill4" in out and "skill5" not in out
    assert "project2" in out and "project3" not in out


# ─── sentinel-string handling (the behaviour Phase 0 replaces with types) ──

def test_no_data_sentinel_string_does_not_crash_and_renders_nothing():
    """
    retrieve_skills() returns either a list or the bare string "NO_DATA", so
    format_context must isinstance-guard. This test pins that the sentinel is
    tolerated and produces no skills section.
    """
    out = format_context({"long_term": {"skills": "NO_DATA", "projects": "NO_DATA"}})
    assert "User Skills:" not in out
    assert "User Projects:" not in out


def test_no_data_status_emits_retrieval_status_lines():
    out = format_context({"long_term": {
        "skills": [], "projects": [],
        "skills_status": "NO_DATA", "projects_status": "NO_DATA",
    }})
    assert "Retrieval Status: No skills data found in vector memory." in out
    assert "Retrieval Status: No projects data found in vector memory." in out


def test_both_statuses_no_data_adds_the_refusal_policy_line():
    out = format_context({"long_term": {
        "skills_status": "NO_DATA", "projects_status": "NO_DATA",
    }})
    assert "I don't have information about that." in out


def test_single_no_data_status_does_not_add_the_policy_line():
    out = format_context({"long_term": {
        "skills_status": "NO_DATA", "projects_status": "OK",
    }})
    assert "I don't have information about that." not in out


def test_fallback_status_emits_provenance_notes():
    out = format_context({"long_term": {
        "skills_status": "FALLBACK", "projects_status": "FALLBACK",
    }})
    assert "Skills retrieved from resume text" in out
    assert "Projects retrieved from resume text" in out


# ─────────────────────────────────────────────────────────────────────────
# _format_chat_history_for_prompt — trimming and budget
# ─────────────────────────────────────────────────────────────────────────

def test_empty_history_renders_nothing():
    assert format_history([]) == ""


def test_history_with_only_blank_content_renders_nothing():
    assert format_history([{"role": "user", "content": "   "}]) == ""


def test_roles_map_to_user_and_assistant_labels():
    out = format_history([
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "system", "content": "note"},
    ])
    assert "User: question" in out
    assert "Assistant: answer" in out
    # Anything that is not "user" is labelled Assistant.
    assert "Assistant: note" in out


def test_only_the_most_recent_messages_are_kept():
    history = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
    out = format_history(history, max_messages=3)
    assert "msg19" in out and "msg17" in out
    assert "msg16" not in out


def test_long_message_is_truncated_with_ellipsis():
    """Budget is max_message_chars inclusive of the three-character ellipsis."""
    out = format_history(
        [{"role": "user", "content": "z" * 500}], max_message_chars=50
    )
    assert "..." in out
    assert "z" * 47 + "..." in out
    assert "z" * 48 not in out


def test_global_char_budget_drops_oldest_lines_first():
    history = [{"role": "user", "content": f"message number {i}"} for i in range(50)]
    out = format_history(history, max_messages=50, max_chars=100)
    assert len(out) < 200
    assert "message number 49" in out
    assert "message number 0" not in out


def test_history_block_carries_its_header():
    out = format_history([{"role": "user", "content": "hi"}])
    assert out.startswith("Recent Conversation:\n")


# ─────────────────────────────────────────────────────────────────────────
# _extract_semantic_resume_chunks — section parsing and fallbacks
# ─────────────────────────────────────────────────────────────────────────

SIMPLE_RESUME = """Vansh Pratap Singh

SKILLS
Python, FastAPI, PostgreSQL

PROJECTS
- Built My_Agent, a personal AI assistant

EDUCATION
B.Tech Computer Science
"""


def test_empty_resume_yields_no_chunks():
    assert extract_chunks("") == []
    assert extract_chunks("   \n\n  ") == []


def test_headings_route_content_into_typed_sections():
    chunks = extract_chunks(SIMPLE_RESUME)
    types = [c["type"] for c in chunks]

    assert "name" in types
    assert "skills" in types
    assert "projects" in types
    assert "education" in types


def test_detected_name_becomes_a_high_importance_identity_chunk():
    name_chunk = next(c for c in extract_chunks(SIMPLE_RESUME) if c["type"] == "name")
    assert name_chunk["content"] == "Vansh Pratap Singh"
    assert name_chunk["importance"] == "high"
    assert "identity" in name_chunk["tags"]


def test_skills_are_normalised_into_a_comma_separated_list():
    skills = next(c for c in extract_chunks(SIMPLE_RESUME) if c["type"] == "skills")
    assert "Python" in skills["content"]
    assert "FastAPI" in skills["content"]
    assert "PostgreSQL" in skills["content"]


def test_bullet_markers_are_stripped_from_project_entries():
    project = next(c for c in extract_chunks(SIMPLE_RESUME) if c["type"] == "projects")
    assert not project["content"].startswith("-")
    assert "My_Agent" in project["content"]


def test_duplicate_content_is_de_duplicated_across_sections():
    """
    The name line is also buffered into "other" (it precedes any heading), so
    the identical content appears twice before de-duplication. Only one chunk
    survives, and it is the earlier — the typed `name` chunk.
    """
    chunks = extract_chunks(SIMPLE_RESUME)
    contents = [c["content"] for c in chunks]
    assert len(contents) == len(set(contents))
    assert contents.count("Vansh Pratap Singh") == 1


def test_name_detection_strips_trailing_contact_digits():
    chunks = extract_chunks("Vansh Pratap Singh+91-6392306428\n\nSKILLS\nPython\n")
    name_chunk = next((c for c in chunks if c["type"] == "name"), None)
    assert name_chunk is not None
    assert name_chunk["content"] == "Vansh Pratap Singh"


def test_contact_lines_are_skipped_when_looking_for_a_name():
    chunks = extract_chunks(
        "someone@example.com\nhttps://github.com/x\nVansh Pratap Singh\n\nSKILLS\nPython\n"
    )
    name_chunk = next(c for c in chunks if c["type"] == "name")
    assert name_chunk["content"] == "Vansh Pratap Singh"


def test_section_heading_is_not_mistaken_for_a_name():
    """
    FIXED (was a strict xfail). `_detect_name` now rejects any line that
    `classify_heading` recognises, so a résumé whose name line extracted as an
    image no longer stores "SKILLS" as the user's identity at high importance
    in a chunk injected into every prompt. Further cases live in
    tests/test_resume_parsing.py.
    """
    chunks = extract_chunks("SKILLS\nPython, FastAPI\n\nEDUCATION\nB.Tech\n")
    assert not any(c["type"] == "name" for c in chunks)


def test_unstructured_text_falls_back_to_paragraph_chunks():
    """No recognisable headings — extraction must still produce usable chunks."""
    prose = "\n\n".join(
        f"This is paragraph number {i} and it contains more than eight words."
        for i in range(4)
    )
    chunks = extract_chunks(prose)
    assert len(chunks) >= 3
    assert all(c["content"].strip() for c in chunks)


def test_every_chunk_carries_the_required_shape():
    for chunk in extract_chunks(SIMPLE_RESUME):
        assert set(chunk) == {"type", "content", "tags", "importance"}
        assert chunk["content"].strip()
        assert chunk["importance"] in {"high", "medium"}
