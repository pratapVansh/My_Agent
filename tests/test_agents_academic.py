"""
The academic agent — timetable, attendance, exams, study plans.

The agent that owns the data a student actually checks daily, and the one whose
"no data" case is most common: an empty timetable is the normal state of a
fresh account, not a fault. So the NO_DATA/TOOL_ERROR split matters here more
than anywhere except the résumé sections.

It also holds more write tools than any other agent — six of ten — and all of
them are LOCAL_WRITE. That is the correct classification (marking attendance is
the user's own record, reversible by them) and it is worth a test, because the
moment one of these starts writing to a university ERP it stops being local.
"""
from __future__ import annotations

import pytest

from app.agents.academic_agent import AcademicAgent
from app.tools.contract import Effect
from tests.support import (
    ScriptedLLM,
    capture_registry,
    drive,
    final,
    state,
    stub_services,
    tool_call,
)


@pytest.fixture
def agent():
    return AcademicAgent()


@pytest.fixture
def services(monkeypatch):
    return stub_services(monkeypatch)


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

async def test_the_real_registry_exposes_the_expected_tools(agent, services):
    tools = await capture_registry(agent)
    for name in (
        "class_suggestions", "add_timetable_entry", "mark_attendance",
        "get_attendance_summary", "add_exam", "get_upcoming_exams",
        "generate_study_schedule", "save_plan", "get_plans", "mark_plan_done",
    ):
        assert name in tools, f"{name} is no longer registered"


async def test_every_academic_write_is_local(agent, services):
    """
    All of these change the user's own records and are reversible by them.
    A tool here that ever writes to a university system would be
    EXTERNAL_WRITE, and this test should be what notices.
    """
    tools = await capture_registry(agent)
    for name, spec in tools.items():
        assert spec["effect"] in (Effect.READ, Effect.LOCAL_WRITE), name


# ═══════════════════════════════════════════════════════════════════════════
# Timetable
# ═══════════════════════════════════════════════════════════════════════════

async def test_class_suggestions_are_returned(agent, monkeypatch):
    stub_services(monkeypatch, suggest_classes={
        "success": True,
        "suggestions": [
            {"subject": "Data Structures", "start": "09:00", "risk": "low"},
            {"subject": "Networks", "start": "11:00", "risk": "high"},
        ],
    })

    result, llm = await drive(
        agent,
        [tool_call("class_suggestions"), final("You have DS at 9 and Networks at 11.")],
        state("what classes do I have today"),
    )

    observed = " ".join(llm.observations())
    assert "Data Structures" in observed
    assert result["answerability"] == "ANSWERABLE"


async def test_an_empty_timetable_is_no_data(agent, services):
    """The normal state of a fresh account, and not an error."""
    result, _ = await drive(
        agent,
        [tool_call("class_suggestions"), final("You don't have a timetable set up yet.")],
        state("what classes do I have today"),
    )
    assert result["answerability"] == "NO_DATA"
    assert result["task_result"]["status"] == "success"


async def test_an_unreachable_timetable_store_is_a_tool_error(agent, monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("postgres unreachable")

    stub_services(monkeypatch, suggest_classes=broken)

    result, _ = await drive(
        agent,
        [tool_call("class_suggestions"), final("I couldn't reach your timetable.")],
        state("what classes do I have today"),
    )
    assert result["answerability"] == "TOOL_ERROR"


async def test_adding_a_timetable_entry_passes_its_fields(agent, monkeypatch):
    seen = {}

    async def store(user_id=None, entries=None, **kw):
        seen["entries"] = entries
        return {"success": True, "stored": len(entries or [])}

    stub_services(monkeypatch, store_timetable=store)

    await drive(
        agent,
        [tool_call("add_timetable_entry", subject="Physics", day="Monday",
                   start_time="09:00", end_time="11:00", room="204"),
         final("Added.")],
        state("add physics monday 9 to 11 room 204"),
    )

    assert seen["entries"], "no timetable entry reached the store"
    entry = seen["entries"][0]
    assert getattr(entry, "subject", None) == "Physics" or entry.get("subject") == "Physics"


# ═══════════════════════════════════════════════════════════════════════════
# Attendance
# ═══════════════════════════════════════════════════════════════════════════

async def test_marking_attendance_writes_a_record(agent, monkeypatch):
    recorded = stub_services(monkeypatch)

    await drive(
        agent,
        [tool_call("mark_attendance", subject="Physics", status="present"),
         final("Marked.")],
        state("mark me present for physics"),
    )
    assert recorded.attendance == [{"subject": "Physics", "status": "present"}]


async def test_an_attendance_summary_is_computed(agent, monkeypatch):
    stub_services(monkeypatch, retrieve_attendance=[
        {"subject": "Physics", "status": "present"},
        {"subject": "Physics", "status": "present"},
        {"subject": "Physics", "status": "absent"},
    ])

    result, llm = await drive(
        agent,
        [tool_call("get_attendance_summary"), final("Physics is at 67%.")],
        state("what is my attendance"),
    )
    assert "Physics" in " ".join(llm.observations())
    assert result["answerability"] == "ANSWERABLE"


async def test_no_attendance_records_is_no_data(agent, services):
    result, _ = await drive(
        agent,
        [tool_call("get_attendance_summary"), final("No attendance recorded yet.")],
        state("what is my attendance"),
    )
    assert result["answerability"] == "NO_DATA"


# ═══════════════════════════════════════════════════════════════════════════
# Exams and plans
# ═══════════════════════════════════════════════════════════════════════════

async def test_upcoming_exams_with_none_scheduled_is_no_data(agent, services):
    result, _ = await drive(
        agent, [tool_call("get_upcoming_exams"), final("No exams scheduled.")],
        state("what exams do I have"),
    )
    assert result["answerability"] == "NO_DATA"


async def test_exams_are_returned_when_present(agent, monkeypatch):
    stub_services(monkeypatch, retrieve_exams=[
        {"subject": "Networks", "exam_date": "2026-08-20", "exam_type": "midterm"},
    ])
    _, llm = await drive(
        agent, [tool_call("get_upcoming_exams"), final("Networks midterm on the 20th.")],
        state("what exams do I have"),
    )
    assert "Networks" in " ".join(llm.observations())


async def test_saving_a_plan_does_not_require_confirmation(agent, services):
    """LOCAL_WRITE — the user's own record, reversible, so not gated."""
    from app.agents.actions import action_gateway

    action_gateway.reset()
    result, _ = await drive(
        agent,
        [tool_call("save_plan", title="Revise networks", scheduled_for="2026-08-15"),
         final("Saved to your plan.")],
        state("remind me to revise networks on the 15th"),
    )
    assert await action_gateway.pending_for(state()["session_id"], state()["user_id"]) == []
    assert result["task_result"]["status"] == "success"


# ═══════════════════════════════════════════════════════════════════════════
# Failure
# ═══════════════════════════════════════════════════════════════════════════

async def test_an_agent_level_failure_produces_a_failed_envelope(agent, monkeypatch):
    stub_services(monkeypatch)
    result, _ = await drive(agent, ScriptedLLM([], fail_after=0), state("what classes today"))

    assert result["task_result"]["status"] == "failed"
    assert "Academic agent error" in result["error"]


async def test_answering_without_a_tool_asserts_nothing(agent, services):
    result, _ = await drive(agent, [final("Studying tips: sleep well.")],
                            state("how should I study"))
    assert result["answerability"] == ""
