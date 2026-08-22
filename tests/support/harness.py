"""
Driving the real agents, with only the outside world replaced.

The temptation when testing an agent is to hand it a registry the test built.
That is a much weaker test than it looks: it proves the loop can call a tool,
and proves nothing about the tools the agent actually registers — their names,
their arguments, their declared effects, or whether the one that sends email is
still marked EXTERNAL_WRITE. Those are precisely the facts a security boundary
rests on, and a fake registry cannot see any of them.

So the seam here is drawn as far out as it will go. The agent, its registry,
its closures, the reasoning loop, the contract, and the action gateway are all
the production objects. Only two things are replaced:

  * **the model**, because otherwise the test is a sample rather than a case
  * **the network and the database**, because a unit test cannot depend on
    Qdrant being up, and because "SMTP times out" is not something a real
    server will do on request

`stub_services` patches methods on the shared singletons the agents already
imported, which works precisely because those imports bind the object rather
than the module. Every stub defaults to the empty, harmless answer, so a test
that forgets to configure something gets NO_DATA rather than a connection
error — and NO_DATA is a state the system is supposed to handle correctly.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from tests.support.fake_llm import ScriptedLLM

# The singletons the agents hold references to.
from app.domain.academic import academic_repository
from app.domain.email import email_repository
from app.domain.jobs import jobs_repository
from app.memory.long_term_memory_qdrant import long_term_memory_qdrant
from app.memory.memory_manager import memory_manager
from app.services.email_sender_service import email_sender_service
from app.tools.email_draft_tool import email_draft_tool
from app.tools.job_search_tool import job_search_tool
from app.tools.timetable_tool import timetable_tool

OWNER = "owner@example.com"
CONVERSATION = "conv-test"


# ── State ────────────────────────────────────────────────────────────────────

def state(
    user_input: str = "hello",
    *,
    user_id: str = OWNER,
    session_id: str = CONVERSATION,
    history: Optional[Sequence[Dict[str, str]]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """An AgentState with the keys the agents and the workflow actually read."""
    base: Dict[str, Any] = {
        "user_input": user_input,
        "user_id": user_id,
        "session_id": session_id,
        "conversation_history": list(history or []),
        "memory_context": None,
        "memory_prompt": "",
        "detected_intent": "",
        "selected_agent": None,
        "planner_confidence": None,
        "needs_clarification": False,
        "clarification_question": "",
        "query_category": None,
        "memory_sources": None,
        "profile_intent": None,
        "followup_subject": None,
        "task_result": None,
        "agent_reasoning": None,
        "execution_plan": [],
        "current_step_index": 0,
        "step_results": {},
        "inter_step_context": None,
        "iteration_count": 0,
        "reflect_outcome": None,
        "reflect_failure_context": None,
        "display_text": None,
        "speech_text": None,
        "current_agent": None,
        "execution_path": [],
        "error": None,
        "scopes": None,
        "answerability": None,
    }
    base.update(overrides)
    return base


def run(coro):
    """Run one coroutine to completion."""
    return asyncio.run(coro)


def register_confirmable(
    name: str,
    fn,
    *,
    effect=None,
    preview: Any = "To: alice@example.com\nSubject: Hi\n\nHello.",
) -> None:
    """
    Point a confirmable tool name at a test double, for this test only.

    Needed because a stored action is executed by *resolving its name* through
    `confirmable_tools`, never by holding a callable — so passing a recorder in
    a spec no longer influences what runs, which is the whole point. Without
    this, a test that intercepts `send_email` and confirms it would reach the
    real SMTP sender.

    Registering here keeps the resolution path fully exercised: name → registry
    → callable, exactly as production does it. The `confirmable_registry`
    fixture restores the original mapping afterwards.
    """
    from app.agents import confirmable_tools
    from app.tools.contract import Effect

    resolved_effect = effect or Effect.EXTERNAL_WRITE

    def builder(owner_id: str) -> Dict[str, Any]:
        return {
            "callable": fn,
            "effect": resolved_effect,
            "preview": preview,
            "description": f"test double for {name}",
        }

    confirmable_tools.CONFIRMABLE_TOOLS[name] = builder


# ── Driving a real agent ─────────────────────────────────────────────────────

async def drive(agent, script: Sequence[str] | ScriptedLLM, agent_state: Dict[str, Any]):
    """
    Run a real agent's `execute()` with a scripted model.

    Everything else is production code: the registry it builds, the loop, the
    gateway, the envelope it returns. The state is mutated in place, as the
    workflow does.
    """
    llm = script if isinstance(script, ScriptedLLM) else ScriptedLLM(script)

    async def call_groq(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
        return await llm(messages, temperature=temperature, max_tokens=max_tokens)

    original = agent.call_groq
    agent.call_groq = call_groq
    try:
        result = await agent.execute(agent_state)
    finally:
        agent.call_groq = original
    return result, llm


async def capture_registry(agent, agent_state: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """
    The tool registry an agent really builds, without running its loop.

    Used by tests that assert on declared effects and preview builders — the
    facts the action gateway depends on. Substituting the loop is what makes
    this cheap; the registry itself is untouched.
    """
    captured = await _capture_loop_call(agent, agent_state)
    return captured.get("tools", {})


async def capture_prompt(agent, agent_state: Optional[Dict[str, Any]] = None) -> str:
    """The base system prompt an agent builds, for prompt-content assertions."""
    captured = await _capture_loop_call(agent, agent_state)
    return captured.get("prompt", "")


async def _capture_loop_call(
    agent, agent_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Intercept the one call an agent makes into its reasoning loop.

    Signature-agnostic on purpose: the agents pass `state=` and
    `base_system_prompt=` as keywords, and a stub with renamed positionals
    raises a TypeError that the agent's own `except Exception` swallows into a
    failed envelope — an empty registry that looks like a missing tool rather
    than a broken test double.
    """
    captured: Dict[str, Any] = {}

    async def capture(*args, **kwargs):
        captured["tools"] = kwargs.get("tools") or (args[2] if len(args) > 2 else {}) or {}
        captured["prompt"] = kwargs.get("base_system_prompt") or (
            args[1] if len(args) > 1 else ""
        )
        return {
            "final_answer": "", "iterations": 0, "tools_used": [], "trace": [],
            "answerability": "", "tools_with_evidence": [], "tools_errored": [],
            "tool_results": [], "pending_actions": [],
        }

    original = agent.execute_reasoning_loop
    agent.execute_reasoning_loop = capture
    try:
        await agent.execute(agent_state or state())
    finally:
        agent.execute_reasoning_loop = original

    if "tools" not in captured:
        raise AssertionError(
            f"{agent.name}.execute() returned without reaching its reasoning "
            "loop — the registry could not be captured."
        )
    return captured


# ── Service doubles ──────────────────────────────────────────────────────────

class RecordedCalls:
    """What the stubbed outside world was asked to do."""

    def __init__(self) -> None:
        self.sent_emails: List[Dict[str, Any]] = []
        self.saved_drafts: List[Dict[str, Any]] = []
        self.saved_facts: List[Dict[str, Any]] = []
        self.forgotten_keys: List[str] = []
        self.bookmarks: List[Dict[str, Any]] = []
        self.attendance: List[Dict[str, Any]] = []

    @property
    def emails_sent(self) -> int:
        return len(self.sent_emails)


def stub_services(monkeypatch, **overrides: Any) -> RecordedCalls:
    """
    Replace every network and database boundary the agents reach.

    Defaults are the empty answer, so an unconfigured lookup yields NO_DATA
    rather than an exception. Pass `retrieve_resume=...` etc. to override any
    individual method with your own async callable or a constant.
    """
    recorded = RecordedCalls()

    def _install(target, name, default):
        supplied = overrides.pop(name, None)
        if supplied is None:
            monkeypatch.setattr(target, name, default, raising=False)
            return
        if callable(supplied) and asyncio.iscoroutinefunction(supplied):
            monkeypatch.setattr(target, name, supplied, raising=False)
        else:
            async def constant(*a, **kw):
                return supplied
            monkeypatch.setattr(target, name, constant, raising=False)

    # ── base_agent's memory hooks (must never touch Postgres in a unit test) ──
    async def _no_insights(*a, **kw):
        return []

    async def _no_save(*a, **kw):
        return None

    _install(memory_manager, "get_tool_insights", _no_insights)
    _install(memory_manager, "save_tool_outcome", _no_save)

    # ── profile ──────────────────────────────────────────────────────────────
    async def _no_facts(*a, **kw):
        return []

    async def _save_fact(user_id=None, key=None, value=None, **kw):
        recorded.saved_facts.append({"user_id": user_id, "key": key, "value": value})
        return "fact-1"

    async def _forget_fact(user_id=None, key=None, **kw):
        recorded.forgotten_keys.append(key)
        return True

    _install(memory_manager, "get_profile_facts", _no_facts)
    _install(memory_manager, "save_profile_fact", _save_fact)
    _install(memory_manager, "forget_profile_fact", _forget_fact)

    async def _none(*a, **kw):
        return None

    async def _empty(*a, **kw):
        return []

    for method in ("retrieve_resume",):
        _install(long_term_memory_qdrant, method, _none)
    for method in ("retrieve_skills", "retrieve_projects", "retrieve_section"):
        _install(long_term_memory_qdrant, method, _empty)
    _install(memory_manager, "retrieve_resume", _none)
    _install(memory_manager, "retrieve_skills", _empty)

    # ── jobs ─────────────────────────────────────────────────────────────────
    async def _no_jobs(*a, **kw):
        return {"tool": "job_search", "success": True, "results": [],
                "total_candidates": 0, "total_filtered": 0, "user_skills": []}

    async def _save_bookmark(user_id=None, title=None, url=None, **kw):
        recorded.bookmarks.append({"title": title, "url": url})
        return "bookmark-1"

    async def _no_bookmarks(*a, **kw):
        return []

    async def _no_urls(*a, **kw):
        return set()

    _install(job_search_tool, "search_jobs", _no_jobs)
    _install(jobs_repository, "save_bookmark", _save_bookmark)
    _install(jobs_repository, "get_bookmarks", _no_bookmarks)
    _install(jobs_repository, "get_bookmarked_urls", _no_urls)

    # ── email ────────────────────────────────────────────────────────────────
    async def _draft(user_id=None, query=None, **kw):
        return {"success": True, "draft": {
            "subject": "Drafted subject", "body": "Drafted body.",
            "greeting": "Hi,", "closing": "Best,", "signature": "Vansh",
        }}

    async def _save_draft(user_id=None, subject=None, body=None, **kw):
        recorded.saved_drafts.append({"subject": subject, "body": body})
        return 1

    async def _no_drafts(*a, **kw):
        return []

    async def _no_templates(*a, **kw):
        return []

    async def _save_template(*a, **kw):
        return 1

    async def _mark_sent(*a, **kw):
        return True

    async def _send(to_email=None, subject=None, body=None, cc=None, **kw):
        recorded.sent_emails.append({"to_email": to_email, "subject": subject, "body": body})
        return {"success": True, "message_id": f"stub-{len(recorded.sent_emails)}"}

    _install(email_draft_tool, "draft_email", _draft)
    _install(email_repository, "save_draft", _save_draft)
    _install(email_repository, "get_drafts", _no_drafts)
    _install(email_repository, "get_templates", _no_templates)
    _install(email_repository, "save_template", _save_template)
    _install(email_repository, "mark_draft_sent", _mark_sent)
    _install(email_sender_service, "send_email", _send)

    # ── academic ─────────────────────────────────────────────────────────────
    async def _no_classes(*a, **kw):
        return {"success": True, "suggestions": [], "message": "No timetable set."}

    async def _store_timetable(*a, **kw):
        return {"success": True, "stored": 1}

    async def _store_attendance(user_id=None, subject=None, status=None, **kw):
        recorded.attendance.append({"subject": subject, "status": status})
        return "att-1"

    _install(timetable_tool, "suggest_classes", _no_classes)
    _install(timetable_tool, "store_timetable", _store_timetable)
    _install(academic_repository, "store_attendance", _store_attendance)
    _install(academic_repository, "retrieve_attendance", _empty)
    _install(academic_repository, "retrieve_timetable", _empty)
    # Was missing, and its absence reached real Postgres from unit tests: the
    # schedule tools call this before reading any rows, so an unstubbed
    # `has_timetable` made every timetable turn depend on a live database and
    # on which event loop happened to still be open. Defaults to False to
    # match `retrieve_timetable`'s empty default — "no timetable stored" and
    # "no rows" are then the same story rather than two halves of different ones.
    async def _no_timetable(*a, **kw):
        return False

    _install(academic_repository, "has_timetable", _no_timetable)
    _install(academic_repository, "retrieve_exams", _empty)
    _install(academic_repository, "store_exam", _none)
    _install(academic_repository, "store_plan", _none)
    _install(academic_repository, "retrieve_plans", _empty)
    _install(academic_repository, "mark_plan_done", _none)

    if overrides:
        raise AssertionError(
            f"stub_services got unknown overrides: {sorted(overrides)}"
        )
    return recorded


__all__ = [
    "OWNER",
    "CONVERSATION",
    "RecordedCalls",
    "capture_prompt",
    "capture_registry",
    "drive",
    "run",
    "state",
    "stub_services",
]
