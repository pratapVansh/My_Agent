"""
Running the scenarios and judging what came back.

The judging is deliberately mechanical — substring presence, which agent ran,
which tools were recorded, what grounding verdict the loop reached. No model
grades another model's output here. An LLM judge would make this suite's
headline number depend on the same component it is supposed to be measuring,
and its failures would be exactly correlated with the failures it is meant to
catch.

That buys reliability at the cost of nuance: this cannot tell a well-phrased
answer from a clumsy one. It can tell a *grounded* answer from an invented one,
which is the property worth gating on.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any, Dict, List, Optional

from evals.metrics import EvalReport, FailureKind, TurnResult, summarize
from evals.scenarios import CHATBOT_TELLS, Scenario, for_mode

OWNER = "eval@example.com"


# ── Deterministic wiring ─────────────────────────────────────────────────────

class _Patch:
    """
    A minimal monkeypatch that works outside pytest.

    `tests.support.stub_services` takes a pytest `monkeypatch`; the eval runner
    has to be usable from a plain script too, so this supplies the same
    `setattr(target, name, value, raising=...)` surface and undoes everything
    on exit.
    """

    def __init__(self):
        self._undo: List[tuple] = []

    def setattr(self, target, name, value, raising=True):
        had = hasattr(target, name)
        old = getattr(target, name, None)
        if raising and not had:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._undo.append((target, name, old, had))
        setattr(target, name, value)

    def undo(self):
        for target, name, old, had in reversed(self._undo):
            if had:
                setattr(target, name, old)
            else:
                with contextlib.suppress(AttributeError):
                    delattr(target, name)
        self._undo.clear()


def _plan_json(agent: str) -> str:
    return json.dumps({
        "intent": "eval", "agent": agent, "confidence": 0.95,
        "needs_clarification": False, "clarification_question": "",
        "execution_plan": [{"step": 1, "agent": agent, "goal": "eval"}],
    })


def _final_json(text: str) -> str:
    return json.dumps({"type": "final", "content": text, "is_complete": True})


def _wire_deterministic(patch: _Patch, scenario: Scenario) -> Dict[str, List[str]]:
    """Script the model; keep every other production object in place."""
    from tests.support.harness import stub_services

    from app.agents.academic_agent import academic_agent
    from app.agents.email_agent import email_agent
    from app.agents.job_agent import job_agent
    from app.agents.planner_agent import planner_agent
    from app.agents.profile_agent import profile_agent
    from app.agents.response_agent import response_agent
    from app.memory.memory_manager import memory_manager

    stub_services(patch)

    async def _none(*a, **kw):
        return None

    async def _prompt(*a, **kw):
        # Populated on purpose: the model is handed plausible context and must
        # still refuse to answer from it when no tool ran.
        return {}, "Résumé excerpt: B.Tech, CGPA 8.80, Python, FastAPI."

    async def _turns(*a, **kw):
        return 1

    for name, fn in (
        ("on_user_input", _none), ("build_memory_prompt", _prompt),
        ("on_agent_response", _none), ("store_episode", _none),
        ("get_session_turn_count", _turns),
    ):
        patch.setattr(memory_manager, name, fn, raising=False)

    queues: Dict[str, List[str]] = {
        "planner": [_plan_json(scenario.planner_agent)],
        scenario.planner_agent: list(scenario.script),
    }
    tool_log: Dict[str, List[str]] = {"calls": []}

    for label, agent in (
        ("planner", planner_agent), ("job", job_agent), ("email", email_agent),
        ("academic", academic_agent), ("profile", profile_agent),
        ("response", response_agent),
    ):
        def make(name):
            async def call_groq(messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs):
                queue = queues.get(name)
                if queue:
                    return queue.pop(0)
                return _final_json("ok")
            return call_groq

        patch.setattr(agent, "call_groq", make(label), raising=False)

    return tool_log


# ── Running one scenario ─────────────────────────────────────────────────────

async def run_scenario(scenario: Scenario, *, mode: str) -> TurnResult:
    from app.agents.workflow import run_workflow

    patch = _Patch()
    started = time.perf_counter()
    try:
        if mode == "deterministic":
            _wire_deterministic(patch, scenario)

        state = await asyncio.wait_for(
            run_workflow(
                user_input=scenario.utterance,
                user_id=OWNER,
                session_id=f"eval-{scenario.id}",
                conversation_history=[],
            ),
            timeout=120.0,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return _judge(scenario, state, latency_ms)

    except asyncio.TimeoutError:
        return TurnResult(
            scenario_id=scenario.id, passed=False,
            failure=FailureKind.TIMEOUT, detail="exceeded 120s",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return TurnResult(
            scenario_id=scenario.id, passed=False,
            failure=FailureKind.EXCEPTION, detail=f"{type(exc).__name__}: {exc}",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    finally:
        patch.undo()


def _judge(scenario: Scenario, state: Dict[str, Any], latency_ms: float) -> TurnResult:
    """Decide whether this turn did the right thing, and if not, why."""
    task_result = state.get("task_result") or {}
    answer = (
        state.get("display_text")
        or (task_result.get("result") or {}).get("content")
        or ""
    )
    agent = state.get("current_agent") or task_result.get("agent") or ""
    grounding = state.get("grounding") or ""
    answerability = state.get("answerability") or ""
    tools = list(task_result.get("evidence") or [])
    path = " ".join(state.get("execution_path") or [])
    retried = "retry" in path

    result = TurnResult(
        scenario_id=scenario.id, passed=True, latency_ms=latency_ms,
        agent=agent, grounding=grounding, answerability=answerability,
        tools_called=tools, answer=answer[:400], retried=retried,
        required_tool=bool(scenario.expect_tools) and not scenario.allow_refusal,
    )

    lowered = answer.lower()

    # ── Safety first. A forbidden string in the answer is a delivered
    # ── invention, and no other property can compensate for it.
    for forbidden in scenario.forbid_substrings:
        if forbidden.lower() in lowered:
            result.passed = False
            result.failure = FailureKind.GROUNDING_VIOLATION
            result.detail = f"delivered forbidden text: {forbidden!r}"
            return result

    # ── Style. Checked for every scenario rather than declared per-scenario:
    # ── "does not sound like a chat product" is a property of the assistant,
    # ── not of one question. Runs after the safety check so a turn that
    # ── invented a fact is reported as the invention it is.
    for tell in CHATBOT_TELLS:
        if tell in lowered:
            result.passed = False
            result.failure = FailureKind.STYLE_VIOLATION
            result.detail = f"answer contains chat-product phrasing: {tell!r}"
            return result

    # ── Routing.
    if scenario.expect_agent and agent and agent != scenario.expect_agent:
        # The response agent is the terminal node for several paths and is not
        # a routing mistake in itself.
        if agent != "response":
            result.passed = False
            result.failure = FailureKind.WRONG_AGENT
            result.detail = f"expected {scenario.expect_agent}, ran {agent}"
            return result

    # ── Tool usage. A scenario that permits refusal has already been judged on
    # ── the safety check above: refusing without a tool is the correct answer
    # ── there, so a missing call is not counted against it.
    if scenario.expect_tools and not scenario.allow_refusal:
        if not any(t in tools for t in scenario.expect_tools):
            result.passed = False
            result.failure = FailureKind.MISSING_TOOL_CALL
            result.detail = (
                f"expected one of {scenario.expect_tools}, called {tools or 'nothing'}"
            )
            return result

    # ── Required content.
    for expected in scenario.expect_substrings:
        if expected.lower() not in lowered:
            result.passed = False
            result.failure = FailureKind.WRONG_CONTENT
            result.detail = f"missing expected text: {expected!r}"
            return result

    if scenario.expect_grounding and grounding != scenario.expect_grounding:
        result.passed = False
        result.failure = FailureKind.WRONG_CONTENT
        result.detail = f"grounding was {grounding!r}, expected {scenario.expect_grounding!r}"
        return result

    if answerability == "TOOL_ERROR" and not scenario.allow_refusal:
        result.passed = False
        result.failure = FailureKind.TOOL_ERROR
        result.detail = "a tool the turn depended on failed"
        return result

    return result


# ── Running a suite ──────────────────────────────────────────────────────────

async def run_suite(
    *,
    mode: str = "deterministic",
    pace_seconds: float = 0.0,
    only: Optional[List[str]] = None,
) -> EvalReport:
    """
    Run every scenario for a mode.

    `pace_seconds` exists for live mode: the Groq account this was built
    against is limited to 8000 tokens/minute, and an unpaced run spends most of
    its wall-clock inside rate-limit backoff, which would be reported as
    latency and would be a lie about the agent.
    """
    scenarios = for_mode(mode)
    if only:
        wanted = set(only)
        scenarios = [s for s in scenarios if s.id in wanted]

    results: List[TurnResult] = []
    for index, scenario in enumerate(scenarios):
        results.append(await run_scenario(scenario, mode=mode))
        if pace_seconds and index < len(scenarios) - 1:
            await asyncio.sleep(pace_seconds)

    return summarize(mode, results)
