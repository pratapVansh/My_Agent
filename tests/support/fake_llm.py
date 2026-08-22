"""
A language model you can script, so agent behaviour becomes testable.

Every agent in this system is a loop around one question — "what does the model
want to do next?" — and until now that question could only be answered by
calling Groq. So the agents were, in practice, untestable: the layer holding
`send_email` had no tests at all while the memory layer had five hundred.

`ScriptedLLM` replaces the answer, not the loop. The real
`execute_reasoning_loop` still parses the JSON, still resolves the tool, still
passes it through the action gateway, still coerces the result. What changes is
that the sequence of decisions is written by the test instead of sampled from a
model — which is what makes an adversarial case expressible at all. "A model
that keeps trying to send" is a scripted list; it is not something you can ask
a real model to reliably be.

The scripts are written with `tool_call(...)` and `final(...)` rather than raw
JSON strings so a change to the wire format breaks one function instead of a
hundred string literals.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from app.agents.base_agent import BaseAgent


# ── Script builders ──────────────────────────────────────────────────────────

def tool_call(tool: str, **tool_input: Any) -> str:
    """One decision to call a tool, in the shape the loop parses."""
    return json.dumps({"type": "tool_call", "tool": tool, "tool_input": tool_input})


def final(content: str, is_complete: bool = True) -> str:
    """One decision to answer."""
    return json.dumps({"type": "final", "content": content, "is_complete": is_complete})


def raw(text: str) -> str:
    """Deliberately malformed model output, for parser-robustness tests."""
    return text


def plan(
    *,
    intent: str = "general query",
    agent: str = "profile",
    confidence: float = 0.9,
    needs_clarification: bool = False,
    clarification_question: str = "",
    steps: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """
    A planner routing decision.

    Separate from `tool_call`/`final` because the planner speaks a different
    JSON dialect — it returns a route and a plan, not a tool invocation.
    """
    body: Dict[str, Any] = {
        "intent": intent,
        "agent": agent,
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }
    if steps is not None:
        body["execution_plan"] = list(steps)
    return json.dumps(body)


# ── The model ────────────────────────────────────────────────────────────────

class ScriptedLLM:
    """
    Returns pre-written responses in order, and records what it was asked.

    Running off the end of the script yields `default` rather than raising. A
    loop that asks one more time than expected is usually a bounded retry doing
    its job, and failing there would obscure the behaviour under test — the
    call count is available for assertions instead.
    """

    def __init__(
        self,
        script: Sequence[str],
        *,
        default: Optional[str] = None,
        fail_after: Optional[int] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._script: List[str] = list(script)
        self._default = default if default is not None else final("done")
        self._fail_after = fail_after
        self._error = error or RuntimeError("Groq API error: simulated failure")
        self.calls: List[List[Dict[str, Any]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def exhausted(self) -> bool:
        return not self._script

    def prompts(self) -> List[str]:
        """Every system prompt the agent built, for prompt-content assertions."""
        return [
            str(messages[0].get("content", ""))
            for messages in self.calls
            if messages and messages[0].get("role") == "system"
        ]

    def observations(self) -> List[str]:
        """Every tool observation fed back to the model across the run."""
        seen: List[str] = []
        for messages in self.calls:
            for message in messages:
                content = str(message.get("content", ""))
                if "observation:" in content:
                    seen.append(content)
        return seen

    async def __call__(self, messages, **kwargs) -> str:
        self.calls.append(list(messages or []))
        if self._fail_after is not None and self.call_count > self._fail_after:
            raise self._error
        return self._script.pop(0) if self._script else self._default


class ScriptedAgent(BaseAgent):
    """
    A minimal real agent whose only fake part is the model.

    For tests about the *loop* — gating, contract coercion, retries — rather
    than about a particular specialist. Specialist tests should drive the real
    agent with `harness.drive` so the registry under test is the real one.
    """

    def __init__(self, script: Sequence[str] | ScriptedLLM, name: str = "scripted"):
        super().__init__(name=name, description="scripted test agent")
        self.llm = script if isinstance(script, ScriptedLLM) else ScriptedLLM(script)

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:  # pragma: no cover
        return state

    async def call_groq(
        self, messages, temperature=0.7, max_tokens=1024, _retries=2, **kwargs
    ) -> str:
        return await self.llm(messages, temperature=temperature, max_tokens=max_tokens)


__all__ = ["ScriptedLLM", "ScriptedAgent", "tool_call", "final", "raw", "plan"]
