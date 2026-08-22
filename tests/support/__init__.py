"""
Reusable test infrastructure for the agent layer.

Three pieces, each replacing exactly one thing the agents cannot be tested
against:

    fake_llm    the model's decisions            → scripted, so cases are exact
    fake_tools  a tool's outcome                 → OK / NO_DATA / error / timeout
    harness     the network and the database     → recorded, in-memory doubles

Everything else — the registries, the reasoning loop, the tool contract, the
action gateway — is the production object. That is the point: a test built on a
fake registry proves the loop works and says nothing about whether the tool
that sends email is still declared EXTERNAL_WRITE.
"""
from tests.support.fake_llm import (
    ScriptedAgent,
    ScriptedLLM,
    final,
    plan,
    raw,
    tool_call,
)
from tests.support.fake_tools import FakeTool, Outcome, registry
from tests.support.harness import (
    CONVERSATION,
    OWNER,
    RecordedCalls,
    capture_prompt,
    capture_registry,
    drive,
    register_confirmable,
    run,
    state,
    stub_services,
)

__all__ = [
    "CONVERSATION",
    "FakeTool",
    "OWNER",
    "Outcome",
    "RecordedCalls",
    "ScriptedAgent",
    "ScriptedLLM",
    "capture_prompt",
    "capture_registry",
    "drive",
    "final",
    "plan",
    "raw",
    "register_confirmable",
    "registry",
    "run",
    "state",
    "stub_services",
    "tool_call",
]
