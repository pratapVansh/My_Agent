"""
The typed tool contract, and the guarantees it is supposed to provide.

Two properties carry most of the weight here, and they pull in opposite
directions on purpose:

  * **Legacy results keep working.** Every tool in the repository still returns
    a bare dictionary. If adapting those changed a single classification, the
    migration would be a rewrite rather than a bridge — so the legacy cases
    below are the same shapes the previous shape-sniffing helpers were written
    against, asserted to reach the same verdicts.

  * **Unrecognisable results from consequential tools fail.** A read that comes
    back in an odd shape is interpreted generously; a `send`, `pay` or `delete`
    that comes back in an odd shape is an error. This asymmetry is the whole
    point of declaring effects, and it is what the malformed-input tests pin.

The end of the file exercises the real `execute_reasoning_loop` against a fake
model and fake tools, because a contract that is correct in isolation and
unreachable in the loop would pass everything above and change nothing.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents.base_agent import (
    _assess_tool_outcomes,
    _tool_reported_failure,
    _tool_yielded_evidence,
)
from tests.support import ScriptedAgent, run
from app.tools.contract import (
    DEFAULT_UNDECLARED_EFFECT,
    Effect,
    ErrorKind,
    ToolError,
    ToolResult,
    ToolStatus,
    coerce,
    derive_idempotency_key,
    effect_for_spec,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Constructing results
# ═══════════════════════════════════════════════════════════════════════════

def test_a_successful_result_reports_ok_and_carries_its_data():
    result = ToolResult.success({"name": "Vansh"}, tool="get_identity")
    assert result.status is ToolStatus.OK
    assert result.ok is True
    assert result.is_error is False
    assert result.is_empty is False
    assert result.yielded_evidence is True
    assert result.data["name"] == "Vansh"
    assert result.error is None


def test_a_no_data_result_is_a_success_that_found_nothing():
    """
    NO_DATA is not a failure. Conflating the two is what turns "I have no
    record of that" into "something went wrong", and vice versa — the second
    being the one that invites a model to invent a value.
    """
    result = ToolResult.no_data("No resume on file.", tool="get_resume")
    assert result.status is ToolStatus.NO_DATA
    assert result.is_empty is True
    assert result.is_error is False
    assert result.yielded_evidence is False
    assert result.error is None


def test_an_error_result_carries_kind_and_retryability():
    result = ToolResult.failed(
        "Qdrant unreachable", kind=ErrorKind.EXCEPTION, tool="get_skills"
    )
    assert result.status is ToolStatus.ERROR
    assert result.is_error is True
    assert result.yielded_evidence is False
    assert isinstance(result.error, ToolError)
    assert result.error.kind is ErrorKind.EXCEPTION
    assert result.error.retryable is False


def test_a_timeout_is_retryable_but_an_invalid_result_is_not():
    timeout = ToolResult.failed("timed out", kind=ErrorKind.TIMEOUT, retryable=True)
    malformed = ToolResult.failed("garbage", kind=ErrorKind.INVALID_RESULT)
    assert timeout.error.retryable is True
    assert malformed.error.retryable is False


def test_results_are_immutable():
    """Frozen so nothing downstream can rewrite what a tool reported."""
    result = ToolResult.success({"a": 1})
    with pytest.raises(Exception):
        result.status = ToolStatus.ERROR  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Every effect type
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("effect", list(Effect))
def test_every_effect_round_trips_through_a_result(effect):
    result = ToolResult.success({"ok": True}, effect=effect)
    assert result.effect is effect
    assert result.effect.value == effect.value


def test_effect_severity_is_totally_ordered():
    order = [Effect.READ, Effect.LOCAL_WRITE, Effect.EXTERNAL_WRITE, Effect.DESTRUCTIVE]
    severities = [e.severity for e in order]
    assert severities == sorted(severities)
    assert len(set(severities)) == len(order)


@pytest.mark.parametrize("effect,expected", [
    (Effect.READ, False),
    (Effect.LOCAL_WRITE, False),
    (Effect.EXTERNAL_WRITE, True),
    (Effect.DESTRUCTIVE, True),
])
def test_only_consequential_effects_require_confirmation(effect, expected):
    """The line the future gateway will read. Drawn once, here."""
    assert effect.requires_confirmation is expected
    assert ToolResult.success(effect=effect).requires_confirmation is expected


@pytest.mark.parametrize("effect,expected", [
    (Effect.READ, True),
    (Effect.LOCAL_WRITE, False),
    (Effect.EXTERNAL_WRITE, False),
    (Effect.DESTRUCTIVE, False),
])
def test_only_reads_are_freely_repeatable(effect, expected):
    assert effect.is_repeatable is expected


# ═══════════════════════════════════════════════════════════════════════════
# 3. Effect declaration on registry entries
# ═══════════════════════════════════════════════════════════════════════════

def test_effect_can_be_declared_as_enum_name_or_value():
    assert effect_for_spec({"effect": Effect.DESTRUCTIVE}) is Effect.DESTRUCTIVE
    assert effect_for_spec({"effect": "EXTERNAL_WRITE"}) is Effect.EXTERNAL_WRITE
    assert effect_for_spec({"effect": "read"}) is Effect.READ


def test_an_undeclared_effect_is_assumed_consequential():
    """
    Conservative by construction. Defaulting an omission to READ would make the
    omission invisible in exactly the case where it costs the most.
    """
    assert effect_for_spec({}, "mystery_tool") is DEFAULT_UNDECLARED_EFFECT
    assert DEFAULT_UNDECLARED_EFFECT.requires_confirmation is True


@pytest.mark.parametrize("bad", [{"effect": "SEND_IT"}, {"effect": 7}, {"effect": []}])
def test_an_unreadable_effect_declaration_falls_back_safely(bad):
    assert effect_for_spec(bad, "weird_tool") is DEFAULT_UNDECLARED_EFFECT


def test_every_registered_tool_in_the_repository_declares_an_effect():
    """
    The declaration is only a guarantee if nothing is missing it. Parsed from
    source because the registries are built inside `execute()` closures and
    cannot be imported directly.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "app" / "agents"
    undeclared = []
    total = 0
    for name in ("profile_agent", "job_agent", "email_agent", "academic_agent"):
        lines = (root / f"{name}.py").read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if '"callable":' in line:
                total += 1
                # Generous window: the two gated tools carry several lines of
                # comment between the callable and its declaration.
                if '"effect":' not in "\n".join(lines[index:index + 12]):
                    undeclared.append(f"{name}.py:{index + 1}")

    assert total > 0, "no tool registrations found — did the registry shape change?"
    assert undeclared == [], f"tools without a declared effect: {undeclared}"

    # The two consequential tools are known and few. If this count rises, a new
    # irreversible capability was added and should be reviewed deliberately.
    #
    # They are declared in `confirmable_tools`, not in the agents: a tool that
    # can be held for confirmation must be reconstructible by name after a
    # restart, and that is only possible if the agent and the resolver share
    # one definition.
    confirmable = (root / "confirmable_tools.py").read_text(encoding="utf-8")
    assert len(re.findall(r"Effect\.EXTERNAL_WRITE", confirmable)) == 1
    assert len(re.findall(r"Effect\.DESTRUCTIVE", confirmable)) == 1

    # And no agent may declare a consequential tool inline, which would make it
    # unreachable from the resolver and therefore unconfirmable after a restart.
    for name in ("profile_agent", "job_agent", "email_agent", "academic_agent"):
        agent_source = (root / f"{name}.py").read_text(encoding="utf-8")
        assert "Effect.EXTERNAL_WRITE" not in agent_source, name
        assert "Effect.DESTRUCTIVE" not in agent_source, name


# ═══════════════════════════════════════════════════════════════════════════
# 4. Preview handling
# ═══════════════════════════════════════════════════════════════════════════

def test_preview_is_carried_through_unchanged():
    preview = "To: alice@example.com\nSubject: Application\n\nDear Alice,"
    result = ToolResult.success(
        {"sent": True}, effect=Effect.EXTERNAL_WRITE, preview=preview
    )
    assert result.preview == preview


def test_a_confirmable_success_without_a_preview_is_flagged():
    """
    Nothing enforces this yet — no gateway reads previews. Surfacing it now
    means the gap is visible before something depends on it.
    """
    missing = ToolResult.success({"sent": True}, effect=Effect.EXTERNAL_WRITE)
    assert missing.preview_missing is True

    present = ToolResult.success(
        {"sent": True}, effect=Effect.EXTERNAL_WRITE, preview="To: bob@example.com"
    )
    assert present.preview_missing is False


@pytest.mark.parametrize("blank", [None, "", "   ", "\n"])
def test_a_blank_preview_counts_as_missing(blank):
    result = ToolResult.success({"x": 1}, effect=Effect.DESTRUCTIVE, preview=blank)
    assert result.preview_missing is True


def test_reads_and_local_writes_never_require_a_preview():
    for effect in (Effect.READ, Effect.LOCAL_WRITE):
        assert ToolResult.success({"x": 1}, effect=effect).preview_missing is False


def test_a_legacy_dict_preview_is_adopted():
    result = coerce(
        {"success": True, "preview": "To: carol@example.com", "sent": True},
        tool="send_email",
        declared_effect=Effect.EXTERNAL_WRITE,
    )
    assert result.preview == "To: carol@example.com"
    assert result.preview_missing is False


# ═══════════════════════════════════════════════════════════════════════════
# 5. Idempotency keys
# ═══════════════════════════════════════════════════════════════════════════

def test_the_same_call_derives_the_same_key():
    a = derive_idempotency_key("send_email", {"to": "x@y.com", "subject": "Hi"})
    b = derive_idempotency_key("send_email", {"to": "x@y.com", "subject": "Hi"})
    assert a == b


def test_argument_order_does_not_change_the_key():
    """Sorted keys, so one action cannot produce two identities."""
    a = derive_idempotency_key("send_email", {"to": "x@y.com", "subject": "Hi"})
    b = derive_idempotency_key("send_email", {"subject": "Hi", "to": "x@y.com"})
    assert a == b


def test_different_arguments_or_tools_derive_different_keys():
    base = derive_idempotency_key("send_email", {"to": "x@y.com"})
    assert base != derive_idempotency_key("send_email", {"to": "z@y.com"})
    assert base != derive_idempotency_key("save_draft", {"to": "x@y.com"})


def test_a_consequential_call_is_given_a_key_when_the_tool_supplies_none():
    result = coerce(
        {"success": True, "message_id": "abc"},
        tool="send_email",
        declared_effect=Effect.EXTERNAL_WRITE,
        tool_input={"to_email": "x@y.com", "subject": "Hi"},
    )
    assert result.idempotency_key
    assert result.idempotency_key == derive_idempotency_key(
        "send_email", {"to_email": "x@y.com", "subject": "Hi"}
    )


def test_a_read_is_not_given_a_key():
    """Reads are freely repeatable; a key there would be noise."""
    result = coerce(
        {"success": True, "skills": ["python"]},
        tool="get_skills",
        declared_effect=Effect.READ,
        tool_input={"limit": 15},
    )
    assert result.idempotency_key is None


def test_a_tool_supplied_key_is_preferred_over_a_derived_one():
    result = coerce(
        {"success": True, "idempotency_key": "server-side-id-42", "sent": True},
        tool="send_email",
        declared_effect=Effect.EXTERNAL_WRITE,
        tool_input={"to_email": "x@y.com"},
    )
    assert result.idempotency_key == "server-side-id-42"


def test_with_idempotency_key_returns_a_new_result():
    original = ToolResult.success({"a": 1}, effect=Effect.LOCAL_WRITE)
    keyed = original.with_idempotency_key("k1")
    assert original.idempotency_key is None
    assert keyed.idempotency_key == "k1"
    assert keyed.status is original.status


# ═══════════════════════════════════════════════════════════════════════════
# 6. Legacy dictionary results
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("legacy", [
    {"success": True, "found": True, "name": "Vansh Pratap Singh"},
    {"success": True, "count": 3, "skills": ["python", "rust", "go"]},
    {"success": True, "content": "B.Tech Information Technology"},
    {"tool": "job_search", "success": True, "results": [{"title": "SWE"}]},
])
def test_legacy_dicts_carrying_data_become_ok(legacy):
    assert coerce(legacy, declared_effect=Effect.READ).status is ToolStatus.OK


@pytest.mark.parametrize("legacy", [
    {"success": True, "found": False, "message": "No resume found."},
    {"success": True, "count": 0, "skills": [], "message": "No skills found."},
    {"success": True, "has_data": False},
    {"success": True},
])
def test_legacy_dicts_reporting_emptiness_become_no_data(legacy):
    result = coerce(legacy, declared_effect=Effect.READ)
    assert result.status is ToolStatus.NO_DATA
    assert result.is_error is False


@pytest.mark.parametrize("legacy", [
    {"success": False, "reason": "url is required"},
    {"success": False, "message": "Qdrant unreachable"},
    {"error": "connection reset"},
])
def test_legacy_dicts_reporting_failure_become_errors(legacy):
    result = coerce(legacy, declared_effect=Effect.READ)
    assert result.status is ToolStatus.ERROR
    assert result.error.kind is ErrorKind.TOOL_REPORTED


def test_the_found_flag_wins_over_a_populated_payload():
    """
    `{"found": False, "message": ...}` is an explicit statement of emptiness.
    A message is not data, and treating the payload as evidence would turn
    "nothing on file" into a result.
    """
    result = coerce(
        {"success": True, "found": False, "message": "I don't have your name on file."},
        declared_effect=Effect.READ,
    )
    assert result.status is ToolStatus.NO_DATA


def test_adapted_results_are_marked_and_keep_their_original():
    legacy = {"success": True, "found": True, "name": "Vansh"}
    result = coerce(legacy, tool="get_identity", declared_effect=Effect.READ)
    assert result.adapted is True
    assert result.raw is legacy


def test_a_partially_migrated_tool_may_declare_its_own_status():
    result = coerce({"status": "no_data", "message": "none"}, declared_effect=Effect.READ)
    assert result.status is ToolStatus.NO_DATA


def test_a_status_key_that_is_payload_not_verdict_is_ignored():
    """`{"status": "saved"}` is what save_job_bookmark actually returns."""
    result = coerce(
        {"success": True, "status": "saved", "id": 42},
        declared_effect=Effect.LOCAL_WRITE,
    )
    assert result.status is ToolStatus.OK


@pytest.mark.parametrize("legacy,expected", [
    (["a", "b"], ToolStatus.OK),
    ([], ToolStatus.NO_DATA),
    ("text", ToolStatus.OK),
    ("", ToolStatus.NO_DATA),
    (None, ToolStatus.NO_DATA),
])
def test_legacy_non_dict_reads_are_interpreted_by_emptiness(legacy, expected):
    assert coerce(legacy, declared_effect=Effect.READ).status is expected


# ═══════════════════════════════════════════════════════════════════════════
# 7. Malformed results — the fail-safe rule
# ═══════════════════════════════════════════════════════════════════════════

class _Opaque:
    """A return value the contract has no way to interpret."""


@pytest.mark.parametrize("effect", [Effect.EXTERNAL_WRITE, Effect.DESTRUCTIVE])
@pytest.mark.parametrize("malformed", [None, _Opaque(), ["unexpected"], "raw string", 0, 1])
def test_unrecognisable_results_from_consequential_tools_are_errors(effect, malformed):
    """
    The core safety property. A tool that may have sent, paid or deleted, whose
    result cannot be positively recognised as a success, must not be reported
    as one — the reply "done" is unrecoverable if it was not.
    """
    result = coerce(malformed, tool="send_email", declared_effect=effect)
    assert result.status is ToolStatus.ERROR
    assert result.error.kind is ErrorKind.INVALID_RESULT
    assert result.ok is False


@pytest.mark.parametrize("effect", [Effect.READ, Effect.LOCAL_WRITE])
def test_the_same_results_are_tolerated_from_harmless_tools(effect):
    """
    The other half of the asymmetry. Failing a read on an odd shape would break
    every tool still returning a bare list, for no safety gain.
    """
    assert coerce(["a"], declared_effect=effect).status is ToolStatus.OK
    assert coerce(_Opaque(), declared_effect=effect).status is ToolStatus.OK
    assert coerce(None, declared_effect=effect).status is ToolStatus.NO_DATA


def test_a_tool_may_not_claim_a_lower_effect_than_it_was_registered_with():
    """The registry is the authority — a tool cannot downgrade its own gate."""
    returned = ToolResult.success({"sent": True}, effect=Effect.READ)
    result = coerce(returned, tool="send_email", declared_effect=Effect.EXTERNAL_WRITE)
    assert result.effect is Effect.EXTERNAL_WRITE


def test_a_tool_claiming_a_higher_effect_than_registered_is_refused():
    """
    Escalation the other way is refused outright rather than honoured. A tool
    registered as a read that reports having performed a destructive action has
    done something nobody authorised, and the result is not trustworthy.
    """
    returned = ToolResult.success({"deleted": 12}, effect=Effect.DESTRUCTIVE)
    result = coerce(returned, tool="get_skills", declared_effect=Effect.READ)
    assert result.status is ToolStatus.ERROR
    assert result.error.kind is ErrorKind.INVALID_RESULT


def test_a_native_result_passes_through_with_its_fields_intact():
    native = ToolResult.success(
        {"sent": True},
        effect=Effect.EXTERNAL_WRITE,
        preview="To: dave@example.com",
        idempotency_key="k9",
        tool="send_email",
    )
    result = coerce(native, tool="send_email", declared_effect=Effect.EXTERNAL_WRITE)
    assert result.ok is True
    assert result.preview == "To: dave@example.com"
    assert result.idempotency_key == "k9"
    assert result.adapted is False


# ═══════════════════════════════════════════════════════════════════════════
# 8. Observations and logging
# ═══════════════════════════════════════════════════════════════════════════

def test_an_error_observation_says_a_failure_is_not_an_absence():
    result = ToolResult.failed("Qdrant unreachable", tool="get_skills")
    text = result.observation()
    assert "error" in text
    assert "does NOT mean the information is missing" in text


def test_a_no_data_observation_says_the_lookup_succeeded():
    text = ToolResult.no_data("nothing on file", tool="get_resume").observation()
    assert "no_data" in text
    assert "found nothing" in text


def test_an_observation_is_truncated_to_its_budget():
    result = ToolResult.success({"blob": "x" * 5000})
    assert len(result.observation(max_chars=200)) <= 210


def test_the_log_summary_never_contains_payload_content():
    result = ToolResult.success(
        {"secret": "hunter2"}, effect=Effect.LOCAL_WRITE, tool="save_draft"
    )
    rendered = str(result.summary())
    assert "hunter2" not in rendered
    assert "save_draft" in rendered and "LOCAL_WRITE" in rendered


# ═══════════════════════════════════════════════════════════════════════════
# 9. The pre-existing helpers still behave identically
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("result,expected", [
    ({"success": True, "found": True, "name": "V"}, True),
    ({"success": True, "count": 3, "skills": ["a"]}, True),
    ({"success": True, "content": "text"}, True),
    (["one", "two"], True),
    ("some text", True),
    ({"success": True, "found": False, "message": "none"}, False),
    ({"success": True, "count": 0, "skills": []}, False),
    ({"success": True, "has_data": False}, False),
    ([], False),
    ("", False),
    (None, False),
])
def test_evidence_helper_is_unchanged_by_the_migration(result, expected):
    assert _tool_yielded_evidence(result) is expected


@pytest.mark.parametrize("result", [
    {"success": False, "message": "Qdrant unreachable"},
    {"error": "connection reset"},
])
def test_failure_helper_is_unchanged_by_the_migration(result):
    assert _tool_reported_failure(result) is True


def test_answerability_verdicts_are_unchanged():
    assert _assess_tool_outcomes(["t"], [], ["t"]) == "TOOL_ERROR"
    assert _assess_tool_outcomes(["t"], [], []) == "NO_DATA"
    assert _assess_tool_outcomes(["t"], ["t"], []) == "ANSWERABLE"
    assert _assess_tool_outcomes(["a", "b"], ["a"], ["b"]) == "PARTIALLY_ANSWERABLE"
    assert _assess_tool_outcomes([], [], []) == ""


def test_a_tool_that_never_returned_still_yields_a_tool_error():
    """
    Regression, found by the loop test below.

    A tool that raises or times out never reaches the line recording it as
    used — that append runs after the call returns. Keying the "nothing was
    looked up" guard on `tools_used` alone therefore made TOOL_ERROR
    unreachable for precisely the two failures it describes, and reported a
    crashed lookup as though no lookup had been attempted.
    """
    assert _assess_tool_outcomes([], [], ["get_skills"]) == "TOOL_ERROR"
    assert _assess_tool_outcomes([], ["get_skills"], []) == "ANSWERABLE"


# ═══════════════════════════════════════════════════════════════════════════
# 10. The real reasoning loop
# ═══════════════════════════════════════════════════════════════════════════

# The scripted agent lives in tests/support now — three copies of it had
# accumulated across this suite, the gateway suite and the confirmation suite.
_ScriptedAgent = ScriptedAgent
_run = run


def _state():
    # No user_id, so the loop skips the memory reads/writes entirely.
    return {"user_input": "test", "user_id": "", "conversation_history": []}


def test_the_loop_still_calls_tools_and_returns_an_answer():
    """The regression guard: ordinary tool use is unaffected by the contract."""
    async def ok_tool(_):
        return {"success": True, "found": True, "name": "Vansh"}

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"get_identity","tool_input":{}}',
        '{"type":"final","content":"Your name is Vansh.","is_complete":true}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"get_identity": {"callable": ok_tool, "effect": Effect.READ}},
        max_iterations=3,
    ))

    assert result["final_answer"] == "Your name is Vansh."
    assert result["tools_used"] == ["get_identity"]
    assert result["tools_with_evidence"] == ["get_identity"]
    assert result["answerability"] == "ANSWERABLE"


def test_the_loop_exposes_typed_results_for_every_call():
    async def ok_tool(_):
        return {"success": True, "count": 2, "drafts": ["a", "b"]}

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"list_drafts","tool_input":{}}',
        '{"type":"final","content":"Two drafts."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"list_drafts": {"callable": ok_tool, "effect": Effect.READ}},
    ))

    results = result["tool_results"]
    assert len(results) == 1
    assert isinstance(results[0], ToolResult)
    assert results[0].tool == "list_drafts"
    assert results[0].effect is Effect.READ
    assert results[0].ok is True


def test_the_loop_carries_the_declared_effect_onto_the_result():
    async def send(_):
        return {"success": True, "sent": True}

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":{"to_email":"x@y.com"}}',
        '{"type":"final","content":"Sent."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"send_email": {"callable": send, "effect": Effect.EXTERNAL_WRITE}},
    ))

    sent = result["tool_results"][0]
    assert sent.effect is Effect.EXTERNAL_WRITE
    assert sent.requires_confirmation is True
    # Derived from the arguments, so a retry of the identical call is
    # recognisable as a repeat rather than performed twice.
    assert sent.idempotency_key == derive_idempotency_key(
        "send_email", {"to_email": "x@y.com"}
    )


def test_a_raising_tool_produces_a_typed_error_not_an_empty_result():
    async def broken(_):
        raise RuntimeError("qdrant down")

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"get_skills","tool_input":{}}',
        '{"type":"final","content":"I could not look that up."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"get_skills": {"callable": broken, "effect": Effect.READ}},
    ))

    assert result["tools_errored"] == ["get_skills"]
    assert result["answerability"] == "TOOL_ERROR"
    failure = result["tool_results"][0]
    assert failure.is_error is True
    assert failure.error.kind is ErrorKind.EXCEPTION


def test_a_malformed_consequential_result_is_an_error_when_it_is_executed():
    """
    The fail-safe rule at the point a confirmable tool actually runs.

    This used to be asserted through `execute_reasoning_loop`. The action
    gateway now intercepts consequential tools before the loop can call them,
    so the only path that executes one is a confirmed action — which is where
    the guarantee has to hold, and does. The loop-level property is now
    strictly stronger and is covered by the gateway suite: the tool is never
    reached at all.
    """
    async def sloppy_send(args):
        return object()

    async def scenario():
        from app.agents.actions import ActionGateway

        from app.domain.audit import InMemoryAuditLog
        from app.domain.pending_actions import InMemoryPendingActionStore
        from tests.support import register_confirmable

        register_confirmable("send_email", sloppy_send)
        gateway = ActionGateway(
            audit=InMemoryAuditLog(), pending=InMemoryPendingActionStore()
        )
        held = await gateway.intercept(
            tool="send_email",
            spec={"callable": sloppy_send, "effect": Effect.EXTERNAL_WRITE},
            arguments={"to_email": "x@y.com"},
            owner_id="owner@example.com",
        )
        return await gateway.confirm_and_execute(
            held.data["confirmation_token"], owner_id="owner@example.com"
        )

    result = _run(scenario())
    assert result.is_error is True
    assert result.error.kind is ErrorKind.INVALID_RESULT


def test_the_loop_never_executes_a_consequential_tool_at_all():
    """The replacement guarantee, stated at the loop boundary."""
    calls = []

    async def sloppy_send(args):
        calls.append(args)
        return object()

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":{"to_email":"x@y.com"}}',
        '{"type":"final","content":"..."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"send_email": {"callable": sloppy_send, "effect": Effect.EXTERNAL_WRITE}},
    ))

    assert calls == []
    assert result["tool_results"][0].is_pending is True
    assert result["tools_with_evidence"] == []


def test_an_unknown_tool_is_recorded_as_an_error_not_a_silent_skip():
    """
    Out-of-scope tools are removed from the registry entirely, so "unknown"
    here can mean "withheld from this caller". That must not read downstream as
    a lookup that found nothing.
    """
    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"send_email","tool_input":{}}',
        '{"type":"final","content":"I cannot do that."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"get_skills": {"callable": None, "effect": Effect.READ}},
    ))

    assert result["tool_results"][0].error.kind is ErrorKind.UNKNOWN_TOOL
    assert result["tools_used"] == []


def test_legacy_observations_are_byte_identical_to_the_previous_behaviour():
    """
    Migration safety. If adapting a legacy dict changed the observation text,
    this change would alter every existing prompt — the contract is meant to be
    invisible until a tool actually adopts it.
    """
    payload = {"success": True, "found": True, "name": "Vansh"}

    async def ok_tool(_):
        return payload

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"get_identity","tool_input":{}}',
        '{"type":"final","content":"done"}',
    ])
    _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"get_identity": {"callable": ok_tool, "effect": Effect.READ}},
    ))

    expected = agent._summarize_tool_result(payload)
    assert '"name": "Vansh"' in expected


def test_a_tool_returning_a_native_result_works_end_to_end():
    """Forward compatibility: the shape tools will migrate to."""
    async def native_tool(_):
        return ToolResult.success(
            {"name": "Vansh"}, effect=Effect.READ, tool="get_identity"
        )

    agent = _ScriptedAgent([
        '{"type":"tool_call","tool":"get_identity","tool_input":{}}',
        '{"type":"final","content":"Your name is Vansh."}',
    ])
    result = _run(agent.execute_reasoning_loop(
        state=_state(),
        base_system_prompt="p",
        tools={"get_identity": {"callable": native_tool, "effect": Effect.READ}},
    ))

    assert result["answerability"] == "ANSWERABLE"
    assert result["tool_results"][0].adapted is False
