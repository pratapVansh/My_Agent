"""
Shadow-mode safety (Phase 2).

Shadow retrieval exists to gather comparison data before the cutover. The
entire feature is only acceptable if it cannot affect production, so these
tests assert the negative: whatever the v2 engine does — succeed, fail, hang —
the served prompt is byte-identical to what the legacy path produced.

See docs/MEMORY_ARCHITECTURE.md §6, Phase 2.
"""
import asyncio

import pytest

import app.memory.memory_manager as mm
from app.config import settings
from app.memory.memory_manager import memory_manager


LEGACY_CONTEXT = {
    "profile_facts": [{"key": "name", "value": "Vansh"}],
    "episodes": [],
    "chat_history": [],
    "preferences": [],
    "long_term": {},
}


@pytest.fixture
def stub_legacy(monkeypatch):
    """Pin the legacy retrieval path to a known result."""
    async def fake_retrieve(**kwargs):
        return dict(LEGACY_CONTEXT)

    monkeypatch.setattr(memory_manager, "retrieve_context", fake_retrieve)
    return LEGACY_CONTEXT


@pytest.fixture
def shadow_off(monkeypatch):
    monkeypatch.setattr(settings, "memory_v2_shadow_read", False)


@pytest.fixture
def shadow_on(monkeypatch):
    monkeypatch.setattr(settings, "memory_v2_shadow_read", True)


async def drain_shadow_tasks():
    """Let detached shadow tasks run to completion."""
    for _ in range(5):
        await asyncio.sleep(0)
    if mm._shadow_tasks:
        await asyncio.gather(*list(mm._shadow_tasks), return_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────

# `_env_file=None` keeps these asserting the *code* default. Without it they
# read the developer's own .env, so enabling a flag locally — which is the whole
# point of the flag — turns the suite red on a machine whose code is correct.
def test_shadow_read_is_off_by_default():
    """
    Comparing against a store that has not been backfilled yet produces noise,
    not signal — so unlike dual-write this defaults off.
    """
    from app.config import Settings
    assert Settings(_env_file=None).memory_v2_shadow_read is False


def test_dual_write_is_on_by_default():
    from app.config import Settings
    assert Settings(_env_file=None).memory_v2_dual_write is True


# ─────────────────────────────────────────────────────────────────────────
# The served result is never affected
# ─────────────────────────────────────────────────────────────────────────

async def test_build_memory_prompt_returns_the_legacy_result(stub_legacy, shadow_off):
    context, prompt = await memory_manager.build_memory_prompt("vansh", "s1", "hello")
    assert context == LEGACY_CONTEXT
    assert "User Profile Facts:" in prompt
    assert "- name: Vansh" in prompt


async def test_enabling_shadow_does_not_change_the_served_prompt(
    stub_legacy, shadow_on, monkeypatch
):
    async def fake_shadow(user_id, query, legacy_prompt, conversation_id='', visibilities=None, memory_owner_id=None):
        return None

    monkeypatch.setattr(memory_manager, "_shadow_compare", fake_shadow)

    _, with_shadow = await memory_manager.build_memory_prompt("vansh", "s1", "hello")
    await drain_shadow_tasks()

    monkeypatch.setattr(settings, "memory_v2_shadow_read", False)
    _, without_shadow = await memory_manager.build_memory_prompt("vansh", "s1", "hello")

    assert with_shadow == without_shadow


async def test_a_failing_shadow_never_breaks_the_turn(
    stub_legacy, shadow_on, monkeypatch
):
    """
    The property that makes shadow mode safe to leave enabled: the v2 engine
    may be entirely broken and the turn still completes normally.
    """
    async def exploding_shadow(user_id, query, legacy_prompt, conversation_id='', visibilities=None, memory_owner_id=None):
        raise RuntimeError("v2 engine is on fire")

    monkeypatch.setattr(memory_manager, "_shadow_compare", exploding_shadow)

    context, prompt = await memory_manager.build_memory_prompt("vansh", "s1", "hello")
    await drain_shadow_tasks()

    assert context == LEGACY_CONTEXT
    assert "- name: Vansh" in prompt


async def test_shadow_runs_detached_rather_than_inline(
    stub_legacy, shadow_on, monkeypatch
):
    """
    Shadow must add no latency. A slow shadow must not delay the return — most
    of all on a spoken turn, where an extra round trip is audible.
    """
    started = asyncio.Event()

    async def slow_shadow(user_id, query, legacy_prompt, conversation_id='', visibilities=None, memory_owner_id=None):
        started.set()
        await asyncio.sleep(5)

    monkeypatch.setattr(memory_manager, "_shadow_compare", slow_shadow)

    _, prompt = await asyncio.wait_for(
        memory_manager.build_memory_prompt("vansh", "s1", "hello"), timeout=1.0
    )
    assert "- name: Vansh" in prompt

    # Clean up the still-running task so it does not leak into other tests.
    for task in list(mm._shadow_tasks):
        task.cancel()
    await asyncio.gather(*list(mm._shadow_tasks), return_exceptions=True)


async def test_shadow_is_not_spawned_when_disabled(stub_legacy, shadow_off, monkeypatch):
    called = []

    async def tracking_shadow(user_id, query, legacy_prompt, conversation_id='', visibilities=None, memory_owner_id=None):
        called.append(query)

    monkeypatch.setattr(memory_manager, "_shadow_compare", tracking_shadow)
    await memory_manager.build_memory_prompt("vansh", "s1", "hello")
    await drain_shadow_tasks()

    assert called == []


async def test_detached_tasks_are_strongly_referenced(
    stub_legacy, shadow_on, monkeypatch
):
    """
    asyncio holds only a weak reference to a running task, so without a strong
    reference a fire-and-forget task can be collected mid-execution and its
    exception never observed.
    """
    gate = asyncio.Event()

    async def waiting_shadow(user_id, query, legacy_prompt, conversation_id='', visibilities=None, memory_owner_id=None):
        await gate.wait()

    monkeypatch.setattr(memory_manager, "_shadow_compare", waiting_shadow)
    await memory_manager.build_memory_prompt("vansh", "s1", "hello")

    assert len(mm._shadow_tasks) == 1
    gate.set()
    await drain_shadow_tasks()
    assert len(mm._shadow_tasks) == 0
