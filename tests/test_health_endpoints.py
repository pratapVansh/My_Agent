"""
The health endpoints, and what they are allowed to cost.

`render.yaml` points `healthCheckPath` at `/health`, and that endpoint used to
issue a real completion against `settings.groq_model` — a 120-billion-parameter
model — on every platform probe. Nobody was waiting on those answers and the
route is unauthenticated, so it was a permanent unattended drain on exactly the
budget the audit found users competing for.

The distinction being defended here is between two different questions that had
been collapsed into one: *is this process able to serve* (answerable locally,
cheap, safe to poll continuously) and *are the providers reachable* (a real
network cost, worth answering at most once a minute).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.cohere_service import cohere_service
from app.services.groq_service import groq_service
from app.services.qdrant_service import qdrant_service


@pytest.fixture(autouse=True)
def _no_throttling(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", False)


@pytest.fixture(autouse=True)
def _fresh_deep_health_cache():
    """The memo is module state; a stale entry would mask the next test."""
    from app import main

    main._deep_health_cache["result"] = None
    main._deep_health_cache["at"] = 0.0
    yield
    main._deep_health_cache["result"] = None
    main._deep_health_cache["at"] = 0.0


@pytest.fixture
def client():
    # No context manager: that skips the lifespan handler, so these tests need
    # no live Postgres, Qdrant or Cohere.
    return TestClient(app)


@pytest.fixture
def provider_probes(monkeypatch):
    """Count how many times each provider health check is actually invoked."""
    calls = {"groq": 0, "cohere": 0, "qdrant": 0}

    async def _groq():
        calls["groq"] += 1
        return True

    async def _cohere():
        calls["cohere"] += 1
        return True

    async def _qdrant():
        calls["qdrant"] += 1
        return True

    monkeypatch.setattr(groq_service, "health_check", _groq)
    monkeypatch.setattr(cohere_service, "health_check", _cohere)
    monkeypatch.setattr(qdrant_service, "health_check", _qdrant)
    return calls


# ── E · /health is liveness only ─────────────────────────────────────────

def test_health_calls_no_provider(client, provider_probes):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert provider_probes == {"groq": 0, "cohere": 0, "qdrant": 0}, (
        "the liveness probe must not spend provider quota"
    )


def test_health_stays_free_under_repeated_probing(client, provider_probes):
    """Render polls this continuously; the cost must stay at zero."""
    for _ in range(25):
        assert client.get("/health").status_code == 200

    assert sum(provider_probes.values()) == 0


def test_the_platform_health_path_is_unchanged():
    """
    Render is configured against this exact path.

    Moving the provider probe was the fix; moving the *route* would have been
    an outage, because the platform would poll a 404 and cycle the service.
    """
    import pathlib
    import re

    render = pathlib.Path("render.yaml").read_text(encoding="utf-8")
    assert re.search(r"healthCheckPath:\s*/health\s*$", render, re.MULTILINE)


# ── F · /health/deep probes, and memoizes ────────────────────────────────

def test_deep_health_reports_every_provider(client, provider_probes):
    body = client.get("/health/deep").json()

    assert body["status"] == "healthy"
    assert body["groq"] == "connected"
    assert body["cohere"] == "connected"
    assert body["qdrant"] == "connected"
    assert provider_probes == {"groq": 1, "cohere": 1, "qdrant": 1}


def test_deep_health_is_memoized(client, provider_probes):
    """
    Without the memo this endpoint is the old `/health` with more providers.

    A dashboard or a misconfigured probe pointed here would otherwise reproduce
    the exact drain that was just removed.
    """
    first = client.get("/health/deep").json()
    for _ in range(10):
        later = client.get("/health/deep").json()

    assert first["cached"] is False
    assert later["cached"] is True
    assert provider_probes == {"groq": 1, "cohere": 1, "qdrant": 1}


def test_deep_health_reports_degraded_without_claiming_health(client, monkeypatch):
    async def _down():
        return False

    async def _up():
        return True

    monkeypatch.setattr(groq_service, "health_check", _down)
    monkeypatch.setattr(cohere_service, "health_check", _up)
    monkeypatch.setattr(qdrant_service, "health_check", _up)

    body = client.get("/health/deep").json()

    assert body["status"] == "degraded"
    assert body["groq"] == "disconnected"
    assert body["cohere"] == "connected"


def test_a_raising_provider_is_reported_not_propagated(client, monkeypatch):
    """
    A health endpoint that 500s tells the platform the *app* is down.

    `gather(return_exceptions=True)` is what keeps an unreachable provider a
    reported condition rather than an outage of the endpoint that reports it.
    """
    async def _explode():
        raise RuntimeError("qdrant unreachable")

    async def _up():
        return True

    monkeypatch.setattr(groq_service, "health_check", _up)
    monkeypatch.setattr(cohere_service, "health_check", _up)
    monkeypatch.setattr(qdrant_service, "health_check", _explode)

    response = client.get("/health/deep")

    assert response.status_code == 200
    assert response.json()["qdrant"] == "disconnected"
    assert response.json()["status"] == "degraded"
