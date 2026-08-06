"""Rate limiter tests (audit finding H1)."""
import pytest

from app.middleware.rate_limit import SlidingWindowCounter, _classify, _limit_for


def test_allows_up_to_limit_then_blocks():
    counter = SlidingWindowCounter(window_seconds=60.0)

    for i in range(5):
        allowed, _ = counter.check_and_record("client:llm", limit=5, now=100.0 + i)
        assert allowed, f"request {i + 1} should be allowed"

    allowed, retry_after = counter.check_and_record("client:llm", limit=5, now=105.0)
    assert not allowed
    assert retry_after > 0


def test_window_slides_so_old_hits_expire():
    counter = SlidingWindowCounter(window_seconds=60.0)

    for i in range(5):
        counter.check_and_record("client:llm", limit=5, now=100.0 + i)

    # Once the original hits age out of the window, capacity returns.
    allowed, _ = counter.check_and_record("client:llm", limit=5, now=200.0)
    assert allowed


def test_clients_are_limited_independently():
    counter = SlidingWindowCounter(window_seconds=60.0)

    for i in range(3):
        counter.check_and_record("alice:llm", limit=3, now=100.0 + i)

    blocked, _ = counter.check_and_record("alice:llm", limit=3, now=103.0)
    assert not blocked

    # A different client is unaffected by alice exhausting her allowance.
    allowed, _ = counter.check_and_record("bob:llm", limit=3, now=103.0)
    assert allowed


def test_prune_drops_inactive_keys():
    counter = SlidingWindowCounter(window_seconds=60.0)
    counter.check_and_record("stale:default", limit=10, now=100.0)

    counter.prune(now=500.0)
    assert "stale:default" not in counter._hits


@pytest.mark.parametrize("path,expected", [
    ("/api/v1/agents/query", "llm"),
    ("/api/v1/agents/tools/attendance/scrape", "expensive"),
    ("/api/v1/voice/token", "expensive"),
    ("/api/v1/agents/memory/upload-pdf", "upload"),
    ("/api/v1/agents/tools/timetable/upload-pdf", "upload"),
    ("/api/v1/agents/memory/episodes/vansh", "default"),
])
def test_path_classification(path, expected):
    assert _classify(path) == expected


def test_expensive_bucket_is_stricter_than_default():
    assert _limit_for("expensive") < _limit_for("llm") <= _limit_for("default")
