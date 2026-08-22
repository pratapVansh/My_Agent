"""
A deliberately tiny live probe against Groq, Cohere and Qdrant.

Not a benchmark and not a load test. It exists to answer questions the offline
harness cannot: what a real round trip costs, whether the router fits inside
its 1.2s budget once the amplification fixes are in, and whether any 429s
appear at rest.

The request budget is fixed and small, and every call is the cheapest form of
itself — `max_tokens` in the single digits where the content does not matter.
Re-running this is safe but pointless: the numbers move with network weather,
not with the code.

Run:  python scripts/probe_providers.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import hybrid_router  # noqa: E402
from app.config import settings  # noqa: E402
from app.services import call_metrics  # noqa: E402
from app.services.cohere_service import cohere_service  # noqa: E402
from app.services.groq_service import groq_service  # noqa: E402
from app.services.qdrant_service import qdrant_service  # noqa: E402

RESULTS: dict[str, object] = {}


async def timed(label: str, coro):
    started = time.perf_counter()
    try:
        value = await coro
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  {label:<34} {elapsed:8.0f} ms   ok")
        return elapsed, value
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  {label:<34} {elapsed:8.0f} ms   FAILED: {type(exc).__name__}: {exc}")
        return elapsed, None


async def main() -> None:
    print(f"\nModel: {settings.groq_model}")
    print(f"Limiter: concurrency={settings.groq_max_concurrency}, "
          f"budget={settings.groq_tokens_per_minute} tok/min\n")

    with call_metrics.turn("live-probe") as metrics:

        # ── Qdrant: a metadata call, no vectors ──────────────────────────
        print("Qdrant")
        await timed("health_check (list collections)", qdrant_service.health_check())

        # ── Cohere: one embedding, then the same one again ───────────────
        print("\nCohere")
        cold, _ = await timed(
            "embed_text (cold)",
            cohere_service.embed_text("what is my CGPA", input_type="search_query"),
        )
        warm, _ = await timed(
            "embed_text (same query, cached)",
            cohere_service.embed_text("what is my CGPA", input_type="search_query"),
        )
        RESULTS["cohere_cold_ms"] = cold
        RESULTS["cohere_warm_ms"] = warm

        # Single-flight, live: five concurrent identical queries.
        before = metrics.embed_calls
        started = time.perf_counter()
        await asyncio.gather(*(
            cohere_service.embed_text("a distinct probe query", input_type="search_query")
            for _ in range(5)
        ))
        fanout_ms = (time.perf_counter() - started) * 1000
        issued = metrics.embed_calls - before
        print(f"  {'5 concurrent identical embeds':<34} {fanout_ms:8.0f} ms   "
              f"{issued} API call(s) issued")
        RESULTS["concurrent_embed_api_calls"] = issued

        # ── Groq: the router model, then the answering model ─────────────
        print("\nGroq")
        router_times = []
        for i in range(3):
            elapsed, _ = await timed(
                f"router classify #{i + 1} ({hybrid_router._ROUTER_MODEL})",
                groq_service.chat_completion(
                    messages=[
                        {"role": "system", "content": hybrid_router._ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": "what is my attendance this week"},
                    ],
                    model=hybrid_router._ROUTER_MODEL,
                    temperature=0.0,
                    max_tokens=hybrid_router._ROUTER_MAX_TOKENS,
                    response_format={"type": "json_object"},
                ),
            )
            router_times.append(elapsed)
        RESULTS["router_ms"] = router_times

        answer_ms, _ = await timed(
            f"one completion ({settings.groq_model}, 16 tok)",
            groq_service.chat_completion(
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=16,
                temperature=0.0,
            ),
        )
        RESULTS["completion_ms"] = answer_ms

    print("\n" + "=" * 68)
    print("TURN_COST for this probe")
    print("=" * 68)
    for key, value in metrics.as_dict().items():
        print(f"  {key:<28} {value}")

    print("\n" + "=" * 68)
    print("Router budget check")
    print("=" * 68)
    budget = hybrid_router._ROUTER_TIMEOUT_SECONDS * 1000
    if router_times:
        worst = max(router_times)
        median = statistics.median(router_times)
        print(f"  budget                       {budget:.0f} ms")
        print(f"  median                       {median:.0f} ms")
        print(f"  worst of {len(router_times)}                   {worst:.0f} ms")
        verdict = "WITHIN budget" if worst < budget else "EXCEEDS budget"
        print(f"  verdict                      {verdict}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
