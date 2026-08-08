"""
Run the memory worker standalone.

    python scripts/run_memory_worker.py --once     # one cycle, then exit
    python scripts/run_memory_worker.py            # poll until interrupted
    python scripts/run_memory_worker.py --status   # queue depth only
    python scripts/run_memory_worker.py --maintain # run lifecycle sweeps now

The worker also runs in-process inside the API when MEMORY_WORKER_ENABLED is
true, which is the right arrangement for a single instance. Run it here instead
when scaling out, so several replicas do not each poll the same queue — and set
MEMORY_WORKER_ENABLED=false on the API processes when you do.
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.memory.cognition.worker import memory_worker  # noqa: E402
from app.memory.stores.postgres_event_queue import postgres_event_queue  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("memory-worker")


async def show_status() -> int:
    pending = await postgres_event_queue.pending_count()
    from app.memory.stores.postgres_record_store import postgres_record_store

    total = await postgres_record_store.count()
    awaiting = len(await postgres_record_store.pending_embeddings(limit=1000))

    print(f"\nmemory queue")
    print(f"  pending events      : {pending}")
    print(f"  memory records      : {total}")
    print(f"  awaiting embedding  : {awaiting}")
    print(f"\nbatch size {settings.memory_extraction_batch_size} turns, "
          f"idle flush {settings.memory_extraction_idle_flush_seconds:.0f}s\n")
    return 0


async def run_maintenance() -> int:
    """Run the lifecycle sweeps immediately, without waiting for the interval."""
    from app.memory.cognition.maintenance import memory_maintenance

    stats = await memory_maintenance.run_once()
    print(f"\nmaintenance complete: {stats.summary()}\n")
    return 1 if stats.failed_jobs else 0


async def run_once() -> int:
    stats = await memory_worker.run_once()
    print(f"\ncycle complete: {stats.summary()}\n")
    return 0


async def run_forever() -> int:
    stop = asyncio.Event()
    try:
        await memory_worker.run_forever(stop)
    except KeyboardInterrupt:
        stop.set()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the memory worker.")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    parser.add_argument("--status", action="store_true", help="Show queue depth and exit.")
    parser.add_argument("--maintain", action="store_true",
                        help="Run decay, dedup, conflict and guest sweeps now.")
    args = parser.parse_args()

    try:
        if args.status:
            return asyncio.run(show_status())
        if args.maintain:
            return asyncio.run(run_maintenance())
        if args.once:
            return asyncio.run(run_once())
        return asyncio.run(run_forever())
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
