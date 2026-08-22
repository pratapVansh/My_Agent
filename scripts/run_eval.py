"""
Run the agent evaluation suite and print the metrics.

    python scripts/run_eval.py                      # deterministic (default)
    python scripts/run_eval.py --mode live          # real model, real API
    python scripts/run_eval.py --json report.json   # machine-readable too
    python scripts/run_eval.py --only cgpa_invented schedule_today_compliant

Deterministic mode scripts the model's decisions, so it measures the SYSTEM —
routing, grounding enforcement, tool wiring, the retry path — reproducibly and
without spending a token. Live mode measures the MODEL: whether it actually
calls the tools it is told to. The two are reported separately because they
answer different questions; see `evals/__init__.py`.

Exit code is 1 if there was any grounding violation, or if task success fell
below --min-success (default 0.9). That makes this usable as a CI gate.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.runner import run_suite  # noqa: E402


async def main(args) -> int:
    report = await run_suite(
        mode=args.mode,
        pace_seconds=args.pace,
        only=args.only,
    )

    print(report.render())

    if args.json:
        Path(args.json).write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        print(f"  Wrote {args.json}")

    if report.grounding_violations:
        print(
            f"\nFAIL: {report.grounding_violations} grounding violation(s) — "
            "an unsupported personal fact reached the user."
        )
        return 1
    if report.task_success_rate < args.min_success:
        print(
            f"\nFAIL: task success {report.task_success_rate:.0%} is below the "
            f"{args.min_success:.0%} threshold."
        )
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("deterministic", "live"), default="deterministic"
    )
    parser.add_argument(
        "--pace", type=float, default=None,
        help="seconds between turns; defaults to 0 deterministic, 20 live "
             "(the Groq free tier is 8000 TPM and an unpaced live run "
             "measures backoff rather than the agent)",
    )
    parser.add_argument("--json", default=None, help="also write a JSON report here")
    parser.add_argument("--only", nargs="*", default=None, help="scenario ids to run")
    parser.add_argument("--min-success", type=float, default=0.9)
    args = parser.parse_args()

    if args.pace is None:
        args.pace = 20.0 if args.mode == "live" else 0.0

    try:
        code = asyncio.run(main(args))
    finally:
        with __import__("contextlib").suppress(Exception):
            from app.db.session import engine
            asyncio.run(engine.dispose())
    sys.exit(code)
