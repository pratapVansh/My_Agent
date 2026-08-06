"""
Apply post-audit schema changes to an EXISTING database.

SQLAlchemy's create_all() only creates missing tables — it does not add
constraints or indexes to tables that already exist. New deployments get these
automatically from the models; existing databases need this script once.

Adds:
  * uq_attendance_user_date_subject   — makes attendance upserts idempotent
                                        (H10: re-scraping duplicated rows and
                                        skewed attendance percentages)
  * uq_job_bookmark_user_url          — closes the bookmark insert race (H7)
  * ix_chat_history_user_session_time — matches the actual chat read pattern (L16)

Duplicate rows are collapsed before each unique constraint is added, keeping
the most recent record for every key.

Usage:
    python scripts/migrate_add_constraints.py            # apply
    python scripts/migrate_add_constraints.py --dry-run  # report only
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402

from app.memory.short_term_memory import short_term_memory  # noqa: E402

# Keep the newest row per key so the surviving record reflects the latest scrape.
DEDUPE_ATTENDANCE = """
DELETE FROM attendance a
USING attendance b
WHERE a.user_id = b.user_id
  AND a.date = b.date
  AND a.subject = b.subject
  AND a.created_at < b.created_at;
"""

DEDUPE_BOOKMARKS = """
DELETE FROM job_bookmarks a
USING job_bookmarks b
WHERE a.user_id = b.user_id
  AND a.url = b.url
  AND a.created_at < b.created_at;
"""

STEPS = [
    (
        "Collapse duplicate attendance rows",
        DEDUPE_ATTENDANCE,
    ),
    (
        "Add unique constraint on (user_id, date, subject)",
        "ALTER TABLE attendance ADD CONSTRAINT uq_attendance_user_date_subject "
        "UNIQUE (user_id, date, subject);",
    ),
    (
        "Collapse duplicate job bookmarks",
        DEDUPE_BOOKMARKS,
    ),
    (
        "Add unique constraint on (user_id, url)",
        "ALTER TABLE job_bookmarks ADD CONSTRAINT uq_job_bookmark_user_url "
        "UNIQUE (user_id, url);",
    ),
    (
        "Add composite chat-history index",
        "CREATE INDEX IF NOT EXISTS ix_chat_history_user_session_time "
        "ON chat_history (user_id, session_id, created_at);",
    ),
]


async def migrate(dry_run: bool) -> int:
    engine = short_term_memory.engine
    failures = 0

    async with engine.begin() as conn:
        for description, statement in STEPS:
            if dry_run:
                print(f"[dry-run] {description}")
                continue
            try:
                result = await conn.execute(text(statement))
                rowcount = getattr(result, "rowcount", -1)
                suffix = f" ({rowcount} rows)" if rowcount and rowcount > 0 else ""
                print(f"[ok] {description}{suffix}")
            except Exception as exc:
                message = str(exc).lower()
                if "already exists" in message or "duplicate" in message:
                    print(f"[skip] {description} — already applied")
                else:
                    failures += 1
                    print(f"[FAIL] {description}: {exc}")

    await engine.dispose()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show steps without applying")
    args = parser.parse_args()

    failures = asyncio.run(migrate(args.dry_run))

    if args.dry_run:
        print("\nDry run complete — nothing was changed.")
    elif failures:
        print(f"\nMigration finished with {failures} failure(s). Review the output above.")
        sys.exit(1)
    else:
        print("\nMigration complete.")


if __name__ == "__main__":
    main()
