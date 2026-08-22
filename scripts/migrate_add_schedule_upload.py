"""
Create the schedule-upload table and link timetable rows to it.

New deployments get both from `create_all()` via the models. An existing
database needs this once, and it is safe to run at any time: the table creation
is `IF NOT EXISTS` and the column addition is `ADD COLUMN IF NOT EXISTS`, so
nothing is dropped, rewritten or backfilled.

Existing timetable rows keep working with `schedule_upload_id IS NULL` — that
is the correct value for a class added conversationally ("add Physics Monday
9am"), which has no document behind it. The column is nullable precisely so the
two cases stay distinguishable rather than one being faked into the other.

Usage:
    python scripts/migrate_add_schedule_upload.py            # apply
    python scripts/migrate_add_schedule_upload.py --dry-run  # report only
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import async_session_maker, engine  # noqa: E402

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS schedule_upload (
    id             UUID PRIMARY KEY,
    user_id        VARCHAR(255) NOT NULL,
    filename       VARCHAR(512) NOT NULL,
    content_hash   VARCHAR(64)  NOT NULL,
    source_title   TEXT,
    semester       VARCHAR(32),
    page_count     INTEGER,
    entry_count    INTEGER      NOT NULL DEFAULT 0,
    parse_notes    TEXT,
    ingest_method  VARCHAR(32)  NOT NULL DEFAULT 'pdf_parser',
    is_active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT uq_schedule_upload_hash UNIQUE (user_id, content_hash)
);
"""

INDEXES = [
    ("ix_schedule_upload_user_id",
     "CREATE INDEX IF NOT EXISTS ix_schedule_upload_user_id "
     "ON schedule_upload (user_id);"),
    ("ix_schedule_upload_active",
     "CREATE INDEX IF NOT EXISTS ix_schedule_upload_active "
     "ON schedule_upload (user_id, is_active);"),
]

ALTER_TIMETABLE = [
    ("timetable.schedule_upload_id",
     "ALTER TABLE timetable "
     "ADD COLUMN IF NOT EXISTS schedule_upload_id UUID;"),
]

TIMETABLE_INDEXES = [
    ("ix_timetable_schedule_upload_id",
     "CREATE INDEX IF NOT EXISTS ix_timetable_schedule_upload_id "
     "ON timetable (schedule_upload_id);"),
    # The read path is always (user, active, day) — the schedule lookup runs on
    # every "what have I got tomorrow", including spoken ones.
    ("ix_timetable_user_active_day",
     "CREATE INDEX IF NOT EXISTS ix_timetable_user_active_day "
     "ON timetable (user_id, is_active, day_of_week);"),
]


async def _table_exists(session, name: str) -> bool:
    return bool(await session.scalar(text("SELECT to_regclass(:n)"), {"n": name}))


async def _column_exists(session, table: str, column: str) -> bool:
    return bool(await session.scalar(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    ))


async def main(dry_run: bool) -> int:
    async with async_session_maker() as session:
        if not await _table_exists(session, "timetable"):
            print(
                "The 'timetable' table does not exist. Start the application "
                "once so create_all() builds the base schema, then re-run this."
            )
            return 1

        upload_exists = await _table_exists(session, "schedule_upload")
        column_exists = await _column_exists(
            session, "timetable", "schedule_upload_id"
        )

        print(f"schedule_upload table        : {'present' if upload_exists else 'MISSING'}")
        print(f"timetable.schedule_upload_id : {'present' if column_exists else 'MISSING'}")

        if dry_run:
            print("\n--dry-run: no changes applied.")
            if not upload_exists or not column_exists:
                print("Run without --dry-run to create them.")
            return 0

        await session.execute(text(CREATE_TABLE))
        for _, statement in INDEXES:
            await session.execute(text(statement))
        for _, statement in ALTER_TIMETABLE:
            await session.execute(text(statement))
        for _, statement in TIMETABLE_INDEXES:
            await session.execute(text(statement))
        await session.commit()
        print("\nApplied. Timetable uploads can now be recorded and traced.")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()
    try:
        code = asyncio.run(main(args.dry_run))
    finally:
        asyncio.run(engine.dispose())
    sys.exit(code)
