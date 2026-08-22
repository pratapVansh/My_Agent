"""
Replace `schedule_upload`'s full uniqueness constraint with a partial one.

`uq_schedule_upload_hash` — a plain `UNIQUE (user_id, content_hash)` — applies
to every row ever inserted, active or retired. That makes a document
permanently unre-uploadable the moment it is superseded: a user going back to
an earlier semester's timetable, or simply re-running an upload, hits
`UniqueViolationError` against their own history. The re-upload flow's actual
requirement is narrower — at most one *active* document per user at a time —
which is exactly what a partial index expresses and a plain constraint
cannot. Same pattern as `action_audit`'s
`uq_action_audit_executed_idempotency` (see `migrate_add_action_audit.py`).

New deployments get the partial index directly from the model
(`app/domain/models.py`); this migration brings an existing database in line
with it. Safe to run at any time: the drop is `IF EXISTS`, the create is
`IF NOT EXISTS`, and neither touches a row.

Usage:
    python scripts/migrate_fix_schedule_upload_uniqueness.py            # apply
    python scripts/migrate_fix_schedule_upload_uniqueness.py --dry-run  # report only
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import async_session_maker, engine  # noqa: E402

DROP_OLD_CONSTRAINT = (
    "ALTER TABLE schedule_upload DROP CONSTRAINT IF EXISTS uq_schedule_upload_hash;"
)

CREATE_PARTIAL_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_schedule_upload_active_hash "
    "ON schedule_upload (user_id, content_hash) WHERE is_active = true;"
)


async def _constraint_exists(session, name: str) -> bool:
    return bool(await session.scalar(
        text("SELECT 1 FROM pg_constraint WHERE conname = :n"), {"n": name},
    ))


async def _index_exists(session, name: str) -> bool:
    return bool(await session.scalar(
        text("SELECT 1 FROM pg_indexes WHERE indexname = :n"), {"n": name},
    ))


async def main(dry_run: bool) -> int:
    async with async_session_maker() as session:
        if not await _index_exists(session, "uq_schedule_upload_active_hash") and not (
            await session.scalar(text("SELECT to_regclass('schedule_upload')"))
        ):
            print(
                "The 'schedule_upload' table does not exist. Run "
                "migrate_add_schedule_upload.py first."
            )
            return 1

        old_present = await _constraint_exists(session, "uq_schedule_upload_hash")
        new_present = await _index_exists(session, "uq_schedule_upload_active_hash")

        print(f"uq_schedule_upload_hash (old, full)      : {'present' if old_present else 'absent'}")
        print(f"uq_schedule_upload_active_hash (new, partial): {'present' if new_present else 'absent'}")

        if dry_run:
            print("\n--dry-run: no changes applied.")
            if old_present or not new_present:
                print("Run without --dry-run to apply.")
            return 0

        await session.execute(text(DROP_OLD_CONSTRAINT))
        await session.execute(text(CREATE_PARTIAL_INDEX))
        await session.commit()
        print("\nApplied. Re-uploading a previously-retired document no longer conflicts.")
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
