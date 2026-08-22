"""
Create the action audit table on an EXISTING database.

New deployments get this from `create_all()` via the model. Existing databases
need it once — and they need it *before* the application starts, because the
gateway fails closed: without this table, every EXTERNAL_WRITE and DESTRUCTIVE
action is refused rather than silently unaudited.

The partial unique index is the part that matters. It is what makes "this
action already ran" a fact the database enforces rather than something a
process remembers, so a retry after a restart — or on a second replica —
cannot send the same email twice.

Usage:
    python scripts/migrate_add_action_audit.py            # apply
    python scripts/migrate_add_action_audit.py --dry-run  # report only
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import async_session_maker, engine  # noqa: E402

CREATE_PENDING = """
CREATE TABLE IF NOT EXISTS pending_action (
    id                UUID PRIMARY KEY,
    token_hash        VARCHAR(64)  NOT NULL,
    audit_id          UUID         NOT NULL,
    owner_id          VARCHAR(255) NOT NULL,
    conversation_id   VARCHAR(255),
    tool              VARCHAR(128) NOT NULL,
    effect            VARCHAR(32)  NOT NULL,
    arguments         JSONB        NOT NULL DEFAULT '{}'::jsonb,
    preview           TEXT         NOT NULL DEFAULT '',
    content_hash      VARCHAR(64)  NOT NULL,
    idempotency_key   VARCHAR(64)  NOT NULL,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at        TIMESTAMPTZ  NOT NULL,
    CONSTRAINT uq_pending_action_token UNIQUE (token_hash)
);
"""

PENDING_INDEXES = [
    ("ix_pending_action_owner_id",
     "CREATE INDEX IF NOT EXISTS ix_pending_action_owner_id "
     "ON pending_action (owner_id);"),
    ("ix_pending_action_audit_id",
     "CREATE INDEX IF NOT EXISTS ix_pending_action_audit_id "
     "ON pending_action (audit_id);"),
    ("ix_pending_action_idempotency_key",
     "CREATE INDEX IF NOT EXISTS ix_pending_action_idempotency_key "
     "ON pending_action (idempotency_key);"),
    ("ix_pending_action_owner_conversation",
     "CREATE INDEX IF NOT EXISTS ix_pending_action_owner_conversation "
     "ON pending_action (owner_id, conversation_id);"),
    ("ix_pending_action_expires_at",
     "CREATE INDEX IF NOT EXISTS ix_pending_action_expires_at "
     "ON pending_action (expires_at);"),
]

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS action_audit (
    id                UUID PRIMARY KEY,
    owner_id          VARCHAR(255) NOT NULL,
    conversation_id   VARCHAR(255),
    effect            VARCHAR(32)  NOT NULL,
    tool              VARCHAR(128) NOT NULL,
    arguments_hash    VARCHAR(64)  NOT NULL,
    preview_hash      VARCHAR(64)  NOT NULL,
    token_hash        VARCHAR(64)  NOT NULL,
    idempotency_key   VARCHAR(64)  NOT NULL,
    status            VARCHAR(16)  NOT NULL DEFAULT 'pending',
    outcome           TEXT,
    error             TEXT,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    confirmed_at      TIMESTAMPTZ,
    executed_at       TIMESTAMPTZ
);
"""

INDEXES = [
    ("ix_action_audit_owner_id",
     "CREATE INDEX IF NOT EXISTS ix_action_audit_owner_id "
     "ON action_audit (owner_id);"),
    ("ix_action_audit_conversation_id",
     "CREATE INDEX IF NOT EXISTS ix_action_audit_conversation_id "
     "ON action_audit (conversation_id);"),
    ("ix_action_audit_token_hash",
     "CREATE INDEX IF NOT EXISTS ix_action_audit_token_hash "
     "ON action_audit (token_hash);"),
    ("ix_action_audit_idempotency_key",
     "CREATE INDEX IF NOT EXISTS ix_action_audit_idempotency_key "
     "ON action_audit (idempotency_key);"),
    ("ix_action_audit_status",
     "CREATE INDEX IF NOT EXISTS ix_action_audit_status "
     "ON action_audit (status);"),
    ("ix_action_audit_owner_created",
     "CREATE INDEX IF NOT EXISTS ix_action_audit_owner_created "
     "ON action_audit (owner_id, created_at);"),
    # The durable idempotency guarantee: at most one executed row per key.
    ("uq_action_audit_executed_idempotency",
     "CREATE UNIQUE INDEX IF NOT EXISTS uq_action_audit_executed_idempotency "
     "ON action_audit (idempotency_key) WHERE status = 'executed';"),
]

CHECK_TABLE = """
SELECT EXISTS (
    SELECT 1 FROM information_schema.tables WHERE table_name = 'action_audit'
);
"""

CHECK_PENDING_TABLE = """
SELECT EXISTS (
    SELECT 1 FROM information_schema.tables WHERE table_name = 'pending_action'
);
"""

CHECK_INDEX = """
SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = :name);
"""


async def main(dry_run: bool) -> int:
    async with async_session_maker() as session:
        exists = (await session.execute(text(CHECK_TABLE))).scalar()
        print(f"action_audit table: {'present' if exists else 'MISSING'}")

        pending_exists = (
            await session.execute(text(CHECK_PENDING_TABLE))
        ).scalar()
        print(f"pending_action table: {'present' if pending_exists else 'MISSING'}")

        missing = []
        if exists:
            for name, _ in INDEXES:
                present = (
                    await session.execute(text(CHECK_INDEX), {"name": name})
                ).scalar()
                print(f"  {name}: {'present' if present else 'MISSING'}")
                if not present:
                    missing.append(name)
        if pending_exists:
            for name, _ in PENDING_INDEXES:
                present = (
                    await session.execute(text(CHECK_INDEX), {"name": name})
                ).scalar()
                print(f"  {name}: {'present' if present else 'MISSING'}")
                if not present:
                    missing.append(name)

        if dry_run:
            print("\n--dry-run: no changes applied.")
            if not exists or not pending_exists or missing:
                print("Run without --dry-run to create them.")
            return 0

        if exists and pending_exists and not missing:
            print("\nNothing to do.")
            return 0

        await session.execute(text(CREATE_TABLE))
        for name, statement in INDEXES:
            await session.execute(text(statement))
        await session.execute(text(CREATE_PENDING))
        for name, statement in PENDING_INDEXES:
            await session.execute(text(statement))
        await session.commit()
        print("\nApplied. The audit and pending-action tables are in place.")
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
