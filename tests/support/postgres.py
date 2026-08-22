"""
Opt-in access to a *dedicated* PostgreSQL database, and the guards that keep it
dedicated.

Everything the durable action system rests on is SQL that had only ever been
compiled: `DELETE … RETURNING` deciding a race between two workers, a partial
unique index refusing a second executed row, JSONB round-tripping the arguments
an approved action will replay. None of that is proven by a Python double, so
these tests need a real server.

Which creates the risk this module exists to eliminate. The tests are
deliberately destructive — they `TRUNCATE` both tables between cases, and
`take_expired` issues a table-wide `DELETE` with no owner filter. Pointed at the
development database, a single run would silently delete real outstanding
confirmations. So the safety is layered, and every layer fails closed:

    opt-in            nothing runs unless POSTGRES_INTEGRATION_TESTS=1
    name guard        the target must not be settings.postgres_db
    suffix guard      the target must end in `_test`
    identity guard    after connecting, SELECT current_database() must match

The last one is the only one that cannot be fooled by a misconfiguration: it
asks the server where we actually are rather than trusting where we meant to
go. A mismatch aborts the run instead of proceeding.

Nothing here touches the application's own engine. `PostgresAuditLog` and
`PostgresPendingActionStore` both accept a session factory precisely so a test
can supply its own, and the production singletons are never reconfigured.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# ── Opt-in ───────────────────────────────────────────────────────────────────

ENABLE_FLAG = "POSTGRES_INTEGRATION_TESTS"
DB_ENV = "TEST_POSTGRES_DB"
DEFAULT_TEST_DB = "my_agent_test"

# The two tables these tests own. Nothing else is created, read or truncated —
# so even inside the test database the blast radius is bounded.
OWNED_TABLES = ("pending_action", "action_audit")


def integration_enabled() -> bool:
    """Whether the caller has explicitly asked for database tests."""
    return os.getenv(ENABLE_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


def test_database_name() -> str:
    """
    The database these tests may use.

    Deliberately *not* defaulted from `settings.postgres_db`: a fallback to the
    application's own database is the one mistake that must be impossible, so
    there is no code path that could produce it.
    """
    return (os.getenv(DB_ENV) or DEFAULT_TEST_DB).strip()


class UnsafeTestDatabase(RuntimeError):
    """The configured target is not a database these tests may touch."""


def assert_safe_target(name: str) -> None:
    """
    Refuse anything that is not an obviously dedicated test database.

    Raises rather than skips: a misconfiguration here is not a reason to run
    fewer tests, it is a reason to stop before something is deleted.
    """
    if not name:
        raise UnsafeTestDatabase(
            f"No test database configured. Set {DB_ENV} or use the default "
            f"'{DEFAULT_TEST_DB}'."
        )
    if name == settings.postgres_db:
        raise UnsafeTestDatabase(
            f"REFUSING TO RUN: the test database resolved to '{name}', which is "
            f"the application's own database. These tests TRUNCATE tables and "
            f"issue table-wide DELETEs. Point {DB_ENV} at a dedicated database."
        )
    if not name.endswith("_test"):
        raise UnsafeTestDatabase(
            f"REFUSING TO RUN: '{name}' does not end in '_test'. The suffix is "
            f"required so a destructive run cannot be aimed at a real database "
            f"by a typo."
        )


def test_database_url(name: str) -> str:
    """Connection URL for the test database, reusing the app's credentials."""
    return (
        f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{name}"
    )


# ── Connection ───────────────────────────────────────────────────────────────

@dataclass
class TestDatabase:
    """One connection pool onto the dedicated test database."""

    name: str
    engine: object
    session_maker: async_sessionmaker

    async def verify_identity(self) -> None:
        """
        Ask the server which database we are actually in.

        The guard that cannot be defeated by a wrong URL, a stale environment
        variable, or a connection-level `search_path` surprise — every other
        check tests intent, this one tests reality.
        """
        async with self.engine.connect() as conn:
            actual = await conn.scalar(text("SELECT current_database()"))
        if actual != self.name:
            raise UnsafeTestDatabase(
                f"REFUSING TO RUN: connected to '{actual}' but expected "
                f"'{self.name}'."
            )
        if actual == settings.postgres_db:
            raise UnsafeTestDatabase(
                f"REFUSING TO RUN: connected to the application database "
                f"'{actual}'."
            )

    async def create_schema(self, statements: List[str]) -> None:
        """Apply the migration DDL. Every statement is `IF NOT EXISTS`."""
        async with self.engine.begin() as conn:
            for statement in statements:
                await conn.execute(text(statement))

    async def truncate(self) -> None:
        """
        Empty only the two tables these tests own.

        Named explicitly rather than discovered, so a table added to the test
        database by something else is never in scope.
        """
        await self.verify_identity()
        async with self.engine.begin() as conn:
            await conn.execute(
                text(f"TRUNCATE TABLE {', '.join(OWNED_TABLES)} RESTART IDENTITY")
            )

    async def count(self, table: str) -> int:
        if table not in OWNED_TABLES:
            raise UnsafeTestDatabase(f"'{table}' is not one of the owned tables")
        async with self.engine.connect() as conn:
            return int(await conn.scalar(text(f"SELECT count(*) FROM {table}")))

    async def dispose(self) -> None:
        await self.engine.dispose()


async def connect(name: str | None = None, *, pool_size: int = 5) -> TestDatabase:
    """
    Open a pool onto the dedicated test database, refusing anything else.

    A fresh call yields an independent pool, which is how the tests model a
    separate worker process: nothing is shared between two `connect()` results
    except the database itself.
    """
    resolved = name or test_database_name()
    assert_safe_target(resolved)

    engine = create_async_engine(
        test_database_url(resolved),
        pool_size=pool_size,
        max_overflow=pool_size,
        pool_pre_ping=True,
        echo=False,
    )
    db = TestDatabase(
        name=resolved,
        engine=engine,
        session_maker=async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        ),
    )
    try:
        await db.verify_identity()
    except Exception:
        await engine.dispose()
        raise
    return db


def migration_statements() -> List[str]:
    """
    The DDL from the real migration script.

    Imported rather than re-typed so these tests exercise the statements a
    deployment would actually run — a migration that only works in
    `create_all()` is not a migration.
    """
    import importlib.util
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts" / "migrate_add_action_audit.py"
    )
    spec = importlib.util.spec_from_file_location("_migration_ddl", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return [
        module.CREATE_TABLE,
        *[sql for _, sql in module.INDEXES],
        module.CREATE_PENDING,
        *[sql for _, sql in module.PENDING_INDEXES],
    ]


__all__ = [
    "DB_ENV",
    "DEFAULT_TEST_DB",
    "ENABLE_FLAG",
    "OWNED_TABLES",
    "TestDatabase",
    "UnsafeTestDatabase",
    "assert_safe_target",
    "connect",
    "integration_enabled",
    "migration_statements",
    "test_database_name",
    "test_database_url",
]
