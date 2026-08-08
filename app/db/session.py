"""
Process-wide async database engine and session factory.

Every repository — memory and domain alike — shares this one engine. That is
not incidental: the LiveKit voice worker runs as a task inside the FastAPI
process, so a second engine would mean a second connection pool competing for
the same Postgres connection limit. Engine lifecycle belongs to the application
lifespan handler and nowhere else (see the note in `app/livekit_worker.py`).

Pool sizing rationale: each chat turn issues several concurrent writes and each
LiveKit participant holds its own workflow, so SQLAlchemy's 5+10 default is
exhausted under modest concurrency. `pool_timeout` makes exhaustion fail fast
and loudly instead of hanging the request.
"""
import logging

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)


engine = create_async_engine(
    settings.postgres_url,
    echo=False,  # SQL echo is far too verbose for request-path logging
    pool_pre_ping=True,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    pool_timeout=settings.postgres_pool_timeout,
    pool_recycle=settings.postgres_pool_recycle,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """
    Create any missing tables.

    Both model modules are imported here rather than at module scope: they
    import `Base` from `app.db.base`, so importing them at the top of this file
    would be a cycle. Importing them inside the function is also what
    *guarantees* every table is registered on `Base.metadata` before
    `create_all` runs — a model module that is never imported is a table that
    is never created.
    """
    import app.memory.models  # noqa: F401  (registers memory tables)
    import app.domain.models  # noqa: F401  (registers application tables)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Close every pooled connection. Called once, from the app lifespan."""
    await engine.dispose()
