"""
Database initialization script.
Creates PostgreSQL database and tables for short-term memory.
"""
import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

# Fix Windows console encoding
if sys.platform == "win32":
    os.system('')  # Enable ANSI escape codes
    sys.stdout.reconfigure(encoding='utf-8')

from app.config import settings  # noqa: E402
from app.db import Base  # noqa: E402


async def create_database():
    """Create the database if it doesn't exist."""
    # Connect to postgres default database
    default_url = f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/postgres"

    engine = create_async_engine(default_url, isolation_level="AUTOCOMMIT")

    try:
        async with engine.connect() as conn:
            # Check if database exists
            result = await conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{settings.postgres_db}'")
            )
            exists = result.fetchone()

            if not exists:
                print(f"Creating database '{settings.postgres_db}'...")
                await conn.execute(text(f"CREATE DATABASE {settings.postgres_db}"))
                print(f"[OK] Database '{settings.postgres_db}' created successfully")
            else:
                print(f"[OK] Database '{settings.postgres_db}' already exists")

    except Exception as e:
        print(f"[ERROR] Error creating database: {str(e)}")
        print("\nMake sure PostgreSQL is running and credentials are correct.")
        sys.exit(1)
    finally:
        await engine.dispose()


async def create_tables():
    """Create all tables defined in the models."""
    engine = create_async_engine(settings.postgres_url, echo=True)

    try:
        print("\nCreating tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("[OK] All tables created successfully")

    except Exception as e:
        print(f"[ERROR] Error creating tables: {str(e)}")
        sys.exit(1)
    finally:
        await engine.dispose()


async def main():
    """Main initialization routine."""
    print("=" * 60)
    print("PostgreSQL Database Initialization")
    print("=" * 60)
    print(f"\nHost: {settings.postgres_host}:{settings.postgres_port}")
    print(f"Database: {settings.postgres_db}")
    print(f"User: {settings.postgres_user}")
    print()

    # Step 1: Create database
    await create_database()

    # Step 2: Create tables
    await create_tables()

    print("\n" + "=" * 60)
    print("[SUCCESS] Database initialization complete!")
    print("=" * 60)
    print("\nYour PostgreSQL database is ready for use.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        sys.exit(1)
