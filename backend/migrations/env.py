"""
backend/migrations/env.py

Alembic migration environment configuration for the Authentix backend.
Configured for async SQLAlchemy with asyncpg (PostgreSQL).

Key features:
  - Reads DATABASE_URL from the application Settings singleton.
  - Imports all ORM models via backend.app.database.models to enable
    autogenerate (--autogenerate flag on `alembic revision`).
  - Supports both offline and online migration modes.
"""

import asyncio
from logging.config import fileConfig
import sys
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# ---------------------------------------------------------------------------
# Ensure project root is in sys.path so backend.app.* imports resolve
# ---------------------------------------------------------------------------
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ---------------------------------------------------------------------------
# Load Alembic config object (reads from alembic.ini)
# ---------------------------------------------------------------------------
config = context.config

# Setup Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Import ALL ORM models so Alembic can diff against Base.metadata
# ---------------------------------------------------------------------------
from backend.app.database.base import Base
import backend.app.database.models  # noqa: F401 — registers all mapped tables

target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Override DATABASE_URL from application Settings
# (takes precedence over alembic.ini sqlalchemy.url)
# ---------------------------------------------------------------------------
from backend.app.config import get_settings as _get_settings
_settings = _get_settings()
config.set_main_option("sqlalchemy.url", _settings.DATABASE_URL)


# ---------------------------------------------------------------------------
# Offline migration mode: emit SQL script without a live DB connection
# ---------------------------------------------------------------------------
def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    Generates SQL statements to stdout without connecting to the database.
    Useful for reviewing changes before applying them.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online migration mode: connect to live DB and apply migrations
# ---------------------------------------------------------------------------
def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Creates an async engine and runs migrations within an async context.
    Required for asyncpg driver compatibility.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live PostgreSQL connection."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
