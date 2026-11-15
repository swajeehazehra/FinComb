"""Alembic environment. Reads the DB URL from application settings (env vars)."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from fincomb.config import get_settings
from fincomb.db import models  # noqa: F401 - registers tables on Base.metadata
from fincomb.db.base import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tests pass an explicit URL via `config.attributes["url"]`.
url = config.attributes.get("url") or get_settings().database_url()
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(url=str(url), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": url}, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
