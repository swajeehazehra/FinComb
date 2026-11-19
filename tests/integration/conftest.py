"""Integration fixtures: a real Postgres test database, migrated with Alembic.

Uses POSTGRES_TEST_DB (default fincomb_test) with the credentials from .env.
All integration tests are skipped if that database can't be reached.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from fincomb.config import get_settings
from fincomb.db.base import make_engine, make_session_factory

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(url) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["url"] = url
    return cfg


@pytest.fixture(scope="session")
def engine():
    url = get_settings().database_url(test=True)
    eng = make_engine(url)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except OperationalError as exc:
        pytest.skip(
            f"test database not reachable ({exc.orig}); start it with `docker compose up -d db`"
        )

    cfg = alembic_config(url)
    command.downgrade(cfg, "base")  # start from an empty schema every session
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE portfolio_weights, portfolios, data_quality_issues, price_bars, "
                "ingestion_runs, "
                "instruments RESTART IDENTITY CASCADE"
            )
        )
    return make_session_factory(engine)
