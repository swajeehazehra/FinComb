"""Regression tests for the three data-integrity bugs found in the Milestone 2 review:

1. adjusted prices "stitched" together by incremental ingests,
2. unfinished (in-progress) daily bars being stored,
3. one symbol's history mixing several data sources.
"""

from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest
from alembic import command
from sqlalchemy import func, select, text

from fincomb.db.base import session_scope
from fincomb.db.models import DataQualityIssue, Instrument, RunStatus
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.ingestion.audit import audit_data
from fincomb.ingestion.base import MarketDataSource
from fincomb.ingestion.pipeline import find_adjustment_change, run_ingestion

pytestmark = pytest.mark.integration

# A Wednesday evening in New York: Wednesday 2023-12-27's bar is final.
EVENING = datetime(2023, 12, 27, 23, 0, tzinfo=UTC)
DAYS = pd.bdate_range("2023-01-02", "2023-12-29", name="date")


class FakeSource(MarketDataSource):
    """In-memory source whose history can be re-adjusted, like a real vendor's."""

    def __init__(self, name: str = "yfinance", seed: int = 0) -> None:
        self.name = name
        rng = np.random.default_rng(seed)
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(DAYS)))
        self.frame = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": 1e6,
            },
            index=DAYS,
        )
        self.calls: list[tuple[date, date]] = []

    def _fetch_raw(self, symbol, start, end):
        self.calls.append((start, end))
        return self.frame.reset_index()

    def pay_dividend(self, ex_date: str, pct: float) -> None:
        """Vendors scale ALL adjusted prices before the ex-date down by the dividend."""
        before = self.frame.index < pd.Timestamp(ex_date)
        self.frame.loc[before, "adj_close"] *= 1 - pct

    def split(self, ratio: float) -> None:
        """After a split, shares trade at 1/ratio of the old price, and Yahoo divides the
        whole earlier history (close AND adj_close) by the ratio so it stays continuous."""
        for col in ("open", "high", "low", "close", "adj_close"):
            self.frame[col] /= ratio


def stored(factory, symbol="XYZ") -> pd.DataFrame:
    with session_scope(factory) as s:
        inst = InstrumentRepository(s).get_by_symbol(symbol)
        return PriceRepository(s).bars_for_instrument(inst.id)


def ingest(factory, src, start, end, **kw):
    return run_ingestion(
        factory,
        src,
        ["XYZ"],
        date.fromisoformat(start),
        date.fromisoformat(end),
        now=kw.pop("now", EVENING),
        **kw,
    )


# --- 1. adjusted-price stitching ----------------------------------------------------------


def test_dividend_between_ingests_triggers_full_refresh(session_factory):
    src = FakeSource()
    ingest(session_factory, src, "2023-01-01", "2023-11-30")
    src.pay_dividend("2023-12-08", 0.003)  # 0.3% dividend re-adjusts all earlier history

    summary = ingest(session_factory, src, "2023-12-01", "2023-12-27")
    res = summary.results[0]
    assert summary.status is RunStatus.SUCCESS and res.replaced
    assert any("adjustment changed" in n for n in res.notes)

    db = stored(session_factory)
    expected = src.frame.loc[:"2023-12-27"]
    np.testing.assert_allclose(db["adj_close"], expected["adj_close"], rtol=1e-12)
    # No fake jump at the old/new boundary: stored returns equal the vendor's returns.
    np.testing.assert_allclose(
        db["adj_close"].pct_change().dropna(),
        expected["adj_close"].pct_change().dropna(),
        rtol=1e-9,
    )
    with session_scope(session_factory) as s:
        rules = set(s.scalars(select(DataQualityIssue.rule)))
    assert "adjustment_changed" in rules


def test_split_between_ingests_is_repaired(session_factory):
    src = FakeSource()
    ingest(session_factory, src, "2023-01-01", "2023-11-30")
    src.split(4.0)  # e.g. a 4:1 split on 11 Dec
    ingest(session_factory, src, "2023-12-01", "2023-12-27")
    db = stored(session_factory)
    worst_move = db["adj_close"].pct_change().abs().max()
    assert worst_move < 0.1  # a stitched 4:1 split would show as a -75% "crash"
    np.testing.assert_allclose(db["close"], src.frame.loc[:"2023-12-27", "close"], rtol=1e-12)


def test_unchanged_history_does_not_refresh(session_factory):
    src = FakeSource()
    ingest(session_factory, src, "2023-01-01", "2023-11-30")
    summary = ingest(session_factory, src, "2023-12-01", "2023-12-27")
    res = summary.results[0]
    assert not res.replaced and res.notes == []
    # The update re-downloaded about two weeks of stored days to check them.
    assert src.calls[-1][0] == date(2023, 11, 16)
    assert len(stored(session_factory)) == len(DAYS[DAYS <= "2023-12-27"])


def test_find_adjustment_change_tolerance():
    a = pd.DataFrame(
        {"adj_close": [100.0, 101.0], "close": [100.0, 101.0]},
        index=pd.bdate_range("2024-01-01", periods=2),
    )
    assert find_adjustment_change(a, a * (1 + 1e-7)) is None  # rounding noise
    msg = find_adjustment_change(a, a * 0.999)
    assert msg and "adj_close" in msg and "0.1000%" in msg


# --- 2. unfinished bars -------------------------------------------------------------------


def test_in_progress_bar_is_never_stored(session_factory):
    src = FakeSource()
    market_open = datetime(2023, 12, 27, 15, 0, tzinfo=UTC)  # 10:00 in New York, Wednesday
    summary = ingest(session_factory, src, "2023-12-01", "2023-12-29", now=market_open)
    assert summary.end_capped_to == date(2023, 12, 26)
    assert stored(session_factory).index.max() == pd.Timestamp("2023-12-26")


def test_requesting_only_today_during_market_hours_fails_cleanly(session_factory):
    market_open = datetime(2023, 12, 27, 15, 0, tzinfo=UTC)
    summary = ingest(session_factory, FakeSource(), "2023-12-27", "2023-12-27", now=market_open)
    assert summary.status is RunStatus.FAILED
    assert "after the last completed session" in summary.results[0].error


# --- 3. one source per symbol -------------------------------------------------------------


def test_owner_is_recorded_and_other_sources_refused(session_factory):
    ingest(session_factory, FakeSource("yfinance"), "2023-01-01", "2023-06-30")
    with session_scope(session_factory) as s:
        assert InstrumentRepository(s).get_by_symbol("XYZ").data_source == "yfinance"

    summary = ingest(session_factory, FakeSource("tiingo"), "2023-07-01", "2023-12-27")
    assert summary.status is RunStatus.FAILED
    assert "fincomb refresh XYZ --source tiingo" in summary.results[0].error
    assert set(stored(session_factory).index) <= set(DAYS[DAYS <= "2023-06-30"])


def test_refresh_switches_source_and_replaces_all_rows(session_factory):
    ingest(session_factory, FakeSource("yfinance", seed=1), "2023-01-01", "2023-12-27")
    tiingo = FakeSource("tiingo", seed=2)
    summary = run_ingestion(
        session_factory, tiingo, ["XYZ"], None, date(2023, 12, 27), mode="replace", now=EVENING
    )
    assert summary.status is RunStatus.SUCCESS and summary.results[0].replaced
    with session_scope(session_factory) as s:
        inst = InstrumentRepository(s).get_by_symbol("XYZ")
        assert inst.data_source == "tiingo"
        assert PriceRepository(s).sources(inst.id) == {"tiingo"}
    np.testing.assert_allclose(
        stored(session_factory)["close"], tiingo.frame.loc[:"2023-12-27", "close"]
    )


def test_refresh_keeps_old_data_if_download_fails(session_factory):
    ingest(session_factory, FakeSource(), "2023-01-01", "2023-12-27")
    before = stored(session_factory)

    class Broken(FakeSource):
        def _fetch_raw(self, symbol, start, end):
            raise ConnectionError("network down")

    summary = run_ingestion(
        session_factory, Broken(), ["XYZ"], None, date(2023, 12, 27), mode="replace", now=EVENING
    )
    assert summary.status is RunStatus.FAILED
    pd.testing.assert_frame_equal(stored(session_factory), before)


def _make_legacy_mixed_history(factory):
    """What existing databases look like: AAPL-style history from two sources, no owner."""
    yahoo, tiingo = FakeSource("yfinance"), FakeSource("tiingo")
    ingest(factory, yahoo, "2023-01-01", "2023-12-27")
    with session_scope(factory) as s:
        s.execute(text("UPDATE price_bars SET source = 'tiingo' WHERE bar_date >= '2023-07-01'"))
        s.execute(text("UPDATE instruments SET data_source = NULL"))
    return yahoo, tiingo


def test_legacy_mixed_history_is_reported_refused_and_repaired(session_factory):
    yahoo, _ = _make_legacy_mixed_history(session_factory)
    with session_scope(session_factory) as s:
        (row,) = audit_data(s)
    assert row.status == "MIXED SOURCES" and row.sources == ["tiingo", "yfinance"]

    summary = ingest(session_factory, yahoo, "2023-12-01", "2023-12-27")
    assert summary.status is RunStatus.FAILED
    assert "several sources" in summary.results[0].error

    run_ingestion(
        session_factory, yahoo, ["XYZ"], None, date(2023, 12, 27), mode="replace", now=EVENING
    )
    with session_scope(session_factory) as s:
        (row,) = audit_data(s)
    assert (row.status, row.owner, row.sources) == ("OK", "yfinance", ["yfinance"])
    assert len(stored(session_factory)) == len(DAYS[DAYS <= "2023-12-27"])


def test_audit_flags_suspicious_jumps(session_factory):
    src = FakeSource()
    src.frame.loc["2023-06-01":, ["open", "high", "low", "close", "adj_close"]] /= 4
    src.frame.loc["2023-06-01", "high"] = src.frame.loc["2023-06-01", "close"] * 1.01
    ingest(session_factory, src, "2023-01-01", "2023-12-27")
    with session_scope(session_factory) as s:
        (row,) = audit_data(s)
    assert row.status == "SUSPICIOUS JUMPS" and row.jumps == 1


def test_migration_backfills_owner_only_for_single_source_symbols(engine, session_factory):
    from tests.integration.conftest import alembic_config

    cfg = alembic_config(engine.url)
    command.downgrade(cfg, "0001_market_data")
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO instruments (id, symbol, asset_type, currency, is_active) VALUES "
                    "(1, 'ONE', 'stock', 'USD', true), (2, 'MIX', 'stock', 'USD', true)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO price_bars (instrument_id, bar_date, open, high, low, close, "
                    "adj_close, volume, source) VALUES "
                    "(1, '2024-01-02', 1, 1, 1, 1, 1, 1, 'yfinance'), "
                    "(2, '2024-01-02', 1, 1, 1, 1, 1, 1, 'yfinance'), "
                    "(2, '2024-01-03', 1, 1, 1, 1, 1, 1, 'tiingo')"
                )
            )
    finally:
        command.upgrade(cfg, "head")
    with session_scope(session_factory) as s:
        owners = dict(s.execute(select(Instrument.symbol, Instrument.data_source)).all())
        assert s.scalar(select(func.count()).select_from(Instrument)) == 2
    assert owners == {"ONE": "yfinance", "MIX": None}


# --- CLI ----------------------------------------------------------------------------------


@pytest.fixture
def cli(session_factory, monkeypatch):
    from typer.testing import CliRunner

    from fincomb.config import get_settings

    monkeypatch.setenv("POSTGRES_DB", get_settings().postgres_test_db)
    monkeypatch.setenv("DEFAULT_DATA_SOURCE", "csv")
    get_settings.cache_clear()
    yield CliRunner()
    get_settings.cache_clear()


def test_cli_check_data_and_refresh_repair_mixed_history(cli, session_factory):
    from fincomb.cli import app

    assert (
        cli.invoke(
            app, ["ingest", "SYNA", "SYNB", "--source", "csv", "--start", "2023-01-01"]
        ).exit_code
        == 0
    )
    with session_scope(session_factory) as s:
        s.execute(
            text(
                "UPDATE price_bars SET source = 'tiingo' FROM instruments i WHERE "
                "i.id = price_bars.instrument_id AND i.symbol = 'SYNA' "
                "AND bar_date >= '2023-07-01'"
            )
        )
        s.execute(text("UPDATE instruments SET data_source = NULL WHERE symbol = 'SYNA'"))

    res = cli.invoke(app, ["check-data"])
    assert res.exit_code == 1 and "MIXED SOURCES" in res.output and "fincomb refresh" in res.output

    res = cli.invoke(app, ["ingest", "SYNA", "--start", "2023-01-01"])
    assert res.exit_code == 1 and "several sources" in res.output

    res = cli.invoke(app, ["refresh", "--all"])
    assert res.exit_code == 0, res.output
    assert "history replaced" in res.output

    res = cli.invoke(app, ["check-data"])
    assert res.exit_code == 0 and "consistent history" in res.output


def test_cli_ingest_uses_each_symbols_owner_by_default(cli, session_factory):
    from fincomb.cli import app

    assert (
        cli.invoke(app, ["ingest", "SYNA", "--source", "csv", "--start", "2023-01-01"]).exit_code
        == 0
    )
    # No --source: SYNA keeps its owner (csv) instead of failing on the default.
    res = cli.invoke(app, ["ingest", "SYNA", "--start", "2023-01-01"])
    assert res.exit_code == 0, res.output
