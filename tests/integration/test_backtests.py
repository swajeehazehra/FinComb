"""Backtest service against the migrated test database (fixture CSVs, no network)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest
from alembic import command
from sqlalchemy import text

from fincomb.db.base import session_scope
from fincomb.exceptions import BacktestNotFoundError, InsufficientDataError, UnknownStrategyError
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.services import backtest_service as svc
from fincomb.services import portfolio_service
from fincomb.services.market_data import load_prices

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNM"]
BENCH = "SYNM"
FAST = {
    "momentum": {"lookback": 60, "skip": 5, "top_n": 1},
    "optimized": {"lookback": 60},
    "mean_reversion": {"window": 20, "entry_z": 1.5},
    "buy_and_hold": {},
    "fixed_weights": {},
}


@pytest.fixture
def loaded(session_factory, fixtures_dir):
    run_ingestion(session_factory, CsvSource(fixtures_dir), SYMBOLS, START, END)
    return session_factory


def spec(strategy="fixed_weights", **kw):
    base = {
        "strategy": strategy,
        "symbols": ["SYNA", "SYNB"],
        "params": FAST.get(strategy, {}),
        "benchmark": BENCH,
    }
    base.update(kw)
    return svc.BacktestSpec(**base)


def run(factory, s):
    with session_scope(factory) as session:
        return svc.run_backtest(session, s)


# --- schema ---------------------------------------------------------------------------------


def test_migration_created_the_tables(engine):
    with engine.connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
        )
        revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert {"backtest_runs", "backtest_equity", "backtest_trades"} <= tables
    assert revision == "0004_backtests"


def test_migration_downgrade_and_upgrade(engine):
    from tests.integration.conftest import alembic_config

    cfg = alembic_config(engine.url)
    command.downgrade(cfg, "0003_portfolios")
    try:
        with engine.connect() as conn:
            tables = set(
                conn.execute(
                    text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                ).scalars()
            )
        assert "backtest_runs" not in tables and "portfolios" in tables
    finally:
        command.upgrade(cfg, "head")


# --- the acceptance identity (C1) -----------------------------------------------------------


@pytest.mark.parametrize("execution", ["next_close", "same_close"])
def test_zero_cost_buy_and_hold_of_the_benchmark_matches_it(loaded, execution):
    out = run(
        loaded,
        spec("buy_and_hold", symbols=[BENCH], cost_bps=0, slippage_bps=0, execution=execution),
    )
    eq = out.equity
    assert np.array_equal(eq["equity"].to_numpy(), eq["benchmark_equity"].to_numpy())
    m = out.record.metrics
    assert m["strategy"]["total_return"] == m["benchmark"]["total_return"]
    assert m["relative"]["excess_total_return"] == 0
    prices = out.result.equity
    with session_scope(loaded) as s:
        p = load_prices(s, [BENCH])[BENCH].loc[prices.index[0] :]
    np.testing.assert_allclose(eq["equity"], 100_000 * p / p.iloc[0], rtol=1e-12)


# --- running and storing --------------------------------------------------------------------


@pytest.mark.parametrize("strategy", list(FAST))
def test_every_strategy_runs_and_is_stored(loaded, strategy):
    out = run(loaded, spec(strategy, symbols=SYMBOLS[:2] if strategy != "optimized" else SYMBOLS))
    r = out.record
    assert r.id is not None and r.created_at is not None
    assert r.strategy == strategy
    assert len(r.data_hash) == 64 and r.code_version
    with session_scope(loaded) as s:
        stored = svc.get_backtest(s, r.id)
        equity = svc.backtest_equity(s, r.id)
        trades, total = svc.backtest_trades(s, r.id)
    assert stored.metrics == r.metrics
    assert len(equity) == len(out.result.equity)
    np.testing.assert_allclose(equity["equity"], out.result.equity, rtol=1e-12)
    assert total == len(trades) == len(out.result.trades)
    assert all((t.kind == "init") == (t.date == r.start) for t in trades)
    if strategy != "mean_reversion":  # mean reversion may start fully in cash
        assert trades[0].kind == "init"


def test_warmup_moves_the_start_with_a_note(loaded):
    out = run(loaded, spec("momentum", start=date(2023, 1, 2)))
    r = out.record
    with session_scope(loaded) as s:
        dates = load_prices(s, ["SYNA", "SYNB", BENCH]).index
    # lookback 60 under next_close: the first signal needs 61 rows, so day 0 is row 61.
    assert r.start == dates[61].date()
    assert r.warmup_start == dates[0].date()
    assert any("Effective start moved" in n for n in r.notes)


def test_warmup_is_loaded_before_start(loaded):
    with session_scope(loaded) as s:
        dates = load_prices(s, ["SYNA", "SYNB", BENCH]).index
    start = dates[150].date()
    r = run(loaded, spec("momentum", start=start)).record
    assert r.start == start
    assert r.warmup_start == dates[150 - 61].date()
    assert not any("Effective start" in n for n in r.notes)


def test_no_save_writes_nothing(loaded):
    out = run(loaded, spec(save=False))
    assert out.record.id is None
    with session_scope(loaded) as s:
        assert svc.list_backtests(s) == []


def test_list_filter_and_delete_cascades(loaded):
    a = run(loaded, spec("fixed_weights")).record
    b = run(loaded, spec("buy_and_hold")).record
    with session_scope(loaded) as s:
        assert [r.id for r in svc.list_backtests(s)] == [b.id, a.id]
        assert [r.id for r in svc.list_backtests(s, "fixed-weights")] == [a.id]
        svc.delete_backtest(s, a.id)
    with session_scope(loaded) as s:
        n_eq = s.execute(
            text("SELECT count(*) FROM backtest_equity WHERE run_id = :i"), {"i": a.id}
        ).scalar()
        n_tr = s.execute(
            text("SELECT count(*) FROM backtest_trades WHERE run_id = :i"), {"i": a.id}
        ).scalar()
        assert n_eq == n_tr == 0
        with pytest.raises(BacktestNotFoundError):
            svc.get_backtest(s, a.id)


def test_costs_lower_the_result(loaded):
    free = run(loaded, spec(cost_bps=0, slippage_bps=0, save=False)).record.metrics
    paid = run(loaded, spec(cost_bps=20, slippage_bps=20, save=False)).record.metrics
    assert paid["strategy"]["total_return"] < free["strategy"]["total_return"]
    assert paid["trading"]["total_turnover"] == pytest.approx(free["trading"]["total_turnover"])


# --- saved portfolios -----------------------------------------------------------------------


def test_backtest_a_saved_portfolio(loaded):
    with session_scope(loaded) as s:
        p = portfolio_service.create_manual(s, "60-40", {"SYNA": 0.6, "SYNB": 0.4})
    r = run(loaded, spec("fixed_weights", symbols=[], portfolio="60-40")).record
    assert r.portfolio_id == p.id
    assert r.params["weights"] == {"SYNA": 0.6, "SYNB": 0.4}
    assert r.symbols == ["SYNA", "SYNB"]
    with session_scope(loaded) as s:
        portfolio_service.delete_portfolio(s, p.id)
    with session_scope(loaded) as s:
        assert svc.get_backtest(s, r.id).portfolio_id is None


def test_constructed_portfolio_overlap_warns_in_sample(loaded):
    with session_scope(loaded) as s:
        portfolio_service.create_constructed(
            s, "MV", portfolio_service.ConstructionSpec(["SYNA", "SYNB"], "min_variance")
        )
    r = run(loaded, spec("fixed_weights", symbols=[], portfolio="MV")).record
    assert any("in-sample" in w for w in r.warnings)


def test_portfolio_errors(loaded):
    with session_scope(loaded) as s:
        portfolio_service.create_manual(s, "P", {"SYNA": 1.0})
    with pytest.raises(ValueError, match="only be backtested"):
        run(loaded, spec("momentum", portfolio="P"))
    with pytest.raises(ValueError, match="holds"):
        run(loaded, spec("fixed_weights", symbols=["SYNB"], portfolio="P"))
    with pytest.raises(ValueError, match="not both"):
        run(
            loaded,
            spec("fixed_weights", symbols=[], portfolio="P", params={"weights": {"SYNA": 1.0}}),
        )


# --- input errors ---------------------------------------------------------------------------


def test_input_errors(loaded):
    with pytest.raises(UnknownStrategyError):
        run(loaded, spec("martingale"))
    with pytest.raises(ValueError, match="never rebalances"):
        run(loaded, spec("buy_and_hold", rebalance="monthly"))
    with pytest.raises(ValueError, match="unknown parameter"):
        run(loaded, spec("momentum", params={"lookbak": 10}))
    with pytest.raises(ValueError, match="execution"):
        run(loaded, spec(execution="next_open"))
    with pytest.raises(InsufficientDataError):
        run(loaded, spec("momentum", params={"lookback": 400}))
    with pytest.raises(InsufficientDataError):
        run(loaded, spec(start=date(2030, 1, 1)))


# --- reproducibility ------------------------------------------------------------------------


def test_rerun_is_identical_and_verify_passes(loaded):
    first = run(loaded, spec("momentum"))
    again = run(loaded, replace(svc.spec_from_record(first.record), save=False))
    assert again.record.data_hash == first.record.data_hash
    assert again.record.metrics == first.record.metrics
    with session_scope(loaded) as s:
        v = svc.verify_backtest(s, first.record.id)
    assert v.reproducible and v.data_hash_matches and v.metrics_match
    assert v.mismatched_metrics == []


def test_verify_detects_changed_data(loaded):
    r = run(loaded, spec("fixed_weights")).record
    with session_scope(loaded) as s:
        s.execute(
            text(
                "UPDATE price_bars SET adj_close = adj_close * 1.01 "
                "WHERE bar_date = :d AND instrument_id = "
                "(SELECT id FROM instruments WHERE symbol = 'SYNA')"
            ),
            {"d": date(2023, 6, 1)},
        )
    with session_scope(loaded) as s:
        v = svc.verify_backtest(s, r.id)
    assert not v.reproducible and not v.data_hash_matches
    assert v.mismatched_metrics  # the curve changed, so metrics changed too
