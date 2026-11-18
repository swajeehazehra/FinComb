"""Portfolio tables, repository and service against the migrated test database."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from fincomb.db.base import session_scope
from fincomb.db.models import PortfolioMethod
from fincomb.db.repositories import InstrumentRepository, PortfolioRepository
from fincomb.exceptions import (
    InsufficientDataError,
    PortfolioExistsError,
    PortfolioNotFoundError,
    SymbolNotFoundError,
)
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.portfolio.construction import construct
from fincomb.services import portfolio_service as svc
from fincomb.services.market_data import load_prices

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNM"]


@pytest.fixture
def loaded(session_factory, fixtures_dir):
    run_ingestion(session_factory, CsvSource(fixtures_dir), SYMBOLS, START, END)
    return session_factory


def spec(method="min_variance", **kw):
    return svc.ConstructionSpec(symbols=SYMBOLS, method=method, **kw)


# --- schema / repository --------------------------------------------------------------------


def test_migration_created_the_tables(engine):
    with engine.connect() as conn:
        tables = set(
            conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            ).scalars()
        )
        labels = set(
            conn.execute(text("SELECT unnest(enum_range(NULL::portfolio_method))::text")).scalars()
        )
    assert {"portfolios", "portfolio_weights"} <= tables
    assert labels == {m.value for m in PortfolioMethod}


def test_repository_round_trip_and_cascade(loaded):
    with session_scope(loaded) as s:
        p = PortfolioRepository(s).create(
            "mix", {"SYNA": 0.6, "SYNB": 0.4}, PortfolioMethod.MANUAL, PortfolioMethod.MANUAL
        )
        pid = p.id
    with session_scope(loaded) as s:
        repo = PortfolioRepository(s)
        assert repo.weights_of(repo.get(pid)) == {"SYNA": 0.6, "SYNB": 0.4}
        assert repo.get_by_name("mix").id == pid
        repo.delete(repo.get(pid))
    with session_scope(loaded) as s:
        n = s.execute(text("SELECT count(*) FROM portfolio_weights")).scalar_one()
    assert n == 0


def test_database_rejects_short_weights_and_deleting_a_used_instrument(loaded):
    with pytest.raises(IntegrityError), session_scope(loaded) as s:
        PortfolioRepository(s).create(
            "short", {"SYNA": 1.5, "SYNB": -0.5}, PortfolioMethod.MANUAL, PortfolioMethod.MANUAL
        )
    with session_scope(loaded) as s:
        PortfolioRepository(s).create(
            "uses-a", {"SYNA": 1.0}, PortfolioMethod.MANUAL, PortfolioMethod.MANUAL
        )
    with pytest.raises(IntegrityError), session_scope(loaded) as s:
        s.execute(text("DELETE FROM instruments WHERE symbol = 'SYNA'"))


# --- construction --------------------------------------------------------------------------


def test_construct_every_method(loaded):
    with session_scope(loaded) as s:
        for method in ("equal_weight", "inverse_vol", "min_variance", "max_sharpe"):
            out = svc.construct_portfolio(s, spec(method, max_weight=0.6))
            w = out.result.weights
            assert list(w.index) == SYMBOLS
            assert w.sum() == pytest.approx(1.0, abs=1e-9)
            assert w.max() <= 0.6 + 1e-9
            assert out.contributions["percent"].sum() == pytest.approx(1.0)


def test_min_variance_beats_equal_weight_on_fixtures(loaded):
    """The architecture's 'done' criterion for Milestone 4."""
    with session_scope(loaded) as s:
        prices = load_prices(s, SYMBOLS, START, END)
        cov = prices.pct_change().iloc[1:].cov().to_numpy()
        mv = svc.construct_portfolio(s, spec("min_variance")).result.weights.to_numpy()
        ew = svc.construct_portfolio(s, spec("equal_weight")).result.weights.to_numpy()
    assert mv @ cov @ mv <= ew @ cov @ ew


def test_lookback_takes_the_last_returns(loaded):
    with session_scope(loaded) as s:
        out = svc.construct_portfolio(s, spec("equal_weight", lookback=100))
        prices = load_prices(s, SYMBOLS, START, END)
    assert out.result.observations == 100
    assert out.result.end == prices.index[-1]
    assert out.result.start == prices.index[-101]


def test_explicit_start_ignores_lookback_and_matches_pure_function(loaded):
    with session_scope(loaded) as s:
        out = svc.construct_portfolio(
            s, spec("max_sharpe", start=date(2023, 3, 1), end=date(2023, 10, 31), lookback=5)
        )
        prices = load_prices(s, SYMBOLS, date(2023, 3, 1), date(2023, 10, 31))
    direct = construct(prices, "max_sharpe")
    np.testing.assert_allclose(out.result.weights, direct.weights)


def test_short_window_note_and_insufficient_data(loaded):
    with session_scope(loaded) as s:
        out = svc.construct_portfolio(s, spec("equal_weight", lookback=1000))
        assert any("fewer than the lookback" in n for n in out.notes)
        with pytest.raises(InsufficientDataError):
            svc.construct_portfolio(s, spec("min_variance", start=date(2023, 12, 1)))


def test_construct_input_errors(loaded):
    with session_scope(loaded) as s:
        with pytest.raises(SymbolNotFoundError):
            svc.construct_portfolio(s, svc.ConstructionSpec(["SYNA", "NOPE"], "min_variance"))
        with pytest.raises(ValueError, match="only once"):
            svc.construct_portfolio(s, svc.ConstructionSpec(["SYNA", "syna"], "min_variance"))
        with pytest.raises(ValueError, match="too small"):
            svc.construct_portfolio(s, spec("min_variance", max_weight=0.3))


# --- saving ---------------------------------------------------------------------------------


def test_create_manual_and_reproducible_constructed(loaded):
    with session_scope(loaded) as s:
        manual = svc.create_manual(s, "60-40", {"syna": 0.6, "SYNB": 0.4}, "classic")
        built, outcome = svc.create_constructed(s, "MV", spec("min_variance", max_weight=0.7))
    assert manual.weights == {"SYNA": 0.6, "SYNB": 0.4}
    assert manual.construction_method == manual.effective_method == "manual"
    assert built.construction_method == "min_variance"
    assert built.estimation_start == outcome.result.start.date()
    params = built.construction_params
    assert params["covariance"] == "ledoit_wolf" and params["max_weight"] == 0.7
    assert 0 <= params["shrinkage"] <= 1
    # Re-running with the stored window and parameters reproduces the weights exactly.
    with session_scope(loaded) as s:
        again = svc.construct_portfolio(
            s,
            spec(
                "min_variance",
                start=built.estimation_start,
                end=built.estimation_end,
                max_weight=params["max_weight"],
            ),
        )
    for sym, w in built.weights.items():
        assert again.result.weights[sym] == pytest.approx(w, abs=1e-12)


def test_saved_portfolio_keeps_the_warning(loaded):
    with session_scope(loaded) as s:
        rec, _ = svc.create_constructed(s, "hopeless", spec("max_sharpe", risk_free=0.9))
    assert rec.construction_params["non_positive_excess_return"] is True
    assert rec.effective_method == "max_sharpe"
    with session_scope(loaded) as s:
        again = svc.get_portfolio(s, "hopeless")
    assert any("not an attractive investment" in w for w in again.warnings)


def test_manual_weight_validation(loaded):
    with session_scope(loaded) as s:
        with pytest.raises(ValueError, match="sum to 0.980000"):
            svc.create_manual(s, "x", {"SYNA": 0.58, "SYNB": 0.4})
        with pytest.raises(ValueError, match="long-only"):
            svc.create_manual(s, "x", {"SYNA": 1.2, "SYNB": -0.2})
        with pytest.raises(ValueError, match="only once"):
            svc.create_manual(s, "x", {"SYNA": 0.5, "syna": 0.5})
        with pytest.raises(SymbolNotFoundError):
            svc.create_manual(s, "x", {"SYNA": 0.5, "NOPE": 0.5})
        with pytest.raises(ValueError, match="name"):
            svc.create_manual(s, "  ", {"SYNA": 1.0})


def test_duplicate_name_and_lookup(loaded):
    with session_scope(loaded) as s:
        rec = svc.create_manual(s, "dup", {"SYNA": 1.0})
    with session_scope(loaded) as s:
        with pytest.raises(PortfolioExistsError):
            svc.create_manual(s, "dup", {"SYNB": 1.0})
        assert svc.get_portfolio(s, rec.id).name == "dup"
        assert svc.get_portfolio(s, str(rec.id)).name == "dup"
        assert [p.name for p in svc.list_portfolios(s)] == ["dup"]
        with pytest.raises(PortfolioNotFoundError):
            svc.get_portfolio(s, "missing")
        svc.delete_portfolio(s, "dup")
        with pytest.raises(PortfolioNotFoundError):
            svc.delete_portfolio(s, rec.id)


def test_instrument_rows_unchanged_by_portfolios(loaded):
    with session_scope(loaded) as s:
        before = InstrumentRepository(s).all_symbols()
        svc.create_manual(s, "p", {"SYNA": 1.0})
        assert InstrumentRepository(s).all_symbols() == before


# --- risk ---------------------------------------------------------------------------------


def test_portfolio_risk_report(loaded):
    with session_scope(loaded) as s:
        svc.create_manual(s, "mix", {"SYNA": 0.5, "SYNB": 0.3, "SYNM": 0.2})
        out = svc.portfolio_risk(s, "mix", benchmark="SYNM", confidence=0.95, risk_free=0.02)
    rep = out.report
    assert out.portfolio.name == "mix"
    assert rep.benchmark == "SYNM" and rep.beta is not None
    assert rep.cvar_historical >= rep.var_historical > 0
    assert rep.contributions["component"].sum() == pytest.approx(rep.volatility)
    assert rep.observations == 259


def test_risk_with_benchmark_outside_portfolio_and_without_one(loaded):
    with session_scope(loaded) as s:
        out = svc.weights_risk(s, {"SYNA": 0.5, "SYNB": 0.5}, benchmark="SYNM")
        assert out.portfolio is None and out.report.beta is not None
        assert list(out.report.contributions.index) == ["SYNA", "SYNB"]
        none = svc.weights_risk(s, {"SYNA": 1.0}, benchmark=None)
        assert none.report.beta is None
        with pytest.raises(SymbolNotFoundError):
            svc.weights_risk(s, {"SYNA": 1.0}, benchmark="SPY")  # not stored in the tests
