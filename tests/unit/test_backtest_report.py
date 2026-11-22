"""Report metrics against independent computations, and the reproducibility helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fincomb.analytics.risk import beta_alpha
from fincomb.analytics.summary import performance_summary
from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.engine import benchmark_curve, run
from fincomb.backtest.report import build_report, hac_regression, newey_west_lags
from fincomb.backtest.reproducibility import CODE_VERSION_ENV, code_version, data_hash
from fincomb.strategies.buy_and_hold import BuyAndHold
from fincomb.strategies.fixed_weights import FixedWeights
from fincomb.strategies.momentum import Momentum
from tests.unit.backtest_helpers import make_prices

CAP = 100_000.0


DEFAULT_COSTS = FixedBpsCost(5, 5)


def _run(costs=DEFAULT_COSTS):
    prices = make_prices(700, 5, seed=2)
    syms = ["S0", "S1", "S2", "S3"]
    strat = Momentum(syms, {"lookback": 120, "skip": 10, "top_n": 2})
    start = prices.index[150]
    res = run(prices[syms], strat, start, initial_capital=CAP, costs=costs)
    bench = benchmark_curve(prices["S4"], start, CAP)
    return res, bench


def test_strategy_stats_match_performance_summary_with_zero_costs():
    res, bench = _run(FixedBpsCost())
    rep = build_report(res, bench, risk_free=0.02)
    ref = performance_summary(res.equity, 0.02).iloc[0]
    for key, value in rep.strategy.items():
        assert value == pytest.approx(float(ref[key]), rel=1e-12), key
    bref = performance_summary(bench, 0.02).iloc[0]
    assert rep.benchmark["sharpe"] == pytest.approx(float(bref["sharpe"]), rel=1e-12)


def test_total_return_includes_the_initial_cost():
    res, bench = _run(FixedBpsCost(10, 10))
    rep = build_report(res, bench)
    assert rep.strategy["total_return"] == pytest.approx(res.equity.iloc[-1] / CAP - 1, rel=1e-12)
    assert rep.trading["initial_turnover"] == pytest.approx(1.0)
    assert rep.trading["total_costs"] == pytest.approx(res.costs.sum())
    assert rep.trading["costs_pct_of_capital"] == pytest.approx(res.costs.sum() / CAP)


def test_tracking_error_and_information_ratio_against_numpy():
    res, bench = _run()
    rep = build_report(res, bench)
    rs = np.diff(res.equity.to_numpy()) / res.equity.to_numpy()[:-1]
    rb = np.diff(bench.to_numpy()) / bench.to_numpy()[:-1]
    a = rs - rb
    te = a.std(ddof=1) * math.sqrt(252)
    assert rep.relative["tracking_error"] == pytest.approx(te, rel=1e-10)
    assert rep.relative["information_ratio"] == pytest.approx(a.mean() * 252 / te, rel=1e-10)
    assert rep.relative["correlation"] == pytest.approx(np.corrcoef(rs, rb)[0, 1], rel=1e-10)


def test_hac_beta_and_alpha_equal_the_closed_form():
    res, bench = _run()
    rs, rb = res.equity.pct_change().iloc[1:], bench.pct_change().iloc[1:]
    reg = hac_regression(rs, rb, risk_free=0.03)
    ref = beta_alpha(rs, rb, risk_free=0.03)
    assert reg.beta == pytest.approx(ref.beta, rel=1e-10)
    assert reg.alpha == pytest.approx(ref.alpha, rel=1e-8, abs=1e-12)
    assert reg.r_squared == pytest.approx(ref.r_squared, rel=1e-10)


def test_newey_west_errors_match_a_manual_sandwich():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(0, 0.01, n)
    e = np.convolve(rng.normal(0, 0.005, n + 2), [1, 0.5, 0.25], mode="valid")
    y = 0.0002 + 0.8 * x + e
    idx = pd.bdate_range("2020-01-01", periods=n)
    reg = hac_regression(pd.Series(y, idx), pd.Series(x, idx))
    L = newey_west_lags(n)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    S = (X * u[:, None]).T @ (X * u[:, None])
    for lag in range(1, L + 1):
        w = 1 - lag / (L + 1)
        G = (X[lag:] * u[lag:, None]).T @ (X[:-lag] * u[:-lag, None])
        S += w * (G + G.T)
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ S @ bread  # no small-sample correction (statsmodels default for HAC)
    se = np.sqrt(np.diag(cov))
    assert reg.beta_t == pytest.approx(beta[1] / se[1], rel=1e-8)
    assert reg.alpha_t == pytest.approx(beta[0] / se[0], rel=1e-8)
    assert reg.hac_lags == L == 5


def test_regression_on_itself_is_beta_one_with_null_t_stats_allowed():
    prices = make_prices(300, 1)
    res = run(prices, BuyAndHold(["S0"]), None, initial_capital=CAP)
    bench = benchmark_curve(prices["S0"], None, CAP)
    rep = build_report(res, bench)
    assert rep.relative["beta"] == pytest.approx(1.0, rel=1e-12)
    assert rep.relative["tracking_error"] == 0
    assert math.isnan(rep.relative["information_ratio"])
    assert rep.relative["excess_total_return"] == 0
    d = rep.to_dict()
    assert d["relative"]["information_ratio"] is None  # JSON has no NaN


def test_flat_benchmark_gives_nan_regression():
    idx = pd.bdate_range("2024-01-01", periods=50)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 50), idx)
    reg = hac_regression(r, pd.Series(0.0, idx))
    assert math.isnan(reg.beta) and math.isnan(reg.alpha)


def test_report_without_benchmark_and_trading_stats():
    prices = make_prices(300, 3)
    res = run(prices, FixedWeights(["S0", "S1", "S2"]), None, costs=FixedBpsCost(5, 5))
    rep = build_report(res)
    assert rep.benchmark is None and rep.relative is None
    t = rep.trading
    assert t["rebalances"] == int((res.turnover.iloc[1:] > 0).sum()) > 0
    years = (res.equity.index[-1] - res.equity.index[0]).days / 365.25
    assert t["annual_turnover"] == pytest.approx(res.turnover.iloc[1:].sum() / years)
    assert t["liquidation_cost"] == pytest.approx(res.equity.iloc[-1] * 0.001, rel=1e-9)
    assert t["average_cash"] == pytest.approx(0.0, abs=1e-12)


# --- reproducibility ------------------------------------------------------------------------


def test_data_hash_is_deterministic_and_sensitive():
    prices = make_prices(100, 3)
    h = data_hash({"adj": prices, "raw": prices * 2})
    assert h == data_hash({"raw": prices * 2, "adj": prices})
    assert h == data_hash({"adj": prices[["S2", "S0", "S1"]], "raw": prices * 2})
    changed = prices.copy()
    changed.iloc[50, 1] += 1e-9
    assert data_hash({"adj": changed, "raw": prices * 2}) != h
    shifted = prices.copy()
    shifted.index = shifted.index + pd.Timedelta(days=1)
    assert data_hash({"adj": shifted, "raw": prices * 2}) != h
    assert len(h) == 64


def test_code_version_env_override(monkeypatch):
    monkeypatch.setenv(CODE_VERSION_ENV, "abc123")
    assert code_version() == "abc123"
    monkeypatch.delenv(CODE_VERSION_ENV)
    v = code_version()
    assert v == "unknown" or len(v) == 40
