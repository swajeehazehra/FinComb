"""Timing targets (plan clarification C2), measured on Linux.

Ordinary strategies: under 2 s for 20 years x 50 assets. `optimized` is dominated by
Milestone 4's optimiser and has its own targets; the 50-asset max-Sharpe case (about
90 s) is in scripts/bench_backtest.py, not here. Skip all of these with
`pytest -m "not performance"`.
"""

from __future__ import annotations

import time

import pytest

from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.engine import run
from fincomb.strategies.buy_and_hold import BuyAndHold
from fincomb.strategies.fixed_weights import FixedWeights
from fincomb.strategies.mean_reversion import MeanReversion
from fincomb.strategies.momentum import Momentum
from fincomb.strategies.optimized import Optimized
from tests.unit.backtest_helpers import make_prices

pytestmark = pytest.mark.performance
DAYS = 20 * 252 + 253  # 20 evaluation years plus a year of warm-up


@pytest.fixture(scope="module")
def prices50():
    return make_prices(DAYS, 50, seed=7, start="2004-01-01")


def _timed(prices, strategy, **kw):
    t = time.perf_counter()
    run(prices, strategy, prices.index[253], costs=FixedBpsCost(5, 5), **kw)
    return time.perf_counter() - t


@pytest.mark.parametrize(
    "make",
    [
        lambda s: BuyAndHold(s),
        lambda s: FixedWeights(s),
        lambda s: Momentum(s, {"top_n": 10}),
        lambda s: MeanReversion(s),
    ],
    ids=["buy_and_hold", "fixed_weights", "momentum", "mean_reversion"],
)
def test_ordinary_strategies_under_two_seconds(prices50, make):
    elapsed = _timed(prices50, make(list(prices50.columns)))
    assert elapsed < 2.0, f"{elapsed:.2f}s"


@pytest.mark.parametrize("method", ["equal_weight", "inverse_vol", "min_variance"])
def test_optimized_cheap_methods_under_ten_seconds(prices50, method):
    elapsed = _timed(prices50, Optimized(list(prices50.columns), {"method": method}))
    assert elapsed < 10.0, f"{elapsed:.2f}s"


def test_optimized_max_sharpe_ten_assets_under_fifteen_seconds():
    prices = make_prices(DAYS, 10, seed=8, start="2004-01-01")
    elapsed = _timed(prices, Optimized(list(prices.columns), {"method": "max_sharpe"}))
    assert elapsed < 15.0, f"{elapsed:.2f}s"
