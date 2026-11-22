"""Strategies: parameters, signals, and the look-ahead perturbation test for every one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fincomb.backtest.engine import run
from fincomb.exceptions import UnknownStrategyError
from fincomb.portfolio.construction import WeightBounds, construct
from fincomb.strategies.base import Strategy
from fincomb.strategies.buy_and_hold import BuyAndHold
from fincomb.strategies.fixed_weights import FixedWeights
from fincomb.strategies.mean_reversion import MeanReversion
from fincomb.strategies.momentum import Momentum
from fincomb.strategies.optimized import Optimized
from fincomb.strategies.params import ParamSpec, json_schema, parse_params
from fincomb.strategies.registry import STRATEGIES, describe_all, get_strategy
from tests.unit.backtest_helpers import leaks, make_prices

SYMS = ["S0", "S1", "S2", "S3"]


# --- parameters -----------------------------------------------------------------------------


def test_params_defaults_and_coercion():
    specs = Momentum.param_specs
    p = parse_params(specs, {"lookback": "126", "top_n": 2.0, "mode": "time-series"})
    assert p == {"mode": "time_series", "lookback": 126, "skip": 21, "top_n": 2}


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"lookbak": 10}, "unknown parameter"),
        ({"lookback": 1}, "at least"),
        ({"lookback": 2.5}, "whole number"),
        ({"mode": "sideways"}, "one of"),
        ({"top_n": True}, "whole number"),
    ],
)
def test_params_rejected(given, match):
    with pytest.raises(ValueError, match=match):
        parse_params(Momentum.param_specs, given)


def test_weights_param_from_text():
    spec = (ParamSpec("weights", "weights", None, ""),)
    assert parse_params(spec, {"weights": "spy=0.6, TLT=0.4"}) == {
        "weights": {"SPY": 0.6, "TLT": 0.4}
    }
    with pytest.raises(ValueError, match="twice"):
        parse_params(spec, {"weights": "SPY=0.5,spy=0.5"})


def test_json_schema_shape():
    schema = json_schema(Momentum.param_specs)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["lookback"] == {
        "type": "integer",
        "minimum": 2,
        "maximum": 2520,
        "default": 252,
        "description": "Trading days back to the start of the window",
    }
    assert schema["properties"]["mode"]["enum"] == ["cross_sectional", "time_series"]


def test_registry():
    assert set(STRATEGIES) == {
        "buy_and_hold",
        "fixed_weights",
        "optimized",
        "momentum",
        "mean_reversion",
    }
    assert get_strategy("Mean-Reversion") is MeanReversion
    with pytest.raises(UnknownStrategyError):
        get_strategy("martingale")
    described = {d["name"]: d for d in describe_all()}
    assert described["buy_and_hold"]["default_rebalance"] == "never"
    assert described["fixed_weights"]["uses_prices"] is False


def test_fixed_weights_validation():
    with pytest.raises(ValueError, match="outside the universe"):
        FixedWeights(["A"], {"weights": {"B": 1.0}})
    with pytest.raises(ValueError, match="sum"):
        FixedWeights(["A", "B"], {"weights": {"A": 0.7, "B": 0.7}})
    with pytest.raises(ValueError, match="long-only"):
        FixedWeights(["A", "B"], {"weights": {"A": -0.1, "B": 1.0}})
    w = FixedWeights(["A", "B"]).target_weights(pd.DataFrame(), None)
    assert w.to_dict() == {"A": 0.5, "B": 0.5}


# --- momentum -------------------------------------------------------------------------------


def _trend(slopes, days=300, start="2020-01-01"):
    t = np.arange(days)
    idx = pd.bdate_range(start, periods=days)
    return pd.DataFrame({f"T{i}": 100 * np.exp(s * t) for i, s in enumerate(slopes)}, index=idx)


def test_cross_sectional_picks_strongest_trends():
    prices = _trend([0.001, 0.003, -0.001, 0.002])
    m = Momentum(list(prices.columns), {"lookback": 252, "skip": 21, "top_n": 2})
    w = m.target_weights(prices, None)
    assert w.to_dict() == {"T0": 0.0, "T1": 0.5, "T2": 0.0, "T3": 0.5}


def test_skip_window_is_ignored():
    prices = _trend([0.001, 0.002])
    m = Momentum(list(prices.columns), {"lookback": 252, "skip": 21, "top_n": 1})
    spiked = prices.copy()
    spiked.iloc[-21:, 0] *= 5  # a huge jump inside the skipped month
    assert m.target_weights(spiked, None).equals(m.target_weights(prices, None))


def test_ties_go_to_the_first_symbol():
    prices = _trend([0.001, 0.001, 0.001])
    m = Momentum(list(prices.columns), {"lookback": 100, "skip": 0, "top_n": 2})
    assert m.target_weights(prices, None).to_dict() == {"T0": 0.5, "T1": 0.5, "T2": 0.0}


def test_time_series_goes_to_cash():
    prices = _trend([-0.001, 0.002, -0.002])
    m = Momentum(list(prices.columns), {"mode": "time_series", "lookback": 200, "skip": 10})
    w = m.target_weights(prices, None)
    assert w.to_dict() == {"T0": 0.0, "T1": pytest.approx(1 / 3), "T2": 0.0}
    down = _trend([-0.001, -0.002])
    m2 = Momentum(list(down.columns), {"mode": "time_series", "lookback": 200, "skip": 10})
    assert m2.target_weights(down, None).sum() == 0


def test_momentum_skip_must_be_smaller():
    with pytest.raises(ValueError, match="skip"):
        Momentum(["A"], {"lookback": 20, "skip": 20})


# --- mean reversion -------------------------------------------------------------------------


def _path(values):
    return pd.DataFrame({"A": values}, index=pd.bdate_range("2024-01-01", periods=len(values)))


def test_mean_reversion_enters_and_exits_on_the_expected_days():
    base = [100.0, 101.0] * 10  # 20 calm days
    path = base + [90.0, 95.0, 104.0, 104.0]  # crash, partial recovery, full recovery
    prices = _path(path)
    m = MeanReversion(["A"], {"window": 20, "entry_z": 2.0, "exit_z": 0.0})
    res = run(prices, m, prices.index[19], execution="same_close")
    kinds = [(t.date, t.side) for t in res.trades]
    # day 20 (price 90): z far below -2 -> buy; day 21 (95): still below the mean -> hold;
    # day 22 (104): above the mean -> sell.
    assert kinds == [(prices.index[20], "buy"), (prices.index[22], "sell")]


def test_mean_reversion_no_change_returns_none():
    prices = _path([100.0, 101.0] * 15)
    m = MeanReversion(["A"])
    first = m.target_weights(prices, None)
    assert first.sum() == 0
    assert m.target_weights(prices, first) is None


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 5000), scale=st.floats(0.01, 100), cut=st.integers(30, 150))
def test_mean_reversion_signals_are_scale_invariant(seed, scale, cut):
    prices = make_prices(200, 3, seed=seed)
    adjusted = prices.copy()
    adjusted.iloc[:cut] *= scale  # a re-adjustment rescales all history before a date
    m = MeanReversion(list(prices.columns), {"window": 20})
    a = m.zscores(prices.iloc[:cut])
    b = m.zscores(adjusted.iloc[:cut])
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_momentum_scores_are_scale_invariant():
    prices = make_prices(300, 3)
    m = Momentum(list(prices.columns), {"lookback": 252})
    np.testing.assert_allclose(m.scores(prices), m.scores(prices * 37.5), rtol=1e-12)


def test_mean_reversion_param_check():
    with pytest.raises(ValueError, match="exit_z"):
        MeanReversion(["A"], {"entry_z": 1.0, "exit_z": -1.5})


def test_adf_diagnostic_separates_random_walk_from_ar1():
    rng = np.random.default_rng(3)
    n = 1000
    walk = np.cumsum(rng.normal(0, 0.01, n))
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + rng.normal(0, 0.01)
    idx = pd.bdate_range("2018-01-01", periods=n)
    prices = pd.DataFrame({"W": 100 * np.exp(walk), "R": 100 * np.exp(ar)}, index=idx)
    d = MeanReversion(["W", "R"]).diagnostics(prices)
    assert d["adf"]["W"]["p_value"] > 0.05
    assert d["adf"]["R"]["p_value"] < 0.01
    assert "In-sample" in d["adf_note"]


# --- optimized ------------------------------------------------------------------------------


def test_optimized_equals_construct_on_the_trailing_window():
    prices = make_prices(500, 4)
    opt = Optimized(SYMS, {"method": "min_variance", "lookback": 120, "max_weight": 0.5})
    res = run(prices, opt, prices.index[200], rebalance="monthly")
    for sig, row in zip(res.signal_dates, res.targets.to_numpy(), strict=True):
        k = prices.index.get_loc(sig)
        window = prices.iloc[k - 120 : k + 1]
        expected = construct(window, "min_variance", WeightBounds(0, 0.5))
        np.testing.assert_allclose(row, expected.weights.to_numpy(), atol=1e-12)


def test_optimized_window_ends_at_signal_date():
    seen = []

    class Probe(Optimized):
        def target_weights(self, history, previous):
            seen.append(history.index[-1])
            return super().target_weights(history, previous)

    prices = make_prices(400, 3)
    res = run(prices, Probe(["S0", "S1", "S2"], {"lookback": 100}), prices.index[150])
    assert seen == res.signal_dates


def test_optimized_counts_fallbacks(monkeypatch):
    import fincomb.strategies.optimized as mod

    real = mod.construct

    def failing(*a, **k):
        r = real(*a, **k)
        r.fallback_used = True
        return r

    monkeypatch.setattr(mod, "construct", failing)
    prices = make_prices(400, 3)
    opt = Optimized(["S0", "S1", "S2"], {"lookback": 100})
    run(prices, opt, prices.index[150])
    (msg,) = opt.run_warnings()
    assert "could not be solved reliably" in msg and "inverse volatility" in msg


def test_optimized_lookback_must_cover_min_observations():
    with pytest.raises(ValueError, match="lookback"):
        Optimized(SYMS, {"lookback": 30})


# --- look-ahead -----------------------------------------------------------------------------


def _factories():
    return {
        "buy_and_hold": lambda p: BuyAndHold(SYMS),
        "fixed_weights": lambda p: FixedWeights(SYMS),
        "optimized": lambda p: Optimized(SYMS, {"lookback": 120, "method": "min_variance"}),
        "optimized_max_sharpe": lambda p: Optimized(
            SYMS, {"lookback": 120, "method": "max_sharpe"}
        ),
        "momentum": lambda p: Momentum(SYMS, {"lookback": 120, "skip": 10, "top_n": 2}),
        "momentum_ts": lambda p: Momentum(SYMS, {"mode": "time_series", "lookback": 120}),
        "mean_reversion": lambda p: MeanReversion(SYMS, {"window": 20, "entry_z": 1.0}),
    }


@pytest.mark.parametrize("execution", ["next_close", "same_close"])
@pytest.mark.parametrize("name", list(_factories()))
def test_future_perturbation_changes_nothing_before_the_cut(name, execution):
    prices = make_prices(450, 4, seed=11)
    make = _factories()[name]
    rebalance = "weekly" if name in ("fixed_weights", "momentum") else None
    for cut in (200, 333):
        assert not leaks(make, prices, cut, prices.index[150], execution, rebalance)


def test_every_registered_strategy_is_covered_by_the_perturbation_test():
    covered = {type(f(None)).name for f in _factories().values()}
    assert covered == set(STRATEGIES)


def test_strategy_only_ever_sees_history_up_to_the_signal_date():
    seen = []

    class Probe(Momentum):
        def target_weights(self, history, previous):
            seen.append(history.index[-1])
            return super().target_weights(history, previous)

    prices = make_prices(400, 4)
    for execution in ("next_close", "same_close"):
        seen.clear()
        res = run(
            prices,
            Probe(SYMS, {"lookback": 60, "skip": 5}),
            prices.index[100],
            execution=execution,
            rebalance="weekly",
        )
        assert seen == res.signal_dates
        idx = prices.index
        lag = 1 if execution == "next_close" else 0
        for sig, ex in zip(res.signal_dates[1:], res.targets.index[1:], strict=True):
            assert idx.get_loc(ex) - idx.get_loc(sig) == lag


class _Cheat(Strategy):
    """A deliberately broken strategy that peeks at tomorrow's return through a closure."""

    name = "cheat"
    description = "negative control"
    default_rebalance = "daily"

    def __init__(self, symbols, full_prices):
        super().__init__(symbols)
        self.full = full_prices

    def target_weights(self, history, previous):
        t = self.full.index.get_loc(history.index[-1])
        if t + 1 >= len(self.full):
            return self._weights({})
        tomorrow = self.full.iloc[t + 1] / self.full.iloc[t] - 1
        return self._weights({tomorrow.idxmax(): 1.0})


def test_perturbation_test_catches_a_cheating_strategy():
    prices = make_prices(300, 4, seed=5)
    make = lambda p: _Cheat(SYMS, p)  # noqa: E731
    found = [
        leaks(make, prices, cut, prices.index[50], "same_close") for cut in range(100, 260, 20)
    ]
    # The cheat's pick only changes when the perturbed day's best asset differs, which a
    # random perturbation misses about one time in four; most cuts must catch it.
    assert sum(found) >= 6
