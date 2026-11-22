"""Engine exactness: drift, timing, the initialisation trade (C1), costs and turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.engine import benchmark_curve, min_start_index, run
from fincomb.backtest.schedule import rebalance_dates, rebalance_mask
from fincomb.exceptions import InsufficientDataError, MissingDataError
from fincomb.portfolio.analysis import portfolio_value
from fincomb.strategies.buy_and_hold import BuyAndHold
from fincomb.strategies.fixed_weights import FixedWeights
from fincomb.strategies.momentum import Momentum
from tests.unit.backtest_helpers import make_prices

CAP = 100_000.0
MODES = ["next_close", "same_close"]


# --- schedule --------------------------------------------------------------------------------


def test_monthly_dates_are_last_trading_day_of_each_month():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    got = rebalance_dates(idx, "monthly")
    assert list(got.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-02-29", "2024-03-29"]


def test_last_date_is_never_a_rebalance_date():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")  # ends on a month end
    assert not rebalance_mask(idx, "monthly")[-1]
    assert not rebalance_mask(idx, "daily")[-1]
    assert rebalance_mask(idx, "daily")[:-1].all()


@pytest.mark.parametrize(
    ("freq", "count"), [("weekly", 52), ("quarterly", 3), ("annual", 0), ("never", 0)]
)
def test_frequency_counts(freq, count):
    idx = pd.bdate_range("2024-01-01", "2024-12-31")  # 2024-12-31 is a Tuesday
    assert rebalance_mask(idx, freq).sum() == count


def test_unknown_frequency():
    with pytest.raises(ValueError, match="rebalance"):
        rebalance_mask(pd.bdate_range("2024-01-01", periods=5), "hourly")


def test_schedule_uses_data_calendar_not_business_calendar():
    idx = pd.bdate_range("2024-01-01", "2024-02-29").drop(pd.Timestamp("2024-01-31"))
    assert pd.Timestamp("2024-01-30") in rebalance_dates(idx, "monthly")


# --- C1: initialisation trade and the buy-and-hold identity ----------------------------------


@pytest.mark.parametrize("execution", MODES)
def test_zero_cost_buy_and_hold_matches_price_ratio_and_benchmark(execution):
    prices = make_prices(400, 3)
    start = prices.index[100]
    res = run(prices, BuyAndHold(["S0"]), start, initial_capital=CAP, execution=execution)
    p = prices["S0"].loc[start:]
    expected = CAP * p / p.iloc[0]
    np.testing.assert_allclose(res.equity.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=0)
    bench = benchmark_curve(prices["S0"], start, CAP)
    assert np.array_equal(res.equity.to_numpy(), bench.to_numpy())  # bit for bit
    assert res.equity.index[0] == start
    assert len(res.trades) == 1
    assert res.trades[0].kind == "init" and res.trades[0].date == start
    assert res.turnover.iloc[1:].sum() == 0


@pytest.mark.parametrize("execution", MODES)
def test_costed_buy_and_hold_pays_once_at_day_zero(execution):
    prices = make_prices(300, 2)
    start = prices.index[50]
    costs = FixedBpsCost(7, 3)
    res = run(
        prices, BuyAndHold(["S1"]), start, initial_capital=CAP, costs=costs, execution=execution
    )
    c = 10 / 10_000
    v0 = CAP * (1 - c)
    assert res.equity.iloc[0] == pytest.approx(v0, rel=1e-15)
    p = prices["S1"].loc[start:]
    np.testing.assert_allclose(res.equity, v0 * p / p.iloc[0], rtol=1e-12)
    assert res.costs.iloc[0] == pytest.approx(CAP * c)
    assert res.costs.iloc[1:].sum() == 0
    t = res.trades[0]
    assert t.commission == pytest.approx(CAP * 7e-4) and t.slippage_cost == pytest.approx(
        CAP * 3e-4
    )


def test_multi_asset_buy_and_hold_drifts():
    prices = make_prices(250, 3)
    w = {"S0": 0.5, "S1": 0.3, "S2": 0.2}
    res = run(prices, BuyAndHold(list(w), {"weights": w}), None, initial_capital=CAP)
    rel = prices / prices.iloc[0]
    expected = CAP * sum(wi * rel[s] for s, wi in w.items())
    np.testing.assert_allclose(res.equity, expected, rtol=1e-12)


def test_partial_weights_leave_cash_that_earns_nothing():
    prices = make_prices(100, 1)
    res = run(prices, BuyAndHold(["S0"], {"weights": {"S0": 0.6}}), None, initial_capital=CAP)
    np.testing.assert_allclose(res.cash, 0.4 * CAP)
    rel = prices["S0"] / prices["S0"].iloc[0]
    np.testing.assert_allclose(res.equity, 0.4 * CAP + 0.6 * CAP * rel, rtol=1e-12)


def test_init_signal_uses_last_warmup_close_under_next_close():
    prices = make_prices(400, 4)
    strat = Momentum(list(prices.columns), {"lookback": 60, "skip": 5, "top_n": 2})
    start = prices.index[200]
    res = run(prices, strat, start, execution="next_close")
    assert res.signal_dates[0] == prices.index[199]
    expected = strat.target_weights(prices.iloc[:200], None)
    np.testing.assert_array_equal(res.targets.iloc[0].to_numpy(), expected.to_numpy())
    same = run(prices, strat, start, execution="same_close")
    assert same.signal_dates[0] == start


def test_warmup_requirement():
    prices = make_prices(100, 2)
    strat = Momentum(["S0", "S1"], {"lookback": 60, "skip": 5})
    assert min_start_index(strat, "next_close") == 61
    assert min_start_index(strat, "same_close") == 60
    with pytest.raises(InsufficientDataError):
        run(prices, strat, prices.index[60], execution="next_close")
    run(prices, strat, prices.index[60], execution="same_close")
    assert min_start_index(BuyAndHold(["S0"]), "next_close") == 0


# --- daily rebalanced = Milestone 4 constant mix --------------------------------------------


def test_daily_fixed_weights_equals_constant_mix_portfolio_value():
    prices = make_prices(300, 3)
    w = {"S0": 0.2, "S1": 0.5, "S2": 0.3}
    res = run(
        prices,
        FixedWeights(list(w), {"weights": w}),
        None,
        rebalance="daily",
        execution="same_close",
        initial_capital=1.0,
    )
    expected = portfolio_value(prices, pd.Series(w))
    np.testing.assert_allclose(res.equity.to_numpy(), expected.to_numpy(), rtol=1e-12)


# --- a hand-worked example ------------------------------------------------------------------

IDX = pd.DatetimeIndex(
    [
        "2024-01-29",
        "2024-01-30",
        "2024-01-31",
        "2024-02-01",
        "2024-02-02",
        "2024-02-05",
        "2024-02-06",
        "2024-02-07",
    ]
)
HAND = pd.DataFrame(
    {"A": [100, 110, 110, 99, 99, 99, 99, 99], "B": [100, 100, 90, 90, 90, 90, 90, 90]},
    index=IDX,
    dtype="float64",
)


def _hand(execution):
    return run(
        HAND,
        FixedWeights(["A", "B"]),
        None,
        rebalance="monthly",
        initial_capital=1000,
        costs=FixedBpsCost(6, 4),
        execution=execution,
    )


def test_hand_worked_next_close():
    res = _hand("next_close")
    # Day 0: buy 50/50 with 1000; turnover 1, cost 1000 x 1 x 0.001 = 1.0, V = 999.
    # Day 1: A +10%: h = [549.45, 499.5], V = 1048.95.
    # Day 2 (Jan 31, month end): B -10%: h = [549.45, 449.55], V = 999.0; signal formed.
    # Day 3: A -10%: h = [494.505, 449.55], V = 944.055; execute: A is 22.4775 over 50%,
    #   turnover = 2 x 22.4775 / 944.055 = 1/21, cost = 44.955 x 0.001 = 0.044955.
    expected = [999.0, 1048.95, 999.0, 944.010045, 944.010045, 944.010045, 944.010045, 944.010045]
    np.testing.assert_allclose(res.equity, expected, rtol=1e-12)
    np.testing.assert_allclose(res.costs, [1.0, 0, 0, 0.044955, 0, 0, 0, 0], atol=1e-12)
    assert res.turnover.iloc[3] == pytest.approx(1 / 21, rel=1e-12)
    rebal = [t for t in res.trades if t.kind == "rebalance"]
    assert [(t.symbol, t.side) for t in rebal] == [("A", "sell"), ("B", "buy")]
    assert all(t.notional == pytest.approx(22.4775) for t in rebal)
    assert rebal[0].commission == pytest.approx(22.4775 * 6e-4)
    assert rebal[0].slippage_cost == pytest.approx(22.4775 * 4e-4)
    assert rebal[0].fill_price == pytest.approx(99 * (1 - 4e-4))
    assert rebal[1].fill_price == pytest.approx(90 * (1 + 4e-4))
    assert rebal[0].quantity == pytest.approx(22.4775 / (99 * (1 - 4e-4)))
    assert res.signal_dates[1] == IDX[2] and res.targets.index[1] == IDX[3]


def test_hand_worked_same_close():
    res = _hand("same_close")
    # Day 2: V = 999, A is 49.95 over 50%: cost 0.002 x 49.95 = 0.0999, V = 998.9001,
    # h = 499.45005 each; day 3: A -10% -> V = 449.505045 + 499.45005 = 948.955095.
    np.testing.assert_allclose(res.equity.iloc[2], 998.9001, rtol=1e-12)
    np.testing.assert_allclose(res.equity.iloc[3:], 948.955095, rtol=1e-12)
    assert res.targets.index[1] == IDX[2]


# --- the final date: no trade executes there -------------------------------------------------

# Data ends the trading day after a month end: 2023-03-31 (Fri) is the last monthly signal
# date and the penultimate row, 2023-04-03 (Mon) is the final row.
_END_IDX = pd.bdate_range("2023-01-02", "2023-04-03")


def _ends_after_signal():
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.02, (len(_END_IDX), 2))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=_END_IDX, columns=["A", "B"])


@pytest.mark.parametrize("execution", MODES)
@pytest.mark.parametrize("rebalance", ["monthly", "daily"])
def test_no_trade_executes_on_the_final_date(execution, rebalance):
    prices = _ends_after_signal()
    penultimate, final = _END_IDX[-2], _END_IDX[-1]
    assert rebalance_mask(_END_IDX, rebalance)[-2]  # a scheduled signal on the penultimate row
    res = run(
        prices,
        FixedWeights(["A", "B"]),
        None,
        rebalance=rebalance,
        costs=FixedBpsCost(10),
        execution=execution,
    )
    # Nothing is traded or charged on the final date in either mode.
    assert res.turnover.loc[final] == 0 and res.costs.loc[final] == 0
    assert all(t.date < final for t in res.trades)
    assert final not in res.targets.index
    # The final day is pure drift from the previous close's holdings.
    drift = res.weights.loc[penultimate].to_numpy() @ (prices.iloc[-1] / prices.iloc[-2]).to_numpy()
    assert res.equity.loc[final] == pytest.approx(res.equity.loc[penultimate] * drift, rel=1e-12)
    if execution == "next_close":
        # The penultimate-row signal is not taken: its fill would land on the final date.
        assert penultimate not in res.signal_dates
    else:
        # same_close trades at the penultimate close and holds over the final day's return.
        assert res.targets.index[-1] == penultimate and res.signal_dates[-1] == penultimate
        assert res.turnover.loc[penultimate] > 0


def test_next_close_still_fills_on_the_penultimate_date():
    """Only the fill on the final date is dropped: one more row and the Mar 31 signal fills."""
    idx = pd.bdate_range("2023-01-02", "2023-04-04")
    rng = np.random.default_rng(7)
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.02, (len(idx), 2)), axis=0),
        index=idx,
        columns=["A", "B"],
    )
    res = run(prices, FixedWeights(["A", "B"]), None, costs=FixedBpsCost(10))
    assert res.signal_dates[-1] == pd.Timestamp("2023-03-31")
    assert res.targets.index[-1] == pd.Timestamp("2023-04-03")  # the penultimate row
    assert res.turnover.loc["2023-04-03"] > 0 and res.turnover.iloc[-1] == 0


# --- costs and turnover ---------------------------------------------------------------------


def test_cost_product_identity():
    prices = make_prices(500, 4)
    strat = FixedWeights(list(prices.columns))
    gross = run(prices, strat, None, initial_capital=CAP)
    net = run(prices, strat, None, initial_capital=CAP, costs=FixedBpsCost(20, 10))
    factor = np.prod(1 - net.turnover.to_numpy() * 0.003)
    assert net.equity.iloc[-1] == pytest.approx(gross.equity.iloc[-1] * factor, rel=1e-12)
    np.testing.assert_allclose(net.turnover, gross.turnover, rtol=1e-12, atol=1e-15)


def test_cost_scales_linearly_for_one_trade():
    prices = make_prices(50, 2)
    one = run(prices, BuyAndHold(["S0", "S1"]), None, costs=FixedBpsCost(5, 0))
    two = run(prices, BuyAndHold(["S0", "S1"]), None, costs=FixedBpsCost(10, 0))
    assert two.costs.iloc[0] == pytest.approx(2 * one.costs.iloc[0], rel=1e-15)
    both = run(prices, BuyAndHold(["S0", "S1"]), None, costs=FixedBpsCost(5, 5))
    assert both.costs.iloc[0] == pytest.approx(two.costs.iloc[0], rel=1e-15)


class _Switch(FixedWeights):
    """All in A until the first rebalance, then all in B."""

    name = "switch"

    def target_weights(self, history, previous):
        return self._weights({"A": 1.0} if previous is None else {"B": 1.0})


def test_full_switch_turnover_is_two():
    res = run(HAND, _Switch(["A", "B"]), None, rebalance="monthly")
    assert res.turnover.iloc[0] == pytest.approx(1.0)
    assert res.turnover.iloc[3] == pytest.approx(2.0)


def test_no_change_no_trade():
    prices = pd.DataFrame({"A": np.full(60, 10.0)}, index=pd.bdate_range("2024-01-01", periods=60))
    res = run(prices, FixedWeights(["A"]), None, rebalance="weekly", costs=FixedBpsCost(10, 10))
    assert len(res.trades) == 1
    assert res.costs.iloc[1:].sum() == 0


def test_cost_bounds():
    with pytest.raises(ValueError):
        FixedBpsCost(-1, 0)
    with pytest.raises(ValueError):
        FixedBpsCost(0, 2000)


@settings(max_examples=60, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    bps=st.floats(0, 200),
    raw=st.lists(st.floats(0.01, 1.0), min_size=3, max_size=3),
    freq=st.sampled_from(["daily", "weekly", "monthly"]),
)
def test_engine_invariants(seed, bps, raw, freq):
    prices = make_prices(120, 3, seed=seed)
    w = {f"S{i}": x / sum(raw) * 0.95 for i, x in enumerate(raw)}
    strat = FixedWeights(list(w), {"weights": w})
    res = run(prices, strat, None, rebalance=freq, costs=FixedBpsCost(bps, bps / 2))
    assert (res.equity > 0).all()
    assert ((res.turnover >= 0) & (res.turnover <= 2 + 1e-12)).all()
    assert (res.costs >= 0).all()
    assert (res.cash >= -1e-9).all()
    assert ((res.weights >= -1e-12) & (res.weights <= 1 + 1e-12)).all().all()
    cheaper = run(prices, strat, None, rebalance=freq, costs=FixedBpsCost(bps / 2, bps / 4))
    assert cheaper.equity.iloc[-1] >= res.equity.iloc[-1] - 1e-9


# --- input checks ---------------------------------------------------------------------------


def test_rejects_gaps_and_bad_inputs():
    prices = make_prices(50, 2)
    gappy = prices.copy()
    gappy.iloc[10, 0] = np.nan
    with pytest.raises(MissingDataError):
        run(gappy, BuyAndHold(["S0", "S1"]), None)
    with pytest.raises(ValueError, match="execution"):
        run(prices, BuyAndHold(["S0"]), None, execution="next_open")
    with pytest.raises(ValueError, match="capital"):
        run(prices, BuyAndHold(["S0"]), None, initial_capital=0)
    with pytest.raises(InsufficientDataError):
        run(prices, BuyAndHold(["S0"]), prices.index[-1])


class _Bad(FixedWeights):
    name = "bad"

    def target_weights(self, history, previous):
        return self._weights({"S0": 0.8, "S1": 0.4})


def test_rejects_weights_above_one():
    with pytest.raises(ValueError, match="summing above 1"):
        run(make_prices(30, 2), _Bad(["S0", "S1"]), None)


def test_raw_prices_feed_only_the_trade_log():
    prices = make_prices(60, 1)
    raw = prices * 3
    a = run(prices, FixedWeights(["S0"]), None, rebalance="weekly")
    b = run(prices, FixedWeights(["S0"]), None, rebalance="weekly", raw_prices=raw)
    np.testing.assert_array_equal(a.equity, b.equity)
    assert b.trades[0].reference_price == pytest.approx(3 * a.trades[0].reference_price)
