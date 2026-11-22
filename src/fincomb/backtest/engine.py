"""The daily backtest engine: the one place that decides timing, drift and costs.

Weight-based and long-only. Each day:

1. Holdings drift with the day's returns (from adjusted closes, so dividends count):
   h_i <- h_i (1 + r_i), V = sum(h) + cash. Cash earns 0.
2. A trade scheduled for this close is executed (see below).
3. If this close is a rebalance date, the strategy is called with prices up to and
   including this close only, and its targets are executed now (`same_close`) or
   at the next close (`next_close`, the default).

**The final date.** No trade ever executes on the last date of the data: a trade
there would pay costs with no later price move to earn anything. The schedule never
marks the final date, and under `next_close` the engine also skips the signal on the
penultimate date, whose fill would land on the final date. Under `same_close` a
penultimate-date trade still executes, since it is held over the final day's return.

**Initialisation.** The first position is a special trade at the close of the first
evaluation date (day 0) in both modes, charged normal costs. Strategies whose
targets don't depend on prices are called at day 0. Price-dependent strategies use
the signal from the close before day 0 under `next_close` (so even the first trade
lags its signal by one bar) and the day 0 signal under `same_close`. With zero
costs buy-and-hold therefore has equity V(t) = V(0) P(t) / P(0) in both modes.

**Executing targets w.** With drifted weights w~ = h / V:
turnover tau = sum_i |w_i - w~_i|, cost C = V tau c, V+ = V - C,
new holdings h = w V+, cash = (1 - sum w) V+.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from fincomb.analytics._utils import check_prices
from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.schedule import rebalance_mask
from fincomb.exceptions import InsufficientDataError, MissingDataError
from fincomb.strategies.base import Strategy

Execution = Literal["next_close", "same_close"]
EXECUTIONS: tuple[str, ...] = ("next_close", "same_close")
WEIGHT_TOL = 1e-9
TRADE_TOL = 1e-12  # weight changes below this are not traded


@dataclass(frozen=True, slots=True)
class Trade:
    date: pd.Timestamp
    symbol: str
    kind: str  # "init" (day 0) or "rebalance"
    side: str  # "buy" or "sell"
    weight_before: float
    weight_after: float
    notional: float  # money traded, positive
    reference_price: float  # raw close that day
    fill_price: float  # reference x (1 +/- slippage)
    quantity: float  # indicative (fractional) shares: notional / fill price
    commission: float
    slippage_cost: float


@dataclass
class EngineResult:
    equity: pd.Series  # portfolio value at each close of the evaluation period
    cash: pd.Series
    gross_exposure: pd.Series  # invested value / equity
    turnover: pd.Series  # 0 on days without a trade
    costs: pd.Series  # money paid on each day
    trades: list[Trade]
    targets: pd.DataFrame  # targets by execution date
    signal_dates: list[pd.Timestamp]  # signal date of each executed target, same order
    initial_capital: float
    cost_rate: float
    execution: str
    rebalance: str
    weights: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)  # end-of-day


def first_signal_index(strategy: Strategy, day0: int, execution: str) -> int:
    """Index of the close whose signal the day-0 initialisation trade uses."""
    if not strategy.uses_prices or execution == "same_close":
        return day0
    return day0 - 1


def min_start_index(strategy: Strategy, execution: str) -> int:
    """Smallest day-0 position that leaves the strategy enough history for its first signal."""
    if not strategy.uses_prices:
        return 0
    need = strategy.warmup_bars  # rows up to and including the signal date
    return need if execution == "next_close" else need - 1


def _check_targets(strategy: Strategy, target: pd.Series, when: pd.Timestamp) -> np.ndarray:
    if not isinstance(target, pd.Series):
        raise TypeError(f"{strategy.name} must return a pandas Series of weights or None")
    if target.index.equals(strategy.symbol_index):
        w = target.to_numpy(dtype="float64")
    else:
        extra = sorted(set(map(str, target.index)) - set(strategy.symbols))
        if extra:
            raise ValueError(f"{strategy.name} returned weights for unknown symbols {extra}")
        w = target.reindex(strategy.symbols, fill_value=0.0).to_numpy(dtype="float64")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{strategy.name} returned non-finite weights on {when.date()}")
    if (w < -WEIGHT_TOL).any() or (w > 1 + WEIGHT_TOL).any() or w.sum() > 1 + WEIGHT_TOL:
        raise ValueError(
            f"{strategy.name} returned weights outside [0, 1] or summing above 1 on {when.date()}"
        )
    return np.clip(w, 0.0, 1.0)


def run(
    prices: pd.DataFrame,
    strategy: Strategy,
    start: pd.Timestamp | str | None = None,
    *,
    raw_prices: pd.DataFrame | None = None,
    rebalance: str | None = None,
    initial_capital: float = 100_000.0,
    costs: FixedBpsCost | None = None,
    execution: str = "next_close",
) -> EngineResult:
    """Simulate `strategy` on `prices` (adjusted closes, inner-aligned, warm-up included).

    `start` is the first evaluation date (day 0); rows before it are warm-up history the
    strategy may read but never trades on. `raw_prices` (raw closes) only feed the trade
    log's reference prices; they default to `prices`.
    """
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {', '.join(EXECUTIONS)}")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be a positive number")
    costs = costs or FixedBpsCost()
    rebalance = rebalance or strategy.default_rebalance
    check_prices(prices)
    missing = [s for s in strategy.symbols if s not in prices.columns]
    if missing:
        raise ValueError(f"no prices for {', '.join(missing)}")
    adj = prices[strategy.symbols]
    if adj.isna().to_numpy().any():
        raise MissingDataError("prices have gaps; align the assets first (inner alignment)")
    raw = adj if raw_prices is None else raw_prices.reindex(index=adj.index)[strategy.symbols]
    if raw.isna().to_numpy().any():
        raise MissingDataError("raw prices are missing on some dates")

    index = adj.index
    day0 = 0 if start is None else int(index.searchsorted(pd.Timestamp(start)))
    if day0 >= len(index) - 1:
        raise InsufficientDataError("the evaluation period needs at least 2 price dates")
    if day0 < min_start_index(strategy, execution):
        raise InsufficientDataError(
            f"{strategy.name} needs {strategy.warmup_bars} price rows before its first signal; "
            "load more history before the start date"
        )

    P = adj.to_numpy(dtype="float64")
    RAW = raw.to_numpy(dtype="float64")
    growth = P[1:] / P[:-1]  # growth[k - 1] = P[k] / P[k - 1]
    mask = rebalance_mask(index, rebalance)
    c = costs.rate
    symbols = strategy.symbols
    symbol_index = pd.Index(symbols, name="symbol")
    n_days = len(index) - day0

    equity = np.empty(n_days)
    cash_s = np.empty(n_days)
    gross = np.empty(n_days)
    turnover = np.zeros(n_days)
    cost_s = np.zeros(n_days)
    weights_eod = np.empty((n_days, len(symbols)))
    trades: list[Trade] = []
    target_rows: list[np.ndarray] = []
    target_dates: list[pd.Timestamp] = []
    signal_dates: list[pd.Timestamp] = []

    h = np.zeros(len(symbols))
    cash = float(initial_capital)
    previous: pd.Series | None = None

    def signal(k: int) -> tuple[np.ndarray | None, pd.Timestamp]:
        nonlocal previous
        when = index[k]
        target = strategy.target_weights(adj.iloc[: k + 1], previous)
        if target is None:
            return None, when
        w = _check_targets(strategy, target, when)
        previous = pd.Series(w, index=symbol_index, name="weight")
        return w, when

    def execute(k: int, w: np.ndarray, kind: str, signal_date: pd.Timestamp) -> None:
        nonlocal h, cash
        value = float(h.sum() + cash)
        drifted = h / value
        dw = w - drifted
        tau = float(np.abs(dw).sum())
        cost = value * tau * c
        after = value - cost
        i = k - day0
        turnover[i] += tau
        cost_s[i] += cost
        today = index[k]
        target_rows.append(w.copy())
        target_dates.append(today)
        signal_dates.append(signal_date)
        traded_idx = np.flatnonzero(np.abs(dw) > TRADE_TOL)
        if len(traded_idx):
            slip = costs.slippage_bps / 10_000.0
            d = dw[traded_idx]
            traded = np.abs(d) * value
            buy = d > 0
            ref = RAW[k, traded_idx]
            fill = ref * np.where(buy, 1 + slip, 1 - slip)
            commission = traded * (costs.cost_bps / 10_000.0)
            slippage = traded * slip
            for j, b, tv, rp, fp, cm, sl in zip(
                traded_idx.tolist(),
                buy.tolist(),
                traded.tolist(),
                ref.tolist(),
                fill.tolist(),
                commission.tolist(),
                slippage.tolist(),
                strict=True,
            ):
                trades.append(
                    Trade(
                        date=today,
                        symbol=symbols[j],
                        kind=kind,
                        side="buy" if b else "sell",
                        weight_before=float(drifted[j]),
                        weight_after=float(w[j]),
                        notional=tv,
                        reference_price=rp,
                        fill_price=fp,
                        quantity=tv / fp,
                        commission=cm,
                        slippage_cost=sl,
                    )
                )
        h = w * after
        cash = after * (1.0 - float(w.sum()))

    pending: tuple[np.ndarray, pd.Timestamp] | None = None
    last = len(index) - 1
    # Under next_close a signal at `last - 1` would fill on the final date, where the
    # trade costs money but can't earn a return before the data ends: don't take it.
    last_signal = last - 1 if execution == "next_close" else last
    for k in range(day0, len(index)):
        i = k - day0
        if k == day0:
            w0, when = signal(first_signal_index(strategy, day0, execution))
            if w0 is None:
                raise ValueError(f"{strategy.name} returned no initial weights")
            execute(k, w0, "init", when)
        else:
            h = h * growth[k - 1]
            if pending is not None:
                execute(k, pending[0], "rebalance", pending[1])
                pending = None
        # A day-0 signal under same_close is the initialisation trade itself.
        if mask[k] and k < last_signal and not (k == day0 and execution == "same_close"):
            w, when = signal(k)
            if w is not None:
                if execution == "same_close":
                    execute(k, w, "rebalance", when)
                else:
                    pending = (w, when)
        value = float(h.sum() + cash)
        equity[i] = value
        cash_s[i] = cash
        gross[i] = float(h.sum()) / value
        weights_eod[i] = h / value

    eval_index = index[day0:]
    return EngineResult(
        equity=pd.Series(equity, index=eval_index, name="equity"),
        cash=pd.Series(cash_s, index=eval_index, name="cash"),
        gross_exposure=pd.Series(gross, index=eval_index, name="gross_exposure"),
        turnover=pd.Series(turnover, index=eval_index, name="turnover"),
        costs=pd.Series(cost_s, index=eval_index, name="costs"),
        trades=trades,
        targets=pd.DataFrame(
            np.array(target_rows).reshape(len(target_rows), len(symbols)),
            index=pd.DatetimeIndex(target_dates, name="date"),
            columns=symbols,
        ),
        signal_dates=signal_dates,
        initial_capital=float(initial_capital),
        cost_rate=c,
        execution=execution,
        rebalance=rebalance,
        weights=pd.DataFrame(weights_eod, index=eval_index, columns=symbols),
    )


def benchmark_curve(
    prices: pd.Series, start: pd.Timestamp | str | None, initial_capital: float
) -> pd.Series:
    """Capital invested in one asset at the day-0 close, no costs, computed exactly the way
    the engine drifts holdings (so a zero-cost buy-and-hold of it matches to the bit)."""
    check_prices(prices)
    p = prices.to_numpy(dtype="float64")
    day0 = 0 if start is None else int(prices.index.searchsorted(pd.Timestamp(start)))
    out = np.empty(len(p) - day0)
    value = np.array([float(initial_capital)])
    out[0] = value[0]
    for i, k in enumerate(range(day0 + 1, len(p)), start=1):
        value = value * (p[k] / p[k - 1])
        out[i] = value[0]
    return pd.Series(out, index=prices.index[day0:], name="benchmark_equity")
