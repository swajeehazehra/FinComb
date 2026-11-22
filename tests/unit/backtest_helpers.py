"""Synthetic price frames and a look-ahead checker shared by the backtest tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.engine import run


def make_prices(days: int = 600, assets: int = 4, seed: int = 0, start: str = "2015-01-01"):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, (days, assets)) + rng.normal(0, 0.006, (days, 1))
    idx = pd.bdate_range(start, periods=days)
    cols = [f"S{i}" for i in range(assets)]
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=cols)


def perturb_after(prices: pd.DataFrame, cut: int, seed: int = 99) -> pd.DataFrame:
    """Replace every price after row `cut` with unrelated random (positive) values."""
    rng = np.random.default_rng(seed)
    out = prices.copy()
    tail = len(prices) - cut - 1
    out.iloc[cut + 1 :] = 50 + 100 * rng.random((tail, prices.shape[1]))
    return out


def leaks(make_strategy, prices, cut, start, execution="next_close", rebalance=None) -> bool:
    """True if anything up to row `cut` changes when prices after it change."""
    a = run(
        prices,
        make_strategy(prices),
        start,
        execution=execution,
        rebalance=rebalance,
        costs=FixedBpsCost(5, 5),
    )
    p2 = perturb_after(prices, cut)
    b = run(
        p2,
        make_strategy(p2),
        start,
        execution=execution,
        rebalance=rebalance,
        costs=FixedBpsCost(5, 5),
    )
    t = prices.index[cut]
    if not np.array_equal(a.equity.loc[:t].to_numpy(), b.equity.loc[:t].to_numpy()):
        return True
    sig_a = [i for i, d in enumerate(a.signal_dates) if d <= t]
    sig_b = [i for i, d in enumerate(b.signal_dates) if d <= t]
    if len(sig_a) != len(sig_b):
        return True
    ta, tb = a.targets.to_numpy(), b.targets.to_numpy()
    if not all(np.array_equal(ta[i], tb[j]) for i, j in zip(sig_a, sig_b, strict=True)):
        return True
    trades_a = [x for x in a.trades if x.date <= t]
    trades_b = [x for x in b.trades if x.date <= t]
    return trades_a != trades_b
