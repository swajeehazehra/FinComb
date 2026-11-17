"""Shared input checks for the analytics functions."""

from __future__ import annotations

import pandas as pd

from fincomb.exceptions import InsufficientDataError

Frame = pd.Series | pd.DataFrame


def check_prices(prices: Frame, min_rows: int = 2) -> Frame:
    if not isinstance(prices, pd.Series | pd.DataFrame):
        raise TypeError(f"expected a pandas Series or DataFrame, got {type(prices).__name__}")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must be indexed by a DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices must be sorted by date (oldest first)")
    if prices.index.has_duplicates:
        raise ValueError("prices contain duplicate dates")
    if len(prices) < min_rows:
        raise InsufficientDataError(f"need at least {min_rows} prices, got {len(prices)}")
    values = prices.to_numpy(dtype="float64")
    if (values[~pd.isna(values)] <= 0).any():
        raise ValueError("prices must be strictly positive")
    return prices


def check_returns(returns: Frame, min_rows: int = 2) -> Frame:
    if not isinstance(returns, pd.Series | pd.DataFrame):
        raise TypeError(f"expected a pandas Series or DataFrame, got {type(returns).__name__}")
    # Count observations without dropping anything: gaps must stay visible so the
    # compounding functions can apply the missing-data policy.
    n = (
        int(returns.notna().any(axis=1).sum())
        if isinstance(returns, pd.DataFrame)
        else int(returns.notna().sum())
    )
    if n < min_rows:
        raise InsufficientDataError(f"need at least {min_rows} returns, got {n}")
    values = returns.to_numpy(dtype="float64")
    if (values[~pd.isna(values)] <= -1).any():
        raise ValueError("a simple return of -100% or less is impossible for a long position")
    return returns
