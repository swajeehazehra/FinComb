"""Return calculations.

All functions work on a Series (one asset) or a DataFrame (one column per asset).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fincomb.analytics._utils import Frame, check_prices, check_returns
from fincomb.analytics.constants import TRADING_DAYS

_FREQ = {"daily": None, "weekly": "W-FRI", "monthly": "ME", "yearly": "YE"}


def simple_returns(prices: Frame) -> Frame:
    """r_t = P_t / P_{t-1} - 1. The first row (no previous price) is dropped.

    Missing prices are NOT filled: a gap produces NaN rather than an invented return.
    """
    check_prices(prices)
    return prices.pct_change(fill_method=None).iloc[1:]


def log_returns(prices: Frame) -> Frame:
    """ln(P_t / P_{t-1}). Additive over time, useful for statistics."""
    check_prices(prices)
    return np.log(prices / prices.shift(1)).iloc[1:]


def wealth_index(returns: Frame, start_value: float = 1.0) -> Frame:
    """Value over time of `start_value` invested before the first return."""
    returns = check_returns(returns, min_rows=1)
    return (1.0 + returns.fillna(0.0)).cumprod() * start_value


def cumulative_returns(returns: Frame) -> Frame:
    """Running total return: prod(1 + r) - 1 up to each date."""
    returns = check_returns(returns, min_rows=1)
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def total_return(returns: Frame) -> float | pd.Series:
    """Total compounded return over the whole period."""
    returns = check_returns(returns, min_rows=1)
    return (1.0 + returns.fillna(0.0)).prod() - 1.0


def cagr(returns: Frame, periods_per_year: int = TRADING_DAYS) -> float | pd.Series:
    """Compound annual growth rate: (1 + total)^(periods_per_year / n) - 1.

    n is the number of return observations, so 252 daily returns = 1 year.
    """
    returns = check_returns(returns)
    n = returns.count()
    return (1.0 + total_return(returns)) ** (periods_per_year / n) - 1.0


def annualized_mean_return(returns: Frame, periods_per_year: int = TRADING_DAYS):
    """Arithmetic mean return x periods_per_year (used in Sharpe-style ratios)."""
    returns = check_returns(returns)
    return returns.mean() * periods_per_year


def resample_returns(returns: Frame, frequency: str = "monthly") -> Frame:
    """Compound daily returns into weekly, monthly or yearly returns.

    Each period's return is prod(1 + r_daily) - 1 over the days in that period.
    Periods with no data are dropped.
    """
    if frequency not in _FREQ:
        raise ValueError(f"frequency must be one of {list(_FREQ)}")
    returns = check_returns(returns, min_rows=1)
    rule = _FREQ[frequency]
    if rule is None:
        return returns
    grouped = (1.0 + returns).resample(rule)
    out = grouped.prod(min_count=1) - 1.0
    return out.dropna(how="all")
