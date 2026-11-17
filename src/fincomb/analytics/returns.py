"""Return calculations.

All functions work on a Series (one asset) or a DataFrame (one column per asset).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from fincomb.analytics._utils import Frame, check_prices, check_returns
from fincomb.analytics.constants import DAYS_PER_YEAR, TRADING_DAYS
from fincomb.exceptions import MissingDataError

Missing = Literal["raise", "zero"]

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


def _gaps(returns: pd.Series) -> pd.Series:
    """NaNs strictly inside the series (between its first and last value).

    Leading/trailing NaNs are not gaps: the asset simply hadn't started trading yet
    (or the table runs past its last price), which happens with outer alignment.
    """
    first, last = returns.first_valid_index(), returns.last_valid_index()
    if first is None:
        return returns.isna() & False
    inside = (returns.index >= first) & (returns.index <= last)
    return returns.isna() & inside


def handle_missing(returns: Frame, missing: Missing = "raise") -> Frame:
    """Apply the missing-data policy before compounding.

    missing="raise" (default): a gap inside the series raises MissingDataError naming
    the asset and dates. missing="zero": gaps are treated as a 0% (flat) day.
    Leading and trailing NaNs are left as they are in both modes.
    """
    if missing not in ("raise", "zero"):
        raise ValueError("missing must be 'raise' or 'zero'")
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    out = frame.copy()
    for col in frame.columns:
        gaps = _gaps(frame[col])
        if not gaps.any():
            continue
        if missing == "raise":
            dates = [d.date().isoformat() for d in frame.index[gaps.to_numpy()][:3]]
            raise MissingDataError(
                f"{col}: {int(gaps.sum())} missing return(s) inside the series "
                f"(first: {', '.join(dates)}). Align the data first, or pass missing='zero' "
                "to treat them as flat days."
            )
        out.loc[gaps, col] = 0.0
    return out.iloc[:, 0].rename(returns.name) if isinstance(returns, pd.Series) else out


def wealth_index(returns: Frame, start_value: float = 1.0, missing: Missing = "raise") -> Frame:
    """Value over time of `start_value` invested before the first return.

    Dates before an asset's first return stay NaN.
    """
    returns = handle_missing(check_returns(returns, min_rows=1), missing)
    return (1.0 + returns).cumprod() * start_value


def cumulative_returns(returns: Frame, missing: Missing = "raise") -> Frame:
    """Running total return: prod(1 + r) - 1 up to each date."""
    return wealth_index(returns, 1.0, missing) - 1.0


def total_return(returns: Frame, missing: Missing = "raise") -> float | pd.Series:
    """Total compounded return over the whole period."""
    returns = handle_missing(check_returns(returns, min_rows=1), missing)
    return (1.0 + returns).prod() - 1.0


def _default_start(index: pd.DatetimeIndex) -> pd.Timestamp:
    """When only returns are known, assume the starting price was one business day earlier."""
    return index[0] - pd.offsets.BDay(1)


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Elapsed calendar time in years (365.25 days per year)."""
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    if days <= 0:
        raise ValueError(f"end {end} must be after start {start}")
    return days / DAYS_PER_YEAR


def cagr(
    returns: Frame, start: pd.Timestamp | str | None = None, missing: Missing = "raise"
) -> float | pd.Series:
    """Compound annual growth rate over elapsed CALENDAR time.

        CAGR = (1 + total return) ** (1 / years) - 1,  years = days / 365.25

    `start` is the date of the starting price (the day the money was invested).
    Pass it whenever you know it; performance_summary does. If omitted it is taken
    as one business day before the first return. The end is the date of the last
    return of each asset.
    """
    returns = check_returns(returns)
    total = total_return(returns, missing)
    if isinstance(returns, pd.Series):
        begin = pd.Timestamp(start) if start is not None else _default_start(returns.index)
        return (1.0 + total) ** (1.0 / years_between(begin, returns.last_valid_index())) - 1.0
    out = {}
    for col in returns.columns:
        col_r = returns[col].dropna()
        begin = pd.Timestamp(start) if start is not None else _default_start(col_r.index)
        out[col] = (1.0 + total[col]) ** (1.0 / years_between(begin, col_r.index[-1])) - 1.0
    return pd.Series(out)


def annualized_mean_return(returns: Frame, periods_per_year: int = TRADING_DAYS):
    """Arithmetic mean return x periods_per_year (used in Sharpe-style ratios)."""
    returns = check_returns(returns)
    return returns.mean() * periods_per_year


def resample_returns(returns: Frame, frequency: str = "monthly") -> Frame:
    """Compound daily returns into weekly, monthly or yearly returns.

    Each period's return is prod(1 + r_daily) - 1 over the days in that period.
    Each period is labelled with its LAST ACTUAL TRADING DAY in the data (e.g.
    2024-08-30, not the calendar month-end 2024-08-31, which was a Saturday), so
    the result lines up with daily data. Periods with no data are dropped.
    """
    if frequency not in _FREQ:
        raise ValueError(f"frequency must be one of {list(_FREQ)}")
    returns = check_returns(returns, min_rows=1)
    rule = _FREQ[frequency]
    if rule is None:
        return returns
    out = (1.0 + returns).resample(rule).prod(min_count=1) - 1.0
    last_day = pd.Series(returns.index, index=returns.index).resample(rule).max()
    out.index = pd.DatetimeIndex(last_day.reindex(out.index).to_numpy(), name=returns.index.name)
    return out.dropna(how="all")
