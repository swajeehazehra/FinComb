"""Risk and risk-adjusted performance measures.

Risk-free rates are given as ANNUAL decimals (0.04 = 4 %) and converted to a
per-period rate by compounding: (1 + rf) ** (1 / periods_per_year) - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fincomb.analytics._utils import Frame, check_returns
from fincomb.analytics.constants import TRADING_DAYS
from fincomb.analytics.returns import cagr, wealth_index


def periodic_rate(annual_rate: float, periods_per_year: int = TRADING_DAYS) -> float:
    if annual_rate <= -1:
        raise ValueError("annual rate must be greater than -100%")
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


_ZERO = 1e-12  # below this a standard deviation is treated as zero (float noise)


def _safe_div(num, den):
    """num / den, giving NaN instead of a huge number or inf when den is ~0."""
    if isinstance(den, pd.Series):
        return num / den.where(den.abs() > _ZERO)
    return float("nan") if np.isnan(den) or abs(den) <= _ZERO else num / den


def annualized_volatility(returns: Frame, periods_per_year: int = TRADING_DAYS):
    """Sample standard deviation (ddof=1) x sqrt(periods_per_year)."""
    returns = check_returns(returns)
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: Frame, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """Annualised Sharpe: mean(excess) / std(excess) x sqrt(periods_per_year)."""
    returns = check_returns(returns)
    excess = returns - periodic_rate(risk_free, periods_per_year)
    return _safe_div(excess.mean(), excess.std(ddof=1)) * np.sqrt(periods_per_year)


def downside_deviation(returns: Frame, target: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """sqrt(mean(min(r - target, 0)^2)) x sqrt(periods_per_year), target per period."""
    returns = check_returns(returns)
    shortfall = (returns - target).clip(upper=0.0)
    return np.sqrt((shortfall**2).mean()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: Frame, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """Annualised mean excess return / annualised downside deviation (target = risk-free)."""
    returns = check_returns(returns)
    rf = periodic_rate(risk_free, periods_per_year)
    excess_mean = (returns - rf).mean() * periods_per_year
    return _safe_div(excess_mean, downside_deviation(returns, rf, periods_per_year))


def drawdown_series(returns: Frame) -> Frame:
    """Percentage below the running peak at each date (0 at a new high, negative otherwise).

    The starting capital counts as the first peak, so a loss on day one is a drawdown.
    """
    wealth = wealth_index(returns)
    peak = wealth.cummax().clip(lower=1.0)
    return wealth / peak - 1.0


def max_drawdown(returns: Frame):
    """Worst peak-to-trough loss, as a negative number in [-1, 0]."""
    return drawdown_series(returns).min()


def calmar_ratio(returns: Frame, periods_per_year: int = TRADING_DAYS):
    """CAGR / |max drawdown|."""
    return _safe_div(cagr(returns, periods_per_year), abs(max_drawdown(returns)))


@dataclass(frozen=True)
class DrawdownEpisode:
    peak: pd.Timestamp | None  # last date at the old high (None = the starting capital)
    trough: pd.Timestamp
    recovery: pd.Timestamp | None  # first date back at the old high (None = not yet)
    depth: float  # negative
    days_to_trough: int  # trading days
    days_to_recover: int | None


def drawdown_episodes(returns: pd.Series, top: int = 5) -> list[DrawdownEpisode]:
    """The `top` deepest separate drawdowns of one asset, deepest first."""
    if not isinstance(returns, pd.Series):
        raise TypeError("drawdown_episodes takes one asset (a Series)")
    dd = drawdown_series(returns)
    underwater = dd < 0
    episodes: list[DrawdownEpisode] = []
    # Each run of consecutive underwater days is one episode.
    run_id = (underwater != underwater.shift(fill_value=False)).cumsum()
    positions = {ts: i for i, ts in enumerate(dd.index)}
    for _, block in dd[underwater].groupby(run_id[underwater]):
        start_pos = positions[block.index[0]]
        end_pos = positions[block.index[-1]]
        trough = block.idxmin()
        peak = dd.index[start_pos - 1] if start_pos > 0 else None
        recovery = dd.index[end_pos + 1] if end_pos + 1 < len(dd) else None
        peak_pos = start_pos - 1
        episodes.append(
            DrawdownEpisode(
                peak=peak,
                trough=trough,
                recovery=recovery,
                depth=float(block.min()),
                days_to_trough=positions[trough] - peak_pos,
                days_to_recover=(end_pos + 1 - positions[trough]) if recovery is not None else None,
            )
        )
    episodes.sort(key=lambda e: e.depth)
    return episodes[:top]


def rolling_volatility(
    returns: Frame, window: int = 63, periods_per_year: int = TRADING_DAYS
) -> Frame:
    """Annualised volatility over a trailing window (only past data, no centering)."""
    if window < 2:
        raise ValueError("window must be at least 2")
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: Frame, window: int = 252, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> Frame:
    """Sharpe ratio over a trailing window."""
    if window < 2:
        raise ValueError("window must be at least 2")
    excess = returns - periodic_rate(risk_free, periods_per_year)
    roll = excess.rolling(window, min_periods=window)
    std = roll.std(ddof=1)
    return roll.mean() / std.where(std.abs() > _ZERO) * np.sqrt(periods_per_year)
