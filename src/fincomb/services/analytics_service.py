"""Load stored prices and run the pure analytics on them. Shared by the CLI and the API.

Each function returns pandas objects plus the alignment notes from `load_prices`,
so every caller can show why a common period is shorter than the full data.
No maths here: it all lives in `fincomb.analytics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from fincomb.analytics.correlation import Method, correlation_matrix
from fincomb.analytics.returns import cumulative_returns, handle_missing, resample_returns
from fincomb.analytics.returns import simple_returns as _simple_returns
from fincomb.analytics.risk import (
    DrawdownEpisode,
    drawdown_episodes,
    drawdown_series,
    rolling_sharpe,
    rolling_volatility,
)
from fincomb.analytics.summary import performance_summary
from fincomb.services.market_data import load_prices

Align = Literal["inner", "outer"]
Frequency = Literal["daily", "weekly", "monthly", "yearly"]
RollingMetric = Literal["volatility", "sharpe"]

DEFAULT_ROLLING_WINDOW = {"volatility": 63, "sharpe": 252}


@dataclass
class AnalysisResult:
    """A result table (rows = dates or symbols) and the alignment notes behind it."""

    table: pd.DataFrame
    notes: list[str] = field(default_factory=list)


@dataclass
class DrawdownResult:
    symbol: str
    series: pd.Series  # drawdown at each date, 0 at a new high
    episodes: list[DrawdownEpisode]


def _load(session, symbols, start, end, align: Align = "inner") -> pd.DataFrame:
    if align not in ("inner", "outer"):
        raise ValueError("align must be 'inner' or 'outer'")
    return load_prices(session, symbols, start, end, align=align)


def _notes(prices: pd.DataFrame) -> list[str]:
    return list(prices.attrs.get("alignment_notes", []))


def performance(
    session: Session,
    symbols: list[str],
    start: date | None = None,
    end: date | None = None,
    risk_free: float = 0.0,
    align: Align = "inner",
) -> AnalysisResult:
    """One row per symbol: total return, CAGR, volatility, Sharpe, Sortino, drawdown..."""
    prices = _load(session, symbols, start, end, align)
    return AnalysisResult(performance_summary(prices, risk_free=risk_free), _notes(prices))


def returns(
    session: Session,
    symbols: list[str],
    start: date | None = None,
    end: date | None = None,
    frequency: Frequency = "daily",
    cumulative: bool = False,
    align: Align = "inner",
) -> AnalysisResult:
    """Return series per symbol (rows = dates).

    Compounding (weekly/monthly/yearly or cumulative) follows the missing-data
    policy: a gap inside a series raises MissingDataError instead of being skipped.
    """
    prices = _load(session, symbols, start, end, align)
    r = _simple_returns(prices)
    if frequency != "daily" or cumulative:
        r = handle_missing(r, "raise")
    r = resample_returns(r, frequency)
    if cumulative:
        r = cumulative_returns(r)
    return AnalysisResult(r, _notes(prices))


def rolling(
    session: Session,
    symbols: list[str],
    start: date | None = None,
    end: date | None = None,
    metric: RollingMetric = "volatility",
    window: int | None = None,
    risk_free: float = 0.0,
    align: Align = "inner",
) -> AnalysisResult:
    """Trailing-window volatility or Sharpe ratio. Dates before the first full window are
    dropped."""
    if metric not in DEFAULT_ROLLING_WINDOW:
        raise ValueError(f"metric must be one of {list(DEFAULT_ROLLING_WINDOW)}")
    window = window or DEFAULT_ROLLING_WINDOW[metric]
    prices = _load(session, symbols, start, end, align)
    r = _simple_returns(prices)
    if metric == "volatility":
        out = rolling_volatility(r, window)
    else:
        out = rolling_sharpe(r, window, risk_free)
    return AnalysisResult(out.dropna(how="all"), _notes(prices))


def correlation(
    session: Session,
    symbols: list[str],
    start: date | None = None,
    end: date | None = None,
    method: Method = "pearson",
    align: Align = "inner",
) -> AnalysisResult:
    """Correlation matrix of daily returns. With align="outer" each pair uses the days
    both symbols traded."""
    prices = _load(session, symbols, start, end, align)
    return AnalysisResult(
        correlation_matrix(_simple_returns(prices), method=method), _notes(prices)
    )


def drawdowns(
    session: Session,
    symbol: str,
    start: date | None = None,
    end: date | None = None,
    top: int = 5,
) -> DrawdownResult:
    """Drawdown series and the `top` deepest episodes of one symbol."""
    prices = _load(session, [symbol], start, end)
    r = _simple_returns(prices.iloc[:, 0])
    return DrawdownResult(
        symbol=str(prices.columns[0]),
        series=drawdown_series(r),
        episodes=drawdown_episodes(r, top=top),
    )
