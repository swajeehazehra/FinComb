"""Analytics endpoints. All numbers come from `fincomb.services.analytics_service`."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Query

from fincomb.api.deps import DateRangeDep, SessionDep, SymbolsDep
from fincomb.api.schemas.analytics import (
    CorrelationResponse,
    DrawdownResponse,
    EpisodeOut,
    PerformanceResponse,
    ReturnsResponse,
    RollingResponse,
)
from fincomb.api.schemas.common import ERROR_RESPONSES, TimeSeries, finite
from fincomb.services import analytics_service as svc

router = APIRouter(prefix="/analytics", tags=["analytics"], responses=ERROR_RESPONSES)

AlignQuery = Annotated[
    Literal["inner", "outer"],
    Query(
        description="inner = only dates every symbol has (default); "
        "outer = each symbol over its own history"
    ),
]
RiskFreeQuery = Annotated[float, Query(gt=-1, lt=1, description="Annual risk-free rate, 0.04 = 4%")]


@router.get("/performance", response_model=PerformanceResponse)
def performance(
    session: SessionDep,
    symbols: SymbolsDep,
    dates: DateRangeDep,
    risk_free: RiskFreeQuery = 0.0,
    align: AlignQuery = "inner",
):
    """Total return, CAGR, volatility, Sharpe, Sortino, max drawdown and Calmar per symbol."""
    result = svc.performance(session, symbols, dates.start, dates.end, risk_free, align)
    return PerformanceResponse.build(result.table, align, risk_free, result.notes)


@router.get("/returns", response_model=ReturnsResponse)
def returns(
    session: SessionDep,
    symbols: SymbolsDep,
    dates: DateRangeDep,
    frequency: Annotated[Literal["daily", "weekly", "monthly", "yearly"], Query()] = "daily",
    cumulative: bool = False,
    align: AlignQuery = "inner",
):
    """Simple returns per symbol, compounded to the chosen frequency. Weekly and monthly
    periods are labelled with their last trading day."""
    result = svc.returns(session, symbols, dates.start, dates.end, frequency, cumulative, align)
    return ReturnsResponse(
        **TimeSeries.from_frame(result.table).model_dump(),
        frequency=frequency,
        cumulative=cumulative,
        align=align,
        notes=result.notes,
    )


@router.get("/rolling", response_model=RollingResponse)
def rolling(
    session: SessionDep,
    symbols: SymbolsDep,
    dates: DateRangeDep,
    metric: Annotated[Literal["volatility", "sharpe"], Query()] = "volatility",
    window: Annotated[
        int | None,
        Query(ge=2, le=2520, description="Trading days. Default 63 (volatility), 252 (sharpe)"),
    ] = None,
    risk_free: RiskFreeQuery = 0.0,
    align: AlignQuery = "inner",
):
    """Annualised volatility or Sharpe ratio over a trailing window (past data only)."""
    window = window or svc.DEFAULT_ROLLING_WINDOW[metric]
    result = svc.rolling(session, symbols, dates.start, dates.end, metric, window, risk_free, align)
    return RollingResponse(
        **TimeSeries.from_frame(result.table).model_dump(),
        metric=metric,
        window=window,
        align=align,
        notes=result.notes,
    )


@router.get("/correlation", response_model=CorrelationResponse)
def correlation(
    session: SessionDep,
    symbols: SymbolsDep,
    dates: DateRangeDep,
    method: Annotated[Literal["pearson", "spearman"], Query()] = "pearson",
    align: AlignQuery = "inner",
):
    """Correlation matrix of daily returns (needs at least two symbols)."""
    result = svc.correlation(session, symbols, dates.start, dates.end, method, align)
    return CorrelationResponse.build(result.table, method, align, result.notes)


@router.get("/drawdown/{symbol}", response_model=DrawdownResponse)
def drawdown(
    symbol: str,
    session: SessionDep,
    dates: DateRangeDep,
    top: Annotated[int, Query(ge=1, le=50, description="Deepest episodes to list")] = 5,
):
    """Drawdown below the running peak at each date, and the deepest episodes."""
    result = svc.drawdowns(session, symbol, dates.start, dates.end, top)
    return DrawdownResponse(
        symbol=result.symbol,
        max_drawdown=finite(result.series.min()),
        dates=[ts.date() for ts in result.series.index],
        drawdown=[finite(v) for v in result.series],
        episodes=[EpisodeOut.from_episode(e) for e in result.episodes],
    )
