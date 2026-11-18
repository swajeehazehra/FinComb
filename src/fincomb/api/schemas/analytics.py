from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, Field

from fincomb.analytics.risk import DrawdownEpisode
from fincomb.api.schemas.common import TimeSeries, finite, to_date

NOTES_FIELD = Field(
    default_factory=list,
    description="Why the common period is shorter than the full data (inner alignment)",
)


class PerformanceRow(BaseModel):
    symbol: str
    start: date
    end: date
    observations: int
    total_return: float | None
    cagr: float | None
    annual_volatility: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    calmar: float | None
    best_day: float | None
    worst_day: float | None


class PerformanceResponse(BaseModel):
    align: str
    risk_free: float
    rows: list[PerformanceRow]
    notes: list[str] = NOTES_FIELD

    @classmethod
    def build(cls, table: pd.DataFrame, align: str, risk_free: float, notes: list[str]):
        rows = []
        for symbol, r in table.iterrows():
            rows.append(
                PerformanceRow(
                    symbol=str(symbol),
                    start=r["start"],
                    end=r["end"],
                    observations=int(r["observations"]),
                    **{
                        k: finite(r[k])
                        for k in PerformanceRow.model_fields
                        if k not in ("symbol", "start", "end", "observations")
                    },
                )
            )
        return cls(align=align, risk_free=risk_free, rows=rows, notes=notes)


class ReturnsResponse(TimeSeries):
    frequency: str
    cumulative: bool
    align: str
    notes: list[str] = NOTES_FIELD


class RollingResponse(TimeSeries):
    metric: str
    window: int
    align: str
    notes: list[str] = NOTES_FIELD


class CorrelationResponse(BaseModel):
    method: str
    align: str
    symbols: list[str]
    matrix: list[list[float | None]] = Field(
        description="matrix[i][j] = correlation of symbols[i] with symbols[j]"
    )
    notes: list[str] = NOTES_FIELD

    @classmethod
    def build(cls, corr: pd.DataFrame, method: str, align: str, notes: list[str]):
        return cls(
            method=method,
            align=align,
            symbols=[str(c) for c in corr.columns],
            matrix=[[finite(v) for v in row] for row in corr.to_numpy()],
            notes=notes,
        )


class EpisodeOut(BaseModel):
    peak: date | None = Field(description="Last date at the old high (null = starting capital)")
    trough: date
    recovery: date | None = Field(description="First date back at the old high (null = not yet)")
    depth: float
    days_to_trough: int
    days_to_recover: int | None

    @classmethod
    def from_episode(cls, e: DrawdownEpisode) -> EpisodeOut:
        return cls(
            peak=to_date(e.peak),
            trough=e.trough.date(),
            recovery=to_date(e.recovery),
            depth=e.depth,
            days_to_trough=e.days_to_trough,
            days_to_recover=e.days_to_recover,
        )


class DrawdownResponse(BaseModel):
    symbol: str
    max_drawdown: float | None
    dates: list[date]
    drawdown: list[float | None]
    episodes: list[EpisodeOut]
