"""Shared response pieces and pandas-to-JSON helpers."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str = Field(examples=["symbol_not_found"])
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


# Documented on every route, so /docs shows the error shape.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Unknown symbol or resource"},
    422: {"model": ErrorResponse, "description": "Invalid request or not enough data"},
}


def finite(x: Any) -> float | None:
    """JSON has no NaN or infinity: an undefined statistic becomes null."""
    if x is None:
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def to_date(ts: Any) -> date | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or ts is pd.NaT:
        return None
    return pd.Timestamp(ts).date()


class TimeSeries(BaseModel):
    """Column-oriented series: `series[symbol][i]` is the value on `dates[i]`."""

    dates: list[date]
    series: dict[str, list[float | None]]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame | pd.Series) -> TimeSeries:
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(frame.name or "value")
        return cls(
            dates=[ts.date() for ts in frame.index],
            series={str(col): [finite(v) for v in frame[col]] for col in frame.columns},
        )
