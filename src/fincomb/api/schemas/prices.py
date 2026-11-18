from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel

from fincomb.api.schemas.common import finite


class Bar(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int | None


class PricesResponse(BaseModel):
    symbol: str
    adjusted: bool
    count: int
    bars: list[Bar]

    @classmethod
    def from_frame(cls, symbol: str, adjusted: bool, bars: pd.DataFrame) -> PricesResponse:
        rows = [
            Bar(
                date=ts.date(),
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                adj_close=row.adj_close,
                volume=None if finite(row.volume) is None else int(row.volume),
            )
            for ts, row in zip(bars.index, bars.itertuples(index=False), strict=True)
        ]
        return cls(symbol=symbol, adjusted=adjusted, count=len(rows), bars=rows)
