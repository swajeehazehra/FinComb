from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from fincomb.api.deps import DateRangeDep, SessionDep
from fincomb.api.schemas.common import ERROR_RESPONSES
from fincomb.api.schemas.prices import PricesResponse
from fincomb.services.market_data import price_bars

router = APIRouter(prefix="/prices", tags=["prices"], responses=ERROR_RESPONSES)


@router.get("/{symbol}", response_model=PricesResponse)
def get_prices(
    symbol: str,
    session: SessionDep,
    dates: DateRangeDep,
    adjusted: Annotated[
        bool,
        Query(description="Scale open/high/low/close by adj_close / close"),
    ] = False,
):
    """Stored daily bars, oldest first. An empty list means none in that date range."""
    bars = price_bars(session, symbol, dates.start, dates.end, adjusted=adjusted)
    return PricesResponse.from_frame(symbol.strip().upper(), adjusted, bars)
