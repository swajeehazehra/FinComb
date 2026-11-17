"""Loading stored prices into the wide frames the analytics expect."""

from __future__ import annotations

import logging
from datetime import date
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from fincomb.db.repositories import PRICE_COLUMNS, PriceRepository
from fincomb.exceptions import InsufficientDataError, SymbolNotFoundError

logger = logging.getLogger(__name__)

Field = Literal["open", "high", "low", "close", "adj_close", "volume"]


def load_prices(
    session: Session,
    symbols: list[str],
    start: date | None = None,
    end: date | None = None,
    field: Field = "adj_close",
    align: Literal["inner", "outer"] = "inner",
) -> pd.DataFrame:
    """Wide frame of one price field: rows = dates, columns = symbols.

    align="inner" (default) keeps only dates where every symbol has a price, so
    returns for different assets always cover the same periods. "outer" keeps all
    dates and leaves NaN gaps; nothing is ever forward-filled.
    """
    if field not in PRICE_COLUMNS:
        raise ValueError(f"field must be one of {PRICE_COLUMNS}")
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not symbols:
        raise ValueError("no symbols given")

    repo = PriceRepository(session)
    columns = {}
    for symbol in symbols:
        bars = repo.get_bars(symbol, start, end)
        if bars.empty:
            raise SymbolNotFoundError(
                f"no stored prices for {symbol} in that range; run `fincomb ingest {symbol}` first"
            )
        columns[symbol] = bars[field]

    wide = pd.concat(columns, axis=1, join=align).sort_index()
    wide.columns.name = "symbol"
    if align == "inner":
        dropped = max(len(c) for c in columns.values()) - len(wide)
        if dropped:
            logger.info(
                "aligned %d symbols on %d common dates (%d dates dropped)",
                len(symbols),
                len(wide),
                dropped,
            )
    if len(wide) < 2:
        raise InsufficientDataError("fewer than 2 common dates across the requested symbols")
    return wide
