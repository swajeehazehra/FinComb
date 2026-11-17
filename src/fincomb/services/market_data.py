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

    With inner alignment, the plain-English reasons the common period is shorter
    than the full data (see `alignment_warnings`) are logged and also kept in
    ``wide.attrs["alignment_notes"]``.
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
    notes = alignment_warnings(columns, wide) if align == "inner" else []
    for message in notes:
        logger.warning(message)
    wide.attrs["alignment_notes"] = notes  # shown to the user by the CLI
    if len(wide) < 2:
        raise InsufficientDataError("fewer than 2 common dates across the requested symbols")
    return wide


def alignment_warnings(columns: dict[str, pd.Series], aligned: pd.DataFrame) -> list[str]:
    """Explain which symbol(s) shortened the common period, in plain words.

    Example: "Common period starts 2021-03-01 because QQQM has no data before then
    (other symbols start as early as 2015-01-02)."
    """
    if len(columns) < 2 or aligned.empty:
        return []
    msgs = []
    firsts = {s: c.index.min() for s, c in columns.items()}
    lasts = {s: c.index.max() for s, c in columns.items()}
    earliest, latest = min(firsts.values()), max(lasts.values())
    common_start, common_end = aligned.index.min(), aligned.index.max()
    if common_start > earliest:
        limiting = sorted(s for s, d in firsts.items() if d == max(firsts.values()))
        msgs.append(
            f"Common period starts {common_start.date()} because {', '.join(limiting)} "
            f"has no data before then (other symbols start as early as {earliest.date()}). "
            "Use --align outer to analyse each symbol over its own history."
        )
    if common_end < latest:
        limiting = sorted(s for s, d in lasts.items() if d == min(lasts.values()))
        msgs.append(
            f"Common period ends {common_end.date()} because {', '.join(limiting)} "
            f"has no data after then (others run to {latest.date()})."
        )
    in_window = {
        sym: c.index[(c.index >= common_start) & (c.index <= common_end)]
        for sym, c in columns.items()
    }
    dropped = max(len(idx.difference(aligned.index)) for idx in in_window.values())
    if dropped:
        msgs.append(
            f"{dropped} date(s) inside the common period were dropped because at "
            "least one symbol has no price on them."
        )
    return msgs
