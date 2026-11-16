"""The MarketDataSource interface every adapter implements.

Every adapter returns the same normalised frame, so nothing downstream
knows or cares which provider the data came from:

    index:   DatetimeIndex named "date" (tz-naive, midnight, sorted, unique)
    columns: open, high, low, close, adj_close (float64), volume (float64, may be NaN)
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import date
from typing import TypeVar

import pandas as pd

from fincomb.exceptions import DataSourceAuthError, DataSourceError, SymbolNotFoundError

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

T = TypeVar("T")


class MarketDataSource(ABC):
    name: str = "base"

    @abstractmethod
    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return provider data for [start, end] (inclusive). May use any column names."""

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        raw = self._fetch_raw(symbol.upper(), start, end)
        df = normalise(raw)
        df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        if df.empty:
            raise SymbolNotFoundError(
                f"{self.name}: no data for {symbol} between {start} and {end}"
            )
        return df


_ALIASES = {
    "date": "date",
    "datetime": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
    "volume": "volume",
}


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert any provider frame to the standard shape described above."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS, index=pd.DatetimeIndex([], name="date"))

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):  # yfinance may return (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]

    if "date" in df.columns:
        df = df.set_index("date")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:  # keep the exchange-local calendar date, then drop tz
        idx = idx.tz_localize(None)
    df.index = idx.normalize().as_unit("ns")  # one resolution across pandas versions
    df.index.name = "date"

    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = float("nan")

    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"source frame is missing columns: {missing}")

    df = df[STANDARD_COLUMNS].apply(pd.to_numeric, errors="coerce").astype("float64")
    return df.sort_index()


def with_retries(
    fn: Callable[[], T], *, attempts: int = 3, base_delay: float = 1.0, what: str = "request"
) -> T:
    """Call fn with exponential backoff. Re-raises the last error as DataSourceError."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (SymbolNotFoundError, DataSourceAuthError):
            raise  # retrying won't help
        except Exception as exc:  # noqa: BLE001 - provider libraries raise anything
            last = exc
            if attempt < attempts:
                delay = base_delay * 2 ** (attempt - 1)
                logger.warning(
                    "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                    what,
                    attempt,
                    attempts,
                    exc,
                    delay,
                )
                time.sleep(delay)
    raise DataSourceError(f"{what} failed after {attempts} attempts: {last}") from last
