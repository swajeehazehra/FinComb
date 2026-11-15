"""Stooq.com daily CSV endpoint: the fallback source.

Note: Stooq's prices are already split/dividend adjusted and it does not
publish a separate unadjusted series, so adj_close == close for this source.
US tickers need the ".us" suffix, which is added automatically.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from datetime import date

import pandas as pd

from fincomb.exceptions import DataSourceError, SymbolNotFoundError
from fincomb.ingestion.base import MarketDataSource, with_retries

STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"


def _http_get(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "fincomb/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return resp.read().decode("utf-8", errors="replace")


class StooqSource(MarketDataSource):
    name = "stooq"

    def __init__(
        self, http_get: Callable[[str], str] = _http_get, attempts: int = 3, base_delay: float = 2.0
    ) -> None:
        self.http_get = http_get  # injectable so tests never touch the network
        self.attempts = attempts
        self.base_delay = base_delay

    @staticmethod
    def stooq_symbol(symbol: str) -> str:
        s = symbol.lower()
        return s if "." in s or s.startswith("^") else f"{s}.us"

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        url = STOOQ_URL.format(
            symbol=self.stooq_symbol(symbol), d1=start.strftime("%Y%m%d"), d2=end.strftime("%Y%m%d")
        )

        def call() -> pd.DataFrame:
            text = self.http_get(url)
            head = text.lstrip()[:200].lower()
            if not head or head.startswith("no data"):
                raise SymbolNotFoundError(f"stooq has no data for {symbol}")
            if not head.startswith("date"):
                raise DataSourceError(f"stooq returned an unexpected response: {head[:80]!r}")
            return pd.read_csv(io.StringIO(text))

        return with_retries(
            call,
            attempts=self.attempts,
            base_delay=self.base_delay,
            what=f"stooq download {symbol}",
        )
