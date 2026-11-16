"""Yahoo Finance via the unofficial `yfinance` library (free, no API key)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from fincomb.exceptions import SymbolNotFoundError
from fincomb.ingestion.base import MarketDataSource, with_retries


class YFinanceSource(MarketDataSource):
    name = "yfinance"

    def __init__(self, attempts: int = 3, base_delay: float = 2.0) -> None:
        self.attempts = attempts
        self.base_delay = base_delay

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf  # imported lazily so tests never need it

        def call() -> pd.DataFrame:
            df = yf.download(
                symbol,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
                auto_adjust=False,  # keep raw OHLC *and* Adj Close
                actions=False,
                progress=False,
                threads=False,
                multi_level_index=False,
            )
            if df is None or df.empty:
                # yfinance logs errors instead of raising; treat empty as "not found".
                raise SymbolNotFoundError(
                    f"yfinance returned no data for {symbol} (unknown ticker, no "
                    "internet, or Yahoo rate limit; try again later or use --source tiingo)"
                )
            return df

        return with_retries(
            call,
            attempts=self.attempts,
            base_delay=self.base_delay,
            what=f"yfinance download {symbol}",
        )
