"""Reads `<SYMBOL>.csv` files from a folder. Used for tests, demos and offline work.

Expected columns: Date, Open, High, Low, Close, Adj Close (optional), Volume (optional).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from fincomb.exceptions import SymbolNotFoundError
from fincomb.ingestion.base import MarketDataSource


class CsvSource(MarketDataSource):
    name = "csv"

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def _fetch_raw(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        path = self.directory / f"{symbol.upper()}.csv"
        if not path.exists():
            raise SymbolNotFoundError(f"no CSV file for {symbol} at {path}")
        return pd.read_csv(path)
