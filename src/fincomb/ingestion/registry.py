"""Builds a data source by name, so the CLI/API can choose one at runtime."""

from __future__ import annotations

from fincomb.config import Settings
from fincomb.ingestion.base import MarketDataSource
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.tiingo_source import TiingoSource
from fincomb.ingestion.yfinance_source import YFinanceSource

SOURCES = ("yfinance", "tiingo", "csv")


def get_source(name: str, settings: Settings) -> MarketDataSource:
    if name == "yfinance":
        return YFinanceSource()
    if name == "tiingo":
        key = settings.tiingo_api_key
        return TiingoSource(key.get_secret_value() if key else None)
    if name == "csv":
        return CsvSource(settings.csv_data_dir)
    raise ValueError(f"unknown data source {name!r}; choose one of {SOURCES}")
