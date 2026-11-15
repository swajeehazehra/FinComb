"""Builds a data source by name, so the CLI/API can choose one at runtime."""

from __future__ import annotations

from fincomb.config import Settings
from fincomb.ingestion.base import MarketDataSource
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.stooq_source import StooqSource
from fincomb.ingestion.yfinance_source import YFinanceSource

SOURCES = ("yfinance", "stooq", "csv")


def get_source(name: str, settings: Settings) -> MarketDataSource:
    if name == "yfinance":
        return YFinanceSource()
    if name == "stooq":
        return StooqSource()
    if name == "csv":
        return CsvSource(settings.csv_data_dir)
    raise ValueError(f"unknown data source {name!r}; choose one of {SOURCES}")
