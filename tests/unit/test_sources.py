"""Adapter tests. The network is never touched: HTTP and yfinance are faked."""

from datetime import date

import pandas as pd
import pytest

from fincomb.exceptions import DataSourceError, SymbolNotFoundError
from fincomb.ingestion.stooq_source import StooqSource
from fincomb.ingestion.yfinance_source import YFinanceSource

STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,187.15,188.44,183.89,185.64,82488700
2024-01-03,184.22,185.88,183.43,184.25,58414500
"""


def test_stooq_parses_csv_and_builds_url():
    seen = {}

    def fake_get(url):
        seen["url"] = url
        return STOOQ_CSV

    df = StooqSource(http_get=fake_get).fetch_daily("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert "s=aapl.us" in seen["url"] and "d1=20240101" in seen["url"]
    assert len(df) == 2
    assert (df["adj_close"] == df["close"]).all()  # stooq is already adjusted


def test_stooq_no_data():
    src = StooqSource(http_get=lambda _u: "No data", attempts=1)
    with pytest.raises(SymbolNotFoundError):
        src.fetch_daily("ZZZZ", date(2024, 1, 1), date(2024, 1, 5))


def test_stooq_unexpected_response(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    src = StooqSource(http_get=lambda _u: "<html>captcha</html>", attempts=2)
    with pytest.raises(DataSourceError):
        src.fetch_daily("AAPL", date(2024, 1, 1), date(2024, 1, 5))


@pytest.mark.parametrize(
    "sym,expected", [("AAPL", "aapl.us"), ("vod.uk", "vod.uk"), ("^spx", "^spx")]
)
def test_stooq_symbol_mapping(sym, expected):
    assert StooqSource.stooq_symbol(sym) == expected


def test_yfinance_end_is_made_inclusive(monkeypatch):
    seen = {}

    def fake_download(symbol, **kw):
        seen.update(kw)
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-05"], name="Date")
        return pd.DataFrame(
            {
                "Open": [1, 1],
                "High": [2, 2],
                "Low": [0.5, 0.5],
                "Close": [1.5, 1.6],
                "Adj Close": [1.4, 1.5],
                "Volume": [10, 11],
            },
            index=idx,
        )

    import yfinance

    monkeypatch.setattr(yfinance, "download", fake_download)
    df = YFinanceSource().fetch_daily("AAPL", date(2024, 1, 2), date(2024, 1, 5))
    assert seen["end"] == "2024-01-06"
    assert seen["auto_adjust"] is False
    assert len(df) == 2 and df["adj_close"].iloc[-1] == 1.5


def test_yfinance_empty_is_not_found(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(SymbolNotFoundError):
        YFinanceSource(attempts=1).fetch_daily("ZZZZ", date(2024, 1, 2), date(2024, 1, 5))
