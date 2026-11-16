"""Adapter tests. The network is never touched: HTTP and yfinance are faked."""

from datetime import date

import pandas as pd
import pytest

from fincomb.exceptions import DataSourceAuthError, DataSourceError, SymbolNotFoundError
from fincomb.ingestion.tiingo_source import TiingoSource
from fincomb.ingestion.yfinance_source import YFinanceSource

TIINGO_JSON = [
    {
        "date": "2024-01-02T00:00:00.000Z",
        "open": 187.15,
        "high": 188.44,
        "low": 183.89,
        "close": 185.64,
        "adjClose": 184.29,
        "volume": 82488700,
        "divCash": 0.0,
        "splitFactor": 1.0,
    },
    {
        "date": "2024-01-03T00:00:00.000Z",
        "open": 184.22,
        "high": 185.88,
        "low": 183.43,
        "close": 184.25,
        "adjClose": 182.91,
        "volume": 58414500,
        "divCash": 0.0,
        "splitFactor": 1.0,
    },
]


def test_tiingo_parses_json_and_sends_token_in_header_not_url():
    seen = {}

    def fake_get(url, token):
        seen["url"], seen["token"] = url, token
        return TIINGO_JSON

    df = TiingoSource("secret123", http_get_json=fake_get).fetch_daily(
        "aapl", date(2024, 1, 1), date(2024, 1, 5)
    )
    assert "/daily/AAPL/prices" in seen["url"]
    assert "startDate=2024-01-01" in seen["url"] and "endDate=2024-01-05" in seen["url"]
    assert "secret123" not in seen["url"] and seen["token"] == "secret123"
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df["adj_close"].tolist() == [184.29, 182.91]  # adjusted close kept separately
    assert df["close"].iloc[0] == 185.64


@pytest.mark.parametrize("key", [None, ""])
def test_tiingo_without_key_explains_how_to_get_one(key):
    called = []
    src = TiingoSource(key, http_get_json=lambda *a: called.append(a))
    with pytest.raises(DataSourceAuthError, match="tiingo.com/account/api/token"):
        src.fetch_daily("AAPL", date(2024, 1, 1), date(2024, 1, 5))
    assert called == []  # no request made without a key


def test_tiingo_auth_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = []

    def bad_token(url, token):
        calls.append(url)
        raise DataSourceAuthError("Tiingo rejected the API token")

    with pytest.raises(DataSourceAuthError):
        TiingoSource("wrong", http_get_json=bad_token).fetch_daily(
            "AAPL", date(2024, 1, 1), date(2024, 1, 5)
        )
    assert len(calls) == 1


def test_tiingo_empty_list_is_not_found():
    src = TiingoSource("k", http_get_json=lambda *a: [], attempts=1)
    with pytest.raises(SymbolNotFoundError):
        src.fetch_daily("AAPL", date(2024, 1, 1), date(2024, 1, 5))


def test_tiingo_error_detail_is_surfaced(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    src = TiingoSource("k", http_get_json=lambda *a: {"detail": "Error: something"}, attempts=1)
    with pytest.raises(DataSourceError, match="something"):
        src.fetch_daily("AAPL", date(2024, 1, 1), date(2024, 1, 5))


@pytest.mark.parametrize("sym,expected", [("AAPL", "AAPL"), ("brk.b", "BRK-B")])
def test_tiingo_ticker_mapping(sym, expected):
    assert TiingoSource.tiingo_ticker(sym) == expected


def test_registry_reads_tiingo_key_from_settings():
    from pydantic import SecretStr

    from fincomb.config import Settings
    from fincomb.ingestion.registry import get_source

    settings = Settings(postgres_password=SecretStr("x"), tiingo_api_key=SecretStr("abc"))
    src = get_source("tiingo", settings)
    assert isinstance(src, TiingoSource) and src.api_key == "abc"
    with pytest.raises(ValueError, match="unknown data source"):
        get_source("stooq", settings)


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


@pytest.mark.parametrize(
    "status,exc,match",
    [
        (401, DataSourceAuthError, "rejected the API token"),
        (404, SymbolNotFoundError, "not found"),
        (429, DataSourceError, "rate limit"),
        (500, DataSourceError, "HTTP 500"),
    ],
)
def test_tiingo_http_errors_are_mapped(monkeypatch, status, exc, match):
    import io
    import urllib.error

    from fincomb.ingestion import tiingo_source

    def fake_urlopen(req, timeout):
        assert req.get_header("Authorization") == "Token k"
        raise urllib.error.HTTPError(req.full_url, status, "err", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(tiingo_source.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(exc, match=match):
        tiingo_source._http_get_json("https://api.tiingo.com/x", "k")
