from datetime import date

import numpy as np
import pandas as pd
import pytest

from fincomb.exceptions import DataSourceError, SymbolNotFoundError
from fincomb.ingestion.base import STANDARD_COLUMNS, normalise, with_retries
from fincomb.ingestion.csv_source import CsvSource


def test_normalise_yfinance_style_frame():
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-02"], name="Date")
    raw = pd.DataFrame(
        {
            "Open": [2, 1],
            "High": [3, 2],
            "Low": [1, 0.5],
            "Close": [2.5, 1.5],
            "Adj Close": [2.4, 1.4],
            "Volume": [10, 20],
        },
        index=idx,
    )
    df = normalise(raw)
    assert list(df.columns) == STANDARD_COLUMNS
    assert df.index.is_monotonic_increasing
    assert df.index.name == "date"
    assert df.loc["2024-01-02", "adj_close"] == 1.4
    assert all(df.dtypes == np.float64)


def test_normalise_flattens_multiindex_and_drops_timezone():
    idx = pd.DatetimeIndex(["2024-01-02 00:00"], tz="America/New_York")
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"], ["AAPL"]])
    raw = pd.DataFrame([[1, 2, 0.5, 1.5, 100]], index=idx, columns=cols)
    df = normalise(raw)
    assert df.index.tz is None
    assert df.index[0] == pd.Timestamp("2024-01-02")  # local calendar date is kept
    assert df["adj_close"].iloc[0] == 1.5  # falls back to close when no adj close


def test_normalise_missing_column_raises():
    raw = pd.DataFrame({"Date": ["2024-01-02"], "Open": [1.0]})
    with pytest.raises(DataSourceError, match="missing columns"):
        normalise(raw)


def test_csv_source_filters_to_requested_dates(fixtures_dir):
    df = CsvSource(fixtures_dir).fetch_daily("syna", date(2023, 3, 1), date(2023, 3, 31))
    assert df.index.min() >= pd.Timestamp("2023-03-01")
    assert df.index.max() <= pd.Timestamp("2023-03-31")
    assert len(df) == 23  # business days in March 2023


def test_csv_source_unknown_symbol(fixtures_dir):
    with pytest.raises(SymbolNotFoundError):
        CsvSource(fixtures_dir).fetch_daily("NOPE", date(2023, 1, 1), date(2023, 2, 1))


def test_csv_source_empty_range_raises(fixtures_dir):
    with pytest.raises(SymbolNotFoundError):
        CsvSource(fixtures_dir).fetch_daily("SYNA", date(2030, 1, 1), date(2030, 2, 1))


def test_start_after_end_rejected(fixtures_dir):
    with pytest.raises(ValueError):
        CsvSource(fixtures_dir).fetch_daily("SYNA", date(2023, 2, 1), date(2023, 1, 1))


def test_with_retries_recovers(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("boom")
        return "ok"

    assert with_retries(flaky, attempts=3) == "ok"
    assert calls["n"] == 3


def test_with_retries_gives_up(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    with pytest.raises(DataSourceError, match="after 2 attempts"):
        with_retries(lambda: (_ for _ in ()).throw(TimeoutError("slow")), attempts=2)


def test_with_retries_does_not_retry_not_found(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = {"n": 0}

    def missing():
        calls["n"] += 1
        raise SymbolNotFoundError("nope")

    with pytest.raises(SymbolNotFoundError):
        with_retries(missing, attempts=5)
    assert calls["n"] == 1
