from datetime import date

import pandas as pd
import pytest

from fincomb.db.models import Severity
from fincomb.ingestion.csv_source import CsvSource
from fincomb.validation.rules import validate_bars


def bars(rows):
    """rows: list of (date, open, high, low, close) with adj_close = close."""
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    df["adj_close"] = df["close"]
    df["volume"] = 1000.0
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def rules(result, severity=None):
    return sorted(i.rule for i in result.issues if severity is None or i.severity is severity)


def test_clean_data_passes_untouched():
    df = bars([("2024-01-02", 10, 11, 9, 10.5), ("2024-01-03", 10.5, 11, 10, 10.8)])
    res = validate_bars(df, today=date(2024, 6, 1))
    assert res.issues == []
    pd.testing.assert_frame_equal(res.clean, df)


def test_input_is_not_mutated():
    df = bars([("2024-01-02", 10, 11, 9, -1)])
    before = df.copy()
    validate_bars(df, today=date(2024, 6, 1))
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(
    "row,rule",
    [
        (("2024-01-03", 10, 11, 9, float("nan")), "missing_value"),
        (("2024-01-03", 10, 11, 0, 10), "non_positive_price"),
        (("2024-01-03", 10, 9, 8, 10), "ohlc_inconsistent"),  # high < open
        (("2024-01-03", 10, 12, 10.5, 11), "ohlc_inconsistent"),  # low > open
    ],
)
def test_error_rules_reject_row(row, rule):
    df = bars([("2024-01-02", 10, 11, 9, 10), row])
    res = validate_bars(df, today=date(2024, 6, 1))
    assert rule in rules(res, Severity.ERROR)
    assert len(res.clean) == 1
    assert res.rejected == 1


def test_negative_volume_rejected():
    df = bars([("2024-01-02", 10, 11, 9, 10)])
    df["volume"] = -5.0
    res = validate_bars(df, today=date(2024, 6, 1))
    assert rules(res) == ["negative_volume"] and res.clean.empty


def test_future_dates_rejected():
    df = bars([("2024-01-02", 10, 11, 9, 10), ("2099-01-02", 10, 11, 9, 10)])
    res = validate_bars(df, today=date(2024, 6, 1))
    assert rules(res) == ["future_date"]
    assert len(res.clean) == 1


def test_duplicates_keep_first():
    df = bars([("2024-01-02", 10, 11, 9, 10), ("2024-01-02", 20, 21, 19, 20)])
    res = validate_bars(df, today=date(2024, 6, 1))
    assert rules(res) == ["duplicate_date"]
    assert res.clean["close"].tolist() == [10]


def test_extreme_return_is_warning_not_rejection():
    df = bars([("2024-01-02", 10, 11, 9, 10), ("2024-01-03", 10, 21, 9, 20)])  # +100%
    res = validate_bars(df, today=date(2024, 6, 1), extreme_return_threshold=0.4)
    assert rules(res, Severity.WARNING) == ["extreme_return"]
    assert len(res.clean) == 2


def test_stale_price_flagged_once():
    rows = [(f"2024-01-{d:02d}", 10, 10, 10, 10) for d in range(2, 12)]  # 10 equal closes
    res = validate_bars(bars(rows), today=date(2024, 6, 1), stale_price_run=5)
    assert rules(res) == ["stale_price"]


def test_bad_fixture_catches_every_rule(fixtures_dir):
    raw = CsvSource(fixtures_dir).fetch_daily("BADDATA", date(2023, 1, 1), date(2023, 12, 31))
    res = validate_bars(raw, today=date(2024, 1, 1))
    found = set(rules(res))
    assert {
        "missing_value",
        "non_positive_price",
        "ohlc_inconsistent",
        "negative_volume",
        "duplicate_date",
        "extreme_return",
        "stale_price",
    } <= found
    # Everything that survived would satisfy the database CHECK constraints.
    c = res.clean
    assert (c[["open", "high", "low", "close", "adj_close"]] > 0).all().all()
    assert (c["high"] >= c[["open", "close", "low"]].max(axis=1)).all()
    assert (c["low"] <= c[["open", "close"]].min(axis=1)).all()
    assert not c.index.duplicated().any()
