import math

import numpy as np
import pandas as pd
import pytest

from fincomb.analytics import returns as R
from fincomb.exceptions import InsufficientDataError, MissingDataError


def series(values, start="2024-01-01", name="X"):
    idx = pd.bdate_range(start, periods=len(values), name="date")
    return pd.Series(values, index=idx, dtype="float64", name=name)


def test_simple_returns_hand_computed():
    r = R.simple_returns(series([100, 110, 99]))
    assert r.tolist() == pytest.approx([0.10, -0.10])
    assert len(r) == 2  # first price has no return


def test_log_returns_hand_computed():
    r = R.log_returns(series([100, 110, 99]))
    assert r.tolist() == pytest.approx([math.log(1.1), math.log(0.9)])


def test_total_and_cumulative_return():
    r = R.simple_returns(series([100, 110, 99]))
    assert R.total_return(r) == pytest.approx(-0.01)  # 1.1 * 0.9 - 1
    assert R.cumulative_returns(r).tolist() == pytest.approx([0.10, -0.01])
    assert R.wealth_index(r, 1000).tolist() == pytest.approx([1100, 990])


def test_cagr_uses_calendar_time_from_start_price():
    # 100 -> 121 over exactly two calendar years (731 days incl. a leap day).
    idx = pd.DatetimeIndex(["2023-01-02", "2024-01-02", "2025-01-02"])
    prices = pd.Series([100.0, 110.0, 121.0], index=idx)
    r = R.simple_returns(prices)
    years = 731 / 365.25
    assert R.cagr(r, start=idx[0]) == pytest.approx(1.21 ** (1 / years) - 1)


def test_cagr_is_not_inflated_by_missing_days():
    """Dropping trading days must not change CAGR: only the dates matter."""
    idx = pd.bdate_range("2023-01-02", "2024-01-02")
    prices = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    full = R.cagr(R.simple_returns(prices), start=idx[0])
    sparse = prices.iloc[::5]  # keep every 5th day, same first and last date
    sparse = pd.concat([sparse, prices.iloc[[-1]]]).loc[lambda s: ~s.index.duplicated()]
    thin = R.cagr(R.simple_returns(sparse), start=idx[0])
    assert thin == pytest.approx(full)
    assert full == pytest.approx(1.2 ** (365.25 / 365) - 1)


def test_cagr_default_start_is_one_business_day_before_first_return():
    r = series([0.10, 0.0], start="2024-01-02")  # returns on Tue 2 Jan and Wed 3 Jan
    expected_years = (pd.Timestamp("2024-01-03") - pd.Timestamp("2024-01-01")).days / 365.25
    assert R.cagr(r) == pytest.approx(1.1 ** (1 / expected_years) - 1)


def test_cagr_dataframe_uses_each_assets_own_dates():
    idx = pd.bdate_range("2024-01-01", periods=4)
    r = pd.DataFrame({"A": [0.01, 0.01, 0.01, 0.01], "B": [np.nan, np.nan, 0.02, 0.02]}, index=idx)
    out = R.cagr(r)
    assert out["A"] == pytest.approx(R.cagr(r["A"]))
    assert out["B"] == pytest.approx(R.cagr(r["B"].dropna()))


def test_annualized_mean_return():
    assert R.annualized_mean_return(series([0.01, 0.03])) == pytest.approx(0.02 * 252)


def test_dataframe_input_is_columnwise():
    prices = pd.DataFrame(
        {"A": [100, 110, 121], "B": [50, 50, 25]}, index=pd.bdate_range("2024-01-01", periods=3)
    )
    total = R.total_return(R.simple_returns(prices))
    assert total["A"] == pytest.approx(0.21)
    assert total["B"] == pytest.approx(-0.5)


def test_gaps_are_not_filled():
    r = R.simple_returns(series([100, np.nan, 110, 121]))
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])  # no return invented across the gap
    assert r.iloc[2] == pytest.approx(0.10)
    gap = R.simple_returns(series([100, 105, np.nan, 110, 121]))  # gap mid-series
    with pytest.raises(MissingDataError):  # compounding refuses to skip it silently
        R.total_return(gap)


def test_resample_monthly_compounds_daily_returns():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-02-01"])
    r = pd.Series([0.10, 0.10, -0.50], index=idx)
    monthly = R.resample_returns(r, "monthly")
    assert monthly.tolist() == pytest.approx([0.21, -0.50])
    assert list(monthly.index.month) == [1, 2]


def test_resample_rejects_unknown_frequency():
    with pytest.raises(ValueError, match="frequency"):
        R.resample_returns(series([0.01, 0.02]), "hourly")


@pytest.mark.parametrize(
    "prices,error",
    [
        ([100], InsufficientDataError),
        ([100, 0, 110], ValueError),
        ([100, -5], ValueError),
    ],
)
def test_bad_prices_rejected(prices, error):
    with pytest.raises(error):
        R.simple_returns(series(prices))


def test_unsorted_prices_rejected():
    s = series([100, 101, 102]).iloc[::-1]
    with pytest.raises(ValueError, match="sorted"):
        R.simple_returns(s)


def test_non_datetime_index_rejected():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        R.simple_returns(pd.Series([100.0, 101.0]))


def test_impossible_return_rejected():
    with pytest.raises(ValueError, match="-100%"):
        R.total_return(series([0.1, -1.0]))


def test_gap_inside_series_raises_by_default():
    r = series([0.01, np.nan, 0.02])
    with pytest.raises(MissingDataError, match="missing return"):
        R.total_return(r)
    with pytest.raises(MissingDataError):
        R.wealth_index(r)


def test_gap_can_be_treated_as_flat_explicitly():
    r = series([0.10, np.nan, 0.10])
    assert R.total_return(r, missing="zero") == pytest.approx(0.21)
    assert R.wealth_index(r, missing="zero").tolist() == pytest.approx([1.1, 1.1, 1.21])


def test_leading_and_trailing_nans_are_not_gaps():
    """An asset that started trading later (outer alignment) is not an error."""
    r = pd.DataFrame(
        {"A": [0.1, 0.1, 0.1], "B": [np.nan, 0.1, 0.1]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    w = R.wealth_index(r)
    assert np.isnan(w["B"].iloc[0])
    assert w["B"].iloc[-1] == pytest.approx(1.21)
    assert R.total_return(r)["A"] == pytest.approx(0.331)


def test_missing_policy_validated():
    with pytest.raises(ValueError, match="missing"):
        R.total_return(series([0.1, 0.1]), missing="drop")


def test_resample_labels_use_last_trading_day():
    # August 2024 ended on Saturday the 31st; the last trading day was Friday the 30th.
    idx = pd.DatetimeIndex(["2024-08-29", "2024-08-30", "2024-09-03"])
    monthly = R.resample_returns(pd.Series([0.01, 0.02, 0.03], index=idx), "monthly")
    assert list(monthly.index) == [pd.Timestamp("2024-08-30"), pd.Timestamp("2024-09-03")]
    assert monthly.iloc[0] == pytest.approx(1.01 * 1.02 - 1)


def test_resample_weekly_and_yearly_labels_are_trading_days():
    r = series([0.01] * 15, start="2024-12-23")  # crosses a year end
    weekly = R.resample_returns(r, "weekly")
    yearly = R.resample_returns(r, "yearly")
    assert set(weekly.index) <= set(r.index)
    assert list(yearly.index) == [pd.Timestamp("2024-12-31"), r.index[-1]]
