import math

import numpy as np
import pandas as pd
import pytest

from fincomb.analytics import returns as R
from fincomb.exceptions import InsufficientDataError


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


def test_cagr_one_year_equals_total_return():
    r = series([0.001] * 252)
    assert R.cagr(r) == pytest.approx(1.001**252 - 1)


def test_cagr_two_years_is_annualised():
    r = series([0.001] * 504)
    assert R.cagr(r) == pytest.approx(1.001**252 - 1)


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
