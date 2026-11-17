from datetime import date

import numpy as np
import pandas as pd
import pytest

from fincomb.analytics.correlation import correlation_matrix, rolling_correlation
from fincomb.analytics.returns import simple_returns
from fincomb.analytics.summary import performance_summary
from fincomb.exceptions import InsufficientDataError
from fincomb.ingestion.csv_source import CsvSource


def frame(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame(cols, index=pd.bdate_range("2024-01-01", periods=n), dtype="float64")


def test_correlation_perfect_and_inverse():
    a = np.random.default_rng(0).normal(0, 0.01, 50)
    c = correlation_matrix(frame(A=a, B=2 * a, C=-a))
    assert c.loc["A", "B"] == pytest.approx(1.0)
    assert c.loc["A", "C"] == pytest.approx(-1.0)
    assert (np.diag(c) == 1).all() and c.equals(c.T)


def test_spearman_is_rank_based():
    a = np.linspace(-0.02, 0.02, 30)
    c = correlation_matrix(frame(A=a, B=a**3), method="spearman")
    assert c.loc["A", "B"] == pytest.approx(1.0)


def test_correlation_validation():
    with pytest.raises(ValueError, match="two assets"):
        correlation_matrix(frame(A=[0.01] * 30))
    with pytest.raises(ValueError, match="method"):
        correlation_matrix(frame(A=[0.01] * 30, B=[0.02] * 30), method="kendall")
    with pytest.raises(InsufficientDataError):
        correlation_matrix(frame(A=[0.01] * 5, B=[0.02] * 5))


def test_rolling_correlation():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0, 0.01, 40), index=pd.bdate_range("2024-01-01", periods=40))
    rc = rolling_correlation(a, a * 3, window=10)
    assert rc.iloc[:9].isna().all()
    assert rc.iloc[9:].to_numpy() == pytest.approx(1.0)
    with pytest.raises(InsufficientDataError):
        rolling_correlation(a.iloc[:5], a.iloc[:5], window=10)


def test_performance_summary_on_fixtures(fixtures_dir):
    src = CsvSource(fixtures_dir)
    prices = pd.concat(
        {
            s: src.fetch_daily(s, date(2023, 1, 1), date(2023, 12, 31))["adj_close"]
            for s in ["SYNA", "SYNM"]
        },
        axis=1,
    )
    table = performance_summary(prices, risk_free=0.04)
    assert list(table.index) == ["SYNA", "SYNM"]
    assert set(table.columns) >= {
        "total_return",
        "cagr",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
    }
    synm = table.loc["SYNM"]
    p = prices["SYNM"]
    assert synm.total_return == pytest.approx(p.iloc[-1] / p.iloc[0] - 1)
    assert synm.observations == len(p) - 1
    assert -1 <= synm.max_drawdown <= 0
    assert synm.best_day == pytest.approx(simple_returns(p).max())


def test_performance_summary_accepts_series():
    s = pd.Series([100.0, 101, 99, 103], index=pd.bdate_range("2024-01-01", periods=4), name="AAA")
    assert list(performance_summary(s).index) == ["AAA"]
