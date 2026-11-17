import math

import numpy as np
import pandas as pd
import pytest

from fincomb.analytics import risk
from fincomb.analytics.returns import cagr, simple_returns
from fincomb.exceptions import MissingDataError
from fincomb.ingestion.csv_source import CsvSource

SQRT252 = math.sqrt(252)


def series(values, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(values), name="date")
    return pd.Series(values, index=idx, dtype="float64")


def test_periodic_rate_compounds_back_to_annual():
    daily = risk.periodic_rate(0.04)
    assert (1 + daily) ** 252 == pytest.approx(1.04)
    with pytest.raises(ValueError):
        risk.periodic_rate(-1.5)


def test_volatility_hand_computed():
    # std of [0.01, -0.01] with ddof=1 is sqrt(0.0002)
    assert risk.annualized_volatility(series([0.01, -0.01])) == pytest.approx(
        math.sqrt(0.0002) * SQRT252
    )


def test_sharpe_hand_computed():
    # mean 0.02, std sqrt(0.0002)
    assert risk.sharpe_ratio(series([0.01, 0.03])) == pytest.approx(
        0.02 / math.sqrt(0.0002) * SQRT252
    )


def test_sharpe_subtracts_risk_free():
    r = series([0.01, 0.03, 0.02, 0.00])
    rf = risk.periodic_rate(0.05)
    expected = (r - rf).mean() / (r - rf).std(ddof=1) * SQRT252
    assert risk.sharpe_ratio(r, risk_free=0.05) == pytest.approx(expected)
    assert risk.sharpe_ratio(r, risk_free=0.05) < risk.sharpe_ratio(r)


def test_sharpe_of_constant_returns_is_nan_not_inf():
    assert np.isnan(risk.sharpe_ratio(series([0.01] * 10)))


def test_sortino_hand_computed():
    r = series([0.02, -0.01, 0.03, -0.02])
    downside = math.sqrt((0.01**2 + 0.02**2) / 4)
    assert risk.sortino_ratio(r) == pytest.approx(0.005 * 252 / (downside * SQRT252))


def test_sortino_with_no_losses_is_nan():
    assert np.isnan(risk.sortino_ratio(series([0.01, 0.02, 0.03])))


def test_max_drawdown_hand_computed():
    r = simple_returns(series([100, 120, 90, 130, 65]))
    dd = risk.drawdown_series(r)
    assert dd.tolist() == pytest.approx([0.0, -0.25, 0.0, -0.5])
    assert risk.max_drawdown(r) == pytest.approx(-0.5)


def test_loss_on_first_day_counts_as_drawdown():
    assert risk.max_drawdown(series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_drawdown_of_rising_series_is_zero():
    assert risk.max_drawdown(series([0.01, 0.02, 0.0, 0.03])) == 0.0


def test_calmar():
    prices = series([100, 120, 90, 130, 65])
    r = simple_returns(prices)
    assert risk.calmar_ratio(r, start=prices.index[0]) == pytest.approx(
        cagr(r, start=prices.index[0]) / 0.5
    )


def test_calmar_negative_when_cagr_negative_and_nan_without_drawdown():
    assert risk.calmar_ratio(series([-0.1, 0.05])) < 0
    assert np.isnan(risk.calmar_ratio(series([0.01, 0.02])))


def test_drawdown_episodes():
    prices = series([100, 120, 90, 130, 65])
    eps = risk.drawdown_episodes(simple_returns(prices))
    d = prices.index
    assert len(eps) == 2
    worst, second = eps
    assert (worst.depth, worst.peak, worst.trough, worst.recovery) == (
        pytest.approx(-0.5),
        d[3],
        d[4],
        None,
    )
    assert worst.days_to_recover is None
    assert (second.depth, second.peak, second.trough, second.recovery) == (
        pytest.approx(-0.25),
        d[1],
        d[2],
        d[3],
    )
    assert (second.days_to_trough, second.days_to_recover) == (1, 1)


def test_drawdown_episode_from_starting_capital():
    eps = risk.drawdown_episodes(series([-0.1, 0.2]))
    assert eps[0].peak is None and eps[0].days_to_trough == 1


def test_drawdown_episodes_top_limit():
    r = series([0.1, -0.1, 0.2, -0.05, 0.1, -0.2, 0.3])
    assert len(risk.drawdown_episodes(r, top=2)) == 2


def test_rolling_volatility_matches_full_window_and_has_warmup():
    r = series(np.random.default_rng(0).normal(0, 0.01, 30))
    rv = risk.rolling_volatility(r, window=10)
    assert rv.iloc[:9].isna().all()
    assert rv.iloc[9] == pytest.approx(risk.annualized_volatility(r.iloc[:10]))


def test_rolling_stats_never_look_ahead():
    """Changing future returns must not change any earlier rolling value."""
    r = series(np.random.default_rng(1).normal(0, 0.01, 60))
    changed = r.copy()
    changed.iloc[40:] = changed.iloc[40:] * 5 + 0.03
    for fn in (
        lambda x: risk.rolling_volatility(x, 10),
        lambda x: risk.rolling_sharpe(x, 10, risk_free=0.02),
    ):
        pd.testing.assert_series_equal(fn(r).iloc[:40], fn(changed).iloc[:40])


def test_rolling_sharpe_matches_full_window():
    r = series(np.random.default_rng(2).normal(0.001, 0.01, 20))
    assert risk.rolling_sharpe(r, window=20).iloc[-1] == pytest.approx(risk.sharpe_ratio(r))


def test_matches_independent_numpy_calculation(fixtures_dir):
    """Cross-check against a direct numpy implementation on the fixture data."""
    from datetime import date

    prices = CsvSource(fixtures_dir).fetch_daily("SYNM", date(2023, 1, 1), date(2023, 12, 31))
    p = prices["adj_close"].to_numpy()
    r = p[1:] / p[:-1] - 1
    ours = simple_returns(prices["adj_close"])
    assert risk.annualized_volatility(ours) == pytest.approx(np.std(r, ddof=1) * SQRT252)
    assert risk.sharpe_ratio(ours) == pytest.approx(r.mean() / r.std(ddof=1) * SQRT252)
    wealth = np.concatenate([[1.0], np.cumprod(1 + r)])
    assert risk.max_drawdown(ours) == pytest.approx(
        (wealth / np.maximum.accumulate(wealth) - 1).min()
    )


def test_sortino_with_risk_free_hand_computed():
    r = series([0.02, -0.01, 0.03, -0.02, 0.0])
    rf = risk.periodic_rate(0.04)
    excess = [x - rf for x in [0.02, -0.01, 0.03, -0.02, 0.0]]
    downside = math.sqrt(sum(min(e, 0) ** 2 for e in excess) / 5) * SQRT252
    expected = (sum(excess) / 5) * 252 / downside
    assert risk.sortino_ratio(r, risk_free=0.04) == pytest.approx(expected)


def test_ratios_on_dataframe_match_per_column():
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        rng.normal(0.0005, 0.01, (100, 3)),
        columns=list("ABC"),
        index=pd.bdate_range("2024-01-01", periods=100),
    )
    for fn in (
        risk.sharpe_ratio,
        risk.sortino_ratio,
        risk.annualized_volatility,
        risk.max_drawdown,
    ):
        together = fn(df)
        for col in df.columns:
            assert together[col] == pytest.approx(fn(df[col]))


def test_drawdown_recovery_on_last_day():
    prices = series([100, 110, 99, 110])
    (ep,) = risk.drawdown_episodes(simple_returns(prices))
    assert ep.recovery == prices.index[-1] and ep.days_to_recover == 1


def test_back_to_back_drawdowns_are_separate_episodes():
    prices = series([100, 90, 100, 95, 100])  # recover exactly, then fall again
    eps = risk.drawdown_episodes(simple_returns(prices))
    assert sorted(e.depth for e in eps) == pytest.approx([-0.10, -0.05])
    assert all(e.recovery is not None for e in eps)


def test_drawdown_trough_ties_pick_first_date():
    prices = series([100, 80, 90, 80, 100])
    (ep,) = risk.drawdown_episodes(simple_returns(prices))
    assert ep.trough == prices.index[1] and ep.depth == pytest.approx(-0.2)


def test_drawdown_gap_raises_unless_zero_policy():
    r = series([0.1, np.nan, -0.2])
    with pytest.raises(MissingDataError):
        risk.max_drawdown(r)
    assert risk.max_drawdown(r, missing="zero") == pytest.approx(-0.2)
