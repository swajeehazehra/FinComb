"""Portfolio risk analysis (portfolio/analysis.py)."""

import numpy as np
import pandas as pd
import pytest

from fincomb.analytics import risk
from fincomb.analytics.summary import performance_summary
from fincomb.exceptions import MissingDataError
from fincomb.portfolio import analysis as PA
from tests.unit.test_portfolio_construction import prices_from_returns, random_prices


def _weights(d):
    return pd.Series(d, dtype="float64")


def test_check_weights():
    PA.check_weights(_weights({"A": 0.6, "B": 0.4}))
    PA.check_weights(_weights({"A": 0.6, "B": 0.4000005}))  # within 1e-6
    with pytest.raises(ValueError, match="sum to 0.980000"):
        PA.check_weights(_weights({"A": 0.58, "B": 0.4}))
    with pytest.raises(ValueError, match="long-only"):
        PA.check_weights(_weights({"A": 1.2, "B": -0.2}))
    with pytest.raises(ValueError, match="once"):
        PA.check_weights(pd.Series([0.5, 0.5], index=["A", "A"]))
    with pytest.raises(ValueError):
        PA.check_weights(pd.Series(dtype="float64"))
    with pytest.raises(ValueError, match="finite"):
        PA.check_weights(_weights({"A": np.nan, "B": 1.0}))


def test_portfolio_returns_are_constant_mix():
    prices = random_prices(1, n_assets=3)
    w = _weights({"A0": 0.5, "A1": 0.3, "A2": 0.2})
    r = prices.pct_change().iloc[1:]
    expected = 0.5 * r["A0"] + 0.3 * r["A1"] + 0.2 * r["A2"]
    pd.testing.assert_series_equal(
        PA.portfolio_returns(r, w), expected.rename("portfolio"), check_names=True
    )


def test_portfolio_value_starts_at_one_on_first_price_date():
    prices = random_prices(2, n_assets=2)
    v = PA.portfolio_value(prices, _weights({"A0": 0.5, "A1": 0.5}))
    assert v.index[0] == prices.index[0] and v.iloc[0] == 1.0
    assert len(v) == len(prices)


def test_portfolio_returns_reject_gaps_and_unknown_assets():
    r = random_prices(3, n_assets=2).pct_change().iloc[1:]
    with pytest.raises(ValueError, match="ZZZ"):
        PA.portfolio_returns(r, _weights({"A0": 0.5, "ZZZ": 0.5}))
    r.iloc[5, 0] = np.nan
    with pytest.raises(MissingDataError):
        PA.portfolio_returns(r, _weights({"A0": 0.5, "A1": 0.5}))


def test_risk_contributions_euler_identity():
    prices = random_prices(4, n_assets=4, mean=None)
    w = _weights({"A0": 0.1, "A1": 0.2, "A2": 0.3, "A3": 0.4})
    cov = prices.pct_change().iloc[1:].cov()
    rc = PA.risk_contributions(w, cov)
    vol = np.sqrt(w @ cov.to_numpy() @ w * 252)
    assert rc["component"].sum() == pytest.approx(vol, rel=1e-12)
    assert rc["percent"].sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rc["component"], rc["weight"] * rc["marginal"])


def test_risk_contributions_two_assets_by_hand():
    # vols 20% and 10% (annual), correlation 0, weights 50/50:
    # sigma_p^2 = 0.25*0.04 + 0.25*0.01 = 0.0125; CR_1 = 0.5*0.5*0.04/sigma_p
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["X", "Y"], columns=["X", "Y"]) / 252
    rc = PA.risk_contributions(_weights({"X": 0.5, "Y": 0.5}), cov)
    sigma = np.sqrt(0.0125)
    assert rc.loc["X", "component"] == pytest.approx(0.01 / sigma)
    assert rc.loc["X", "percent"] == pytest.approx(0.8)
    assert rc.loc["Y", "percent"] == pytest.approx(0.2)


def test_negative_contribution_from_a_hedge():
    cov = pd.DataFrame([[0.04, -0.015], [-0.015, 0.01]], index=["EQ", "HEDGE"]).T / 252
    cov.index = cov.columns = ["EQ", "HEDGE"]
    rc = PA.risk_contributions(_weights({"EQ": 0.8, "HEDGE": 0.2}), cov)
    assert rc.loc["HEDGE", "component"] < 0
    assert rc["percent"].sum() == pytest.approx(1.0)


def test_risk_report_matches_the_building_blocks():
    prices = random_prices(5, n_assets=3, mean=None)
    w = _weights({"A0": 0.5, "A1": 0.3, "A2": 0.2})
    rep = PA.risk_report(prices, w, benchmark="A0", confidence=0.95, risk_free=0.02)
    r_p = PA.portfolio_returns(prices.pct_change().iloc[1:], w)
    assert rep.observations == len(prices) - 1
    assert rep.var_historical == pytest.approx(risk.historical_var(r_p, 0.95))
    assert rep.cvar_historical == pytest.approx(risk.historical_cvar(r_p, 0.95))
    assert rep.cvar_historical >= rep.var_historical
    assert rep.var_normal == pytest.approx(risk.parametric_var(r_p, 0.95))
    assert rep.volatility == pytest.approx(risk.annualized_volatility(r_p))
    assert rep.contributions["component"].sum() == pytest.approx(rep.volatility)
    assert rep.summary["sharpe"] == pytest.approx(risk.sharpe_ratio(r_p, 0.02))
    expected = performance_summary(PA.portfolio_value(prices, w).to_frame(), 0.02).iloc[0]
    assert rep.summary["total_return"] == pytest.approx(expected["total_return"])
    bench = risk.beta_alpha(r_p, prices["A0"].pct_change().iloc[1:], 0.02)
    assert rep.beta.beta == pytest.approx(bench.beta)
    assert rep.start == prices.index[0] and rep.end == prices.index[-1]


def test_single_asset_portfolio_against_itself():
    prices = random_prices(6, n_assets=2)
    rep = PA.risk_report(prices, _weights({"A0": 1.0}), benchmark="A0")
    assert rep.beta.beta == pytest.approx(1.0) and rep.beta.r_squared == pytest.approx(1.0)
    assert rep.contributions.loc["A0", "percent"] == pytest.approx(1.0)


def test_no_benchmark_and_unknown_benchmark():
    prices = random_prices(7, n_assets=2)
    w = _weights({"A0": 0.5, "A1": 0.5})
    assert PA.risk_report(prices, w).beta is None
    with pytest.raises(ValueError, match="benchmark"):
        PA.risk_report(prices, w, benchmark="SPY")


def test_small_tail_warning():
    prices = random_prices(8, n_assets=2, n_days=252)
    w = _weights({"A0": 0.5, "A1": 0.5})
    assert PA.risk_report(prices, w, confidence=0.95).warnings == []  # 12.6 in the tail
    warned = PA.risk_report(prices, w, confidence=0.99)  # 2.5 in the tail
    assert any("unreliable" in m for m in warned.warnings)


def test_portfolio_volatility_identity():
    prices = prices_from_returns(np.random.default_rng(9).normal(0, 0.01, (200, 3)))
    w = _weights({"A0": 0.2, "A1": 0.3, "A2": 0.5})
    cov = prices.pct_change().iloc[1:].cov().to_numpy()
    rep = PA.risk_report(prices, w)
    assert rep.volatility == pytest.approx(np.sqrt(w @ cov @ w * 252))
