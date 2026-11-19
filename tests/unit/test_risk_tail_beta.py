"""VaR, CVaR, beta and alpha (analytics/risk.py, Milestone 4)."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fincomb.analytics import risk
from fincomb.exceptions import InsufficientDataError


def _series(values, start="2023-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


SMALL = _series([0.02, -0.01, -0.05, 0.0, -0.03])  # sorted: -.05 -.03 -.01 0 .02


# --- historical VaR / CVaR: hand-computed ---------------------------------------------------


def test_historical_var_linear_interpolation():
    # p = 0.2, position h = (5 - 1) * 0.2 = 0.8: Q = -0.05 + 0.8 * 0.02 = -0.034
    assert risk.historical_var(SMALL, 0.8) == pytest.approx(0.034)
    assert risk.historical_var(SMALL, 0.5) == pytest.approx(0.01)  # the median


def test_historical_cvar_fractional_tail():
    # Integral of Q over [0, 0.2]: one partial trapezoid from Q(0) = -0.05 to Q(0.2) = -0.034,
    # width 0.2, average -0.042 -> CVaR = 0.042. (The tail holds 5 * 0.2 = 1 observation's
    # worth of probability, but the interpolated quantile is between two data points.)
    assert risk.historical_cvar(SMALL, 0.8) == pytest.approx(0.042)


def test_historical_cvar_whole_trapezoids():
    # p = 0.5 -> h = 2: trapezoids (-.05,-.03) and (-.03,-.01), each 0.25 wide:
    # integral = 0.25 * (-0.04 - 0.02) = -0.015 -> CVaR = 0.015 / 0.5 = 0.03
    assert risk.historical_cvar(SMALL, 0.5) == pytest.approx(0.03)


def test_historical_cvar_matches_dense_numerical_integral():
    r = _series(np.random.default_rng(7).standard_t(4, 57) * 0.01)
    x = np.sort(r.to_numpy())
    p = 0.05
    u = np.linspace(0.0, p, 400_001)
    numeric = -np.trapezoid(np.quantile(x, u), u) / p
    assert risk.historical_cvar(r, 0.95) == pytest.approx(numeric, rel=1e-8)


def test_tail_risk_converges_to_gaussian_values():
    r = _series(np.random.default_rng(0).normal(0.0, 0.01, 200_000), start="1900-01-01")
    assert risk.historical_var(r, 0.95) == pytest.approx(0.016449, abs=2e-4)
    assert risk.historical_cvar(r, 0.95) == pytest.approx(0.020627, abs=2e-4)


def test_parametric_var_and_cvar_known_values():
    # A series with mean 0 and sample std exactly 1%: z(0.05) = -1.6449,
    # pdf(z) / 0.05 = 2.0627
    base = np.array([1.0, -1.0] * 50)
    r = _series(base / base.std(ddof=1) * 0.01)
    assert risk.parametric_var(r, 0.95) == pytest.approx(0.0164485, rel=1e-5)
    assert risk.parametric_cvar(r, 0.95) == pytest.approx(0.0206271, rel=1e-5)


def test_constant_series_var_is_minus_the_return():
    r = _series([0.001] * 30)
    assert risk.historical_var(r, 0.95) == pytest.approx(-0.001)
    assert risk.historical_cvar(r, 0.95) == pytest.approx(-0.001)


def test_tail_observations_and_bad_confidence():
    r = _series(np.zeros(252) + 0.001)
    assert risk.tail_observations(r, 0.99) == pytest.approx(2.52)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            risk.historical_var(r, bad)


def test_tail_risk_needs_two_returns_and_a_series():
    with pytest.raises(InsufficientDataError):
        risk.historical_var(_series([0.01]))
    with pytest.raises(TypeError):
        risk.historical_cvar(pd.DataFrame({"a": [0.01, 0.02]}))


def test_nans_are_skipped():
    with_gap = _series([0.02, np.nan, -0.01, -0.05, 0.0, -0.03])
    assert risk.historical_var(with_gap, 0.8) == pytest.approx(0.034)


# --- properties ----------------------------------------------------------------------------

returns_strategy = st.lists(
    st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=2, max_size=300
).map(_series)
confidence_strategy = st.floats(min_value=0.5, max_value=0.999)


@settings(max_examples=300, deadline=None)
@given(returns_strategy, confidence_strategy)
def test_cvar_is_at_least_var(r, confidence):
    var = risk.historical_var(r, confidence)
    assert risk.historical_cvar(r, confidence) >= var - 1e-12


@settings(max_examples=150, deadline=None)
@given(returns_strategy, confidence_strategy, confidence_strategy)
def test_var_and_cvar_rise_with_confidence(r, a, b):
    lo, hi = sorted((a, b))
    assert risk.historical_var(r, hi) >= risk.historical_var(r, lo) - 1e-12
    assert risk.historical_cvar(r, hi) >= risk.historical_cvar(r, lo) - 1e-12


@settings(max_examples=150, deadline=None)
@given(returns_strategy, st.floats(min_value=0.1, max_value=4.0))
def test_var_is_positively_homogeneous(r, c):
    scaled = r * c
    assert risk.historical_var(scaled) == pytest.approx(c * risk.historical_var(r), abs=1e-12)
    assert risk.historical_cvar(scaled) == pytest.approx(c * risk.historical_cvar(r), abs=1e-12)


@settings(max_examples=150, deadline=None)
@given(returns_strategy)
def test_cvar_never_exceeds_worst_loss(r):
    assert risk.historical_cvar(r, 0.95) <= -r.min() + 1e-12


# --- beta / alpha ---------------------------------------------------------------------------


def test_exact_linear_relation_gives_beta_and_zero_alpha():
    rng = np.random.default_rng(3)
    rb = _series(rng.normal(0.0004, 0.01, 300))
    res = risk.beta_alpha(1.5 * rb, rb)
    assert res.beta == pytest.approx(1.5)
    assert res.alpha == pytest.approx(0.0, abs=1e-12)
    assert res.correlation == pytest.approx(1.0)
    assert res.r_squared == pytest.approx(1.0)
    assert res.observations == 300


def test_beta_against_itself_is_one_and_alpha_uses_excess_returns():
    rb = _series(np.random.default_rng(4).normal(0.0005, 0.01, 200))
    assert risk.beta_alpha(rb, rb).beta == pytest.approx(1.0)
    # r_p = 0.5 r_b + 0.0002 a day: alpha = 0.0002 - (1 - beta) * rf per day, annualised
    rp = 0.5 * rb + 0.0002
    rf = risk.periodic_rate(0.04)
    expected = (rp.mean() - rf - 0.5 * (rb.mean() - rf)) * 252
    res = risk.beta_alpha(rp, rb, risk_free=0.04)
    assert res.beta == pytest.approx(0.5)
    assert res.alpha == pytest.approx(expected)
    assert res.alpha == pytest.approx((0.0002 - 0.5 * rf) * 252)


def test_beta_matches_ols_slope():
    rng = np.random.default_rng(5)
    rb = _series(rng.normal(0, 0.01, 500))
    rp = 0.8 * rb + _series(rng.normal(0, 0.005, 500))
    slope, intercept = np.polyfit(rb.to_numpy(), rp.to_numpy(), 1)
    res = risk.beta_alpha(rp, rb)
    assert res.beta == pytest.approx(slope)
    assert res.alpha == pytest.approx(intercept * 252)
    assert res.r_squared == pytest.approx(np.corrcoef(rp, rb)[0, 1] ** 2)


def test_beta_uses_common_dates_only():
    rb = _series(np.random.default_rng(6).normal(0, 0.01, 100))
    rp = (2 * rb).iloc[10:]
    res = risk.beta_alpha(rp, rb)
    assert res.observations == 90
    assert res.beta == pytest.approx(2.0)


def test_zero_variance_benchmark_gives_nan():
    rb = _series([0.001] * 50)
    rp = _series(np.random.default_rng(8).normal(0, 0.01, 50))
    res = risk.beta_alpha(rp, rb)
    assert math.isnan(res.beta) and math.isnan(res.alpha) and math.isnan(res.correlation)
