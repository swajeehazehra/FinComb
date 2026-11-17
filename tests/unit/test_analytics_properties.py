"""Property-based tests: invariants that must hold for ANY valid price series."""

import math

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fincomb.analytics import returns as R
from fincomb.analytics import risk

# Realistic daily moves (-20% to +20%) compounded from a random starting price.
prices_strategy = st.tuples(
    st.floats(min_value=1, max_value=5_000),
    st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False), min_size=20, max_size=300),
).map(
    lambda t: pd.Series(
        t[0] * np.cumprod([1.0, *[1 + r for r in t[1]]]),
        index=pd.bdate_range("2020-01-01", periods=len(t[1]) + 1),
    )
)


@settings(max_examples=150, deadline=None)
@given(prices_strategy)
def test_total_return_equals_last_over_first(prices):
    r = R.simple_returns(prices)
    assert math.isclose(
        R.total_return(r), prices.iloc[-1] / prices.iloc[0] - 1, rel_tol=1e-9, abs_tol=1e-9
    )


@settings(max_examples=150, deadline=None)
@given(prices_strategy)
def test_log_returns_sum_to_log_price_ratio(prices):
    assert math.isclose(
        R.log_returns(prices).sum(), math.log(prices.iloc[-1] / prices.iloc[0]), abs_tol=1e-9
    )


@settings(max_examples=150, deadline=None)
@given(prices_strategy)
def test_drawdown_bounds(prices):
    r = R.simple_returns(prices)
    dd = risk.drawdown_series(r)
    assert (dd <= 1e-12).all() and (dd >= -1).all()
    mdd = risk.max_drawdown(r)
    # The worst drawdown can never be smaller than the loss from the highest to the lowest
    # later price, nor worse than from the overall max to the overall min.
    assert -1 <= mdd <= 0
    assert mdd >= prices.min() / max(prices.max(), prices.iloc[0]) - 1 - 1e-9


@settings(max_examples=100, deadline=None)
@given(prices_strategy, st.floats(min_value=0.01, max_value=1000))
def test_scale_invariance(prices, k):
    """Stats depend on returns only, so multiplying all prices by k changes nothing."""
    a, b = R.simple_returns(prices), R.simple_returns(prices * k)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12)
    assert math.isclose(risk.max_drawdown(a), risk.max_drawdown(b), abs_tol=1e-9)


@settings(max_examples=100, deadline=None)
@given(prices_strategy)
def test_cagr_has_same_sign_as_total_return(prices):
    r = R.simple_returns(prices)
    total, growth = R.total_return(r), R.cagr(r)
    assert np.sign(round(total, 12)) == np.sign(round(growth, 12))
