"""Portfolio construction (portfolio/construction.py, portfolio/optimizer.py).

Optimiser answers are checked against closed-form solutions, optimality (KKT)
conditions and an independent convex reformulation, not just against themselves.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from fincomb.analytics.risk import annualized_volatility, periodic_rate, sharpe_ratio
from fincomb.exceptions import InsufficientDataError, MissingDataError
from fincomb.portfolio import construction as PC
from fincomb.portfolio.construction import WeightBounds, bounded_proportional, construct
from fincomb.portfolio.optimizer import SolveResult


def prices_from_returns(returns: np.ndarray, columns=None, start="2020-01-02") -> pd.DataFrame:
    returns = np.asarray(returns, dtype="float64")
    n, k = returns.shape
    columns = columns or [f"A{i}" for i in range(k)]
    values = 100.0 * np.vstack([np.ones(k), np.cumprod(1.0 + returns, axis=0)])
    return pd.DataFrame(values, index=pd.bdate_range(start, periods=n + 1), columns=columns)


def random_prices(seed: int, n_assets: int = 4, n_days: int = 300, mean=0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vols = rng.uniform(0.005, 0.02, n_assets)
    corr = 0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets)
    cov = np.outer(vols, vols) * corr
    means = rng.uniform(-0.0002, 0.001, n_assets) if mean is None else np.full(n_assets, mean)
    return prices_from_returns(rng.multivariate_normal(means, cov, n_days))


def returns_of(prices):
    return prices.pct_change().iloc[1:]


def sample_cov(prices):
    return np.cov(returns_of(prices).T, ddof=1)


def port_var(w, prices):
    return float(w @ sample_cov(prices) @ w)


# --- bounds -------------------------------------------------------------------------------


def test_bounds_validation_and_feasibility():
    with pytest.raises(ValueError):
        WeightBounds(0.5, 0.4)
    with pytest.raises(ValueError):
        WeightBounds(-0.1, 1.0)
    with pytest.raises(ValueError, match="too small for 3 assets"):
        WeightBounds(0.0, 0.3).check_feasible(3)
    with pytest.raises(ValueError, match="too large for 4 assets"):
        WeightBounds(0.3, 1.0).check_feasible(4)
    WeightBounds(0.25, 0.25).check_feasible(4)  # exactly feasible


# --- bounded inverse-vol scaling (clarification C1) ---------------------------------------


def _check_bounded(w, raw, bounds):
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(w >= bounds.min_weight - 1e-12) and np.all(w <= bounds.max_weight + 1e-12)
    free = (w > bounds.min_weight + 1e-9) & (w < bounds.max_weight - 1e-9)
    if free.sum() >= 2:  # free assets keep their raw proportions
        ratios = w[free] / raw[free]
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)


def test_bounded_proportional_unbounded_is_plain_normalisation():
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(bounded_proportional(raw, WeightBounds()), raw / raw.sum())


def test_bounded_proportional_max_only():
    raw = np.array([10.0, 1.0, 1.0, 1.0])  # raw share of the first: 77%
    b = WeightBounds(0.0, 0.4)
    w = bounded_proportional(raw, b)
    np.testing.assert_allclose(w, [0.4, 0.2, 0.2, 0.2])
    _check_bounded(w, raw, b)


def test_bounded_proportional_min_only():
    raw = np.array([100.0, 1.0, 1.0, 50.0])
    b = WeightBounds(0.1, 1.0)
    w = bounded_proportional(raw, b)
    # Two smallest pinned at 0.1; the rest 0.8 split 100:50.
    np.testing.assert_allclose(w, [0.8 * 2 / 3, 0.1, 0.1, 0.8 / 3])
    _check_bounded(w, raw, b)


def test_bounded_proportional_both_bounds_bind():
    raw = np.array([100.0, 30.0, 20.0, 1.0, 0.5])
    b = WeightBounds(0.1, 0.35)
    w = bounded_proportional(raw, b)
    assert w[0] == pytest.approx(0.35)
    assert w[3] == pytest.approx(0.1) and w[4] == pytest.approx(0.1)
    np.testing.assert_allclose(w[1:3], np.array([30.0, 20.0]) / 50.0 * 0.45)
    _check_bounded(w, raw, b)


def test_bounded_proportional_min_equals_max():
    w = bounded_proportional(np.array([5.0, 1.0, 2.0, 9.0]), WeightBounds(0.25, 0.25))
    np.testing.assert_allclose(w, 0.25)


def test_bounded_proportional_infeasible_and_bad_raw():
    with pytest.raises(ValueError):
        bounded_proportional(np.array([1.0, 2.0, 3.0]), WeightBounds(0.0, 0.3))
    with pytest.raises(ValueError):
        bounded_proportional(np.array([1.0, 0.0]), WeightBounds())


@settings(max_examples=300, deadline=None)
@given(
    st.lists(st.floats(min_value=0.01, max_value=100.0), min_size=1, max_size=20),
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_bounded_proportional_always_feasible(raw, a, b):
    raw = np.array(raw)
    n = len(raw)
    lo, hi = sorted((a, b))
    lo = min(lo, 1.0 / n)  # make the bounds feasible
    hi = max(hi, 1.0 / n)
    bounds = WeightBounds(lo, hi)
    _check_bounded(bounded_proportional(raw, bounds), raw, bounds)


# --- equal weight and inverse volatility -----------------------------------------------------


def test_equal_weight():
    res = construct(random_prices(0), "equal_weight")
    np.testing.assert_allclose(res.weights, 0.25)
    assert res.effective_method == "equal_weight" and res.covariance is None


def test_inverse_vol_two_to_one():
    pattern = np.array([1.0, -1.0] * 50)
    r = np.column_stack([0.01 * pattern, 0.02 * pattern])
    res = construct(prices_from_returns(r, ["LOW", "HIGH"]), "inverse_vol")
    assert res.weights["LOW"] == pytest.approx(2 / 3)
    assert res.weights["HIGH"] == pytest.approx(1 / 3)


def test_inverse_vol_respects_both_bounds():
    rng = np.random.default_rng(2)
    vols = np.array([0.002, 0.01, 0.012, 0.03, 0.05])
    prices = prices_from_returns(rng.normal(0, 1, (300, 5)) * vols)
    b = WeightBounds(0.08, 0.4)
    res = construct(prices, "inverse_vol", b)
    w = res.weights.to_numpy()
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert w.max() <= 0.4 + 1e-12 and w.min() >= 0.08 - 1e-12
    assert w[0] == pytest.approx(0.4)  # the lowest-vol asset hits the cap
    assert w[4] == pytest.approx(0.08)  # the highest-vol asset is lifted to the floor


def test_inverse_vol_equal_risk_contributions_when_uncorrelated():
    # Exactly uncorrelated columns (orthogonal +-1 patterns) with different scales.
    base = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]] * 30, dtype=float)
    r = base * np.array([0.005, 0.01, 0.03])
    prices = prices_from_returns(r)
    res = construct(prices, "inverse_vol")
    w = res.weights.to_numpy()
    cov = sample_cov(prices)
    contrib = w * (cov @ w)
    np.testing.assert_allclose(contrib, contrib.mean(), rtol=1e-6)


def test_inverse_vol_rejects_flat_asset():
    rng = np.random.default_rng(3)
    r = np.column_stack([rng.normal(0, 0.01, 100), np.zeros(100)])
    with pytest.raises(ValueError, match="zero volatility"):
        construct(prices_from_returns(r, ["A", "FLAT"]), "inverse_vol")


# --- minimum variance ------------------------------------------------------------------------


def test_min_variance_two_asset_closed_form():
    prices = random_prices(11, n_assets=2)
    s = sample_cov(prices)
    expected = (s[1, 1] - s[0, 1]) / (s[0, 0] + s[1, 1] - 2 * s[0, 1])
    assert 0 < expected < 1  # interior: the bounds don't bind
    res = construct(prices, "min_variance", covariance="sample")
    assert res.weights.iloc[0] == pytest.approx(expected, abs=1e-6)


def similar_vol_prices(seed: int, n_assets: int = 5) -> pd.DataFrame:
    """Similar volatilities, mild correlation: the unconstrained solutions are interior."""
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.01, 0.014, n_assets)
    cov = np.outer(vols, vols) * (0.2 + 0.8 * np.eye(n_assets))
    return prices_from_returns(rng.multivariate_normal(np.zeros(n_assets), cov, 400))


def test_min_variance_n_asset_closed_form():
    prices = similar_vol_prices(12)
    s = sample_cov(prices)
    inv1 = np.linalg.solve(s, np.ones(5))
    expected = inv1 / inv1.sum()
    assert np.all(expected > 0)
    res = construct(prices, "min_variance", covariance="sample")
    np.testing.assert_allclose(res.weights, expected, atol=1e-6)
    assert res.optimizer is not None and res.optimizer.success
    assert res.shrinkage is None and res.covariance == "sample"


def test_min_variance_kkt_when_bounds_bind():
    rng = np.random.default_rng(13)
    vols = np.array([0.004, 0.006, 0.012, 0.02, 0.025])
    prices = prices_from_returns(rng.normal(0, 1, (400, 5)) * vols)
    b = WeightBounds(0.0, 0.4)
    res = construct(prices, "min_variance", b, covariance="sample")
    w = res.weights.to_numpy()
    mv = sample_cov(prices) @ w  # marginal variance (up to the factor 2)
    free = (w > 1e-7) & (w < 0.4 - 1e-7)
    capped = w >= 0.4 - 1e-7
    zero = w <= 1e-7
    assert capped.any() and free.sum() >= 2
    level = mv[free].mean()
    np.testing.assert_allclose(mv[free], level, rtol=1e-4)
    assert np.all(mv[capped] <= level * (1 + 1e-4))
    assert np.all(mv[zero] >= level * (1 - 1e-4))


def test_min_variance_ledoit_wolf_reports_shrinkage():
    res = construct(random_prices(14), "min_variance")
    assert res.covariance == "ledoit_wolf"
    assert 0.0 <= res.shrinkage <= 1.0
    assert res.warnings == []


def test_duplicate_asset_gets_ridge_and_equal_split():
    rng = np.random.default_rng(15)
    a = rng.normal(0, 0.01, 300)
    r = np.column_stack([a, a, rng.normal(0, 0.012, 300)])
    res = construct(prices_from_returns(r, ["A", "A2", "B"]), "min_variance", covariance="sample")
    assert res.ridge > 0
    assert any("singular" in w for w in res.warnings)
    assert res.weights["A"] == pytest.approx(res.weights["A2"], abs=1e-6)
    assert not res.fallback_used


# --- maximum Sharpe --------------------------------------------------------------------------


def _excess(prices, risk_free=0.0):
    return returns_of(prices).mean().to_numpy() - periodic_rate(risk_free)


def _sharpe(w, prices, risk_free=0.0):
    s = sample_cov(prices)
    return float(w @ _excess(prices, risk_free)) / np.sqrt(w @ s @ w) * np.sqrt(252)


def test_max_sharpe_interior_closed_form():
    # Set the sample means so that the tangency weights are exactly 0.5 / 0.3 / 0.2:
    # the solution of S w = e is proportional to w* when e = c * S w*.
    rng = np.random.default_rng(21)
    z = rng.normal(0, 0.01, (300, 3)) @ np.array([[1, 0.3, 0.2], [0, 1, 0.3], [0, 0, 1]])
    s0 = np.cov(z.T, ddof=1)
    target = np.array([0.5, 0.3, 0.2])
    r = z - z.mean(axis=0) + 0.5 * s0 @ target / (target @ s0 @ target) * 0.0008
    prices = prices_from_returns(r)
    s = sample_cov(prices)
    raw = np.linalg.solve(s, _excess(prices))
    expected = raw / raw.sum()
    np.testing.assert_allclose(expected, target, atol=1e-6)
    res = construct(prices, "max_sharpe", covariance="sample")
    np.testing.assert_allclose(res.weights, expected, atol=1e-5)
    assert not res.non_positive_excess_return


def test_max_sharpe_matches_convex_reformulation():
    # Long-only tangency: min y'Sy s.t. e'y = 1, y >= 0, then w = y / sum(y).
    prices = random_prices(22, n_assets=6, mean=None)
    s = sample_cov(prices)
    e = _excess(prices)
    ref = minimize(
        lambda y: y @ s @ y,
        np.ones(6) / max(e.max(), 1e-6),
        jac=lambda y: 2 * s @ y,
        method="SLSQP",
        bounds=[(0, None)] * 6,
        constraints=[{"type": "eq", "fun": lambda y: e @ y - 1}],
        options={"ftol": 1e-16, "maxiter": 1000},
    )
    expected = ref.x / ref.x.sum()
    res = construct(prices, "max_sharpe", covariance="sample")
    assert _sharpe(res.weights.to_numpy(), prices) == pytest.approx(
        _sharpe(expected, prices), rel=1e-6
    )
    np.testing.assert_allclose(res.weights, expected, atol=1e-4)


def test_reported_statistics_match_existing_analytics():
    prices = random_prices(23, mean=0.0005)
    res = construct(prices, "max_sharpe", covariance="sample", risk_free=0.02)
    r_p = returns_of(prices) @ res.weights.to_numpy()
    assert res.sharpe == pytest.approx(sharpe_ratio(r_p, 0.02))
    assert res.sharpe == pytest.approx(_sharpe(res.weights.to_numpy(), prices, 0.02))
    assert res.volatility == pytest.approx(annualized_volatility(r_p))
    w = res.weights.to_numpy()
    assert res.volatility == pytest.approx(np.sqrt(w @ sample_cov(prices) @ w * 252))
    assert res.expected_return == pytest.approx(r_p.mean() * 252)


def test_max_sharpe_respects_cap():
    prices = random_prices(24, n_assets=5, mean=None)
    res = construct(prices, "max_sharpe", WeightBounds(0.0, 0.3))
    assert res.weights.max() <= 0.3 + 1e-9
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-9)


def test_max_sharpe_non_positive_excess_return_warns_and_keeps_method():
    prices = random_prices(25, n_assets=4, mean=0.0001)
    res = construct(prices, "max_sharpe", risk_free=0.5)  # 50% a year: nothing beats it
    assert res.non_positive_excess_return
    assert res.method == res.effective_method == "max_sharpe"
    assert not res.fallback_used
    assert any("not an attractive investment" in w for w in res.warnings)
    assert res.sharpe < 0
    # Still the best long-only Sharpe among simple alternatives.
    for other in ("equal_weight", "inverse_vol", "min_variance"):
        alt = construct(prices, other, risk_free=0.5)
        assert res.sharpe >= alt.sharpe - 1e-6


def test_max_sharpe_with_some_positive_excess_has_no_warning():
    res = construct(random_prices(26, mean=0.0008), "max_sharpe")
    assert not res.non_positive_excess_return and res.warnings == []


# --- fallback ---------------------------------------------------------------------------------


def test_fallback_when_solver_fails(monkeypatch):
    monkeypatch.setattr(
        PC, "solve", lambda *a, **k: SolveResult(None, False, "Iteration limit reached", 500, 0.0)
    )
    prices = random_prices(31)
    b = WeightBounds(0.1, 0.4)
    res = construct(prices, "min_variance", b)
    expected = construct(prices, "inverse_vol", b).weights
    pd.testing.assert_series_equal(res.weights, expected)
    assert res.fallback_used and res.method == "min_variance"
    assert res.effective_method == "inverse_vol"
    assert "Iteration limit reached" in res.fallback_reason
    assert any("bounded inverse volatility" in w for w in res.warnings)


def test_fallback_when_scipy_reports_failure(monkeypatch):
    import fincomb.portfolio.optimizer as opt

    class Failed:
        success = False
        message = "Positive directional derivative for linesearch"
        nit = 3
        x = np.full(4, 0.25)

    monkeypatch.setattr(opt, "minimize", lambda *a, **k: Failed())
    res = construct(random_prices(32), "max_sharpe")
    assert res.fallback_used and "linesearch" in res.fallback_reason


def test_fallback_when_solution_is_worse_than_a_simple_portfolio(monkeypatch):
    # A "successful" answer that is worse than equal weight must not be trusted.
    monkeypatch.setattr(
        PC,
        "solve",
        lambda obj, grad, starts, lo, hi: SolveResult(
            np.array([1.0, 0, 0, 0]), True, "ok", 1, obj(np.array([1.0, 0, 0, 0]))
        ),
    )
    res = construct(random_prices(33), "min_variance")
    assert res.fallback_used and "worse point" in res.fallback_reason


def test_fallback_when_covariance_cannot_be_repaired(monkeypatch):
    from fincomb.portfolio.covariance import PDResult

    monkeypatch.setattr(PC, "make_positive_definite", lambda cov: PDResult(cov, 0.0, False))
    res = construct(random_prices(34), "max_sharpe")
    assert res.fallback_used and "positive definite" in res.fallback_reason
    assert res.optimizer is None


# --- input checks ------------------------------------------------------------------------------


def test_insufficient_history():
    prices = random_prices(41, n_assets=4, n_days=59)
    with pytest.raises(InsufficientDataError, match="at least 60"):
        construct(prices, "min_variance")
    with pytest.raises(InsufficientDataError, match="at least 80"):
        construct(random_prices(41, n_assets=40, n_days=79), "equal_weight")


def test_gaps_rejected():
    prices = random_prices(42)
    prices.iloc[10, 1] = np.nan
    with pytest.raises(MissingDataError):
        construct(prices, "equal_weight")


def test_bad_method_and_infeasible_bounds():
    prices = random_prices(43)
    with pytest.raises(ValueError, match="method"):
        construct(prices, "risk_parity")
    with pytest.raises(ValueError, match="covariance"):
        construct(prices, "min_variance", covariance="shrunk")
    with pytest.raises(ValueError, match="too small"):
        construct(prices, "equal_weight", WeightBounds(0, 0.2))


def test_window_dates_and_observations():
    prices = random_prices(44, n_days=120)
    res = construct(prices, "equal_weight")
    assert res.start == prices.index[0] and res.end == prices.index[-1]
    assert res.observations == 120


# --- invariants on random data (Hypothesis) ----------------------------------------------------

seeds = st.integers(min_value=0, max_value=10_000)


def _bounds_for(n, cap):
    return WeightBounds(0.0, max(cap, 1.0 / n))


@settings(max_examples=40, deadline=None)
@given(seeds, st.integers(min_value=2, max_value=8), st.floats(min_value=0.1, max_value=1.0))
def test_every_method_gives_feasible_weights(seed, n, cap):
    prices = random_prices(seed, n_assets=n, n_days=150, mean=None)
    b = _bounds_for(n, cap)
    for method in PC.METHODS:
        w = construct(prices, method, b).weights.to_numpy()
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-12) and np.all(w <= b.max_weight + 1e-8)


@settings(max_examples=40, deadline=None)
@given(seeds, st.integers(min_value=2, max_value=8), st.floats(min_value=0.1, max_value=1.0))
def test_min_variance_beats_the_other_methods(seed, n, cap):
    prices = random_prices(seed, n_assets=n, n_days=150, mean=None)
    b = _bounds_for(n, cap)
    mv = port_var(construct(prices, "min_variance", b, covariance="sample").weights, prices)
    for method in ("equal_weight", "inverse_vol", "max_sharpe"):
        other = construct(prices, method, b, covariance="sample").weights
        assert mv <= port_var(other.to_numpy(), prices) * (1 + 1e-7)


@settings(max_examples=40, deadline=None)
@given(seeds, st.integers(min_value=2, max_value=8))
def test_max_sharpe_beats_the_other_methods(seed, n):
    prices = random_prices(seed, n_assets=n, n_days=150, mean=None)
    res = construct(prices, "max_sharpe", covariance="sample")
    if res.non_positive_excess_return:
        return
    best = _sharpe(res.weights.to_numpy(), prices)
    for method in ("equal_weight", "inverse_vol", "min_variance"):
        other = construct(prices, method, covariance="sample").weights.to_numpy()
        assert best >= _sharpe(other, prices) - 1e-6


@settings(max_examples=30, deadline=None)
@given(seeds)
def test_reordering_assets_reorders_weights(seed):
    prices = random_prices(seed, n_assets=5, n_days=150, mean=None)
    shuffled = prices[list(reversed(prices.columns))]
    for method in ("min_variance", "max_sharpe"):
        a = construct(prices, method).weights
        b = construct(shuffled, method).weights
        pd.testing.assert_series_equal(a.sort_index(), b.sort_index(), atol=1e-6)


@settings(max_examples=30, deadline=None)
@given(seeds, st.floats(min_value=0.3, max_value=3.0))
def test_scaling_returns_leaves_weights_unchanged(seed, c):
    prices = random_prices(seed, n_assets=4, n_days=150, mean=None)
    scaled = prices_from_returns(returns_of(prices).to_numpy() * c, list(prices.columns))
    for method in ("min_variance", "max_sharpe"):
        a = construct(prices, method, covariance="sample").weights
        b = construct(scaled, method, covariance="sample").weights
        pd.testing.assert_series_equal(a, b, atol=1e-5)


@settings(max_examples=30, deadline=None)
@given(seeds)
def test_tighter_cap_never_lowers_minimum_variance(seed):
    prices = random_prices(seed, n_assets=5, n_days=150, mean=None)
    variances = [
        port_var(construct(prices, "min_variance", WeightBounds(0, cap)).weights, prices)
        for cap in (1.0, 0.5, 0.3, 0.2)
    ]
    assert all(b >= a * (1 - 1e-7) for a, b in zip(variances, variances[1:], strict=False))
