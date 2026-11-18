"""Covariance estimation and the positive-definiteness guard (portfolio/covariance.py)."""

import numpy as np
import pandas as pd
import pytest

from fincomb.exceptions import MissingDataError
from fincomb.portfolio import covariance as C


def _deterministic_returns() -> pd.DataFrame:
    t = np.arange(40)
    return pd.DataFrame(
        {
            "A": np.sin(t * 0.7) * 0.01 + 0.001 * np.cos(t * 1.3),
            "B": np.cos(t * 0.4) * 0.02 + 0.004 * np.sin(t * 2.1),
            "C": np.sin(t * 1.9 + 1) * 0.015 + 0.5 * np.sin(t * 0.7) * 0.01,
        }
    )


# Computed once with scikit-learn 1.x (sklearn.covariance.ledoit_wolf) on the data above.
# scikit-learn is not a dependency; the values are pinned here.
SKLEARN_DELTA = 0.20290322865524335
SKLEARN_SHRUNK = np.array(
    [
        [6.60624040e-05, 1.19030662e-06, 1.78716626e-05],
        [1.19030662e-06, 1.96150971e-04, 1.12659645e-05],
        [1.78716626e-05, 1.12659645e-05, 1.22601965e-04],
    ]
)


def test_sample_covariance_matches_numpy_ddof1():
    r = _deterministic_returns()
    np.testing.assert_allclose(C.sample_covariance(r).to_numpy(), np.cov(r.T, ddof=1))


def test_ledoit_wolf_matches_scikit_learn():
    r = _deterministic_returns()
    shrunk, delta = C.ledoit_wolf(r)
    assert delta == pytest.approx(SKLEARN_DELTA, rel=1e-10)
    n = len(r)
    # scikit-learn shrinks the ddof=0 matrix; ours shrinks ddof=1: a factor n / (n - 1).
    np.testing.assert_allclose(shrunk.to_numpy() * (n - 1) / n, SKLEARN_SHRUNK, rtol=1e-7)


def test_shrinkage_in_unit_interval_and_positive_definite():
    rng = np.random.default_rng(0)
    for n_rows, n_cols in [(30, 20), (252, 6), (61, 30)]:
        r = pd.DataFrame(rng.normal(0, 0.01, (n_rows, n_cols)))
        shrunk, delta = C.ledoit_wolf(r)
        assert 0.0 <= delta <= 1.0
        assert C.is_positive_definite(shrunk.to_numpy())

    # Correlated assets (one common factor): fewer rows per asset -> more shrinkage.
    def factor_returns(n_rows, n_cols):
        common = rng.normal(0, 0.01, (n_rows, 1))
        return pd.DataFrame(common + rng.normal(0, 0.01, (n_rows, n_cols)))

    few = C.ledoit_wolf_shrinkage(factor_returns(2000, 5))
    many = C.ledoit_wolf_shrinkage(factor_returns(40, 30))
    assert many > few


def test_ledoit_wolf_keeps_the_trace():
    r = _deterministic_returns()
    shrunk, _ = C.ledoit_wolf(r)
    assert np.trace(shrunk) == pytest.approx(np.trace(C.sample_covariance(r)))


def test_estimate_covariance_dispatch():
    r = _deterministic_returns()
    cov, delta = C.estimate_covariance(r, "sample")
    assert delta is None
    with pytest.raises(ValueError):
        C.estimate_covariance(r, "exotic")


def test_gaps_are_rejected():
    r = _deterministic_returns()
    r.iloc[3, 1] = np.nan
    with pytest.raises(MissingDataError, match="B"):
        C.sample_covariance(r)


def test_positive_definite_matrix_is_unchanged():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]])
    res = C.make_positive_definite(cov)
    assert res.ok and res.ridge == 0.0
    assert res.cov is cov


def test_singular_matrix_gets_smallest_ridge():
    x = np.random.default_rng(1).normal(0, 0.01, 100)
    r = pd.DataFrame({"A": x, "A2": x, "B": np.random.default_rng(2).normal(0, 0.01, 100)})
    cov = C.sample_covariance(r)
    assert not C.is_positive_definite(cov.to_numpy())
    res = C.make_positive_definite(cov)
    assert res.ok and 0 < res.ridge <= C.RIDGE_MAX
    assert C.is_positive_definite(res.cov.to_numpy())
    # The ridge is tiny relative to the variances.
    assert np.abs(res.cov.to_numpy() - cov.to_numpy()).max() < 1e-6 * np.trace(cov) / 3


def test_hopeless_matrix_is_reported():
    assert not C.make_positive_definite(pd.DataFrame(np.zeros((2, 2)))).ok
    assert not C.make_positive_definite(pd.DataFrame([[np.nan, 0], [0, 1.0]])).ok
