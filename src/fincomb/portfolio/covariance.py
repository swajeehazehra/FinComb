"""Covariance estimation for the optimisers: sample, Ledoit-Wolf shrinkage, and a
positive-definiteness guard.

All functions take a returns DataFrame (rows = dates, columns = assets) with no
missing values and return covariance matrices in the same per-period units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from fincomb.exceptions import InsufficientDataError, MissingDataError

CovarianceMethod = Literal["sample", "ledoit_wolf"]
COVARIANCE_METHODS: tuple[str, ...] = ("sample", "ledoit_wolf")

RIDGE_START = 1e-10  # first ridge tried, relative to the average variance
RIDGE_MAX = 1e-4  # largest ridge tried before giving up
PD_TOL = 1e-12  # smallest/largest eigenvalue ratio below which a matrix counts as singular


def _check(returns: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("covariance functions take a returns DataFrame (one column per asset)")
    if returns.isna().to_numpy().any():
        cols = [str(c) for c in returns.columns[returns.isna().any()]]
        raise MissingDataError(f"returns have gaps for {', '.join(cols)}; align the assets first")
    if len(returns) < 2:
        raise InsufficientDataError("need at least 2 returns to estimate a covariance")
    return returns.to_numpy(dtype="float64")


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance (ddof = 1), the same convention as `annualized_volatility`."""
    x = _check(returns)
    cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> float:
    """The Ledoit-Wolf (2004) optimal intensity for shrinking toward a scaled identity.

    With S = X'X / n (X demeaned), m = tr(S) / p:
        d^2   = ||S - m I||^2 / p                      (how far S is from the target)
        b_bar = sum_k ||x_k x_k' - S||^2 / (n^2 p)     (how noisy S is)
        delta = min(b_bar, d^2) / d^2
    (||.|| is the Frobenius norm.) Equal to scikit-learn's `ledoit_wolf_shrinkage`.
    """
    x = _check(returns)
    n, p = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / n
    m = np.trace(s) / p
    d2 = float(np.sum((s - m * np.eye(p)) ** 2)) / p
    if d2 <= 0:
        return 0.0  # S already is a scaled identity: nothing to shrink
    # sum_k ||x_k x_k' - S||^2 = sum_k ||x_k||^4 - n ||S||^2
    b_bar = (float(np.sum(np.sum(x**2, axis=1) ** 2)) - n * float(np.sum(s**2))) / (n**2 * p)
    return float(min(max(b_bar, 0.0), d2) / d2)


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Shrunk covariance delta * F + (1 - delta) * S and the intensity delta.

    S is the ddof=1 sample covariance and F = tr(S)/p * I. (Scikit-learn shrinks the
    ddof=0 matrix; the two results differ only by the factor n / (n - 1).)
    """
    delta = ledoit_wolf_shrinkage(returns)
    s = sample_covariance(returns)
    p = s.shape[0]
    target = np.trace(s.to_numpy()) / p * np.eye(p)
    shrunk = delta * target + (1.0 - delta) * s.to_numpy()
    return pd.DataFrame(shrunk, index=s.index, columns=s.columns), delta


def estimate_covariance(
    returns: pd.DataFrame, method: CovarianceMethod = "ledoit_wolf"
) -> tuple[pd.DataFrame, float | None]:
    """Covariance by `method`, plus the shrinkage intensity (None for "sample")."""
    if method == "sample":
        return sample_covariance(returns), None
    if method == "ledoit_wolf":
        return ledoit_wolf(returns)
    raise ValueError(f"covariance must be one of {COVARIANCE_METHODS}")


def is_positive_definite(cov: np.ndarray) -> bool:
    """True if the smallest eigenvalue is clearly above zero: > PD_TOL x the largest.

    A plain Cholesky test is not enough: it "succeeds" on an exactly singular matrix
    when rounding leaves its zero eigenvalue at +1e-20, and the optimiser then works
    with a matrix that is numerically singular.
    """
    cov = np.asarray(cov, dtype="float64")
    if not np.all(np.isfinite(cov)):
        return False
    eig = np.linalg.eigvalsh((cov + cov.T) / 2.0)
    return bool(eig[-1] > 0 and eig[0] > PD_TOL * eig[-1])


@dataclass(frozen=True)
class PDResult:
    cov: pd.DataFrame
    ridge: float  # ridge added, as a fraction of the average variance (0 = none)
    ok: bool  # False: still not positive definite at the largest ridge


def make_positive_definite(cov: pd.DataFrame) -> PDResult:
    """Return `cov` unchanged if it is positive definite; otherwise add the smallest ridge
    eps * (tr/p) * I, eps = 1e-10, 1e-9, ... 1e-4, that makes it so (adding eps * tr/p
    to every eigenvalue).

    Singular matrices come from duplicate or perfectly collinear assets, or from an
    asset whose price never moved.
    """
    values = cov.to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        return PDResult(cov, 0.0, False)
    if is_positive_definite(values):
        return PDResult(cov, 0.0, True)
    p = values.shape[0]
    scale = np.trace(values) / p
    if scale <= 0:
        return PDResult(cov, 0.0, False)
    eps = RIDGE_START
    while eps <= RIDGE_MAX * (1 + 1e-9):
        candidate = values + eps * scale * np.eye(p)
        if is_positive_definite(candidate):
            return PDResult(
                pd.DataFrame(candidate, index=cov.index, columns=cov.columns), eps, True
            )
        eps *= 10.0
    return PDResult(cov, 0.0, False)
