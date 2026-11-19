"""Portfolio construction: equal weight, inverse volatility, minimum variance and
maximum Sharpe, all long-only and fully invested with per-asset bounds.

`construct` is the one entry point the services (and later the backtester) use: it
takes a window of prices, estimates what it needs, solves, verifies, and falls back
to bounded inverse volatility (with a warning) if an optimiser can't produce a
trustworthy answer. Pure functions: no database, no HTTP, no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from fincomb.analytics._utils import check_prices
from fincomb.analytics.constants import TRADING_DAYS
from fincomb.analytics.returns import simple_returns
from fincomb.analytics.risk import annualized_volatility, periodic_rate, sharpe_ratio
from fincomb.exceptions import InsufficientDataError, MissingDataError
from fincomb.portfolio.covariance import (
    COVARIANCE_METHODS,
    CovarianceMethod,
    estimate_covariance,
    make_positive_definite,
)
from fincomb.portfolio.optimizer import solve

Method = Literal["equal_weight", "inverse_vol", "min_variance", "max_sharpe"]
METHODS: tuple[str, ...] = ("equal_weight", "inverse_vol", "min_variance", "max_sharpe")

MIN_OBSERVATIONS = 60  # absolute minimum number of returns in the estimation window
_FEAS_TOL = 1e-12
_HEAVY = 1e6  # relative score that puts one asset at its upper bound in a starting point


@dataclass(frozen=True)
class WeightBounds:
    """Per-asset bounds min_weight <= w_i <= max_weight (long-only: 0 <= min <= max <= 1)."""

    min_weight: float = 0.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_weight <= self.max_weight <= 1.0):
            raise ValueError(
                "weight bounds must satisfy 0 <= min_weight <= max_weight <= 1 "
                f"(got min_weight={self.min_weight}, max_weight={self.max_weight})"
            )

    def check_feasible(self, n: int) -> None:
        """Raise ValueError unless n weights inside the bounds can sum to 1."""
        if n < 1:
            raise ValueError("need at least one asset")
        if n * self.max_weight < 1.0 - _FEAS_TOL:
            raise ValueError(
                f"max_weight {self.max_weight:g} is too small for {n} assets: even with every "
                f"asset at the cap the weights sum to {n * self.max_weight:.0%}. "
                f"Use max_weight >= {1 / n:.4f}."
            )
        if n * self.min_weight > 1.0 + _FEAS_TOL:
            raise ValueError(
                f"min_weight {self.min_weight:g} is too large for {n} assets: the minimums "
                f"alone sum to {n * self.min_weight:.0%}. Use min_weight <= {1 / n:.4f}."
            )


def min_observations(n_assets: int) -> int:
    """Returns needed in the window: max(60, 2N). Fewer makes the covariance unreliable."""
    return max(MIN_OBSERVATIONS, 2 * n_assets)


def bounded_proportional(raw: np.ndarray, bounds: WeightBounds) -> np.ndarray:
    """Weights proportional to `raw` (all > 0), clipped to the bounds and summing to 1.

    w_i = clip(c * raw_i, min_weight, max_weight) with the one scalar c that makes them
    sum to 1. The sum is continuous and non-decreasing in c, from n*min_weight (c = 0)
    to n*max_weight (c large), so a solution exists whenever the bounds are feasible.
    c is bracketed by bisection, then solved exactly for the assets not pinned at a
    bound; those keep their raw proportions.
    """
    raw = np.asarray(raw, dtype="float64")
    if raw.ndim != 1 or len(raw) == 0 or not np.all(np.isfinite(raw)) or np.any(raw <= 0):
        raise ValueError("raw scores must be finite and positive")
    n = len(raw)
    lo, hi = bounds.min_weight, bounds.max_weight
    bounds.check_feasible(n)
    if hi - lo <= _FEAS_TOL:  # min == max: the only feasible portfolio is 1/n each
        return np.full(n, 1.0 / n)

    def total(c: float) -> float:
        return float(np.clip(c * raw, lo, hi).sum())

    c_lo, c_hi = 0.0, hi / raw.min()
    for _ in range(200):
        mid = 0.5 * (c_lo + c_hi)
        if total(mid) < 1.0:
            c_lo = mid
        else:
            c_hi = mid
        if c_hi - c_lo <= 1e-16 * max(c_hi, 1.0):
            break
    c = 0.5 * (c_lo + c_hi)
    w = np.clip(c * raw, lo, hi)
    free = (c * raw > lo) & (c * raw < hi)
    if free.any():
        pinned = w[~free].sum()
        c_exact = (1.0 - pinned) / raw[free].sum()
        candidate = w.copy()
        candidate[free] = c_exact * raw[free]
        if np.all(candidate >= lo - 1e-12) and np.all(candidate <= hi + 1e-12):
            w = np.clip(candidate, lo, hi)
    return w


def equal_weight(symbols: list[str] | pd.Index, bounds: WeightBounds | None = None) -> pd.Series:
    """1/N for every asset. Always inside any feasible bounds."""
    n = len(symbols)
    (bounds or WeightBounds()).check_feasible(n)
    return pd.Series(1.0 / n, index=pd.Index(symbols, name="symbol"), name="weight")


def inverse_volatility(returns: pd.DataFrame, bounds: WeightBounds | None = None) -> pd.Series:
    """Weights proportional to 1 / volatility, clipped to both bounds (see
    `bounded_proportional`). An asset with zero volatility is rejected."""
    vols = returns.std(ddof=1)
    flat = [str(s) for s, v in vols.items() if not np.isfinite(v) or v <= 1e-12]
    if flat:
        raise ValueError(
            f"{', '.join(flat)} had zero volatility in the window (the price never moved), "
            "so inverse-volatility weights are undefined"
        )
    w = bounded_proportional(1.0 / vols.to_numpy(), bounds or WeightBounds())
    return pd.Series(w, index=pd.Index(returns.columns, name="symbol"), name="weight")


@dataclass(frozen=True)
class OptimizerInfo:
    success: bool
    message: str
    iterations: int


@dataclass
class ConstructionResult:
    """Weights plus everything needed to understand and reproduce them."""

    weights: pd.Series
    method: str  # what was asked for
    effective_method: str  # what produced the weights ("inverse_vol" after a fallback)
    start: pd.Timestamp  # first price date of the estimation window
    end: pd.Timestamp  # last price date
    observations: int  # daily returns used
    bounds: WeightBounds
    covariance: str | None  # covariance method (optimisers only)
    shrinkage: float | None  # Ledoit-Wolf intensity delta
    ridge: float  # ridge added to make the covariance positive definite (0 = none)
    risk_free: float
    # In-sample (flattering) statistics of the constant-mix portfolio over the window:
    expected_return: float  # annualised arithmetic mean
    volatility: float
    sharpe: float
    optimizer: OptimizerInfo | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None
    non_positive_excess_return: bool = False
    warnings: list[str] = field(default_factory=list)


def _window_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("construct takes a prices DataFrame (one column per asset)")
    check_prices(prices)
    if prices.columns.has_duplicates:
        raise ValueError("each asset may appear only once")
    returns = simple_returns(prices)
    if returns.isna().to_numpy().any():
        cols = [str(c) for c in returns.columns[returns.isna().any()]]
        raise MissingDataError(
            f"the estimation window has missing prices for {', '.join(cols)}; "
            "use dates every asset traded (inner alignment)"
        )
    return returns


def _variance_problem(cov: np.ndarray):
    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov @ w

    return objective, gradient


def _sharpe_problem(excess: np.ndarray, cov: np.ndarray):
    """Minus the Sharpe ratio w'e / sqrt(w'Cw) and its gradient (quotient rule)."""

    def objective(w: np.ndarray) -> float:
        vol = np.sqrt(max(float(w @ cov @ w), 1e-300))
        return -float(w @ excess) / vol

    def gradient(w: np.ndarray) -> np.ndarray:
        var = max(float(w @ cov @ w), 1e-300)
        vol = np.sqrt(var)
        return -(excess / vol - float(w @ excess) * (cov @ w) / (var * vol))

    return objective, gradient


def _heavy_starts(n: int, bounds: WeightBounds) -> list[np.ndarray]:
    """One feasible point per asset with that asset as large as the bounds allow."""
    starts = []
    for i in range(n):
        raw = np.ones(n)
        raw[i] = _HEAVY
        starts.append(bounded_proportional(raw, bounds))
    return starts


def construct(
    prices: pd.DataFrame,
    method: Method,
    bounds: WeightBounds | None = None,
    covariance: CovarianceMethod = "ledoit_wolf",
    risk_free: float = 0.0,
    min_obs: int | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> ConstructionResult:
    """Weights for the assets in `prices` (rows = dates, inner-aligned, no gaps) by `method`.

    The optimisers (min_variance, max_sharpe) use the `covariance` estimate (annualised),
    made positive definite with a tiny ridge if needed. If the covariance can't be
    repaired, SLSQP fails, or its answer fails the checks, the weights fall back to
    bounded inverse volatility; `fallback_used`, `effective_method` and `warnings`
    say so. Max Sharpe never switches method when no asset beats the risk-free rate:
    it returns the (unattractive) answer with `non_positive_excess_return` and a warning.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if covariance not in COVARIANCE_METHODS:
        raise ValueError(f"covariance must be one of {COVARIANCE_METHODS}")
    bounds = bounds or WeightBounds()
    returns = _window_returns(prices)
    symbols = returns.columns
    n = len(symbols)
    bounds.check_feasible(n)
    needed = min_obs if min_obs is not None else min_observations(n)
    if len(returns) < needed:
        raise InsufficientDataError(
            f"need at least {needed} daily returns for {n} assets, the window has "
            f"{len(returns)}; use a longer lookback or an earlier start"
        )

    warnings: list[str] = []
    info: OptimizerInfo | None = None
    fallback_reason: str | None = None
    shrinkage: float | None = None
    ridge = 0.0
    non_positive = False
    cov_method: str | None = None
    w: np.ndarray | None = None

    if method == "equal_weight":
        w = equal_weight(symbols, bounds).to_numpy()
    elif method == "inverse_vol":
        w = inverse_volatility(returns, bounds).to_numpy()
    else:
        cov_method = covariance
        cov_df, shrinkage = estimate_covariance(returns, covariance)
        repaired = make_positive_definite(cov_df * periods_per_year)
        if not repaired.ok:
            fallback_reason = "the covariance matrix is not positive definite even after a ridge"
        else:
            ridge = repaired.ridge
            if ridge > 0:
                warnings.append(
                    f"The covariance matrix was singular (duplicate, collinear or flat assets); "
                    f"a ridge of {ridge:g} x the average variance was added so it could be "
                    "optimised."
                )
            # Divide by the average variance so the solver sees numbers of order 1 whatever
            # the asset volatilities: SLSQP's tolerance is absolute. The optimal weights
            # don't change (variance scales, the Sharpe ratio only gains a constant factor).
            cov = repaired.cov.to_numpy()
            cov = cov / (np.trace(cov) / n)
            ew = np.full(n, 1.0 / n)
            iv = bounded_proportional(1.0 / np.sqrt(np.diag(cov)), bounds)
            if method == "min_variance":
                objective, gradient = _variance_problem(cov)
                candidates = [ew, iv]
                starts = [ew]
            else:
                rf = periodic_rate(risk_free, periods_per_year)
                excess = (returns.mean().to_numpy() - rf) * periods_per_year
                non_positive = bool(np.all(excess <= 0))
                objective, gradient = _sharpe_problem(excess, cov)
                heavy = _heavy_starts(n, bounds)
                candidates = [ew, iv, *heavy]
                # The Sharpe ratio is not convex, and with no positive excess return it is
                # not even quasi-concave, and SLSQP's line search can stall from a single
                # start. Several starts (equal weight, inverse vol, one corner per asset)
                # cost little for tens of assets and make the answer robust.
                starts = [ew, iv, *heavy]
            solved = solve(objective, gradient, starts, bounds.min_weight, bounds.max_weight)
            info = OptimizerInfo(solved.success, solved.message, solved.iterations)
            if solved.weights is None:
                fallback_reason = f"SLSQP did not converge: {solved.message}"
            else:
                best_candidate = min(objective(c) for c in candidates)
                if solved.objective > best_candidate + 1e-9 * max(1.0, abs(best_candidate)):
                    fallback_reason = (
                        "SLSQP stopped at a worse point than a simple feasible portfolio"
                    )
                else:
                    w = solved.weights
        if fallback_reason is not None:
            w = inverse_volatility(returns, bounds).to_numpy()
            warnings.append(
                f"{method} could not be solved reliably ({fallback_reason}); the weights are "
                "bounded inverse volatility instead."
            )
        if non_positive:
            warnings.append(
                f"No asset had a positive estimated excess return over the risk-free rate "
                f"({risk_free:.2%} a year) in the estimation window, so every long-only "
                "portfolio has a negative expected Sharpe ratio. Maximising a negative Sharpe "
                "ratio favours HIGHER volatility, so these weights can look unintuitive: they "
                "are the mathematical answer to the question asked, not an attractive "
                "investment. Consider min_variance, another window or a lower risk-free rate."
            )

    assert w is not None
    weights = pd.Series(w, index=pd.Index([str(s) for s in symbols], name="symbol"), name="weight")
    r_p = returns @ weights.to_numpy()
    return ConstructionResult(
        weights=weights,
        method=method,
        effective_method="inverse_vol" if fallback_reason else method,
        start=prices.index[0],
        end=prices.index[-1],
        observations=len(returns),
        bounds=bounds,
        covariance=cov_method,
        shrinkage=shrinkage,
        ridge=ridge,
        risk_free=risk_free,
        expected_return=float(r_p.mean() * periods_per_year),
        volatility=float(annualized_volatility(r_p, periods_per_year)),
        sharpe=float(sharpe_ratio(r_p, risk_free, periods_per_year)),
        optimizer=info,
        fallback_used=fallback_reason is not None,
        fallback_reason=fallback_reason,
        non_positive_excess_return=non_positive,
        warnings=warnings,
    )
