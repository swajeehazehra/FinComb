"""Portfolio risk analysis for a fixed set of weights.

The portfolio is "constant mix": weights are reset every day, so the daily portfolio
return is r_p = sum_i w_i r_i. That is the standard convention for ex-ante risk and
makes the identities exact (for example std(r_p) = sqrt(w' S w) with S the sample
covariance). Drift between rebalances is modelled by the backtester, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fincomb.analytics._utils import check_prices
from fincomb.analytics.constants import TRADING_DAYS
from fincomb.analytics.returns import simple_returns
from fincomb.analytics.risk import _ZERO as ZERO_STD  # one zero tolerance everywhere
from fincomb.analytics.risk import (
    BetaResult,
    beta_alpha,
    historical_cvar,
    historical_var,
    parametric_cvar,
    parametric_var,
    tail_observations,
)
from fincomb.analytics.summary import performance_summary
from fincomb.exceptions import MissingDataError
from fincomb.portfolio.covariance import sample_covariance

MIN_TAIL_OBSERVATIONS = 10  # below this the historical VaR/CVaR estimate gets a warning
WEIGHT_SUM_TOL = 1e-6


def check_weights(weights: pd.Series) -> pd.Series:
    """Long-only, fully invested weights: finite, 0 <= w <= 1, unique assets, sum 1 +/- 1e-6."""
    if not isinstance(weights, pd.Series) or weights.empty:
        raise ValueError("weights must be a non-empty Series indexed by symbol")
    if weights.index.has_duplicates:
        raise ValueError("each asset may appear only once in the weights")
    values = weights.to_numpy(dtype="float64")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite numbers")
    bad = [f"{s}={w:g}" for s, w in weights.items() if w < 0 or w > 1]
    if bad:
        raise ValueError(f"weights must be between 0 and 1 (long-only): {', '.join(bad)}")
    total = float(values.sum())
    if abs(total - 1.0) > WEIGHT_SUM_TOL:
        raise ValueError(f"weights must sum to 1, they sum to {total:.6f}")
    return weights.astype("float64")


def _asset_returns(prices: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    missing = [s for s in weights.index if s not in prices.columns]
    if missing:
        raise ValueError(f"no prices for {', '.join(map(str, missing))}")
    returns = simple_returns(prices[list(weights.index)])
    if returns.isna().to_numpy().any():
        cols = [str(c) for c in returns.columns[returns.isna().any()]]
        raise MissingDataError(f"missing prices for {', '.join(cols)} inside the period")
    return returns


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily constant-mix portfolio returns sum_i w_i r_i (no gaps allowed)."""
    weights = check_weights(weights)
    missing = [s for s in weights.index if s not in returns.columns]
    if missing:
        raise ValueError(f"no returns for {', '.join(map(str, missing))}")
    sub = returns[list(weights.index)]
    if sub.isna().to_numpy().any():
        raise MissingDataError("asset returns have gaps; align the assets first")
    return (sub @ weights.to_numpy()).rename("portfolio")


def portfolio_value(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Value of 1 invested at the first price date, rebalanced daily to `weights`."""
    check_prices(prices)
    weights = check_weights(weights)
    r_p = portfolio_returns(_asset_returns(prices, weights), weights)
    value = pd.concat([pd.Series([1.0], index=prices.index[:1]), (1.0 + r_p).cumprod()])
    return value.rename("portfolio")


def risk_contributions(
    weights: pd.Series, cov: pd.DataFrame, periods_per_year: int = TRADING_DAYS
) -> pd.DataFrame:
    """Each asset's share of portfolio volatility (Euler decomposition), annualised.

    marginal_i  = (S w)_i / sigma_p      extra volatility per unit of weight
    component_i = w_i * marginal_i       sums exactly to sigma_p
    percent_i   = component_i / sigma_p  sums to 1; negative = the asset reduces risk

    `cov` is the per-period (daily) covariance. When the portfolio's per-period
    volatility is at or below the analytics zero tolerance (1e-12, the same one the
    Sharpe ratio uses), the decomposition is undefined: marginal, component and
    percent are NaN rather than ratios of rounding noise. A covariance with NaN or
    infinite entries is rejected.
    """
    weights = check_weights(weights)
    daily = cov.loc[weights.index, weights.index].to_numpy(dtype="float64")
    if not np.all(np.isfinite(daily)):
        raise ValueError("the covariance matrix contains NaN or infinite values")
    w = weights.to_numpy()
    # Compare the DAILY volatility with the tolerance, then annualise: the tolerance is
    # defined per period, like the standard deviations it guards elsewhere.
    daily_vol = np.sqrt(max(float(w @ daily @ w), 0.0))
    if daily_vol <= ZERO_STD:
        undefined = np.full(len(w), np.nan)
        marginal = component = percent = undefined
    else:
        s = daily * periods_per_year
        vol = daily_vol * np.sqrt(periods_per_year)
        marginal = (s @ w) / vol
        component = w * marginal
        percent = component / vol
    return pd.DataFrame(
        {"weight": w, "marginal": marginal, "component": component, "percent": percent},
        index=pd.Index(weights.index, name="symbol"),
    )


@dataclass
class RiskReport:
    start: pd.Timestamp
    end: pd.Timestamp
    observations: int
    confidence: float
    risk_free: float
    summary: dict  # performance_summary row for the portfolio value series
    volatility: float  # annualised
    var_historical: float  # 1-day, positive = loss
    cvar_historical: float
    var_normal: float
    cvar_normal: float
    tail_observations: float
    benchmark: str | None
    beta: BetaResult | None
    contributions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def risk_report(
    prices: pd.DataFrame,
    weights: pd.Series,
    benchmark: str | None = None,
    confidence: float = 0.95,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> RiskReport:
    """Risk of `weights` over the dates in `prices` (inner-aligned; may include the
    benchmark column, which can also be a portfolio member)."""
    weights = check_weights(weights)
    returns = _asset_returns(prices, weights)
    r_p = portfolio_returns(returns, weights)
    value = portfolio_value(prices, weights)
    summary = performance_summary(value.to_frame(), risk_free, periods_per_year).iloc[0].to_dict()

    beta = None
    if benchmark is not None:
        if benchmark not in prices.columns:
            raise ValueError(f"no prices for benchmark {benchmark}")
        bench_r = simple_returns(prices[benchmark])
        beta = beta_alpha(r_p, bench_r, risk_free, periods_per_year)

    warnings: list[str] = []
    contributions = risk_contributions(weights, sample_covariance(returns), periods_per_year)
    if contributions["component"].isna().all():
        warnings.append(
            "Risk contributions are undefined because the portfolio's volatility is "
            "effectively zero (daily standard deviation at or below 1e-12): there is no "
            "risk to divide between the assets."
        )
    tail = tail_observations(r_p, confidence)
    if tail < MIN_TAIL_OBSERVATIONS:
        warnings.append(
            f"Only {tail:.1f} daily returns fall in the {1 - confidence:.1%} tail, so the "
            "historical VaR and CVaR rest on very few observations and are unreliable. "
            "Use a longer period or a lower confidence level."
        )

    return RiskReport(
        start=prices.index[0],
        end=prices.index[-1],
        observations=len(r_p),
        confidence=confidence,
        risk_free=risk_free,
        summary=summary,
        volatility=float(summary["annual_volatility"]),
        var_historical=historical_var(r_p, confidence),
        cvar_historical=historical_cvar(r_p, confidence),
        var_normal=parametric_var(r_p, confidence),
        cvar_normal=parametric_cvar(r_p, confidence),
        tail_observations=tail,
        benchmark=benchmark,
        beta=beta,
        contributions=contributions,
        warnings=warnings,
    )
