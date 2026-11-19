"""Risk and risk-adjusted performance measures.

Risk-free rates are given as ANNUAL decimals (0.04 = 4 %) and converted to a
per-period rate by compounding: (1 + rf) ** (1 / periods_per_year) - 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fincomb.analytics._utils import Frame, check_returns
from fincomb.analytics.constants import TRADING_DAYS
from fincomb.analytics.returns import Missing, cagr, wealth_index


def periodic_rate(annual_rate: float, periods_per_year: int = TRADING_DAYS) -> float:
    if annual_rate <= -1:
        raise ValueError("annual rate must be greater than -100%")
    return (1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0


_ZERO = 1e-12  # below this a standard deviation is treated as zero (float noise)
_DD_TOL = 1e-10  # drawdowns smaller than this are rounding noise


def _safe_div(num, den):
    """num / den, giving NaN instead of a huge number or inf when den is ~0."""
    if isinstance(den, pd.Series):
        return num / den.where(den.abs() > _ZERO)
    return float("nan") if np.isnan(den) or abs(den) <= _ZERO else num / den


def annualized_volatility(returns: Frame, periods_per_year: int = TRADING_DAYS):
    """Sample standard deviation (ddof=1) x sqrt(periods_per_year)."""
    returns = check_returns(returns)
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: Frame, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """Annualised Sharpe: mean(excess) / std(excess) x sqrt(periods_per_year)."""
    returns = check_returns(returns)
    excess = returns - periodic_rate(risk_free, periods_per_year)
    return _safe_div(excess.mean(), excess.std(ddof=1)) * np.sqrt(periods_per_year)


def downside_deviation(returns: Frame, target: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """sqrt(mean(min(r - target, 0)^2)) x sqrt(periods_per_year), target per period."""
    returns = check_returns(returns)
    shortfall = (returns - target).clip(upper=0.0)
    return np.sqrt((shortfall**2).mean()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: Frame, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS):
    """Annualised mean excess return / annualised downside deviation (target = risk-free)."""
    returns = check_returns(returns)
    rf = periodic_rate(risk_free, periods_per_year)
    excess_mean = (returns - rf).mean() * periods_per_year
    return _safe_div(excess_mean, downside_deviation(returns, rf, periods_per_year))


def drawdown_series(returns: Frame, missing: Missing = "raise") -> Frame:
    """Percentage below the running peak at each date (0 at a new high, negative otherwise).

    The starting capital counts as the first peak, so a loss on day one is a drawdown.
    """
    wealth = wealth_index(returns, missing=missing)
    peak = wealth.cummax().clip(lower=1.0)
    dd = wealth / peak - 1.0
    # Compounding rounding can leave a fully recovered value at -1e-16; that is a
    # recovery, not a drawdown. Snap tiny values to exactly 0 (NaN stays NaN).
    return dd.mask(dd.abs() < _DD_TOL, 0.0)


def max_drawdown(returns: Frame, missing: Missing = "raise"):
    """Worst peak-to-trough loss, as a negative number in [-1, 0]."""
    return drawdown_series(returns, missing).min()


def calmar_ratio(
    returns: Frame, start: pd.Timestamp | str | None = None, missing: Missing = "raise"
):
    """CAGR / |max drawdown|, both over the WHOLE period given (not a trailing 36 months).

    `start` is the date of the starting price, as in `cagr`.
    """
    return _safe_div(cagr(returns, start, missing), abs(max_drawdown(returns, missing)))


@dataclass(frozen=True)
class DrawdownEpisode:
    peak: pd.Timestamp | None  # last date at the old high (None = the starting capital)
    trough: pd.Timestamp
    recovery: pd.Timestamp | None  # first date back at the old high (None = not yet)
    depth: float  # negative
    days_to_trough: int  # trading days
    days_to_recover: int | None


def drawdown_episodes(
    returns: pd.Series, top: int = 5, missing: Missing = "raise"
) -> list[DrawdownEpisode]:
    """The `top` deepest separate drawdowns of one asset, deepest first."""
    if not isinstance(returns, pd.Series):
        raise TypeError("drawdown_episodes takes one asset (a Series)")
    dd = drawdown_series(returns, missing).dropna()  # leading NaNs: not trading yet
    underwater = dd < 0
    episodes: list[DrawdownEpisode] = []
    # Each run of consecutive underwater days is one episode.
    run_id = (underwater != underwater.shift(fill_value=False)).cumsum()
    positions = {ts: i for i, ts in enumerate(dd.index)}
    for _, block in dd[underwater].groupby(run_id[underwater]):
        start_pos = positions[block.index[0]]
        end_pos = positions[block.index[-1]]
        # First date within rounding distance of the low, so ties pick the earliest.
        trough = block[block <= block.min() + _DD_TOL].index[0]
        peak = dd.index[start_pos - 1] if start_pos > 0 else None
        recovery = dd.index[end_pos + 1] if end_pos + 1 < len(dd) else None
        peak_pos = start_pos - 1
        episodes.append(
            DrawdownEpisode(
                peak=peak,
                trough=trough,
                recovery=recovery,
                depth=float(block.min()),
                days_to_trough=positions[trough] - peak_pos,
                days_to_recover=(end_pos + 1 - positions[trough]) if recovery is not None else None,
            )
        )
    episodes.sort(key=lambda e: e.depth)
    return episodes[:top]


def rolling_volatility(
    returns: Frame, window: int = 63, periods_per_year: int = TRADING_DAYS
) -> Frame:
    """Annualised volatility over a trailing window (only past data, no centering)."""
    if window < 2:
        raise ValueError("window must be at least 2")
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: Frame, window: int = 252, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> Frame:
    """Sharpe ratio over a trailing window."""
    if window < 2:
        raise ValueError("window must be at least 2")
    excess = returns - periodic_rate(risk_free, periods_per_year)
    roll = excess.rolling(window, min_periods=window)
    std = roll.std(ddof=1)
    return roll.mean() / std.where(std.abs() > _ZERO) * np.sqrt(periods_per_year)


# --- Tail risk: Value at Risk and Conditional VaR ------------------------------------------
#
# Both are reported as POSITIVE numbers meaning a loss over one period (one day for
# daily returns): "95% VaR = 0.018" means a loss larger than 1.8% happens on only
# 5% of days. The quantile is NumPy's default linear interpolation (Excel's
# PERCENTILE.INC); CVaR averages that SAME quantile function over the tail, so the
# two are consistent for any sample size and CVaR >= VaR always holds.


def _check_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1, e.g. 0.95")
    return float(confidence)


def _clean_series(returns: pd.Series, min_rows: int = 2) -> np.ndarray:
    if not isinstance(returns, pd.Series):
        raise TypeError("tail-risk functions take one return series (a Series)")
    check_returns(returns, min_rows)
    return returns.dropna().to_numpy(dtype="float64")


def _quantile_integral(sorted_x: np.ndarray, p: float) -> float:
    """Exact integral of the linear-interpolation quantile function Q(u) over [0, p].

    Q is piecewise linear with knots Q(k / (n - 1)) = x_(k), so the integral is a sum
    of trapezoids: whole ones up to the last knot below p, plus a partial one.
    """
    n = len(sorted_x)
    step = 1.0 / (n - 1)
    h = (n - 1) * p  # position of p in knot units
    m = int(np.floor(h))
    full = step * float(np.sum(sorted_x[:m] + sorted_x[1 : m + 1]) / 2.0) if m > 0 else 0.0
    frac = h - m
    if frac > 0:
        q_p = sorted_x[m] + frac * (sorted_x[m + 1] - sorted_x[m])
        full += frac * step * (sorted_x[m] + q_p) / 2.0
    return full


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: minus the (1 - confidence) quantile of the returns (a positive loss)."""
    confidence = _check_confidence(confidence)
    x = _clean_series(returns)
    return float(-np.quantile(x, 1.0 - confidence))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR (expected shortfall): the average loss in the worst (1 - confidence) tail.

    Defined as -(1/p) * integral_0^p Q(u) du with p = 1 - confidence and Q the same
    linear-interpolation quantile function as `historical_var`. A fractional number of
    tail observations needs no rounding, and because Q never decreases, the average
    of Q below p is at most Q(p): CVaR >= VaR by construction.
    """
    confidence = _check_confidence(confidence)
    x = np.sort(_clean_series(returns))
    p = 1.0 - confidence
    return float(-_quantile_integral(x, p) / p)


def tail_observations(returns: pd.Series, confidence: float = 0.95) -> float:
    """How many observations the (1 - confidence) tail holds: n x (1 - confidence)."""
    confidence = _check_confidence(confidence)
    return float(returns.notna().sum()) * (1.0 - confidence)


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian VaR: -(mean + z * std), z the (1 - confidence) standard normal quantile."""
    from scipy.stats import norm

    confidence = _check_confidence(confidence)
    x = _clean_series(returns)
    z = norm.ppf(1.0 - confidence)
    return float(-(x.mean() + z * x.std(ddof=1)))


def parametric_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian CVaR: std * pdf(z) / (1 - confidence) - mean."""
    from scipy.stats import norm

    confidence = _check_confidence(confidence)
    x = _clean_series(returns)
    z = norm.ppf(1.0 - confidence)
    return float(x.std(ddof=1) * norm.pdf(z) / (1.0 - confidence) - x.mean())


# --- Beta and alpha against a benchmark -----------------------------------------------------


@dataclass(frozen=True)
class BetaResult:
    beta: float  # Cov(r_p, r_b) / Var(r_b): the OLS slope
    alpha: float  # Jensen's alpha, annualised (x periods_per_year)
    correlation: float
    r_squared: float
    observations: int


def beta_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> BetaResult:
    """Beta, Jensen's alpha, correlation and R^2 of `returns` against `benchmark`.

    Closed-form OLS on the dates both series have. With a constant risk-free rate the
    slope is the same for raw and excess returns; only alpha uses the excess returns:
    alpha = mean(r - rf) - beta * mean(r_b - rf), annualised. A benchmark with zero
    variance gives NaN beta, alpha and correlation.
    """
    if not isinstance(returns, pd.Series) or not isinstance(benchmark, pd.Series):
        raise TypeError("beta_alpha takes two return series")
    both = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    check_returns(both.iloc[:, 0], 3)
    rp = both.iloc[:, 0].to_numpy(dtype="float64")
    rb = both.iloc[:, 1].to_numpy(dtype="float64")
    var_b = rb.var(ddof=1)
    var_p = rp.var(ddof=1)
    cov = float(np.cov(rp, rb, ddof=1)[0, 1])
    beta = _safe_div(cov, var_b)
    rf = periodic_rate(risk_free, periods_per_year)
    alpha = (rp.mean() - rf - beta * (rb.mean() - rf)) * periods_per_year
    corr = _safe_div(cov, np.sqrt(var_p * var_b))
    return BetaResult(
        beta=float(beta),
        alpha=float(alpha),
        correlation=float(corr),
        r_squared=float(corr**2),
        observations=len(both),
    )
