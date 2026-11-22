"""Backtest metrics and the comparison against a benchmark.

* Strategy and benchmark statistics come from the existing analytics code applied to
  their equity curves. Total return and CAGR of the strategy are measured from the
  initial capital, so they include the cost of the initial allocation.
* Tracking error = std(r_s - r_b) x sqrt(252); information ratio = mean(r_s - r_b) x
  252 / tracking error.
* Alpha and beta: OLS of strategy excess returns on benchmark excess returns
  (Statsmodels), with Newey-West (HAC) standard errors, lag floor(4 (T/100)^(2/9)),
  because daily returns are heteroskedastic and autocorrelated.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fincomb.analytics.constants import TRADING_DAYS
from fincomb.analytics.returns import years_between
from fincomb.analytics.risk import _ZERO, periodic_rate
from fincomb.analytics.summary import performance_summary
from fincomb.backtest.engine import EngineResult

SUMMARY_KEYS = (
    "total_return",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "best_day",
    "worst_day",
)


def newey_west_lags(n: int) -> int:
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class RegressionResult:
    alpha: float  # annualised (x periods_per_year)
    beta: float
    alpha_t: float
    alpha_p: float
    beta_t: float
    beta_p: float
    r_squared: float
    hac_lags: int
    observations: int


def hac_regression(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> RegressionResult:
    """OLS r - rf = a + b (r_b - rf) + e with Newey-West standard errors."""
    import statsmodels.api as sm

    both = pd.concat([returns, benchmark], axis=1, join="inner").dropna()
    n = len(both)
    lags = newey_west_lags(n)
    nan = float("nan")
    rf = periodic_rate(risk_free, periods_per_year)
    y = both.iloc[:, 0].to_numpy(dtype="float64") - rf
    x = both.iloc[:, 1].to_numpy(dtype="float64") - rf
    if n < 3 or x.std(ddof=1) <= _ZERO:
        return RegressionResult(nan, nan, nan, nan, nan, nan, nan, lags, n)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        t = np.asarray(fit.tvalues, dtype="float64")
        p = np.asarray(fit.pvalues, dtype="float64")
    params = np.asarray(fit.params, dtype="float64")
    resid_ss = float(np.sum((y - params[0] - params[1] * x) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - resid_ss / total_ss if total_ss > 0 else nan

    def clean(v: float) -> float:
        return float(v) if math.isfinite(v) else nan

    return RegressionResult(
        alpha=float(params[0] * periods_per_year),
        beta=float(params[1]),
        alpha_t=clean(t[0]),
        alpha_p=clean(p[0]),
        beta_t=clean(t[1]),
        beta_p=clean(p[1]),
        r_squared=clean(r2),
        hac_lags=lags,
        observations=n,
    )


@dataclass
class BacktestReport:
    start: pd.Timestamp
    end: pd.Timestamp
    observations: int  # daily returns in the evaluation period
    strategy: dict[str, float]
    benchmark: dict[str, float] | None
    relative: dict[str, float] | None
    trading: dict[str, float]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready: NaN and infinity become None."""
        return _jsonable(
            {
                "start": self.start.date().isoformat(),
                "end": self.end.date().isoformat(),
                "observations": self.observations,
                "strategy": self.strategy,
                "benchmark": self.benchmark,
                "relative": self.relative,
                "trading": self.trading,
                "diagnostics": self.diagnostics,
            }
        )


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, bool | str) or x is None:
        return x
    if isinstance(x, int | np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        return float(x) if math.isfinite(x) else None
    return x


def _summary(equity: pd.Series, risk_free: float, capital: float | None = None) -> dict:
    row = performance_summary(equity, risk_free).iloc[0]
    out = {k: float(row[k]) for k in SUMMARY_KEYS}
    if capital is not None:
        # Measured from the money put in, so the initial allocation's cost counts.
        years = years_between(equity.index[0], equity.index[-1])
        total = float(equity.iloc[-1] / capital - 1.0)
        out["total_return"] = total
        out["cagr"] = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
        mdd = out["max_drawdown"]
        out["calmar"] = out["cagr"] / abs(mdd) if mdd < 0 else float("nan")
    return out


def build_report(
    result: EngineResult,
    benchmark_equity: pd.Series | None = None,
    risk_free: float = 0.0,
    diagnostics: dict[str, Any] | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> BacktestReport:
    eq = result.equity
    capital = result.initial_capital
    strat = _summary(eq, risk_free, capital)
    years = years_between(eq.index[0], eq.index[-1])

    rebal = result.turnover.iloc[1:]
    final_value = float(eq.iloc[-1])
    trading = {
        "initial_capital": capital,
        "final_equity": final_value,
        "rebalances": int((rebal > 0).sum()),
        "initial_turnover": float(result.turnover.iloc[0]),
        "total_turnover": float(result.turnover.sum()),
        "annual_turnover": float(rebal.sum() / years) if years > 0 else float("nan"),
        "total_costs": float(result.costs.sum()),
        "costs_pct_of_capital": float(result.costs.sum() / capital),
        "annual_cost_drag": float(
            (result.turnover * result.cost_rate).sum() / years if years > 0 else float("nan")
        ),
        "average_cash": float((result.cash / eq).mean()),
        "average_gross_exposure": float(result.gross_exposure.mean()),
        "liquidation_cost": float(result.gross_exposure.iloc[-1] * final_value * result.cost_rate),
    }

    bench = relative = None
    if benchmark_equity is not None:
        b = benchmark_equity.reindex(eq.index)
        bench = _summary(b, risk_free, capital)
        r_s = eq.pct_change().iloc[1:]
        r_b = b.pct_change().iloc[1:]
        active = r_s - r_b
        te = float(active.std(ddof=1) * math.sqrt(periods_per_year))
        mean_active = float(active.mean() * periods_per_year)
        reg = hac_regression(r_s, r_b, risk_free, periods_per_year)
        corr = float(np.corrcoef(r_s, r_b)[0, 1]) if r_s.std() > 0 and r_b.std() > 0 else np.nan
        relative = {
            "excess_total_return": strat["total_return"] - bench["total_return"],
            "excess_cagr": strat["cagr"] - bench["cagr"],
            "tracking_error": te,
            "information_ratio": mean_active / te if te > _ZERO else float("nan"),
            "alpha": reg.alpha,
            "alpha_t": reg.alpha_t,
            "alpha_p": reg.alpha_p,
            "beta": reg.beta,
            "beta_t": reg.beta_t,
            "beta_p": reg.beta_p,
            "r_squared": reg.r_squared,
            "correlation": float(corr),
            "hac_lags": reg.hac_lags,
        }

    return BacktestReport(
        start=eq.index[0],
        end=eq.index[-1],
        observations=len(eq) - 1,
        strategy=strat,
        benchmark=bench,
        relative=relative,
        trading=trading,
        diagnostics=diagnostics or {},
    )
