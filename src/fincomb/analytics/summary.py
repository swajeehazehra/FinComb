"""One-call performance report for one or more assets."""

from __future__ import annotations

import pandas as pd

from fincomb.analytics import returns as R
from fincomb.analytics import risk
from fincomb.analytics._utils import check_prices
from fincomb.analytics.constants import TRADING_DAYS


def performance_summary(
    prices: pd.Series | pd.DataFrame, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS
) -> pd.DataFrame:
    """Key statistics per asset (one row each) from adjusted close prices."""
    prices = check_prices(prices)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(prices.name or "asset")

    rows = {}
    for symbol in prices.columns:
        p = prices[symbol].dropna()
        r = R.simple_returns(p)
        rows[symbol] = {
            "start": p.index[0].date(),
            "end": p.index[-1].date(),
            "observations": int(r.count()),
            "total_return": R.total_return(r),
            "cagr": R.cagr(r, periods_per_year),
            "annual_volatility": risk.annualized_volatility(r, periods_per_year),
            "sharpe": risk.sharpe_ratio(r, risk_free, periods_per_year),
            "sortino": risk.sortino_ratio(r, risk_free, periods_per_year),
            "max_drawdown": risk.max_drawdown(r),
            "calmar": risk.calmar_ratio(r, periods_per_year),
            "best_day": r.max(),
            "worst_day": r.min(),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "symbol"
    return out
