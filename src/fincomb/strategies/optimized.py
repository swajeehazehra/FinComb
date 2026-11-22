"""Re-run Milestone 4's portfolio construction at each rebalance on a trailing window."""

from __future__ import annotations

import pandas as pd

from fincomb.portfolio.construction import METHODS, WeightBounds, construct, min_observations
from fincomb.strategies.base import Strategy
from fincomb.strategies.params import ParamSpec


class Optimized(Strategy):
    name = "optimized"
    description = (
        "At each rebalance, weights from portfolio construction (equal weight, inverse "
        "volatility, minimum variance or maximum Sharpe) fitted on the trailing `lookback` "
        "daily returns ending at that date only."
    )
    default_rebalance = "monthly"
    param_specs = (
        ParamSpec("method", "str", "min_variance", "Construction method", choices=METHODS),
        ParamSpec("lookback", "int", 252, "Daily returns in each window", 20, 2520),
        ParamSpec("min_weight", "float", 0.0, "Smallest weight per asset", 0, 1),
        ParamSpec("max_weight", "float", 1.0, "Largest weight per asset", 0, 1),
        ParamSpec(
            "covariance",
            "str",
            "ledoit_wolf",
            "Covariance estimate",
            choices=("ledoit_wolf", "sample"),
        ),
        ParamSpec("risk_free", "float", 0.0, "Annual risk-free rate for max Sharpe", -1, 1),
    )

    def validate(self) -> None:
        p = self.params
        self.bounds = WeightBounds(p["min_weight"], p["max_weight"])
        self.bounds.check_feasible(len(self.symbols))
        need = min_observations(len(self.symbols))
        if p["lookback"] < need:
            raise ValueError(
                f"lookback must be at least {need} returns for {len(self.symbols)} assets"
            )
        self.fallbacks: list[str] = []

    @property
    def warmup_bars(self) -> int:
        return self.params["lookback"] + 1

    def target_weights(self, history: pd.DataFrame, previous: pd.Series | None) -> pd.Series:
        p = self.params
        window = history.iloc[-(p["lookback"] + 1) :]
        result = construct(
            window, p["method"], self.bounds, covariance=p["covariance"], risk_free=p["risk_free"]
        )
        if result.fallback_used:
            self.fallbacks.append(str(window.index[-1].date()))
        return self._weights(result.weights.to_dict())

    def run_warnings(self) -> list[str]:
        if not self.fallbacks:
            return []
        shown = ", ".join(self.fallbacks[:5]) + (" ..." if len(self.fallbacks) > 5 else "")
        return [
            f"{self.params['method']} could not be solved reliably at {len(self.fallbacks)} "
            f"rebalance(s) ({shown}); bounded inverse volatility was used there instead."
        ]
