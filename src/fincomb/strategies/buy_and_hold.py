"""Buy once at the start, never rebalance: holdings drift with prices."""

from __future__ import annotations

import pandas as pd

from fincomb.strategies.base import Strategy
from fincomb.strategies.params import ParamSpec

WEIGHTS = ParamSpec(
    "weights", "weights", None, "Symbol -> weight, summing to at most 1 (default: equal weight)"
)


class FixedTargets(Strategy):
    """Shared by buy-and-hold and fixed weights: the same target every time."""

    uses_prices = False
    param_specs = (WEIGHTS,)

    def validate(self) -> None:
        self._check_weights_param(self.params["weights"])

    def target_weights(self, history: pd.DataFrame, previous: pd.Series | None) -> pd.Series:
        w = self.params["weights"]
        if w is None:
            w = {s: 1.0 / len(self.symbols) for s in self.symbols}
        return self._weights(w)


class BuyAndHold(FixedTargets):
    name = "buy_and_hold"
    description = (
        "Buy the given weights (default equal) at the first close and hold; weights drift "
        "with prices and are never reset."
    )
    default_rebalance = "never"
