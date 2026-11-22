"""Constant target weights, reset at every rebalance (e.g. a saved portfolio)."""

from __future__ import annotations

from fincomb.strategies.buy_and_hold import FixedTargets


class FixedWeights(FixedTargets):
    name = "fixed_weights"
    description = (
        "Hold the given weights (default equal, or a saved portfolio's) and reset them to "
        "target at every rebalance date."
    )
    default_rebalance = "monthly"
