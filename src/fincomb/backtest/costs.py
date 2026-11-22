"""Transaction costs: a fixed number of basis points per unit of turnover.

Traded value on a rebalance = turnover x pre-trade portfolio value, where turnover
is sum_i |w_new_i - w_drifted_i| over the risky assets (cash is not charged).
Commission = traded value x cost_bps / 10 000, slippage = traded value x
slippage_bps / 10 000. Both are deducted from the portfolio on that day.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MAX_BPS = 1000.0  # 10% per unit traded; above this the model makes no sense


@dataclass(frozen=True)
class FixedBpsCost:
    cost_bps: float = 0.0
    slippage_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cost_bps", "slippage_bps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or value > MAX_BPS:
                raise ValueError(f"{name} must be between 0 and {MAX_BPS:g}, got {value}")

    @property
    def rate(self) -> float:
        """Total cost per unit of traded value."""
        return (self.cost_bps + self.slippage_bps) / 10_000.0

    def split(self, traded_value: float) -> tuple[float, float]:
        """(commission, slippage) in money for a traded value."""
        return (
            traded_value * self.cost_bps / 10_000.0,
            traded_value * self.slippage_bps / 10_000.0,
        )
