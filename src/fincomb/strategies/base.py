"""The Strategy interface.

A strategy only says what it wants to hold. The engine decides when that trade
happens, what it costs and how holdings drift afterwards, so look-ahead control
lives in one place (`backtest/engine.py`):

* At each rebalance date t the engine calls `target_weights(history, previous)`
  with `history` = adjusted closes **up to and including t** (never later), and
  `previous` = the targets this strategy returned last time (None at the start).
* The answer is a Series of weights per symbol, each in [0, 1], summing to at most 1
  (the rest is cash), or None for "no change, don't trade".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from fincomb.strategies.params import ParamSpec, parse_params


class Strategy(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    default_rebalance: ClassVar[str]
    # False: the targets don't depend on prices (buy-and-hold, fixed weights), so the
    # first allocation needs no history before the first evaluation date.
    uses_prices: ClassVar[bool] = True
    param_specs: ClassVar[tuple[ParamSpec, ...]] = ()

    def __init__(self, symbols: list[str], params: dict[str, Any] | None = None) -> None:
        if not symbols:
            raise ValueError("a strategy needs at least one symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("each symbol may appear only once")
        self.symbols = list(symbols)
        self.symbol_index = pd.Index(self.symbols, name="symbol")
        self.params = parse_params(self.param_specs, params)
        self._warnings: list[str] = []
        self.validate()

    def validate(self) -> None:
        """Check parameters against each other and the universe; raise ValueError."""
        return None  # nothing to check by default

    @property
    def warmup_bars(self) -> int:
        """Price rows (including the signal date) the history must have at a signal."""
        return 1

    @abstractmethod
    def target_weights(self, history: pd.DataFrame, previous: pd.Series | None) -> pd.Series | None:
        """Target weights from prices up to the signal date (the last row of `history`)."""

    def run_warnings(self) -> list[str]:
        """Warnings collected during a run (e.g. optimiser fallbacks)."""
        return list(self._warnings)

    def diagnostics(self, prices: pd.DataFrame) -> dict[str, Any]:
        """Extra statistics for the report, computed after the run (not used to trade)."""
        return {}

    def _weights(self, values: dict[str, float]) -> pd.Series:
        s = pd.Series(values, dtype="float64").reindex(self.symbols, fill_value=0.0)
        s.index.name = "symbol"
        return s.rename("weight")

    def _weight_array(self, values: np.ndarray) -> pd.Series:
        """Weights in `self.symbols` order, from an array (fast path for daily strategies)."""
        return pd.Series(values, index=self.symbol_index, name="weight")

    def _tail(self, history: pd.DataFrame, rows: int) -> np.ndarray:
        """The last `rows` prices of this strategy's symbols, as an array (dates x symbols)."""
        tail = history.iloc[-rows:]
        if not tail.columns.equals(self.symbol_index):
            tail = tail[self.symbols]
        return tail.to_numpy(dtype="float64")

    def _check_weights_param(self, weights: dict[str, float] | None) -> None:
        if weights is None:
            return
        extra = sorted(set(weights) - set(self.symbols))
        if extra:
            raise ValueError(f"weights name symbols outside the universe: {', '.join(extra)}")
        bad = [f"{s}={w:g}" for s, w in weights.items() if w < 0 or w > 1]
        if bad:
            raise ValueError(f"weights must be between 0 and 1 (long-only): {', '.join(bad)}")
        total = sum(weights.values())
        if total <= 0 or total > 1 + 1e-6:
            raise ValueError(
                f"weights must sum to more than 0 and at most 1, they sum to {total:g}"
            )
