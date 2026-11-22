"""Momentum: trailing return skipping the most recent month (12-1 by default).

score_i = P_i(t - skip) / P_i(t - lookback) - 1, from adjusted closes (a ratio, so a
re-adjustment of history changes nothing).

* cross_sectional: hold the `top_n` highest scores, equal weight; ties go to the
  symbol listed first.
* time_series: each asset has a fixed 1/n slot, held when its score is positive and
  in cash otherwise.
"""

from __future__ import annotations

import pandas as pd

from fincomb.strategies.base import Strategy
from fincomb.strategies.params import ParamSpec


class Momentum(Strategy):
    name = "momentum"
    description = (
        "Trailing return from t-lookback to t-skip (default 12-1 months). cross_sectional "
        "holds the top_n equally; time_series holds each asset with a positive score in a "
        "1/n slot, the rest in cash."
    )
    default_rebalance = "monthly"
    param_specs = (
        ParamSpec(
            "mode",
            "str",
            "cross_sectional",
            "Ranking or absolute momentum",
            choices=("cross_sectional", "time_series"),
        ),
        ParamSpec("lookback", "int", 252, "Trading days back to the start of the window", 2, 2520),
        ParamSpec("skip", "int", 21, "Most recent trading days left out", 0, 252),
        ParamSpec("top_n", "int", 3, "Assets held (cross_sectional)", 1, 50),
    )

    def validate(self) -> None:
        if self.params["skip"] >= self.params["lookback"]:
            raise ValueError("skip must be smaller than lookback")

    @property
    def warmup_bars(self) -> int:
        return self.params["lookback"] + 1

    def scores(self, history: pd.DataFrame) -> pd.Series:
        lb, skip = self.params["lookback"], self.params["skip"]
        p = self._tail(history, lb + 1)
        return pd.Series(p[-1 - skip] / p[0] - 1.0, index=self.symbols)

    def target_weights(self, history: pd.DataFrame, previous: pd.Series | None) -> pd.Series:
        score = self.scores(history)
        n = len(self.symbols)
        if self.params["mode"] == "time_series":
            return self._weights({s: 1.0 / n for s in self.symbols if score[s] > 0})
        k = min(self.params["top_n"], n)
        order = sorted(range(n), key=lambda i: (-score.iloc[i], i))
        return self._weights({self.symbols[i]: 1.0 / k for i in order[:k]})
