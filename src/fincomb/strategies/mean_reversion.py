"""Mean reversion: z-score of price against its trailing mean.

z_t = (P_t - mean(P over the last `window` days)) / std(same), from adjusted closes.
z is scale-invariant: multiplying all past prices by a constant (a re-adjustment)
leaves it unchanged. Long/flat per asset with a fixed 1/n slot: enter when
z < -entry_z, exit when z >= exit_z. Trades only on days a position opens or closes.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from fincomb.strategies.base import Strategy
from fincomb.strategies.params import ParamSpec


class MeanReversion(Strategy):
    name = "mean_reversion"
    description = (
        "Buy an asset (1/n slot) when its z-score against the trailing `window`-day mean "
        "falls below -entry_z; sell when it recovers to exit_z. Evaluated daily, trades only "
        "when a position changes. ADF tests are reported as an in-sample diagnostic."
    )
    default_rebalance = "daily"
    param_specs = (
        ParamSpec("window", "int", 20, "Trading days in the rolling mean and std", 3, 504),
        ParamSpec("entry_z", "float", 2.0, "Enter when z < -entry_z", 0, 10),
        ParamSpec("exit_z", "float", 0.0, "Exit when z >= exit_z", -10, 10),
    )

    def validate(self) -> None:
        if self.params["exit_z"] <= -self.params["entry_z"]:
            raise ValueError("exit_z must be above -entry_z, or a position would exit at once")

    @property
    def warmup_bars(self) -> int:
        return self.params["window"]

    def _z(self, history: pd.DataFrame) -> np.ndarray:
        tail = self._tail(history, self.params["window"])
        mean = tail.mean(axis=0)
        std = tail.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(std > 0, (tail[-1] - mean) / std, np.nan)

    def zscores(self, history: pd.DataFrame) -> pd.Series:
        """z-score of the last price in `history` per symbol (NaN for a flat window)."""
        return pd.Series(self._z(history), index=self.symbols)

    def target_weights(self, history: pd.DataFrame, previous: pd.Series | None) -> pd.Series | None:
        z = self._z(history)
        if previous is None:
            held = np.zeros(len(self.symbols), dtype=bool)
        else:
            if not previous.index.equals(self.symbol_index):
                previous = previous.reindex(self.symbols, fill_value=0.0)
            held = previous.to_numpy(dtype="float64") > 0
        with np.errstate(invalid="ignore"):
            stay = held & ~(z >= self.params["exit_z"])  # NaN (flat window) keeps a position
            enter = ~held & (z < -self.params["entry_z"])
        now = stay | enter
        if previous is not None and np.array_equal(now, held):
            return None
        return self._weight_array(np.where(now, 1.0 / len(self.symbols), 0.0))

    def diagnostics(self, prices: pd.DataFrame) -> dict[str, Any]:
        """Augmented Dickey-Fuller test on log prices over the whole period (in-sample)."""
        from statsmodels.tsa.stattools import adfuller

        out: dict[str, Any] = {}
        for s in self.symbols:
            x = np.log(prices[s].to_numpy(dtype="float64"))
            if len(x) < 20 or np.ptp(x) == 0:
                out[s] = None
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, pvalue, lags, nobs, _, _ = adfuller(x, autolag="AIC")
            out[s] = {
                "statistic": float(stat),
                "p_value": float(pvalue),
                "lags": int(lags),
                "observations": int(nobs),
            }
        return {
            "adf": out,
            "adf_note": "In-sample diagnostic over the whole period; not used by the trading "
            "rule. A low p-value means the log price looked mean-reverting in this sample.",
        }
