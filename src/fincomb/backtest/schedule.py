"""Rebalance dates: the last trading day of each calendar period in the data."""

from __future__ import annotations

import numpy as np
import pandas as pd

FREQUENCIES: tuple[str, ...] = ("daily", "weekly", "monthly", "quarterly", "annual", "never")
_PERIODS = {"weekly": "W", "monthly": "M", "quarterly": "Q", "annual": "Y"}


def rebalance_mask(index: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """True on each date that is the last date of its period in `index`.

    The last date of `index` is never a rebalance date: its period may be incomplete,
    and a trade there could not earn anything before the data ends.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"rebalance must be one of {', '.join(FREQUENCIES)}")
    n = len(index)
    if frequency == "never" or n == 0:
        return np.zeros(n, dtype=bool)
    if frequency == "daily":
        mask = np.ones(n, dtype=bool)
    else:
        periods = index.to_period(_PERIODS[frequency]).asi8
        mask = np.append(periods[1:] != periods[:-1], True)
    mask[-1] = False
    return mask


def rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    return index[rebalance_mask(index, frequency)]
