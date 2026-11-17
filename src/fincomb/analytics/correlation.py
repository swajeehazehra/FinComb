"""Correlation between assets, computed on returns (never on prices)."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from fincomb.analytics._utils import check_returns
from fincomb.exceptions import InsufficientDataError

Method = Literal["pearson", "spearman"]


def correlation_matrix(
    returns: pd.DataFrame, method: Method = "pearson", min_periods: int = 20
) -> pd.DataFrame:
    """Pairwise correlation of asset returns.

    Uses only dates where both assets of a pair have a return, and needs at
    least `min_periods` such dates per pair (otherwise that cell is NaN).
    """
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        raise ValueError("correlation needs a DataFrame with at least two assets")
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    check_returns(returns, min_rows=min_periods)
    return returns.corr(method=method, min_periods=min_periods)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    """Correlation of two return series over a trailing window."""
    if window < 3:
        raise ValueError("window must be at least 3")
    aligned = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(aligned) < window:
        raise InsufficientDataError(f"need at least {window} overlapping returns")
    x, y = aligned.iloc[:, 0], aligned.iloc[:, 1]
    return x.rolling(window, min_periods=window).corr(y)
