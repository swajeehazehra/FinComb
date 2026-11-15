"""Data-quality rules for normalised daily bars.

`validate_bars` never mutates its input. It returns the rows that are safe to
store plus a list of issues:

* severity "error"   -> the row is rejected (would also violate DB constraints)
* severity "warning" -> the row is kept but flagged for review
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from fincomb.db.models import Severity

PRICE_FIELDS = ["open", "high", "low", "close", "adj_close"]
EPS = 1e-9  # tolerance for float rounding in OHLC comparisons


@dataclass(frozen=True)
class Issue:
    bar_date: date | None
    rule: str
    severity: Severity
    detail: str


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    issues: list[Issue] = field(default_factory=list)

    @property
    def rejected(self) -> int:
        return len({i.bar_date for i in self.issues if i.severity is Severity.ERROR})


def _flag(mask: pd.Series, rule: str, severity: Severity, detail: pd.Series | str) -> list[Issue]:
    out = []
    for ts in mask[mask].index:
        d = detail if isinstance(detail, str) else str(detail.loc[ts])
        out.append(Issue(ts.date(), rule, severity, d))
    return out


def validate_bars(
    df: pd.DataFrame,
    *,
    today: date | None = None,
    extreme_return_threshold: float = 0.40,
    stale_price_run: int = 5,
) -> ValidationResult:
    today = today or date.today()
    issues: list[Issue] = []
    E, W = Severity.ERROR, Severity.WARNING

    # 1. Duplicate dates: keep the first occurrence.
    dup = pd.Series(df.index.duplicated(keep="first"), index=df.index)
    if dup.any():
        issues += [
            Issue(ts.date(), "duplicate_date", E, "duplicate bar dropped")
            for ts in df.index[dup.to_numpy()]
        ]
    df = df[~dup.to_numpy()]

    # 2. Missing prices.
    missing = df[PRICE_FIELDS].isna().any(axis=1)
    issues += _flag(missing, "missing_value", E, "one or more prices are NaN")

    # 3. Non-positive prices.
    nonpos = (df[PRICE_FIELDS] <= 0).any(axis=1) & ~missing
    issues += _flag(nonpos, "non_positive_price", E, "price <= 0")

    # 4. Negative volume.
    negvol = df["volume"] < 0
    issues += _flag(negvol, "negative_volume", E, "volume < 0")

    # 5. OHLC consistency: low <= open, close <= high.
    hi_ok = df["high"] + EPS >= df[["open", "close", "low"]].max(axis=1)
    lo_ok = df["low"] - EPS <= df[["open", "close"]].min(axis=1)
    bad_ohlc = ~(hi_ok & lo_ok) & ~missing
    issues += _flag(
        bad_ohlc,
        "ohlc_inconsistent",
        E,
        "O="
        + df["open"].astype(str)
        + " H="
        + df["high"].astype(str)
        + " L="
        + df["low"].astype(str)
        + " C="
        + df["close"].astype(str),
    )

    # 6. Bars dated in the future.
    future = pd.Series(df.index > pd.Timestamp(today), index=df.index)
    issues += _flag(future, "future_date", E, f"date after {today}")

    reject = (missing | nonpos | negvol | bad_ohlc | future).to_numpy()
    clean = df[~reject]

    # --- Warnings, computed on rows that passed. ---
    # 7. Extreme daily move on the adjusted series (often an unadjusted split).
    ret = clean["adj_close"].pct_change()
    extreme = ret.abs() > extreme_return_threshold
    issues += _flag(
        extreme, "extreme_return", W, "daily return " + (ret * 100).round(1).astype(str) + "%"
    )

    # 8. Stale prices: the same close N days in a row (halted or bad feed).
    if len(clean) >= stale_price_run:
        same = clean["close"].diff().eq(0)
        run_id = (~same).cumsum()
        run_len = same.groupby(run_id).cumsum()
        stale = run_len == stale_price_run - 1  # flag once, when the run reaches N
        issues += _flag(
            stale, "stale_price", W, f"close unchanged for {stale_price_run} consecutive bars"
        )

    issues.sort(key=lambda i: (i.bar_date or date.min, i.rule))
    return ValidationResult(clean=clean.copy(), issues=issues)


__all__ = ["Issue", "ValidationResult", "validate_bars"]
