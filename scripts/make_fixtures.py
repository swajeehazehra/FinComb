"""Generate the small *synthetic* price files in tests/fixtures/.

These are NOT real market prices. They are seeded random walks so tests and
offline demos are deterministic and never need the network.

    python scripts/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DATES = pd.bdate_range("2023-01-02", "2023-12-29")


def synthetic(seed: int, start_price: float, mu: float, sigma: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu / 252, sigma / np.sqrt(252), len(DATES))
    close = start_price * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, 0.003, len(DATES)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(DATES)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(DATES)))
    volume = rng.integers(1_000_000, 5_000_000, len(DATES))
    return pd.DataFrame(
        {
            "Date": DATES.strftime("%Y-%m-%d"),
            "Open": open_.round(4),
            "High": high.round(4),
            "Low": low.round(4),
            "Close": close.round(4),
            "Adj Close": close.round(4),
            "Volume": volume,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic(1, 100.0, 0.12, 0.25).to_csv(OUT / "SYNA.csv", index=False)
    synthetic(2, 50.0, 0.05, 0.35).to_csv(OUT / "SYNB.csv", index=False)
    synthetic(3, 400.0, 0.08, 0.15).to_csv(OUT / "SYNM.csv", index=False)

    # A file with one deliberate problem per validation rule.
    bad = synthetic(4, 20.0, 0.0, 0.2).head(30).copy()
    bad.loc[3, "Close"] = np.nan  # missing_value
    bad.loc[5, "Low"] = -1.0  # non_positive_price
    bad.loc[7, "High"] = bad.loc[7, "Low"] * 0.9  # ohlc_inconsistent
    bad.loc[9, "Volume"] = -100  # negative_volume
    bad.loc[12, ["Close", "Adj Close"]] = bad.loc[11, "Close"] * 2  # extreme_return (+100%)
    bad.loc[12, "High"] = bad.loc[12, "Close"] * 1.01
    bad = pd.concat([bad, bad.iloc[[15]]])  # duplicate_date
    for i in range(20, 26):  # stale_price (6 identical closes)
        bad.loc[i, ["Open", "High", "Low", "Close", "Adj Close"]] = 21.0
    bad.to_csv(OUT / "BADDATA.csv", index=False)
    print(f"wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
