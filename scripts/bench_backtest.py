"""Time the backtest engine on synthetic data (no database, no network).

    python scripts/bench_backtest.py            # every case, 20 years x 50 assets
    python scripts/bench_backtest.py --quick    # skip the 50-asset max-Sharpe case (~90 s)

Targets (Linux, plan clarification C2): ordinary strategies < 2 s; optimized with
equal weight, inverse vol or min variance < 10 s; optimized max Sharpe with 10
assets < 15 s and with 50 assets < 180 s. Windows is usually slower.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import pandas as pd

from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.engine import run
from fincomb.strategies.registry import get_strategy

DAYS = 20 * 252 + 253  # 20 evaluation years plus a year of warm-up
WARMUP = 253


def prices(assets: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, (DAYS, assets)) + rng.normal(0, 0.006, (DAYS, 1))
    idx = pd.bdate_range("2004-01-01", periods=DAYS)
    return pd.DataFrame(
        100 * np.cumprod(1 + r, axis=0), index=idx, columns=[f"S{i}" for i in range(assets)]
    )


CASES = [
    # (label, strategy, params, assets, target seconds)
    ("buy_and_hold", "buy_and_hold", {}, 50, 2),
    ("fixed_weights monthly", "fixed_weights", {}, 50, 2),
    ("momentum monthly", "momentum", {"top_n": 10}, 50, 2),
    ("mean_reversion daily", "mean_reversion", {}, 50, 2),
    ("optimized equal_weight", "optimized", {"method": "equal_weight"}, 50, 10),
    ("optimized inverse_vol", "optimized", {"method": "inverse_vol"}, 50, 10),
    ("optimized min_variance", "optimized", {"method": "min_variance"}, 50, 10),
    ("optimized max_sharpe", "optimized", {"method": "max_sharpe"}, 10, 15),
    ("optimized max_sharpe", "optimized", {"method": "max_sharpe"}, 50, 180),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the 50-asset max-Sharpe case")
    args = parser.parse_args()
    print(f"{platform.system()} {platform.machine()}, Python {platform.python_version()}")
    print(f"{'case':<26}{'assets':>7}{'seconds':>10}{'target':>9}  result")
    data = {n: prices(n, seed=n) for n in (10, 50)}
    for label, name, params, assets, target in CASES:
        if args.quick and name == "optimized" and params["method"] == "max_sharpe" and assets == 50:
            continue
        p = data[assets]
        strategy = get_strategy(name)(list(p.columns), params)
        t = time.perf_counter()
        run(p, strategy, p.index[WARMUP], costs=FixedBpsCost(5, 5))
        elapsed = time.perf_counter() - t
        ok = "ok" if elapsed < target else "SLOWER THAN TARGET"
        print(f"{label:<26}{assets:>7}{elapsed:>10.2f}{'< ' + str(target):>9}  {ok}")


if __name__ == "__main__":
    main()
