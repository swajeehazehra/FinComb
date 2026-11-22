# Milestone 5: Strategies and backtest engine

## 1. What this milestone builds

Milestones 1 to 4 answer "how did each asset do, and how should I combine them?".
This one answers "what would a rule have earned, after costs, without seeing the
future?".

- **Five long-only strategies**: `buy_and_hold`, `fixed_weights` (constant weights
  reset at a rebalance frequency, also how a saved Milestone 4 portfolio is
  backtested), `optimized` (Milestone 4's `construct()` re-run at each rebalance on a
  trailing window), `momentum` (12-1 cross-sectional or time-series) and
  `mean_reversion` (z-score, with an ADF diagnostic).
- **One engine** (`backtest/engine.py`) that owns timing, execution, drift and costs.
  Strategies only say what they want to hold.
- **A benchmark report**: CAGR, Sharpe, drawdown and the rest for the strategy and
  the benchmark, excess return, tracking error, information ratio, and alpha and beta
  from a Statsmodels regression with Newey-West t-statistics.
- **Stored, reproducible runs** (migration `0004_backtests`): the inputs, the daily
  curve, the trade log, the git commit and a SHA-256 hash of the exact prices used,
  so `fincomb backtest verify` can prove a result still reproduces.
- The same features from the **CLI** (`fincomb strategies`, `fincomb backtest ...`)
  and the **API** (`/api/v1/strategies`, `/api/v1/backtests`), both calling
  `services/backtest_service.py`.

The formulas and conventions are in [METHODOLOGY.md](METHODOLOGY.md#backtesting).

## 2. How the components work

```
CLI (fincomb backtest ...)      API (/api/v1/backtests)
                \                 /
          services/backtest_service.py   load prices with warm-up, resolve a saved
                |                        portfolio, run, report, hash, store
     backtest/engine.py   <- the only place that decides WHEN a trade happens
        |          |
 strategies/*   backtest/costs.py, schedule.py      pure functions, no I/O
        |
 portfolio/construction.py (optimized)   backtest/report.py -> analytics/, statsmodels
```

| Module | Role |
|---|---|
| `strategies/base.py` | `Strategy`: `target_weights(history, previous)`, `warmup_bars`, `uses_prices`, `default_rebalance` |
| `strategies/params.py` | `ParamSpec`: one declaration gives parsing, validation, defaults and the JSON Schema |
| `strategies/*.py`, `registry.py` | the five strategies, looked up by name |
| `backtest/schedule.py` | rebalance dates: last trading day of each week, month, quarter or year |
| `backtest/costs.py` | `FixedBpsCost`: commission and slippage in bps of traded value |
| `backtest/engine.py` | the daily simulation, the initialisation trade, the trade log |
| `backtest/report.py` | metrics, benchmark comparison, HAC regression |
| `backtest/reproducibility.py` | `data_hash`, `code_version` |
| `services/backtest_service.py` | run, store, list, get, equity, trades, verify, delete |
| `api/routers/strategies.py`, `backtests.py`, `api/schemas/backtests.py` | HTTP only |
| `cli.py` | `strategies` and the `backtest` command group, formatting only |

### How a trade happens

1. **Day 0** (the first evaluation date): the initial allocation is bought at the
   close, with normal costs. Strategies that read prices use the signal from the
   previous close (warm-up history is loaded before your `--start`).
2. **Each rebalance date t**: the strategy gets prices up to and including close t,
   never later, and returns target weights (or "no change").
3. **Execution**: at close t+1 by default (`next_close`), or at close t
   (`same_close`, optimistic).
4. **Between trades** holdings drift with prices; cash earns nothing.

### Design choices worth mentioning in interviews

- **Look-ahead is impossible by construction, then tested anyway.** The engine
  slices the history it hands a strategy, and a perturbation test replaces every
  price after a cut date with random numbers and checks that nothing before it
  changes, for every strategy and both execution modes. A deliberately cheating
  strategy proves the test can catch a leak.
- **Exact accounting.** Zero-cost buy-and-hold equals the benchmark to the bit;
  daily-rebalanced fixed weights equal Milestone 4's constant-mix portfolio; with
  costs, final equity = zero-cost equity × Π(1 − turnover × cost rate). A
  hand-worked eight-day example sits in the test file.
- **Reproducibility you can check**: `verify` reruns a stored backtest and compares
  the data hash and every metric.

## 3. Get the code

In PowerShell:

```powershell
cd D:\Projects\FinComb
.venv\Scripts\Activate.ps1
git fetch origin
git checkout milestone-5      # only needed until the Milestone 5 PR is merged into main
python -m pip install --no-cache-dir -e ".[dev]"   # adds statsmodels
```

Start the database and apply the new migration:

```powershell
docker compose up -d db
alembic upgrade head          # creates backtest_runs, backtest_equity, backtest_trades
fincomb check-db              # Schema revision: 0004_backtests
```

## 4. Run the tests

```powershell
pytest
```

Expected: **473 passed** (342 from before, plus 131 new). Without the database the
integration tests are skipped instead. Only the backtest tests:

```powershell
pytest tests/unit/test_backtest_engine.py tests/unit/test_strategies.py `
       tests/unit/test_backtest_report.py tests/unit/test_backtest_performance.py `
       tests/integration/test_backtests.py tests/integration/test_backtest_api.py `
       tests/integration/test_backtest_cli.py
```

The timing tests are marked `performance`; skip them with
`pytest -m "not performance"`. The full benchmark, including the slow 50-asset
maximum-Sharpe case, is a script:

```powershell
python scripts/bench_backtest.py           # about 2 minutes on Linux
python scripts/bench_backtest.py --quick   # skips the 50-asset max-Sharpe case
```

## 5. Try it from the CLI

These use the six instruments you already have (AAPL, GLD, MSFT, QQQ, SPY, TLT).
Each run is stored unless you add `--no-save`.

```powershell
# What's available, with every parameter and its default
fincomb strategies

# The acceptance check: buy-and-hold SPY with zero costs equals the SPY benchmark
fincomb backtest run buy_and_hold SPY --start 2018-01-01 --cost-bps 0 --slippage-bps 0

# 12-1 momentum on five assets, top 2, monthly, next-close execution, default 5 + 5 bps
fincomb backtest run momentum AAPL MSFT QQQ TLT GLD --start 2018-01-01 --param top_n=2

# Time-series momentum (cash when the trend is down)
fincomb backtest run momentum SPY TLT GLD --start 2018-01-01 --param mode=time_series

# Mean reversion, daily signals, entry at z < -1.5, with ADF diagnostics
fincomb backtest run mean_reversion AAPL MSFT QQQ --start 2019-01-01 --param entry_z=1.5

# Minimum variance re-optimised each month on the trailing year, 40% cap
fincomb backtest run optimized AAPL MSFT QQQ SPY TLT GLD --start 2018-01-01 `
    --param method=min_variance --param max_weight=0.4

# A 60/40 reset every quarter, and a saved Milestone 4 portfolio
fincomb backtest run fixed_weights SPY TLT --weights SPY=0.6,TLT=0.4 --rebalance quarterly
fincomb backtest run fixed_weights --portfolio "Core MV" --start 2023-01-01

# Same-close execution (optimistic) to see how much timing matters
fincomb backtest run momentum AAPL MSFT QQQ TLT GLD --start 2018-01-01 --execution same-close

# Compare, inspect, export, verify, delete
fincomb backtest list
fincomb backtest show 2
fincomb backtest trades 2 --limit 20
fincomb backtest equity 2 --csv momentum_equity.csv
fincomb backtest verify 2
fincomb backtest delete 2
```

`--param` takes `NAME=VALUE` and can be repeated; names and limits are listed by
`fincomb strategies`. `--benchmark ""` runs without a benchmark. Warnings (an
optimiser fallback, an in-sample saved portfolio) print in red above the numbers;
notes (an effective start moved later for warm-up) print in yellow under them.

## 6. Try it from the API

Start `fincomb serve`, then in a second PowerShell window:

```powershell
$api = "http://127.0.0.1:8000/api/v1"

# Strategies and their parameter schemas
Invoke-RestMethod "$api/strategies" | Format-Table name, default_rebalance, description

# Run momentum and keep it
$body = @{ strategy = "momentum"; symbols = @("AAPL","MSFT","QQQ","TLT","GLD");
           params = @{ top_n = 2 }; start = "2018-01-01"; benchmark = "SPY" } | ConvertTo-Json -Depth 3
$bt = Invoke-RestMethod "$api/backtests" -Method Post -Body $body -ContentType "application/json"
$bt.metrics.strategy; $bt.metrics.relative; $bt.metrics.trading; $bt.warnings; $bt.notes

# Explore without saving (HTTP 200, id = null)
$body = @{ strategy = "buy_and_hold"; symbols = @("SPY"); cost_bps = 0; slippage_bps = 0;
           save = $false } | ConvertTo-Json
Invoke-RestMethod "$api/backtests" -Method Post -Body $body -ContentType "application/json"

# Stored runs, curves, trades and the reproducibility check
Invoke-RestMethod "$api/backtests" | Format-Table id, strategy, cagr, sharpe, max_drawdown
$eq = Invoke-RestMethod "$api/backtests/$($bt.id)/equity"
$eq.curves.dates[0..4]; $eq.curves.series.equity[0..4]
(Invoke-RestMethod "$api/backtests/$($bt.id)/trades?limit=10").trades | Format-Table
Invoke-RestMethod "$api/backtests/$($bt.id)/verify"

# Errors: an unknown parameter (422), an unknown run (404)
$bad = @{ strategy = "momentum"; symbols = @("SPY"); params = @{ lookbak = 3 } } | ConvertTo-Json
curl.exe -s -X POST "$api/backtests" -H "Content-Type: application/json" -d $bad
curl.exe -s "$api/backtests/99999"
```

### Things to check

- `/health` shows `"schema_revision": "0004_backtests"`.
- Buy-and-hold SPY with zero costs shows the same total return and CAGR for the
  strategy and the SPY benchmark, excess return 0 and beta 1.00.
- The default momentum run shows `next_close`; rerunning with `--execution
  same-close` changes the numbers a little (timing matters).
- Doubling `--cost-bps` leaves turnover unchanged and lowers the return.
- `fincomb backtest verify` says "reproducible"; after `fincomb refresh` of a symbol
  whose history the vendor re-adjusted, it may say the data changed, which is the
  point.
- `fincomb backtest show ID` and `GET /backtests/{ID}` give the same numbers.

## 7. Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/strategies` | Strategies, default rebalance, parameter JSON Schema |
| POST | `/backtests` | Run; stored (201) or not (`save: false`, 200) |
| GET | `/backtests` | Stored runs with headline metrics (`strategy`, `limit`) |
| GET | `/backtests/{id}` | Inputs, metrics, warnings, notes, reproducibility |
| GET | `/backtests/{id}/equity` | Daily equity, cash, exposure, benchmark, drawdown, turnover, costs |
| GET | `/backtests/{id}/trades` | Trade log (`limit`, `offset`) |
| GET | `/backtests/{id}/verify` | Rerun and compare data hash and metrics |
| DELETE | `/backtests/{id}` | Delete with its curve and trades |

## 8. Performance (Linux reference)

20 years of daily data (plus one year of warm-up), from `scripts/bench_backtest.py`
in the build container; Windows will be slower:

| Case | Assets | Seconds | Target |
|---|---|---|---|
| buy_and_hold | 50 | 0.02 | < 2 |
| fixed_weights, monthly | 50 | 0.15 | < 2 |
| momentum, monthly | 50 | 0.23 | < 2 |
| mean_reversion, daily | 50 | 1.25 | < 2 |
| optimized equal_weight | 50 | 0.58 | < 10 |
| optimized inverse_vol | 50 | 0.63 | < 10 |
| optimized min_variance | 50 | 2.24 | < 10 |
| optimized max_sharpe | 10 | 3.89 | < 15 |
| optimized max_sharpe | 50 | 103 | < 180 |

Maximum Sharpe is slow because Milestone 4 solves it from 52 starting points at
each of the 240 monthly rebalances.

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `relation "backtest_runs" does not exist` | Run `alembic upgrade head` |
| `ModuleNotFoundError: statsmodels` | Reinstall: `python -m pip install -e ".[dev]"` |
| `Effective start moved from ... to ...` | The strategy needs that much history first (momentum: 253 trading days); ingest earlier data or accept the later start |
| `no stored prices for SPY` | The default benchmark isn't ingested: ingest it or pass `--benchmark ""` |
| `unknown parameter(s): ...` | A typo in `--param`; `fincomb strategies` lists the names |
| `buy_and_hold never rebalances` | Use `fixed_weights` with `--rebalance` |
| `lookback must be at least 60 returns` | `optimized` needs max(60, 2 × assets) returns per window |
| `verify` says NOT reproducible | Stored prices changed since the run (re-ingest or re-adjustment); rerun it to get current numbers |
