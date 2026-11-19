# Milestone 4: Portfolio construction and risk

## 1. What this milestone builds

Milestones 1 to 3 answer "how did each asset do?". This one answers "how should I
combine assets, and how risky is the combination?".

- **Four weighting methods**: equal weight, inverse volatility, minimum variance
  and maximum Sharpe, all long-only and fully invested with per-asset
  `min_weight` / `max_weight` bounds.
- **Saved portfolios** (migration `0003_portfolios`): the weights plus the exact
  window and parameters that produced them, so they can be reproduced.
- **Risk reports**: performance, 1-day historical and Gaussian VaR and CVaR,
  beta and alpha against a benchmark, and each asset's share of the risk.
- The same features from the **CLI** (`fincomb portfolio ...`) and the **API**
  (`/api/v1/portfolios`), both calling one service.

The formulas, conventions and limitations are in
[METHODOLOGY.md](METHODOLOGY.md#portfolio-construction). Read the section on
maximum Sharpe before trusting its weights.

## 2. How the components work

```
CLI (fincomb portfolio ...)   API (/api/v1/portfolios)
              \                 /
          services/portfolio_service.py   load prices, pick the window, validate,
              |            |               save, turn rows into plain records
   portfolio/construction  portfolio/analysis       pure functions, no I/O
     |            |              |
 covariance   optimizer     analytics/risk.py (VaR, CVaR, beta)
```

| Module | Role |
|---|---|
| `analytics/risk.py` | `historical_var`, `historical_cvar`, `parametric_var`, `parametric_cvar`, `beta_alpha` (new) |
| `portfolio/covariance.py` | sample covariance, Ledoit-Wolf shrinkage (same numbers as scikit-learn), smallest ridge that makes a singular matrix usable |
| `portfolio/optimizer.py` | SciPy SLSQP with analytic gradients and several starting points; every answer is checked independently (sums to 1, inside the bounds) |
| `portfolio/construction.py` | the four methods, the bounds check, the fallback, `construct()` (which the Milestone 5 backtester will call at each rebalance) |
| `portfolio/analysis.py` | constant-mix portfolio returns, risk contributions, `risk_report()` |
| `services/portfolio_service.py` | estimation window, construct, create (manual or constructed), list, get, delete, risk |
| `api/routers/portfolios.py`, `api/schemas/portfolios.py` | HTTP layer only |
| `cli.py` | `portfolio` command group, formatting only |

### Design choices worth mentioning in interviews

- **Fallback that can't hide.** If SLSQP fails or its answer fails a check, the
  weights become bounded inverse volatility, but the result says so in
  `fallback_used`, `fallback_reason` and `effective_method`, and a saved
  portfolio stores both methods.
- **No silent method switch for maximum Sharpe.** When nothing beats the
  risk-free rate the answer is still returned, flagged with
  `non_positive_excess_return` and a warning that it is not an attractive
  investment.
- **CVaR consistent with VaR.** CVaR averages the same interpolated quantile
  function that VaR reads, so fractional tails need no rounding and CVaR ≥ VaR is
  guaranteed (checked on hundreds of random series).
- **Server-side construction.** `POST /portfolios` with `construction` runs the
  optimiser on the server; a client can't save "optimised" weights it made up.
- **Tested against independent answers**: closed-form solutions, optimality
  (KKT) conditions, a convex reformulation of maximum Sharpe, and scikit-learn's
  Ledoit-Wolf values, plus property tests on random data.

## 3. Get the code

In PowerShell:

```powershell
cd D:\Projects\FinComb
.venv\Scripts\Activate.ps1
git fetch origin
git checkout milestone-4      # only needed until the Milestone 4 PR is merged into main
python -m pip install --no-cache-dir -e ".[dev]"   # adds SciPy
```

Start the database and apply the new migration:

```powershell
docker compose up -d db
alembic upgrade head          # creates portfolios and portfolio_weights
fincomb check-db              # Schema revision: 0003_portfolios
```

## 4. Run the tests

```powershell
pytest
```

Expected: **342 passed** (212 from before, plus 130 new). Without the database,
107 integration tests are skipped instead (`235 passed, 107 skipped`). Only the
portfolio and risk tests:

```powershell
pytest tests/unit/test_risk_tail_beta.py tests/unit/test_portfolio_covariance.py `
       tests/unit/test_portfolio_construction.py tests/unit/test_portfolio_analysis.py `
       tests/integration/test_portfolios.py tests/integration/test_portfolio_api.py `
       tests/integration/test_portfolio_cli.py
```

If the full suite is still slow on Windows, try `$env:POSTGRES_HOST = "127.0.0.1"`
before `pytest` (`localhost` may try IPv6 first) and `pytest --durations=10` to
see which tests take the time.

## 5. Try it from the CLI

These use the six instruments you already have (AAPL, GLD, MSFT, QQQ, SPY, TLT).

```powershell
# All four methods side by side over the last 252 trading days, 40% cap per asset
fincomb portfolio compare AAPL MSFT QQQ SPY TLT GLD --max-weight 0.4 --risk-free 0.02

# One method in detail, with each asset's share of the risk
fincomb portfolio construct AAPL MSFT QQQ SPY TLT GLD --method min-variance --max-weight 0.4

# Save it (constructed on the spot) and a manual 60/40
fincomb portfolio construct AAPL MSFT QQQ SPY TLT GLD --method min-variance `
    --max-weight 0.4 --save "Core MV" --description "min variance, 40% cap"
fincomb portfolio create "60-40" --weights SPY=0.6,TLT=0.4

fincomb portfolio list
fincomb portfolio show "Core MV"

# Risk over the whole common history, then out of sample (after the fitting window)
fincomb portfolio risk "Core MV" --benchmark SPY
fincomb portfolio risk "60-40" --start 2022-01-01 --end 2022-12-31 --confidence 0.99

# The warning case: nothing beats a 60% risk-free rate
fincomb portfolio construct AAPL MSFT SPY --method max-sharpe --risk-free 0.6

fincomb portfolio delete "60-40"
```

Options accept `min-variance` or `min_variance`, and `ledoit-wolf` or `sample`
for `--covariance`. `--start` sets an explicit window (then `--lookback` is
ignored). `--benchmark ""` skips beta.

## 6. Try it from the API

Start `fincomb serve`, then in a second PowerShell window:

```powershell
$api = "http://127.0.0.1:8000/api/v1"

# Construct (not saved)
$body = @{ symbols = @("AAPL","MSFT","QQQ","SPY","TLT","GLD"); method = "max_sharpe";
           max_weight = 0.4; risk_free = 0.02 } | ConvertTo-Json
$r = Invoke-RestMethod "$api/portfolios/construct" -Method Post -Body $body -ContentType "application/json"
$r.weights | Format-Table symbol, weight, percent_of_risk
$r.in_sample; $r.warnings

# Save a constructed portfolio, then its risk report
$body = @{ name = "API MV"; construction = @{ symbols = @("SPY","TLT","GLD"); method = "min_variance" } } |
    ConvertTo-Json -Depth 3
$p = Invoke-RestMethod "$api/portfolios" -Method Post -Body $body -ContentType "application/json"
$risk = Invoke-RestMethod "$api/portfolios/$($p.id)/risk?benchmark=SPY&confidence=0.95"
$risk.var; $risk.cvar; $risk.beta
$risk.contributions | Format-Table symbol, weight, risk_contribution, percent_of_risk

# Risk of weights you haven't saved
$body = @{ weights = @{ SPY = 0.6; TLT = 0.4 } } | ConvertTo-Json
Invoke-RestMethod "$api/portfolios/risk" -Method Post -Body $body -ContentType "application/json"

# Errors: weights that don't sum to 1 (422), a duplicate name (409)
$bad = @{ name = "bad"; weights = @{ SPY = 0.6; TLT = 0.3 } } | ConvertTo-Json
curl.exe -s -X POST "$api/portfolios" -H "Content-Type: application/json" -d $bad
```

### Things to check

- `/health` shows `"schema_revision": "0003_portfolios"`.
- In `compare`, minimum variance has the lowest volatility of the four columns
  and maximum Sharpe the highest Sharpe (both in-sample). With `--covariance
  sample` this holds exactly; with the default Ledoit-Wolf the optimisers work
  on the shrunk matrix, so a near-tie can very occasionally go the other way.
- Weights sum to 100% and respect `--max-weight` / `--min-weight`.
- `portfolio risk` and `GET /portfolios/{id}/risk` give identical numbers.
- CVaR is never below VaR; at 99% on one year of data you get a warning that
  the tail is too thin.
- The max-Sharpe warning appears above the weights in the CLI and in
  `warnings` in the API.
- `fincomb portfolio show` repeats the warning for a saved portfolio.

## 7. Endpoint summary

| Method | Path | Purpose |
|---|---|---|
| POST | `/portfolios/construct` | Weights by method; not saved |
| POST | `/portfolios` | Save manual weights or a server-side construction |
| GET | `/portfolios` | List saved portfolios |
| GET | `/portfolios/{id}` | Detail, weights and construction parameters |
| DELETE | `/portfolios/{id}` | Delete |
| GET | `/portfolios/{id}/risk` | Risk report for a saved portfolio |
| POST | `/portfolios/risk` | Risk report for unsaved weights |

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `relation "portfolios" does not exist` | Run `alembic upgrade head` |
| `ModuleNotFoundError: scipy` | Reinstall: `python -m pip install -e ".[dev]"` |
| `max_weight ... is too small for N assets` | N × max_weight must be at least 1, e.g. 0.2 or more for 5 assets |
| `need at least 60 daily returns` | Use a longer `--lookback` or an earlier `--start` |
| `no stored prices for SPY` in a risk report | The default benchmark isn't ingested: ingest it or pass `--benchmark ""` |
| Maximum Sharpe puts everything in one asset | Expected with noisy mean returns; use `--max-weight` |
