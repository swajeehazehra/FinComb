# Milestone 2: Analytics core

## 1. What this milestone builds

The quantitative engine: functions that turn stored prices into returns, risk
and correlation statistics, plus three new commands to use them from the
terminal. Every formula and convention is written down in
[METHODOLOGY.md](METHODOLOGY.md).

```
 PostgreSQL ──load_prices()──▶ wide frame of adjusted closes (dates × symbols)
                                        │
                     ┌──────────────────┼────────────────────┐
                     ▼                  ▼                    ▼
              analytics/returns   analytics/risk     analytics/correlation
              simple/log returns  volatility, Sharpe correlation matrix,
              CAGR, resampling    Sortino, drawdowns rolling correlation
                     └──────────────────┼────────────────────┘
                                        ▼
                         analytics/summary.performance_summary
                                        ▼
                  fincomb analyze · fincomb correlation · fincomb drawdowns
```

## 2. How the components work

| File | Role |
|---|---|
| `analytics/constants.py` | `TRADING_DAYS = 252`, the one place the year length is defined. |
| `analytics/_utils.py` | Input checks shared by every function: date index, sorted, no duplicates, positive prices, enough rows, no return of −100 % or worse. Bad input fails loudly instead of giving a wrong number. |
| `analytics/returns.py` | Simple and log returns, wealth index, cumulative and total return, CAGR, and resampling daily returns to weekly/monthly/yearly by compounding. |
| `analytics/risk.py` | Annualised volatility, Sharpe, Sortino, drawdown series, maximum drawdown, drawdown episodes (peak, trough, recovery, durations), Calmar, rolling volatility and rolling Sharpe. Division by a zero volatility returns NaN, never infinity. |
| `analytics/correlation.py` | Pearson or Spearman correlation matrix of returns, and rolling correlation between two assets. |
| `analytics/summary.py` | `performance_summary()`: one row of key statistics per symbol. |
| `services/market_data.py` | `load_prices()`: reads stored bars and pivots them into a dates × symbols table. By default it keeps only dates where **every** symbol traded, so all statistics cover the same days. Gaps are never forward-filled. |
| `cli.py` | New commands `analyze`, `correlation` and `drawdowns`. |

### Design choices worth mentioning in interviews

- **Pure functions.** The analytics never touch the database or the network.
  That makes them fast to test, easy to reuse in the API (Milestone 3) and the
  backtester (Milestone 5), and deterministic.
- **Conventions are explicit.** 252 days, sample standard deviation, compounded
  risk-free rate, adjusted prices. All are documented and tested, because
  "my Sharpe doesn't match Yahoo's" is almost always a convention difference.
- **No look-ahead in rolling statistics.** A test changes the future and checks
  that the past doesn't move.
- **Three kinds of tests.** Hand-computed examples (you can check them on paper),
  a cross-check against a separate NumPy implementation, and property-based tests
  that try hundreds of random price series to check invariants.

## 3. Get the code

You're on `main` with Milestone 1 merged. In PowerShell:

```powershell
cd D:\Projects\FinComb
.venv\Scripts\Activate.ps1
git fetch origin
git checkout milestone-2      # only needed until PR #2 is merged into main
pip install -e ".[dev]"       # adds the hypothesis test library
```

Make sure Docker Desktop is running and the database is up:

```powershell
docker compose up -d db
fincomb check-db
```

No database migration is needed for this milestone: nothing new is stored.

## 4. Run the tests

```powershell
pytest
```

Expected: **102 passed**. To see which lines are tested:

```powershell
pytest --cov=fincomb --cov-report=term-missing
```

## 5. Try it on real data

Get a broader set of stocks and ETFs first (this updates what you already have):

```powershell
fincomb ingest AAPL MSFT NVDA JPM XOM --start 2015-01-01
fincomb ingest SPY QQQ TLT GLD --start 2015-01-01 --asset-type etf
```

**Performance table** (with a 4 % risk-free rate):

```powershell
fincomb analyze AAPL MSFT NVDA JPM XOM SPY --start 2015-01-01 --risk-free 0.04
```

You get total return, CAGR, volatility, Sharpe, Sortino, max drawdown and Calmar
for each symbol over the same dates.

**Correlation of daily returns:**

```powershell
fincomb correlation SPY QQQ TLT GLD AAPL XOM --start 2018-01-01
fincomb correlation SPY QQQ TLT GLD --method spearman
```

Things to look for: SPY and QQQ should be highly correlated (about 0.9), and TLT
(long-term US Treasuries) much less correlated with stocks, often near zero or
negative.

**Worst drawdowns:**

```powershell
fincomb drawdowns SPY --top 5
```

The 2020 COVID crash (about −34 %, February to March 2020) should be at or near
the top, and the 2022 bear market should also appear.

## 6. Use the functions directly in Python

```powershell
python
```
```python
from fincomb.config import get_settings
from fincomb.db.base import make_engine, make_session_factory, session_scope
from fincomb.services.market_data import load_prices
from fincomb.analytics.returns import simple_returns, resample_returns
from fincomb.analytics.risk import rolling_volatility

factory = make_session_factory(make_engine(get_settings().database_url()))
with session_scope(factory) as s:
    prices = load_prices(s, ["SPY", "TLT"])

r = simple_returns(prices)
print(resample_returns(r, "yearly").round(3))  # calendar-year returns
print(rolling_volatility(r, window=63).tail())  # 3-month rolling volatility
exit()
```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'hypothesis'` when running `pytest` | Run `pip install -e ".[dev]"` again with `(.venv)` active. |
| `no stored prices for XYZ … run fincomb ingest XYZ first` | Ingest the symbol, or check its spelling with `fincomb coverage`. |
| `fewer than 2 common dates` | The symbols don't overlap in time. Check their date ranges with `fincomb coverage`, or widen `--start`/`--end`. |
| A Sharpe or Sortino shows `n/a` | Zero volatility (or no losing days for Sortino) in that window, so the ratio is undefined. |
| Numbers differ slightly from a website | Check the conventions in METHODOLOGY.md: risk-free rate, 252 days, adjusted prices and exact date range all change the result. |

If something else goes wrong, send me the full command and its full output.
