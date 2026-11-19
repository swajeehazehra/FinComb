# FinComb: Financial Intelligence & Quantitative Research Platform

**Design document, v0.3. Milestones 1 to 4 implemented** (data pipeline, analytics core, REST API, portfolio construction and risk).

Changes since v0.1: Stooq was replaced by Tiingo as the fallback source (Stooq now requires an API key and a browser check); each symbol has one owning data source ([DATA_INTEGRITY.md](DATA_INTEGRITY.md)); the API section below describes what Milestone 3 actually built.

Changes in v0.3 (Milestone 4): the portfolio tables in §2.4 are long-only and record the requested and effective method, the construction parameters and the estimation window; `rebalance_frequency` moved to Milestone 5; `portfolio/` gained `covariance.py` and `optimizer.py`; `POST /portfolios/risk` was added for unsaved weights; beta uses the closed-form OLS slope (Statsmodels arrives with Milestone 5).

Python package name: `fincomb`.

---

## 1. Architecture

### 1.1 Guiding principles

1. **Layers only depend downward.** The dashboard never touches the database, and the quant code never touches the database or HTTP. That makes the maths testable as pure functions.
2. **Pure quant core.** Everything in `analytics/`, `portfolio/`, `strategies/` and `backtest/` takes pandas objects in and returns pandas objects / dataclasses out. No I/O, no globals, no randomness without an explicit seed.
3. **Pluggable data sources.** One `MarketDataSource` interface; yfinance is the default, Tiingo (free API key) is the fallback, and a CSV adapter feeds tests so the test suite never hits the network.
4. **Reproducible by construction.** Every backtest run stores its parameters, the code version (git SHA), and a hash of the exact price data it used.
5. **Configuration from environment.** `pydantic-settings` reads `.env` / env vars. No credentials in code; `.env` is git-ignored and `.env.example` documents every variable.

### 1.2 Layer diagram

```
┌──────────────────────────────────────────────────────────────┐
│  Presentation      Streamlit dashboard  (talks HTTP only)    │
└───────────────────────────────┬──────────────────────────────┘
                                │ httpx (JSON)
┌───────────────────────────────▼──────────────────────────────┐
│  API               FastAPI routers + Pydantic schemas        │
│                    (validation, error mapping, no maths)     │
└───────────────────────────────┬──────────────────────────────┘
                                │ calls
┌───────────────────────────────▼──────────────────────────────┐
│  Services          Orchestration: load data → run quant →    │
│                    persist results. Owns DB sessions.        │
└──────┬─────────────────────────────┬─────────────────────────┘
       │                             │
┌──────▼───────────────┐   ┌─────────▼──────────────────────────┐
│ Data layer           │   │ Quant core (pure functions)        │
│  ingestion/ adapters │   │  analytics/  returns, risk, corr   │
│  validation/ rules   │   │  portfolio/  weights, optimisation │
│  db/ models + repos  │   │  strategies/ momentum, mean-rev    │
│  (SQLAlchemy 2.0)    │   │  backtest/   engine, costs, report │
└──────┬───────────────┘   └────────────────────────────────────┘
       │
┌──────▼───────────────┐
│ PostgreSQL 16        │   schema managed by Alembic migrations
└──────────────────────┘
```

### 1.3 Key technology choices

| Concern | Choice | Why |
|---|---|---|
| ORM / migrations | SQLAlchemy 2.0 + Alembic | Industry standard, typed, versioned schema |
| Config | pydantic-settings | Typed env-var config, fails fast on missing values |
| API | FastAPI + Pydantic v2 | Auto OpenAPI docs at `/docs`, request validation |
| Data source | yfinance (default), Tiingo REST API (fallback, free key) | Free; adapter pattern isolates breakage. Each symbol is owned by one source so histories are never mixed |
| Optimisation | SciPy `minimize` (SLSQP) | Constrained min-variance / max-Sharpe without extra deps |
| Statistics | NumPy closed form (Milestone 4), Statsmodels (Milestone 5) | OLS beta/alpha vs benchmark; t-statistics and the ADF test for mean reversion |
| Charts | Plotly | Same figures work in Streamlit and notebooks |
| Logging | stdlib `logging` with JSON formatter option | No extra deps; structured logs in Docker |
| Tests | Pytest, pytest-cov, Hypothesis (property tests) | Unit tests on pure core, integration tests on a real Postgres |
| Lint / types | Ruff + mypy | Cheap quality signal for reviewers |
| Packaging | `pyproject.toml`, src layout, pip | Simple for reviewers to install |
| Runtime | Docker Compose: `db`, `api`, `dashboard` | One command to run the whole thing |

Data frequency for the MVP is **daily bars** only. Intraday adds storage and calendar complexity without adding much to the portfolio story.

---

## 2. Database schema

PostgreSQL 16. Prices are stored as `DOUBLE PRECISION` (analytics run in float64 anyway; money-exact `NUMERIC` buys nothing here and slows pandas conversion).

### 2.1 Reference data

**`instruments`**

| column | type | notes |
|---|---|---|
| id | SERIAL PK | |
| symbol | VARCHAR(20) UNIQUE NOT NULL | e.g. `AAPL`, `SPY` |
| name | VARCHAR(200) | |
| asset_type | ENUM(`stock`,`etf`,`index`) | |
| exchange | VARCHAR(20) | |
| currency | CHAR(3) DEFAULT 'USD' | |
| is_active | BOOLEAN DEFAULT true | |
| data_source | VARCHAR(20) | the one source that owns this symbol's history (migration 0002) |
| created_at | TIMESTAMPTZ DEFAULT now() | |

### 2.2 Market data

**`price_bars`** (daily OHLCV)

| column | type | notes |
|---|---|---|
| instrument_id | INT FK → instruments | |
| bar_date | DATE | trading date (exchange-local) |
| open, high, low, close | DOUBLE PRECISION NOT NULL | raw (unadjusted) |
| adj_close | DOUBLE PRECISION NOT NULL | split + dividend adjusted |
| volume | BIGINT | |
| source | VARCHAR(20) NOT NULL | `yfinance`, `tiingo`, `csv` |
| ingested_at | TIMESTAMPTZ DEFAULT now() | |
| ingestion_run_id | INT FK → ingestion_runs | lineage |

- **PK `(instrument_id, bar_date)`**, which also makes re-ingestion idempotent via `INSERT … ON CONFLICT DO UPDATE`.
- CHECK constraints: `high >= greatest(open, close, low)`, `low <= least(open, close)`, prices `> 0`, `volume >= 0`.

### 2.3 Data pipeline audit

**`ingestion_runs`**: id, source, symbols TEXT[], start_date, end_date, started_at, finished_at, status ENUM(`running`,`success`,`partial`,`failed`), rows_inserted, rows_updated, rows_rejected, error_message.

**`data_quality_issues`**: id, ingestion_run_id FK, instrument_id FK, bar_date, rule (e.g. `non_positive_price`, `ohlc_inconsistent`, `extreme_return`, `stale_price`, `gap`), severity ENUM(`warning`,`error`), detail TEXT. Errors are rejected, warnings are stored and flagged.

### 2.4 Portfolios (migration 0003)

**`portfolios`**

| column | type | notes |
|---|---|---|
| id | SERIAL PK | |
| name | VARCHAR(100) UNIQUE NOT NULL | the CLI refers to portfolios by name or id |
| description | TEXT | |
| construction_method | ENUM `portfolio_method` | `manual`, `equal_weight`, `inverse_vol`, `min_variance`, `max_sharpe`: what was asked for |
| effective_method | `portfolio_method` | what produced the weights (`inverse_vol` after an optimiser fallback) |
| construction_params | JSONB NOT NULL DEFAULT '{}' | bounds, lookback, covariance method, shrinkage, ridge, risk-free rate, optimiser status, fallback reason, warnings |
| estimation_start, estimation_end | DATE NULL | first and last price date of the window (null for manual) |
| created_at | TIMESTAMPTZ DEFAULT now() | |

**`portfolio_weights`**: portfolio_id FK (ON DELETE CASCADE), instrument_id FK (ON DELETE RESTRICT), weight DOUBLE PRECISION, PK `(portfolio_id, instrument_id)`, CHECK `weight BETWEEN 0 AND 1` (long-only). The service enforces sum = 1 within 1e-6. Saved portfolios are immutable: re-running construction with the stored window and parameters reproduces the weights exactly.

### 2.5 Backtests

**`backtest_runs`**

| column | type | notes |
|---|---|---|
| id | SERIAL PK | |
| strategy | VARCHAR(50) | `momentum`, `mean_reversion`, `buy_and_hold` |
| params | JSONB | lookback, thresholds, etc. |
| symbols | TEXT[] | universe |
| benchmark_symbol | VARCHAR(20) | e.g. `SPY` |
| start_date, end_date | DATE | |
| initial_capital | DOUBLE PRECISION | |
| cost_bps, slippage_bps | DOUBLE PRECISION | |
| code_version | VARCHAR(40) | git SHA |
| data_hash | CHAR(64) | SHA-256 of the input price frame |
| metrics | JSONB | CAGR, vol, Sharpe, Sortino, MDD, turnover, alpha/beta… |
| status | ENUM(`success`,`failed`) | |
| created_at | TIMESTAMPTZ | |

**`backtest_equity`**: run_id FK, bar_date, equity, cash, gross_exposure, benchmark_equity, drawdown. PK `(run_id, bar_date)`.

**`backtest_trades`**: id, run_id FK, bar_date, instrument_id FK, quantity, fill_price, notional, commission, slippage_cost.

Computed analytics (returns, correlations) are **not** stored; they are cheap to recompute from `price_bars` and storing them risks staleness. Only backtests are persisted because they are the "experiments" a researcher wants to compare later.

---

## 3. Folder structure

```
FinComb/
├── pyproject.toml              # deps, ruff, mypy, pytest config
├── README.md
├── .env.example                # every env var documented, no secrets
├── .gitignore
├── docker-compose.yml          # db, api, dashboard
├── docker/
│   ├── api.Dockerfile
│   └── dashboard.Dockerfile
├── alembic.ini
├── migrations/                 # Alembic versions
├── docs/
│   ├── ARCHITECTURE.md         # this document
│   ├── METHODOLOGY.md          # formulas, assumptions, bias controls
│   └── adr/                    # short architecture decision records
├── src/fincomb/
│   ├── config.py               # pydantic-settings Settings
│   ├── logging_config.py
│   ├── exceptions.py           # domain errors → mapped to HTTP codes
│   ├── db/
│   │   ├── base.py             # engine, session factory
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   └── repositories.py     # InstrumentRepo, PriceRepo, BacktestRepo
│   ├── ingestion/
│   │   ├── base.py             # MarketDataSource protocol
│   │   ├── yfinance_source.py
│   │   ├── tiingo_source.py    # fallback, needs TIINGO_API_KEY
│   │   ├── csv_source.py       # offline / tests
│   │   ├── registry.py         # source by name
│   │   ├── calendar.py         # last completed trading session
│   │   ├── audit.py            # `check-data` / GET /data/audit
│   │   └── pipeline.py         # fetch → validate → upsert → audit
│   ├── validation/
│   │   └── rules.py            # data-quality rules returning issues
│   ├── analytics/
│   │   ├── returns.py          # simple/log returns, CAGR, rolling
│   │   ├── risk.py             # vol, Sharpe, Sortino, MDD, VaR, CVaR, beta
│   │   └── correlation.py
│   ├── portfolio/
│   │   ├── covariance.py       # sample, Ledoit-Wolf, positive-definite guard
│   │   ├── optimizer.py        # SLSQP wrapper with independent checks
│   │   ├── construction.py     # equal, inverse-vol, min-var, max-Sharpe, fallback
│   │   └── analysis.py         # portfolio returns, VaR/CVaR, beta, risk contribution
│   ├── strategies/
│   │   ├── base.py             # Strategy ABC: generate_weights(prices)
│   │   ├── momentum.py         # cross-sectional / time-series momentum
│   │   ├── mean_reversion.py   # z-score / Bollinger reversion
│   │   └── buy_and_hold.py
│   ├── backtest/
│   │   ├── engine.py           # vectorised daily engine
│   │   ├── costs.py            # commission + slippage models
│   │   └── report.py           # metrics + benchmark comparison
│   ├── services/               # orchestration used by API and CLI
│   │   ├── market_data.py      # load_prices (aligned wide frame), price_bars
│   │   ├── ingestion_service.py # choose source per symbol, run the pipeline
│   │   ├── analytics_service.py # load prices → analytics, with alignment notes
│   │   ├── portfolio_service.py # estimation window, construct, save, risk report
│   │   └── backtest_service.py
│   ├── api/
│   │   ├── main.py             # create_app(settings, session_factory) factory
│   │   ├── deps.py             # settings, session per request, date range, symbols
│   │   ├── errors.py           # exception → HTTP status + error JSON, one table
│   │   ├── schemas/            # Pydantic request/response models
│   │   └── routers/            # health, instruments, prices, analytics,
│   │                           # portfolios, strategies, backtests, ingestion
│   └── cli.py                  # `fincomb ingest`, `fincomb serve`, … (thin: calls services)
├── dashboard/
│   ├── app.py                  # Streamlit entry
│   ├── api_client.py           # thin httpx wrapper
│   ├── charts.py               # Plotly figure builders
│   └── pages/                  # Market, Performance, Correlation,
│                               # Portfolio, Backtest
└── tests/
    ├── conftest.py
    ├── fixtures/               # small CSV price files (checked in)
    ├── unit/                   # pure quant + validation tests
    └── integration/            # Postgres + API tests
```

---

## 4. API endpoints

Base path `/api/v1`. All responses JSON. Interactive docs at `/docs` (Swagger UI) and `/redoc`; the schema is at `/api/v1/openapi.json`. Run locally with `fincomb serve` (binds to 127.0.0.1 by default).

**Conventions**

- Dates are ISO `YYYY-MM-DD`; `start`/`end` are inclusive and `start > end` is a 422.
- `symbols` is comma-separated (`symbols=AAPL,MSFT`) or repeated (`symbols=AAPL&symbols=MSFT`), at most 50 per analytics request.
- Multi-symbol analytics use **inner alignment** by default (dates every symbol has). Every analytics response has a `notes` list explaining when that shortened the period; `align=outer` analyses each symbol over its own history.
- Statistics that are undefined (e.g. Sharpe with zero volatility) are `null`; JSON never contains NaN.
- Time series are column-oriented: `{"dates": [...], "series": {"AAPL": [...]}}`.
- Routes are synchronous (`def`): SQLAlchemy is synchronous, so FastAPI runs them in its thread pool.

**Errors** always look like `{"error": {"code": "...", "message": "...", "details": {...}}}`. Domain exceptions are mapped in one table (`api/errors.py`):

| Exception / case | HTTP | code |
|---|---|---|
| `SymbolNotFoundError` (no stored prices) | 404 | `symbol_not_found` |
| unknown instrument / run, unknown route | 404 | `not_found` |
| symbol already registered | 409 | `already_exists` |
| `DataOwnershipError` | 409 | `data_ownership_conflict` |
| bad query/body, `ValueError` from analytics | 422 | `invalid_request` |
| `InsufficientDataError` / `MissingDataError` | 422 | `insufficient_data` / `missing_data` |
| `DataSourceError` / `DataSourceAuthError` | 502 / 503 | `data_source_error` / `data_source_auth` |
| database unreachable | 503 | `database_unavailable` |
| anything else | 500 | `internal_error` (details only in the server log) |

### System
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + DB connectivity + Alembic revision (503 if the DB is unreachable) |

### Instruments & data
| Method | Path | Purpose |
|---|---|---|
| GET | `/instruments` | List (filter `asset_type`, `active`) |
| POST | `/instruments` | Register a symbol `{symbol, asset_type, name}` without downloading (201; 409 if it exists) |
| GET | `/instruments/{symbol}` | Detail + owning source + coverage (first/last date, bar count) |
| PATCH | `/instruments/{symbol}` | Change `asset_type` (same as `fincomb set-type`) |
| GET | `/prices/{symbol}` | Bars; query `start`, `end`, `adjusted` (true = OHLC scaled by `adj_close / close`) |
| POST | `/ingestion/runs` | Ingest `{symbols, start, end, source, mode, asset_type}` synchronously; returns one run summary per source used (201) |
| GET | `/ingestion/runs` | Recent runs, newest first (`limit`, default 20) |
| GET | `/ingestion/runs/{id}` | Run detail incl. data-quality issues |
| GET | `/data/audit` | Same checks as `fincomb check-data` |

`POST /ingestion/runs` follows the pipeline's rules exactly: each symbol keeps its owning source unless `source` is given; `mode=refresh` replaces the whole history from one source (and is how ownership is switched); the end date is capped to the last completed session (`end_capped_to`). One bad symbol never fails the others, so a per-symbol problem such as an ownership conflict is reported in `results[].error` and the run `status` (`partial` / `failed`), not as an HTTP error. A request may name at most 20 symbols, because downloads run inside the request.

### Analytics
| Method | Path | Purpose |
|---|---|---|
| GET | `/analytics/performance` | `symbols`, `start`, `end`, `risk_free`, `align` → total return, CAGR, ann. vol, Sharpe, Sortino, max drawdown, Calmar, best/worst day per symbol |
| GET | `/analytics/returns` | Return series (`frequency=daily|weekly|monthly|yearly`, `cumulative=bool`, `align`) |
| GET | `/analytics/drawdown/{symbol}` | Drawdown series + worst `top` episodes |
| GET | `/analytics/rolling` | Rolling vol / Sharpe (`metric`, `window`, default 63 / 252 days) |
| GET | `/analytics/correlation` | Correlation matrix (`method=pearson|spearman`, `align`) |

### Portfolios
| Method | Path | Purpose |
|---|---|---|
| POST | `/portfolios/construct` | Weights for `{symbols, method, start?, end?, lookback, min_weight, max_weight, covariance, risk_free}`, with in-sample stats and risk contributions; not saved |
| POST | `/portfolios` | Save `{name, description?, weights}` (manual) or `{name, description?, construction}` (computed on the server); 201, 409 if the name exists |
| GET | `/portfolios` / `/portfolios/{id}` | List / detail with weights and construction parameters |
| DELETE | `/portfolios/{id}` | Remove (204) |
| GET | `/portfolios/{id}/risk` | `start, end, benchmark=SPY, confidence=0.95, risk_free`: performance, 1-day historical + Gaussian VaR/CVaR, beta/alpha/R², risk contribution per asset |
| POST | `/portfolios/risk` | The same report for unsaved `{weights, ...}` |

Every portfolio response has a `warnings` list, separate from the alignment `notes`: an optimiser fallback, a covariance ridge, a max-Sharpe portfolio with no positive excess return, or too few observations in the VaR tail. The CLI prints them as a highlighted `WARNING:` block above the numbers.

### Strategies & backtests
| Method | Path | Purpose |
|---|---|---|
| GET | `/strategies` | Available strategies + JSON schema of their parameters |
| POST | `/backtests` | Run `{strategy, params, symbols, benchmark, start, end, initial_capital, cost_bps, slippage_bps}`; persisted, returns id + metrics |
| GET | `/backtests` | List past runs (compare experiments) |
| GET | `/backtests/{id}` | Run detail + metrics + benchmark comparison |
| GET | `/backtests/{id}/equity` | Equity, benchmark and drawdown curves |
| GET | `/backtests/{id}/trades` | Trade log |

Backtests run **synchronously** in the MVP: daily data on tens of symbols over 10 to 20 years takes well under a second vectorised. A job queue is listed as a stretch goal, not built.

---

## 5. Quant methodology (summary; full version goes in `METHODOLOGY.md`)

- **Returns:** simple returns from `adj_close`; log returns where additivity is needed.
- **Annualisation:** 252 trading days. Vol = std(daily) × √252. Sharpe = mean(excess daily) / std(daily) × √252, risk-free rate as an annual input converted to daily.
- **Max drawdown:** min of `equity / equity.cummax() − 1`.
- **Portfolio construction:** equal weight; inverse volatility; minimum variance and maximum Sharpe via SLSQP with long-only, fully-invested, per-asset min/max bounds. Covariance from a trailing window with Ledoit-Wolf shrinkage by default; bounded inverse volatility is the visible fallback. Details in METHODOLOGY.md.
- **Portfolio risk:** constant-mix returns; 1-day historical VaR (linear-interpolation quantile) and CVaR (the average of the same quantile function over the tail, so CVaR ≥ VaR), Gaussian VaR/CVaR alongside; closed-form OLS beta and Jensen's alpha; Euler risk contributions.
- **Momentum:** rank by trailing 12-1 month return (skip last month), hold top-N equal weight, monthly rebalance. Time-series variant: long if trailing return > 0.
- **Mean reversion:** z-score of price vs rolling mean; enter when z < −k, exit when z crosses 0. Optional ADF test (Statsmodels) reported as a diagnostic.
- **Benchmark comparison:** excess return, tracking error, information ratio, OLS alpha/beta (Statsmodels).

### 5.1 No look-ahead bias: how it is enforced

1. Strategies receive prices and return **target weights indexed by the date the signal was formed** (using data up to and including close *t*).
2. The engine **shifts weights by one bar**: weights decided at close *t* earn the return from *t* to *t+1*. This lives in exactly one place (`engine.py`) so no strategy can bypass it.
3. Rolling windows use only trailing data (`rolling(...)` with no `center=True`; no `bfill`).
4. Portfolio optimisation inside a backtest uses only the trailing covariance window at each rebalance date.
5. **Test that proves it:** a "future perturbation" test randomises all prices after date *T* and asserts every weight and equity value up to *T* is unchanged. Run for every strategy.

### 5.2 Costs and slippage

- Turnover at each rebalance = Σ|w_new − w_drifted|.
- Commission cost = turnover × `cost_bps` / 10 000.
- Slippage cost = turnover × `slippage_bps` / 10 000 (fixed-bps model in MVP; volume-based model is a stretch).
- Costs are deducted from equity on the rebalance bar. A test checks that a zero-cost run matches the analytical gross return, and that costs scale linearly with turnover.

---

## 6. Milestones

Each milestone ends with: code, explanation, exact commands, tests passing, and your sign-off before the next one.

| # | Milestone | Deliverables | "Done" means |
|---|---|---|---|
| **1** ✅ | **Foundation + data pipeline** | Repo scaffold, `pyproject.toml`, config, logging, Docker Compose Postgres, SQLAlchemy models, first Alembic migration, `MarketDataSource` + yfinance/Tiingo/CSV adapters (Stooq dropped), validation rules, idempotent upsert, ingestion audit, `fincomb ingest` CLI | `docker compose up db` + `fincomb ingest AAPL MSFT SPY --start 2015-01-01` fills Postgres; re-running inserts no duplicates; unit tests for validation, integration test for upsert |
| **2** ✅ | **Analytics core** | `returns.py`, `risk.py`, `correlation.py` as pure functions | Metrics match hand-computed fixtures and known values; property tests (e.g. MDD ∈ [−1, 0]) |
| **3** ✅ | **FastAPI backend** | App factory, error handling, routers for health/instruments/prices/ingestion/analytics, Pydantic schemas | OpenAPI docs live; integration tests via `TestClient` against test Postgres |
| **4** ✅ | **Portfolio construction & risk** | Weight methods, risk metrics, portfolio endpoints | Weights sum to 1, respect bounds; min-var ≤ equal-weight variance on fixture data |
| 5 | Strategies + backtest engine | Strategy ABC, momentum, mean reversion, buy-and-hold, engine with costs/slippage, benchmark report, persisted runs, backtest endpoints | Look-ahead perturbation test passes; buy-and-hold matches benchmark return exactly with zero costs |
| 6 | Streamlit dashboard | Pages: Market data, Performance, Correlation heatmap, Portfolio builder, Backtest lab | Runs in Docker against the API; no direct DB access |
| 7 | Polish for portfolio | Full Docker Compose, GitHub Actions CI (ruff, mypy, pytest + Postgres service), README with screenshots, METHODOLOGY.md, seed script, coverage badge | Fresh clone → `docker compose up` → working demo in one command |

Stretch (only if time allows): async job queue for long backtests, walk-forward / parameter sweep with out-of-sample split, volume-based slippage, factor exposures, auth on write endpoints.

---

## 7. Technical risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **yfinance is unofficial and breaks / rate-limits** | Ingestion fails, demos break | Adapter interface + Tiingo fallback; retry with backoff; checked-in CSV fixtures so tests and demos never need the network |
| **Look-ahead bias** | Backtests look unrealistically good, which reviewers will spot | Single shift point in engine; perturbation test per strategy; documented in METHODOLOGY.md |
| **Survivorship bias** | Universe of today's tickers overstates past returns | Acknowledge explicitly in docs; let users pick universes; delisted-data is out of scope for a free source |
| **Adjusted vs raw prices** | Splits/dividends distort returns | Store both; analytics use `adj_close`; validation flags >40% one-day moves for review |
| **Misaligned calendars** (ETFs vs stocks, holidays, missing bars) | Wrong correlations, NaN propagation | Align on inner-joined trading dates; explicit, documented missing-data policy (no forward-fill beyond N days) |
| **Metric convention ambiguity** (252 vs 365, arithmetic vs geometric) | Numbers differ from other tools, credibility hit | One constants module; conventions written down; tests against known reference values |
| **Optimiser instability** (singular covariance, non-convergence) | Bad or crashing weights | Minimum history check (max(60, 2N) returns), Ledoit-Wolf shrinkage, smallest ridge that makes the covariance positive definite, independent check of every solution, fall back to bounded inverse-vol with a visible warning |
| **Overfitting / data snooping** | Strategy results not meaningful | Report in-sample vs out-of-sample split (stretch: walk-forward); keep strategies simple |
| **Flaky integration tests** | CI noise | Tests use fixture CSVs; Postgres via Compose service in CI; each test in a rolled-back transaction |
| **Docker friction on your machine** (Windows/Apple Silicon) | Setup time | Multi-arch slim images; app also runs locally against Dockerised Postgres only |
| **Scope creep** | Nothing finished | Milestone gates; stretch list kept separate |

---
