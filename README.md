# FinComb

An end-to-end financial intelligence and quantitative research platform: it ingests
and validates market data, stores it in PostgreSQL, analyses performance and risk, builds and risk-checks
portfolios, serves it all through a REST API, and (in later milestones) backtests
trading strategies without look-ahead bias.

Tech stack: Python · PostgreSQL · SQLAlchemy · Alembic · FastAPI · pandas / NumPy ·
SciPy / Statsmodels · Plotly · Streamlit · Pytest · Docker

> Status: **Milestone 4 of 7 done** (data pipeline, analytics core, REST API, and
> portfolio construction and risk). See
> the [roadmap](#roadmap).

## What works today

- Download daily OHLCV bars from **Yahoo Finance** (default) or **Tiingo** (fallback, free API key),
  or load them from CSV files, all through one pluggable `MarketDataSource` interface.
- **Data-quality validation**: missing values, non-positive prices, inconsistent
  OHLC, negative volume, duplicate and future dates are rejected; extreme moves and
  stale prices are kept but flagged.
- **Idempotent storage** in PostgreSQL: re-running an ingestion updates rows and
  never duplicates them. Database CHECK constraints are a second line of defence.
- A full **audit trail**: every run records which symbols succeeded or failed and
  every data-quality issue it found.
- **Consistent history** ([docs/DATA_INTEGRITY.md](docs/DATA_INTEGRITY.md)): each
  symbol is owned by one data source, unfinished intraday bars are never stored,
  and a vendor re-adjustment (dividend/split) triggers a full re-download instead
  of a stitched history. `fincomb check-data` audits it and `fincomb refresh`
  repairs it.
- Schema managed with **Alembic** migrations; config from environment variables only.
- **Analytics core:** returns, CAGR, volatility, Sharpe, Sortino, maximum drawdown
  and drawdown episodes, Calmar, rolling statistics and correlation matrices, as
  pure, tested functions. Conventions are documented in
  [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
- **REST API** (FastAPI): instruments, prices, ingestion runs, data audit and all
  analytics as JSON, with interactive OpenAPI docs at `/docs`, one error format and
  no business logic in the routes (they call the same services as the CLI). See
  [docs/MILESTONE_3.md](docs/MILESTONE_3.md).
- **Portfolio construction and risk:** equal weight, inverse volatility, minimum
  variance and maximum Sharpe (SciPy SLSQP, Ledoit-Wolf covariance, per-asset
  bounds, a visible fallback), saved portfolios that can be reproduced exactly,
  and risk reports with historical and Gaussian VaR/CVaR, beta/alpha and risk
  contributions, from both `fincomb portfolio ...` and `/api/v1/portfolios`. See
  [docs/MILESTONE_4.md](docs/MILESTONE_4.md).

## Quick start

You need Python 3.11+, Git and Docker Desktop. Step-by-step install instructions
for Windows, macOS and Linux are in [docs/MILESTONE_1.md](docs/MILESTONE_1.md).

```bash
git clone https://github.com/swajeehazehra/FinComb.git
cd FinComb
cp .env.example .env              # then edit POSTGRES_PASSWORD
docker compose up -d db           # start PostgreSQL
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
alembic upgrade head              # create the tables
fincomb check-db
fincomb ingest AAPL MSFT SPY --start 2015-01-01
fincomb coverage
fincomb analyze AAPL MSFT SPY --risk-free 0.04
fincomb correlation AAPL MSFT SPY
fincomb drawdowns SPY
fincomb portfolio compare AAPL MSFT SPY --max-weight 0.6
pytest
fincomb serve                     # API on http://127.0.0.1:8000, docs at /docs
```

## REST API

`fincomb serve` starts the API on `127.0.0.1:8000` (this computer only; the write
endpoints have no authentication yet). Open <http://127.0.0.1:8000/docs> to try
every endpoint in the browser. A few examples:

```bash
curl "http://127.0.0.1:8000/api/v1/health"
curl "http://127.0.0.1:8000/api/v1/instruments/AAPL"
curl "http://127.0.0.1:8000/api/v1/analytics/performance?symbols=AAPL,MSFT,SPY&risk_free=0.04"
curl "http://127.0.0.1:8000/api/v1/analytics/correlation?symbols=SPY,QQQ,TLT,GLD"
```

Errors always have the shape `{"error": {"code", "message", "details"}}`. The full
endpoint list is in [docs/ARCHITECTURE.md §4](docs/ARCHITECTURE.md#4-api-endpoints).

## Project layout

```
src/fincomb/
  config.py            settings from env vars / .env
  logging_config.py    plain or JSON logs
  db/                  SQLAlchemy models, engine, repositories (all SQL lives here)
  ingestion/           MarketDataSource + yfinance, tiingo, csv adapters, pipeline
  validation/          data-quality rules
  analytics/           returns, risk, VaR/CVaR, beta, correlation (pure functions, no I/O)
  portfolio/           covariance, optimiser, weighting methods, portfolio risk (pure)
  services/            shared by CLI and API: load prices, analytics, portfolios, ingestion
  api/                 FastAPI app factory, routers, Pydantic schemas, error mapping
  cli.py               `fincomb` command (thin: parses arguments, calls services)
migrations/            Alembic schema migrations
tests/unit/            fast tests, no database, no network
tests/integration/     tests against a real PostgreSQL test database
tests/fixtures/        small synthetic price files (not real market data)
docs/                  architecture and milestone guides
```

## Roadmap

1. ✅ Foundation and data pipeline
2. ✅ Analytics core: returns, volatility, Sharpe, drawdown, correlation
3. ✅ FastAPI backend: REST API over the data and analytics
4. ✅ Portfolio construction and risk
5. Strategies (momentum, mean reversion) and a backtest engine with costs and slippage
6. Streamlit dashboard
7. CI, documentation and polish

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data sources and disclaimer

Yahoo Finance data is accessed through the unofficial `yfinance` library and Tiingo
through its official API on the free plan. Both are meant for personal research and
education; check their terms before any other use. Nothing in this project is
investment advice.

## License

MIT, see [LICENSE](LICENSE).
