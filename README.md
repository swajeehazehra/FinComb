# FinComb

An end-to-end financial intelligence and quantitative research platform: it ingests
and validates market data, stores it in PostgreSQL, and (in later milestones) analyses
performance and risk, optimises portfolios and backtests trading strategies without
look-ahead bias.

Tech stack: Python · PostgreSQL · SQLAlchemy · Alembic · FastAPI · pandas / NumPy ·
SciPy / Statsmodels · Plotly · Streamlit · Pytest · Docker

> Status: **Milestone 1 of 7 done** (foundation and data pipeline). See the
> [roadmap](#roadmap).

## What works today

- Download daily OHLCV bars from **Yahoo Finance** (default) or **Stooq** (fallback),
  or load them from CSV files, all through one pluggable `MarketDataSource` interface.
- **Data-quality validation**: missing values, non-positive prices, inconsistent
  OHLC, negative volume, duplicate and future dates are rejected; extreme moves and
  stale prices are kept but flagged.
- **Idempotent storage** in PostgreSQL: re-running an ingestion updates rows and
  never duplicates them. Database CHECK constraints are a second line of defence.
- A full **audit trail**: every run records which symbols succeeded or failed and
  every data-quality issue it found.
- Schema managed with **Alembic** migrations; config from environment variables only.

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
pytest
```

## Project layout

```
src/fincomb/
  config.py            settings from env vars / .env
  logging_config.py    plain or JSON logs
  db/                  SQLAlchemy models, engine, repositories (all SQL lives here)
  ingestion/           MarketDataSource + yfinance, stooq, csv adapters, pipeline
  validation/          data-quality rules
  cli.py               `fincomb` command
migrations/            Alembic schema migrations
tests/unit/            fast tests, no database, no network
tests/integration/     tests against a real PostgreSQL test database
tests/fixtures/        small synthetic price files (not real market data)
docs/                  architecture and milestone guides
```

## Roadmap

1. ✅ Foundation and data pipeline
2. Analytics core: returns, volatility, Sharpe, drawdown, correlation
3. FastAPI backend
4. Portfolio construction and risk
5. Strategies (momentum, mean reversion) and a backtest engine with costs and slippage
6. Streamlit dashboard
7. CI, documentation and polish

Full design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data sources and disclaimer

Yahoo Finance data is accessed through the unofficial `yfinance` library and Stooq
through its public CSV endpoint. Both are free and meant for personal research and
education; check their terms before any other use. Nothing in this project is
investment advice.

## License

MIT, see [LICENSE](LICENSE).
