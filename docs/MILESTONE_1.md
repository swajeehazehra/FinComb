# Milestone 1: Foundation and data pipeline

## 1. What this milestone builds

The base everything else stands on: a Python package, configuration, logging, a
PostgreSQL database in Docker with a migrated schema, and a pipeline that downloads
daily prices, checks them, and stores them.

```
 fincomb ingest AAPL SPY
          │
          ▼
 ┌──────────────────┐   fetch_daily()   ┌──────────────────────────────┐
 │ MarketDataSource │ ────────────────▶ │ normalise(): one standard    │
 │ yfinance / stooq │                   │ frame for every provider     │
 │ / csv            │                   └──────────────┬───────────────┘
 └──────────────────┘                                  ▼
                                         ┌──────────────────────────────┐
                                         │ validate_bars(): reject bad  │
                                         │ rows, flag suspicious ones   │
                                         └──────────────┬───────────────┘
                                                        ▼
                                         ┌──────────────────────────────┐
                                         │ PriceRepository.upsert_bars  │
                                         │ INSERT ... ON CONFLICT UPDATE│
                                         └──────────────┬───────────────┘
                                                        ▼
                                  price_bars · instruments · ingestion_runs ·
                                  data_quality_issues   (PostgreSQL)
```

## 2. How the components work

| File | Role |
|---|---|
| `config.py` | `Settings` reads `.env` and environment variables with pydantic. The DB password has **no default**, so the app refuses to start without one. No credentials appear in code. |
| `logging_config.py` | One call sets up logging. `LOG_JSON=true` switches to one JSON object per line, which is what log tools in Docker expect. |
| `exceptions.py` | Domain errors (`DataSourceError`, `SymbolNotFoundError`, …). The pipeline catches these per symbol, and Milestone 3 will map them to HTTP status codes. |
| `db/models.py` | SQLAlchemy 2.0 typed models for the four tables. Postgres enums for asset type, run status and severity. CHECK constraints reject impossible bars (for example high < low) even if a bug let one through validation. |
| `db/repositories.py` | All SQL lives here. `upsert_bars` uses `INSERT … ON CONFLICT (instrument_id, bar_date) DO UPDATE`, so ingesting the same data twice changes nothing. It also reports inserted vs updated counts using Postgres's `xmax = 0` trick, and splits big batches into chunks of 1,000 rows to stay under Postgres's 65,535-parameter limit. |
| `ingestion/base.py` | The `MarketDataSource` abstract class. Each adapter only implements `_fetch_raw`; the base class normalises column names, time zones, sort order and dtypes, and filters to the requested dates. `with_retries` adds exponential backoff (but never retries "symbol not found"). |
| `ingestion/yfinance_source.py` | Yahoo via `yfinance`, with `auto_adjust=False` so both raw OHLC and adjusted close are stored. yfinance's `end` is exclusive, so one day is added to make our ranges inclusive. |
| `ingestion/stooq_source.py` | Fallback. Fetches Stooq's CSV directly. Stooq prices are already adjusted, so `adj_close = close` for this source. The HTTP function is injectable so tests never use the network. |
| `ingestion/csv_source.py` | Reads `SYMBOL.csv` from a folder. Used by tests and for offline demos. |
| `validation/rules.py` | Pure function: takes a frame, returns `clean` rows plus a list of `Issue`s. Errors reject the row; warnings keep it but record it. |
| `ingestion/pipeline.py` | Orchestrates one run. Each symbol gets its own transaction, so one bad ticker doesn't lose the others. The run ends as `success`, `partial` or `failed`. |
| `cli.py` | The `fincomb` command: `check-db`, `ingest`, `coverage`, `show`. |
| `migrations/` | Alembic. `0001_market_data` creates the schema. The DB URL comes from `.env`, not `alembic.ini`. |

### Validation rules

| Rule | Severity | Why it matters |
|---|---|---|
| `duplicate_date` | error | Two bars for one day would double-count returns |
| `missing_value` | error | NaN prices break every calculation downstream |
| `non_positive_price` | error | Prices ≤ 0 are impossible for stocks/ETFs |
| `negative_volume` | error | Impossible |
| `ohlc_inconsistent` | error | High must be ≥ open, close and low; low ≤ open and close |
| `future_date` | error | A bar dated after today is a feed bug, and a look-ahead risk |
| `extreme_return` | warning | A daily move above 40 % is often an unadjusted split |
| `stale_price` | warning | 5+ identical closes in a row suggests a halted or broken feed |

### Design choices worth mentioning in interviews

- **Validation is separate from storage.** Rules are pure functions, so they are
  tested without a database, and the DB constraints act as a safety net.
- **Idempotent ingestion.** You can re-run any ingestion safely; overlapping date
  ranges only add the new days.
- **Lineage.** Every bar records the source and the ingestion run that wrote it.
- **The test suite never touches the internet.** Network adapters are tested with
  fakes, and the pipeline is tested with small synthetic CSV files.

## 3. One-time setup on your computer

You need four things: **Git**, **Python 3.11 or newer**, **Docker Desktop**, and a
code editor (VS Code is a good free choice: https://code.visualstudio.com).
PostgreSQL itself runs inside Docker, so you don't install it separately.

### Windows 10/11

1. **Git:** download from https://git-scm.com/download/win and install with the defaults.
2. **Python:** download Python 3.12 from https://www.python.org/downloads/windows/.
   On the first installer screen, **tick "Add python.exe to PATH"**, then Install.
3. **Docker Desktop:** download from https://www.docker.com/products/docker-desktop/,
   install it, keep "Use WSL 2" ticked, and restart if asked. Open Docker Desktop
   once and wait until it says "Engine running".
4. Open **PowerShell** and check everything:
   ```powershell
   git --version
   python --version
   docker --version
   docker compose version
   ```
5. Allow the virtual-environment activation script to run (once):
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

### macOS

1. Install Homebrew (a package manager) from https://brew.sh, following the
   instructions it prints at the end.
2. In Terminal:
   ```bash
   brew install git python@3.12
   brew install --cask docker
   ```
3. Open **Docker** from Applications once and wait until it says "Engine running".
4. Check: `git --version`, `python3 --version`, `docker compose version`.

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER     # then log out and back in
docker compose version
```

## 4. Run Milestone 1

### Windows: every command in order (PowerShell)

Open PowerShell in the folder where you keep projects (e.g. `cd ~\Documents`), make
sure Docker Desktop says "Engine running", then:

```powershell
git clone https://github.com/swajeehazehra/FinComb.git
cd FinComb
copy .env.example .env
notepad .env                      # change POSTGRES_PASSWORD, save, close
docker compose up -d db
docker compose ps                 # wait until STATUS shows "healthy"
python -m venv .venv
.venv\Scripts\Activate.ps1        # prompt now starts with (.venv)
python -m pip install --upgrade pip
pip install -e ".[dev]"
alembic upgrade head
fincomb check-db
pytest
fincomb ingest AAPL MSFT NVDA JPM --start 2015-01-01
fincomb ingest SPY QQQ TLT GLD --start 2015-01-01 --asset-type etf
fincomb coverage
```

The sections below explain each step and the macOS/Linux equivalents.

In the commands below, `python` means `python3` on macOS/Linux.

### 4.1 Get the code

```bash
git clone https://github.com/swajeehazehra/FinComb.git
cd FinComb
```

### 4.2 Create your `.env`

```bash
cp .env.example .env          # Windows PowerShell: copy .env.example .env
```

Open `.env` in your editor and change `POSTGRES_PASSWORD=change-me` to a password
of your choice. `.env` is git-ignored, so it never gets committed.

### 4.3 Start PostgreSQL

```bash
docker compose up -d db
docker compose ps              # STATUS should become "healthy" after a few seconds
```

### 4.4 Create the Python environment

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Then, on any OS (your prompt now starts with `(.venv)`):
```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

You need to activate `.venv` again each time you open a new terminal.

### 4.5 Create the tables and check the connection

```bash
alembic upgrade head
fincomb check-db
```
Expected:
```
Connected: PostgreSQL 16.x ...
Schema revision: 0001_market_data
```

### 4.6 Run the tests

```bash
pytest
```
Expected: `41 passed`. If you see `30 passed, 11 skipped`, the integration tests
could not reach the test database (see troubleshooting below).

With a coverage report:
```bash
pytest --cov=fincomb --cov-report=term-missing
```

### 4.7 Ingest real data

```bash
fincomb ingest AAPL MSFT NVDA JPM --start 2015-01-01
fincomb ingest SPY QQQ TLT GLD --start 2015-01-01 --asset-type etf
fincomb coverage
fincomb show SPY --tail 5
```

Run the first command again: you should see `0 inserted` and roughly 2,500
`updated` per symbol, which proves ingestion is idempotent.

Try the fallback source and the offline demo data:
```bash
fincomb ingest AAPL --start 2024-01-01 --source stooq
fincomb ingest SYNA SYNB SYNM BADDATA --source csv --start 2023-01-01 --end 2023-12-31
```
`BADDATA` is a deliberately broken file: the run shows rows rejected and warnings.

### 4.8 Look inside the database (optional)

```bash
docker compose exec db psql -U fincomb -d fincomb
```
```sql
\dt
SELECT id, source, status, rows_inserted, rows_updated, rows_rejected FROM ingestion_runs;
SELECT rule, severity, count(*) FROM data_quality_issues GROUP BY 1, 2;
\q
```

### 4.9 Stop / reset

```bash
docker compose down        # stop (data is kept in a Docker volume)
docker compose down -v     # stop AND delete all stored data
```

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `fincomb: command not found` / `not recognized` | The venv isn't active. Run the activate command from 4.4 again. |
| `Field required … postgres_password` | `.env` is missing or you're not in the `FinComb` folder. |
| `Cannot connect to Postgres … Connection refused` | Docker Desktop isn't running, or the container isn't up: `docker compose up -d db`, then `docker compose ps`. |
| `password authentication failed` | You changed the password in `.env` after the DB was first created. Either put the old password back, or reset with `docker compose down -v` and `docker compose up -d db`. |
| `port is already allocated` / `address already in use` on 5432 | Another Postgres is running. Set `POSTGRES_PORT=5433` in `.env` and rerun `docker compose up -d db`. |
| Integration tests are **skipped** with `database "fincomb_test" does not exist` | The test DB is only created when the volume is first made. Run `docker compose exec db createdb -U fincomb fincomb_test`. |
| `yfinance returned no data for …` | Wrong ticker, no internet, or Yahoo is rate-limiting. Wait a minute, or use `--source stooq`. |
| Windows: `running scripts is disabled on this system` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. |
| Windows: `docker/initdb/...sh: not found` in `docker compose logs db` | The script got Windows line endings. The repo's `.gitattributes` prevents this on a fresh clone; if it happened, delete the folder, clone it again, then run `docker compose down -v` and `docker compose up -d db`. |

If something else goes wrong, send me the full command and its full output.
