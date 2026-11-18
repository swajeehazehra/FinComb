# Milestone 3: REST API (FastAPI)

## 1. What this milestone builds

A REST API over everything Milestones 1 and 2 built: instruments, stored prices,
ingestion runs, the data-integrity audit and the analytics. It adds no new
maths and no new tables. The routes are thin: they check the input, call the same
services the CLI uses, and turn the result into JSON.

```
 HTTP request ──▶ router (api/routers/*)        validate query/body (Pydantic)
                     │
                     ▼
                  services/                      shared with the CLI
                  ├─ analytics_service ──▶ load_prices ──▶ analytics/ (pure functions)
                  ├─ ingestion_service ──▶ ingestion/pipeline (ownership, capping, audit)
                  └─ market_data.price_bars
                     │
                     ▼
                  db/repositories ──▶ PostgreSQL
                     │
 HTTP response ◀── schemas (api/schemas/*)       NaN → null, one error shape
```

## 2. How the components work

| File | Role |
|---|---|
| `api/main.py` | `create_app(settings=None, session_factory=None)`: builds the FastAPI app, mounts every router under `/api/v1`, installs the error handlers. With no arguments it reads `.env` like the CLI; tests pass the test database instead. |
| `api/deps.py` | Dependencies: one database session per request (commit on success, roll back on error), the settings, a `start`/`end` date range that rejects `start > end`, and the `symbols` list (comma-separated or repeated, at most 50). |
| `api/errors.py` | One table that maps exceptions to HTTP status codes and error codes, and the handlers that always answer `{"error": {"code", "message", "details"}}`. Unexpected errors return a generic 500; the traceback goes to the server log only. |
| `api/schemas/` | Pydantic models for every request and response. They also convert pandas results: undefined statistics (NaN) become `null`, time series become `{"dates": [...], "series": {...}}`. |
| `api/routers/` | `health`, `instruments`, `prices`, `ingestion` (runs and `/data/audit`), `analytics`. |
| `services/analytics_service.py` | New. "Load prices, then call the analytics" for performance, returns, rolling, correlation and drawdowns, returning the alignment notes with every result. The CLI's `analyze`, `correlation` and `drawdowns` now use it too. |
| `services/ingestion_service.py` | New. Chooses the source for each symbol (explicit source, else the symbol's owner, else `DEFAULT_DATA_SOURCE`) and runs the pipeline once per source. Moved out of `cli.py` unchanged, so `ingest` and `refresh` behave exactly as before. |
| `services/market_data.py` | Adds `price_bars()` for `GET /prices/{symbol}`, including the optional adjusted OHLC. |
| `db/repositories.py` | Adds `list_instruments(asset_type, active)`, `IngestionRunRepository.recent(limit)` and `issues_with_symbols(run_id)`. All SQL still lives here. |
| `cli.py` | New `fincomb serve` command. Existing commands now call the services; their output is unchanged. |

### Design choices worth mentioning in interviews

- **No business logic in the API.** Routers never compute a statistic or build
  a query. The CLI and the API share one service layer, so they can't disagree.
  The integration tests check this directly: every API number equals the
  analytics function called on the same data.
- **Injectable app factory.** `create_app` takes its settings and session factory
  as arguments, so the tests run against `fincomb_test` and never touch your
  development data. There is no global app object created at import time.
- **One error format.** Clients only ever parse one error shape. The mapping
  from domain exceptions to HTTP codes is one table in `api/errors.py`.
- **Honest numbers.** JSON has no NaN, so undefined statistics are `null`, never 0.
  The missing-data policy from Milestone 2 still holds: compounding across a gap
  returns a 422 `missing_data` instead of a made-up number.
- **Ingestion keeps its rules.** `POST /ingestion/runs` calls the same pipeline as
  `fincomb ingest`: one owning source per symbol, completed bars only, and a full
  re-download when the vendor re-adjusts. One bad symbol doesn't fail the others,
  so per-symbol problems (an ownership conflict, an unknown ticker) are reported
  in the run's `results`, not as an HTTP error.
- **Safe defaults.** `fincomb serve` listens on `127.0.0.1` only, because the write
  endpoints have no authentication yet. Ingestion requests are capped at 20
  symbols because the download runs inside the request.
- **Synchronous routes on purpose.** SQLAlchemy is used synchronously, so routes
  are plain `def`. FastAPI runs them in a thread pool, so one slow download
  doesn't block other requests.

## 3. Get the code

In PowerShell:

```powershell
cd D:\Projects\FinComb
.venv\Scripts\Activate.ps1
git fetch origin
git checkout milestone-3      # only needed until the Milestone 3 PR is merged into main
pip install -e ".[dev]"       # adds fastapi, uvicorn and httpx
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

Expected: **212 passed** (147 from before, plus 65 new API and service tests).
If the database isn't running, 65 integration tests are skipped instead
(`147 passed, 65 skipped`). To run only the API tests:

```powershell
pytest tests/unit/test_api_app.py tests/integration/test_api.py -v
```

## 5. Run the API

```powershell
fincomb serve
```

You should see `Uvicorn running on http://127.0.0.1:8000`. Leave that window
open, then open **<http://127.0.0.1:8000/docs>** in your browser. Every endpoint
is listed there; click one, then **Try it out** and **Execute**.

Use `fincomb serve --reload` while editing code (it restarts on every save) and
`fincomb serve --port 8001` if port 8000 is taken. Stop it with `Ctrl+C`.

### Try it from a second PowerShell window

PowerShell's `Invoke-RestMethod` parses the JSON for you:

```powershell
$api = "http://127.0.0.1:8000/api/v1"

Invoke-RestMethod "$api/health"
Invoke-RestMethod "$api/instruments" | Format-Table symbol, asset_type, data_source
Invoke-RestMethod "$api/instruments/AAPL" | Select-Object -ExpandProperty coverage
(Invoke-RestMethod "$api/prices/SPY?start=2026-11-01").bars | Format-Table

# Performance table (same numbers as `fincomb analyze`)
(Invoke-RestMethod "$api/analytics/performance?symbols=AAPL,MSFT,SPY&risk_free=0.04").rows |
    Format-Table symbol, start, end, total_return, cagr, sharpe, max_drawdown

# Correlation, monthly returns, rolling volatility, drawdowns
Invoke-RestMethod "$api/analytics/correlation?symbols=SPY,QQQ,TLT,GLD"
Invoke-RestMethod "$api/analytics/returns?symbols=SPY&frequency=monthly&start=2026-01-01"
Invoke-RestMethod "$api/analytics/rolling?symbols=SPY&metric=volatility&window=63"
(Invoke-RestMethod "$api/analytics/drawdown/SPY?top=3").episodes | Format-Table

# Data audit (same as `fincomb check-data`) and recent ingestion runs
Invoke-RestMethod "$api/data/audit" | Select-Object -ExpandProperty rows | Format-Table
Invoke-RestMethod "$api/ingestion/runs?limit=5" | Format-Table id, source, status, rows_inserted
```

Update prices through the API (same as `fincomb ingest SPY QQQ`):

```powershell
$body = @{ symbols = @("SPY", "QQQ") } | ConvertTo-Json
(Invoke-RestMethod "$api/ingestion/runs" -Method Post -Body $body -ContentType "application/json").runs
```

See an error response (an unknown symbol gives a 404):

```powershell
curl.exe -s "$api/analytics/performance?symbols=NOPE"
```

Use `curl.exe`, not `curl`: in Windows PowerShell, `curl` is an alias for
`Invoke-WebRequest`.

### Things to check

- `/health` shows `"database": "ok"` and `"schema_revision": "0002_instrument_data_source"`.
- The performance numbers match `fincomb analyze AAPL MSFT SPY --risk-free 0.04`.
- Asking for a symbol with a shorter history fills `notes`, for example
  `symbols=SPY,QQQM` if you have ingested QQQM.
- `POST /ingestion/runs` with `"source": "tiingo"` for a symbol owned by yfinance
  returns status `failed`, and the error in `results` tells you to use refresh.

## 6. Endpoint summary

All under `/api/v1`. Full details are in [ARCHITECTURE.md §4](ARCHITECTURE.md#4-api-endpoints)
and in `/docs`.

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | DB connectivity and migration version (503 if the DB is down) |
| GET / POST | `/instruments` | List (filter `asset_type`, `active`) / register a symbol |
| GET / PATCH | `/instruments/{symbol}` | Detail and coverage / change asset type |
| GET | `/prices/{symbol}` | Daily bars (`start`, `end`, `adjusted`) |
| POST / GET | `/ingestion/runs` | Ingest or refresh symbols / recent runs |
| GET | `/ingestion/runs/{id}` | One run with its data-quality issues |
| GET | `/data/audit` | Source and jump checks per symbol |
| GET | `/analytics/performance` | Return, risk and drawdown statistics per symbol |
| GET | `/analytics/returns` | Daily/weekly/monthly/yearly, optional cumulative |
| GET | `/analytics/rolling` | Rolling volatility or Sharpe |
| GET | `/analytics/correlation` | Pearson or Spearman matrix |
| GET | `/analytics/drawdown/{symbol}` | Drawdown series and deepest episodes |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `fincomb: The term 'fincomb' is not recognized` or `No such command 'serve'` | Activate the venv (`.venv\Scripts\Activate.ps1`) and run `pip install -e ".[dev]"` again. |
| `[Errno 10048] ... address already in use` | Something else uses port 8000. Run `fincomb serve --port 8001`. |
| `/health` returns 503, `"database": "unreachable"` | Start Postgres: `docker compose up -d db`, then `fincomb check-db`. |
| `"schema_revision": null` | The database isn't migrated: `alembic upgrade head`. |
| `404 symbol_not_found` | The symbol has no stored prices. Ingest it first (CLI or `POST /ingestion/runs`). |
| `422 insufficient_data` | Fewer than 2 common dates in the range. Widen `start`/`end` or check `fincomb coverage`. |
| `422 missing_data` | You asked for compounded returns with `align=outer` and a symbol has a gap. Use the default `align=inner`. |
| A statistic is `null` | Undefined for that data (e.g. Sharpe with zero volatility), the same as `n/a` in the CLI. |
| The browser can't reach the API from another device | By design: it only listens on 127.0.0.1. `--host 0.0.0.0` opens it to your network, but there's no authentication yet, so only do that on a network you trust. |

If something else goes wrong, send me the full command and its full output.
