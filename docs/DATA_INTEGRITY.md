# Data integrity: how FinComb keeps price history consistent

The Milestone 2 review found three ways the stored price history could become
silently wrong. None of them showed up in the analytics numbers yet, but each
would corrupt returns and backtests later. This page explains each problem, the
fix, and how to repair a database created before the fix.

## 1. Adjusted prices stitched together by incremental updates

**Problem.** Adjusted prices are recalculated backwards by the data vendor
every time a dividend or split happens. Suppose SPY was stored up to November,
then pays a dividend in December, and you run
`fincomb ingest SPY --start 2026-12-01`. Before the fix, only December onwards
was rewritten. The older rows kept the *old* adjustment, so the step from
30 November to 1 December looked like a real return. For a dividend that's
about 0.3%. For a 4:1 split it would look like a −75% crash. The validation
rules couldn't see it, because they only check the newly downloaded rows.

**Fix.** Every update now also re-downloads about two weeks of days that are
**already stored** and compares them with what's in the database. If the
adjusted close or close of any of those days moved by more than 0.01%, the
vendor has re-adjusted the history. FinComb then downloads the symbol's
**whole** history again and replaces the stored rows. You'll see a yellow note
like this:

```
SPY        0     2992        0        1
           note: adjustment changed (adj_close on 2026-11-28 changed from 680.1200 to
           678.0800 (0.3000%) since it was stored); full history re-downloaded
```

The event is also recorded in `data_quality_issues` with the rule
`adjustment_changed`. When nothing changed, the update behaves as before, apart
from those ~10 extra overlapping days it re-checks.

## 2. Unfinished daily bars

**Problem.** The default end date was *today*. During market hours, Yahoo
returns today's in-progress bar, and its "close" is just the latest price. That
bar was stored as if it were final.

**Fix.** The end date is now capped at the **last completed trading session**,
meaning the latest weekday whose bar is final. A bar counts as final from 18:00
New York time, two hours after the close, so vendors have time to publish the
final numbers. On weekends the cap is Friday. Any bar later than that is
dropped even if a vendor returns it. When the cap applies, `ingest` says so:

```
(end date capped to 2026-11-16, the last completed trading session; later bars are not final yet)
```

The time-zone data comes from the `tzdata` package, which Windows needs because
it has no built-in time-zone database.

## 3. One symbol, several data sources

**Problem.** Yahoo and Tiingo define the plain `close` differently: Yahoo's is
split-adjusted and Tiingo's is the raw traded price. Their adjustment factors
also differ slightly. AAPL ended up with Yahoo rows before 2024 and Tiingo rows
from 2024 onwards.

**Fix.** Each instrument now records the source that **owns** its history, in
the new column `instruments.data_source` (migration `0002`). Owners work like
this:

- `ingest` without `--source` uses each symbol's owner. New symbols use
  `DEFAULT_DATA_SOURCE`. Commands you already use keep working.
- `ingest` with a different `--source` than the owner is **refused**, with the
  exact command to run instead.
- `fincomb refresh SYMBOL --source NAME` is how to switch sources. It downloads
  the whole history from the new source and replaces the stored rows in **one
  transaction**, so if the download fails, the old data stays as it was.
- A symbol whose stored bars already mix sources has **no owner** after the
  migration. Ingestion refuses to touch it until it's refreshed.

## New commands

| Command | What it does |
|---|---|
| `fincomb check-data` | Audits every symbol and shows one status: OK, MIXED SOURCES, NO OWNER, OWNER MISMATCH, or SUSPICIOUS JUMPS (a daily adjusted move above 40%, the usual sign of a stitched split). Exits with code 1 if anything needs attention. |
| `fincomb refresh SYMBOL…` | Rebuilds those symbols' full history from one source, starting at their first stored date. Options: `--source NAME` to switch sources, `--start YYYY-MM-DD` to change the start, `--all` for every stored symbol. |

## Repairing a database created before this fix

Run these once, in PowerShell from `D:\Projects\FinComb` with `(.venv)` active
and Docker Desktop running:

```powershell
git fetch origin
git checkout data-integrity        # only needed until this PR is merged
pip install -e ".[dev]"            # adds tzdata
alembic upgrade head               # adds instruments.data_source and fills in owners
fincomb check-data                 # AAPL should show MIXED SOURCES
fincomb refresh --all              # rebuild every symbol's history from one source
fincomb check-data                 # every row should now say OK
```

What happens to your data:

- **The migration** sets the owner of every symbol whose bars all come from one
  source, such as SPY, MSFT, QQQ, TLT and GLD (all Yahoo). AAPL has bars from
  both Yahoo and Tiingo, so it's deliberately left without an owner.
- **`refresh --all`** rebuilds each symbol from its owner. AAPL has no owner, so
  it uses `DEFAULT_DATA_SOURCE` (yfinance). Its 2,985 mixed rows are deleted and
  replaced by Yahoo's full history from 2 Jan 2015, and AAPL is then owned by
  yfinance. If you'd rather keep AAPL on Tiingo, run
  `fincomb refresh AAPL --source tiingo` instead.
- **Every other symbol is refreshed too.** That's on purpose. Any stitched
  adjustment, or an unfinished bar stored during market hours, looks exactly
  like normal data once it's in the database. The only way to be sure is to
  download each history once more, in one piece.
- **Symbols from the offline demo files** (SYNA, SYNB, SYNM, BADDATA) are
  refreshed from the CSV files, since they're owned by `csv`.
- **The audit trail is kept:** past `ingestion_runs` and `data_quality_issues`
  rows aren't deleted, and each refresh adds its own run.
