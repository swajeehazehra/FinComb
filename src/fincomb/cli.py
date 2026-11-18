"""Command-line interface.

fincomb check-db
fincomb ingest AAPL MSFT SPY --start 2015-01-01
fincomb coverage
fincomb show AAPL --start 2024-01-01
fincomb serve            # REST API on http://127.0.0.1:8000 (docs at /docs)

Commands only parse arguments and print results: the work is done by the shared
services in `fincomb.services`, which the API uses too.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Annotated

import typer
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError

from fincomb.config import get_settings
from fincomb.db.base import make_engine, make_session_factory, session_scope
from fincomb.db.models import AssetType, RunStatus
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.exceptions import FinCombError
from fincomb.ingestion.audit import audit_data
from fincomb.ingestion.registry import SOURCES
from fincomb.logging_config import configure_logging
from fincomb.services import analytics_service, ingestion_service

app = typer.Typer(help="FinComb command-line tools.", no_args_is_help=True)
logger = logging.getLogger("fincomb.cli")


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"{value!r} is not a YYYY-MM-DD date") from exc


def _session_factory():
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    return settings, make_session_factory(make_engine(settings.database_url()))


@app.command("check-db")
def check_db() -> None:
    """Check that the database is reachable and migrated."""
    settings, factory = _session_factory()
    try:
        with session_scope(factory) as s:
            version = s.execute(text("SELECT version()")).scalar_one()
        with session_scope(factory) as s:
            try:
                rev = s.execute(text("SELECT version_num FROM alembic_version")).scalar()
            except ProgrammingError:  # table doesn't exist yet: never migrated
                rev = None
    except OperationalError as exc:
        typer.secho(
            f"Cannot connect to Postgres at {settings.postgres_host}:"
            f"{settings.postgres_port}. Is the container running?\n{exc.orig}",
            fg="red",
            err=True,
        )
        raise typer.Exit(1) from exc
    typer.secho(f"Connected: {version.split(',')[0]}", fg="green")
    typer.echo(f"Schema revision: {rev or 'NONE (run: alembic upgrade head)'}")


def _print_summary(summary) -> None:
    typer.echo(f"\nRun #{summary.run_id}: {summary.status.value.upper()}")
    if summary.end_capped_to:
        typer.echo(
            f"(end date capped to {summary.end_capped_to}, the last completed trading "
            "session; later bars are not final yet)"
        )
    typer.echo(f"{'symbol':<8}{'inserted':>10}{'updated':>10}{'rejected':>10}{'warnings':>10}")
    for r in summary.results:
        if r.error:
            typer.secho(f"{r.symbol:<8}  ERROR: {r.error}", fg="red")
            continue
        typer.echo(f"{r.symbol:<8}{r.inserted:>10}{r.updated:>10}{r.rejected:>10}{r.warnings:>10}")
        for note in r.notes:
            typer.secho(f"{'':<8}  note: {note}", fg="yellow")


def _run(factory, settings, symbols, start_d, end_d, mode, source, asset_type=None) -> bool:
    """Run ingestion through the shared service, print each run, True if all succeeded."""
    summaries = ingestion_service.ingest(
        factory, settings, symbols, start_d, end_d, source=source, mode=mode, asset_type=asset_type
    )
    for summary in summaries:
        _print_summary(summary)
    return all(summary.status is RunStatus.SUCCESS for summary in summaries)


@app.command()
def ingest(
    symbols: Annotated[list[str], typer.Argument(help="Tickers, e.g. AAPL MSFT SPY")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2015-01-01",
    end: Annotated[
        str | None, typer.Option(help="YYYY-MM-DD, default: last completed trading session")
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            help=f"One of {SOURCES}. Default: the source that already owns each "
            "symbol, or DEFAULT_DATA_SOURCE for new symbols."
        ),
    ] = None,
    asset_type: Annotated[
        AssetType | None,
        typer.Option(
            help="stock, etf or index. New symbols default to stock; "
            "passing it also corrects symbols already stored."
        ),
    ] = None,
) -> None:
    """Download daily bars, validate them and store them in Postgres."""
    settings, factory = _session_factory()
    start_d = _parse_date(start)
    end_d = _parse_date(end) or date.today()
    if not _run(factory, settings, symbols, start_d, end_d, "update", source, asset_type):
        raise typer.Exit(1)


@app.command()
def refresh(
    symbols: Annotated[list[str] | None, typer.Argument(help="Tickers to rebuild")] = None,
    all_symbols: Annotated[bool, typer.Option("--all", help="Rebuild every stored symbol")] = False,
    source: Annotated[
        str | None,
        typer.Option(
            help="Source to rebuild from (switches ownership). Default: each "
            "symbol's current owner, or DEFAULT_DATA_SOURCE if it has none."
        ),
    ] = None,
    start: Annotated[
        str | None, typer.Option(help="YYYY-MM-DD. Default: each symbol's first stored date")
    ] = None,
) -> None:
    """Re-download a symbol's whole history from ONE source and replace what is stored.

    Use it to repair mixed-source or inconsistently adjusted history, or to switch
    a symbol to another source. Each symbol is replaced in a single transaction:
    if the download fails, its old data is left untouched.
    """
    settings, factory = _session_factory()
    if all_symbols:
        symbols = ingestion_service.stored_symbols(factory)
    if not symbols:
        typer.secho("Give symbols to refresh, or --all.", fg="red", err=True)
        raise typer.Exit(1)
    if not _run(factory, settings, symbols, _parse_date(start), date.today(), "replace", source):
        raise typer.Exit(1)


@app.command("check-data")
def check_data() -> None:
    """Audit stored history: one source per symbol, an owner, and no suspicious jumps."""
    settings, factory = _session_factory()
    with session_scope(factory) as s:
        rows = audit_data(s, settings.extreme_return_threshold)
    if not rows:
        typer.echo("No instruments yet.")
        return
    typer.echo(f"{'symbol':<8}{'owner':<10}{'sources in bars':<20}{'bars':>7}{'jumps':>7}  status")
    problems = 0
    for r in rows:
        bad = r.status != "OK"
        problems += bad
        typer.secho(
            f"{r.symbol:<8}{r.owner or '-':<10}{', '.join(r.sources) or '-':<20}{r.bars:>7}"
            f"{r.jumps:>7}  {r.status}",
            fg="red" if bad else None,
        )
    if problems:
        typer.echo(
            f"\n{problems} symbol(s) need attention. Repair with: "
            "fincomb refresh SYMBOL [--source NAME]   (or fincomb refresh --all)"
        )
        raise typer.Exit(1)
    typer.secho("\nAll symbols have a single-source, consistent history.", fg="green")


@app.command("set-type")
def set_type(
    symbol: str,
    asset_type: Annotated[AssetType, typer.Argument(help="stock, etf or index")],
) -> None:
    """Change the asset type of a stored symbol, e.g. `fincomb set-type SPY etf`."""
    _, factory = _session_factory()
    with session_scope(factory) as s:
        instrument = InstrumentRepository(s).set_asset_type(symbol, asset_type)
    if instrument is None:
        typer.secho(f"{symbol.upper()} is not stored yet.", fg="red", err=True)
        raise typer.Exit(1)
    typer.echo(f"{symbol.upper()} is now {asset_type.value}.")


@app.command()
def coverage() -> None:
    """List stored instruments and their date coverage."""
    _, factory = _session_factory()
    with session_scope(factory) as s:
        rows = InstrumentRepository(s).coverage()
    if not rows:
        typer.echo("No instruments yet. Try: fincomb ingest AAPL SPY")
        return
    typer.echo(f"{'symbol':<8}{'type':<7}{'first':>12}{'last':>12}{'bars':>8}")
    for r in rows:
        typer.echo(
            f"{r.symbol:<8}{r.asset_type:<7}{str(r.first_date):>12}"
            f"{str(r.last_date):>12}{r.bars:>8}"
        )


@app.command()
def show(
    symbol: str,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    tail: Annotated[int, typer.Option(help="Rows to print")] = 10,
) -> None:
    """Print the most recent stored bars for a symbol."""
    _, factory = _session_factory()
    with session_scope(factory) as s:
        df = PriceRepository(s).get_bars(symbol, _parse_date(start), _parse_date(end))
    if df.empty:
        typer.echo(f"No bars stored for {symbol.upper()}.")
        raise typer.Exit(1)
    typer.echo(df.tail(tail).to_string())


def _analyse(func, *args, **kwargs):
    """Call an analytics_service function in a session; print domain errors and exit 1."""
    _, factory = _session_factory()
    try:
        with session_scope(factory) as s:
            return func(s, *args, **kwargs)
    except (ValueError, FinCombError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc


def _check_align(align: str) -> None:
    if align not in ("inner", "outer"):
        raise typer.BadParameter("align must be 'inner' or 'outer'")


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x * 100:,.2f}%"  # x != x is True only for NaN


def _num(x: float) -> str:
    return "n/a" if x != x else f"{x:,.2f}"


@app.command()
def analyze(
    symbols: Annotated[list[str], typer.Argument(help="Tickers already ingested")],
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    risk_free: Annotated[float, typer.Option(help="Annual risk-free rate, 0.04 = 4%")] = 0.0,
    align: Annotated[
        str,
        typer.Option(
            help="inner = only dates all symbols share (default); "
            "outer = each symbol over its own full history"
        ),
    ] = "inner",
) -> None:
    """Return, volatility, Sharpe, Sortino and drawdown for each symbol."""
    _check_align(align)
    result = _analyse(
        analytics_service.performance,
        symbols,
        _parse_date(start),
        _parse_date(end),
        risk_free=risk_free,
        align=align,
    )
    table = result.table
    if align == "outer":
        typer.echo(f"\nEach symbol over its own history, risk-free {risk_free:.2%}\n")
    else:
        typer.echo(
            f"\nPeriod {table['start'].min()} to {table['end'].max()}, "
            f"{int(table['observations'].iloc[0])} daily returns, "
            f"risk-free {risk_free:.2%}\n"
        )
    header = (
        f"{'symbol':<8}{'from':>12}{'to':>12}"
        f"{'total':>10}{'CAGR':>9}{'vol':>9}{'sharpe':>8}"
        f"{'sortino':>9}{'max DD':>10}{'calmar':>8}"
    )
    typer.echo(header)
    typer.echo("-" * len(header))
    for sym, r in table.iterrows():
        typer.echo(
            f"{sym:<8}{str(r.start):>12}{str(r.end):>12}"
            f"{_pct(r.total_return):>10}{_pct(r.cagr):>9}"
            f"{_pct(r.annual_volatility):>9}{_num(r.sharpe):>8}{_num(r.sortino):>9}"
            f"{_pct(r.max_drawdown):>10}{_num(r.calmar):>8}"
        )
    _print_alignment_notes(result.notes)


def _print_alignment_notes(notes: list[str]) -> None:
    """Show, next to the results, why inner alignment shortened the period."""
    if notes:
        typer.echo("")
    for note in notes:
        typer.secho(f"Note: {note}", fg="yellow")


@app.command()
def correlation(
    symbols: Annotated[list[str], typer.Argument(help="At least two tickers")],
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    method: Annotated[str, typer.Option(help="pearson or spearman")] = "pearson",
    align: Annotated[
        str,
        typer.Option(
            help="inner = only dates all symbols share (default); "
            "outer = each symbol over its own full history"
        ),
    ] = "inner",
) -> None:
    """Correlation matrix of daily returns.

    With --align outer each pair uses the days both symbols traded.
    """
    _check_align(align)
    result = _analyse(
        analytics_service.correlation,
        symbols,
        _parse_date(start),
        _parse_date(end),
        method=method,
        align=align,
    )
    typer.echo(f"\n{method.title()} correlation of daily returns\n")
    typer.echo(result.table.round(2).to_string())
    _print_alignment_notes(result.notes)


@app.command()
def drawdowns(
    symbol: str,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    top: Annotated[int, typer.Option(help="How many episodes")] = 5,
) -> None:
    """The deepest drawdowns of one symbol, with how long each took to recover."""
    result = _analyse(
        analytics_service.drawdowns, symbol, _parse_date(start), _parse_date(end), top=top
    )
    episodes = result.episodes
    typer.echo(
        f"\n{'depth':>9}  {'peak':<12}{'trough':<12}{'recovered':<12}"
        f"{'days down':>10}{'days up':>9}"
    )
    for e in episodes:
        typer.echo(
            f"{_pct(e.depth):>9}  {str(e.peak.date() if e.peak else 'start'):<12}"
            f"{str(e.trough.date()):<12}"
            f"{str(e.recovery.date()) if e.recovery else 'not yet':<12}"
            f"{e.days_to_trough:>10}{e.days_to_recover if e.days_to_recover else '-':>9}"
        )


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


@app.command()
def serve(
    host: Annotated[
        str,
        typer.Option(help="Interface to listen on. Default: this computer only."),
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Restart when source files change")] = False,
) -> None:
    """Run the REST API. Interactive docs at http://127.0.0.1:8000/docs."""
    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    if host not in LOOPBACK_HOSTS:
        typer.secho(
            f"Warning: listening on {host} makes the API reachable from other machines. "
            "Its write endpoints (ingestion, instruments) have no authentication.",
            fg="yellow",
            err=True,
        )
    typer.echo(f"FinComb API on http://{host}:{port}  (docs: http://{host}:{port}/docs)")
    uvicorn.run(
        "fincomb.api.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    app()
