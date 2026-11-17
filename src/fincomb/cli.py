"""Command-line interface.

fincomb check-db
fincomb ingest AAPL MSFT SPY --start 2015-01-01
fincomb coverage
fincomb show AAPL --start 2024-01-01
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
from fincomb.ingestion.audit import audit_data
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.ingestion.registry import SOURCES, get_source
from fincomb.logging_config import configure_logging

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


def _owners(factory, symbols: list[str]) -> dict[str, str | None]:
    with session_scope(factory) as s:
        repo = InstrumentRepository(s)
        out = {}
        for sym in symbols:
            inst = repo.get_by_symbol(sym)
            out[sym.upper()] = inst.data_source if inst else None
        return out


def _group_by_source(
    symbols: list[str], explicit: str | None, owners: dict[str, str | None], default: str
) -> dict[str, list[str]]:
    """Explicit --source wins; otherwise each symbol keeps its owner; else the default."""
    groups: dict[str, list[str]] = {}
    for sym in dict.fromkeys(s.strip().upper() for s in symbols if s.strip()):
        name = explicit or owners.get(sym) or default
        groups.setdefault(name, []).append(sym)
    return groups


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


def _run_groups(factory, settings, groups, start_d, end_d, mode, asset_type=None) -> bool:
    ok = True
    for name, syms in groups.items():
        src = get_source(name, settings)
        summary = run_ingestion(
            factory,
            src,
            syms,
            start_d,
            end_d,
            mode=mode,
            asset_type=asset_type,
            extreme_return_threshold=settings.extreme_return_threshold,
            stale_price_run=settings.stale_price_run,
        )
        _print_summary(summary)
        ok = ok and summary.status is RunStatus.SUCCESS
    return ok


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
    groups = _group_by_source(
        symbols, source, _owners(factory, symbols), settings.default_data_source
    )
    if not _run_groups(factory, settings, groups, start_d, end_d, "update", asset_type):
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
        with session_scope(factory) as s:
            symbols = InstrumentRepository(s).all_symbols()
    if not symbols:
        typer.secho("Give symbols to refresh, or --all.", fg="red", err=True)
        raise typer.Exit(1)
    groups = _group_by_source(
        symbols, source, _owners(factory, symbols), settings.default_data_source
    )
    if not _run_groups(factory, settings, groups, _parse_date(start), date.today(), "replace"):
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


if __name__ == "__main__":
    app()
