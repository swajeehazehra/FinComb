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


@app.command()
def ingest(
    symbols: Annotated[list[str], typer.Argument(help="Tickers, e.g. AAPL MSFT SPY")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")] = "2015-01-01",
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD, default today")] = None,
    source: Annotated[str | None, typer.Option(help=f"One of {SOURCES}")] = None,
    asset_type: Annotated[
        AssetType, typer.Option(help="Used when a symbol is new")
    ] = AssetType.STOCK,
) -> None:
    """Download daily bars, validate them and store them in Postgres."""
    settings, factory = _session_factory()
    start_d = _parse_date(start)
    end_d = _parse_date(end) or date.today()
    src = get_source(source or settings.default_data_source, settings)

    summary = run_ingestion(
        factory,
        src,
        symbols,
        start_d,
        end_d,
        asset_type=asset_type,
        extreme_return_threshold=settings.extreme_return_threshold,
        stale_price_run=settings.stale_price_run,
    )

    typer.echo(f"\nRun #{summary.run_id}: {summary.status.value.upper()}")
    typer.echo(f"{'symbol':<8}{'inserted':>10}{'updated':>10}{'rejected':>10}{'warnings':>10}")
    for r in summary.results:
        if r.error:
            typer.secho(f"{r.symbol:<8}  ERROR: {r.error}", fg="red")
        else:
            typer.echo(
                f"{r.symbol:<8}{r.inserted:>10}{r.updated:>10}{r.rejected:>10}{r.warnings:>10}"
            )
    if summary.status is not RunStatus.SUCCESS:
        raise typer.Exit(1)


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
