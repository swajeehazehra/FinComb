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

from fincomb.analytics.correlation import correlation_matrix
from fincomb.analytics.returns import simple_returns
from fincomb.analytics.risk import drawdown_episodes
from fincomb.analytics.summary import performance_summary
from fincomb.config import get_settings
from fincomb.db.base import make_engine, make_session_factory, session_scope
from fincomb.db.models import AssetType, RunStatus
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.exceptions import FinCombError
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.ingestion.registry import SOURCES, get_source
from fincomb.logging_config import configure_logging
from fincomb.services.market_data import load_prices

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


def _load(symbols: list[str], start: str | None, end: str | None, align: str = "inner"):
    if align not in ("inner", "outer"):
        raise typer.BadParameter("align must be 'inner' or 'outer'")
    _, factory = _session_factory()
    try:
        with session_scope(factory) as s:
            return load_prices(s, symbols, _parse_date(start), _parse_date(end), align=align)  # type: ignore[arg-type]
    except FinCombError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc


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
    prices = _load(symbols, start, end, align)
    table = performance_summary(prices, risk_free=risk_free)
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
    _print_alignment_notes(prices)


def _print_alignment_notes(prices) -> None:
    """Show, next to the results, why inner alignment shortened the period."""
    notes = prices.attrs.get("alignment_notes", [])
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
    prices = _load(symbols, start, end, align)
    try:
        corr = correlation_matrix(simple_returns(prices), method=method)  # type: ignore[arg-type]
    except (ValueError, FinCombError) as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"\n{method.title()} correlation of daily returns\n")
    typer.echo(corr.round(2).to_string())
    _print_alignment_notes(prices)


@app.command()
def drawdowns(
    symbol: str,
    start: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="YYYY-MM-DD")] = None,
    top: Annotated[int, typer.Option(help="How many episodes")] = 5,
) -> None:
    """The deepest drawdowns of one symbol, with how long each took to recover."""
    prices = _load([symbol], start, end)
    episodes = drawdown_episodes(simple_returns(prices.iloc[:, 0]), top=top)
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


if __name__ == "__main__":
    app()
