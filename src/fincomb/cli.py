"""Command-line interface.

fincomb check-db
fincomb ingest AAPL MSFT SPY --start 2015-01-01
fincomb coverage
fincomb show AAPL --start 2024-01-01
fincomb portfolio compare AAPL MSFT SPY TLT --max-weight 0.4
fincomb backtest run momentum AAPL MSFT QQQ TLT GLD --start 2018-01-01
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
from fincomb.services import (
    analytics_service,
    backtest_service,
    ingestion_service,
    portfolio_service,
)

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


# --- portfolios (Milestone 4) ---------------------------------------------------------------

portfolio_app = typer.Typer(
    help="Build, save and analyse portfolios (long-only, fully invested).",
    no_args_is_help=True,
)
app.add_typer(portfolio_app, name="portfolio")

METHOD_CHOICES = ("equal-weight", "inverse-vol", "min-variance", "max-sharpe")


def _method(value: str) -> str:
    method = value.strip().lower().replace("-", "_")
    if method not in ("equal_weight", "inverse_vol", "min_variance", "max_sharpe"):
        raise typer.BadParameter(f"method must be one of {', '.join(METHOD_CHOICES)}")
    return method


def _covariance(value: str) -> str:
    cov = value.strip().lower().replace("-", "_")
    if cov not in ("ledoit_wolf", "sample"):
        raise typer.BadParameter("covariance must be ledoit-wolf or sample")
    return cov


def _print_warnings(warnings: list[str]) -> None:
    """Warnings go ABOVE the numbers, highlighted, so they can't be missed."""
    for w in warnings:
        typer.secho(f"\nWARNING: {w}", fg="red", bold=True)
    if warnings:
        typer.echo("")


def _parse_weights(text_value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for part in text_value.split(","):
        if not part.strip():
            continue
        symbol, sep, number = part.partition("=")
        if not sep:
            raise typer.BadParameter(f"{part!r}: write weights as SYMBOL=WEIGHT, e.g. SPY=0.6")
        try:
            value = float(number)
        except ValueError as exc:
            raise typer.BadParameter(f"{number!r} is not a number") from exc
        key = symbol.strip().upper()
        if key in weights:
            raise typer.BadParameter(f"{key} is listed twice")
        weights[key] = value
    if not weights:
        raise typer.BadParameter("give at least one SYMBOL=WEIGHT")
    return weights


def _spec(symbols, method, start, end, lookback, min_weight, max_weight, covariance, risk_free):
    return portfolio_service.ConstructionSpec(
        symbols=symbols,
        method=_method(method),
        start=_parse_date(start),
        end=_parse_date(end),
        lookback=lookback,
        min_weight=min_weight,
        max_weight=max_weight,
        covariance=_covariance(covariance),
        risk_free=risk_free,
    )


def _print_construction(outcome) -> None:
    r = outcome.result
    method = r.method if not r.fallback_used else f"{r.method} (fell back to {r.effective_method})"
    typer.echo(
        f"\n{method}: window {r.start.date()} to {r.end.date()} "
        f"({r.observations} daily returns), bounds {r.bounds.min_weight:.0%} to "
        f"{r.bounds.max_weight:.0%}"
    )
    if r.covariance:
        shrink = "" if r.shrinkage is None else f", shrinkage {r.shrinkage:.1%}"
        typer.echo(f"covariance: {r.covariance}{shrink}")
    _print_warnings(r.warnings)
    typer.echo(f"\n{'symbol':<8}{'weight':>9}{'risk share':>12}")
    for sym, row in outcome.contributions.iterrows():
        typer.echo(f"{sym:<8}{_pct(row.weight):>9}{_pct(row.percent):>12}")
    typer.echo(
        f"\nIn-sample (the window the weights were fitted on, so flattering): "
        f"return {_pct(r.expected_return)}, volatility {_pct(r.volatility)}, "
        f"Sharpe {_num(r.sharpe)} (risk-free {r.risk_free:.2%})"
    )
    _print_alignment_notes(outcome.notes)


LookbackOpt = Annotated[
    int, typer.Option(help="Daily returns in the window (ignored with --start)")
]
MinWeightOpt = Annotated[float, typer.Option(help="Smallest weight per asset, 0.05 = 5%")]
MaxWeightOpt = Annotated[float, typer.Option(help="Largest weight per asset, 0.4 = 40%")]
CovOpt = Annotated[str, typer.Option(help="ledoit-wolf (default) or sample")]
RiskFreeOpt = Annotated[float, typer.Option(help="Annual risk-free rate, 0.04 = 4%")]
DateOpt = Annotated[str | None, typer.Option(help="YYYY-MM-DD")]


@portfolio_app.command("construct")
def portfolio_construct(
    symbols: Annotated[list[str], typer.Argument(help="Tickers already ingested")],
    method: Annotated[str, typer.Option(help=", ".join(METHOD_CHOICES))] = "min-variance",
    start: DateOpt = None,
    end: DateOpt = None,
    lookback: LookbackOpt = portfolio_service.DEFAULT_LOOKBACK,
    min_weight: MinWeightOpt = 0.0,
    max_weight: MaxWeightOpt = 1.0,
    covariance: CovOpt = "ledoit-wolf",
    risk_free: RiskFreeOpt = 0.0,
    save: Annotated[str | None, typer.Option(help="Save under this name")] = None,
    description: Annotated[str | None, typer.Option(help="Saved with --save")] = None,
) -> None:
    """Compute portfolio weights from stored prices (and optionally save them)."""
    spec = _spec(
        symbols, method, start, end, lookback, min_weight, max_weight, covariance, risk_free
    )
    if save:
        record, outcome = _analyse(portfolio_service.create_constructed, save, spec, description)
        _print_construction(outcome)
        typer.secho(f"\nSaved as {record.name!r} (id {record.id}).", fg="green")
    else:
        _print_construction(_analyse(portfolio_service.construct_portfolio, spec))


@portfolio_app.command("compare")
def portfolio_compare(
    symbols: Annotated[list[str], typer.Argument(help="Tickers already ingested")],
    start: DateOpt = None,
    end: DateOpt = None,
    lookback: LookbackOpt = portfolio_service.DEFAULT_LOOKBACK,
    min_weight: MinWeightOpt = 0.0,
    max_weight: MaxWeightOpt = 1.0,
    covariance: CovOpt = "ledoit-wolf",
    risk_free: RiskFreeOpt = 0.0,
) -> None:
    """All four methods side by side on the same window."""
    outcomes = {}
    for m in METHOD_CHOICES:
        spec = _spec(
            symbols, m, start, end, lookback, min_weight, max_weight, covariance, risk_free
        )
        outcomes[m] = _analyse(portfolio_service.construct_portfolio, spec)
    first = next(iter(outcomes.values())).result
    typer.echo(
        f"\nWindow {first.start.date()} to {first.end.date()} ({first.observations} daily "
        f"returns), risk-free {risk_free:.2%}"
    )
    for m, out in outcomes.items():
        for w in out.result.warnings:
            typer.secho(f"\nWARNING ({m}): {w}", fg="red", bold=True)
    header = f"\n{'':<12}" + "".join(f"{m:>14}" for m in outcomes)
    typer.echo(header)
    for sym in first.weights.index:
        typer.echo(
            f"{sym:<12}" + "".join(f"{_pct(o.result.weights[sym]):>14}" for o in outcomes.values())
        )
    typer.echo("-" * (12 + 14 * len(outcomes)))
    for label, attr, fmt in (
        ("return", "expected_return", _pct),
        ("volatility", "volatility", _pct),
        ("Sharpe", "sharpe", _num),
    ):
        typer.echo(
            f"{label:<12}"
            + "".join(f"{fmt(getattr(o.result, attr)):>14}" for o in outcomes.values())
        )
    typer.echo("\nReturn, volatility and Sharpe are in-sample (the fitting window).")
    _print_alignment_notes(next(iter(outcomes.values())).notes)


@portfolio_app.command("create")
def portfolio_create(
    name: str,
    weights: Annotated[str, typer.Option(help="SYMBOL=WEIGHT pairs, e.g. SPY=0.6,TLT=0.4")],
    description: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Save a portfolio with weights you choose (they must sum to 1)."""
    record = _analyse(portfolio_service.create_manual, name, _parse_weights(weights), description)
    typer.secho(
        f"Saved {record.name!r} (id {record.id}) with {len(record.weights)} assets.", fg="green"
    )


@portfolio_app.command("list")
def portfolio_list() -> None:
    """Saved portfolios."""
    records = _analyse(portfolio_service.list_portfolios)
    if not records:
        typer.echo("No portfolios yet. Try: fincomb portfolio construct AAPL SPY TLT --save NAME")
        return
    typer.echo(f"{'id':>4}  {'name':<24}{'method':<34}{'assets':>7}  created")
    for p in records:
        method = p.construction_method
        if p.effective_method != p.construction_method:
            method += f" (fell back to {p.effective_method})"
        flag = "  [warnings]" if p.warnings else ""
        typer.echo(
            f"{p.id:>4}  {p.name:<24}{method:<34}{len(p.weights):>7}  {p.created_at:%Y-%m-%d}{flag}"
        )


@portfolio_app.command("show")
def portfolio_show(name: Annotated[str, typer.Argument(help="Name or id")]) -> None:
    """Weights and the parameters that produced them."""
    p = _analyse(portfolio_service.get_portfolio, name)
    typer.echo(f"\n{p.name} (id {p.id}): {p.construction_method}")
    if p.effective_method != p.construction_method:
        typer.echo(f"weights actually from: {p.effective_method}")
    if p.description:
        typer.echo(p.description)
    if p.estimation_start:
        params = p.construction_params
        typer.echo(
            f"estimated on {p.estimation_start} to {p.estimation_end} "
            f"({params.get('observations')} returns), bounds {params.get('min_weight'):.0%} "
            f"to {params.get('max_weight'):.0%}, covariance {params.get('covariance') or '-'}, "
            f"risk-free {params.get('risk_free', 0):.2%}"
        )
    _print_warnings(p.warnings)
    typer.echo(f"\n{'symbol':<8}{'weight':>9}")
    for sym, w in p.weights.items():
        typer.echo(f"{sym:<8}{_pct(w):>9}")


@portfolio_app.command("risk")
def portfolio_risk(
    name: Annotated[str, typer.Argument(help="Name or id")],
    start: DateOpt = None,
    end: DateOpt = None,
    benchmark: Annotated[
        str, typer.Option(help="Benchmark for beta; empty string for none")
    ] = portfolio_service.DEFAULT_BENCHMARK,
    confidence: Annotated[float, typer.Option(help="VaR confidence, 0.95 = 95%")] = 0.95,
    risk_free: RiskFreeOpt = 0.0,
) -> None:
    """Volatility, VaR/CVaR, beta and risk contributions of a saved portfolio."""
    out = _analyse(
        portfolio_service.portfolio_risk,
        name,
        _parse_date(start),
        _parse_date(end),
        benchmark or None,
        confidence,
        risk_free,
    )
    rep = out.report
    typer.echo(
        f"\n{out.portfolio.name}: {rep.start.date()} to {rep.end.date()} "
        f"({rep.observations} daily returns), daily rebalanced to its weights"
    )
    _print_warnings(out.portfolio.warnings + rep.warnings)
    s = rep.summary
    typer.echo(
        f"\nreturn {_pct(s['total_return'])}  CAGR {_pct(s['cagr'])}  volatility "
        f"{_pct(rep.volatility)}  Sharpe {_num(s['sharpe'])}  max DD {_pct(s['max_drawdown'])}"
    )
    c = f"{rep.confidence:.1%}".replace(".0%", "%")
    typer.echo(f"\n1-day loss at {c} confidence   {'historical':>11}{'normal':>9}")
    typer.echo(f"{'VaR':<34}{_pct(rep.var_historical):>11}{_pct(rep.var_normal):>9}")
    typer.echo(
        f"{'CVaR (average beyond VaR)':<34}{_pct(rep.cvar_historical):>11}"
        f"{_pct(rep.cvar_normal):>9}"
    )
    if rep.beta is not None:
        b = rep.beta
        typer.echo(
            f"\nvs {rep.benchmark}: beta {_num(b.beta)}, alpha {_pct(b.alpha)} a year, "
            f"correlation {_num(b.correlation)}, R^2 {_num(b.r_squared)}"
        )
    typer.echo(f"\n{'symbol':<8}{'weight':>9}{'risk share':>12}{'marginal':>10}")
    for sym, row in rep.contributions.iterrows():
        typer.echo(f"{sym:<8}{_pct(row.weight):>9}{_pct(row.percent):>12}{_pct(row.marginal):>10}")
    _print_alignment_notes(out.notes)


@portfolio_app.command("delete")
def portfolio_delete(name: Annotated[str, typer.Argument(help="Name or id")]) -> None:
    """Delete a saved portfolio."""
    record = _analyse(portfolio_service.delete_portfolio, name)
    typer.echo(f"Deleted {record.name!r} (id {record.id}).")


# --- strategies and backtests (Milestone 5) ---------------------------------------------------

backtest_app = typer.Typer(
    help="Run, list, inspect and verify backtests (long-only, daily).", no_args_is_help=True
)
app.add_typer(backtest_app, name="backtest")


@app.command()
def strategies() -> None:
    """Available strategies and their parameters."""
    for s in backtest_service.strategies():
        typer.secho(f"\n{s['name']}", bold=True)
        typer.echo(f"  {s['description']}")
        typer.echo(f"  default rebalance: {s['default_rebalance']}")
        for p in s["params"]:
            limits = ""
            if p["choices"]:
                limits = f" ({', '.join(p['choices'])})"
            elif p["minimum"] is not None:
                limits = f" ({p['minimum']:g} to {p['maximum']:g})"
            typer.echo(f"  --param {p['name']}=...  default {p['default']}{limits}: {p['help']}")


def _parse_params(items: list[str] | None) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise typer.BadParameter(f"{item!r}: write parameters as NAME=VALUE, e.g. top_n=2")
        if key in params:
            raise typer.BadParameter(f"{key} is given twice")
        params[key] = value.strip()
    return params


def _bt_line(label: str, s: dict, keys) -> str:
    return f"{label:<12}" + "".join(f"{fmt(s.get(k)):>11}" for k, fmt in keys)


def _f(fmt):
    return lambda x: "n/a" if x is None else fmt(x)


_PCT = _f(_pct)
_NUM = _f(_num)
_STAT_COLUMNS = (
    ("total_return", _PCT),
    ("cagr", _PCT),
    ("annual_volatility", _PCT),
    ("sharpe", _NUM),
    ("sortino", _NUM),
    ("max_drawdown", _PCT),
    ("calmar", _NUM),
)


def _print_backtest(r) -> None:
    m = r.metrics
    saved = f"run {r.id}" if r.id is not None else "not saved"
    typer.echo(
        f"\n{r.strategy} ({saved}) on {', '.join(r.symbols)}: {r.start} to {r.end} "
        f"({m['observations']} daily returns)"
    )
    params = ", ".join(f"{k}={v}" for k, v in r.params.items() if v is not None) or "defaults"
    typer.echo(
        f"params: {params}; rebalance {r.rebalance}, execution {r.execution}, capital "
        f"{r.initial_capital:,.0f}, costs {r.cost_bps:g} + {r.slippage_bps:g} bps"
    )
    _print_warnings(r.warnings)
    header = f"\n{'':<12}" + "".join(
        f"{h:>11}" for h in ("total", "CAGR", "vol", "Sharpe", "Sortino", "max DD", "Calmar")
    )
    typer.echo(header)
    typer.echo(_bt_line("strategy", m["strategy"], _STAT_COLUMNS))
    if m.get("benchmark"):
        typer.echo(_bt_line(r.benchmark, m["benchmark"], _STAT_COLUMNS))
        rel = m["relative"]
        typer.echo(
            f"\nvs {r.benchmark}: excess CAGR {_PCT(rel['excess_cagr'])}, tracking error "
            f"{_PCT(rel['tracking_error'])}, information ratio {_NUM(rel['information_ratio'])}"
        )
        typer.echo(
            f"alpha {_PCT(rel['alpha'])} a year (t = {_NUM(rel['alpha_t'])}, p = "
            f"{_NUM(rel['alpha_p'])}), beta {_NUM(rel['beta'])} (t = {_NUM(rel['beta_t'])}), "
            f"R^2 {_NUM(rel['r_squared'])}; Newey-West errors, {rel['hac_lags']} lags"
        )
    t = m["trading"]
    typer.echo(
        f"\ntrading: {t['rebalances']} rebalances, turnover {_NUM(t['annual_turnover'])}x a "
        f"year, costs {t['total_costs']:,.2f} ({_PCT(t['costs_pct_of_capital'])} of capital), "
        f"average cash {_PCT(t['average_cash'])}"
    )
    typer.echo(
        f"final equity {t['final_equity']:,.2f}; selling everything now would cost "
        f"{t['liquidation_cost']:,.2f}"
    )
    adf = (m.get("diagnostics") or {}).get("adf")
    if adf:
        typer.echo("\nADF on log prices (in-sample diagnostic, not used to trade):")
        for sym, d in adf.items():
            typer.echo(f"  {sym:<8}" + ("n/a" if d is None else f"p = {d['p_value']:.3f}"))
    typer.echo(f"\ndata hash {r.data_hash[:16]}..., code {r.code_version[:12]}")
    _print_alignment_notes(r.notes)


@backtest_app.command("run")
def backtest_run(
    strategy: Annotated[str, typer.Argument(help="See `fincomb strategies`")],
    symbols: Annotated[
        list[str] | None, typer.Argument(help="Tickers already ingested (none with --portfolio)")
    ] = None,
    start: DateOpt = None,
    end: DateOpt = None,
    benchmark: Annotated[
        str, typer.Option(help="Benchmark symbol; empty string for none")
    ] = backtest_service.DEFAULT_BENCHMARK,
    capital: Annotated[
        float, typer.Option(help="Initial capital")
    ] = backtest_service.DEFAULT_CAPITAL,
    cost_bps: Annotated[
        float, typer.Option(help="Commission, bps of traded value")
    ] = backtest_service.DEFAULT_COST_BPS,
    slippage_bps: Annotated[
        float, typer.Option(help="Slippage, bps of traded value")
    ] = backtest_service.DEFAULT_SLIPPAGE_BPS,
    rebalance: Annotated[
        str | None,
        typer.Option(
            help="daily, weekly, monthly, quarterly, annual or never (default: the strategy's own)"
        ),
    ] = None,
    execution: Annotated[
        str, typer.Option(help="next-close (default) or same-close (optimistic)")
    ] = "next-close",
    param: Annotated[
        list[str] | None, typer.Option(help="Strategy parameter NAME=VALUE (repeatable)")
    ] = None,
    weights: Annotated[
        str | None, typer.Option(help="buy_and_hold / fixed_weights: SPY=0.6,TLT=0.4")
    ] = None,
    portfolio: Annotated[
        str | None, typer.Option(help="Saved portfolio name or id (fixed_weights, buy_and_hold)")
    ] = None,
    risk_free: RiskFreeOpt = 0.0,
    save: Annotated[bool, typer.Option(help="Store the run (--no-save to explore)")] = True,
) -> None:
    """Run a backtest on stored prices and print its report."""
    params: dict = _parse_params(param)
    if weights is not None:
        params["weights"] = weights
    spec = backtest_service.BacktestSpec(
        strategy=strategy,
        symbols=list(symbols or []),
        params=params,
        start=_parse_date(start),
        end=_parse_date(end),
        benchmark=benchmark or None,
        initial_capital=capital,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        rebalance=rebalance.strip().lower() if rebalance else None,
        execution=execution.strip().lower().replace("-", "_"),
        portfolio=portfolio,
        risk_free=risk_free,
        save=save,
    )
    out = _analyse(backtest_service.run_backtest, spec)
    _print_backtest(out.record)
    if out.record.id is not None:
        typer.secho(f"\nSaved as run {out.record.id}.", fg="green")


@backtest_app.command("list")
def backtest_list(
    strategy: Annotated[str | None, typer.Option(help="Only this strategy")] = None,
    limit: Annotated[int, typer.Option(help="How many, newest first")] = 20,
) -> None:
    """Stored backtests with their headline numbers."""
    runs = _analyse(backtest_service.list_backtests, strategy, limit)
    if not runs:
        typer.echo("No backtests yet. Try: fincomb backtest run buy_and_hold SPY")
        return
    typer.echo(
        f"{'id':>4}  {'strategy':<15}{'symbols':<24}{'from':>11}{'to':>11}"
        f"{'CAGR':>9}{'Sharpe':>8}{'max DD':>9}{'turnover':>9}"
    )
    for r in runs:
        s = r.metrics.get("strategy", {})
        syms = ",".join(r.symbols)
        syms = syms if len(syms) <= 22 else syms[:21] + "~"
        flag = "  [warnings]" if r.warnings else ""
        typer.echo(
            f"{r.id:>4}  {r.strategy:<15}{syms:<24}{str(r.start):>11}{str(r.end):>11}"
            f"{_PCT(s.get('cagr')):>9}{_NUM(s.get('sharpe')):>8}{_PCT(s.get('max_drawdown')):>9}"
            f"{_NUM(r.metrics.get('trading', {}).get('annual_turnover')):>9}{flag}"
        )


@backtest_app.command("show")
def backtest_show(run_id: Annotated[int, typer.Argument(help="Run id")]) -> None:
    """The full report of a stored run."""
    _print_backtest(_analyse(backtest_service.get_backtest, run_id))


@backtest_app.command("trades")
def backtest_trades(
    run_id: Annotated[int, typer.Argument(help="Run id")],
    limit: Annotated[int, typer.Option(help="Rows to show")] = 50,
) -> None:
    """Trade log (quantities are indicative fractional shares)."""
    trades, total = _analyse(backtest_service.backtest_trades, run_id, limit)
    typer.echo(
        f"{'date':<12}{'kind':<10}{'symbol':<8}{'side':<6}{'from':>8}{'to':>8}"
        f"{'notional':>14}{'fill':>11}{'quantity':>12}{'costs':>10}"
    )
    for t in trades:
        typer.echo(
            f"{str(t.date):<12}{t.kind:<10}{t.symbol:<8}{t.side:<6}{_pct(t.weight_before):>8}"
            f"{_pct(t.weight_after):>8}{t.notional:>14,.2f}{t.fill_price:>11,.2f}"
            f"{t.quantity:>12,.3f}{t.commission + t.slippage_cost:>10,.2f}"
        )
    if total > len(trades):
        typer.echo(f"... {total - len(trades)} more (use --limit)")


@backtest_app.command("equity")
def backtest_equity(
    run_id: Annotated[int, typer.Argument(help="Run id")],
    csv: Annotated[str | None, typer.Option(help="Write the full daily curve to this file")] = None,
) -> None:
    """Equity curve summary, or the whole curve as CSV."""
    frame = _analyse(backtest_service.backtest_equity, run_id)
    if csv:
        frame.to_csv(csv, index_label="date")
        typer.secho(f"Wrote {len(frame)} rows to {csv}", fg="green")
        return
    show = frame if len(frame) <= 12 else frame.iloc[list(range(5)) + list(range(-5, 0))]
    typer.echo(f"{'date':<12}{'equity':>14}{'benchmark':>14}{'drawdown':>10}{'cash':>12}")
    for i, (ts, r) in enumerate(show.iterrows()):
        if len(frame) > 12 and i == 5:
            typer.echo("...")
        bench = "n/a" if r.benchmark_equity != r.benchmark_equity else f"{r.benchmark_equity:,.2f}"
        typer.echo(
            f"{str(ts.date()):<12}{r.equity:>14,.2f}{bench:>14}{_pct(r.drawdown):>10}"
            f"{r.cash:>12,.2f}"
        )


@backtest_app.command("verify")
def backtest_verify(run_id: Annotated[int, typer.Argument(help="Run id")]) -> None:
    """Rerun a stored backtest from current data and compare the data hash and metrics."""
    v = _analyse(backtest_service.verify_backtest, run_id)
    if v.reproducible:
        typer.secho(f"Run {v.run_id} is reproducible: same data hash, same metrics.", fg="green")
    else:
        typer.secho(f"Run {v.run_id} is NOT reproducible from current data.", fg="red", bold=True)
        if not v.data_hash_matches:
            typer.echo("The stored prices changed since the run (re-ingested or re-adjusted).")
        if v.mismatched_metrics:
            typer.echo("Metrics that differ: " + ", ".join(v.mismatched_metrics[:10]))
    typer.echo(f"code then {v.stored_code_version[:12]}, now {v.current_code_version[:12]}")
    if not v.reproducible:
        raise typer.Exit(1)


@backtest_app.command("delete")
def backtest_delete(run_id: Annotated[int, typer.Argument(help="Run id")]) -> None:
    """Delete a stored run with its equity curve and trades."""
    r = _analyse(backtest_service.delete_backtest, run_id)
    typer.echo(f"Deleted run {r.id} ({r.strategy}).")


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
