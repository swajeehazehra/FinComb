"""Run, store and check backtests. Shared by the CLI and the API.

No maths here: the strategy decides targets (`fincomb.strategies`), the engine
simulates (`fincomb.backtest.engine`) and the report computes the metrics. This
module loads prices with enough warm-up history, resolves a saved portfolio,
stores the run and turns rows into plain records.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from fincomb.analytics.risk import drawdown_series
from fincomb.backtest.costs import FixedBpsCost
from fincomb.backtest.engine import (
    EXECUTIONS,
    EngineResult,
    Trade,
    benchmark_curve,
    min_start_index,
    run,
)
from fincomb.backtest.report import BacktestReport, build_report
from fincomb.backtest.reproducibility import code_version, data_hash
from fincomb.backtest.schedule import FREQUENCIES
from fincomb.db.models import BacktestRun
from fincomb.db.repositories import BacktestRepository
from fincomb.exceptions import BacktestNotFoundError, InsufficientDataError
from fincomb.services import portfolio_service
from fincomb.services.market_data import load_prices
from fincomb.strategies.registry import describe_all, get_strategy

DEFAULT_BENCHMARK = "SPY"
DEFAULT_CAPITAL = 100_000.0
DEFAULT_COST_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 5.0
MAX_BACKTEST_SYMBOLS = 50
PORTFOLIO_STRATEGIES = ("buy_and_hold", "fixed_weights")
METRIC_RTOL = 1e-9  # verify: relative tolerance when comparing a rerun's metrics


@dataclass(frozen=True)
class BacktestSpec:
    """Every input of a backtest."""

    strategy: str
    symbols: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    start: date | None = None
    end: date | None = None
    benchmark: str | None = DEFAULT_BENCHMARK
    initial_capital: float = DEFAULT_CAPITAL
    cost_bps: float = DEFAULT_COST_BPS
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    rebalance: str | None = None  # default: the strategy's own
    execution: str = "next_close"
    portfolio: int | str | None = None  # saved portfolio for buy_and_hold / fixed_weights
    risk_free: float = 0.0
    save: bool = True


@dataclass(frozen=True)
class BacktestRecord:
    id: int | None  # None when not saved
    strategy: str
    params: dict[str, Any]
    symbols: list[str]
    benchmark: str | None
    start: date
    end: date
    requested_start: date | None
    requested_end: date | None
    warmup_start: date | None
    rebalance: str
    execution: str
    initial_capital: float
    cost_bps: float
    slippage_bps: float
    risk_free: float
    portfolio_id: int | None
    code_version: str
    data_hash: str
    metrics: dict[str, Any]
    warnings: list[str]
    notes: list[str]
    duration_ms: int | None
    created_at: datetime | None


@dataclass
class BacktestOutcome:
    record: BacktestRecord
    report: BacktestReport
    result: EngineResult
    benchmark_equity: pd.Series | None
    equity: pd.DataFrame  # the rows stored in backtest_equity


@dataclass(frozen=True)
class TradeRecord:
    date: date
    symbol: str
    kind: str
    side: str
    weight_before: float
    weight_after: float
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    commission: float
    slippage_cost: float


@dataclass(frozen=True)
class VerifyResult:
    run_id: int
    reproducible: bool
    data_hash_matches: bool
    metrics_match: bool
    stored_data_hash: str
    current_data_hash: str
    stored_code_version: str
    current_code_version: str
    mismatched_metrics: list[str]


def strategies() -> list[dict[str, Any]]:
    return describe_all()


# --- running ---------------------------------------------------------------------------------


def _check_spec(spec: BacktestSpec) -> None:
    if spec.execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {', '.join(EXECUTIONS)}")
    if spec.rebalance is not None and spec.rebalance not in FREQUENCIES:
        raise ValueError(f"rebalance must be one of {', '.join(FREQUENCIES)}")
    if not math.isfinite(spec.initial_capital) or spec.initial_capital <= 0:
        raise ValueError("initial_capital must be a positive number")
    if spec.start and spec.end and spec.start > spec.end:
        raise ValueError(f"start {spec.start} is after end {spec.end}")
    if not -1 < spec.risk_free < 1:
        raise ValueError("risk_free must be an annual rate between -1 and 1")


def _resolve_portfolio(
    session: Session, spec: BacktestSpec, strategy_name: str
) -> tuple[list[str], dict[str, Any], portfolio_service.PortfolioRecord | None]:
    symbols = [s.strip().upper() for s in spec.symbols if s and s.strip()]
    params = dict(spec.params)
    if spec.portfolio is None:
        return symbols, params, None
    if strategy_name not in PORTFOLIO_STRATEGIES:
        raise ValueError(f"a saved portfolio can only be backtested with {PORTFOLIO_STRATEGIES}")
    if params.get("weights") is not None:
        raise ValueError("give either a portfolio or weights, not both")
    record = portfolio_service.get_portfolio(session, spec.portfolio)
    if symbols and set(symbols) != set(record.weights):
        raise ValueError(
            f"portfolio {record.name!r} holds {', '.join(record.weights)}; leave symbols empty "
            "or list exactly those"
        )
    params["weights"] = dict(record.weights)
    return list(record.weights), params, record


def run_backtest(session: Session, spec: BacktestSpec) -> BacktestOutcome:
    """Run `spec` on stored prices; store it unless `spec.save` is False."""
    started = time.perf_counter()
    _check_spec(spec)
    cls = get_strategy(spec.strategy)
    symbols, params, portfolio = _resolve_portfolio(session, spec, cls.name)
    if not symbols:
        raise ValueError("give at least one symbol (or a saved portfolio)")
    symbols = portfolio_service._normalise_symbols(symbols)
    if len(symbols) > MAX_BACKTEST_SYMBOLS:
        raise ValueError(f"at most {MAX_BACKTEST_SYMBOLS} symbols per backtest")
    strategy = cls(symbols, params)
    rebalance = spec.rebalance or cls.default_rebalance
    if cls.name == "buy_and_hold" and rebalance != "never":
        raise ValueError("buy_and_hold never rebalances; use fixed_weights to rebalance")
    benchmark = (
        spec.benchmark.strip().upper() if spec.benchmark and spec.benchmark.strip() else None
    )

    # Load everything up to `end`, from the start of the stored history, so the strategy
    # can warm up before `start`. Universe and benchmark share one inner-aligned calendar.
    universe = list(symbols) + ([benchmark] if benchmark and benchmark not in symbols else [])
    adj_all = load_prices(session, universe, None, spec.end, field="adj_close", align="inner")
    raw_all = load_prices(session, universe, None, spec.end, field="close", align="inner")
    notes = list(adj_all.attrs.get("alignment_notes", []))
    index = adj_all.index

    need = min_start_index(strategy, spec.execution)
    day0 = 0 if spec.start is None else int(index.searchsorted(pd.Timestamp(spec.start)))
    if day0 >= len(index):
        raise InsufficientDataError(f"no common prices on or after {spec.start}")
    if day0 < need:
        if need >= len(index) - 1:
            raise InsufficientDataError(
                f"{cls.name} needs {strategy.warmup_bars} trading days of history before its "
                f"first signal, but only {len(index)} common dates are stored"
            )
        if spec.start is not None:
            notes.append(
                f"Effective start moved from {spec.start} to {index[need].date()}: {cls.name} "
                f"needs {strategy.warmup_bars} trading days of history before its first signal."
            )
        day0 = need
    if day0 >= len(index) - 1:
        raise InsufficientDataError("the evaluation period needs at least 2 common price dates")
    first = day0 - need
    if first > 0:
        notes = [n for n in notes if not n.startswith("Common period starts")]
    adj, raw = adj_all.iloc[first:], raw_all.iloc[first:]
    day0_date = index[day0]

    costs = FixedBpsCost(spec.cost_bps, spec.slippage_bps)
    result = run(
        adj[symbols],
        strategy,
        day0_date,
        raw_prices=raw[symbols],
        rebalance=rebalance,
        initial_capital=spec.initial_capital,
        costs=costs,
        execution=spec.execution,
    )
    bench_eq = (
        benchmark_curve(adj[benchmark], day0_date, spec.initial_capital) if benchmark else None
    )
    diagnostics = strategy.diagnostics(adj.loc[day0_date:, symbols])
    report = build_report(result, bench_eq, spec.risk_free, diagnostics)

    warnings = strategy.run_warnings()
    if (
        portfolio is not None
        and portfolio.estimation_end
        and day0_date.date() <= portfolio.estimation_end
    ):
        warnings.append(
            f"Portfolio {portfolio.name!r} was estimated on {portfolio.estimation_start} to "
            f"{portfolio.estimation_end}, which overlaps this backtest: results over that "
            "stretch are in-sample and flattering."
        )

    frames: dict[str, Any] = {"adj_close": adj[symbols], "close": raw[symbols]}
    if benchmark:
        frames["benchmark"] = adj[[benchmark]]
    equity = _equity_frame(result, bench_eq)
    record = BacktestRecord(
        id=None,
        strategy=cls.name,
        params=report_params(strategy.params),
        symbols=symbols,
        benchmark=benchmark,
        start=day0_date.date(),
        end=index[-1].date(),
        requested_start=spec.start,
        requested_end=spec.end,
        warmup_start=adj.index[0].date(),
        rebalance=rebalance,
        execution=spec.execution,
        initial_capital=float(spec.initial_capital),
        cost_bps=float(spec.cost_bps),
        slippage_bps=float(spec.slippage_bps),
        risk_free=float(spec.risk_free),
        portfolio_id=portfolio.id if portfolio else None,
        code_version=code_version(),
        data_hash=data_hash(frames),
        metrics=report.to_dict(),
        warnings=warnings,
        notes=notes,
        duration_ms=int((time.perf_counter() - started) * 1000),
        created_at=None,
    )
    if spec.save:
        record = _save(session, record, equity, result.trades)
    return BacktestOutcome(record, report, result, bench_eq, equity)


def report_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strategy parameters as stored (JSON types)."""
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()}


def _equity_frame(result: EngineResult, bench: pd.Series | None) -> pd.DataFrame:
    eq = result.equity
    return pd.DataFrame(
        {
            "equity": eq,
            "cash": result.cash,
            "gross_exposure": result.gross_exposure,
            "benchmark_equity": bench.reindex(eq.index) if bench is not None else float("nan"),
            "drawdown": drawdown_series(eq.pct_change().fillna(0.0)),
            "turnover": result.turnover,
            "costs": result.costs,
        }
    )


def _nan_none(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def _save(
    session: Session, record: BacktestRecord, equity: pd.DataFrame, trades: list[Trade]
) -> BacktestRecord:
    row = BacktestRun(
        strategy=record.strategy,
        params=record.params,
        symbols=record.symbols,
        benchmark_symbol=record.benchmark,
        start_date=record.start,
        end_date=record.end,
        requested_start=record.requested_start,
        requested_end=record.requested_end,
        warmup_start=record.warmup_start,
        rebalance_frequency=record.rebalance,
        execution=record.execution,
        initial_capital=record.initial_capital,
        cost_bps=record.cost_bps,
        slippage_bps=record.slippage_bps,
        risk_free=record.risk_free,
        portfolio_id=record.portfolio_id,
        code_version=record.code_version,
        data_hash=record.data_hash,
        metrics=record.metrics,
        warnings=record.warnings,
        notes=record.notes,
        duration_ms=record.duration_ms,
    )
    equity_rows = [
        {
            "bar_date": ts.date(),
            "equity": float(r.equity),
            "cash": float(r.cash),
            "gross_exposure": float(r.gross_exposure),
            "benchmark_equity": _nan_none(r.benchmark_equity),
            "drawdown": float(r.drawdown),
            "turnover": float(r.turnover),
            "costs": float(r.costs),
        }
        for ts, r in equity.iterrows()
    ]
    trade_rows = [
        {
            "symbol": t.symbol,
            "bar_date": t.date.date(),
            "kind": t.kind,
            "side": t.side,
            "weight_before": t.weight_before,
            "weight_after": t.weight_after,
            "quantity": t.quantity,
            "reference_price": t.reference_price,
            "fill_price": t.fill_price,
            "notional": t.notional,
            "commission": t.commission,
            "slippage_cost": t.slippage_cost,
        }
        for t in trades
    ]
    BacktestRepository(session).create(row, equity_rows, trade_rows)
    session.refresh(row)
    return _record(row)


# --- stored runs -----------------------------------------------------------------------------


def _record(r: BacktestRun) -> BacktestRecord:
    return BacktestRecord(
        id=r.id,
        strategy=r.strategy,
        params=dict(r.params),
        symbols=list(r.symbols),
        benchmark=r.benchmark_symbol,
        start=r.start_date,
        end=r.end_date,
        requested_start=r.requested_start,
        requested_end=r.requested_end,
        warmup_start=r.warmup_start,
        rebalance=r.rebalance_frequency,
        execution=r.execution,
        initial_capital=r.initial_capital,
        cost_bps=r.cost_bps,
        slippage_bps=r.slippage_bps,
        risk_free=r.risk_free,
        portfolio_id=r.portfolio_id,
        code_version=r.code_version,
        data_hash=r.data_hash,
        metrics=dict(r.metrics),
        warnings=list(r.warnings),
        notes=list(r.notes),
        duration_ms=r.duration_ms,
        created_at=r.created_at,
    )


def _find(session: Session, run_id: int) -> BacktestRun:
    row = BacktestRepository(session).get(int(run_id))
    if row is None:
        raise BacktestNotFoundError(f"no backtest run {run_id}")
    return row


def get_backtest(session: Session, run_id: int) -> BacktestRecord:
    return _record(_find(session, run_id))


def list_backtests(
    session: Session, strategy: str | None = None, limit: int = 50
) -> list[BacktestRecord]:
    if strategy:
        strategy = get_strategy(strategy).name
    return [_record(r) for r in BacktestRepository(session).list_runs(strategy, limit)]


def delete_backtest(session: Session, run_id: int) -> BacktestRecord:
    row = _find(session, run_id)
    record = _record(row)
    BacktestRepository(session).delete(row)
    return record


def backtest_equity(session: Session, run_id: int) -> pd.DataFrame:
    """Stored daily curve: equity, cash, exposure, benchmark, drawdown, turnover, costs."""
    _find(session, run_id)
    rows = BacktestRepository(session).equity(run_id)
    cols = [
        "equity",
        "cash",
        "gross_exposure",
        "benchmark_equity",
        "drawdown",
        "turnover",
        "costs",
    ]
    frame = pd.DataFrame(
        [[getattr(r, c) for c in cols] for r in rows],
        index=pd.DatetimeIndex([pd.Timestamp(r.bar_date) for r in rows], name="date"),
        columns=cols,
        dtype="float64",
    )
    return frame


def backtest_trades(
    session: Session, run_id: int, limit: int | None = None, offset: int = 0
) -> tuple[list[TradeRecord], int]:
    """(trades in date order, total number of trades)."""
    _find(session, run_id)
    repo = BacktestRepository(session)
    out = [
        TradeRecord(
            date=t.bar_date,
            symbol=symbol,
            kind=t.kind,
            side=t.side,
            weight_before=t.weight_before,
            weight_after=t.weight_after,
            quantity=t.quantity,
            reference_price=t.reference_price,
            fill_price=t.fill_price,
            notional=t.notional,
            commission=t.commission,
            slippage_cost=t.slippage_cost,
        )
        for t, symbol in repo.trades(run_id, limit, offset)
    ]
    return out, repo.trade_count(run_id)


def spec_from_record(record: BacktestRecord) -> BacktestSpec:
    """The spec that reruns a stored backtest over exactly its effective period."""
    return BacktestSpec(
        strategy=record.strategy,
        symbols=list(record.symbols),
        params=dict(record.params),
        start=record.start,
        end=record.end,
        benchmark=record.benchmark,
        initial_capital=record.initial_capital,
        cost_bps=record.cost_bps,
        slippage_bps=record.slippage_bps,
        rebalance=record.rebalance,
        execution=record.execution,
        risk_free=record.risk_free,
        save=False,
    )


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            _diff(a.get(key), b.get(key), f"{path}.{key}" if path else key, out)
    elif isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
        if not math.isclose(a, b, rel_tol=METRIC_RTOL, abs_tol=1e-12):
            out.append(path)
    elif a != b:
        out.append(path)


def verify_backtest(session: Session, run_id: int) -> VerifyResult:
    """Rerun a stored backtest from current data and compare hash and metrics."""
    record = get_backtest(session, run_id)
    rerun = run_backtest(session, replace(spec_from_record(record), save=False)).record
    mismatched: list[str] = []
    _diff(record.metrics, rerun.metrics, "", mismatched)
    hash_ok = rerun.data_hash == record.data_hash
    return VerifyResult(
        run_id=record.id or run_id,
        reproducible=hash_ok and not mismatched,
        data_hash_matches=hash_ok,
        metrics_match=not mismatched,
        stored_data_hash=record.data_hash,
        current_data_hash=rerun.data_hash,
        stored_code_version=record.code_version,
        current_code_version=rerun.code_version,
        mismatched_metrics=mismatched,
    )
