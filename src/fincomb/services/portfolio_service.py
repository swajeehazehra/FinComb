"""Build, save and analyse portfolios. Shared by the CLI and the API.

No maths here: weights come from `fincomb.portfolio.construction`, risk from
`fincomb.portfolio.analysis`. This module loads prices, picks the estimation
window, validates input, and turns saved portfolios into plain records so callers
never hold database objects after the session closes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd
from sqlalchemy.orm import Session

from fincomb.analytics.returns import simple_returns
from fincomb.db.models import Portfolio, PortfolioMethod
from fincomb.db.repositories import InstrumentRepository, PortfolioRepository, PriceRepository
from fincomb.exceptions import PortfolioExistsError, PortfolioNotFoundError, SymbolNotFoundError
from fincomb.portfolio.analysis import RiskReport, check_weights, risk_contributions, risk_report
from fincomb.portfolio.construction import ConstructionResult, WeightBounds, construct
from fincomb.portfolio.covariance import sample_covariance
from fincomb.services.market_data import load_prices

DEFAULT_LOOKBACK = 252  # daily returns in the estimation window when no start is given
DEFAULT_BENCHMARK = "SPY"
MAX_NAME_LENGTH = 100


@dataclass(frozen=True)
class ConstructionSpec:
    """Everything that determines a constructed portfolio's weights."""

    symbols: list[str]
    method: str
    start: date | None = None  # explicit window start (then `lookback` is ignored)
    end: date | None = None  # default: the latest date every symbol has
    lookback: int = DEFAULT_LOOKBACK
    min_weight: float = 0.0
    max_weight: float = 1.0
    covariance: str = "ledoit_wolf"
    risk_free: float = 0.0


@dataclass
class ConstructOutcome:
    result: ConstructionResult
    contributions: pd.DataFrame  # risk contributions over the estimation window
    notes: list[str] = field(default_factory=list)  # alignment / window notes


@dataclass(frozen=True)
class PortfolioRecord:
    id: int
    name: str
    description: str | None
    construction_method: str
    effective_method: str
    construction_params: dict
    estimation_start: date | None
    estimation_end: date | None
    created_at: datetime
    weights: dict[str, float]

    @property
    def warnings(self) -> list[str]:
        return list(self.construction_params.get("warnings", []))


@dataclass
class RiskOutcome:
    portfolio: PortfolioRecord | None  # None for unsaved weights
    weights: pd.Series
    report: RiskReport
    notes: list[str] = field(default_factory=list)


def _normalise_symbols(symbols: list[str]) -> list[str]:
    out = [s.strip().upper() for s in symbols if s and s.strip()]
    if not out:
        raise ValueError("give at least one symbol")
    dupes = sorted({s for s in out if out.count(s) > 1})
    if dupes:
        raise ValueError(f"each symbol may appear only once (repeated: {', '.join(dupes)})")
    return out


# --- construction --------------------------------------------------------------------------


def _load_window(
    session: Session, spec: ConstructionSpec, symbols: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Prices for the estimation window: [start, end], or the last `lookback` returns."""
    if spec.start is None and spec.lookback < 2:
        raise ValueError("lookback must be at least 2 returns")
    prices = load_prices(session, symbols, spec.start, spec.end, align="inner")
    notes = list(prices.attrs.get("alignment_notes", []))
    if spec.start is None and len(prices) > spec.lookback + 1:
        prices = prices.iloc[-(spec.lookback + 1) :]
        # The window no longer reaches the start of the common history, so a note about
        # which symbol limits that start doesn't apply.
        notes = [n for n in notes if not n.startswith("Common period starts")]
    elif spec.start is None and len(prices) < spec.lookback + 1:
        notes.append(
            f"Only {len(prices) - 1} common daily returns are available, fewer than the "
            f"lookback of {spec.lookback}; the whole common history was used."
        )
    return prices, notes


def construct_portfolio(session: Session, spec: ConstructionSpec) -> ConstructOutcome:
    """Weights by `spec.method` from stored prices. Nothing is saved."""
    symbols = _normalise_symbols(spec.symbols)
    bounds = WeightBounds(spec.min_weight, spec.max_weight)
    prices, notes = _load_window(session, spec, symbols)
    result = construct(
        prices, spec.method, bounds, covariance=spec.covariance, risk_free=spec.risk_free
    )
    contributions = risk_contributions(result.weights, sample_covariance(simple_returns(prices)))
    return ConstructOutcome(result, contributions, notes)


def _json_float(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def construction_params(spec: ConstructionSpec, outcome: ConstructOutcome) -> dict:
    """JSON-safe record of how the weights were made (stored with the portfolio)."""
    r = outcome.result
    return {
        "symbols": list(r.weights.index),
        "requested_start": spec.start.isoformat() if spec.start else None,
        "requested_end": spec.end.isoformat() if spec.end else None,
        "lookback": None if spec.start else spec.lookback,
        "observations": r.observations,
        "min_weight": r.bounds.min_weight,
        "max_weight": r.bounds.max_weight,
        "covariance": r.covariance,
        "shrinkage": _json_float(r.shrinkage),
        "ridge": r.ridge,
        "risk_free": r.risk_free,
        "optimizer": None
        if r.optimizer is None
        else {
            "success": r.optimizer.success,
            "message": r.optimizer.message,
            "iterations": r.optimizer.iterations,
        },
        "fallback_used": r.fallback_used,
        "fallback_reason": r.fallback_reason,
        "non_positive_excess_return": r.non_positive_excess_return,
        "in_sample": {
            "expected_return": _json_float(r.expected_return),
            "volatility": _json_float(r.volatility),
            "sharpe": _json_float(r.sharpe),
        },
        "warnings": list(r.warnings),
        "notes": list(outcome.notes),
    }


# --- saved portfolios ----------------------------------------------------------------------


def _record(p: Portfolio) -> PortfolioRecord:
    return PortfolioRecord(
        id=p.id,
        name=p.name,
        description=p.description,
        construction_method=p.construction_method.value,
        effective_method=p.effective_method.value,
        construction_params=dict(p.construction_params or {}),
        estimation_start=p.estimation_start,
        estimation_end=p.estimation_end,
        created_at=p.created_at,
        weights=PortfolioRepository.weights_of(p),
    )


def _check_name(session: Session, name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("a portfolio needs a name")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"portfolio names are at most {MAX_NAME_LENGTH} characters")
    if PortfolioRepository(session).get_by_name(name) is not None:
        raise PortfolioExistsError(f"a portfolio named {name!r} already exists")
    return name


def _check_stored(session: Session, symbols: list[str]) -> None:
    instruments, prices = InstrumentRepository(session), PriceRepository(session)
    for symbol in symbols:
        inst = instruments.get_by_symbol(symbol)
        if inst is None or prices.stored_range(inst.id)[2] == 0:
            raise SymbolNotFoundError(
                f"no stored prices for {symbol}; run `fincomb ingest {symbol}` first"
            )


def manual_weights(weights: dict[str, float]) -> pd.Series:
    """Validate user-given weights: unique symbols, long-only, summing to 1 (+/- 1e-6).
    Never normalised: a typo should fail, not be quietly fixed."""
    if not weights:
        raise ValueError("give at least one weight")
    symbols = _normalise_symbols(list(weights))
    series = pd.Series(
        [float(v) for v in weights.values()], index=pd.Index(symbols, name="symbol"), name="weight"
    )
    return check_weights(series)


def create_manual(
    session: Session, name: str, weights: dict[str, float], description: str | None = None
) -> PortfolioRecord:
    name = _check_name(session, name)
    series = manual_weights(weights)
    _check_stored(session, list(series.index))
    p = PortfolioRepository(session).create(
        name,
        series.to_dict(),
        PortfolioMethod.MANUAL,
        PortfolioMethod.MANUAL,
        params={"warnings": [], "notes": []},
        description=description,
    )
    return _record(p)


def create_constructed(
    session: Session, name: str, spec: ConstructionSpec, description: str | None = None
) -> tuple[PortfolioRecord, ConstructOutcome]:
    """Construct on the server and save weights + parameters (clients never send
    'optimised' weights)."""
    name = _check_name(session, name)
    outcome = construct_portfolio(session, spec)
    r = outcome.result
    p = PortfolioRepository(session).create(
        name,
        r.weights.to_dict(),
        PortfolioMethod(r.method),
        PortfolioMethod(r.effective_method),
        params=construction_params(spec, outcome),
        description=description,
        estimation_start=r.start.date(),
        estimation_end=r.end.date(),
    )
    return _record(p), outcome


def _find(session: Session, key: int | str) -> Portfolio:
    repo = PortfolioRepository(session)
    p = None
    if isinstance(key, int):
        p = repo.get(key)
    else:
        p = repo.get_by_name(key.strip())
        if p is None and key.strip().isdigit():
            p = repo.get(int(key.strip()))
    if p is None:
        raise PortfolioNotFoundError(f"no portfolio {key!r}")
    return p


def get_portfolio(session: Session, key: int | str) -> PortfolioRecord:
    """By id (int), or by name (str; a string of digits that is no name is tried as an id)."""
    return _record(_find(session, key))


def list_portfolios(session: Session) -> list[PortfolioRecord]:
    return [_record(p) for p in PortfolioRepository(session).list_all()]


def delete_portfolio(session: Session, key: int | str) -> PortfolioRecord:
    p = _find(session, key)
    record = _record(p)
    PortfolioRepository(session).delete(p)
    return record


# --- risk ---------------------------------------------------------------------------------


def _risk(
    session: Session,
    weights: pd.Series,
    start: date | None,
    end: date | None,
    benchmark: str | None,
    confidence: float,
    risk_free: float,
) -> tuple[RiskReport, list[str]]:
    benchmark = benchmark.strip().upper() if benchmark and benchmark.strip() else None
    symbols = list(weights.index)
    if benchmark and benchmark not in symbols:
        symbols.append(benchmark)
    prices = load_prices(session, symbols, start, end, align="inner")
    notes = list(prices.attrs.get("alignment_notes", []))
    report = risk_report(prices, weights, benchmark, confidence, risk_free)
    return report, notes


def portfolio_risk(
    session: Session,
    key: int | str,
    start: date | None = None,
    end: date | None = None,
    benchmark: str | None = DEFAULT_BENCHMARK,
    confidence: float = 0.95,
    risk_free: float = 0.0,
) -> RiskOutcome:
    """Risk report for a saved portfolio over [start, end] (default: all common dates)."""
    record = get_portfolio(session, key)
    weights = pd.Series(record.weights, name="weight", dtype="float64")
    weights.index.name = "symbol"
    report, notes = _risk(session, weights, start, end, benchmark, confidence, risk_free)
    return RiskOutcome(record, weights, report, notes)


def weights_risk(
    session: Session,
    weights: dict[str, float],
    start: date | None = None,
    end: date | None = None,
    benchmark: str | None = DEFAULT_BENCHMARK,
    confidence: float = 0.95,
    risk_free: float = 0.0,
) -> RiskOutcome:
    """The same report for weights that are not saved (e.g. a portfolio being designed)."""
    series = manual_weights(weights)
    report, notes = _risk(session, series, start, end, benchmark, confidence, risk_free)
    return RiskOutcome(None, series, report, notes)
