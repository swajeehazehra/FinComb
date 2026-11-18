"""Ingestion pipeline: fetch -> validate -> store -> audit.

One bad symbol never aborts the run: each symbol is processed in its own
transaction and failures are recorded on the ingestion run.

Three rules keep every symbol's stored history consistent:

1. **Only completed bars.** The end date is capped at the last completed trading
   session, so an in-progress bar (whose "close" is just the latest price) is
   never stored.
2. **One source per symbol.** Each instrument records the source that owns its
   history. Ingesting it from a different source is refused; `fincomb refresh
   --source X` switches by replacing the whole history.
3. **No stitched adjustments.** Vendors re-adjust the whole history after each
   dividend or split. Every update re-downloads a few days that are already
   stored; if those prices changed, the adjustment changed, and the symbol's
   FULL history is downloaded again and replaces the old rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Literal

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from fincomb.db.base import session_scope
from fincomb.db.models import AssetType, DataQualityIssue, RunStatus, Severity
from fincomb.db.repositories import (
    IngestionRunRepository,
    InstrumentRepository,
    PriceRepository,
)
from fincomb.exceptions import DataOwnershipError, FinCombError, SymbolNotFoundError
from fincomb.ingestion.base import MarketDataSource
from fincomb.ingestion.calendar import last_completed_session
from fincomb.validation.rules import Issue, validate_bars

logger = logging.getLogger(__name__)


Mode = Literal["update", "replace"]

# How far back an update re-downloads already-stored days to detect re-adjustment.
OVERLAP_DAYS = 14
# Relative difference on overlapping days above which the history counts as re-adjusted.
# Small enough to catch a 0.01% dividend adjustment, far above float/rounding noise.
ADJUSTMENT_TOLERANCE = 1e-4


@dataclass
class SymbolResult:
    symbol: str
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    warnings: int = 0
    replaced: bool = False  # True if the whole stored history was rebuilt
    notes: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class IngestionSummary:
    run_id: int
    status: RunStatus
    results: list[SymbolResult] = field(default_factory=list)
    end_capped_to: date | None = None  # set when the requested end was after the last final bar
    source: str = ""  # name of the data source this run used

    @property
    def inserted(self) -> int:
        return sum(r.inserted for r in self.results)

    @property
    def updated(self) -> int:
        return sum(r.updated for r in self.results)

    @property
    def rejected(self) -> int:
        return sum(r.rejected for r in self.results)


@dataclass
class _StoredState:
    instrument_id: int | None
    owner: str | None
    sources: set[str]
    first: date | None
    last: date | None


def _stored_state(session: Session, symbol: str) -> _StoredState:
    instrument = InstrumentRepository(session).get_by_symbol(symbol)
    if instrument is None:
        return _StoredState(None, None, set(), None, None)
    prices = PriceRepository(session)
    first, last, _ = prices.stored_range(instrument.id)
    return _StoredState(
        instrument.id, instrument.data_source, prices.sources(instrument.id), first, last
    )


def check_ownership(symbol: str, state: _StoredState, source_name: str) -> None:
    """Refuse an update that would mix sources in one symbol's history."""
    if not state.sources:
        return  # nothing stored yet: this source will become the owner
    owner = state.owner
    if owner is None and len(state.sources) > 1:
        raise DataOwnershipError(
            f"{symbol} has stored bars from several sources ({', '.join(sorted(state.sources))}). "
            f"Rebuild it from one source first: fincomb refresh {symbol} --source {source_name}"
        )
    owner = owner or next(iter(state.sources))
    if owner != source_name:
        raise DataOwnershipError(
            f"{symbol}'s history comes from {owner}, not {source_name}. To switch sources, "
            f"replace the whole history: fincomb refresh {symbol} --source {source_name}"
        )


def find_adjustment_change(
    stored: pd.DataFrame, fetched: pd.DataFrame, tolerance: float = ADJUSTMENT_TOLERANCE
) -> str | None:
    """Compare overlapping days; describe the first re-adjusted price, or None if consistent."""
    common = stored.index.intersection(fetched.index)
    if common.empty:
        return None
    for col in ("adj_close", "close"):
        rel = (fetched.loc[common, col] / stored.loc[common, col] - 1.0).abs()
        if (rel > tolerance).any():
            day = rel.idxmax()
            return (
                f"{col} on {day.date()} changed from {stored.loc[day, col]:.4f} to "
                f"{fetched.loc[day, col]:.4f} ({rel.max():.4%}) since it was stored"
            )
    return None


def _issues(run_id: int, instrument_id: int, vr) -> list[DataQualityIssue]:
    return [
        DataQualityIssue(
            ingestion_run_id=run_id,
            instrument_id=instrument_id,
            bar_date=i.bar_date,
            rule=i.rule,
            severity=i.severity,
            detail=i.detail,
        )
        for i in vr.issues
    ]


def _ingest_symbol(
    factory: sessionmaker[Session],
    source: MarketDataSource,
    symbol: str,
    start: date | None,
    end: date,
    run_id: int,
    mode: Mode,
    asset_type: AssetType | None,
    validate_kwargs: dict,
) -> SymbolResult:
    res = SymbolResult(symbol)
    with session_scope(factory) as s:
        state = _stored_state(s, symbol)

    if mode == "update":
        check_ownership(symbol, state, source.name)
        if start is None:
            raise ValueError("start is required for an update")
        fetch_start, fetch_end = start, end
        if state.first is not None and state.last is not None:
            # Make sure the download overlaps days already stored, so that a
            # re-adjustment by the vendor is noticed (see find_adjustment_change).
            if end < state.first:  # requested range lies entirely before stored history
                fetch_end = min(state.first + timedelta(days=OVERLAP_DAYS), state.last)
            elif start > state.last:  # appending new days (the usual case)
                fetch_start = state.last - timedelta(days=OVERLAP_DAYS)
    else:  # replace: rebuild the whole history from this source
        fetch_start = start or state.first
        if fetch_start is None:
            raise ValueError(f"{symbol} has no stored history; give a start date to refresh it")
        fetch_end = end

    raw = source.fetch_daily(symbol, fetch_start, fetch_end)
    # `end` is already capped at the last completed session; drop anything later
    # defensively (some vendors return today's in-progress bar regardless).
    raw = raw.loc[raw.index <= pd.Timestamp(max(end, fetch_end))]
    if raw.empty:
        raise SymbolNotFoundError(f"{symbol}: no completed bars between {fetch_start} and {end}")
    vr = validate_bars(raw, **validate_kwargs)

    if mode == "update" and state.instrument_id is not None and state.last is not None:
        with session_scope(factory) as s:
            stored = PriceRepository(s).bars_for_instrument(
                state.instrument_id, vr.clean.index.min().date(), vr.clean.index.max().date()
            )
        change = find_adjustment_change(stored, vr.clean)
        if change:
            # The vendor re-adjusted history: patching a range would leave a fake jump
            # at the boundary, so rebuild the whole history instead.
            full_start = min(state.first, fetch_start)
            logger.warning("%s: %s; downloading full history from %s", symbol, change, full_start)
            res.notes.append(f"adjustment changed ({change}); full history re-downloaded")
            raw = source.fetch_daily(symbol, full_start, end)
            raw = raw.loc[raw.index <= pd.Timestamp(end)]
            vr = validate_bars(raw, **validate_kwargs)
            vr.issues.append(Issue(None, "adjustment_changed", Severity.WARNING, change))
            mode = "replace"

    with session_scope(factory) as s:
        instrument = InstrumentRepository(s).get_or_create(symbol, asset_type)
        prices = PriceRepository(s)
        if mode == "replace":
            removed = prices.delete_bars(instrument.id)
            up = prices.upsert_bars(instrument.id, vr.clean, source.name, run_id)
            res.replaced = True
            res.notes.append(
                f"history replaced: {removed} old rows removed, "
                f"{up.inserted} rows from {source.name}"
            )
        else:
            up = prices.upsert_bars(instrument.id, vr.clean, source.name, run_id)
        instrument.data_source = source.name
        IngestionRunRepository(s).add_issues(_issues(run_id, instrument.id, vr))

    res.inserted, res.updated, res.rejected = up.inserted, up.updated, vr.rejected
    res.warnings = sum(1 for i in vr.issues if i.severity is Severity.WARNING)
    return res


def run_ingestion(
    session_factory: sessionmaker[Session],
    source: MarketDataSource,
    symbols: list[str],
    start: date | None,
    end: date,
    *,
    mode: Mode = "update",
    asset_type: AssetType | None = None,
    extreme_return_threshold: float = 0.40,
    stale_price_run: int = 5,
    now: datetime | None = None,
) -> IngestionSummary:
    """Ingest symbols from one source.

    mode="update" (the `ingest` command) adds/updates bars in [start, end].
    mode="replace" (the `refresh` command) deletes each symbol's stored bars and
    downloads its history again from `source`, which becomes its owner. With
    start=None, each symbol is refreshed from its first stored date.
    """
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not symbols:
        raise ValueError("no symbols given")
    if mode not in ("update", "replace"):
        raise ValueError("mode must be 'update' or 'replace'")

    final_day = last_completed_session(now)
    capped = end > final_day
    end = min(end, final_day)
    if capped:
        logger.info("end date capped to %s: later bars are not final yet", end)

    with session_scope(session_factory) as s:
        run_id = IngestionRunRepository(s).create(source.name, symbols, start or end, end).id
    logger.info(
        "ingestion run %d (%s) started: %s via %s (%s to %s)",
        run_id,
        mode,
        ",".join(symbols),
        source.name,
        start or "stored start",
        end,
    )

    validate_kwargs = {
        "extreme_return_threshold": extreme_return_threshold,
        "stale_price_run": stale_price_run,
    }
    results: list[SymbolResult] = []
    for symbol in symbols:
        try:
            if start is not None and start > end:
                raise ValueError(f"start {start} is after the last completed session {end}")
            res = _ingest_symbol(
                session_factory,
                source,
                symbol,
                start,
                end,
                run_id,
                mode,
                asset_type,
                validate_kwargs,
            )
            logger.info(
                "%s: %d inserted, %d updated, %d rejected, %d warnings%s",
                symbol,
                res.inserted,
                res.updated,
                res.rejected,
                res.warnings,
                " (history replaced)" if res.replaced else "",
            )
        except (FinCombError, ValueError) as exc:
            res = SymbolResult(symbol, error=str(exc))
            logger.error("%s: %s", symbol, exc)
        except Exception as exc:  # unexpected: log the traceback but keep going
            res = SymbolResult(symbol, error=f"{type(exc).__name__}: {exc}")
            logger.exception("%s: unexpected error", symbol)
        results.append(res)

    failed = [r for r in results if r.error]
    status = (
        RunStatus.SUCCESS
        if not failed
        else RunStatus.FAILED
        if len(failed) == len(results)
        else RunStatus.PARTIAL
    )

    with session_scope(session_factory) as s:
        run = IngestionRunRepository(s).get(run_id)
        assert run is not None
        run.status = status
        run.finished_at = func.now()
        run.rows_inserted = sum(r.inserted for r in results)
        run.rows_updated = sum(r.updated for r in results)
        run.rows_rejected = sum(r.rejected for r in results)
        run.error_message = "; ".join(f"{r.symbol}: {r.error}" for r in failed) or None

    logger.info("ingestion run %d finished: %s", run_id, status.value)
    return IngestionSummary(
        run_id=run_id,
        status=status,
        results=results,
        end_capped_to=end if capped else None,
        source=source.name,
    )
