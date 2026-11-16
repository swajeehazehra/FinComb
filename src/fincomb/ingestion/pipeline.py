"""Ingestion pipeline: fetch -> validate -> upsert -> audit.

One bad symbol never aborts the run: each symbol is processed in its own
transaction and failures are recorded on the ingestion run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session, sessionmaker

from fincomb.db.base import session_scope
from fincomb.db.models import AssetType, DataQualityIssue, RunStatus
from fincomb.db.repositories import (
    IngestionRunRepository,
    InstrumentRepository,
    PriceRepository,
)
from fincomb.exceptions import FinCombError
from fincomb.ingestion.base import MarketDataSource
from fincomb.validation.rules import validate_bars

logger = logging.getLogger(__name__)


@dataclass
class SymbolResult:
    symbol: str
    inserted: int = 0
    updated: int = 0
    rejected: int = 0
    warnings: int = 0
    error: str | None = None


@dataclass
class IngestionSummary:
    run_id: int
    status: RunStatus
    results: list[SymbolResult] = field(default_factory=list)

    @property
    def inserted(self) -> int:
        return sum(r.inserted for r in self.results)

    @property
    def updated(self) -> int:
        return sum(r.updated for r in self.results)

    @property
    def rejected(self) -> int:
        return sum(r.rejected for r in self.results)


def run_ingestion(
    session_factory: sessionmaker[Session],
    source: MarketDataSource,
    symbols: list[str],
    start: date,
    end: date,
    *,
    asset_type: AssetType | None = None,
    extreme_return_threshold: float = 0.40,
    stale_price_run: int = 5,
) -> IngestionSummary:
    symbols = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    if not symbols:
        raise ValueError("no symbols given")

    with session_scope(session_factory) as s:
        run_id = IngestionRunRepository(s).create(source.name, symbols, start, end).id
    logger.info(
        "ingestion run %d started: %s via %s (%s to %s)",
        run_id,
        ",".join(symbols),
        source.name,
        start,
        end,
    )

    results: list[SymbolResult] = []
    for symbol in symbols:
        res = SymbolResult(symbol)
        try:
            raw = source.fetch_daily(symbol, start, end)
            vr = validate_bars(
                raw,
                extreme_return_threshold=extreme_return_threshold,
                stale_price_run=stale_price_run,
            )
            with session_scope(session_factory) as s:
                instrument = InstrumentRepository(s).get_or_create(symbol, asset_type)
                up = PriceRepository(s).upsert_bars(instrument.id, vr.clean, source.name, run_id)
                IngestionRunRepository(s).add_issues(
                    [
                        DataQualityIssue(
                            ingestion_run_id=run_id,
                            instrument_id=instrument.id,
                            bar_date=i.bar_date,
                            rule=i.rule,
                            severity=i.severity,
                            detail=i.detail,
                        )
                        for i in vr.issues
                    ]
                )
            res.inserted, res.updated, res.rejected = up.inserted, up.updated, vr.rejected
            res.warnings = sum(1 for i in vr.issues if i.severity.value == "warning")
            logger.info(
                "%s: %d inserted, %d updated, %d rejected, %d warnings",
                symbol,
                res.inserted,
                res.updated,
                res.rejected,
                res.warnings,
            )
        except FinCombError as exc:
            res.error = str(exc)
            logger.error("%s: %s", symbol, exc)
        except Exception as exc:  # unexpected: log the traceback but keep going
            res.error = f"{type(exc).__name__}: {exc}"
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
    return IngestionSummary(run_id=run_id, status=status, results=results)
