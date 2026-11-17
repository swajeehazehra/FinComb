"""Audit of stored price history (used by `fincomb check-data`)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from fincomb.db.repositories import InstrumentRepository, PriceRepository


@dataclass(frozen=True)
class AuditRow:
    symbol: str
    owner: str | None
    sources: list[str]
    bars: int
    jumps: int  # daily adjusted moves larger than the extreme-return threshold
    status: str


def audit_data(session: Session, jump_threshold: float = 0.40) -> list[AuditRow]:
    """One row per instrument. Status is OK, MIXED SOURCES, NO OWNER, OWNER MISMATCH,
    or SUSPICIOUS JUMPS (which usually means a split was stitched into the history)."""
    instruments = InstrumentRepository(session)
    prices = PriceRepository(session)
    rows = []
    for symbol in instruments.all_symbols():
        inst = instruments.get_by_symbol(symbol)
        assert inst is not None
        sources = sorted(prices.sources(inst.id))
        bars = prices.bars_for_instrument(inst.id)
        jumps = int((bars["adj_close"].pct_change(fill_method=None).abs() > jump_threshold).sum())
        if len(sources) > 1:
            status = "MIXED SOURCES"
        elif sources and inst.data_source is None:
            status = "NO OWNER"
        elif sources and inst.data_source != sources[0]:
            status = "OWNER MISMATCH"
        elif jumps:
            status = "SUSPICIOUS JUMPS"
        else:
            status = "OK"
        rows.append(AuditRow(symbol, inst.data_source, sources, len(bars), jumps, status))
    return rows
