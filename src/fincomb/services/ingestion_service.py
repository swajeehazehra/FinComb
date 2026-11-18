"""Choosing data sources and running ingestion. Shared by the CLI and the API.

The rules (one owning source per symbol, completed bars only, full re-download on
re-adjustment) live in `fincomb.ingestion.pipeline`; this module only decides
which source each symbol is ingested from and runs one pipeline call per source.
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy.orm import Session, sessionmaker

from fincomb.config import Settings
from fincomb.db.base import session_scope
from fincomb.db.models import AssetType
from fincomb.db.repositories import InstrumentRepository
from fincomb.ingestion.pipeline import IngestionSummary, Mode, run_ingestion
from fincomb.ingestion.registry import get_source


def normalise_symbols(symbols: list[str]) -> list[str]:
    """Strip, upper-case and de-duplicate, keeping the given order."""
    return list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))


def owners(factory: sessionmaker[Session], symbols: list[str]) -> dict[str, str | None]:
    """The source that currently owns each symbol's history (None = not stored yet)."""
    with session_scope(factory) as s:
        repo = InstrumentRepository(s)
        out = {}
        for sym in symbols:
            inst = repo.get_by_symbol(sym)
            out[sym.upper()] = inst.data_source if inst else None
        return out


def stored_symbols(factory: sessionmaker[Session]) -> list[str]:
    with session_scope(factory) as s:
        return InstrumentRepository(s).all_symbols()


def group_by_source(
    symbols: list[str], explicit: str | None, owners: dict[str, str | None], default: str
) -> dict[str, list[str]]:
    """An explicit source wins; otherwise each symbol keeps its owner; else the default."""
    groups: dict[str, list[str]] = {}
    for sym in normalise_symbols(symbols):
        name = explicit or owners.get(sym) or default
        groups.setdefault(name, []).append(sym)
    return groups


def ingest(
    factory: sessionmaker[Session],
    settings: Settings,
    symbols: list[str],
    start: date | None,
    end: date,
    *,
    source: str | None = None,
    mode: Mode = "update",
    asset_type: AssetType | None = None,
    now: datetime | None = None,
) -> list[IngestionSummary]:
    """Ingest (mode="update") or rebuild (mode="replace") symbols.

    Symbols are grouped by source (see `group_by_source`) and each group becomes one
    ingestion run, so the result has one summary per source used. Per-symbol
    failures, such as a source-ownership conflict, are reported inside the
    summaries, exactly as the pipeline records them.
    """
    groups = group_by_source(
        symbols, source, owners(factory, symbols), settings.default_data_source
    )
    if not groups:
        raise ValueError("no symbols given")
    summaries = []
    for name, syms in groups.items():
        summaries.append(
            run_ingestion(
                factory,
                get_source(name, settings),
                syms,
                start,
                end,
                mode=mode,
                asset_type=asset_type,
                extreme_return_threshold=settings.extreme_return_threshold,
                stale_price_run=settings.stale_price_run,
                now=now,
            )
        )
    return summaries
