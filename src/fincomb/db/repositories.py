"""Data-access layer. All SQL lives here, so services never build queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from fincomb.db.models import AssetType, DataQualityIssue, IngestionRun, Instrument, PriceBar

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
# Postgres allows at most 65,535 bind parameters per statement (10 per row here).
UPSERT_CHUNK_ROWS = 1000


@dataclass(frozen=True)
class UpsertResult:
    inserted: int
    updated: int


@dataclass(frozen=True)
class Coverage:
    symbol: str
    asset_type: str
    first_date: date | None
    last_date: date | None
    bars: int


class InstrumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_symbol(self, symbol: str) -> Instrument | None:
        return self.session.scalar(select(Instrument).where(Instrument.symbol == symbol.upper()))

    def get_or_create(self, symbol: str, asset_type: AssetType = AssetType.STOCK) -> Instrument:
        existing = self.get_by_symbol(symbol)
        if existing is not None:
            return existing
        instrument = Instrument(symbol=symbol.upper(), asset_type=asset_type)
        self.session.add(instrument)
        self.session.flush()  # assigns instrument.id
        return instrument

    def coverage(self) -> list[Coverage]:
        stmt = (
            select(
                Instrument.symbol,
                Instrument.asset_type,
                func.min(PriceBar.bar_date),
                func.max(PriceBar.bar_date),
                func.count(PriceBar.bar_date),
            )
            .outerjoin(PriceBar, PriceBar.instrument_id == Instrument.id)
            .group_by(Instrument.id)
            .order_by(Instrument.symbol)
        )
        return [
            Coverage(sym, at.value, first, last, n)
            for sym, at, first, last, n in self.session.execute(stmt)
        ]


class PriceRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_bars(
        self, instrument_id: int, bars: pd.DataFrame, source: str, run_id: int | None
    ) -> UpsertResult:
        """Insert or update daily bars. Idempotent: re-running never duplicates rows.

        `bars` is indexed by date with the columns in PRICE_COLUMNS.
        """
        if bars.empty:
            return UpsertResult(0, 0)

        records = [
            {
                "instrument_id": instrument_id,
                "bar_date": idx.date() if hasattr(idx, "date") else idx,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "adj_close": float(row.adj_close),
                "volume": None if pd.isna(row.volume) else int(row.volume),
                "source": source,
                "ingestion_run_id": run_id,
            }
            for idx, row in bars[PRICE_COLUMNS].iterrows()
        ]

        inserted = updated = 0
        for i in range(0, len(records), UPSERT_CHUNK_ROWS):
            stmt = pg_insert(PriceBar).values(records[i : i + UPSERT_CHUNK_ROWS])
            stmt = stmt.on_conflict_do_update(
                index_elements=[PriceBar.instrument_id, PriceBar.bar_date],
                set_={
                    **{c: stmt.excluded[c] for c in PRICE_COLUMNS},
                    "source": stmt.excluded.source,
                    "ingestion_run_id": stmt.excluded.ingestion_run_id,
                    "ingested_at": func.now(),
                },
            )
            # Postgres trick: xmax = 0 only for freshly inserted rows.
            stmt = stmt.returning(literal_column("(xmax = 0)").label("inserted"))
            flags = self.session.execute(stmt).scalars().all()
            n_new = sum(1 for f in flags if f)
            inserted += n_new
            updated += len(flags) - n_new
        return UpsertResult(inserted=inserted, updated=updated)

    def get_bars(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        stmt = (
            select(PriceBar.bar_date, *[getattr(PriceBar, c) for c in PRICE_COLUMNS])
            .join(Instrument)
            .where(Instrument.symbol == symbol.upper())
            .order_by(PriceBar.bar_date)
        )
        if start:
            stmt = stmt.where(PriceBar.bar_date >= start)
        if end:
            stmt = stmt.where(PriceBar.bar_date <= end)
        rows = self.session.execute(stmt).all()
        df = pd.DataFrame(rows, columns=["date", *PRICE_COLUMNS])
        df["date"] = pd.to_datetime(df["date"]).dt.as_unit("ns")
        df[PRICE_COLUMNS] = df[PRICE_COLUMNS].astype("float64")  # same dtypes as the sources
        return df.set_index("date")


class IngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, source: str, symbols: list[str], start: date, end: date) -> IngestionRun:
        run = IngestionRun(source=source, symbols=symbols, start_date=start, end_date=end)
        self.session.add(run)
        self.session.flush()
        return run

    def add_issues(self, issues: list[DataQualityIssue]) -> None:
        self.session.add_all(issues)

    def get(self, run_id: int) -> IngestionRun | None:
        return self.session.get(IngestionRun, run_id)
