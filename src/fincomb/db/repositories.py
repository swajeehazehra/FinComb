"""Data-access layer. All SQL lives here, so services never build queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import delete, func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, selectinload

from fincomb.db.models import (
    AssetType,
    BacktestEquity,
    BacktestRun,
    BacktestTrade,
    DataQualityIssue,
    IngestionRun,
    Instrument,
    Portfolio,
    PortfolioMethod,
    PortfolioWeight,
    PriceBar,
)

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

    def get_or_create(self, symbol: str, asset_type: AssetType | None = None) -> Instrument:
        """Return the instrument, creating it if needed.

        A new instrument defaults to "stock". If asset_type is given explicitly,
        an existing instrument is updated to it (so a wrong label can be fixed).
        """
        existing = self.get_by_symbol(symbol)
        if existing is not None:
            if asset_type is not None and existing.asset_type is not asset_type:
                existing.asset_type = asset_type
            return existing
        instrument = Instrument(symbol=symbol.upper(), asset_type=asset_type or AssetType.STOCK)
        self.session.add(instrument)
        self.session.flush()  # assigns instrument.id
        return instrument

    def set_asset_type(self, symbol: str, asset_type: AssetType) -> Instrument | None:
        instrument = self.get_by_symbol(symbol)
        if instrument is not None:
            instrument.asset_type = asset_type
        return instrument

    def list_instruments(
        self, asset_type: AssetType | None = None, active: bool | None = None
    ) -> list[Instrument]:
        stmt = select(Instrument).order_by(Instrument.symbol)
        if asset_type is not None:
            stmt = stmt.where(Instrument.asset_type == asset_type)
        if active is not None:
            stmt = stmt.where(Instrument.is_active == active)
        return list(self.session.scalars(stmt))

    def all_symbols(self) -> list[str]:
        return list(self.session.scalars(select(Instrument.symbol).order_by(Instrument.symbol)))

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

    def stored_range(self, instrument_id: int) -> tuple[date | None, date | None, int]:
        """(first date, last date, number of bars) stored for an instrument."""
        first, last, n = self.session.execute(
            select(func.min(PriceBar.bar_date), func.max(PriceBar.bar_date), func.count()).where(
                PriceBar.instrument_id == instrument_id
            )
        ).one()
        return first, last, n

    def sources(self, instrument_id: int) -> set[str]:
        """Distinct data sources found in an instrument's stored bars."""
        return set(
            self.session.scalars(
                select(PriceBar.source).where(PriceBar.instrument_id == instrument_id).distinct()
            )
        )

    def bars_for_instrument(
        self, instrument_id: int, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """Stored bars indexed by date (same shape as a data source's frame)."""
        stmt = select(PriceBar.bar_date, *[getattr(PriceBar, c) for c in PRICE_COLUMNS]).where(
            PriceBar.instrument_id == instrument_id
        )
        if start:
            stmt = stmt.where(PriceBar.bar_date >= start)
        if end:
            stmt = stmt.where(PriceBar.bar_date <= end)
        rows = self.session.execute(stmt.order_by(PriceBar.bar_date)).all()
        df = pd.DataFrame(rows, columns=["date", *PRICE_COLUMNS])
        df["date"] = pd.to_datetime(df["date"]).dt.as_unit("ns")
        df[PRICE_COLUMNS] = df[PRICE_COLUMNS].astype("float64")
        return df.set_index("date")

    def delete_bars(self, instrument_id: int) -> int:
        """Delete every stored bar of an instrument (used when replacing its history)."""
        result = self.session.execute(
            delete(PriceBar).where(PriceBar.instrument_id == instrument_id)
        )
        return result.rowcount or 0

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

    def recent(self, limit: int = 20) -> list[IngestionRun]:
        """Most recent runs first."""
        stmt = select(IngestionRun).order_by(IngestionRun.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def issues_with_symbols(self, run_id: int) -> list[tuple[DataQualityIssue, str | None]]:
        """A run's data-quality issues, each with its instrument's symbol."""
        stmt = (
            select(DataQualityIssue, Instrument.symbol)
            .outerjoin(Instrument, Instrument.id == DataQualityIssue.instrument_id)
            .where(DataQualityIssue.ingestion_run_id == run_id)
            .order_by(DataQualityIssue.id)
        )
        return [(issue, symbol) for issue, symbol in self.session.execute(stmt)]


class PortfolioRepository:
    """Saved portfolios. Weights are stored per instrument; callers pass symbols."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        name: str,
        weights: dict[str, float],
        construction_method: PortfolioMethod,
        effective_method: PortfolioMethod,
        params: dict | None = None,
        description: str | None = None,
        estimation_start: date | None = None,
        estimation_end: date | None = None,
    ) -> Portfolio:
        """Insert a portfolio. Every symbol must already be a stored instrument."""
        instruments = InstrumentRepository(self.session)
        portfolio = Portfolio(
            name=name,
            description=description,
            construction_method=construction_method,
            effective_method=effective_method,
            construction_params=params or {},
            estimation_start=estimation_start,
            estimation_end=estimation_end,
        )
        for symbol, weight in weights.items():
            instrument = instruments.get_by_symbol(symbol)
            if instrument is None:
                raise ValueError(f"{symbol} is not a stored instrument")
            portfolio.weights.append(PortfolioWeight(instrument_id=instrument.id, weight=weight))
        self.session.add(portfolio)
        self.session.flush()
        return portfolio

    def _query(self):
        return select(Portfolio).options(
            selectinload(Portfolio.weights).selectinload(PortfolioWeight.instrument)
        )

    def get(self, portfolio_id: int) -> Portfolio | None:
        return self.session.scalars(self._query().where(Portfolio.id == portfolio_id)).first()

    def get_by_name(self, name: str) -> Portfolio | None:
        return self.session.scalars(self._query().where(Portfolio.name == name)).first()

    def list_all(self) -> list[Portfolio]:
        return list(self.session.scalars(self._query().order_by(Portfolio.name)))

    def delete(self, portfolio: Portfolio) -> None:
        self.session.delete(portfolio)
        self.session.flush()

    @staticmethod
    def weights_of(portfolio: Portfolio) -> dict[str, float]:
        """{symbol: weight}, in symbol order."""
        pairs = sorted((w.instrument.symbol, w.weight) for w in portfolio.weights)
        return dict(pairs)


class BacktestRepository:
    """Stored backtest runs with their equity curves and trades."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, run: BacktestRun, equity: list[dict], trades: list[dict]) -> BacktestRun:
        """Insert a run, its equity rows and its trades (`symbol` per trade row)."""
        self.session.add(run)
        self.session.flush()
        if equity:
            self.session.execute(
                pg_insert(BacktestEquity), [{**row, "run_id": run.id} for row in equity]
            )
        if trades:
            ids = {
                i.symbol: i.id
                for i in self.session.scalars(
                    select(Instrument).where(Instrument.symbol.in_({t["symbol"] for t in trades}))
                )
            }
            rows = []
            for t in trades:
                row = {k: v for k, v in t.items() if k != "symbol"}
                rows.append({**row, "run_id": run.id, "instrument_id": ids[t["symbol"]]})
            for i in range(0, len(rows), UPSERT_CHUNK_ROWS):
                self.session.execute(pg_insert(BacktestTrade), rows[i : i + UPSERT_CHUNK_ROWS])
        self.session.flush()
        return run

    def get(self, run_id: int) -> BacktestRun | None:
        return self.session.get(BacktestRun, run_id)

    def list_runs(self, strategy: str | None = None, limit: int = 50) -> list[BacktestRun]:
        stmt = select(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit)
        if strategy:
            stmt = stmt.where(BacktestRun.strategy == strategy)
        return list(self.session.scalars(stmt))

    def equity(self, run_id: int) -> list[BacktestEquity]:
        stmt = (
            select(BacktestEquity)
            .where(BacktestEquity.run_id == run_id)
            .order_by(BacktestEquity.bar_date)
        )
        return list(self.session.scalars(stmt))

    def trades(
        self, run_id: int, limit: int | None = None, offset: int = 0
    ) -> list[tuple[BacktestTrade, str]]:
        stmt = (
            select(BacktestTrade, Instrument.symbol)
            .join(Instrument, Instrument.id == BacktestTrade.instrument_id)
            .where(BacktestTrade.run_id == run_id)
            .order_by(BacktestTrade.bar_date, BacktestTrade.id)
            .offset(offset)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return [(t, s) for t, s in self.session.execute(stmt)]

    def trade_count(self, run_id: int) -> int:
        stmt = select(func.count()).select_from(BacktestTrade).where(BacktestTrade.run_id == run_id)
        return int(self.session.scalar(stmt) or 0)

    def delete(self, run: BacktestRun) -> None:
        self.session.delete(run)
        self.session.flush()
