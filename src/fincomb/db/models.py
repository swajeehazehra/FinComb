"""SQLAlchemy ORM models: market data (Milestone 1), portfolios (Milestone 4) and
backtest runs (Milestone 5)."""

from __future__ import annotations

import enum
from datetime import date, datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fincomb.db.base import Base


class AssetType(enum.StrEnum):
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class PortfolioMethod(enum.StrEnum):
    MANUAL = "manual"
    EQUAL_WEIGHT = "equal_weight"
    INVERSE_VOL = "inverse_vol"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"


class Severity(enum.StrEnum):
    WARNING = "warning"
    ERROR = "error"


def _pg_enum(cls: type[enum.Enum], name: str) -> Enum:
    # Store the lowercase *values* ("stock"), not the member names ("STOCK").
    return Enum(cls, name=name, values_callable=lambda e: [m.value for m in e])


class Instrument(Base):
    __tablename__ = "instruments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(200))
    asset_type: Mapped[AssetType] = mapped_column(
        _pg_enum(AssetType, "asset_type"), nullable=False, default=AssetType.STOCK
    )
    exchange: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    is_active: Mapped[bool] = mapped_column(nullable=False, default=True)
    # The one source this symbol's price history comes from ("yfinance", "tiingo",
    # "csv"). Mixing sources breaks the history: vendors adjust prices differently.
    data_source: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    bars: Mapped[list[PriceBar]] = relationship(back_populates="instrument")

    def __repr__(self) -> str:
        return f"Instrument({self.symbol!r}, {self.asset_type.value})"


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    symbols: Mapped[list[str]] = mapped_column(ARRAY(String(20)), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[RunStatus] = mapped_column(
        _pg_enum(RunStatus, "run_status"), nullable=False, default=RunStatus.RUNNING
    )
    rows_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_rejected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)

    issues: Mapped[list[DataQualityIssue]] = relationship(back_populates="run")


class PriceBar(Base):
    __tablename__ = "price_bars"
    __table_args__ = (
        CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0 AND adj_close > 0",
            name="ck_price_bars_positive",
        ),
        CheckConstraint(
            "high >= low AND high >= open AND high >= close AND low <= open AND low <= close",
            name="ck_price_bars_ohlc",
        ),
        CheckConstraint("volume IS NULL OR volume >= 0", name="ck_price_bars_volume"),
    )

    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), primary_key=True
    )
    bar_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int | None] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL")
    )

    instrument: Mapped[Instrument] = relationship(back_populates="bars")


class DataQualityIssue(Base):
    __tablename__ = "data_quality_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ingestion_run_id: Mapped[int] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_id: Mapped[int | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE")
    )
    bar_date: Mapped[date | None] = mapped_column(Date)
    rule: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[Severity] = mapped_column(_pg_enum(Severity, "issue_severity"), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)

    run: Mapped[IngestionRun] = relationship(back_populates="issues")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # What was asked for, and what actually produced the weights (they differ only
    # when an optimiser fell back to inverse volatility).
    construction_method: Mapped[PortfolioMethod] = mapped_column(
        _pg_enum(PortfolioMethod, "portfolio_method"), nullable=False
    )
    effective_method: Mapped[PortfolioMethod] = mapped_column(
        _pg_enum(PortfolioMethod, "portfolio_method"), nullable=False
    )
    # Everything needed to reproduce the weights: bounds, covariance method,
    # shrinkage, risk-free rate, optimiser status, warnings...
    construction_params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    estimation_start: Mapped[date | None] = mapped_column(Date)  # first price date used
    estimation_end: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    weights: Mapped[list[PortfolioWeight]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", passive_deletes=True
    )

    def __repr__(self) -> str:
        return f"Portfolio({self.name!r}, {self.construction_method.value})"


class PortfolioWeight(Base):
    __tablename__ = "portfolio_weights"
    __table_args__ = (
        CheckConstraint("weight >= 0 AND weight <= 1", name="ck_portfolio_weights_long_only"),
    )

    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), primary_key=True
    )
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"), primary_key=True
    )
    weight: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped[Portfolio] = relationship(back_populates="weights")
    instrument: Mapped[Instrument] = relationship()


class BacktestRun(Base):
    """One stored backtest: every input needed to rerun it, and its metrics.

    Only successful runs are stored. `data_hash` and `code_version` make it checkable
    later (`fincomb backtest verify`).
    """

    __tablename__ = "backtest_runs"
    __table_args__ = (
        CheckConstraint("initial_capital > 0", name="ck_backtest_runs_capital"),
        CheckConstraint("cost_bps >= 0 AND slippage_bps >= 0", name="ck_backtest_runs_costs"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    symbols: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    benchmark_symbol: Mapped[str | None] = mapped_column(String(20))
    start_date: Mapped[date] = mapped_column(Date, nullable=False)  # effective period
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    requested_start: Mapped[date | None] = mapped_column(Date)
    requested_end: Mapped[date | None] = mapped_column(Date)
    warmup_start: Mapped[date | None] = mapped_column(Date)  # first price date read
    rebalance_frequency: Mapped[str] = mapped_column(String(10), nullable=False)
    execution: Mapped[str] = mapped_column(String(12), nullable=False)
    initial_capital: Mapped[float] = mapped_column(Float, nullable=False)
    cost_bps: Mapped[float] = mapped_column(Float, nullable=False)
    slippage_bps: Mapped[float] = mapped_column(Float, nullable=False)
    risk_free: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    portfolio_id: Mapped[int | None] = mapped_column(
        ForeignKey("portfolios.id", ondelete="SET NULL")
    )
    code_version: Mapped[str] = mapped_column(String(40), nullable=False)
    data_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    warnings: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    notes: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"BacktestRun({self.id}, {self.strategy})"


class BacktestEquity(Base):
    __tablename__ = "backtest_equity"

    run_id: Mapped[int] = mapped_column(
        ForeignKey("backtest_runs.id", ondelete="CASCADE"), primary_key=True
    )
    bar_date: Mapped[date] = mapped_column(Date, primary_key=True)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    cash: Mapped[float] = mapped_column(Float, nullable=False)
    gross_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_equity: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float] = mapped_column(Float, nullable=False)
    turnover: Mapped[float] = mapped_column(Float, nullable=False)
    costs: Mapped[float] = mapped_column(Float, nullable=False)


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False
    )  # indexed with bar_date by migration 0004
    bar_date: Mapped[date] = mapped_column(Date, nullable=False)
    instrument_id: Mapped[int] = mapped_column(
        ForeignKey("instruments.id", ondelete="RESTRICT"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(10), nullable=False)  # init | rebalance
    side: Mapped[str] = mapped_column(String(4), nullable=False)  # buy | sell
    weight_before: Mapped[float] = mapped_column(Float, nullable=False)
    weight_after: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    reference_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    notional: Mapped[float] = mapped_column(Float, nullable=False)
    commission: Mapped[float] = mapped_column(Float, nullable=False)
    slippage_cost: Mapped[float] = mapped_column(Float, nullable=False)

    instrument: Mapped[Instrument] = relationship()
