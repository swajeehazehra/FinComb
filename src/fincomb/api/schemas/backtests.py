"""Request and response models for the strategy and backtest endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from fincomb.api.schemas.common import TimeSeries, finite
from fincomb.services.backtest_service import (
    DEFAULT_BENCHMARK,
    DEFAULT_CAPITAL,
    DEFAULT_COST_BPS,
    DEFAULT_SLIPPAGE_BPS,
    MAX_BACKTEST_SYMBOLS,
    BacktestRecord,
    BacktestSpec,
    TradeRecord,
    VerifyResult,
)

Rebalance = Literal["daily", "weekly", "monthly", "quarterly", "annual", "never"]
Execution = Literal["next_close", "same_close"]


class StrategyParam(BaseModel):
    name: str
    type: str
    default: Any = None
    help: str
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] | None = None


class StrategyOut(BaseModel):
    name: str
    description: str
    default_rebalance: str
    uses_prices: bool = Field(description="False: targets don't depend on prices")
    params: list[StrategyParam]
    params_schema: dict[str, Any] = Field(description="JSON Schema of `params`")


class BacktestRequest(BaseModel):
    strategy: str = Field(examples=["momentum"])
    symbols: list[str] = Field(
        default_factory=list,
        max_length=MAX_BACKTEST_SYMBOLS,
        description="Universe; may be empty when `portfolio` is given",
        examples=[["AAPL", "MSFT", "QQQ", "TLT", "GLD"]],
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters (see GET /strategies)"
    )
    start: date | None = Field(default=None, description="First evaluation date")
    end: date | None = None
    benchmark: str | None = Field(default=DEFAULT_BENCHMARK, description="Empty or null = none")
    initial_capital: float = Field(default=DEFAULT_CAPITAL, gt=0)
    cost_bps: float = Field(default=DEFAULT_COST_BPS, ge=0, le=1000)
    slippage_bps: float = Field(default=DEFAULT_SLIPPAGE_BPS, ge=0, le=1000)
    rebalance: Rebalance | None = Field(default=None, description="Default: the strategy's own")
    execution: Execution = Field(
        default="next_close",
        description="next_close: trade at the close after the signal (default); "
        "same_close: at the signal's own close (optimistic)",
    )
    portfolio: int | str | None = Field(
        default=None, description="Saved portfolio (id or name) for buy_and_hold / fixed_weights"
    )
    risk_free: float = Field(default=0.0, gt=-1, lt=1, description="Annual rate, 0.04 = 4%")
    save: bool = Field(default=True, description="false = run without storing it")

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        out = [s.strip().upper() for s in v if s and s.strip()]
        dupes = sorted({s for s in out if out.count(s) > 1})
        if dupes:
            raise ValueError(f"each symbol may appear only once (repeated: {', '.join(dupes)})")
        return out

    @model_validator(mode="after")
    def _check(self) -> BacktestRequest:
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if not self.symbols and self.portfolio is None:
            raise ValueError("give symbols or a portfolio")
        return self

    def to_spec(self) -> BacktestSpec:
        return BacktestSpec(**self.model_dump())


class BacktestSummary(BaseModel):
    id: int | None
    strategy: str
    symbols: list[str]
    benchmark: str | None
    start: date
    end: date
    rebalance: str
    execution: str
    total_return: float | None
    cagr: float | None
    sharpe: float | None
    max_drawdown: float | None
    annual_turnover: float | None
    excess_cagr: float | None
    has_warnings: bool
    created_at: datetime | None

    @classmethod
    def from_record(cls, r: BacktestRecord) -> BacktestSummary:
        s = r.metrics.get("strategy") or {}
        rel = r.metrics.get("relative") or {}
        return cls(
            id=r.id,
            strategy=r.strategy,
            symbols=r.symbols,
            benchmark=r.benchmark,
            start=r.start,
            end=r.end,
            rebalance=r.rebalance,
            execution=r.execution,
            total_return=finite(s.get("total_return")),
            cagr=finite(s.get("cagr")),
            sharpe=finite(s.get("sharpe")),
            max_drawdown=finite(s.get("max_drawdown")),
            annual_turnover=finite((r.metrics.get("trading") or {}).get("annual_turnover")),
            excess_cagr=finite(rel.get("excess_cagr")),
            has_warnings=bool(r.warnings),
            created_at=r.created_at,
        )


class Reproducibility(BaseModel):
    code_version: str
    data_hash: str
    warmup_start: date | None
    requested_start: date | None
    requested_end: date | None


class BacktestDetail(BaseModel):
    id: int | None
    strategy: str
    params: dict[str, Any]
    symbols: list[str]
    benchmark: str | None
    start: date = Field(description="Effective first evaluation date (day 0)")
    end: date
    rebalance: str
    execution: str
    initial_capital: float
    cost_bps: float
    slippage_bps: float
    risk_free: float
    portfolio_id: int | None
    metrics: dict[str, Any] = Field(
        description="strategy, benchmark, relative (excess return, tracking error, "
        "information ratio, HAC alpha/beta), trading (turnover, costs) and diagnostics"
    )
    warnings: list[str] = Field(description="Anything that weakens the result. Read these.")
    notes: list[str] = Field(description="Alignment and effective-period notes")
    reproducibility: Reproducibility
    duration_ms: int | None
    created_at: datetime | None

    @classmethod
    def from_record(cls, r: BacktestRecord) -> BacktestDetail:
        return cls(
            id=r.id,
            strategy=r.strategy,
            params=r.params,
            symbols=r.symbols,
            benchmark=r.benchmark,
            start=r.start,
            end=r.end,
            rebalance=r.rebalance,
            execution=r.execution,
            initial_capital=r.initial_capital,
            cost_bps=r.cost_bps,
            slippage_bps=r.slippage_bps,
            risk_free=r.risk_free,
            portfolio_id=r.portfolio_id,
            metrics=r.metrics,
            warnings=r.warnings,
            notes=r.notes,
            reproducibility=Reproducibility(
                code_version=r.code_version,
                data_hash=r.data_hash,
                warmup_start=r.warmup_start,
                requested_start=r.requested_start,
                requested_end=r.requested_end,
            ),
            duration_ms=r.duration_ms,
            created_at=r.created_at,
        )


class EquityResponse(BaseModel):
    run_id: int
    curves: TimeSeries = Field(
        description="equity, cash, gross_exposure, benchmark_equity, drawdown, turnover, costs"
    )


class TradeOut(BaseModel):
    date: date
    symbol: str
    kind: str = Field(description="init (day 0) or rebalance")
    side: str
    weight_before: float
    weight_after: float
    quantity: float = Field(description="Indicative fractional shares at the fill price")
    reference_price: float = Field(description="Raw close that day")
    fill_price: float
    notional: float
    commission: float
    slippage_cost: float

    @classmethod
    def from_record(cls, t: TradeRecord) -> TradeOut:
        return cls(**t.__dict__)


class TradesResponse(BaseModel):
    run_id: int
    total: int
    trades: list[TradeOut]


class VerifyResponse(BaseModel):
    run_id: int
    reproducible: bool
    data_hash_matches: bool
    metrics_match: bool
    stored_data_hash: str
    current_data_hash: str
    stored_code_version: str
    current_code_version: str
    mismatched_metrics: list[str]

    @classmethod
    def from_result(cls, v: VerifyResult) -> VerifyResponse:
        return cls(**v.__dict__)
