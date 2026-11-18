"""Request and response models for the portfolio endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from fincomb.api.schemas.common import finite
from fincomb.portfolio.analysis import RiskReport
from fincomb.services.portfolio_service import (
    DEFAULT_BENCHMARK,
    DEFAULT_LOOKBACK,
    MAX_NAME_LENGTH,
    ConstructionSpec,
    ConstructOutcome,
    PortfolioRecord,
    RiskOutcome,
)

MAX_PORTFOLIO_SYMBOLS = 50

WARNINGS_FIELD = Field(
    default_factory=list,
    description="Anything that weakens the result: an optimiser fallback, a covariance "
    "ridge, no asset beating the risk-free rate, too few tail observations. Read these.",
)
NOTES_FIELD = Field(
    default_factory=list,
    description="Why the common period is shorter than the full data (inner alignment)",
)

ConstructMethod = Literal["equal_weight", "inverse_vol", "min_variance", "max_sharpe"]


def _upper_unique(symbols: list[str]) -> list[str]:
    out = [s.strip().upper() for s in symbols if s and s.strip()]
    dupes = sorted({s for s in out if out.count(s) > 1})
    if dupes:
        raise ValueError(f"each symbol may appear only once (repeated: {', '.join(dupes)})")
    return out


class ConstructionRequest(BaseModel):
    symbols: list[str] = Field(
        min_length=1, max_length=MAX_PORTFOLIO_SYMBOLS, examples=[["AAPL", "MSFT", "SPY", "TLT"]]
    )
    method: ConstructMethod = Field(examples=["min_variance"])
    start: date | None = Field(
        default=None, description="Window start; when given, `lookback` is ignored"
    )
    end: date | None = Field(default=None, description="Window end (default: latest common date)")
    lookback: int = Field(
        default=DEFAULT_LOOKBACK, ge=20, le=5040, description="Daily returns in the window"
    )
    min_weight: float = Field(default=0.0, ge=0, le=1)
    max_weight: float = Field(default=1.0, ge=0, le=1)
    covariance: Literal["ledoit_wolf", "sample"] = "ledoit_wolf"
    risk_free: float = Field(default=0.0, gt=-1, lt=1, description="Annual rate, 0.04 = 4%")

    @field_validator("symbols")
    @classmethod
    def _symbols(cls, v: list[str]) -> list[str]:
        out = _upper_unique(v)
        if not out:
            raise ValueError("give at least one symbol")
        return out

    @model_validator(mode="after")
    def _ranges(self) -> ConstructionRequest:
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must not exceed max_weight")
        return self

    def to_spec(self) -> ConstructionSpec:
        return ConstructionSpec(**self.model_dump())


class WeightOut(BaseModel):
    symbol: str
    weight: float


class ContributionOut(BaseModel):
    symbol: str
    weight: float
    marginal_risk: float | None = Field(description="Extra annual volatility per unit of weight")
    risk_contribution: float | None = Field(description="Share of annual volatility; sums to it")
    percent_of_risk: float | None = Field(description="Sums to 1; negative = reduces risk")


def _contributions(table: pd.DataFrame) -> list[ContributionOut]:
    return [
        ContributionOut(
            symbol=str(sym),
            weight=float(r["weight"]),
            marginal_risk=finite(r["marginal"]),
            risk_contribution=finite(r["component"]),
            percent_of_risk=finite(r["percent"]),
        )
        for sym, r in table.iterrows()
    ]


class OptimizerOut(BaseModel):
    success: bool
    message: str
    iterations: int


class EstimationWindow(BaseModel):
    start: date = Field(description="First price date")
    end: date = Field(description="Last price date")
    observations: int = Field(description="Daily returns used")


class InSampleStats(BaseModel):
    """Statistics of the weights over the window they were fitted on (flattering)."""

    expected_return: float | None
    volatility: float | None
    sharpe: float | None


class ConstructionResponse(BaseModel):
    method: str
    effective_method: str = Field(description="inverse_vol if the optimiser fell back")
    window: EstimationWindow
    min_weight: float
    max_weight: float
    covariance: str | None
    shrinkage: float | None = Field(description="Ledoit-Wolf intensity, 0 = none, 1 = full")
    ridge: float
    risk_free: float
    in_sample: InSampleStats
    optimizer: OptimizerOut | None
    fallback_used: bool
    fallback_reason: str | None
    non_positive_excess_return: bool = Field(
        description="True when no asset beat the risk-free rate (see warnings)"
    )
    weights: list[ContributionOut]
    warnings: list[str] = WARNINGS_FIELD
    notes: list[str] = NOTES_FIELD

    @classmethod
    def build(cls, outcome: ConstructOutcome) -> ConstructionResponse:
        r = outcome.result
        return cls(
            method=r.method,
            effective_method=r.effective_method,
            window=EstimationWindow(
                start=r.start.date(), end=r.end.date(), observations=r.observations
            ),
            min_weight=r.bounds.min_weight,
            max_weight=r.bounds.max_weight,
            covariance=r.covariance,
            shrinkage=finite(r.shrinkage),
            ridge=r.ridge,
            risk_free=r.risk_free,
            in_sample=InSampleStats(
                expected_return=finite(r.expected_return),
                volatility=finite(r.volatility),
                sharpe=finite(r.sharpe),
            ),
            optimizer=None
            if r.optimizer is None
            else OptimizerOut(
                success=r.optimizer.success,
                message=r.optimizer.message,
                iterations=r.optimizer.iterations,
            ),
            fallback_used=r.fallback_used,
            fallback_reason=r.fallback_reason,
            non_positive_excess_return=r.non_positive_excess_return,
            weights=_contributions(outcome.contributions),
            warnings=r.warnings,
            notes=outcome.notes,
        )


class PortfolioCreate(BaseModel):
    """Give exactly one of `weights` (manual) or `construction` (computed on the server)."""

    name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH, examples=["Core min-var"])
    description: str | None = None
    weights: dict[str, float] | None = Field(
        default=None, examples=[{"SPY": 0.6, "TLT": 0.4}], description="Must sum to 1"
    )
    construction: ConstructionRequest | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> PortfolioCreate:
        if (self.weights is None) == (self.construction is None):
            raise ValueError("give exactly one of 'weights' or 'construction'")
        if not self.name.strip():
            raise ValueError("name must not be blank")
        return self


class PortfolioSummary(BaseModel):
    id: int
    name: str
    description: str | None
    construction_method: str
    effective_method: str
    assets: int
    created_at: datetime
    warnings: list[str] = WARNINGS_FIELD

    @classmethod
    def from_record(cls, p: PortfolioRecord) -> PortfolioSummary:
        return cls(
            id=p.id,
            name=p.name,
            description=p.description,
            construction_method=p.construction_method,
            effective_method=p.effective_method,
            assets=len(p.weights),
            created_at=p.created_at,
            warnings=p.warnings,
        )


class PortfolioDetail(PortfolioSummary):
    estimation_start: date | None
    estimation_end: date | None
    weights: list[WeightOut]
    construction_params: dict[str, Any]

    @classmethod
    def from_record(cls, p: PortfolioRecord) -> PortfolioDetail:
        return cls(
            **PortfolioSummary.from_record(p).model_dump(),
            estimation_start=p.estimation_start,
            estimation_end=p.estimation_end,
            weights=[WeightOut(symbol=s, weight=w) for s, w in p.weights.items()],
            construction_params=p.construction_params,
        )


class RiskRequest(BaseModel):
    weights: dict[str, float] = Field(examples=[{"SPY": 0.6, "TLT": 0.4}])
    start: date | None = None
    end: date | None = None
    benchmark: str | None = DEFAULT_BENCHMARK
    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    risk_free: float = Field(default=0.0, gt=-1, lt=1)

    @model_validator(mode="after")
    def _dates(self) -> RiskRequest:
        if self.start and self.end and self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


class PerformanceOut(BaseModel):
    total_return: float | None
    cagr: float | None
    annual_volatility: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None
    calmar: float | None
    best_day: float | None
    worst_day: float | None


class TailRisk(BaseModel):
    historical: float | None = Field(description="From the actual returns")
    normal: float | None = Field(description="Assuming normally distributed returns")


class BetaOut(BaseModel):
    benchmark: str
    beta: float | None
    alpha: float | None = Field(description="Jensen's alpha, annualised")
    correlation: float | None
    r_squared: float | None
    observations: int


class RiskResponse(BaseModel):
    portfolio_id: int | None
    name: str | None
    start: date
    end: date
    observations: int
    confidence: float
    risk_free: float
    horizon_days: int = Field(default=1, description="VaR and CVaR are 1-day losses")
    performance: PerformanceOut
    var: TailRisk = Field(description="Value at Risk: positive number = loss")
    cvar: TailRisk = Field(description="Average loss beyond VaR (expected shortfall)")
    tail_observations: float
    beta: BetaOut | None
    contributions: list[ContributionOut]
    warnings: list[str] = WARNINGS_FIELD
    notes: list[str] = NOTES_FIELD

    @classmethod
    def build(cls, outcome: RiskOutcome) -> RiskResponse:
        rep: RiskReport = outcome.report
        b = rep.beta
        return cls(
            portfolio_id=outcome.portfolio.id if outcome.portfolio else None,
            name=outcome.portfolio.name if outcome.portfolio else None,
            start=rep.start.date(),
            end=rep.end.date(),
            observations=rep.observations,
            confidence=rep.confidence,
            risk_free=rep.risk_free,
            performance=PerformanceOut(
                **{k: finite(rep.summary.get(k)) for k in PerformanceOut.model_fields}
            ),
            var=TailRisk(historical=finite(rep.var_historical), normal=finite(rep.var_normal)),
            cvar=TailRisk(historical=finite(rep.cvar_historical), normal=finite(rep.cvar_normal)),
            tail_observations=rep.tail_observations,
            beta=None
            if b is None
            else BetaOut(
                benchmark=str(rep.benchmark),
                beta=finite(b.beta),
                alpha=finite(b.alpha),
                correlation=finite(b.correlation),
                r_squared=finite(b.r_squared),
                observations=b.observations,
            ),
            contributions=_contributions(rep.contributions),
            warnings=rep.warnings,
            notes=outcome.notes,
        )
