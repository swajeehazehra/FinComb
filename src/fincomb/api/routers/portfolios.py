"""Portfolio endpoints. All work is done by `fincomb.services.portfolio_service`."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from fincomb.api.deps import DateRangeDep, SessionDep
from fincomb.api.schemas.common import ERROR_RESPONSES, ErrorResponse
from fincomb.api.schemas.portfolios import (
    ConstructionRequest,
    ConstructionResponse,
    PortfolioCreate,
    PortfolioDetail,
    PortfolioSummary,
    RiskRequest,
    RiskResponse,
)
from fincomb.services import portfolio_service as svc

router = APIRouter(prefix="/portfolios", tags=["portfolios"], responses=ERROR_RESPONSES)


@router.post("/construct", response_model=ConstructionResponse)
def construct(body: ConstructionRequest, session: SessionDep):
    """Compute weights by one of four methods over a trailing window. Nothing is saved.

    `in_sample` statistics describe the window the weights were fitted on, which
    flatters them. Check `warnings`: an optimiser fallback or a max-Sharpe portfolio
    with no positive excess return is reported there.
    """
    return ConstructionResponse.build(svc.construct_portfolio(session, body.to_spec()))


@router.post("/risk", response_model=RiskResponse)
def risk_for_weights(body: RiskRequest, session: SessionDep):
    """Risk report for weights that are not saved (same numbers as a saved portfolio's)."""
    outcome = svc.weights_risk(
        session,
        body.weights,
        body.start,
        body.end,
        body.benchmark,
        body.confidence,
        body.risk_free,
    )
    return RiskResponse.build(outcome)


@router.post(
    "",
    response_model=PortfolioDetail,
    status_code=201,
    responses={409: {"model": ErrorResponse, "description": "Name already taken"}},
)
def create_portfolio(body: PortfolioCreate, session: SessionDep):
    """Save a portfolio: manual `weights`, or a `construction` run on the server so the
    stored weights always match their stored parameters."""
    if body.weights is not None:
        record = svc.create_manual(session, body.name, body.weights, body.description)
    else:
        assert body.construction is not None
        record, _ = svc.create_constructed(
            session, body.name, body.construction.to_spec(), body.description
        )
    return PortfolioDetail.from_record(record)


@router.get("", response_model=list[PortfolioSummary])
def list_portfolios(session: SessionDep):
    """Saved portfolios, alphabetically."""
    return [PortfolioSummary.from_record(p) for p in svc.list_portfolios(session)]


@router.get("/{portfolio_id}", response_model=PortfolioDetail)
def get_portfolio(portfolio_id: int, session: SessionDep):
    """Weights, method and the parameters that produced them."""
    return PortfolioDetail.from_record(svc.get_portfolio(session, portfolio_id))


@router.delete("/{portfolio_id}", status_code=204)
def delete_portfolio(portfolio_id: int, session: SessionDep):
    svc.delete_portfolio(session, portfolio_id)
    return Response(status_code=204)


@router.get("/{portfolio_id}/risk", response_model=RiskResponse)
def portfolio_risk(
    portfolio_id: int,
    session: SessionDep,
    dates: DateRangeDep,
    benchmark: Annotated[
        str | None, Query(description="Benchmark for beta; empty = none")
    ] = svc.DEFAULT_BENCHMARK,
    confidence: Annotated[float, Query(ge=0.5, le=0.999)] = 0.95,
    risk_free: Annotated[float, Query(gt=-1, lt=1, description="Annual rate, 0.04 = 4%")] = 0.0,
):
    """Volatility, 1-day historical and Gaussian VaR/CVaR, beta/alpha against the
    benchmark, and each asset's contribution to risk, over [start, end]
    (default: every date all assets and the benchmark share)."""
    outcome = svc.portfolio_risk(
        session, portfolio_id, dates.start, dates.end, benchmark, confidence, risk_free
    )
    return RiskResponse.build(outcome)
