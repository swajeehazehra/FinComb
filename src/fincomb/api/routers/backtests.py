"""Backtest endpoints. All work is done by `fincomb.services.backtest_service`."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from fincomb.api.deps import SessionDep
from fincomb.api.schemas.backtests import (
    BacktestDetail,
    BacktestRequest,
    BacktestSummary,
    EquityResponse,
    TradeOut,
    TradesResponse,
    VerifyResponse,
)
from fincomb.api.schemas.common import ERROR_RESPONSES, TimeSeries
from fincomb.services import backtest_service as svc

router = APIRouter(prefix="/backtests", tags=["backtests"], responses=ERROR_RESPONSES)


@router.post("", response_model=BacktestDetail, status_code=201)
def run_backtest(body: BacktestRequest, session: SessionDep, response: Response):
    """Run a backtest synchronously. Stored (201) unless `save` is false (200, `id` null).

    Targets formed at close t trade at close t+1 by default (`execution=next_close`);
    the initial allocation is bought at the first evaluation close. Check `warnings`.
    """
    outcome = svc.run_backtest(session, body.to_spec())
    if not body.save:
        response.status_code = 200
    return BacktestDetail.from_record(outcome.record)


@router.get("", response_model=list[BacktestSummary])
def list_backtests(
    session: SessionDep,
    strategy: Annotated[str | None, Query(description="Only this strategy")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
):
    """Stored runs, newest first, with headline metrics for comparing experiments."""
    return [BacktestSummary.from_record(r) for r in svc.list_backtests(session, strategy, limit)]


@router.get("/{run_id}", response_model=BacktestDetail)
def get_backtest(run_id: int, session: SessionDep):
    """Inputs, metrics, benchmark comparison, warnings and reproducibility fields."""
    return BacktestDetail.from_record(svc.get_backtest(session, run_id))


@router.get("/{run_id}/equity", response_model=EquityResponse)
def get_equity(run_id: int, session: SessionDep):
    """Daily equity, cash, exposure, benchmark, drawdown, turnover and costs."""
    return EquityResponse(
        run_id=run_id, curves=TimeSeries.from_frame(svc.backtest_equity(session, run_id))
    )


@router.get("/{run_id}/trades", response_model=TradesResponse)
def get_trades(
    run_id: int,
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=10_000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Trade log in date order (`kind` = init for the day-0 allocation)."""
    trades, total = svc.backtest_trades(session, run_id, limit, offset)
    return TradesResponse(
        run_id=run_id, total=total, trades=[TradeOut.from_record(t) for t in trades]
    )


@router.get("/{run_id}/verify", response_model=VerifyResponse)
def verify(run_id: int, session: SessionDep):
    """Rerun from current stored prices and compare the data hash and every metric."""
    return VerifyResponse.from_result(svc.verify_backtest(session, run_id))


@router.delete("/{run_id}", status_code=204)
def delete_backtest(run_id: int, session: SessionDep):
    svc.delete_backtest(session, run_id)
    return Response(status_code=204)
