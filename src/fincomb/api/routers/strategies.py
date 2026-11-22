"""Strategy catalogue."""

from __future__ import annotations

from fastapi import APIRouter

from fincomb.api.schemas.backtests import StrategyOut
from fincomb.services import backtest_service as svc

router = APIRouter(prefix="/strategies", tags=["backtests"])


@router.get("", response_model=list[StrategyOut])
def list_strategies():
    """Every strategy with its default rebalance frequency and parameter schema."""
    return svc.strategies()
