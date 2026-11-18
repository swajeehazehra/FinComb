from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query

from fincomb.api.deps import SessionDep
from fincomb.api.errors import ApiError, not_found
from fincomb.api.schemas.common import ERROR_RESPONSES, ErrorResponse
from fincomb.api.schemas.instruments import (
    Coverage,
    InstrumentCreate,
    InstrumentDetail,
    InstrumentOut,
    InstrumentUpdate,
)
from fincomb.db.models import AssetType, Instrument
from fincomb.db.repositories import InstrumentRepository, PriceRepository

router = APIRouter(prefix="/instruments", tags=["instruments"], responses=ERROR_RESPONSES)


def _detail(session, instrument: Instrument) -> InstrumentDetail:
    first, last, n = PriceRepository(session).stored_range(instrument.id)
    return InstrumentDetail(
        **InstrumentOut.model_validate(instrument).model_dump(),
        coverage=Coverage(first_date=first, last_date=last, bars=n),
    )


def _get(session, symbol: str) -> Instrument:
    instrument = InstrumentRepository(session).get_by_symbol(symbol)
    if instrument is None:
        raise not_found("instrument", symbol.upper())
    return instrument


@router.get("", response_model=list[InstrumentOut])
def list_instruments(
    session: SessionDep,
    asset_type: Annotated[AssetType | None, Query()] = None,
    active: Annotated[bool | None, Query()] = None,
):
    """All registered instruments, alphabetically."""
    return InstrumentRepository(session).list_instruments(asset_type, active)


@router.post(
    "",
    response_model=InstrumentDetail,
    status_code=201,
    responses={409: {"model": ErrorResponse, "description": "Symbol already registered"}},
)
def create_instrument(body: InstrumentCreate, session: SessionDep):
    """Register a symbol without downloading anything (use POST /ingestion/runs for data)."""
    repo = InstrumentRepository(session)
    if repo.get_by_symbol(body.symbol) is not None:
        raise ApiError(
            409, "already_exists", f"{body.symbol} is already registered", {"symbol": body.symbol}
        )
    instrument = repo.get_or_create(body.symbol, body.asset_type)
    instrument.name = body.name
    session.flush()
    return _detail(session, instrument)


@router.get("/{symbol}", response_model=InstrumentDetail)
def get_instrument(symbol: str, session: SessionDep):
    """One instrument with its stored date coverage and owning data source."""
    return _detail(session, _get(session, symbol))


@router.patch("/{symbol}", response_model=InstrumentDetail)
def update_instrument(symbol: str, body: InstrumentUpdate, session: SessionDep):
    """Change the asset type (same as `fincomb set-type`)."""
    _get(session, symbol)
    instrument = InstrumentRepository(session).set_asset_type(symbol, body.asset_type)
    assert instrument is not None
    return _detail(session, instrument)
