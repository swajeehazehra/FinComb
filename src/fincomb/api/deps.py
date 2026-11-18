"""FastAPI dependencies: settings, the session factory, and one DB session per request.

Everything comes from `app.state`, which `create_app` fills. Tests build the app
with their own settings and session factory, so the development database is never
touched by the test suite.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Annotated

from fastapi import Depends, Query, Request
from sqlalchemy.orm import Session, sessionmaker

from fincomb.api.errors import ApiError
from fincomb.config import Settings
from fincomb.db.base import session_scope
from fincomb.services.ingestion_service import normalise_symbols


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_session_factory(request: Request) -> sessionmaker[Session]:
    return request.app.state.session_factory


def get_session(
    factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
) -> Iterator[Session]:
    """Commit if the endpoint succeeds, roll back if it raises."""
    with session_scope(factory) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
FactoryDep = Annotated[sessionmaker[Session], Depends(get_session_factory)]


class DateRange:
    """`start` / `end` query parameters (inclusive, ISO dates), checked for order."""

    def __init__(
        self,
        start: Annotated[date | None, Query(description="First date, YYYY-MM-DD")] = None,
        end: Annotated[date | None, Query(description="Last date, YYYY-MM-DD")] = None,
    ) -> None:
        if start and end and start > end:
            raise ApiError(
                422,
                "invalid_request",
                f"start {start} is after end {end}",
                {"start": start.isoformat(), "end": end.isoformat()},
            )
        self.start = start
        self.end = end


DateRangeDep = Annotated[DateRange, Depends()]


MAX_ANALYTICS_SYMBOLS = 50


def symbol_list(
    symbols: Annotated[
        list[str],
        Query(
            description="Tickers, comma-separated (symbols=AAPL,MSFT) or repeated "
            "(symbols=AAPL&symbols=MSFT)",
        ),
    ],
) -> list[str]:
    out = normalise_symbols([part for item in symbols for part in item.split(",")])
    if not out:
        raise ApiError(422, "invalid_request", "symbols: give at least one ticker")
    if len(out) > MAX_ANALYTICS_SYMBOLS:
        raise ApiError(
            422,
            "invalid_request",
            f"symbols: at most {MAX_ANALYTICS_SYMBOLS} tickers per request",
        )
    return out


SymbolsDep = Annotated[list[str], Depends(symbol_list)]
