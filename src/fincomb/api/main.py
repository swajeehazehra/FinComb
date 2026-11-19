"""FastAPI application factory.

    uvicorn fincomb.api.main:create_app --factory     (or: fincomb serve)

`create_app` takes its settings and session factory as arguments so tests can
point it at the test database; with no arguments it reads `.env` like the CLI.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session, sessionmaker

from fincomb import __version__
from fincomb.api.errors import install_error_handlers
from fincomb.api.routers import analytics, health, ingestion, instruments, portfolios, prices
from fincomb.config import Settings, get_settings
from fincomb.db.base import make_engine, make_session_factory

API_PREFIX = "/api/v1"

DESCRIPTION = """
Market data, data-quality audit, performance analytics, and portfolio construction
and risk for daily price history.

* Dates are ISO `YYYY-MM-DD`; ranges are inclusive.
* Multi-symbol endpoints use **inner alignment** by default (only dates every symbol
  has). `notes` explains when that shortens the period; `align=outer` analyses each
  symbol over its own history.
* Statistics that are undefined (e.g. Sharpe with zero volatility) are `null`.
* Portfolio responses carry `warnings` (optimiser fallback, no positive excess
  return, unreliable tail estimates): read them before trusting the numbers.
* Errors always look like `{"error": {"code", "message", "details"}}`.
"""


def create_app(
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    if session_factory is None:
        session_factory = make_session_factory(make_engine(settings.database_url()))

    app = FastAPI(
        title="FinComb API",
        version=__version__,
        description=DESCRIPTION,
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.session_factory = session_factory
    install_error_handlers(app)

    for module in (health, instruments, prices, ingestion, analytics, portfolios):
        app.include_router(module.router, prefix=API_PREFIX)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    return app
