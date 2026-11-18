from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from fincomb import __version__
from fincomb.api.deps import FactoryDep
from fincomb.api.schemas.system import HealthResponse
from fincomb.db.base import session_scope

router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable"}},
)
def health(factory: FactoryDep):
    """Liveness plus database connectivity and the applied migration revision."""
    try:
        with session_scope(factory) as s:
            s.execute(text("SELECT 1"))
            try:
                with s.begin_nested():
                    rev = s.execute(text("SELECT version_num FROM alembic_version")).scalar()
            except ProgrammingError:  # never migrated
                rev = None
    except (OperationalError, DBAPIError):
        body = HealthResponse(
            status="degraded", version=__version__, database="unreachable", schema_revision=None
        )
        return JSONResponse(status_code=503, content=body.model_dump())
    return HealthResponse(status="ok", version=__version__, database="ok", schema_revision=rev)
