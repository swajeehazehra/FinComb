"""Turn exceptions into HTTP responses with one JSON shape.

Every error response looks like::

    {"error": {"code": "symbol_not_found", "message": "...", "details": {...}}}

Domain exceptions from `fincomb.exceptions` are mapped here, in one table, so
routers never build error responses themselves.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

from fincomb.exceptions import (
    BacktestNotFoundError,
    DataOwnershipError,
    DataSourceAuthError,
    DataSourceError,
    FinCombError,
    InsufficientDataError,
    MissingDataError,
    PortfolioExistsError,
    PortfolioNotFoundError,
    SymbolNotFoundError,
    UnknownStrategyError,
    ValidationError,
)

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Raised by routers for request-level problems (not found, conflict, bad input)."""

    def __init__(
        self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def not_found(what: str, key: object) -> ApiError:
    return ApiError(404, "not_found", f"{what} {key} not found", {"key": str(key)})


# Most specific first is not required (handlers are looked up along the exception's
# MRO), but the table reads better that way.
DOMAIN_ERRORS: dict[type[FinCombError], tuple[int, str]] = {
    SymbolNotFoundError: (404, "symbol_not_found"),
    PortfolioNotFoundError: (404, "not_found"),
    BacktestNotFoundError: (404, "not_found"),
    UnknownStrategyError: (422, "invalid_request"),
    PortfolioExistsError: (409, "already_exists"),
    DataOwnershipError: (409, "data_ownership_conflict"),
    InsufficientDataError: (422, "insufficient_data"),
    MissingDataError: (422, "missing_data"),
    ValidationError: (422, "invalid_data"),
    DataSourceAuthError: (503, "data_source_auth"),
    DataSourceError: (502, "data_source_error"),
    FinCombError: (400, "fincomb_error"),
}

_HTTP_CODES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "invalid_request",
}


def error_response(
    status_code: int, code: str, message: str, details: dict[str, Any] | None = None
) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details or {}}}
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


def _domain_handler(status_code: int, code: str):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return error_response(status_code, code, str(exc))

    return handler


async def _api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.code, exc.message, exc.details)


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    errors = [
        {"location": list(e.get("loc", ())), "message": e.get("msg", ""), "type": e.get("type")}
        for e in exc.errors()
    ]
    first = errors[0] if errors else {"location": [], "message": "invalid request"}
    where = ".".join(str(p) for p in first["location"] if p not in ("query", "body", "path"))
    message = f"{where}: {first['message']}" if where else str(first["message"])
    return error_response(422, "invalid_request", message, {"errors": errors})


async def _value_error(request: Request, exc: Exception) -> JSONResponse:
    # The analytics functions raise ValueError for inputs they can't work with
    # (e.g. correlation of a single symbol). That is the caller's problem: 422.
    return error_response(422, "invalid_request", str(exc))


async def _http_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _HTTP_CODES.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail))


async def _db_unavailable(request: Request, exc: Exception) -> JSONResponse:
    logger.error("database unavailable: %s", exc)
    return error_response(503, "database_unavailable", "the database is not reachable")


async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return error_response(500, "internal_error", "unexpected server error; see the server log")


def install_error_handlers(app: FastAPI) -> None:
    for exc_type, (status_code, code) in DOMAIN_ERRORS.items():
        app.add_exception_handler(exc_type, _domain_handler(status_code, code))
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(ValueError, _value_error)
    app.add_exception_handler(OperationalError, _db_unavailable)
    app.add_exception_handler(Exception, _unexpected)
