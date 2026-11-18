"""API tests that need no database: routing, validation, error shape, JSON clean-up.

The app is built with a session factory pointing at a closed port, so any test
that accidentally reached the database would fail instead of using real data.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fincomb.api.errors import ApiError
from fincomb.api.main import API_PREFIX, create_app
from fincomb.api.schemas.common import TimeSeries, finite
from fincomb.config import Settings
from fincomb.db.base import make_engine, make_session_factory
from fincomb.exceptions import (
    DataOwnershipError,
    DataSourceAuthError,
    DataSourceError,
    InsufficientDataError,
    MissingDataError,
    SymbolNotFoundError,
)


@pytest.fixture
def app():
    settings = Settings(_env_file=None, postgres_password="unused", postgres_port=1)
    factory = make_session_factory(make_engine(settings.database_url()))
    return create_app(settings=settings, session_factory=factory)


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _error(res):
    body = res.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    return body["error"]


def test_openapi_lists_every_route(client):
    paths = client.get(f"{API_PREFIX}/openapi.json").json()["paths"]
    expected = {
        "/health": {"get"},
        "/instruments": {"get", "post"},
        "/instruments/{symbol}": {"get", "patch"},
        "/prices/{symbol}": {"get"},
        "/ingestion/runs": {"get", "post"},
        "/ingestion/runs/{run_id}": {"get"},
        "/data/audit": {"get"},
        "/analytics/performance": {"get"},
        "/analytics/returns": {"get"},
        "/analytics/drawdown/{symbol}": {"get"},
        "/analytics/rolling": {"get"},
        "/analytics/correlation": {"get"},
        "/portfolios": {"get", "post"},
        "/portfolios/construct": {"post"},
        "/portfolios/risk": {"post"},
        "/portfolios/{portfolio_id}": {"get", "delete"},
        "/portfolios/{portfolio_id}/risk": {"get"},
    }
    assert {p.removeprefix(API_PREFIX) for p in paths} == set(expected)
    for path, methods in expected.items():
        assert set(paths[API_PREFIX + path]) == methods


def test_docs_page_is_served(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_health_reports_unreachable_database(client):
    res = client.get(f"{API_PREFIX}/health")
    assert res.status_code == 503
    assert res.json()["database"] == "unreachable" and res.json()["status"] == "degraded"


def test_database_down_on_a_data_endpoint_gives_503(client):
    res = client.get(f"{API_PREFIX}/instruments")
    assert res.status_code == 503
    assert _error(res)["code"] == "database_unavailable"


def test_unknown_route_uses_error_shape(client):
    res = client.get(f"{API_PREFIX}/nope")
    assert res.status_code == 404 and _error(res)["code"] == "not_found"


@pytest.mark.parametrize(
    "query",
    [
        "",  # symbols missing
        "symbols=,",  # only separators
        "symbols=AAPL&start=2024-02-01&end=2024-01-01",  # start after end
        "symbols=AAPL&start=yesterday",  # not a date
        "symbols=AAPL&align=sideways",
        "symbols=AAPL&risk_free=5",  # 500%: certainly a units mistake
    ],
)
def test_performance_rejects_bad_query_before_touching_db(client, query):
    res = client.get(f"{API_PREFIX}/analytics/performance?{query}")
    assert res.status_code == 422, res.text
    assert _error(res)["code"] == "invalid_request"


@pytest.mark.parametrize(
    "path",
    [
        "/analytics/returns?symbols=AAPL&frequency=hourly",
        "/analytics/rolling?symbols=AAPL&window=1",
        "/analytics/rolling?symbols=AAPL&metric=beta",
        "/analytics/correlation?symbols=AAPL,SPY&method=kendall",
        "/analytics/drawdown/AAPL?top=0",
        "/ingestion/runs?limit=0",
        "/prices/AAPL?start=2024-02-01&end=2024-01-01",
    ],
)
def test_other_bad_queries_are_422(client, path):
    res = client.get(API_PREFIX + path)
    assert res.status_code == 422, res.text
    assert _error(res)["code"] == "invalid_request"


def test_too_many_symbols(client):
    many = ",".join(f"S{i}" for i in range(51))
    assert client.get(f"{API_PREFIX}/analytics/performance?symbols={many}").status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"symbols": []},
        {"symbols": [f"S{i}" for i in range(21)]},  # more than the 20-symbol cap
        {"symbols": ["AAPL; DROP TABLE"]},
        {"symbols": ["AAPL"], "source": "stooq"},
        {"symbols": ["AAPL"], "mode": "append"},
        {"symbols": ["AAPL"], "start": "2024-02-01", "end": "2024-01-01"},
    ],
)
def test_ingestion_request_validation(client, body):
    res = client.post(f"{API_PREFIX}/ingestion/runs", json=body)
    assert res.status_code == 422, res.text
    err = _error(res)
    assert err["code"] == "invalid_request" and err["details"]["errors"]


def test_instrument_create_validation(client):
    res = client.post(f"{API_PREFIX}/instruments", json={"symbol": "AAPL", "asset_type": "bond"})
    assert res.status_code == 422


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (SymbolNotFoundError("no AAPL"), 404, "symbol_not_found"),
        (DataOwnershipError("owned by yfinance"), 409, "data_ownership_conflict"),
        (InsufficientDataError("need 2"), 422, "insufficient_data"),
        (MissingDataError("gap"), 422, "missing_data"),
        (DataSourceAuthError("no key"), 503, "data_source_auth"),
        (DataSourceError("yahoo down"), 502, "data_source_error"),
        (ValueError("needs two assets"), 422, "invalid_request"),
        (ApiError(409, "already_exists", "dup", {"symbol": "X"}), 409, "already_exists"),
        (RuntimeError("bug"), 500, "internal_error"),
    ],
)
def test_exception_mapping(app, exc, status, code):
    @app.get("/boom")
    def boom():
        raise exc

    res = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert res.status_code == status
    err = _error(res)
    assert err["code"] == code
    if status == 500:
        assert "bug" not in err["message"]  # internals are logged, not returned
    else:
        assert err["message"] == str(exc)


def test_finite_turns_nan_and_inf_into_null():
    assert finite(float("nan")) is None
    assert finite(math.inf) is None and finite(-math.inf) is None
    assert finite(None) is None
    assert finite(0.5) == 0.5


def test_timeseries_from_frame_is_json_safe():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    ts = TimeSeries.from_frame(pd.DataFrame({"A": [float("nan"), 0.01]}, index=idx))
    assert ts.series == {"A": [None, 0.01]}
    assert [d.isoformat() for d in ts.dates] == ["2024-01-02", "2024-01-03"]


def test_serve_defaults_to_localhost_only(monkeypatch):
    import uvicorn
    from typer.testing import CliRunner

    from fincomb import cli

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(
        cli, "get_settings", lambda: Settings(_env_file=None, postgres_password="x")
    )

    res = CliRunner().invoke(cli.app, ["serve"])
    assert res.exit_code == 0, res.output
    (args, kwargs) = calls[0]
    assert args == ("fincomb.api.main:create_app",) and kwargs["factory"] is True
    assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8000
    assert "Warning" not in res.output

    res = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0"])
    assert calls[1][1]["host"] == "0.0.0.0" and "no authentication" in res.output
