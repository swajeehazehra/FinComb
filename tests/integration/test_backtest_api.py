"""Strategy and backtest endpoints through TestClient against the test database."""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from fincomb.api.main import API_PREFIX, create_app
from fincomb.config import get_settings
from fincomb.db.base import session_scope
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.services import backtest_service as svc

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNM"]
B = f"{API_PREFIX}/backtests"
BODY = {
    "strategy": "momentum",
    "symbols": ["SYNA", "SYNB"],
    "params": {"lookback": 60, "skip": 5, "top_n": 1},
    "benchmark": "SYNM",
}


@pytest.fixture
def client(session_factory, fixtures_dir):
    settings = get_settings().model_copy(
        update={"csv_data_dir": fixtures_dir, "default_data_source": "csv"}
    )
    run_ingestion(session_factory, CsvSource(fixtures_dir), SYMBOLS, START, END)
    return TestClient(create_app(settings=settings, session_factory=session_factory))


def _code(res):
    return res.json()["error"]["code"]


def _no_nan(res):
    json.loads(res.text, parse_constant=lambda c: pytest.fail(f"{c} in JSON"))


def test_strategies_catalogue(client):
    res = client.get(f"{API_PREFIX}/strategies")
    assert res.status_code == 200
    by_name = {s["name"]: s for s in res.json()}
    assert set(by_name) == {
        "buy_and_hold",
        "fixed_weights",
        "optimized",
        "momentum",
        "mean_reversion",
    }
    mom = by_name["momentum"]
    assert mom["default_rebalance"] == "monthly"
    assert mom["params_schema"]["properties"]["lookback"]["default"] == 252


def test_run_store_and_read_back(client, session_factory):
    res = client.post(B, json=BODY)
    assert res.status_code == 201, res.text
    _no_nan(res)
    body = res.json()
    run_id = body["id"]
    assert body["execution"] == "next_close" and body["rebalance"] == "monthly"
    assert body["cost_bps"] == 5 and body["slippage_bps"] == 5
    assert len(body["reproducibility"]["data_hash"]) == 64
    assert set(body["metrics"]) >= {"strategy", "benchmark", "relative", "trading"}
    with session_scope(session_factory) as s:
        stored = svc.get_backtest(s, run_id)
    assert body["metrics"] == stored.metrics

    detail = client.get(f"{B}/{run_id}").json()
    assert detail == body
    listing = client.get(B).json()
    assert [r["id"] for r in listing] == [run_id]
    assert listing[0]["cagr"] == pytest.approx(body["metrics"]["strategy"]["cagr"])

    eq = client.get(f"{B}/{run_id}/equity")
    assert eq.status_code == 200
    _no_nan(eq)
    curves = eq.json()["curves"]
    assert set(curves["series"]) == {
        "equity",
        "cash",
        "gross_exposure",
        "benchmark_equity",
        "drawdown",
        "turnover",
        "costs",
    }
    assert curves["dates"][0] == body["start"]

    trades = client.get(f"{B}/{run_id}/trades", params={"limit": 2}).json()
    assert len(trades["trades"]) == 2 and trades["total"] >= 2
    assert trades["trades"][0]["kind"] == "init"

    v = client.get(f"{B}/{run_id}/verify").json()
    assert v["reproducible"] is True

    assert client.delete(f"{B}/{run_id}").status_code == 204
    assert client.get(f"{B}/{run_id}").status_code == 404


def test_unsaved_run_returns_200_and_no_id(client):
    res = client.post(B, json={**BODY, "save": False})
    assert res.status_code == 200
    assert res.json()["id"] is None
    assert client.get(B).json() == []


def test_saved_portfolio_by_name(client):
    assert (
        client.post(
            f"{API_PREFIX}/portfolios", json={"name": "6040", "weights": {"SYNA": 0.6, "SYNB": 0.4}}
        ).status_code
        == 201
    )
    res = client.post(
        B, json={"strategy": "fixed_weights", "portfolio": "6040", "benchmark": "SYNM"}
    )
    assert res.status_code == 201, res.text
    assert res.json()["params"]["weights"] == {"SYNA": 0.6, "SYNB": 0.4}


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({**BODY, "strategy": "martingale"}, 422, "invalid_request"),
        ({**BODY, "params": {"lookbak": 3}}, 422, "invalid_request"),
        ({**BODY, "execution": "next_open"}, 422, "invalid_request"),
        ({**BODY, "rebalance": "hourly"}, 422, "invalid_request"),
        ({**BODY, "start": "2023-06-01", "end": "2023-01-01"}, 422, "invalid_request"),
        ({**BODY, "symbols": ["SYNA", "syna"]}, 422, "invalid_request"),
        ({**BODY, "symbols": []}, 422, "invalid_request"),
        ({**BODY, "symbols": ["NOPE"]}, 404, "symbol_not_found"),
        ({**BODY, "params": {"lookback": 400}}, 422, "insufficient_data"),
        ({**BODY, "cost_bps": -1}, 422, "invalid_request"),
        (
            {**BODY, "strategy": "buy_and_hold", "params": {}, "rebalance": "monthly"},
            422,
            "invalid_request",
        ),
        (
            {**BODY, "portfolio": "missing", "strategy": "fixed_weights", "params": {}},
            404,
            "not_found",
        ),
    ],
)
def test_errors(client, body, status, code):
    res = client.post(B, json=body)
    assert res.status_code == status, res.text
    assert _code(res) == code


def test_unknown_run_is_404(client):
    for path in ("", "/equity", "/trades", "/verify"):
        res = client.get(f"{B}/999{path}")
        assert res.status_code == 404 and _code(res) == "not_found"
    assert client.delete(f"{B}/999").status_code == 404
