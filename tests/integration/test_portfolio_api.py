"""Portfolio endpoints through TestClient against the migrated test database."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fincomb.api.main import API_PREFIX, create_app
from fincomb.config import get_settings
from fincomb.db.base import session_scope
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.services import portfolio_service as svc

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNM"]
P = f"{API_PREFIX}/portfolios"


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


@pytest.mark.parametrize("method", ["equal_weight", "inverse_vol", "min_variance", "max_sharpe"])
def test_construct_each_method(client, method):
    res = client.post(f"{P}/construct", json={"symbols": SYMBOLS, "method": method})
    assert res.status_code == 200, res.text
    body = res.json()
    _no_nan(res)
    weights = {w["symbol"]: w["weight"] for w in body["weights"]}
    assert set(weights) == set(SYMBOLS)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(w["percent_of_risk"] for w in body["weights"]) == pytest.approx(1.0)
    assert body["method"] == body["effective_method"] == method
    assert body["window"]["observations"] == 252
    assert body["fallback_used"] is False


def test_construct_matches_service(client, session_factory):
    req = {"symbols": ["syna", "SYNB"], "method": "min_variance", "max_weight": 0.8}
    body = client.post(f"{P}/construct", json=req).json()
    with session_scope(session_factory) as s:
        out = svc.construct_portfolio(
            s, svc.ConstructionSpec(["SYNA", "SYNB"], "min_variance", max_weight=0.8)
        )
    for w in body["weights"]:
        assert w["weight"] == pytest.approx(out.result.weights[w["symbol"]], abs=1e-12)
    assert body["shrinkage"] == pytest.approx(out.result.shrinkage)


def test_construct_non_positive_excess_warning(client):
    res = client.post(
        f"{P}/construct", json={"symbols": SYMBOLS, "method": "max_sharpe", "risk_free": 0.9}
    )
    body = res.json()
    assert res.status_code == 200
    assert body["non_positive_excess_return"] is True
    assert body["effective_method"] == "max_sharpe"
    assert any("not an attractive investment" in w for w in body["warnings"])


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"symbols": SYMBOLS, "method": "risk_parity"}, 422, "invalid_request"),
        ({"symbols": SYMBOLS, "method": "min_variance", "max_weight": 0.3}, 422, "invalid_request"),
        (
            {"symbols": SYMBOLS, "method": "min_variance", "min_weight": 0.5, "max_weight": 0.4},
            422,
            "invalid_request",
        ),
        ({"symbols": ["SYNA", "syna"], "method": "min_variance"}, 422, "invalid_request"),
        ({"symbols": ["SYNA", "NOPE"], "method": "min_variance"}, 404, "symbol_not_found"),
        (
            {"symbols": SYMBOLS, "method": "min_variance", "start": "2023-12-01"},
            422,
            "insufficient_data",
        ),
        (
            {
                "symbols": SYMBOLS,
                "method": "min_variance",
                "start": "2023-06-01",
                "end": "2023-01-01",
            },
            422,
            "invalid_request",
        ),
    ],
)
def test_construct_errors(client, payload, status, code):
    res = client.post(f"{P}/construct", json=payload)
    assert res.status_code == status, res.text
    assert _code(res) == code


def test_save_get_risk_delete_round_trip(client):
    res = client.post(
        P,
        json={
            "name": "Core MV",
            "description": "test",
            "construction": {"symbols": SYMBOLS, "method": "min_variance", "max_weight": 0.7},
        },
    )
    assert res.status_code == 201, res.text
    detail = res.json()
    pid = detail["id"]
    assert detail["construction_method"] == "min_variance"
    assert detail["construction_params"]["max_weight"] == 0.7
    assert detail["estimation_start"] is not None

    listed = client.get(P).json()
    assert [p["name"] for p in listed] == ["Core MV"] and listed[0]["assets"] == 3
    assert client.get(f"{P}/{pid}").json()["weights"] == detail["weights"]

    risk = client.get(f"{P}/{pid}/risk", params={"benchmark": "SYNM", "risk_free": 0.02})
    assert risk.status_code == 200, risk.text
    _no_nan(risk)
    r = risk.json()
    assert r["portfolio_id"] == pid and r["horizon_days"] == 1
    assert r["cvar"]["historical"] >= r["var"]["historical"]
    assert r["beta"]["benchmark"] == "SYNM"
    total = sum(c["risk_contribution"] for c in r["contributions"])
    assert total == pytest.approx(r["performance"]["annual_volatility"])

    assert client.delete(f"{P}/{pid}").status_code == 204
    assert client.get(f"{P}/{pid}").status_code == 404
    assert _code(client.delete(f"{P}/{pid}")) == "not_found"


def test_saved_and_unsaved_risk_agree(client):
    weights = {"SYNA": 0.5, "SYNB": 0.5}
    pid = client.post(P, json={"name": "half", "weights": weights}).json()["id"]
    saved = client.get(f"{P}/{pid}/risk", params={"benchmark": ""}).json()
    unsaved = client.post(f"{P}/risk", json={"weights": weights, "benchmark": None}).json()
    assert saved["beta"] is None and unsaved["beta"] is None
    assert saved["var"] == unsaved["var"] and saved["cvar"] == unsaved["cvar"]
    assert unsaved["portfolio_id"] is None


def test_manual_portfolio_errors(client):
    ok = {"name": "x", "weights": {"SYNA": 0.6, "SYNB": 0.4}}
    assert client.post(P, json=ok).status_code == 201
    dup = client.post(P, json=ok)
    assert dup.status_code == 409 and _code(dup) == "already_exists"
    bad_sum = client.post(P, json={"name": "y", "weights": {"SYNA": 0.6, "SYNB": 0.38}})
    assert bad_sum.status_code == 422 and "0.980000" in bad_sum.json()["error"]["message"]
    neg = client.post(P, json={"name": "y", "weights": {"SYNA": 1.2, "SYNB": -0.2}})
    assert neg.status_code == 422
    unknown = client.post(P, json={"name": "y", "weights": {"NOPE": 1.0}})
    assert unknown.status_code == 404 and _code(unknown) == "symbol_not_found"
    both = client.post(
        P,
        json={
            "name": "y",
            "weights": {"SYNA": 1.0},
            "construction": {"symbols": ["SYNA"], "method": "equal_weight"},
        },
    )
    assert both.status_code == 422
    neither = client.post(P, json={"name": "y"})
    assert neither.status_code == 422


def test_risk_errors(client):
    assert client.get(f"{P}/999/risk").status_code == 404
    pid = client.post(P, json={"name": "a", "weights": {"SYNA": 1.0}}).json()["id"]
    missing_bench = client.get(f"{P}/{pid}/risk")  # default SPY isn't stored in the tests
    assert missing_bench.status_code == 404 and _code(missing_bench) == "symbol_not_found"
    assert client.get(f"{P}/{pid}/risk", params={"confidence": 0.3}).status_code == 422
    bad_dates = client.get(f"{P}/{pid}/risk", params={"start": "2023-06-01", "end": "2023-01-01"})
    assert bad_dates.status_code == 422


def test_openapi_lists_portfolio_routes(client):
    paths = client.get(f"{API_PREFIX}/openapi.json").json()["paths"]
    for path in (
        "/portfolios",
        "/portfolios/construct",
        "/portfolios/risk",
        "/portfolios/{portfolio_id}",
        "/portfolios/{portfolio_id}/risk",
    ):
        assert f"{API_PREFIX}{path}" in paths


def _add_flat_symbol(factory, symbol="FLAT"):
    """A symbol whose price never moves, on the same dates as the fixtures."""
    idx = pd.bdate_range("2023-01-02", "2023-12-29", name="date")
    bars = pd.DataFrame(
        {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "adj_close": 10.0, "volume": 1},
        index=idx,
    )
    with session_scope(factory) as s:
        inst = InstrumentRepository(s).get_or_create(symbol)
        inst.data_source = "csv"
        PriceRepository(s).upsert_bars(inst.id, bars, "csv", None)


def test_zero_volatility_portfolio_serialises_null_contributions(client, session_factory):
    _add_flat_symbol(session_factory)
    res = client.post(f"{P}/risk", json={"weights": {"FLAT": 1.0}, "benchmark": None})
    assert res.status_code == 200, res.text
    _no_nan(res)
    body = res.json()
    (row,) = body["contributions"]
    assert row["weight"] == 1.0
    assert row["marginal_risk"] is None
    assert row["risk_contribution"] is None
    assert row["percent_of_risk"] is None
    assert any("effectively zero" in w for w in body["warnings"])
    # A flat asset inside a risky portfolio keeps its exact zero contribution.
    mixed = client.post(
        f"{P}/risk", json={"weights": {"FLAT": 0.5, "SYNA": 0.5}, "benchmark": None}
    ).json()
    by_symbol = {c["symbol"]: c for c in mixed["contributions"]}
    assert by_symbol["FLAT"]["percent_of_risk"] == 0.0
    assert by_symbol["SYNA"]["percent_of_risk"] == pytest.approx(1.0)
    assert not any("effectively zero" in w for w in mixed["warnings"])
