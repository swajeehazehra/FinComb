"""API integration tests: TestClient against the migrated test database.

The app is built with the test session factory and test settings, so nothing here
reads or writes the development database. Prices come from the CSV fixtures
through the real ingestion pipeline.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from fincomb.analytics.correlation import correlation_matrix
from fincomb.analytics.returns import simple_returns
from fincomb.analytics.risk import drawdown_episodes, drawdown_series
from fincomb.analytics.summary import performance_summary
from fincomb.api.main import API_PREFIX, create_app
from fincomb.config import get_settings
from fincomb.db.base import session_scope
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.services.market_data import load_prices

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
V1 = API_PREFIX


@pytest.fixture
def settings(fixtures_dir):
    return get_settings().model_copy(
        update={"csv_data_dir": fixtures_dir, "default_data_source": "csv"}
    )


@pytest.fixture
def client(session_factory, settings):
    return TestClient(create_app(settings=settings, session_factory=session_factory))


@pytest.fixture
def loaded(session_factory, fixtures_dir):
    run_ingestion(session_factory, CsvSource(fixtures_dir), ["SYNA", "SYNB"], START, END)
    return session_factory


def _add_short_symbol(factory, n=30):
    idx = pd.bdate_range("2023-06-01", periods=n, name="date")
    df = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "adj_close": 10.0, "volume": 1.0},
        index=idx,
    )
    df["adj_close"] = df["close"] = [10.0 + 0.1 * (i % 3) for i in range(n)]
    df["high"] = 11.0
    with session_scope(factory) as s:
        inst = InstrumentRepository(s).get_or_create("SHORT")
        inst.data_source = "csv"
        PriceRepository(s).upsert_bars(inst.id, df, "csv", None)


def _prices(factory, symbols, align="inner"):
    with session_scope(factory) as s:
        return load_prices(s, symbols, START, END, align=align)


def _error_code(res):
    return res.json()["error"]["code"]


# --- system -------------------------------------------------------------------------------


def test_health_with_database(client):
    res = client.get(f"{V1}/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok" and body["database"] == "ok"
    assert body["schema_revision"]  # migrated by the conftest


# --- instruments --------------------------------------------------------------------------


def test_list_and_detail_instruments(client, loaded):
    res = client.get(f"{V1}/instruments")
    assert res.status_code == 200
    assert [i["symbol"] for i in res.json()] == ["SYNA", "SYNB"]

    detail = client.get(f"{V1}/instruments/syna").json()
    assert detail["symbol"] == "SYNA" and detail["data_source"] == "csv"
    assert detail["coverage"] == {
        "first_date": "2023-01-02",
        "last_date": "2023-12-29",
        "bars": 260,
    }


def test_instrument_filters(client, loaded):
    client.patch(f"{V1}/instruments/SYNB", json={"asset_type": "etf"})
    assert [i["symbol"] for i in client.get(f"{V1}/instruments?asset_type=etf").json()] == ["SYNB"]
    assert len(client.get(f"{V1}/instruments?active=true").json()) == 2
    assert client.get(f"{V1}/instruments?active=false").json() == []


def test_create_instrument_then_conflict(client):
    res = client.post(f"{V1}/instruments", json={"symbol": "spy", "asset_type": "etf"})
    assert res.status_code == 201
    body = res.json()
    assert body["symbol"] == "SPY" and body["asset_type"] == "etf"
    assert body["coverage"]["bars"] == 0 and body["data_source"] is None

    res = client.post(f"{V1}/instruments", json={"symbol": "SPY"})
    assert res.status_code == 409 and _error_code(res) == "already_exists"


def test_patch_asset_type_and_unknown_symbol(client, loaded):
    res = client.patch(f"{V1}/instruments/SYNA", json={"asset_type": "index"})
    assert res.status_code == 200 and res.json()["asset_type"] == "index"
    assert client.get(f"{V1}/instruments/SYNA").json()["asset_type"] == "index"  # committed

    for res in (
        client.get(f"{V1}/instruments/NOPE"),
        client.patch(f"{V1}/instruments/NOPE", json={"asset_type": "etf"}),
    ):
        assert res.status_code == 404 and _error_code(res) == "not_found"


# --- prices -------------------------------------------------------------------------------


def test_prices_raw_and_adjusted(client, loaded, session_factory):
    with session_scope(session_factory) as s:
        # Make one bar's adj_close differ from close, as after a dividend.
        s.execute(
            text(
                "UPDATE price_bars SET adj_close = close * 0.5 FROM instruments i WHERE "
                "i.id = price_bars.instrument_id AND i.symbol = 'SYNA' "
                "AND bar_date = '2023-03-01'"
            )
        )
    raw = client.get(f"{V1}/prices/SYNA?start=2023-03-01&end=2023-03-03").json()
    assert raw["count"] == 3 and raw["adjusted"] is False
    bar = raw["bars"][0]
    assert bar["date"] == "2023-03-01"

    adj = client.get(f"{V1}/prices/SYNA?start=2023-03-01&end=2023-03-03&adjusted=true").json()
    abar = adj["bars"][0]
    for col in ("open", "high", "low", "close"):
        assert abar[col] == pytest.approx(bar[col] * 0.5)
    assert abar["volume"] == bar["volume"]
    assert adj["bars"][1]["close"] == pytest.approx(raw["bars"][1]["close"])  # factor 1


def test_prices_empty_range_and_unknown_symbol(client, loaded):
    res = client.get(f"{V1}/prices/SYNA?start=2030-01-01")
    assert res.status_code == 200 and res.json()["bars"] == []
    res = client.get(f"{V1}/prices/NOPE")
    assert res.status_code == 404 and _error_code(res) == "symbol_not_found"


# --- ingestion ----------------------------------------------------------------------------


def test_post_ingestion_run_and_rerun_adds_no_duplicates(client):
    body = {"symbols": ["syna", "SYNB"], "start": "2023-01-01", "end": "2023-12-31"}
    res = client.post(f"{V1}/ingestion/runs", json=body)
    assert res.status_code == 201, res.text
    (run,) = res.json()["runs"]
    assert run["status"] == "success" and run["source"] == "csv"
    assert run["inserted"] == 520 and [r["symbol"] for r in run["results"]] == ["SYNA", "SYNB"]

    again = client.post(f"{V1}/ingestion/runs", json=body).json()["runs"][0]
    assert again["inserted"] == 0 and again["status"] == "success"
    assert client.get(f"{V1}/instruments/SYNA").json()["coverage"]["bars"] == 260

    runs = client.get(f"{V1}/ingestion/runs").json()
    assert [r["id"] for r in runs] == [again["run_id"], run["run_id"]]  # newest first
    assert client.get(f"{V1}/ingestion/runs?limit=1").json()[0]["id"] == again["run_id"]


def test_ingestion_partial_failure_is_reported_per_symbol(client):
    body = {"symbols": ["SYNA", "MISSING"], "start": "2023-01-01", "end": "2023-12-31"}
    run = client.post(f"{V1}/ingestion/runs", json=body).json()["runs"][0]
    assert run["status"] == "partial"
    errors = {r["symbol"]: r["error"] for r in run["results"]}
    assert errors["SYNA"] is None and "MISSING" in errors["MISSING"]


def test_ingestion_ownership_conflict_is_reported(client, loaded):
    body = {"symbols": ["SYNA"], "start": "2023-01-01", "source": "yfinance"}
    run = client.post(f"{V1}/ingestion/runs", json=body).json()["runs"][0]
    assert run["status"] == "failed"
    assert "csv" in run["results"][0]["error"] and "refresh" in run["results"][0]["error"]


def test_ingestion_end_is_capped_to_last_completed_session(client):
    body = {"symbols": ["SYNA"], "start": "2023-01-01", "end": "2999-01-01"}
    run = client.post(f"{V1}/ingestion/runs", json=body).json()["runs"][0]
    assert run["end_capped_to"] is not None and run["end_capped_to"] < "2999-01-01"


def test_refresh_mode_replaces_history(client, loaded):
    run = client.post(f"{V1}/ingestion/runs", json={"symbols": ["SYNA"], "mode": "refresh"})
    assert run.status_code == 201
    result = run.json()["runs"][0]["results"][0]
    assert result["replaced"] is True and result["error"] is None


def test_run_detail_lists_issues(client):
    body = {"symbols": ["BADDATA"], "start": "2023-01-01", "end": "2023-12-31"}
    run_id = client.post(f"{V1}/ingestion/runs", json=body).json()["runs"][0]["run_id"]
    detail = client.get(f"{V1}/ingestion/runs/{run_id}").json()
    assert detail["id"] == run_id and detail["symbols"] == ["BADDATA"]
    assert detail["issues"], "BADDATA fixture should produce data-quality issues"
    assert {i["symbol"] for i in detail["issues"]} == {"BADDATA"}
    assert {"rule", "severity", "bar_date", "detail"} <= set(detail["issues"][0])

    res = client.get(f"{V1}/ingestion/runs/999999")
    assert res.status_code == 404 and _error_code(res) == "not_found"


def test_data_audit(client, loaded, session_factory):
    body = client.get(f"{V1}/data/audit").json()
    assert body["ok"] is True and {r["status"] for r in body["rows"]} == {"OK"}

    with session_scope(session_factory) as s:
        s.execute(
            text(
                "UPDATE price_bars SET source = 'tiingo' FROM instruments i WHERE "
                "i.id = price_bars.instrument_id AND i.symbol = 'SYNA' "
                "AND bar_date >= '2023-07-01'"
            )
        )
    body = client.get(f"{V1}/data/audit").json()
    assert body["ok"] is False
    syna = next(r for r in body["rows"] if r["symbol"] == "SYNA")
    assert syna["status"] == "MIXED SOURCES" and syna["sources"] == ["csv", "tiingo"]


# --- analytics: numbers equal the analytics functions called directly ---------------------


def test_performance_matches_analytics(client, loaded):
    res = client.get(
        f"{V1}/analytics/performance?symbols=SYNA,SYNB&start=2023-01-01&end=2023-12-31"
        "&risk_free=0.04"
    )
    assert res.status_code == 200, res.text
    body = res.json()
    expected = performance_summary(_prices(loaded, ["SYNA", "SYNB"]), risk_free=0.04)
    assert [r["symbol"] for r in body["rows"]] == ["SYNA", "SYNB"]
    for row in body["rows"]:
        exp = expected.loc[row["symbol"]]
        assert row["start"] == str(exp["start"]) and row["observations"] == exp["observations"]
        for key in ("total_return", "cagr", "annual_volatility", "sharpe", "sortino",
                    "max_drawdown", "calmar", "best_day", "worst_day"):  # fmt: skip
            assert row[key] == pytest.approx(exp[key], rel=1e-12), key
    assert body["notes"] == [] and body["align"] == "inner"


def test_repeated_symbols_query_form(client, loaded):
    a = client.get(f"{V1}/analytics/performance?symbols=SYNA&symbols=SYNB").json()
    b = client.get(f"{V1}/analytics/performance?symbols=SYNA,SYNB").json()
    assert a == b


def test_alignment_notes_are_returned(client, loaded):
    _add_short_symbol(loaded)
    inner = client.get(f"{V1}/analytics/performance?symbols=SYNA,SHORT").json()
    assert any("SHORT has no data before then" in n for n in inner["notes"])
    assert {r["start"] for r in inner["rows"]} == {"2023-06-01"}

    outer = client.get(f"{V1}/analytics/performance?symbols=SYNA,SHORT&align=outer").json()
    assert outer["notes"] == []
    starts = {r["symbol"]: r["start"] for r in outer["rows"]}
    assert starts == {"SYNA": "2023-01-02", "SHORT": "2023-06-01"}


def test_returns_daily_monthly_cumulative(client, loaded):
    prices = _prices(loaded, ["SYNA", "SYNB"])
    daily = client.get(f"{V1}/analytics/returns?symbols=SYNA,SYNB").json()
    expected = simple_returns(prices)
    assert len(daily["dates"]) == len(expected)
    assert daily["series"]["SYNA"] == pytest.approx(expected["SYNA"].tolist())

    monthly = client.get(f"{V1}/analytics/returns?symbols=SYNA&frequency=monthly").json()
    assert len(monthly["dates"]) == 12 and monthly["dates"][0] == "2023-01-31"

    cum = client.get(f"{V1}/analytics/returns?symbols=SYNA&cumulative=true").json()
    total = prices["SYNA"].iloc[-1] / prices["SYNA"].iloc[0] - 1
    assert cum["series"]["SYNA"][-1] == pytest.approx(total)


def test_compounding_across_a_gap_is_refused(client, loaded):
    _add_short_symbol(loaded)
    with session_scope(loaded) as s:
        s.execute(
            text(
                "DELETE FROM price_bars USING instruments i WHERE i.id = price_bars.instrument_id "
                "AND i.symbol = 'SHORT' AND bar_date = '2023-06-05'"
            )
        )
    res = client.get(f"{V1}/analytics/returns?symbols=SYNA,SHORT&align=outer&cumulative=true")
    assert res.status_code == 422 and _error_code(res) == "missing_data"


def test_rolling_volatility_and_sharpe(client, loaded):
    vol = client.get(f"{V1}/analytics/rolling?symbols=SYNA&window=20").json()
    assert vol["metric"] == "volatility" and vol["window"] == 20
    assert len(vol["dates"]) == 259 - 19  # 259 returns, first full window at the 20th
    assert all(v is not None and v > 0 for v in vol["series"]["SYNA"])

    sharpe = client.get(f"{V1}/analytics/rolling?symbols=SYNA&metric=sharpe&window=60").json()
    assert sharpe["window"] == 60 and len(sharpe["dates"]) == 259 - 59


def test_correlation_matches_analytics(client, loaded):
    body = client.get(f"{V1}/analytics/correlation?symbols=SYNA,SYNB&method=spearman").json()
    expected = correlation_matrix(simple_returns(_prices(loaded, ["SYNA", "SYNB"])), "spearman")
    assert body["symbols"] == ["SYNA", "SYNB"] and body["method"] == "spearman"
    flat = [v for row in body["matrix"] for v in row]
    assert flat == pytest.approx(expected.to_numpy().ravel().tolist())
    assert body["matrix"][0][0] == pytest.approx(1.0)


def test_correlation_needs_two_symbols(client, loaded):
    res = client.get(f"{V1}/analytics/correlation?symbols=SYNA")
    assert res.status_code == 422 and _error_code(res) == "invalid_request"


def test_drawdown_matches_analytics(client, loaded):
    body = client.get(f"{V1}/analytics/drawdown/SYNA?top=3").json()
    r = simple_returns(_prices(loaded, ["SYNA"])["SYNA"])
    assert body["drawdown"] == pytest.approx(drawdown_series(r).tolist())
    assert body["max_drawdown"] == pytest.approx(drawdown_series(r).min())
    expected = drawdown_episodes(r, top=3)
    assert len(body["episodes"]) == len(expected)
    assert body["episodes"][0]["depth"] == pytest.approx(expected[0].depth)
    assert body["episodes"][0]["trough"] == expected[0].trough.date().isoformat()


def test_analytics_unknown_symbol_and_too_little_data(client, loaded):
    res = client.get(f"{V1}/analytics/performance?symbols=SYNA,NOPE")
    assert res.status_code == 404 and _error_code(res) == "symbol_not_found"
    res = client.get(f"{V1}/analytics/performance?symbols=SYNA&start=2023-03-01&end=2023-03-01")
    assert res.status_code == 422 and _error_code(res) == "insufficient_data"
