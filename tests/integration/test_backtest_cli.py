"""`fincomb strategies` and `fincomb backtest ...` against the test database, and parity
with the API (both call the same service)."""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from fincomb.api.main import API_PREFIX, create_app
from fincomb.cli import app
from fincomb.config import get_settings
from fincomb.db.base import session_scope
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.services import backtest_service as svc

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNM"]
MOMENTUM = [
    "momentum",
    "SYNA",
    "SYNB",
    "--benchmark",
    "SYNM",
    "--param",
    "lookback=60",
    "--param",
    "skip=5",
    "--param",
    "top_n=1",
]


@pytest.fixture
def cli(session_factory, fixtures_dir, monkeypatch):
    run_ingestion(session_factory, CsvSource(fixtures_dir), SYMBOLS, START, END)
    monkeypatch.setenv("POSTGRES_DB", get_settings().postgres_test_db)
    get_settings.cache_clear()
    yield CliRunner()
    get_settings.cache_clear()


def bt(runner, *args):
    return runner.invoke(app, ["backtest", *args])


def _latest(session_factory):
    with session_scope(session_factory) as s:
        return svc.list_backtests(s)[0]


def test_strategies_command(cli):
    res = cli.invoke(app, ["strategies"])
    assert res.exit_code == 0, res.output
    for name in ("buy_and_hold", "fixed_weights", "optimized", "momentum", "mean_reversion"):
        assert name in res.output
    assert "--param lookback=" in res.output


def test_run_prints_report_and_saves(cli, session_factory):
    res = bt(cli, "run", *MOMENTUM)
    assert res.exit_code == 0, res.output
    assert "momentum (run 1)" in res.output
    assert "vs SYNM: excess CAGR" in res.output
    assert "Newey-West" in res.output
    assert "Saved as run 1." in res.output
    assert _latest(session_factory).id == 1


def test_no_save_and_list(cli, session_factory):
    res = bt(cli, "run", *MOMENTUM, "--no-save")
    assert res.exit_code == 0, res.output
    assert "not saved" in res.output
    assert "No backtests yet" in bt(cli, "list").output
    bt(cli, "run", "buy_and_hold", "SYNM", "--benchmark", "SYNM")
    out = bt(cli, "list").output
    assert "buy_and_hold" in out and "SYNM" in out


def test_show_trades_equity_verify_delete(cli, tmp_path):
    assert (
        bt(
            cli,
            "run",
            "fixed_weights",
            "SYNA",
            "SYNB",
            "--weights",
            "SYNA=0.7,SYNB=0.3",
            "--rebalance",
            "weekly",
            "--benchmark",
            "SYNM",
        ).exit_code
        == 0
    )
    show = bt(cli, "show", "1")
    assert show.exit_code == 0 and "rebalance weekly" in show.output
    trades = bt(cli, "trades", "1", "--limit", "3")
    assert (
        trades.exit_code == 0 and "init" in trades.output and "more (use --limit)" in trades.output
    )
    path = tmp_path / "eq.csv"
    eq = bt(cli, "equity", "1", "--csv", str(path))
    assert eq.exit_code == 0 and path.read_text().startswith("date,equity,cash")
    assert "..." in bt(cli, "equity", "1").output
    v = bt(cli, "verify", "1")
    assert v.exit_code == 0 and "is reproducible" in v.output
    assert "Deleted run 1" in bt(cli, "delete", "1").output
    missing = bt(cli, "show", "1")
    assert missing.exit_code == 1 and "no backtest run 1" in missing.output


def test_errors_exit_1(cli):
    res = bt(cli, "run", "momentum", "SYNA", "--param", "lookbak=3")
    assert res.exit_code == 1 and "unknown parameter" in res.output
    res = bt(cli, "run", "momentum", "SYNA", "--param", "oops")
    assert res.exit_code != 0
    res = bt(cli, "run", "nope", "SYNA")
    assert res.exit_code == 1 and "unknown strategy" in res.output


def test_cli_and_api_give_the_same_numbers(cli, session_factory, fixtures_dir):
    assert bt(cli, "run", *MOMENTUM, "--execution", "same-close", "--cost-bps", "7").exit_code == 0
    via_cli = _latest(session_factory)
    settings = get_settings().model_copy(update={"csv_data_dir": fixtures_dir})
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    res = client.post(
        f"{API_PREFIX}/backtests",
        json={
            "strategy": "momentum",
            "symbols": ["SYNA", "SYNB"],
            "benchmark": "SYNM",
            "params": {"lookback": 60, "skip": 5, "top_n": 1},
            "execution": "same_close",
            "cost_bps": 7,
        },
    )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["metrics"] == via_cli.metrics
    assert body["reproducibility"]["data_hash"] == via_cli.data_hash
    assert body["params"] == via_cli.params
