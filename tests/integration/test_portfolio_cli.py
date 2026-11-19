"""`fincomb portfolio ...` against the test database, and agreement with the API."""

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
from fincomb.services import portfolio_service as svc

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)
SYMBOLS = ["SYNA", "SYNB", "SYNM"]


@pytest.fixture
def cli(session_factory, fixtures_dir, monkeypatch):
    run_ingestion(session_factory, CsvSource(fixtures_dir), SYMBOLS, START, END)
    monkeypatch.setenv("POSTGRES_DB", get_settings().postgres_test_db)
    get_settings.cache_clear()
    yield CliRunner()
    get_settings.cache_clear()


def run(runner, *args):
    return runner.invoke(app, ["portfolio", *args])


def test_construct_prints_weights(cli):
    res = run(cli, "construct", *SYMBOLS, "--method", "min-variance", "--max-weight", "0.6")
    assert res.exit_code == 0, res.output
    assert "min_variance: window" in res.output
    assert "shrinkage" in res.output
    for sym in SYMBOLS:
        assert sym in res.output
    assert "WARNING" not in res.output


def test_construct_shows_the_non_positive_warning(cli):
    res = run(cli, "construct", *SYMBOLS, "--method", "max-sharpe", "--risk-free", "0.9")
    assert res.exit_code == 0, res.output
    warning_at = res.output.index("WARNING:")
    assert "not an attractive investment" in res.output
    assert warning_at < res.output.index("risk share")  # above the weights table


def test_compare_lists_every_method(cli):
    res = run(cli, "compare", *SYMBOLS)
    assert res.exit_code == 0, res.output
    for m in ("equal-weight", "inverse-vol", "min-variance", "max-sharpe"):
        assert m in res.output


def test_save_show_list_risk_delete(cli):
    res = run(cli, "construct", *SYMBOLS, "--method", "inverse-vol", "--save", "IV")
    assert res.exit_code == 0 and "Saved as 'IV'" in res.output
    assert run(cli, "create", "6040", "--weights", "SYNA=0.6,SYNB=0.4").exit_code == 0
    listed = run(cli, "list").output
    assert "IV" in listed and "6040" in listed and "manual" in listed
    shown = run(cli, "show", "IV").output
    assert "inverse_vol" in shown and "estimated on" in shown
    risk = run(cli, "risk", "6040", "--benchmark", "SYNM")
    assert risk.exit_code == 0, risk.output
    assert "VaR" in risk.output and "vs SYNM: beta" in risk.output
    assert run(cli, "delete", "6040").exit_code == 0
    gone = run(cli, "show", "6040")
    assert gone.exit_code == 1 and "no portfolio" in gone.output


def test_cli_errors_exit_1(cli):
    bad_sum = run(cli, "create", "bad", "--weights", "SYNA=0.6,SYNB=0.3")
    assert bad_sum.exit_code == 1 and "sum to 0.900000" in bad_sum.output
    bad_method = run(cli, "construct", *SYMBOLS, "--method", "magic")
    assert bad_method.exit_code == 2
    infeasible = run(cli, "construct", *SYMBOLS, "--max-weight", "0.2")
    assert infeasible.exit_code == 1 and "too small" in infeasible.output
    malformed = run(cli, "create", "m", "--weights", "SYNA:1")
    assert malformed.exit_code == 2


def test_cli_and_api_give_the_same_numbers(cli, session_factory, fixtures_dir):
    """One implementation: both interfaces call the same service functions."""
    assert run(cli, "create", "same", "--weights", "SYNA=0.5,SYNB=0.5").exit_code == 0
    with session_scope(session_factory) as s:
        pid = svc.get_portfolio(s, "same").id
        expected = svc.portfolio_risk(s, pid, benchmark="SYNM").report
    settings = get_settings().model_copy(update={"csv_data_dir": fixtures_dir})
    client = TestClient(create_app(settings=settings, session_factory=session_factory))
    body = client.get(f"{API_PREFIX}/portfolios/{pid}/risk", params={"benchmark": "SYNM"}).json()
    assert body["var"]["historical"] == pytest.approx(expected.var_historical, abs=1e-15)
    assert body["beta"]["beta"] == pytest.approx(expected.beta.beta, abs=1e-15)
    out = run(cli, "risk", "same", "--benchmark", "SYNM").output
    assert f"{expected.var_historical * 100:,.2f}%" in out
    assert f"beta {expected.beta.beta:,.2f}" in out


def test_zero_volatility_portfolio_prints_na_and_the_warning(cli, session_factory):
    from tests.integration.test_portfolio_api import _add_flat_symbol

    _add_flat_symbol(session_factory)
    assert run(cli, "create", "cash", "--weights", "FLAT=1").exit_code == 0
    res = run(cli, "risk", "cash", "--benchmark", "")
    assert res.exit_code == 0, res.output
    assert "WARNING: Risk contributions are undefined" in res.output
    row = next(line for line in res.output.splitlines() if line.startswith("FLAT"))
    assert row.split() == ["FLAT", "100.00%", "n/a", "n/a"]
