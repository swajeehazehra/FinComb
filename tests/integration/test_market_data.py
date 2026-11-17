from datetime import date

import pandas as pd
import pytest
from typer.testing import CliRunner

from fincomb.cli import app
from fincomb.config import get_settings
from fincomb.db.base import session_scope
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.exceptions import SymbolNotFoundError
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion
from fincomb.services.market_data import load_prices

pytestmark = pytest.mark.integration
START, END = date(2023, 1, 1), date(2023, 12, 31)


@pytest.fixture
def loaded(session_factory, fixtures_dir):
    run_ingestion(session_factory, CsvSource(fixtures_dir), ["SYNA", "SYNB"], START, END)
    return session_factory


def _add_short_symbol(factory, n=5):
    idx = pd.bdate_range("2023-06-01", periods=n, name="date")
    df = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "adj_close": 10.0, "volume": 1.0},
        index=idx,
    )
    with session_scope(factory) as s:
        inst = InstrumentRepository(s).get_or_create("SHORT")
        PriceRepository(s).upsert_bars(inst.id, df, "csv", None)


def test_load_prices_wide_frame(loaded):
    with session_scope(loaded) as s:
        wide = load_prices(s, ["syna", "SYNB"], START, END)
    assert list(wide.columns) == ["SYNA", "SYNB"]
    assert len(wide) == 260 and wide.index.is_monotonic_increasing
    assert not wide.isna().any().any()


def test_load_prices_inner_vs_outer_alignment(loaded):
    _add_short_symbol(loaded)
    with session_scope(loaded) as s:
        inner = load_prices(s, ["SYNA", "SHORT"], START, END)
        outer = load_prices(s, ["SYNA", "SHORT"], START, END, align="outer")
    assert len(inner) == 5 and not inner.isna().any().any()
    assert len(outer) == 260 and outer["SHORT"].isna().sum() == 255  # never forward-filled


def test_load_prices_date_filter_and_field(loaded):
    with session_scope(loaded) as s:
        wide = load_prices(s, ["SYNA"], date(2023, 3, 1), date(2023, 3, 31), field="volume")
    assert len(wide) == 23 and wide.index.min() >= pd.Timestamp("2023-03-01")


def test_load_prices_unknown_symbol(loaded):
    with session_scope(loaded) as s, pytest.raises(SymbolNotFoundError, match="fincomb ingest"):
        load_prices(s, ["SYNA", "NOPE"])


@pytest.fixture
def cli_on_test_db(loaded, monkeypatch):
    monkeypatch.setenv("POSTGRES_DB", get_settings().postgres_test_db)
    get_settings.cache_clear()
    yield CliRunner()
    get_settings.cache_clear()


def test_cli_analyze(cli_on_test_db):
    res = cli_on_test_db.invoke(app, ["analyze", "SYNA", "SYNB", "--risk-free", "0.04"])
    assert res.exit_code == 0, res.output
    assert "SYNA" in res.output and "sharpe" in res.output and "risk-free 4.00%" in res.output


def test_cli_correlation_and_drawdowns(cli_on_test_db):
    res = cli_on_test_db.invoke(app, ["correlation", "SYNA", "SYNB"])
    assert res.exit_code == 0 and "1.00" in res.output
    res = cli_on_test_db.invoke(app, ["drawdowns", "SYNA", "--top", "2"])
    assert res.exit_code == 0 and "trough" in res.output


def test_cli_unknown_symbol_exits_nonzero(cli_on_test_db):
    res = cli_on_test_db.invoke(app, ["analyze", "NOPE"])
    assert res.exit_code == 1
