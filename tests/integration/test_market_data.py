from datetime import date

import pandas as pd
import pytest
from sqlalchemy import text
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


def test_load_prices_inner_vs_outer_alignment(loaded, caplog):
    _add_short_symbol(loaded)
    with session_scope(loaded) as s, caplog.at_level("WARNING"):
        inner = load_prices(s, ["SYNA", "SHORT"], START, END)
        outer = load_prices(s, ["SYNA", "SHORT"], START, END, align="outer")
    assert len(inner) == 5 and not inner.isna().any().any()
    assert len(outer) == 260 and outer["SHORT"].isna().sum() == 255  # never forward-filled
    text = caplog.text
    assert "starts 2023-06-01 because SHORT has no data before then" in text
    assert "ends 2023-06-07 because SHORT" in text


def test_load_prices_drops_a_date_missing_for_one_symbol(loaded, caplog):
    with session_scope(loaded) as s:
        s.execute(
            text(
                "DELETE FROM price_bars USING instruments WHERE "
                "price_bars.instrument_id = instruments.id AND symbol = 'SYNB' "
                "AND bar_date = '2023-03-15'"
            )
        )
    with session_scope(loaded) as s, caplog.at_level("WARNING"):
        inner = load_prices(s, ["SYNA", "SYNB"], START, END)
        outer = load_prices(s, ["SYNA", "SYNB"], START, END, align="outer")
    assert pd.Timestamp("2023-03-15") not in inner.index and len(inner) == 259
    assert "1 date(s) inside the common period were dropped" in caplog.text
    assert outer.loc["2023-03-15", "SYNB"] != outer.loc["2023-03-15", "SYNB"]  # NaN kept


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


def test_cli_analyze_outer_shows_each_symbols_start(cli_on_test_db, loaded):
    _add_short_symbol(loaded, n=30)
    res = cli_on_test_db.invoke(app, ["analyze", "SYNA", "SHORT", "--align", "outer"])
    assert res.exit_code == 0, res.output
    assert "own history" in res.output and "2023-06-01" in res.output
    res = cli_on_test_db.invoke(app, ["analyze", "SYNA", "--align", "sideways"])
    assert res.exit_code != 0


def test_cli_correlation_outer(cli_on_test_db, loaded):
    _add_short_symbol(loaded, n=30)
    res = cli_on_test_db.invoke(app, ["correlation", "SYNA", "SYNB", "SHORT", "--align", "outer"])
    assert res.exit_code == 0, res.output


def _add_stale_symbol(factory):
    """STALE trades all year but its data stops two sessions before SYNA/SYNB."""
    idx = pd.bdate_range("2023-01-02", "2023-12-27", name="date")
    df = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "adj_close": 10.0, "volume": 1.0},
        index=idx,
    )
    with session_scope(factory) as s:
        inst = InstrumentRepository(s).get_or_create("STALE")
        PriceRepository(s).upsert_bars(inst.id, df, "csv", None)


def _row(output: str, symbol: str) -> list[str]:
    return next(line.split() for line in output.splitlines() if line.startswith(symbol))


STALE_NOTE = (
    "Note: Common period ends 2023-12-27 because STALE has no data after then "
    "(others run to 2023-12-29)."
)


def test_load_prices_keeps_alignment_notes(loaded):
    _add_stale_symbol(loaded)
    with session_scope(loaded) as s:
        inner = load_prices(s, ["SYNA", "STALE"], START, END)
        outer = load_prices(s, ["SYNA", "STALE"], START, END, align="outer")
        same = load_prices(s, ["SYNA", "SYNB"], START, END)
    assert inner.attrs["alignment_notes"] == [STALE_NOTE.removeprefix("Note: ")]
    assert outer.attrs["alignment_notes"] == [] and same.attrs["alignment_notes"] == []


def test_cli_analyze_outer_shows_each_symbols_own_end_date(cli_on_test_db, loaded):
    _add_stale_symbol(loaded)
    res = cli_on_test_db.invoke(app, ["analyze", "SYNA", "STALE", "--align", "outer"])
    assert res.exit_code == 0, res.output
    assert "from" in res.stdout and " to " in res.stdout
    assert _row(res.stdout, "SYNA")[1:4] == ["2023-01-02", "2023-12-29", "-25.15%"]
    assert _row(res.stdout, "STALE")[1:3] == ["2023-01-02", "2023-12-27"]
    assert "Note:" not in res.stdout


def test_cli_analyze_inner_shows_window_and_warns_on_stdout(cli_on_test_db, loaded):
    _add_stale_symbol(loaded)
    res = cli_on_test_db.invoke(app, ["analyze", "SYNA", "STALE"])
    assert res.exit_code == 0, res.output
    # every row is measured over the shortened common window...
    assert _row(res.stdout, "SYNA")[1:4] == ["2023-01-02", "2023-12-27", "-25.47%"]
    assert _row(res.stdout, "STALE")[1:3] == ["2023-01-02", "2023-12-27"]
    # ...and the reason is printed with the results, not only in the log
    assert STALE_NOTE in res.stdout


def test_cli_analyze_inner_no_note_when_windows_match(cli_on_test_db):
    res = cli_on_test_db.invoke(app, ["analyze", "SYNA", "SYNB"])
    assert res.exit_code == 0, res.output
    assert _row(res.stdout, "SYNA")[1:3] == ["2023-01-02", "2023-12-29"]
    assert "Note:" not in res.stdout


def test_cli_correlation_inner_warns_on_stdout(cli_on_test_db, loaded):
    _add_stale_symbol(loaded)
    res = cli_on_test_db.invoke(app, ["correlation", "SYNA", "SYNB", "STALE"])
    assert res.exit_code == 0, res.output
    assert STALE_NOTE in res.stdout
