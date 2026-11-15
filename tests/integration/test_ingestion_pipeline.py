from datetime import date

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from fincomb.db.base import session_scope
from fincomb.db.models import (
    AssetType,
    DataQualityIssue,
    IngestionRun,
    Instrument,
    PriceBar,
    RunStatus,
)
from fincomb.db.repositories import InstrumentRepository, PriceRepository
from fincomb.ingestion.csv_source import CsvSource
from fincomb.ingestion.pipeline import run_ingestion

pytestmark = pytest.mark.integration

START, END = date(2023, 1, 1), date(2023, 12, 31)


def count(factory, model):
    with session_scope(factory) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_ingest_stores_bars_and_audit(session_factory, fixtures_dir):
    summary = run_ingestion(session_factory, CsvSource(fixtures_dir), ["SYNA", "SYNM"], START, END)
    assert summary.status is RunStatus.SUCCESS
    assert summary.inserted == 520 and summary.updated == 0
    assert count(session_factory, PriceBar) == 520

    with session_scope(session_factory) as s:
        run = s.get(IngestionRun, summary.run_id)
        assert run.status is RunStatus.SUCCESS
        assert run.rows_inserted == 520 and run.finished_at is not None
        assert run.symbols == ["SYNA", "SYNM"]


def test_reingest_is_idempotent(session_factory, fixtures_dir):
    src = CsvSource(fixtures_dir)
    run_ingestion(session_factory, src, ["SYNA"], START, END)
    second = run_ingestion(session_factory, src, ["SYNA"], START, END)
    assert second.inserted == 0 and second.updated == 260
    assert count(session_factory, PriceBar) == 260
    assert count(session_factory, Instrument) == 1


def test_overlapping_ranges_only_add_new_days(session_factory, fixtures_dir):
    src = CsvSource(fixtures_dir)
    first = run_ingestion(session_factory, src, ["SYNA"], date(2023, 1, 1), date(2023, 6, 30))
    second = run_ingestion(session_factory, src, ["SYNA"], date(2023, 6, 1), date(2023, 12, 31))
    assert first.inserted + second.inserted == 260
    assert second.updated == 22  # business days in June 2023 overlap
    assert count(session_factory, PriceBar) == 260


def test_bad_rows_rejected_and_recorded(session_factory, fixtures_dir):
    summary = run_ingestion(session_factory, CsvSource(fixtures_dir), ["BADDATA"], START, END)
    assert summary.status is RunStatus.SUCCESS
    r = summary.results[0]
    assert r.rejected >= 5 and r.warnings >= 2
    assert count(session_factory, PriceBar) == r.inserted

    with session_scope(session_factory) as s:
        rules = set(s.scalars(select(DataQualityIssue.rule)))
    assert {"missing_value", "ohlc_inconsistent", "duplicate_date", "stale_price"} <= rules


def test_one_bad_symbol_gives_partial_run(session_factory, fixtures_dir):
    summary = run_ingestion(
        session_factory, CsvSource(fixtures_dir), ["SYNA", "DOESNOTEXIST"], START, END
    )
    assert summary.status is RunStatus.PARTIAL
    assert count(session_factory, PriceBar) == 260
    with session_scope(session_factory) as s:
        run = s.get(IngestionRun, summary.run_id)
        assert "DOESNOTEXIST" in run.error_message


def test_all_symbols_failing_gives_failed_run(session_factory, fixtures_dir):
    summary = run_ingestion(
        session_factory, CsvSource(fixtures_dir), ["NOPE1", "NOPE2"], START, END
    )
    assert summary.status is RunStatus.FAILED
    assert count(session_factory, Instrument) == 0


def test_symbols_are_deduplicated_and_uppercased(session_factory, fixtures_dir):
    summary = run_ingestion(
        session_factory,
        CsvSource(fixtures_dir),
        ["syna", "SYNA ", "SYNA"],
        START,
        END,
        asset_type=AssetType.ETF,
    )
    assert [r.symbol for r in summary.results] == ["SYNA"]
    with session_scope(session_factory) as s:
        assert InstrumentRepository(s).get_by_symbol("syna").asset_type is AssetType.ETF


def test_read_back_matches_source(session_factory, fixtures_dir):
    src = CsvSource(fixtures_dir)
    run_ingestion(session_factory, src, ["SYNB"], START, END)
    with session_scope(session_factory) as s:
        stored = PriceRepository(s).get_bars("SYNB")
    expected = src.fetch_daily("SYNB", START, END)
    pd.testing.assert_frame_equal(stored, expected, check_freq=False)


def test_large_upsert_is_chunked(session_factory):
    """More rows than one statement can carry (65k bind params) still works."""
    idx = pd.bdate_range("1990-01-01", periods=8000, name="date")
    df = pd.DataFrame(
        {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "adj_close": 10.5, "volume": 1.0},
        index=idx,
    )
    with session_scope(session_factory) as s:
        inst = InstrumentRepository(s).get_or_create("BIG")
        res = PriceRepository(s).upsert_bars(inst.id, df, "csv", None)
    assert res.inserted == 8000


def test_database_constraints_are_a_second_line_of_defence(session_factory):
    with pytest.raises(IntegrityError), session_scope(session_factory) as s:
        inst = InstrumentRepository(s).get_or_create("X")
        s.add(
            PriceBar(
                instrument_id=inst.id,
                bar_date=date(2024, 1, 2),
                open=10,
                high=9,
                low=8,
                close=10,
                adj_close=10,
                volume=1,
                source="csv",
            )
        )
        s.flush()


def test_coverage(session_factory, fixtures_dir):
    run_ingestion(session_factory, CsvSource(fixtures_dir), ["SYNA"], START, END)
    with session_scope(session_factory) as s:
        cov = InstrumentRepository(s).coverage()
    assert [(c.symbol, c.bars, c.first_date, c.last_date) for c in cov] == [
        ("SYNA", 260, date(2023, 1, 2), date(2023, 12, 29))
    ]
