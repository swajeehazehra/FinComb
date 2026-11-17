from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from fincomb.ingestion.calendar import last_completed_session

NY = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    "now_ny,expected",
    [
        (datetime(2026, 11, 16, 10, 0), date(2026, 11, 13)),  # Mon, market open -> Fri
        (datetime(2026, 11, 16, 16, 30), date(2026, 11, 13)),  # Mon, just closed, not final
        (datetime(2026, 11, 16, 18, 0), date(2026, 11, 16)),  # Mon evening -> Mon
        (datetime(2026, 11, 17, 3, 0), date(2026, 11, 16)),  # Tue before open -> Mon
        (datetime(2026, 11, 14, 12, 0), date(2026, 11, 13)),  # Saturday -> Fri
        (datetime(2026, 11, 15, 20, 0), date(2026, 11, 13)),  # Sunday evening -> Fri
    ],
)
def test_last_completed_session(now_ny, expected):
    assert last_completed_session(now_ny.replace(tzinfo=NY)) == expected


def test_converts_from_other_time_zones():
    # 02:40 UTC on Tue 17 Nov 2026 is 21:40 on Mon 16 Nov in New York.
    assert last_completed_session(datetime(2026, 11, 17, 2, 40, tzinfo=UTC)) == date(2026, 11, 16)
    # 15:00 UTC on Mon is 10:00 in New York: Monday's bar is still in progress.
    assert last_completed_session(datetime(2026, 11, 16, 15, 0, tzinfo=UTC)) == date(2026, 11, 13)


def test_naive_datetime_rejected():
    with pytest.raises(ValueError, match="timezone"):
        last_completed_session(datetime(2026, 11, 16, 12, 0))
