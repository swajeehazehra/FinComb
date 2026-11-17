"""When is a daily bar final?

A US daily bar is only final after the market closes at 16:00 New York time,
and data vendors keep correcting it for a while after that. Before then, Yahoo
returns an *in-progress* bar whose "close" is just the latest price. To never
store such a bar, ingestion stops at the last *completed* session.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")  # needs the `tzdata` package on Windows
# Close is 16:00; wait two hours so vendors have published the final bar.
BAR_FINAL_AFTER = time(18, 0)


def last_completed_session(now: datetime | None = None) -> date:
    """The most recent weekday whose daily bar is final at `now`.

    Market holidays are not modelled: on a holiday the vendor simply returns no
    bar for that date, so nothing incomplete can be stored.
    """
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local = now.astimezone(NEW_YORK)
    day = local.date()
    if local.weekday() >= 5 or local.time() < BAR_FINAL_AFTER:
        day -= timedelta(days=1)
    while day.weekday() >= 5:  # Saturday/Sunday -> Friday
        day -= timedelta(days=1)
    return day
