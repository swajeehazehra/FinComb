"""Record which data source owns each symbol's price history.

Backfill: a symbol whose stored bars all come from one source is assigned that
source. A symbol with bars from several sources (e.g. AAPL with Yahoo history
and newer Tiingo rows) is left with NO owner on purpose: ingestion refuses to
touch it until `fincomb refresh` rebuilds its history from one source.

Revision ID: 0002_instrument_data_source
Revises: 0001_market_data
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_instrument_data_source"
down_revision: str | None = "0001_market_data"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("instruments", sa.Column("data_source", sa.String(20), nullable=True))
    op.execute(
        """
        UPDATE instruments AS i
        SET data_source = s.source
        FROM (
            SELECT instrument_id, MIN(source) AS source
            FROM price_bars
            GROUP BY instrument_id
            HAVING COUNT(DISTINCT source) = 1
        ) AS s
        WHERE s.instrument_id = i.id
        """
    )


def downgrade() -> None:
    op.drop_column("instruments", "data_source")
