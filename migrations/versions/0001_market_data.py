"""Market data tables: instruments, price_bars, ingestion_runs, data_quality_issues.

Revision ID: 0001_market_data
Revises:
Create Date: 2026-11-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_market_data"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

asset_type = postgresql.ENUM("stock", "etf", "index", name="asset_type", create_type=False)
run_status = postgresql.ENUM(
    "running", "success", "partial", "failed", name="run_status", create_type=False
)
issue_severity = postgresql.ENUM("warning", "error", name="issue_severity", create_type=False)


def upgrade() -> None:
    bind = op.get_bind()
    for enum in (asset_type, run_status, issue_severity):
        enum.create(bind, checkfirst=True)

    op.create_table(
        "instruments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False, unique=True),
        sa.Column("name", sa.String(200)),
        sa.Column("asset_type", asset_type, nullable=False),
        sa.Column("exchange", sa.String(20)),
        sa.Column("currency", sa.String(3), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column("symbols", postgresql.ARRAY(sa.String(20)), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column(
            "started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", run_status, nullable=False),
        sa.Column("rows_inserted", sa.Integer(), nullable=False),
        sa.Column("rows_updated", sa.Integer(), nullable=False),
        sa.Column("rows_rejected", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text()),
    )

    op.create_table(
        "price_bars",
        sa.Column(
            "instrument_id",
            sa.Integer(),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("bar_date", sa.Date(), primary_key=True),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("adj_close", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger()),
        sa.Column("source", sa.String(20), nullable=False),
        sa.Column(
            "ingested_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "ingestion_run_id",
            sa.Integer(),
            sa.ForeignKey("ingestion_runs.id", ondelete="SET NULL"),
        ),
        sa.CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0 AND adj_close > 0",
            name="ck_price_bars_positive",
        ),
        sa.CheckConstraint(
            "high >= low AND high >= open AND high >= close AND low <= open AND low <= close",
            name="ck_price_bars_ohlc",
        ),
        sa.CheckConstraint("volume IS NULL OR volume >= 0", name="ck_price_bars_volume"),
    )

    op.create_table(
        "data_quality_issues",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "ingestion_run_id",
            sa.Integer(),
            sa.ForeignKey("ingestion_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "instrument_id", sa.Integer(), sa.ForeignKey("instruments.id", ondelete="CASCADE")
        ),
        sa.Column("bar_date", sa.Date()),
        sa.Column("rule", sa.String(40), nullable=False),
        sa.Column("severity", issue_severity, nullable=False),
        sa.Column("detail", sa.Text()),
    )
    op.create_index(
        "ix_data_quality_issues_ingestion_run_id", "data_quality_issues", ["ingestion_run_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_data_quality_issues_ingestion_run_id", table_name="data_quality_issues")
    op.drop_table("data_quality_issues")
    op.drop_table("price_bars")
    op.drop_table("ingestion_runs")
    op.drop_table("instruments")
    bind = op.get_bind()
    for enum in (issue_severity, run_status, asset_type):
        enum.drop(bind, checkfirst=True)
