"""Backtest runs, their daily equity curves and trade logs (Milestone 5).

Only successful runs are stored, with every input needed to rerun them, the git
commit of the code and a SHA-256 hash of the exact prices used.

Revision ID: 0004_backtests
Revises: 0003_portfolios
Create Date: 2026-11-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_backtests"
down_revision: str | None = "0003_portfolios"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("strategy", sa.String(50), nullable=False),
        sa.Column(
            "params", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("symbols", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("benchmark_symbol", sa.String(20)),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("requested_start", sa.Date()),
        sa.Column("requested_end", sa.Date()),
        sa.Column("warmup_start", sa.Date()),
        sa.Column("rebalance_frequency", sa.String(10), nullable=False),
        sa.Column("execution", sa.String(12), nullable=False),
        sa.Column("initial_capital", sa.Float(), nullable=False),
        sa.Column("cost_bps", sa.Float(), nullable=False),
        sa.Column("slippage_bps", sa.Float(), nullable=False),
        sa.Column("risk_free", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "portfolio_id", sa.Integer(), sa.ForeignKey("portfolios.id", ondelete="SET NULL")
        ),
        sa.Column("code_version", sa.String(40), nullable=False),
        sa.Column("data_hash", sa.String(64), nullable=False),
        sa.Column(
            "metrics", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column(
            "warnings", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column(
            "notes", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("duration_ms", sa.Integer()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("initial_capital > 0", name="ck_backtest_runs_capital"),
        sa.CheckConstraint("cost_bps >= 0 AND slippage_bps >= 0", name="ck_backtest_runs_costs"),
    )

    op.create_table(
        "backtest_equity",
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("backtest_runs.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("bar_date", sa.Date(), primary_key=True),
        sa.Column("equity", sa.Float(), nullable=False),
        sa.Column("cash", sa.Float(), nullable=False),
        sa.Column("gross_exposure", sa.Float(), nullable=False),
        sa.Column("benchmark_equity", sa.Float()),
        sa.Column("drawdown", sa.Float(), nullable=False),
        sa.Column("turnover", sa.Float(), nullable=False),
        sa.Column("costs", sa.Float(), nullable=False),
    )

    op.create_table(
        "backtest_trades",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("backtest_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("bar_date", sa.Date(), nullable=False),
        sa.Column(
            "instrument_id",
            sa.Integer(),
            sa.ForeignKey("instruments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("side", sa.String(4), nullable=False),
        sa.Column("weight_before", sa.Float(), nullable=False),
        sa.Column("weight_after", sa.Float(), nullable=False),
        sa.Column("quantity", sa.Float(), nullable=False),
        sa.Column("reference_price", sa.Float(), nullable=False),
        sa.Column("fill_price", sa.Float(), nullable=False),
        sa.Column("notional", sa.Float(), nullable=False),
        sa.Column("commission", sa.Float(), nullable=False),
        sa.Column("slippage_cost", sa.Float(), nullable=False),
    )
    op.create_index("ix_backtest_trades_run_date", "backtest_trades", ["run_id", "bar_date"])


def downgrade() -> None:
    op.drop_index("ix_backtest_trades_run_date", table_name="backtest_trades")
    op.drop_table("backtest_trades")
    op.drop_table("backtest_equity")
    op.drop_table("backtest_runs")
