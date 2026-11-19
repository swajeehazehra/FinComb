"""Portfolios and their weights (Milestone 4).

A portfolio stores its weights and the exact parameters that produced them, so it
can be reproduced later. Risk numbers are not stored: they are recomputed from
price_bars on request.

Revision ID: 0003_portfolios
Revises: 0002_instrument_data_source
Create Date: 2026-11-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_portfolios"
down_revision: str | None = "0002_instrument_data_source"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

portfolio_method = postgresql.ENUM(
    "manual",
    "equal_weight",
    "inverse_vol",
    "min_variance",
    "max_sharpe",
    name="portfolio_method",
    create_type=False,
)


def upgrade() -> None:
    portfolio_method.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "portfolios",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("description", sa.Text()),
        sa.Column("construction_method", portfolio_method, nullable=False),
        sa.Column("effective_method", portfolio_method, nullable=False),
        sa.Column(
            "construction_params",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("estimation_start", sa.Date()),
        sa.Column("estimation_end", sa.Date()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )

    op.create_table(
        "portfolio_weights",
        sa.Column(
            "portfolio_id",
            sa.Integer(),
            sa.ForeignKey("portfolios.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "instrument_id",
            sa.Integer(),
            sa.ForeignKey("instruments.id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.CheckConstraint("weight >= 0 AND weight <= 1", name="ck_portfolio_weights_long_only"),
    )


def downgrade() -> None:
    op.drop_table("portfolio_weights")
    op.drop_table("portfolios")
    portfolio_method.drop(op.get_bind(), checkfirst=True)
