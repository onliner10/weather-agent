"""add snooze_until and make event rule_id nullable with SET NULL

Revision ID: 002_snooze_until
Revises: 001_initial
Create Date: 2026-04-28

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "002_snooze_until"
down_revision: str | None = "001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "notification_rules",
        sa.Column("snooze_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.alter_column(
        "notification_events",
        "rule_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.drop_constraint(
        "notification_events_rule_id_fkey",
        "notification_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "notification_events_rule_id_fkey",
        "notification_events",
        "notification_rules",
        ["rule_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "notification_events_rule_id_fkey",
        "notification_events",
        type_="foreignkey",
    )
    op.create_foreign_key(
        "notification_events_rule_id_fkey",
        "notification_events",
        "notification_rules",
        ["rule_id"],
        ["id"],
    )
    op.alter_column(
        "notification_events",
        "rule_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.drop_column("notification_rules", "snooze_until")
