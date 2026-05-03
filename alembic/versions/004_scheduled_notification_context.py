"""scheduled notification context

Revision ID: 004_sched_ctx
Revises: 003_notification_idempotency
Create Date: 2026-05-03

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "004_sched_ctx"
down_revision: str | None = "003_notification_idempotency"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "notification_rules",
        sa.Column("notification_context", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("notification_rules", "notification_context")
