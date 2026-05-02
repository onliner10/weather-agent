"""notification idempotency

Revision ID: 003_notification_idempotency
Revises: 7e7a759f9505
Create Date: 2026-05-02

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "003_notification_idempotency"
down_revision: str | None = "7e7a759f9505"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "notification_events",
        sa.Column(
            "delivery_status",
            sa.String(20),
            nullable=False,
            server_default="pending",
        ),
    )
    op.add_column(
        "notification_events",
        sa.Column("delivery_claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE notification_events SET delivery_status = "
        "CASE WHEN suppressed IS TRUE THEN 'suppressed' "
        "WHEN sent_at IS NOT NULL THEN 'sent' ELSE 'pending' END"
    )
    op.create_index(
        "uq_notification_events_rule_payload_active",
        "notification_events",
        ["rule_id", "payload_hash"],
        unique=True,
        postgresql_where=sa.text(
            "payload_hash IS NOT NULL AND delivery_status IN ('pending', 'sending', 'sent')"
        ),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_notification_events_rule_payload_active",
        table_name="notification_events",
    )
    op.drop_column("notification_events", "delivery_claimed_at")
    op.drop_column("notification_events", "delivery_status")
