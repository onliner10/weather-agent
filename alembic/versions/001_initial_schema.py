"""initial schema

Revision ID: 001_initial
Revises: None
Create Date: 2026-04-28

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "authorized_users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("telegram_user_id", sa.Integer(), nullable=False, unique=True),
        sa.Column("role", sa.String(50), nullable=False, server_default="user"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "locations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("authorized_users.id"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("aliases", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "global_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(200), nullable=False, unique=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "telegram_contexts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("chat_id", sa.Integer(), nullable=False),
        sa.Column("message_thread_id", sa.Integer(), nullable=True),
        sa.Column("context_key", sa.String(100), nullable=False, unique=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_telegram_contexts_chat_id", "telegram_contexts", ["chat_id"])

    op.create_table(
        "forecast_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
    )

    op.create_table(
        "forecast_points",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "snapshot_id", sa.Integer(), sa.ForeignKey("forecast_snapshots.id"), nullable=False
        ),
        sa.Column("target_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False),
        sa.Column("temperature_2m_c", sa.Float(), nullable=True),
        sa.Column("apparent_temperature_c", sa.Float(), nullable=True),
        sa.Column("precipitation_mm", sa.Float(), nullable=True),
        sa.Column("precipitation_probability_pct", sa.Float(), nullable=True),
        sa.Column("rain_mm", sa.Float(), nullable=True),
        sa.Column("snowfall_cm", sa.Float(), nullable=True),
        sa.Column("cloud_cover_pct", sa.Float(), nullable=True),
        sa.Column("wind_speed_10m_ms", sa.Float(), nullable=True),
        sa.Column("wind_gusts_10m_ms", sa.Float(), nullable=True),
        sa.Column("wind_direction_10m_deg", sa.Float(), nullable=True),
        sa.Column("pressure_msl_hpa", sa.Float(), nullable=True),
        sa.Column("relative_humidity_2m_pct", sa.Float(), nullable=True),
        sa.Column("weather_code", sa.String(50), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_forecast_points_location_target", "forecast_points", ["location_id", "target_time"]
    )
    op.create_index("ix_forecast_points_snapshot", "forecast_points", ["snapshot_id"])

    op.create_table(
        "observations",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("station_id", sa.String(100), nullable=True),
        sa.Column("station_name", sa.String(200), nullable=True),
        sa.Column("distance_km", sa.Float(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("wind_speed_ms", sa.Float(), nullable=True),
        sa.Column("wind_direction_deg", sa.Float(), nullable=True),
        sa.Column("pressure_hpa", sa.Float(), nullable=True),
        sa.Column("humidity_pct", sa.Float(), nullable=True),
        sa.Column("precipitation_mm", sa.Float(), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_observations_location_observed", "observations", ["location_id", "observed_at"]
    )
    op.create_index(
        "ix_observations_provider_station_observed",
        "observations",
        ["provider", "station_id", "observed_at"],
    )

    op.create_table(
        "official_warnings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("provider", sa.String(100), nullable=False),
        sa.Column("external_id", sa.String(200), nullable=False),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False),
        sa.Column("severity", sa.String(50), nullable=True),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("headline", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "provider", "external_id", "valid_from", name="uq_official_warnings_provider_ext_valid"
        ),
    )
    op.create_index(
        "ix_official_warnings_provider_ext_valid",
        "official_warnings",
        ["provider", "external_id", "valid_from"],
    )

    op.create_table(
        "notification_rules",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("short_id", sa.String(10), nullable=False, unique=True),
        sa.Column(
            "user_id", sa.Integer(), sa.ForeignKey("authorized_users.id"), nullable=False
        ),
        sa.Column("telegram_chat_id", sa.Integer(), nullable=False),
        sa.Column("telegram_message_thread_id", sa.Integer(), nullable=True),
        sa.Column("location_id", sa.Integer(), sa.ForeignKey("locations.id"), nullable=False),
        sa.Column(
            "expression_language", sa.String(20), nullable=False, server_default="cel"
        ),
        sa.Column("expression", sa.Text(), nullable=False),
        sa.Column("schedule", sa.String(200), nullable=True),
        sa.Column("lead_time_minutes", sa.Integer(), nullable=True),
        sa.Column("cooldown_minutes", sa.Integer(), nullable=True, server_default="60"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("dry_run", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_notification_rules_user_enabled", "notification_rules", ["user_id", "enabled"]
    )

    op.create_table(
        "rule_evaluation_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "rule_id", sa.Integer(), sa.ForeignKey("notification_rules.id"), nullable=False
        ),
        sa.Column(
            "snapshot_id",
            sa.Integer(),
            sa.ForeignKey("forecast_snapshots.id"),
            nullable=True,
        ),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("result", sa.Boolean(), nullable=False),
        sa.Column("evaluation_detail", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "notification_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "short_id", sa.String(10), nullable=False, unique=True
        ),
        sa.Column(
            "rule_id",
            sa.Integer(),
            sa.ForeignKey("notification_rules.id"),
            nullable=False,
        ),
        sa.Column(
            "evaluation_run_id",
            sa.Integer(),
            sa.ForeignKey("rule_evaluation_runs.id"),
            nullable=True,
        ),
        sa.Column("telegram_chat_id", sa.Integer(), nullable=False),
        sa.Column("telegram_message_thread_id", sa.Integer(), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "suppressed", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("suppress_reason", sa.Text(), nullable=True),
        sa.Column("payload_hash", sa.String(64), nullable=True),
        sa.Column("message_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("context_key", sa.String(100), nullable=True),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_audit_log_event_created", "audit_log", ["event_type", "created_at"])

    # TimescaleDB hypertable creation directives.
    # These require the timescaledb extension to be enabled in PostgreSQL.
    # Run these statements only on a TimescaleDB-enabled database:
    #
    # CREATE EXTENSION IF NOT EXISTS timescaledb;
    # SELECT create_hypertable('forecast_points', 'target_time');
    # SELECT create_hypertable('observations', 'observed_at');
    # SELECT create_hypertable('rule_evaluation_runs', 'evaluated_at');


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("notification_events")
    op.drop_table("rule_evaluation_runs")
    op.drop_table("notification_rules")
    op.drop_table("official_warnings")
    op.drop_table("observations")
    op.drop_table("forecast_points")
    op.drop_table("forecast_snapshots")
    op.drop_table("telegram_contexts")
    op.drop_table("global_settings")
    op.drop_table("locations")
    op.drop_table("authorized_users")