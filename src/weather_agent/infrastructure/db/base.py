from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    ForeignKey,
    Index,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB as PG_JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeEngine


class JSONBVariant(TypeDecorator[dict[str, object]]):
    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: object) -> TypeEngine[dict[str, object]]:
        if getattr(dialect, "name", None) == "postgresql":
            return PG_JSONB()
        return JSON()


class Base(DeclarativeBase):
    type_annotation_map = {
        dict[str, object]: JSONBVariant(),
    }


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AuthorizedUser(Base):
    __tablename__ = "authorized_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_user_id: Mapped[int] = mapped_column(unique=True)
    role: Mapped[str] = mapped_column(String(50), default="user")
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow
    )

    locations: Mapped[list[Location]] = relationship(back_populates="user")
    notification_rules: Mapped[list[NotificationRule]] = relationship(
        back_populates="user"
    )


class Location(Base):
    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("authorized_users.id"))
    name: Mapped[str] = mapped_column(String(200))
    aliases: Mapped[list[str]] = mapped_column(JSONBVariant, default=list)
    latitude: Mapped[float] = mapped_column()
    longitude: Mapped[float] = mapped_column()
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[AuthorizedUser] = relationship(back_populates="locations")
    forecast_snapshots: Mapped[list[ForecastSnapshot]] = relationship(
        back_populates="location"
    )
    forecast_points: Mapped[list[ForecastPoint]] = relationship(
        back_populates="location"
    )
    observations: Mapped[list[Observation]] = relationship(
        back_populates="location"
    )
    official_warnings: Mapped[list[OfficialWarning]] = relationship(
        back_populates="location"
    )
    notification_rules: Mapped[list[NotificationRule]] = relationship(
        back_populates="location"
    )


class GlobalSetting(Base):
    __tablename__ = "global_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(200), unique=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow
    )


class TelegramContext(Base):
    __tablename__ = "telegram_contexts"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column()
    message_thread_id: Mapped[int | None] = mapped_column(nullable=True)
    context_key: Mapped[str] = mapped_column(String(100), unique=True)
    metadata_: Mapped[dict[str, object]] = mapped_column("metadata", default=dict)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow
    )

    __table_args__ = (
        Index("ix_telegram_contexts_chat_id", "chat_id"),
    )


class ForecastSnapshot(Base):
    __tablename__ = "forecast_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(100))
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    fetched_at: Mapped[datetime] = mapped_column()
    raw_payload: Mapped[dict[str, object]] = mapped_column()

    location: Mapped[Location] = relationship(
        back_populates="forecast_snapshots"
    )
    forecast_points: Mapped[list[ForecastPoint]] = relationship(
        back_populates="snapshot"
    )
    rule_evaluation_runs: Mapped[list[RuleEvaluationRun]] = relationship(
        back_populates="snapshot"
    )


class ForecastPoint(Base):
    __tablename__ = "forecast_points"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("forecast_snapshots.id")
    )
    target_time: Mapped[datetime] = mapped_column()
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))

    temperature_2m_c: Mapped[float | None] = mapped_column(nullable=True)
    apparent_temperature_c: Mapped[float | None] = mapped_column(nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(nullable=True)
    precipitation_probability_pct: Mapped[float | None] = mapped_column(
        nullable=True
    )
    rain_mm: Mapped[float | None] = mapped_column(nullable=True)
    snowfall_cm: Mapped[float | None] = mapped_column(nullable=True)
    cloud_cover_pct: Mapped[float | None] = mapped_column(nullable=True)
    wind_speed_10m_ms: Mapped[float | None] = mapped_column(nullable=True)
    wind_gusts_10m_ms: Mapped[float | None] = mapped_column(nullable=True)
    wind_direction_10m_deg: Mapped[float | None] = mapped_column(nullable=True)
    pressure_msl_hpa: Mapped[float | None] = mapped_column(nullable=True)
    relative_humidity_2m_pct: Mapped[float | None] = mapped_column(
        nullable=True
    )
    weather_code: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )
    raw_payload: Mapped[dict[str, object]] = mapped_column()

    snapshot: Mapped[ForecastSnapshot] = relationship(
        back_populates="forecast_points"
    )
    location: Mapped[Location] = relationship(back_populates="forecast_points")

    __table_args__ = (
        Index("ix_forecast_points_location_target", "location_id", "target_time"),
        Index("ix_forecast_points_snapshot", "snapshot_id"),
    )


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(100))
    station_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    station_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )
    distance_km: Mapped[float | None] = mapped_column(nullable=True)
    observed_at: Mapped[datetime] = mapped_column()
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    fetched_at: Mapped[datetime] = mapped_column()
    temperature_c: Mapped[float | None] = mapped_column(nullable=True)
    wind_speed_ms: Mapped[float | None] = mapped_column(nullable=True)
    wind_direction_deg: Mapped[float | None] = mapped_column(nullable=True)
    pressure_hpa: Mapped[float | None] = mapped_column(nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(nullable=True)
    precipitation_mm: Mapped[float | None] = mapped_column(nullable=True)
    raw_payload: Mapped[dict[str, object]] = mapped_column()

    location: Mapped[Location] = relationship(back_populates="observations")

    __table_args__ = (
        Index(
            "ix_observations_location_observed", "location_id", "observed_at"
        ),
        Index(
            "ix_observations_provider_station_observed",
            "provider",
            "station_id",
            "observed_at",
        ),
    )


class OfficialWarning(Base):
    __tablename__ = "official_warnings"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(100))
    external_id: Mapped[str] = mapped_column(String(200))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    severity: Mapped[str | None] = mapped_column(String(50), nullable=True)
    category: Mapped[str] = mapped_column(String(50))
    headline: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text)
    valid_from: Mapped[datetime] = mapped_column()
    valid_to: Mapped[datetime] = mapped_column()
    raw_payload: Mapped[dict[str, object]] = mapped_column()

    location: Mapped[Location] = relationship(
        back_populates="official_warnings"
    )

    __table_args__ = (
        UniqueConstraint(
            "provider",
            "external_id",
            "valid_from",
            name="uq_official_warnings_provider_ext_valid",
        ),
        Index(
            "ix_official_warnings_provider_ext_valid",
            "provider",
            "external_id",
            "valid_from",
        ),
    )


class NotificationRule(Base):
    __tablename__ = "notification_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    short_id: Mapped[str] = mapped_column(String(10), unique=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("authorized_users.id"))
    telegram_chat_id: Mapped[int] = mapped_column()
    telegram_message_thread_id: Mapped[int | None] = mapped_column(
        nullable=True
    )
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    expression_language: Mapped[str] = mapped_column(
        String(20), default="cel"
    )
    expression: Mapped[str] = mapped_column(Text)
    schedule: Mapped[str | None] = mapped_column(String(200), nullable=True)
    lead_time_minutes: Mapped[int | None] = mapped_column(nullable=True)
    cooldown_minutes: Mapped[int | None] = mapped_column(
        nullable=True, default=60
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    dry_run: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[AuthorizedUser] = relationship(
        back_populates="notification_rules"
    )
    location: Mapped[Location] = relationship(
        back_populates="notification_rules"
    )
    notification_events: Mapped[list[NotificationEvent]] = relationship(
        back_populates="rule"
    )
    rule_evaluation_runs: Mapped[list[RuleEvaluationRun]] = relationship(
        back_populates="rule"
    )

    __table_args__ = (
        Index("ix_notification_rules_user_enabled", "user_id", "enabled"),
    )


class NotificationEvent(Base):
    __tablename__ = "notification_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    short_id: Mapped[str] = mapped_column(String(10), unique=True)
    rule_id: Mapped[int] = mapped_column(
        ForeignKey("notification_rules.id")
    )
    evaluation_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("rule_evaluation_runs.id"), nullable=True
    )
    telegram_chat_id: Mapped[int] = mapped_column()
    telegram_message_thread_id: Mapped[int | None] = mapped_column(
        nullable=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(nullable=True)
    suppressed: Mapped[bool] = mapped_column(Boolean, default=False)
    suppress_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    message_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    rule: Mapped[NotificationRule] = relationship(
        back_populates="notification_events"
    )


class RuleEvaluationRun(Base):
    __tablename__ = "rule_evaluation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    rule_id: Mapped[int] = mapped_column(
        ForeignKey("notification_rules.id")
    )
    snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("forecast_snapshots.id"), nullable=True
    )
    evaluated_at: Mapped[datetime] = mapped_column()
    result: Mapped[bool] = mapped_column()
    evaluation_detail: Mapped[dict[str, object]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    rule: Mapped[NotificationRule] = relationship(
        back_populates="rule_evaluation_runs"
    )
    snapshot: Mapped[ForecastSnapshot | None] = relationship(
        back_populates="rule_evaluation_runs"
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(100))
    user_id: Mapped[int | None] = mapped_column(nullable=True)
    context_key: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )
    details: Mapped[dict[str, object]] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=_utcnow)

    __table_args__ = (
        Index("ix_audit_log_event_created", "event_type", "created_at"),
    )