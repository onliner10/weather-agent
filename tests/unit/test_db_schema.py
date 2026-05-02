from __future__ import annotations

import pytest
from sqlalchemy import Inspector, create_engine, inspect

from weather_agent.infrastructure.db.base import (
    AuditLog,
    AuthorizedUser,
    Base,
    ForecastPoint,
    ForecastSnapshot,
    GlobalSetting,
    Location,
    NotificationEvent,
    NotificationRule,
    Observation,
    OfficialWarning,
    RuleEvaluationRun,
    TelegramContext,
)

EXPECTED_TABLES = {
    "authorized_users",
    "locations",
    "global_settings",
    "telegram_contexts",
    "forecast_snapshots",
    "forecast_points",
    "observations",
    "official_warnings",
    "notification_rules",
    "notification_events",
    "rule_evaluation_runs",
    "audit_log",
}


@pytest.fixture()
def sync_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def inspector(sync_engine):
    return inspect(sync_engine)


class TestSchemaTables:
    def test_all_tables_created(self, inspector: Inspector) -> None:
        actual = set(inspector.get_table_names())
        assert actual == EXPECTED_TABLES

    def test_table_names_match_model_tablenames(self) -> None:
        model_tablenames = {cls.__tablename__ for cls in Base.__subclasses__()}
        assert model_tablenames == EXPECTED_TABLES


class TestAuthorizedUsers:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("authorized_users")}
        assert "id" in cols
        assert "telegram_user_id" in cols
        assert "role" in cols
        assert "created_at" in cols
        assert "updated_at" in cols

    def test_telegram_user_id_unique(self, inspector: Inspector) -> None:
        uniques = inspector.get_unique_constraints("authorized_users")
        col_sets = {frozenset(c["column_names"]) for c in uniques}
        assert frozenset({"telegram_user_id"}) in col_sets


class TestLocations:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("locations")}
        assert "id" in cols
        assert "user_id" in cols
        assert "name" in cols
        assert "aliases" in cols
        assert "latitude" in cols
        assert "longitude" in cols
        assert "description" in cols
        assert "enabled" in cols
        assert "created_at" in cols
        assert "updated_at" in cols

    def test_description_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("locations")}
        assert cols["description"]["nullable"] is True

    def test_enabled_not_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("locations")}
        assert cols["enabled"]["nullable"] is False

    def test_fk_to_authorized_users(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("locations")
        fk_targets = [(fk["constrained_columns"], fk["referred_table"]) for fk in fks]
        assert any("authorized_users" == t for _, t in fk_targets)


class TestGlobalSettings:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("global_settings")}
        assert "id" in cols
        assert "key" in cols
        assert "value" in cols
        assert "updated_at" in cols

    def test_key_unique(self, inspector: Inspector) -> None:
        uniques = inspector.get_unique_constraints("global_settings")
        col_sets = {frozenset(c["column_names"]) for c in uniques}
        assert frozenset({"key"}) in col_sets


class TestTelegramContexts:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("telegram_contexts")}
        assert "id" in cols
        assert "chat_id" in cols
        assert "message_thread_id" in cols
        assert "context_key" in cols
        assert "metadata" in cols
        assert "created_at" in cols
        assert "updated_at" in cols

    def test_message_thread_id_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("telegram_contexts")}
        assert cols["message_thread_id"]["nullable"] is True

    def test_context_key_unique(self, inspector: Inspector) -> None:
        uniques = inspector.get_unique_constraints("telegram_contexts")
        col_sets = {frozenset(c["column_names"]) for c in uniques}
        assert frozenset({"context_key"}) in col_sets


class TestForecastSnapshots:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("forecast_snapshots")}
        assert "id" in cols
        assert "provider" in cols
        assert "model" in cols
        assert "location_id" in cols
        assert "fetched_at" in cols
        assert "raw_payload" in cols

    def test_model_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("forecast_snapshots")}
        assert cols["model"]["nullable"] is True

    def test_fk_to_locations(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("forecast_snapshots")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "locations" in fk_tables


class TestForecastPoints:
    FORECAST_POINT_WEATHER_FIELDS = [
        "temperature_2m_c",
        "apparent_temperature_c",
        "precipitation_mm",
        "precipitation_probability_pct",
        "rain_mm",
        "snowfall_cm",
        "cloud_cover_pct",
        "wind_speed_10m_ms",
        "wind_gusts_10m_ms",
        "wind_direction_10m_deg",
        "pressure_msl_hpa",
        "relative_humidity_2m_pct",
        "weather_code",
    ]

    def test_all_columns_present(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("forecast_points")}
        expected = {
            "id",
            "snapshot_id",
            "target_time",
            "location_id",
            "raw_payload",
        } | set(self.FORECAST_POINT_WEATHER_FIELDS)
        assert expected <= cols

    def test_weather_fields_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("forecast_points")}
        for field in self.FORECAST_POINT_WEATHER_FIELDS:
            assert cols[field]["nullable"] is True, f"{field} should be nullable"

    def test_target_time_not_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("forecast_points")}
        assert cols["target_time"]["nullable"] is False

    def test_fks(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("forecast_points")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "forecast_snapshots" in fk_tables
        assert "locations" in fk_tables


class TestObservations:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("observations")}
        expected = {
            "id",
            "provider",
            "station_id",
            "station_name",
            "distance_km",
            "observed_at",
            "location_id",
            "fetched_at",
            "temperature_c",
            "wind_speed_ms",
            "wind_direction_deg",
            "pressure_hpa",
            "humidity_pct",
            "precipitation_mm",
            "raw_payload",
        }
        assert expected <= cols

    def test_nullable_fields(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("observations")}
        for nullable_field in [
            "station_id",
            "station_name",
            "distance_km",
            "temperature_c",
            "wind_speed_ms",
            "wind_direction_deg",
            "pressure_hpa",
            "humidity_pct",
            "precipitation_mm",
        ]:
            assert cols[nullable_field]["nullable"] is True, f"{nullable_field} should be nullable"

    def test_not_nullable_fields(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("observations")}
        not_null_fields = ["provider", "observed_at", "location_id", "fetched_at", "raw_payload"]
        for not_null_field in not_null_fields:
            msg = f"{not_null_field} should not be nullable"
            assert cols[not_null_field]["nullable"] is False, msg

    def test_fk_to_locations(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("observations")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "locations" in fk_tables


class TestOfficialWarnings:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("official_warnings")}
        expected = {
            "id",
            "provider",
            "external_id",
            "location_id",
            "severity",
            "category",
            "headline",
            "description",
            "valid_from",
            "valid_to",
            "raw_payload",
        }
        assert expected <= cols

    def test_severity_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("official_warnings")}
        assert cols["severity"]["nullable"] is True

    def test_unique_constraint(self, inspector: Inspector) -> None:
        uniques = inspector.get_unique_constraints("official_warnings")
        col_sets = {frozenset(c["column_names"]) for c in uniques}
        assert frozenset({"provider", "external_id", "valid_from"}) in col_sets

    def test_fk_to_locations(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("official_warnings")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "locations" in fk_tables


class TestNotificationRules:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("notification_rules")}
        expected = {
            "id",
            "short_id",
            "user_id",
            "telegram_chat_id",
            "telegram_message_thread_id",
            "location_id",
            "expression_language",
            "expression",
            "schedule",
            "lead_time_minutes",
            "cooldown_minutes",
            "enabled",
            "dry_run",
            "description",
            "created_at",
            "updated_at",
        }
        assert expected <= cols

    def test_short_id_unique(self, inspector: Inspector) -> None:
        uniques = inspector.get_unique_constraints("notification_rules")
        col_sets = {frozenset(c["column_names"]) for c in uniques}
        assert frozenset({"short_id"}) in col_sets

    def test_nullable_fields(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("notification_rules")}
        for nullable_field in [
            "telegram_message_thread_id",
            "schedule",
            "lead_time_minutes",
            "cooldown_minutes",
            "description",
        ]:
            assert cols[nullable_field]["nullable"] is True, f"{nullable_field} should be nullable"

    def test_fks(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("notification_rules")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "authorized_users" in fk_tables
        assert "locations" in fk_tables


class TestNotificationEvents:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("notification_events")}
        expected = {
            "id",
            "short_id",
            "rule_id",
            "evaluation_run_id",
            "telegram_chat_id",
            "telegram_message_thread_id",
            "sent_at",
            "suppressed",
            "suppress_reason",
            "payload_hash",
            "message_text",
            "delivery_status",
            "delivery_claimed_at",
            "created_at",
        }
        assert expected <= cols

    def test_short_id_unique(self, inspector: Inspector) -> None:
        uniques = inspector.get_unique_constraints("notification_events")
        col_sets = {frozenset(c["column_names"]) for c in uniques}
        assert frozenset({"short_id"}) in col_sets

    def test_nullable_fields(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("notification_events")}
        for nullable_field in [
            "evaluation_run_id",
            "telegram_message_thread_id",
            "sent_at",
            "suppress_reason",
            "payload_hash",
            "message_text",
            "delivery_claimed_at",
        ]:
            assert cols[nullable_field]["nullable"] is True, f"{nullable_field} should be nullable"

    def test_delivery_status_not_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("notification_events")}
        assert cols["delivery_status"]["nullable"] is False

    def test_fk_to_notification_rules(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("notification_events")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "notification_rules" in fk_tables


class TestRuleEvaluationRuns:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("rule_evaluation_runs")}
        expected = {
            "id",
            "rule_id",
            "snapshot_id",
            "evaluated_at",
            "result",
            "evaluation_detail",
            "created_at",
        }
        assert expected <= cols

    def test_snapshot_id_nullable(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("rule_evaluation_runs")}
        assert cols["snapshot_id"]["nullable"] is True

    def test_fks(self, inspector: Inspector) -> None:
        fks = inspector.get_foreign_keys("rule_evaluation_runs")
        fk_tables = {fk["referred_table"] for fk in fks}
        assert "notification_rules" in fk_tables
        assert "forecast_snapshots" in fk_tables


class TestAuditLog:
    def test_columns(self, inspector: Inspector) -> None:
        cols = {c["name"] for c in inspector.get_columns("audit_log")}
        expected = {"id", "event_type", "user_id", "context_key", "details", "created_at"}
        assert expected <= cols

    def test_nullable_fields(self, inspector: Inspector) -> None:
        cols = {c["name"]: c for c in inspector.get_columns("audit_log")}
        assert cols["user_id"]["nullable"] is True
        assert cols["context_key"]["nullable"] is True


class TestModelTablenames:
    @pytest.mark.parametrize(
        "model_class, expected_tablename",
        [
            (AuthorizedUser, "authorized_users"),
            (Location, "locations"),
            (GlobalSetting, "global_settings"),
            (TelegramContext, "telegram_contexts"),
            (ForecastSnapshot, "forecast_snapshots"),
            (ForecastPoint, "forecast_points"),
            (Observation, "observations"),
            (OfficialWarning, "official_warnings"),
            (NotificationRule, "notification_rules"),
            (NotificationEvent, "notification_events"),
            (RuleEvaluationRun, "rule_evaluation_runs"),
            (AuditLog, "audit_log"),
        ],
    )
    def test_tablename(self, model_class: type, expected_tablename: str) -> None:
        assert model_class.__tablename__ == expected_tablename
