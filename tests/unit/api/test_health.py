from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from weather_agent import __version__
from weather_agent.adapters.telegram.commands import (
    CommandContext,
    SystemStatus,
    handle_status,
)
from weather_agent.api.health import HealthStatus, create_health_app
from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules import NotificationRuleService
from weather_agent.domain.rules.models import NotificationRule


def _make_rule(
    id: int = 1,
    short_id: str = "R7K2",
    user_id: int = 42,
    enabled: bool = True,
    dry_run: bool = False,
    expression: str = "temp > 30",
    description: str | None = None,
) -> NotificationRule:
    return NotificationRule(
        id=id,
        short_id=short_id,
        user_id=user_id,
        telegram_chat_id=100,
        telegram_message_thread_id=None,
        location_id=1,
        expression_language="cel",
        expression=expression,
        schedule=None,
        lead_time_minutes=None,
        cooldown_minutes=60,
        enabled=enabled,
        dry_run=dry_run,
        description=description,
        snooze_until=None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestHealthStatusModel:
    def test_health_status_fields(self) -> None:
        now = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        status = HealthStatus(
            status="healthy",
            version="0.1.0",
            db_connected=True,
            last_forecast_fetch=now,
            last_rule_evaluation=now,
            scheduler_status="running",
            langsmith_enabled=True,
            timestamp=now,
        )
        assert status.status == "healthy"
        assert status.version == "0.1.0"
        assert status.db_connected is True
        assert status.last_forecast_fetch == now
        assert status.last_rule_evaluation == now
        assert status.scheduler_status == "running"
        assert status.langsmith_enabled is True

    def test_health_status_degraded(self) -> None:
        status = HealthStatus(
            status="degraded",
            version="0.1.0",
            db_connected=False,
            last_forecast_fetch=None,
            last_rule_evaluation=None,
            scheduler_status="stopped",
            langsmith_enabled=False,
            timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        assert status.status == "degraded"
        assert status.db_connected is False
        assert status.last_forecast_fetch is None
        assert status.last_rule_evaluation is None
        assert status.scheduler_status == "stopped"


class TestCreateHealthApp:
    @pytest.mark.asyncio()
    async def test_health_endpoint_returns_structure(self) -> None:
        app = create_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "db_connected" in data
        assert "last_forecast_fetch" in data
        assert "last_rule_evaluation" in data
        assert "scheduler_status" in data
        assert "langsmith_enabled" in data
        assert "timestamp" in data

    @pytest.mark.asyncio()
    async def test_health_endpoint_no_db_shows_degraded(self) -> None:
        app = create_health_app(session_factory=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["db_connected"] is False
        assert data["scheduler_status"] == "stopped"
        assert data["last_forecast_fetch"] is None
        assert data["last_rule_evaluation"] is None

    @pytest.mark.asyncio()
    async def test_health_endpoint_reports_version(self) -> None:
        app = create_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        data = response.json()
        assert data["version"] == __version__

    @pytest.mark.asyncio()
    async def test_health_endpoint_with_mock_db_connected(self) -> None:
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
        )

        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock(spec=async_sessionmaker)
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        app = create_health_app(session_factory=mock_factory)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["db_connected"] is True

    @pytest.mark.asyncio()
    async def test_health_endpoint_with_db_failure(self) -> None:
        from sqlalchemy.ext.asyncio import (
            AsyncSession,
            async_sessionmaker,
        )

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute = AsyncMock(side_effect=Exception("connection refused"))

        mock_factory = MagicMock(spec=async_sessionmaker)
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        app = create_health_app(session_factory=mock_factory)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["db_connected"] is False

    @pytest.mark.asyncio()
    async def test_metrics_endpoint_returns_prometheus_text(self) -> None:
        app = create_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "process_" in response.text or "python_" in response.text


class TestTelegramStatusCommand:
    @pytest.mark.asyncio()
    async def test_status_with_system_info(self) -> None:
        from weather_agent.domain.locations import Location

        loc_svc = MagicMock(spec=LocationService)
        loc_svc.list_locations = AsyncMock(
            return_value=[
                Location(
                    id=1,
                    name="Warszawa",
                    aliases=[],
                    latitude=52.2297,
                    longitude=21.0122,
                    description=None,
                    enabled=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    updated_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ]
        )
        rule_svc = MagicMock(spec=NotificationRuleService)
        rule_svc.list_rules = AsyncMock(
            return_value=[
                _make_rule(short_id="R7K2", enabled=True),
            ]
        )

        sys_status = SystemStatus(
            db_connected=True,
            scheduler_status="running",
            last_forecast_fetch=datetime(2026, 4, 28, 10, 0, tzinfo=UTC),
            last_rule_evaluation=datetime(2026, 4, 28, 10, 15, tzinfo=UTC),
            provider_status={"Open-Meteo": "ok", "IMGW": "ok"},
        )

        ctx = CommandContext(
            user_id=42,
            chat_id=100,
            message_thread_id=None,
            location_service=loc_svc,
            rule_service=rule_svc,
            auth_service=MagicMock(spec=AuthorizationService),
            system_status=sys_status,
        )
        result = await handle_status(ctx)
        assert "Baza danych: ✅" in result
        assert "Scheduler: running" in result
        assert "Ostatni pobór prognozy: 2026-04-28 10:00 UTC" in result
        assert "Ostatnia ewaluacja reguł: 2026-04-28 10:15 UTC" in result
        assert "Open-Meteo" in result
        assert "IMGW" in result
        assert f"Wersja: {__version__}" in result

    @pytest.mark.asyncio()
    async def test_status_with_db_down(self) -> None:
        loc_svc = MagicMock(spec=LocationService)
        loc_svc.list_locations = AsyncMock(return_value=[])
        rule_svc = MagicMock(spec=NotificationRuleService)
        rule_svc.list_rules = AsyncMock(return_value=[])

        sys_status = SystemStatus(
            db_connected=False,
            scheduler_status="stopped",
            last_forecast_fetch=None,
            last_rule_evaluation=None,
        )

        ctx = CommandContext(
            user_id=42,
            chat_id=100,
            message_thread_id=None,
            location_service=loc_svc,
            rule_service=rule_svc,
            auth_service=MagicMock(spec=AuthorizationService),
            system_status=sys_status,
        )
        result = await handle_status(ctx)
        assert "Baza danych: ❌" in result
        assert "Scheduler: stopped" in result
        assert "Ostatni pobór prognozy: brak" in result
        assert "Ostatnia ewaluacja reguł: brak" in result
        assert "Problemy z bazą danych" in result

    @pytest.mark.asyncio()
    async def test_status_without_system_status_fallback(self) -> None:
        loc_svc = MagicMock(spec=LocationService)
        loc_svc.list_locations = AsyncMock(return_value=[])
        rule_svc = MagicMock(spec=NotificationRuleService)
        rule_svc.list_rules = AsyncMock(
            return_value=[
                _make_rule(short_id="R7K2", enabled=True, dry_run=True),
            ]
        )

        ctx = CommandContext(
            user_id=42,
            chat_id=100,
            message_thread_id=None,
            location_service=loc_svc,
            rule_service=rule_svc,
            auth_service=MagicMock(spec=AuthorizationService),
            system_status=None,
        )
        result = await handle_status(ctx)
        assert "Dry-run: 1" in result
        assert "Bot działa poprawnie" in result

    @pytest.mark.asyncio()
    async def test_status_langsmith_enabled(self) -> None:
        loc_svc = MagicMock(spec=LocationService)
        loc_svc.list_locations = AsyncMock(return_value=[])
        rule_svc = MagicMock(spec=NotificationRuleService)
        rule_svc.list_rules = AsyncMock(return_value=[])

        sys_status = SystemStatus(db_connected=True, scheduler_status="running")

        ctx = CommandContext(
            user_id=42,
            chat_id=100,
            message_thread_id=None,
            location_service=loc_svc,
            rule_service=rule_svc,
            auth_service=MagicMock(spec=AuthorizationService),
            system_status=sys_status,
        )

        patch_path = "weather_agent.adapters.telegram.commands.LangSmithTracing.is_enabled"
        with patch(patch_path, return_value=True):
            result = await handle_status(ctx)
        assert "LangSmith: włączony" in result

    @pytest.mark.asyncio()
    async def test_status_langsmith_disabled(self) -> None:
        loc_svc = MagicMock(spec=LocationService)
        loc_svc.list_locations = AsyncMock(return_value=[])
        rule_svc = MagicMock(spec=NotificationRuleService)
        rule_svc.list_rules = AsyncMock(return_value=[])

        sys_status = SystemStatus(db_connected=True, scheduler_status="running")

        ctx = CommandContext(
            user_id=42,
            chat_id=100,
            message_thread_id=None,
            location_service=loc_svc,
            rule_service=rule_svc,
            auth_service=MagicMock(spec=AuthorizationService),
            system_status=sys_status,
        )

        patch_path = "weather_agent.adapters.telegram.commands.LangSmithTracing.is_enabled"
        with patch(patch_path, return_value=False):
            result = await handle_status(ctx)
        assert "LangSmith: wyłączony" in result
