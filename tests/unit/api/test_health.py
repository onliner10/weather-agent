from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from weather_agent import __version__
from weather_agent.api.health import ComponentHealth, HealthStatus, create_health_app


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
            migrations_current=True,
            readiness="ready",
            components={
                "db": ComponentHealth(status="healthy", checked_at=now),
            },
            timestamp=now,
        )
        assert status.status == "healthy"
        assert status.version == "0.1.0"
        assert status.db_connected is True
        assert status.last_forecast_fetch == now
        assert status.last_rule_evaluation == now
        assert status.scheduler_status == "running"
        assert status.langsmith_enabled is True
        assert status.migrations_current is True
        assert status.readiness == "ready"

    def test_health_status_degraded(self) -> None:
        status = HealthStatus(
            status="degraded",
            version="0.1.0",
            db_connected=False,
            last_forecast_fetch=None,
            last_rule_evaluation=None,
            scheduler_status="stopped",
            langsmith_enabled=False,
            migrations_current=False,
            readiness="not_ready",
            components={
                "db": ComponentHealth(
                    status="degraded",
                    detail="database connection failed",
                    checked_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
                ),
            },
            timestamp=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
        )
        assert status.status == "degraded"
        assert status.db_connected is False
        assert status.last_forecast_fetch is None
        assert status.last_rule_evaluation is None
        assert status.scheduler_status == "stopped"
        assert status.migrations_current is False
        assert status.readiness == "not_ready"


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
        assert "migrations_current" in data
        assert "readiness" in data
        assert "components" in data
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
        assert data["readiness"] == "not_ready"
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
        assert data["components"]["db"]["status"] == "healthy"

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
    async def test_livez_is_process_only(self) -> None:
        app = create_health_app(session_factory=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/livez")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    @pytest.mark.asyncio()
    async def test_readyz_returns_503_when_not_ready(self) -> None:
        app = create_health_app(session_factory=None)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["readiness"] == "not_ready"

    @pytest.mark.asyncio()
    async def test_metrics_endpoint_returns_prometheus_text(self) -> None:
        app = create_health_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "process_" in response.text or "python_" in response.text
