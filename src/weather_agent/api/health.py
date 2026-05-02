from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from weather_agent import __version__
from weather_agent.infrastructure.db.base import ForecastSnapshot, RuleEvaluationRun
from weather_agent.infrastructure.db.setup import get_migration_heads
from weather_agent.observability.langsmith_tracing import LangSmithTracing
from weather_agent.settings import HealthSettings, ModelSettings, SchedulerSettings


class ComponentHealth(BaseModel):
    status: str
    detail: str | None = None
    checked_at: datetime


class HealthStatus(BaseModel):
    status: str
    version: str
    db_connected: bool
    last_forecast_fetch: datetime | None
    last_rule_evaluation: datetime | None
    scheduler_status: str
    langsmith_enabled: bool
    migrations_current: bool
    readiness: str
    components: dict[str, ComponentHealth]
    timestamp: datetime


async def _check_db(session: AsyncSession) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _get_last_forecast_fetch(session: AsyncSession) -> datetime | None:
    from sqlalchemy import select

    stmt = select(ForecastSnapshot.fetched_at).order_by(ForecastSnapshot.fetched_at.desc()).limit(1)
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    return row


async def _get_last_rule_evaluation(session: AsyncSession) -> datetime | None:
    from sqlalchemy import select

    stmt = (
        select(RuleEvaluationRun.evaluated_at)
        .order_by(RuleEvaluationRun.evaluated_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    row = result.scalar_one_or_none()
    return row


async def _get_database_revisions(session: AsyncSession) -> set[str]:
    try:
        result = await session.execute(text("SELECT version_num FROM alembic_version"))
    except Exception:
        return set()
    return {str(row[0]) for row in result.all()}


def _freshness_status(
    latest: datetime | None,
    threshold: timedelta,
    now: datetime,
    missing_status: str = "degraded",
) -> tuple[str, str | None]:
    if latest is None:
        return missing_status, "no successful run recorded"
    latest_utc = latest if latest.tzinfo is not None else latest.replace(tzinfo=UTC)
    age = now - latest_utc.astimezone(UTC)
    if age > threshold:
        return "degraded", f"stale for {int(age.total_seconds())}s"
    return "healthy", None


def create_health_app(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    health_settings: HealthSettings | None = None,
    scheduler_settings: SchedulerSettings | None = None,
    model_settings: ModelSettings | None = None,
    role: str = "process",
) -> FastAPI:
    app = FastAPI(title="Weather Agent Health", version=__version__)
    effective_health_settings = health_settings or HealthSettings()
    effective_scheduler_settings = scheduler_settings or SchedulerSettings()

    async def build_status() -> HealthStatus:
        now = datetime.now(UTC)
        components: dict[str, ComponentHealth] = {}

        if session_factory is not None:
            async with session_factory() as session:
                db_connected = await _check_db(session)
                last_forecast = await _get_last_forecast_fetch(session) if db_connected else None
                last_eval = await _get_last_rule_evaluation(session) if db_connected else None
                db_revisions = await _get_database_revisions(session) if db_connected else set()
        else:
            db_connected = False
            last_forecast = None
            last_eval = None
            db_revisions = set()

        langsmith_enabled = LangSmithTracing.is_enabled()
        migration_heads = get_migration_heads()
        migrations_current = bool(db_revisions) and db_revisions == migration_heads
        scheduler_status = "running" if db_connected and role == "worker" else "stopped"

        components["db"] = ComponentHealth(
            status="healthy" if db_connected else "degraded",
            detail=None if db_connected else "database connection failed",
            checked_at=now,
        )
        components["migrations"] = ComponentHealth(
            status="healthy" if migrations_current else "degraded",
            detail=None if migrations_current else "database revision is not at alembic head",
            checked_at=now,
        )

        worker_threshold = timedelta(
            minutes=max(
                effective_health_settings.worker_stale_after_minutes,
                effective_scheduler_settings.rule_evaluation_minutes * 2,
            )
        )
        worker_status, worker_detail = _freshness_status(
            last_eval,
            worker_threshold,
            now,
            missing_status="degraded" if role == "worker" else "unknown",
        )
        components["worker_freshness"] = ComponentHealth(
            status=worker_status,
            detail=worker_detail,
            checked_at=now,
        )

        forecast_status, forecast_detail = _freshness_status(
            last_forecast,
            timedelta(minutes=effective_health_settings.forecast_stale_after_minutes),
            now,
            missing_status="degraded" if role == "worker" else "unknown",
        )
        components["forecast_freshness"] = ComponentHealth(
            status=forecast_status,
            detail=forecast_detail,
            checked_at=now,
        )
        components["provider_recent_success"] = ComponentHealth(
            status=forecast_status,
            detail=forecast_detail,
            checked_at=now,
        )
        components["model_configured"] = ComponentHealth(
            status="healthy" if model_settings is not None else "unknown",
            detail=None if model_settings is not None else "model settings not provided",
            checked_at=now,
        )

        readiness_components = ["db", "migrations"]
        if role == "worker":
            readiness_components.extend(["worker_freshness", "forecast_freshness"])
        ready = all(components[name].status == "healthy" for name in readiness_components)
        overall_status = "healthy" if db_connected else "degraded"

        return HealthStatus(
            status=overall_status,
            version=__version__,
            db_connected=db_connected,
            last_forecast_fetch=last_forecast,
            last_rule_evaluation=last_eval,
            scheduler_status=scheduler_status,
            langsmith_enabled=langsmith_enabled,
            migrations_current=migrations_current,
            readiness="ready" if ready else "not_ready",
            components=components,
            timestamp=now,
        )

    @app.get("/health", response_model=HealthStatus)
    async def health_check() -> HealthStatus:
        return await build_status()

    @app.get("/livez")
    async def livez() -> dict[str, str]:
        return {"status": "alive", "version": __version__}

    @app.get("/readyz", response_model=HealthStatus)
    async def readyz(response: Response) -> HealthStatus:
        health = await build_status()
        if health.readiness != "ready":
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return health

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app
