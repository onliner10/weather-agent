from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from weather_agent import __version__
from weather_agent.infrastructure.db.base import ForecastSnapshot, RuleEvaluationRun
from weather_agent.observability.langsmith_tracing import LangSmithTracing


class HealthStatus(BaseModel):
    status: str
    version: str
    db_connected: bool
    last_forecast_fetch: datetime | None
    last_rule_evaluation: datetime | None
    scheduler_status: str
    langsmith_enabled: bool
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


def create_health_app(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> FastAPI:
    app = FastAPI(title="Weather Agent Health", version=__version__)

    @app.get("/health", response_model=HealthStatus)
    async def health_check() -> HealthStatus:
        if session_factory is not None:
            async with session_factory() as session:
                db_connected = await _check_db(session)
                last_forecast = await _get_last_forecast_fetch(session) if db_connected else None
                last_eval = await _get_last_rule_evaluation(session) if db_connected else None
        else:
            db_connected = False
            last_forecast = None
            last_eval = None

        langsmith_enabled = LangSmithTracing.is_enabled()
        scheduler_status = "running" if db_connected else "stopped"

        overall_status = "healthy" if db_connected else "degraded"

        return HealthStatus(
            status=overall_status,
            version=__version__,
            db_connected=db_connected,
            last_forecast_fetch=last_forecast,
            last_rule_evaluation=last_eval,
            scheduler_status=scheduler_status,
            langsmith_enabled=langsmith_enabled,
            timestamp=datetime.now(UTC),
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app
