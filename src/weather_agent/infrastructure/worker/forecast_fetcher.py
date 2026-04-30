from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.locations import _orm_to_domain
from weather_agent.domain.weather import (
    ForecastResolution,
    LocationRef,
    TimeRange,
    WeatherVariable,
)
from weather_agent.infrastructure.db.base import Location as LocationORM
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.observability.logging import get_logger

logger = get_logger(__name__)

_ALL_VARIABLES = list(WeatherVariable)


class ForecastRefreshError(Exception):
    pass


class WorkerForecastFetcher:
    def __init__(
        self,
        session: AsyncSession,
        forecast_provider: Any,
        forecast_repo: ForecastRepository,
    ) -> None:
        self._session = session
        self._provider = forecast_provider
        self._repo = forecast_repo

    async def fetch_fresh_forecast(self, location_id: int) -> int | None:
        stmt = select(LocationORM).where(
            LocationORM.id == location_id,
            LocationORM.enabled.is_(True),
        )
        result = await self._session.execute(stmt)
        location_orm = result.scalar_one_or_none()
        if location_orm is None:
            logger.warning(
                "forecast_fetcher_location_not_found",
                location_id=location_id,
            )
            return None

        domain_loc = _orm_to_domain(location_orm)
        location_ref = LocationRef(
            id=str(domain_loc.id),
            name=domain_loc.name,
            latitude=domain_loc.latitude,
            longitude=domain_loc.longitude,
        )

        now = datetime.now(UTC)
        time_range = TimeRange(
            start=now - timedelta(hours=1),
            end=now + timedelta(days=7),
        )

        forecast_result = await self._provider.get_forecast(
            location=location_ref,
            time_range=time_range,
            variables=_ALL_VARIABLES,
            resolution=ForecastResolution.hourly,
        )

        snapshot_id = await self._repo.save_snapshot(forecast_result)
        await self._session.flush()
        return snapshot_id
