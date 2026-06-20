from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.locations import _orm_to_domain
from weather_agent.domain.weather import (
    ForecastPoint,
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


@dataclass(frozen=True)
class ForecastFetchResult:
    """Result of a fresh forecast fetch for rule evaluation.

    Carries *in-memory* point dicts so the evaluator does not need to
    read forecast_points back from the database -- the points are no longer
    persisted at all.
    """

    snapshot_id: int | None
    point_dicts: list[dict[str, Any]]


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

    async def fetch_fresh_forecast(self, location_id: int) -> ForecastFetchResult | None:
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

        point_dicts = _domain_points_to_dicts(forecast_result.points)

        # Save only the snapshot header, NOT the individual forecast points.
        # This keeps the DB small while still allowing us to track when
        # forecasts were fetched. Points are passed in-memory via the result.
        snapshot_id = await self._repo.save_snapshot(forecast_result, persist_points=False)
        await self._session.flush()
        return ForecastFetchResult(snapshot_id=snapshot_id, point_dicts=point_dicts)


_EVAL_FIELDS: tuple[str, ...] = (
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
)


def _domain_points_to_dicts(
    points: list[ForecastPoint],
) -> list[dict[str, Any]]:
    """Convert domain ForecastPoint list to evaluation-ready dicts."""
    result: list[dict[str, Any]] = []
    for point in points:
        d: dict[str, Any] = {
            "target_time": point.target_time,
            "fetched_at": point.fetched_at,
            "raw_payload": point.raw_payload,
        }
        for field_name in _EVAL_FIELDS:
            val = getattr(point, field_name, None)
            if val is not None:
                d[field_name] = val
        result.append(d)
    return result
