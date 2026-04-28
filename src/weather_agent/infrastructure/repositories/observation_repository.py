from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.weather import ObservationResult
from weather_agent.infrastructure.db.base import Observation as ObservationORM
from weather_agent.infrastructure.repositories.base import BaseRepository


class ObservationRepository(BaseRepository):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def save_observations(self, result: ObservationResult) -> list[int]:
        ids: list[int] = []
        for point in result.points:
            orm_obj = ObservationORM(
                provider=point.provider,
                station_id=point.station_id,
                station_name=point.station_name,
                distance_km=point.distance_km,
                observed_at=point.observed_at,
                location_id=int(result.location.id),
                fetched_at=point.fetched_at,
                temperature_c=point.temperature_c,
                wind_speed_ms=point.wind_speed_ms,
                wind_direction_deg=point.wind_direction_deg,
                pressure_hpa=point.pressure_hpa,
                humidity_pct=point.humidity_pct,
                precipitation_mm=point.precipitation_mm,
                raw_payload=point.raw_payload,
            )
            self._session.add(orm_obj)
            await self._session.flush()
            ids.append(orm_obj.id)
        return ids

    async def get_observations_by_time_range(
        self,
        location_id: str,
        start: datetime,
        end: datetime,
    ) -> list[ObservationORM]:
        stmt = (
            select(ObservationORM)
            .where(
                ObservationORM.location_id == int(location_id),
                ObservationORM.observed_at >= start,
                ObservationORM.observed_at <= end,
            )
            .order_by(ObservationORM.observed_at)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())