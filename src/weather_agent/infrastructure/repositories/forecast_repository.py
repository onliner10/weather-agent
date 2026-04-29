from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.weather import ForecastResult
from weather_agent.infrastructure.db.base import ForecastPoint as ForecastPointORM
from weather_agent.infrastructure.db.base import ForecastSnapshot as ForecastSnapshotORM
from weather_agent.infrastructure.repositories.base import BaseRepository


class ForecastRepository(BaseRepository):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def save_snapshot(self, result: ForecastResult) -> int:
        snapshot = ForecastSnapshotORM(
            provider=result.provider,
            model=result.model,
            location_id=int(result.location.id),
            fetched_at=result.fetched_at,
            raw_payload=result.raw_payload,
        )
        self._session.add(snapshot)
        await self._session.flush()
        snapshot_id = snapshot.id

        for point in result.points:
            orm_point = ForecastPointORM(
                snapshot_id=snapshot_id,
                target_time=point.target_time,
                location_id=int(point.location_id),
                temperature_2m_c=point.temperature_2m_c,
                apparent_temperature_c=point.apparent_temperature_c,
                precipitation_mm=point.precipitation_mm,
                precipitation_probability_pct=point.precipitation_probability_pct,
                rain_mm=point.rain_mm,
                snowfall_cm=point.snowfall_cm,
                cloud_cover_pct=point.cloud_cover_pct,
                wind_speed_10m_ms=point.wind_speed_10m_ms,
                wind_gusts_10m_ms=point.wind_gusts_10m_ms,
                wind_direction_10m_deg=point.wind_direction_10m_deg,
                pressure_msl_hpa=point.pressure_msl_hpa,
                relative_humidity_2m_pct=point.relative_humidity_2m_pct,
                weather_code=point.weather_code,
                raw_payload=point.raw_payload,
            )
            self._session.add(orm_point)

        await self._session.flush()
        return snapshot_id

    async def get_latest_snapshot(self, location_id: str) -> ForecastSnapshotORM | None:
        stmt = (
            select(ForecastSnapshotORM)
            .where(ForecastSnapshotORM.location_id == int(location_id))
            .order_by(ForecastSnapshotORM.fetched_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_points_by_time_range(
        self,
        location_id: str,
        start: datetime,
        end: datetime,
    ) -> list[ForecastPointORM]:
        stmt = (
            select(ForecastPointORM)
            .where(
                ForecastPointORM.location_id == int(location_id),
                ForecastPointORM.target_time >= start,
                ForecastPointORM.target_time <= end,
            )
            .order_by(ForecastPointORM.target_time)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_points_for_snapshot(
        self,
        snapshot_id: int,
        start: datetime,
        end: datetime,
    ) -> list[ForecastPointORM]:
        stmt = (
            select(ForecastPointORM)
            .where(
                ForecastPointORM.snapshot_id == snapshot_id,
                ForecastPointORM.target_time >= start,
                ForecastPointORM.target_time <= end,
            )
            .order_by(ForecastPointORM.target_time)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_previous_snapshot(
        self,
        location_id: str,
        before: datetime,
    ) -> ForecastSnapshotORM | None:
        stmt = (
            select(ForecastSnapshotORM)
            .where(
                ForecastSnapshotORM.location_id == int(location_id),
                ForecastSnapshotORM.fetched_at < before,
            )
            .order_by(ForecastSnapshotORM.fetched_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
