from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.weather import WeatherWarning
from weather_agent.infrastructure.db.base import OfficialWarning as OfficialWarningORM
from weather_agent.infrastructure.repositories.base import BaseRepository


class WarningRepository(BaseRepository):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def save_warnings(self, warnings: list[WeatherWarning]) -> list[int]:
        ids: list[int] = []
        for warning in warnings:
            stmt = select(OfficialWarningORM).where(
                OfficialWarningORM.provider == warning.provider,
                OfficialWarningORM.external_id == warning.external_id,
                OfficialWarningORM.valid_from == warning.valid_from,
            )
            result = await self._session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is not None:
                ids.append(existing.id)
                continue

            orm_obj = OfficialWarningORM(
                provider=warning.provider,
                external_id=warning.external_id,
                location_id=int(warning.location_id),
                severity=warning.severity,
                category=warning.category,
                headline=warning.headline,
                description=warning.description,
                valid_from=warning.valid_from,
                valid_to=warning.valid_to,
                raw_payload=warning.raw_payload,
            )
            self._session.add(orm_obj)
            await self._session.flush()
            ids.append(orm_obj.id)
        return ids

    async def get_warnings(
        self,
        location_id: str,
        start: datetime,
        end: datetime,
    ) -> list[OfficialWarningORM]:
        stmt = (
            select(OfficialWarningORM)
            .where(
                OfficialWarningORM.location_id == int(location_id),
                OfficialWarningORM.valid_from >= start,
                OfficialWarningORM.valid_to <= end,
            )
            .order_by(OfficialWarningORM.valid_from)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())
