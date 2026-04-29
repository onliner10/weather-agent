from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.global_settings import GlobalUnits
from weather_agent.infrastructure.db.base import AuthorizedUser, GlobalSetting


class SqlAlchemyAuthorizedUserRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_user(self, telegram_user_id: int) -> None:
        exists = await self._session.execute(
            select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        if exists.scalar_one_or_none() is not None:
            return
        self._session.add(AuthorizedUser(telegram_user_id=telegram_user_id))
        await self._session.flush()

    async def remove_user(self, telegram_user_id: int) -> None:
        result = await self._session.execute(
            select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await self._session.delete(row)
            await self._session.flush()

    async def list_user_ids(self) -> list[int]:
        result = await self._session.execute(select(AuthorizedUser.telegram_user_id))
        return [row[0] for row in result.all()]


class SqlAlchemyGlobalSettingsRepo:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_units(self, units: GlobalUnits) -> None:
        fields = {
            "temperature": units.temperature,
            "wind_speed": units.wind_speed,
            "precipitation": units.precipitation,
            "pressure": units.pressure,
        }
        for key, value in fields.items():
            result = await self._session.execute(
                select(GlobalSetting).where(GlobalSetting.key == key)
            )
            row = result.scalar_one_or_none()
            if row is not None:
                row.value = value
            else:
                self._session.add(GlobalSetting(key=key, value=value))
        await self._session.flush()

    async def load_units(self) -> GlobalUnits | None:
        result = await self._session.execute(
            select(GlobalSetting).where(
                GlobalSetting.key.in_(["temperature", "wind_speed", "precipitation", "pressure"])
            )
        )
        rows = {row.key: row.value for row in result.scalars().all()}
        if not rows:
            return None
        return GlobalUnits(
            temperature=rows.get("temperature", "celsius"),
            wind_speed=rows.get("wind_speed", "ms"),
            precipitation=rows.get("precipitation", "mm"),
            pressure=rows.get("pressure", "hpa"),
        )
