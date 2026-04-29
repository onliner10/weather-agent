from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.infrastructure.db.base import AuthorizedUser


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create_authorized_user_id(
        self,
        telegram_user_id: int,
    ) -> int:
        result = await self._session.execute(
            select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        authorized_user = result.scalar_one_or_none()
        if authorized_user is None:
            authorized_user = AuthorizedUser(telegram_user_id=telegram_user_id)
            self._session.add(authorized_user)
            await self._session.flush()
        return int(authorized_user.id)
