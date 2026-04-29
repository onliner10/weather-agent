from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.infrastructure.db.base import TelegramContext as TelegramContextORM


class ContextKey(BaseModel):
    chat_id: int
    message_thread_id: int | None = None

    @property
    def context_key(self) -> str:
        if self.message_thread_id is not None:
            return f"{self.chat_id}:{self.message_thread_id}"
        return str(self.chat_id)


class TelegramContext(BaseModel):
    context_key: str
    chat_id: int
    message_thread_id: int | None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class TelegramContextService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def compute_context_key(chat_id: int, message_thread_id: int | None) -> ContextKey:
        return ContextKey(chat_id=chat_id, message_thread_id=message_thread_id)

    async def get_or_create_context(
        self, chat_id: int, message_thread_id: int | None
    ) -> TelegramContext:
        key = self.compute_context_key(chat_id, message_thread_id)
        stmt = select(TelegramContextORM).where(TelegramContextORM.context_key == key.context_key)
        result = await self._session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is not None:
            return TelegramContext(
                context_key=orm_obj.context_key,
                chat_id=orm_obj.chat_id,
                message_thread_id=orm_obj.message_thread_id,
                metadata=orm_obj.metadata_.copy(),
                created_at=orm_obj.created_at,
                updated_at=orm_obj.updated_at,
            )

        now = datetime.now(UTC)
        orm_obj = TelegramContextORM(
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            context_key=key.context_key,
            metadata_={},
            created_at=now,
            updated_at=now,
        )
        self._session.add(orm_obj)
        await self._session.flush()

        return TelegramContext(
            context_key=orm_obj.context_key,
            chat_id=orm_obj.chat_id,
            message_thread_id=orm_obj.message_thread_id,
            metadata=orm_obj.metadata_.copy(),
            created_at=orm_obj.created_at,
            updated_at=orm_obj.updated_at,
        )

    async def update_context(
        self, context_key: str, metadata: dict[str, object]
    ) -> TelegramContext:
        stmt = select(TelegramContextORM).where(TelegramContextORM.context_key == context_key)
        result = await self._session.execute(stmt)
        orm_obj = result.scalar_one()

        orm_obj.metadata_ = metadata
        orm_obj.updated_at = datetime.now(UTC)
        await self._session.flush()

        return TelegramContext(
            context_key=orm_obj.context_key,
            chat_id=orm_obj.chat_id,
            message_thread_id=orm_obj.message_thread_id,
            metadata=orm_obj.metadata_.copy(),
            created_at=orm_obj.created_at,
            updated_at=orm_obj.updated_at,
        )

    async def clear_context(self, context_key: str) -> None:
        stmt = select(TelegramContextORM).where(TelegramContextORM.context_key == context_key)
        result = await self._session.execute(stmt)
        orm_obj = result.scalar_one_or_none()

        if orm_obj is not None:
            orm_obj.metadata_ = {}
            orm_obj.updated_at = datetime.now(UTC)
            await self._session.flush()
