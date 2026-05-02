from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_RULE_EVALUATION_LOCK_KEY = 0x57454154484552


@dataclass(frozen=True)
class WorkerLock:
    acquired: bool


class WorkerCoordinator:
    def __init__(self, session: AsyncSession, lock_key: int = _RULE_EVALUATION_LOCK_KEY) -> None:
        self._session = session
        self._lock_key = lock_key
        self._uses_postgres = session.bind is not None and session.bind.dialect.name == "postgresql"

    async def acquire(self) -> WorkerLock:
        if not self._uses_postgres:
            return WorkerLock(acquired=True)
        result = await self._session.execute(
            text("SELECT pg_try_advisory_lock(:lock_key)"),
            {"lock_key": self._lock_key},
        )
        return WorkerLock(acquired=bool(result.scalar_one()))

    async def release(self) -> None:
        if not self._uses_postgres:
            return
        await self._session.execute(
            text("SELECT pg_advisory_unlock(:lock_key)"),
            {"lock_key": self._lock_key},
        )
