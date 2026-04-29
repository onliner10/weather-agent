from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def _normalize_database_url(url: str) -> str:
    if "+psycopg://" in url:
        url = url.replace("+psycopg://", "+psycopg_async://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg_async://", 1)
    return url


@lru_cache(maxsize=1)
def _get_engine() -> AsyncEngine:
    from weather_agent.settings import load_settings

    settings = load_settings()
    url = _normalize_database_url(settings.database_url)
    return create_async_engine(url, echo=False, pool_size=5, max_overflow=10)


@lru_cache(maxsize=1)
def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        _get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _get_session_factory()() as session:
        yield session


@asynccontextmanager
async def session_context() -> AsyncGenerator[AsyncSession, None]:
    async with _get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
