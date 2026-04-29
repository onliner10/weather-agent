from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def normalize_database_url(url: str) -> str:
    if "+psycopg://" in url:
        return url.replace("+psycopg://", "+psycopg_async://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg_async://", 1)
    return url


def run_migrations() -> None:
    from alembic.config import Config

    from alembic import command

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


def create_engine(database_url: str) -> AsyncEngine:
    normalized = normalize_database_url(database_url)
    return create_async_engine(normalized, echo=False, pool_size=5, max_overflow=10)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
