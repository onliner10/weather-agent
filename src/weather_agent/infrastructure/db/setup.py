from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from alembic.config import Config
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


@contextmanager
def _migration_database_url(database_url: str | None) -> Iterator[None]:
    if database_url is None:
        yield
        return

    old_value = os.environ.get("WEATHER_AGENT_DATABASE_URL")
    os.environ["WEATHER_AGENT_DATABASE_URL"] = database_url
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("WEATHER_AGENT_DATABASE_URL", None)
        else:
            os.environ["WEATHER_AGENT_DATABASE_URL"] = old_value


def _alembic_config(database_url: str | None = None) -> Config:
    alembic_cfg = Config("alembic.ini")
    if database_url is not None:
        alembic_cfg.set_main_option("sqlalchemy.url", normalize_database_url(database_url))
    return alembic_cfg


def run_migrations(database_url: str | None = None) -> None:
    from alembic import command

    alembic_cfg = _alembic_config(database_url)
    with _migration_database_url(database_url):
        command.upgrade(alembic_cfg, "head")


def get_migration_heads() -> set[str]:
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_alembic_config())
    return set(script.get_heads())


def create_engine(database_url: str) -> AsyncEngine:
    normalized = normalize_database_url(database_url)
    return create_async_engine(normalized, echo=False, pool_size=5, max_overflow=10)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
