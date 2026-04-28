from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


def _normalize_database_url(url: str) -> str:
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


def _create_engine(database_url: str) -> AsyncEngine:
    normalized = _normalize_database_url(database_url)
    return create_async_engine(normalized, echo=False, pool_size=5, max_overflow=10)


def _create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def _session_context(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


class _BotServices:
    def __init__(self) -> None:
        from weather_agent.observability.langsmith_tracing import configure_tracing
        from weather_agent.observability.logging import configure_logging
        from weather_agent.settings import load_settings

        try:
            settings = load_settings()
        except Exception as exc:
            print(
                f"Error loading configuration: {exc}\n"
                "Ensure all required environment variables are set "
                "(WEATHER_AGENT_DATABASE_URL, WEATHER_AGENT_TELEGRAM__BOT_TOKEN, etc.).",
                file=sys.stderr,
            )
            sys.exit(1)

        configure_logging()
        configure_tracing(settings.langsmith)

        self.settings = settings
        self.engine = _create_engine(settings.database_url)
        self.session_factory = _create_session_factory(self.engine)


async def _make_message_handler(services: _BotServices) -> Any:
    from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
    from weather_agent.adapters.open_meteo.forecast_provider import OpenMeteoDwdIconProvider
    from weather_agent.adapters.telegram.context import TelegramContextService
    from weather_agent.domain.auth import AuthorizationService
    from weather_agent.domain.cel.evaluator import CELEvaluator
    from weather_agent.domain.date_resolver import DateResolver
    from weather_agent.domain.holidays import CachedHolidayProvider
    from weather_agent.domain.locations import LocationService
    from weather_agent.domain.rules.service import NotificationRuleService
    from weather_agent.graphs.conversation import ConversationDeps, compile_conversation_graph
    from weather_agent.infrastructure.db.repos import SqlAlchemyAuthorizedUserRepo
    from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService
    from weather_agent.llm.model_factory import ModelFactory
    from weather_agent.observability.logging import AuditLogger

    session_factory = services.session_factory
    settings = services.settings

    async with _session_context(session_factory) as session:
        auth_repo = SqlAlchemyAuthorizedUserRepo(session)
        AuthorizationService(
            allowed_user_ids=list(settings.telegram.allowed_user_ids),
            repo=auth_repo,
        )
        cel_evaluator = CELEvaluator()

        holiday_provider = CachedHolidayProvider(
            base_url=settings.nager_date.base_url,
            timeout_seconds=settings.nager_date.timeout_seconds,
        )
        date_resolver = DateResolver(holiday_provider=holiday_provider)

        location_service = LocationService(session)
        rule_service = NotificationRuleService(session=session, cel_evaluator=cel_evaluator)

        forecast_provider = OpenMeteoDwdIconProvider(settings=settings.open_meteo)
        observation_provider = ImgwSynopProvider(settings=settings.imgw)
        model_factory = ModelFactory(settings=settings.model)

        context_service = TelegramContextService(session)
        ThreadMemoryService(
            context_service=context_service,
            default_ttl_days=settings.retention.thread_memory_days,
        )
        AuditLogger(session)

        deps = ConversationDeps(
            location_service=location_service,
            date_resolver=date_resolver,
            forecast_provider=forecast_provider,
            observation_provider=observation_provider,
            model_factory=model_factory,
            cel_evaluator=cel_evaluator,
            rule_service=rule_service,
            user_id=0,
        )
        graph = compile_conversation_graph(deps)

    async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.effective_user is None:
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id if update.effective_chat else 0
        thread_id = (
            update.message.message_thread_id
            if hasattr(update.message, "message_thread_id")
            else None
        )
        text = update.message.text

        if text is None:
            return

        context_key = f"{chat_id}:{thread_id}" if thread_id else str(chat_id)

        from weather_agent.graphs.state import ConversationState

        state: ConversationState = {
            "authorized_user_id": user_id,
            "chat_id": chat_id,
            "message_thread_id": thread_id,
            "context_key": context_key,
            "user_message": text,
        }

        try:
            result = await graph.ainvoke(state)
            answer = result.get("answer", "Przepraszam, nie udało się przetworzyć zapytania.")
        except Exception as exc:
            logger.exception("conversation graph failed")
            answer = f"Przepraszam, wystąpił błąd: {exc}"

        try:
            await update.message.reply_text(answer)
        except Exception:
            logger.exception("failed to send reply")

    return message_handler


def cmd_bot(_args: argparse.Namespace) -> None:
    services = _BotServices()

    try:
        logger.info("Running database migrations...")
        run_migrations()
    except Exception as exc:
        logger.warning("Migration failed: %s", exc)

    from weather_agent.adapters.telegram.bot import TelegramBot
    from weather_agent.domain.auth import AuthorizationService
    from weather_agent.infrastructure.db.repos import SqlAlchemyAuthorizedUserRepo

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        message_handler = loop.run_until_complete(_make_message_handler(services))
    finally:
        loop.close()

    async def _get_auth_service() -> AuthorizationService:
        async with _session_context(services.session_factory) as session:
            auth_repo = SqlAlchemyAuthorizedUserRepo(session)
            return AuthorizationService(
                allowed_user_ids=list(services.settings.telegram.allowed_user_ids),
                repo=auth_repo,
            )

    auth_service = asyncio.run(_get_auth_service())

    bot = TelegramBot(
        settings=services.settings.telegram,
        auth_service=auth_service,
        message_handler=message_handler,
    )
    bot.setup()
    logger.info("Starting Telegram bot...")
    bot.run()


def cmd_worker(_args: argparse.Namespace) -> None:
    services = _BotServices()

    try:
        logger.info("Running database migrations...")
        run_migrations()
    except Exception as exc:
        logger.warning("Migration failed: %s", exc)

    async def _run_worker() -> None:
        from weather_agent.domain.cel.evaluator import CELEvaluator
        from weather_agent.domain.rules.service import NotificationRuleService
        from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
        from weather_agent.infrastructure.worker.rule_evaluator import RuleEvaluationWorker

        async with _session_context(services.session_factory) as session:
            cel_evaluator = CELEvaluator()
            rule_service = NotificationRuleService(
                session=session,
                cel_evaluator=cel_evaluator,
            )
            forecast_repo = ForecastRepository(session=session)

            worker = RuleEvaluationWorker(
                session=session,
                forecast_repo=forecast_repo,
                cel_evaluator=cel_evaluator,
                rule_service=rule_service,
                settings=services.settings.scheduler,
            )

            logger.info("Starting rule evaluation worker...")
            await worker.run_loop()

    asyncio.run(_run_worker())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weather_agent",
        description="Telegram Weather AI Agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bot_parser = subparsers.add_parser("bot", help="Start the Telegram bot")
    bot_parser.set_defaults(func=cmd_bot)

    worker_parser = subparsers.add_parser("worker", help="Start the rule evaluation worker")
    worker_parser.set_defaults(func=cmd_worker)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()