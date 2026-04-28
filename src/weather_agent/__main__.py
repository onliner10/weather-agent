from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from telegram import Update
from telegram.ext import ContextTypes

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.holidays import CachedHolidayProvider
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.model_factory import ModelFactory

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

    def init_services(self) -> None:
        from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
        from weather_agent.adapters.open_meteo.forecast_provider import (
            OpenMeteoDwdIconProvider,
        )

        self.cel_evaluator = CELEvaluator()
        self.holiday_provider = CachedHolidayProvider(
            base_url=self.settings.nager_date.base_url,
            timeout_seconds=self.settings.nager_date.timeout_seconds,
        )
        self.date_resolver = DateResolver(holiday_provider=self.holiday_provider)
        self.forecast_provider = OpenMeteoDwdIconProvider(settings=self.settings.open_meteo)
        self.observation_provider = ImgwSynopProvider(settings=self.settings.imgw)
        self.model_factory = ModelFactory(settings=self.settings.model)
        self.geocoder = Geocoder(model_factory=self.model_factory)
        logger.info("Application services initialized")

    def compile_graph(
        self,
        location_service: Any = None,
        rule_service: Any = None,
        user_id: int = 0,
    ) -> Any:
        from weather_agent.graphs.conversation import ConversationDeps, compile_conversation_graph

        deps = ConversationDeps(
            location_service=location_service,
            date_resolver=self.date_resolver,
            forecast_provider=self.forecast_provider,
            observation_provider=self.observation_provider,
            model_factory=self.model_factory,
            cel_evaluator=self.cel_evaluator,
            rule_service=rule_service,
            geocoder=self.geocoder,
            user_id=user_id,
        )
        return compile_conversation_graph(deps)


async def _make_message_handler(services: _BotServices) -> Any:
    services.init_services()
    logger.info("Application services ready, compiling base graph...")
    services.compile_graph()
    logger.info("Base graph compiled")

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

        from weather_agent.domain.auth import AuthorizationService
        from weather_agent.domain.locations import LocationService
        from weather_agent.domain.rules.service import NotificationRuleService
        from weather_agent.graphs.state import ConversationState
        from weather_agent.infrastructure.db.repos import SqlAlchemyAuthorizedUserRepo

        async with services.session_factory() as session:
            auth_repo = SqlAlchemyAuthorizedUserRepo(session)
            auth_service = AuthorizationService(
                allowed_user_ids=list(services.settings.telegram.allowed_user_ids),
                repo=auth_repo,
            )
            if not auth_service.is_authorized(user_id):
                await update.message.reply_text("Brak uprawnień do korzystania z tego bota.")
                return

            location_service = LocationService(session)
            rule_service = NotificationRuleService(
                session=session,
                cel_evaluator=services.cel_evaluator,
            )

            graph = services.compile_graph(
                location_service=location_service,
                rule_service=rule_service,
                user_id=user_id,
            )

            state: ConversationState = {
                "authorized_user_id": user_id,
                "chat_id": chat_id,
                "message_thread_id": thread_id,
                "context_key": context_key,
                "user_message": text,
            }

            try:
                result_state = await graph.ainvoke(state)
                answer = result_state.get(
                    "answer",
                    "Przepraszam, nie udało się przetworzyć zapytania.",
                )
            except Exception as exc:
                logger.exception("conversation graph failed")
                answer = f"Przepraszam, wystąpił błąd: {exc}"
            await session.commit()

        try:
            await update.message.reply_text(answer)
        except Exception:
            logger.exception("failed to send reply")

    logger.info("Message handler ready")
    return message_handler


def cmd_bot(_args: argparse.Namespace) -> None:
    print("Initializing bot services...")
    services = _BotServices()

    print("Running database migrations...")
    try:
        run_migrations()
    except Exception as exc:
        logger.warning("Migration failed (tables may already exist): %s", exc)

    from weather_agent.adapters.telegram.bot import TelegramBot
    from weather_agent.domain.auth import AuthorizationService

    message_handler = asyncio.run(_make_message_handler(services))

    auth_service = AuthorizationService(
        allowed_user_ids=list(services.settings.telegram.allowed_user_ids),
    )

    bot = TelegramBot(
        settings=services.settings.telegram,
        auth_service=auth_service,
        message_handler=message_handler,
    )
    bot.setup()
    print("Starting Telegram bot polling...")
    logger.info("Starting Telegram bot polling...")
    bot.run()


def cmd_worker(_args: argparse.Namespace) -> None:
    services = _BotServices()

    print("Running database migrations...")
    try:
        run_migrations()
    except Exception as exc:
        logger.warning("Migration failed (tables may already exist): %s", exc)

    async def _run_worker() -> None:
        from weather_agent.domain.rules.service import NotificationRuleService
        from weather_agent.infrastructure.repositories.forecast_repository import (
            ForecastRepository,
        )
        from weather_agent.infrastructure.worker.rule_evaluator import RuleEvaluationWorker

        async with services.session_factory() as session:
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