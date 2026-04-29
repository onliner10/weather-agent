from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any

from langsmith import trace
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from telegram import Update
from telegram.ext import ContextTypes

from weather_agent.api.health import create_health_app
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.holidays import CachedHolidayProvider
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.logging import (
    bound_telegram_context,
    bound_worker_context,
    generate_correlation_id,
    get_logger,
)
from weather_agent.observability.metrics import (
    CONVERSATION_FAILURES_TOTAL,
    CONVERSATION_TURN_DURATION_SECONDS,
    CONVERSATION_TURNS_TOTAL,
    REPLY_CONTEXT_HITS_TOTAL,
    REPLY_SEND_DURATION_SECONDS,
    REPLY_SEND_TOTAL,
)
from weather_agent.observability.server import start_observability_server
from weather_agent.observability.tracing import build_graph_config

logger = get_logger(__name__)


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
        memory_service: Any = None,
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
            memory_service=memory_service,
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

        message_id = update.message.message_id
        reply_to_message_id = None
        if update.message.reply_to_message is not None:
            reply_to_message_id = update.message.reply_to_message.message_id

        context_key = f"{chat_id}:{thread_id}" if thread_id else str(chat_id)

        with bound_telegram_context(
            correlation_id=generate_correlation_id(),
            chat_id=chat_id,
            message_thread_id=thread_id,
            telegram_user_id=user_id,
            message_id=message_id,
            reply_to_message_id=reply_to_message_id,
            context_key=context_key,
        ):
            from weather_agent.adapters.telegram.context import TelegramContextService
            from weather_agent.domain.locations import LocationService
            from weather_agent.domain.rules.service import NotificationRuleService
            from weather_agent.graphs.state import ConversationState
            from weather_agent.infrastructure.memory.thread_memory import ThreadMemoryService

            async with services.session_factory() as session:
                location_service = LocationService(session)
                rule_service = NotificationRuleService(
                    session=session,
                    cel_evaluator=services.cel_evaluator,
                )
                context_service = TelegramContextService(session)
                memory_service = ThreadMemoryService(context_service)

                graph = services.compile_graph(
                    location_service=location_service,
                    rule_service=rule_service,
                    user_id=user_id,
                    memory_service=memory_service,
                )

                state: ConversationState = {
                    "authorized_user_id": user_id,
                    "chat_id": chat_id,
                    "message_thread_id": thread_id,
                    "context_key": context_key,
                    "user_message": text,
                    "message_id": message_id,
                    "reply_to_message_id": reply_to_message_id,
                }

                graph_config = build_graph_config(state)
                if services.model_factory is not None:
                    graph_config["metadata"]["model_provider"] = services.model_factory.provider
                    graph_config["metadata"]["model_name"] = services.model_factory.model_name

                CONVERSATION_TURNS_TOTAL.inc()
                turn_start = time.perf_counter()
                try:
                    result_state = await graph.ainvoke(state, config=graph_config)
                    answer = result_state.get(
                        "answer",
                        "Przepraszam, nie udało się przetworzyć zapytania.",
                    )
                except Exception as exc:
                    CONVERSATION_FAILURES_TOTAL.inc()
                    logger.exception(
                        "conversation_graph_failed",
                        error_class=type(exc).__name__,
                        outcome="failure",
                    )
                    answer = "Przepraszam, wystąpił błąd. Spróbuj ponownie za chwilę."
                finally:
                    CONVERSATION_TURN_DURATION_SECONDS.observe(time.perf_counter() - turn_start)
                await session.commit()

            if reply_to_message_id is not None:
                REPLY_CONTEXT_HITS_TOTAL.labels(source="reply_to").inc()

            reply_start = time.perf_counter()
            try:
                sent_message = await update.message.reply_text(answer)
                bot_message_id = sent_message.message_id if sent_message else None
                REPLY_SEND_TOTAL.labels(outcome="success").inc()
            except Exception as exc:
                REPLY_SEND_TOTAL.labels(outcome="failure").inc()
                logger.exception(
                    "reply_send_failed",
                    error_class=type(exc).__name__,
                    outcome="failure",
                )
                bot_message_id = None
            finally:
                REPLY_SEND_DURATION_SECONDS.observe(time.perf_counter() - reply_start)

            if bot_message_id is not None:
                try:
                    async with services.session_factory() as session:
                        context_service2 = TelegramContextService(session)
                        memory_service2 = ThreadMemoryService(context_service2)
                        await memory_service2.update_last_bot_turn_message_id(
                            context_key, bot_message_id
                        )
                        await session.commit()
                except Exception:
                    logger.warning(
                        "bot_message_id_persist_failed",
                        exc_info=True,
                    )

    logger.info("message_handler_ready")
    return message_handler


def _acquire_lock(name: str) -> str:
    pid_file = f"/tmp/weather-agent-{name}.pid"
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            try:
                old_pid = int(f.read().strip())
            except ValueError:
                old_pid = -1
        if old_pid > 0 and os.path.exists(f"/proc/{old_pid}"):
            print(f"Another {name} instance (PID {old_pid}) is already running.", file=sys.stderr)
            sys.exit(1)
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    return pid_file


def cmd_bot(_args: argparse.Namespace) -> None:
    _pid_file = _acquire_lock("bot")
    print("Initializing bot services...")
    services = _BotServices()

    if services.settings.observability.enabled:
        health_app = create_health_app(session_factory=services.session_factory)
        start_observability_server(
            app=health_app,
            host="0.0.0.0",
            port=services.settings.observability.bot_port,
        )

    print("Running database migrations...")
    try:
        run_migrations()
    except Exception as exc:
        logger.warning(
            "migration_failed",
            error_class=type(exc).__name__,
            error_message=str(exc),
        )

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
        session_factory=services.session_factory,
    )
    bot.setup()
    print("Starting Telegram bot polling...")
    logger.info("Starting Telegram bot polling...")
    try:
        bot.run()
    finally:
        os.unlink(_pid_file)


def cmd_worker(_args: argparse.Namespace) -> None:
    _pid_file = _acquire_lock("worker")
    services = _BotServices()

    if services.settings.observability.enabled:
        health_app = create_health_app(session_factory=services.session_factory)
        start_observability_server(
            app=health_app,
            host="0.0.0.0",
            port=services.settings.observability.worker_port,
        )

    print("Running database migrations...")
    try:
        run_migrations()
    except Exception as exc:
        logger.warning(
            "migration_failed",
            error_class=type(exc).__name__,
            error_message=str(exc),
        )

    async def _run_worker() -> None:
        from weather_agent.domain.rules.service import NotificationRuleService
        from weather_agent.infrastructure.repositories.forecast_repository import (
            ForecastRepository,
        )
        from weather_agent.infrastructure.worker.rule_evaluator import RuleEvaluationWorker

        async with services.session_factory() as session:
            async with trace(
                "worker_startup",
                run_type="tool",
                metadata={"mode": "worker"},
            ):
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

            with bound_worker_context():
                logger.info("starting_rule_evaluation_worker")
                try:
                    await worker.run_loop()
                finally:
                    os.unlink(_pid_file)

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
