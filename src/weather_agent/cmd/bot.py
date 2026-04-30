from __future__ import annotations

import argparse
import asyncio
import os

from weather_agent.api.health import create_health_app
from weather_agent.infrastructure.app_container import AppContainer
from weather_agent.infrastructure.db.setup import run_migrations
from weather_agent.infrastructure.utils import acquire_lock
from weather_agent.observability.logging import get_logger
from weather_agent.observability.server import start_observability_server

logger = get_logger(__name__)


def cmd_bot(_args: argparse.Namespace) -> None:
    pid_file = acquire_lock("bot")

    async def _run() -> None:
        async with AppContainer() as app:
            if app.settings.observability.enabled:
                health_app = create_health_app(session_factory=app.session_factory)
                start_observability_server(
                    app=health_app,
                    host="0.0.0.0",
                    port=app.settings.observability.bot_port,
                )

            try:
                run_migrations()
            except Exception as exc:
                logger.warning(
                    "migration_failed",
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                )

            from weather_agent.adapters.telegram.bot import TelegramBot
            from weather_agent.adapters.telegram.handler import make_message_handler
            from weather_agent.domain.auth import AuthorizationService

            message_handler = await make_message_handler(app)

            auth_service = AuthorizationService(
                allowed_user_ids=list(app.settings.telegram.allowed_user_ids),
            )

            bot = TelegramBot(
                settings=app.settings.telegram,
                auth_service=auth_service,
                message_handler=message_handler,
                session_factory=app.session_factory,
            )
            bot.setup()
            logger.info("Starting Telegram bot polling...")
            try:
                await bot.start()
                while True:
                    await asyncio.sleep(3600)
            finally:
                await bot.stop()

    try:
        asyncio.run(_run())
    finally:
        if os.path.exists(pid_file):
            os.unlink(pid_file)
