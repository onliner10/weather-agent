from __future__ import annotations

import argparse
import asyncio
import os

from langsmith import trace

from weather_agent.api.health import create_health_app
from weather_agent.infrastructure.app_container import AppContainer
from weather_agent.infrastructure.db.setup import run_migrations
from weather_agent.infrastructure.utils import acquire_lock
from weather_agent.observability.logging import (
    bound_worker_context,
    get_logger,
)
from weather_agent.observability.server import start_observability_server

logger = get_logger(__name__)


def cmd_worker(_args: argparse.Namespace) -> None:
    pid_file = acquire_lock("worker")

    async def _run() -> None:
        async with AppContainer() as app:
            if app.settings.observability.enabled:
                health_app = create_health_app(
                    session_factory=app.session_factory,
                    health_settings=app.settings.health,
                    scheduler_settings=app.settings.scheduler,
                    model_settings=app.settings.model,
                    role="worker",
                )
                start_observability_server(
                    app=health_app,
                    host="0.0.0.0",
                    port=app.settings.observability.worker_port,
                )

            from weather_agent.domain.notifications.deduplication import (
                NotificationDeduplicator,
            )
            from weather_agent.domain.notifications.events import (
                NotificationEventService,
            )
            from weather_agent.domain.rules.service import NotificationRuleService
            from weather_agent.infrastructure.repositories.forecast_repository import (
                ForecastRepository,
            )
            from weather_agent.infrastructure.worker.forecast_fetcher import (
                WorkerForecastFetcher,
            )
            from weather_agent.infrastructure.worker.rule_evaluator import (
                RuleEvaluationWorker,
            )

            async with app.session_factory() as session:
                async with trace(
                    "worker_startup",
                    run_type="tool",
                    metadata={"mode": "worker"},
                ):
                    rule_service = NotificationRuleService(
                        session=session,
                        rule_expression_evaluator=app.rule_expression_evaluator,
                    )
                    forecast_repo = ForecastRepository(session=session)

                    forecast_fetcher = WorkerForecastFetcher(
                        session=session,
                        forecast_provider=app.forecast_provider,
                        forecast_repo=forecast_repo,
                    )

                    from weather_agent.adapters.telegram.http_sender import (
                        TelegramHttpNotificationSender,
                    )
                    from weather_agent.observability.logging import get_audit_logger

                    audit_logger = get_audit_logger(session)
                    event_service = NotificationEventService(
                        session=session,
                        audit_logger=audit_logger,
                    )
                    deduplicator = NotificationDeduplicator(session=session)
                    notification_sender = TelegramHttpNotificationSender(
                        bot_token=app.settings.telegram.bot_token,
                        httpx_client=app.httpx_client,
                    )

                    worker = RuleEvaluationWorker(
                        session=session,
                        forecast_repo=forecast_repo,
                        rule_expression_evaluator=app.rule_expression_evaluator,
                        rule_service=rule_service,
                        settings=app.settings.scheduler,
                        forecast_fetcher=forecast_fetcher,
                        notification_sender=notification_sender,
                        event_service=event_service,
                        deduplicator=deduplicator,
                    )

                with bound_worker_context():
                    logger.info("starting_rule_evaluation_worker")
                    try:
                        await worker.run_loop()
                    finally:
                        if os.path.exists(pid_file):
                            os.unlink(pid_file)

    try:
        try:
            from weather_agent.settings import load_settings

            run_migrations(load_settings().database_url)
        except Exception:
            logger.exception("migration_failed")
            raise
        asyncio.run(_run())
    finally:
        if os.path.exists(pid_file):
            os.unlink(pid_file)
