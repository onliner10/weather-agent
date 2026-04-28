from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.infrastructure.db.base import (
    AuditLog,
    ForecastPoint,
    ForecastSnapshot,
    NotificationEvent,
    Observation,
    RuleEvaluationRun,
    TelegramContext,
)
from weather_agent.settings import RetentionSettings

logger = structlog.get_logger(__name__)


def _cutoff(days: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


class RetentionService:
    def __init__(
        self, session: AsyncSession, settings: RetentionSettings
    ) -> None:
        self._session = session
        self._settings = settings

    async def cleanup_thread_memory(
        self, older_than_days: int | None = None, *, dry_run: bool = False
    ) -> int:
        days = older_than_days if older_than_days is not None else self._settings.thread_memory_days
        cutoff = _cutoff(days)

        if dry_run:
            count = (
                await self._session.execute(
                    select(func.count()).select_from(TelegramContext).where(
                        TelegramContext.updated_at < cutoff
                    )
                )
            ).scalar_one()
            logger.info("dry_run_cleanup_thread_memory", days=days, would_delete=count)
            return count

        result = await self._session.execute(
            delete(TelegramContext).where(TelegramContext.updated_at < cutoff)
        )
        await self._session.flush()
        deleted = result.rowcount
        logger.info("cleanup_thread_memory", days=days, deleted=deleted)
        return deleted

    async def cleanup_raw_forecasts(
        self, older_than_days: int | None = None, *, dry_run: bool = False
    ) -> int:
        days = older_than_days if older_than_days is not None else self._settings.raw_forecast_days
        cutoff = _cutoff(days)

        snapshot_ids_stmt = select(ForecastSnapshot.id).where(
            ForecastSnapshot.fetched_at < cutoff
        )

        if dry_run:
            count = (
                await self._session.execute(
                    select(func.count()).select_from(ForecastSnapshot).where(
                        ForecastSnapshot.fetched_at < cutoff
                    )
                )
            ).scalar_one()
            logger.info("dry_run_cleanup_raw_forecasts", days=days, would_delete_snapshots=count)
            return count

        await self._session.execute(
            delete(ForecastPoint).where(
                ForecastPoint.snapshot_id.in_(snapshot_ids_stmt)
            )
        )
        snapshots_result = await self._session.execute(
            delete(ForecastSnapshot).where(ForecastSnapshot.fetched_at < cutoff)
        )
        await self._session.flush()
        deleted = snapshots_result.rowcount
        logger.info("cleanup_raw_forecasts", days=days, deleted=deleted)
        return deleted

    async def cleanup_aggregated_weather(
        self, older_than_days: int | None = None, *, dry_run: bool = False
    ) -> int:
        days = (
            older_than_days
            if older_than_days is not None
            else self._settings.aggregated_weather_days
        )
        cutoff = _cutoff(days)

        if dry_run:
            count = (
                await self._session.execute(
                    select(func.count()).select_from(Observation).where(
                        Observation.observed_at < cutoff
                    )
                )
            ).scalar_one()
            logger.info("dry_run_cleanup_aggregated_weather", days=days, would_delete=count)
            return count

        result = await self._session.execute(
            delete(Observation).where(Observation.observed_at < cutoff)
        )
        await self._session.flush()
        deleted = result.rowcount
        logger.info("cleanup_aggregated_weather", days=days, deleted=deleted)
        return deleted

    async def cleanup_notification_log(
        self, older_than_days: int | None = None, *, dry_run: bool = False
    ) -> int:
        days = (
            older_than_days
            if older_than_days is not None
            else self._settings.notification_log_days
        )
        cutoff = _cutoff(days)

        if dry_run:
            count = (
                await self._session.execute(
                    select(func.count()).select_from(NotificationEvent).where(
                        NotificationEvent.created_at < cutoff
                    )
                )
            ).scalar_one()
            logger.info("dry_run_cleanup_notification_log", days=days, would_delete=count)
            return count

        result = await self._session.execute(
            delete(NotificationEvent).where(NotificationEvent.created_at < cutoff)
        )
        await self._session.flush()
        deleted = result.rowcount
        logger.info("cleanup_notification_log", days=days, deleted=deleted)
        return deleted

    async def cleanup_audit_log(
        self, older_than_days: int | None = None, *, dry_run: bool = False
    ) -> int:
        days = older_than_days if older_than_days is not None else self._settings.audit_log_days
        cutoff = _cutoff(days)

        if dry_run:
            count = (
                await self._session.execute(
                    select(func.count()).select_from(AuditLog).where(
                        AuditLog.created_at < cutoff
                    )
                )
            ).scalar_one()
            logger.info("dry_run_cleanup_audit_log", days=days, would_delete=count)
            return count

        result = await self._session.execute(
            delete(AuditLog).where(AuditLog.created_at < cutoff)
        )
        await self._session.flush()
        deleted = result.rowcount
        logger.info("cleanup_audit_log", days=days, deleted=deleted)
        return deleted

    async def cleanup_trace_data(
        self, older_than_days: int | None = None, *, dry_run: bool = False
    ) -> int:
        days = older_than_days if older_than_days is not None else self._settings.trace_days
        cutoff = _cutoff(days)

        if dry_run:
            count = (
                await self._session.execute(
                    select(func.count()).select_from(RuleEvaluationRun).where(
                        RuleEvaluationRun.created_at < cutoff
                    )
                )
            ).scalar_one()
            logger.info("dry_run_cleanup_trace_data", days=days, would_delete=count)
            return count

        result = await self._session.execute(
            delete(RuleEvaluationRun).where(RuleEvaluationRun.created_at < cutoff)
        )
        await self._session.flush()
        deleted = result.rowcount
        logger.info("cleanup_trace_data", days=days, deleted=deleted)
        return deleted

    async def run_all_cleanup(self, *, dry_run: bool = False) -> dict[str, int]:
        results: dict[str, int] = {}
        results["thread_memory"] = await self.cleanup_thread_memory(dry_run=dry_run)
        results["raw_forecasts"] = await self.cleanup_raw_forecasts(dry_run=dry_run)
        results["aggregated_weather"] = await self.cleanup_aggregated_weather(dry_run=dry_run)
        results["notification_log"] = await self.cleanup_notification_log(dry_run=dry_run)
        results["audit_log"] = await self.cleanup_audit_log(dry_run=dry_run)
        results["trace_data"] = await self.cleanup_trace_data(dry_run=dry_run)
        logger.info("run_all_cleanup", dry_run=dry_run, results=results)
        return results