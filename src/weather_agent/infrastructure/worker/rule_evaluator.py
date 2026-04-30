from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from langsmith import trace
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.rules.models import NotificationRule
from weather_agent.domain.rules.schedule import is_rule_due, last_cron_slot
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.db.base import ForecastPoint as ForecastPointORM
from weather_agent.infrastructure.db.base import NotificationEvent as NotificationEventORM
from weather_agent.infrastructure.db.base import RuleEvaluationRun as RuleEvaluationRunORM
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.observability.logging import (
    bound_worker_context,
    generate_correlation_id,
    get_logger,
)
from weather_agent.observability.metrics import (
    FORECAST_REFRESH_DURATION_SECONDS,
    FORECAST_REFRESH_TOTAL,
    LAST_SUCCESSFUL_FORECAST_REFRESH_TIMESTAMP_SECONDS,
    LAST_SUCCESSFUL_WORKER_CYCLE_TIMESTAMP_SECONDS,
    RULE_EVALUATION_DURATION_SECONDS,
    RULE_EVALUATION_FAILURES_TOTAL,
    RULES_EVALUATED_TOTAL,
    WORKER_CYCLE_DURATION_SECONDS,
    WORKER_CYCLES_TOTAL,
)
from weather_agent.settings import SchedulerSettings

if TYPE_CHECKING:
    from weather_agent.domain.notifications.deduplication import (
        NotificationDeduplicator,
    )
    from weather_agent.domain.notifications.events import NotificationEventService

logger = get_logger(__name__)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: int
    rule_short_id: str
    expression: str
    evaluated: bool
    result: bool | None
    error: str | None = None
    notification_candidate: bool = False
    evaluation_detail: dict[str, Any] | None = None
    dry_run: bool = False


@runtime_checkable
class ForecastFetcher(Protocol):
    async def fetch_fresh_forecast(self, location_id: int) -> int | None: ...


@runtime_checkable
class NotificationSender(Protocol):
    async def send(self, chat_id: int, thread_id: int | None, text: str) -> bool: ...


class RuleEvaluationWorker:
    def __init__(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        cel_evaluator: CELEvaluator,
        rule_service: NotificationRuleService,
        settings: SchedulerSettings,
        forecast_fetcher: ForecastFetcher | None = None,
        notification_sender: NotificationSender | None = None,
        event_service: NotificationEventService | None = None,
        deduplicator: NotificationDeduplicator | None = None,
    ) -> None:
        self._session = session
        self._forecast_repo = forecast_repo
        self._cel = cel_evaluator
        self._rule_service = rule_service
        self._settings = settings
        self._forecast_fetcher = forecast_fetcher
        self._notification_sender = notification_sender
        self._event_service = event_service
        self._deduplicator = deduplicator

    async def evaluate_rules(self, dry_run: bool = False) -> list[EvaluationResult]:
        all_rules = await self._rule_service.list_all_enabled_rules()
        now = datetime.now(UTC)
        rules = [r for r in all_rules if await self._is_rule_due(r, now)]
        async with trace(
            "evaluate_rules",
            run_type="tool",
            metadata={"rule_count": len(rules), "dry_run": dry_run},
        ):
            results: list[EvaluationResult] = []

            for rule in rules:
                rule_dry_run = dry_run or rule.dry_run
                try:
                    result = await self._evaluate_single_rule(rule, rule_dry_run)
                    RULES_EVALUATED_TOTAL.labels(
                        outcome="success" if result.evaluated else "error"
                    ).inc()
                    await self._session.commit()
                except Exception as exc:
                    RULE_EVALUATION_FAILURES_TOTAL.inc()
                    RULES_EVALUATED_TOTAL.labels(outcome="failure").inc()
                    await self._session.rollback()
                    logger.exception(
                        "rule_evaluation_failed",
                        rule_id=rule.id,
                        error_class=type(exc).__name__,
                    )
                    result = EvaluationResult(
                        rule_id=rule.id,
                        rule_short_id=rule.short_id,
                        expression=rule.expression,
                        evaluated=False,
                        result=None,
                        error=str(exc),
                    )
                results.append(result)

            if not dry_run:
                for rule, result in zip(rules, results, strict=True):
                    if result.notification_candidate:
                        if (
                            self._notification_sender is not None
                            and self._event_service is not None
                        ):
                            await self._deliver_notification(rule, result)
                    if (
                        result.notification_candidate
                        and rule.schedule is not None
                        and rule.schedule.startswith("once:")
                    ):
                        await self._rule_service.disable_rule(rule.id)
                await self._session.commit()

            return results

    async def _is_rule_due(self, rule: NotificationRule, now: datetime | None = None) -> bool:
        if now is None:
            now = datetime.now(UTC)
        if not is_rule_due(rule.schedule, now):
            return False
        if rule.schedule is not None and rule.schedule.startswith("cron:"):
            slot = last_cron_slot(rule.schedule, now)
            if slot is not None:
                stmt = select(NotificationEventORM).where(
                    NotificationEventORM.rule_id == rule.id,
                    NotificationEventORM.created_at >= slot,
                )
                result = await self._session.execute(stmt)
                if result.scalar_one_or_none() is not None:
                    return False
        return True

    async def _evaluate_single_rule(
        self, rule: NotificationRule, dry_run: bool
    ) -> EvaluationResult:
        eval_start = time.perf_counter()
        async with trace(
            "evaluate_single_rule",
            run_type="tool",
            metadata={
                "rule_id": rule.id,
                "rule_short_id": rule.short_id,
                "location_id": rule.location_id,
                "dry_run": dry_run,
            },
        ):
            import structlog

            structlog.contextvars.bind_contextvars(
                rule_id=rule.id,
                rule_short_id=rule.short_id,
                location_id=rule.location_id,
            )
            if self._forecast_fetcher is not None:
                refresh_start = time.perf_counter()
                try:
                    async with trace(
                        "forecast_refresh",
                        run_type="tool",
                        metadata={"location_id": rule.location_id},
                    ):
                        await self._forecast_fetcher.fetch_fresh_forecast(rule.location_id)
                    FORECAST_REFRESH_TOTAL.labels(outcome="success").inc()
                    LAST_SUCCESSFUL_FORECAST_REFRESH_TIMESTAMP_SECONDS.set(time.time())
                except Exception as exc:
                    FORECAST_REFRESH_TOTAL.labels(outcome="failure").inc()
                    logger.warning(
                        "forecast_fetch_failed",
                        location_id=rule.location_id,
                        error_class=type(exc).__name__,
                    )
                finally:
                    FORECAST_REFRESH_DURATION_SECONDS.observe(time.perf_counter() - refresh_start)

            data = await self._build_evaluation_data(rule.location_id)
            try:
                result = await self._finish_evaluation(rule, dry_run, data)
            finally:
                RULE_EVALUATION_DURATION_SECONDS.observe(time.perf_counter() - eval_start)
            return result

    async def _finish_evaluation(
        self,
        rule: NotificationRule,
        dry_run: bool,
        data: dict[str, Any],
    ) -> EvaluationResult:

        if data.get("points") is None or len(data["points"]) == 0:
            evaluation_detail: dict[str, Any] = {
                "rule_id": rule.id,
                "rule_short_id": rule.short_id,
                "location_id": rule.location_id,
                "snapshot_id": data.get("snapshot_id"),
                "point_count": 0,
                "error": "no_forecast_data",
            }
            await self._save_evaluation_run(
                rule, evaluated=False, result=False, evaluation_detail=evaluation_detail
            )
            return EvaluationResult(
                rule_id=rule.id,
                rule_short_id=rule.short_id,
                expression=rule.expression,
                evaluated=False,
                result=None,
                error="no_forecast_data",
                evaluation_detail=evaluation_detail,
                dry_run=dry_run,
            )

        cel_result = self._cel.evaluate(rule.expression, data)

        evaluated = cel_result.error is None
        result_value: bool | None = None
        error: str | None = None

        if cel_result.error is not None:
            error = cel_result.error
        elif isinstance(cel_result.result, bool):
            result_value = cel_result.result
        else:
            result_type = type(cel_result.result).__name__
            error = f"Expression did not return boolean, got {result_type}: {cel_result.result}"
            evaluated = False

        notification_candidate = evaluated and result_value is True

        evaluation_detail = {
            "rule_id": rule.id,
            "rule_short_id": rule.short_id,
            "location_id": rule.location_id,
            "snapshot_id": data.get("snapshot_id"),
            "point_count": len(data.get("points", [])),
            "evaluated_metrics": cel_result.evaluated_metrics,
            "evaluated_functions": cel_result.evaluated_functions,
            "expression_result": cel_result.result,
            "expression_error": cel_result.error,
        }

        if notification_candidate:
            first_point = data["points"][0] if data["points"] else {}
            last_point = data["points"][-1] if data["points"] else {}
            evaluation_detail["forecast_window_start"] = str(first_point.get("target_time", ""))
            evaluation_detail["forecast_window_end"] = str(last_point.get("target_time", ""))
            key_metrics: dict[str, float | str | None] = {}
            for metric in cel_result.evaluated_metrics:
                key_metrics[metric] = first_point.get(metric)
            evaluation_detail["key_metrics"] = key_metrics

        eval_run = await self._save_evaluation_run(
            rule,
            evaluated=evaluated,
            result=result_value is True,
            evaluation_detail=evaluation_detail,
        )

        if notification_candidate:
            evaluation_detail["evaluation_run_id"] = eval_run.id

        if dry_run and notification_candidate:
            logger.info(
                "dry_run_notification_candidate",
                rule_id=rule.id,
                rule_short_id=rule.short_id,
                expression=rule.expression,
            )

        with trace(
            "evaluation_result",
            run_type="tool",
            metadata={
                "rule_id": rule.id,
                "rule_short_id": rule.short_id,
                "evaluated": evaluated,
                "result": result_value,
                "notification_candidate": notification_candidate,
                "error": error,
                "dry_run": dry_run,
            },
        ):
            pass

        return EvaluationResult(
            rule_id=rule.id,
            rule_short_id=rule.short_id,
            expression=rule.expression,
            evaluated=evaluated,
            result=result_value,
            error=error,
            notification_candidate=notification_candidate,
            evaluation_detail=evaluation_detail,
            dry_run=dry_run,
        )

    async def _build_evaluation_data(self, location_id: int) -> dict[str, Any]:
        snapshot = await self._forecast_repo.get_latest_snapshot(str(location_id))
        if snapshot is None:
            return {"points": [], "snapshot_id": None}

        now = datetime.now(UTC)
        time_start = now - timedelta(hours=1)
        time_end = now + timedelta(days=7)

        points = await self._forecast_repo.get_points_for_snapshot(
            snapshot.id,
            start=time_start,
            end=time_end,
        )

        point_dicts: list[dict[str, Any]] = []
        for p in points:
            point_dicts.append(self._orm_point_to_dict(p))

        previous_points: list[dict[str, Any]] = []
        if snapshot.fetched_at is not None:
            prev_snapshot = await self._forecast_repo.get_previous_snapshot(
                str(location_id), before=snapshot.fetched_at
            )
            if prev_snapshot is not None:
                prev_points = await self._forecast_repo.get_points_for_snapshot(
                    prev_snapshot.id,
                    start=time_start,
                    end=time_end,
                )
                for p in prev_points:
                    previous_points.append(self._orm_point_to_dict(p))

        return {
            "points": point_dicts,
            "previous_points": previous_points,
            "snapshot_id": snapshot.id,
        }

    @staticmethod
    def _orm_point_to_dict(p: ForecastPointORM) -> dict[str, Any]:
        target_time = p.target_time
        if target_time is not None and target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=ZoneInfo("Europe/Warsaw"))
        d: dict[str, Any] = {
            "target_time": target_time,
            "fetched_at": None,
            "raw_payload": p.raw_payload,
        }
        for field_name in (
            "temperature_2m_c",
            "apparent_temperature_c",
            "precipitation_mm",
            "precipitation_probability_pct",
            "rain_mm",
            "snowfall_cm",
            "cloud_cover_pct",
            "wind_speed_10m_ms",
            "wind_gusts_10m_ms",
            "wind_direction_10m_deg",
            "pressure_msl_hpa",
            "relative_humidity_2m_pct",
            "weather_code",
        ):
            val = getattr(p, field_name, None)
            if val is not None:
                d[field_name] = val
        return d

    async def _save_evaluation_run(
        self,
        rule: NotificationRule,
        evaluated: bool,
        result: bool,
        evaluation_detail: dict[str, Any],
    ) -> RuleEvaluationRunORM:
        snapshot_id = evaluation_detail.get("snapshot_id")
        orm = RuleEvaluationRunORM(
            rule_id=rule.id,
            snapshot_id=snapshot_id,
            evaluated_at=datetime.now(UTC),
            result=result and evaluated,
            evaluation_detail=evaluation_detail,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return orm

    async def _deliver_notification(
        self,
        rule: NotificationRule,
        result: EvaluationResult,
    ) -> None:
        from weather_agent.domain.notifications.deduplication import NotificationCandidate

        assert self._event_service is not None
        assert self._notification_sender is not None

        detail = result.evaluation_detail or {}

        window_start: datetime | None = None
        window_end: datetime | None = None
        ws_raw = detail.get("forecast_window_start")
        we_raw = detail.get("forecast_window_end")
        if isinstance(ws_raw, str):
            try:
                window_start = datetime.fromisoformat(ws_raw)
            except (ValueError, TypeError):
                pass
        if isinstance(we_raw, str):
            try:
                window_end = datetime.fromisoformat(we_raw)
            except (ValueError, TypeError):
                pass

        candidate = NotificationCandidate(
            rule_id=rule.id,
            location_id=rule.location_id,
            expression=rule.expression,
            forecast_window_start=window_start,
            forecast_window_end=window_end,
            payload=detail,
            dry_run=result.dry_run,
        )

        suppressed = False
        suppress_reason: str | None = None
        if self._deduplicator is not None:
            suppressed, suppress_reason = await self._deduplicator.should_suppress(rule, candidate)

        event = await self._event_service.create_event(
            rule=rule,
            evaluation=result,
            dedupe_key=None,
            payload=detail,
        )

        if suppressed:
            await self._event_service.mark_suppressed(event.id, suppress_reason or "suppressed")
            return

        message_text = _build_forecast_message(rule, detail)
        sent = await self._notification_sender.send(
            chat_id=rule.telegram_chat_id,
            thread_id=rule.telegram_message_thread_id,
            text=message_text,
        )
        if sent:
            await self._event_service.mark_sent(event.id, message_text=message_text)
        else:
            await self._event_service.mark_suppressed(event.id, "telegram_send_failed")

    async def run_once(self) -> None:
        await self.evaluate_rules()

    async def run_loop(self) -> None:
        interval_seconds = self._settings.rule_evaluation_minutes * 60
        while True:
            with bound_worker_context(
                correlation_id=generate_correlation_id(),
            ):
                try:
                    cycle_start = time.perf_counter()
                    WORKER_CYCLES_TOTAL.inc()
                    async with trace(
                        "worker_cycle",
                        run_type="tool",
                    ):
                        await self.run_once()
                    await self._session.commit()
                    WORKER_CYCLE_DURATION_SECONDS.observe(time.perf_counter() - cycle_start)
                    LAST_SUCCESSFUL_WORKER_CYCLE_TIMESTAMP_SECONDS.set(time.time())
                except Exception as exc:
                    await self._session.rollback()
                    logger.exception(
                        "rule_evaluation_cycle_failed",
                        error_class=type(exc).__name__,
                        outcome="failure",
                    )
            await asyncio.sleep(interval_seconds)


_METRIC_LABELS: dict[str, str] = {
    "temperature_2m_c": "Temperatura",
    "apparent_temperature_c": "Temperatura odczuwalna",
    "precipitation_mm": "Opady",
    "precipitation_probability_pct": "Prawd. opadów",
    "rain_mm": "Deszcz",
    "snowfall_cm": "Śnieg",
    "cloud_cover_pct": "Zachmurzenie",
    "wind_speed_10m_ms": "Wiatr",
    "wind_gusts_10m_ms": "Porywy wiatru",
    "wind_direction_10m_deg": "Kierunek wiatru",
    "pressure_msl_hpa": "Ciśnienie",
    "relative_humidity_2m_pct": "Wilgotność",
}

_UNIT_SUFFIXES: dict[str, str] = {
    "temperature_2m_c": "°C",
    "apparent_temperature_c": "°C",
    "precipitation_mm": " mm",
    "precipitation_probability_pct": "%",
    "rain_mm": " mm",
    "snowfall_cm": " cm",
    "cloud_cover_pct": "%",
    "wind_speed_10m_ms": " m/s",
    "wind_gusts_10m_ms": " m/s",
    "wind_direction_10m_deg": "°",
    "pressure_msl_hpa": " hPa",
    "relative_humidity_2m_pct": "%",
}


def _build_forecast_message(
    rule: NotificationRule,
    evaluation_detail: dict[str, Any],
) -> str:
    lines: list[str] = []

    if rule.description:
        lines.append(f"{rule.description}")

    key_metrics = evaluation_detail.get("key_metrics")
    if isinstance(key_metrics, dict) and key_metrics:
        for metric, value in key_metrics.items():
            if isinstance(metric, str) and value is not None:
                label = _METRIC_LABELS.get(metric, metric)
                suffix = _UNIT_SUFFIXES.get(metric, "")
                try:
                    formatted = f"{float(value):.1f}{suffix}" if suffix else str(value)
                except (ValueError, TypeError):
                    formatted = str(value)
                lines.append(f"{label}: {formatted}")

    window_start = evaluation_detail.get("forecast_window_start")
    window_end = evaluation_detail.get("forecast_window_end")
    if window_start and window_end:
        lines.append(f"Okres: {window_start}  {window_end}")

    if not lines:
        lines.append("Prognoza niedostepna.")

    return "\n".join(lines)
