from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from langsmith import trace
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.domain.rules.models import NotificationRule
from weather_agent.domain.rules.notification_context import notification_context_fingerprint
from weather_agent.domain.rules.schedule import is_rule_due, last_cron_slot
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.time import WARSAW_TZ, ensure_utc
from weather_agent.infrastructure.db.base import ForecastPoint as ForecastPointORM
from weather_agent.infrastructure.db.base import NotificationEvent as NotificationEventORM
from weather_agent.infrastructure.db.base import RuleEvaluationRun as RuleEvaluationRunORM
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.infrastructure.worker.coordination import WorkerCoordinator
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
_FORECAST_REFRESH_FAILED = "forecast_refresh_failed"


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


@runtime_checkable
class NotificationContentGenerator(Protocol):
    async def generate(
        self,
        rule: NotificationRule,
        evaluation_detail: dict[str, Any],
    ) -> str | None: ...


class RuleEvaluationWorker:
    def __init__(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        rule_expression_evaluator: RuleExpressionEvaluator,
        rule_service: NotificationRuleService,
        settings: SchedulerSettings,
        forecast_fetcher: ForecastFetcher | None = None,
        notification_sender: NotificationSender | None = None,
        event_service: NotificationEventService | None = None,
        deduplicator: NotificationDeduplicator | None = None,
        coordinator: WorkerCoordinator | None = None,
        notification_content_generator: NotificationContentGenerator | None = None,
    ) -> None:
        self._session = session
        self._forecast_repo = forecast_repo
        self._rule_expression = rule_expression_evaluator
        self._rule_service = rule_service
        self._settings = settings
        self._forecast_fetcher = forecast_fetcher
        self._notification_sender = notification_sender
        self._event_service = event_service
        self._deduplicator = deduplicator
        self._coordinator = coordinator or WorkerCoordinator(session)
        self._notification_content_generator = notification_content_generator

    async def evaluate_rules(self, dry_run: bool = False) -> list[EvaluationResult]:
        rules = await self._claim_due_rules()
        if not rules:
            return []

        return await self._evaluate_due_rules(rules=rules, dry_run=dry_run)

    async def _claim_due_rules(self) -> list[NotificationRule]:
        worker_lock = await self._coordinator.acquire()
        if not worker_lock.acquired:
            logger.info("rule_evaluation_worker_lock_busy")
            return []

        try:
            all_rules = await self._rule_service.list_all_enabled_rules()
            now = datetime.now(UTC)
            return [r for r in all_rules if await self._is_rule_due(r, now)]
        finally:
            await self._coordinator.release()

    async def _evaluate_due_rules(
        self,
        rules: list[NotificationRule],
        dry_run: bool = False,
    ) -> list[EvaluationResult]:
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
                delivered_scheduled_keys: set[str] = set()
                for rule, result in zip(rules, results, strict=True):
                    if result.notification_candidate:
                        scheduled_key = _scheduled_notification_key(rule)
                        if scheduled_key is not None:
                            if scheduled_key in delivered_scheduled_keys:
                                if rule.schedule is not None and rule.schedule.startswith("once:"):
                                    await self._rule_service.disable_rule(rule.id)
                                continue
                            delivered_scheduled_keys.add(scheduled_key)
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
                    NotificationEventORM.delivery_status.in_(("sent", "suppressed")),
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
            try:
                fresh_snapshot_id: int | None = None
                if self._forecast_fetcher is not None:
                    refresh_failed = False
                    refresh_start = time.perf_counter()
                    try:
                        async with trace(
                            "forecast_refresh",
                            run_type="tool",
                            metadata={"location_id": rule.location_id},
                        ):
                            fresh_snapshot_id = await self._forecast_fetcher.fetch_fresh_forecast(
                                rule.location_id
                            )
                    except Exception as exc:
                        refresh_failed = True
                        FORECAST_REFRESH_TOTAL.labels(outcome="failure").inc()
                        logger.warning(
                            "forecast_fetch_failed",
                            location_id=rule.location_id,
                            error_class=type(exc).__name__,
                        )
                    finally:
                        FORECAST_REFRESH_DURATION_SECONDS.observe(
                            time.perf_counter() - refresh_start
                        )
                    if fresh_snapshot_id is None:
                        if not refresh_failed:
                            FORECAST_REFRESH_TOTAL.labels(outcome="failure").inc()
                            logger.warning(
                                "forecast_fetch_failed",
                                location_id=rule.location_id,
                                error_class="NoSnapshot",
                            )
                        return await self._finish_forecast_refresh_failed(rule, dry_run)
                    FORECAST_REFRESH_TOTAL.labels(outcome="success").inc()
                    LAST_SUCCESSFUL_FORECAST_REFRESH_TIMESTAMP_SECONDS.set(time.time())

                data = await self._build_evaluation_data(rule.location_id, fresh_snapshot_id)
                return await self._finish_evaluation(rule, dry_run, data)
            finally:
                RULE_EVALUATION_DURATION_SECONDS.observe(time.perf_counter() - eval_start)

    async def _finish_forecast_refresh_failed(
        self,
        rule: NotificationRule,
        dry_run: bool,
    ) -> EvaluationResult:
        evaluation_detail: dict[str, Any] = {
            "rule_id": rule.id,
            "rule_short_id": rule.short_id,
            "location_id": rule.location_id,
            "snapshot_id": None,
            "point_count": 0,
            "error": _FORECAST_REFRESH_FAILED,
        }
        await self._save_evaluation_run(
            rule,
            evaluated=False,
            result=False,
            evaluation_detail=evaluation_detail,
        )
        return EvaluationResult(
            rule_id=rule.id,
            rule_short_id=rule.short_id,
            expression=rule.expression,
            evaluated=False,
            result=None,
            error=_FORECAST_REFRESH_FAILED,
            notification_candidate=False,
            evaluation_detail=evaluation_detail,
            dry_run=dry_run,
        )

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

        rule_expression_result = self._rule_expression.evaluate(rule.expression, data)

        evaluated = rule_expression_result.error is None
        result_value: bool | None = None
        error: str | None = None

        if rule_expression_result.error is not None:
            error = rule_expression_result.error
        elif isinstance(rule_expression_result.result, bool):
            result_value = rule_expression_result.result
        else:
            result_type = type(rule_expression_result.result).__name__
            error = (
                "Expression did not return boolean, "
                f"got {result_type}: {rule_expression_result.result}"
            )
            evaluated = False

        notification_candidate = evaluated and result_value is True

        evaluation_detail = {
            "rule_id": rule.id,
            "rule_short_id": rule.short_id,
            "location_id": rule.location_id,
            "snapshot_id": data.get("snapshot_id"),
            "point_count": len(data.get("points", [])),
            "evaluated_metrics": rule_expression_result.evaluated_metrics,
            "evaluated_functions": rule_expression_result.evaluated_functions,
            "expression_result": rule_expression_result.result,
            "expression_error": rule_expression_result.error,
        }

        if notification_candidate:
            first_point = data["points"][0] if data["points"] else {}
            last_point = data["points"][-1] if data["points"] else {}
            evaluation_detail["forecast_window_start"] = str(first_point.get("target_time", ""))
            evaluation_detail["forecast_window_end"] = str(last_point.get("target_time", ""))
            evaluation_detail["forecast_points"] = _notification_forecast_points(data["points"])
            key_metrics: dict[str, float | str | None] = {}
            for metric in rule_expression_result.evaluated_metrics:
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

    async def _build_evaluation_data(
        self,
        location_id: int,
        snapshot_id: int | None = None,
    ) -> dict[str, Any]:
        if snapshot_id is None:
            snapshot = await self._forecast_repo.get_latest_snapshot(str(location_id))
        else:
            snapshot = await self._forecast_repo.get_snapshot(snapshot_id)
        if snapshot is None:
            return {"points": [], "snapshot_id": None}
        if snapshot.location_id != location_id:
            return {"points": [], "snapshot_id": snapshot.id}

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
        if target_time is not None:
            target_time = ensure_utc(target_time)
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
        from weather_agent.domain.notifications.deduplication import (
            NotificationCandidate,
            compute_dedupe_key,
        )

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

        dedupe_key = compute_dedupe_key(
            rule_id=rule.id,
            location_id=rule.location_id,
            expression=rule.expression,
            window_start=window_start,
            window_end=window_end,
        )
        event, created = await self._event_service.create_event_once(
            rule=rule,
            evaluation=result,
            dedupe_key=dedupe_key,
            payload=detail,
        )
        if not created:
            return

        if suppressed:
            await self._event_service.mark_suppressed(event.id, suppress_reason or "suppressed")
            return

        message_text = await self._build_notification_message(rule, detail)
        sent = await self._notification_sender.send(
            chat_id=rule.telegram_chat_id,
            thread_id=rule.telegram_message_thread_id,
            text=message_text,
        )
        if sent:
            await self._event_service.mark_sent(event.id, message_text=message_text)
        else:
            await self._event_service.mark_suppressed(event.id, "telegram_send_failed")

    async def _build_notification_message(
        self,
        rule: NotificationRule,
        detail: dict[str, Any],
    ) -> str:
        if rule.schedule is not None:
            if (
                rule.notification_context is not None
                and self._notification_content_generator is not None
            ):
                generated = await self._notification_content_generator.generate(rule, detail)
                if generated is not None:
                    return generated
            return _build_scheduled_fallback_message(rule, detail)
        return _build_forecast_message(rule, detail)

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


def _notification_forecast_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for point in points[:24]:
        target_time = point.get("target_time")
        if isinstance(target_time, datetime):
            target_text = target_time.astimezone(WARSAW_TZ).strftime("%Y-%m-%d %H:%M")
        else:
            target_text = str(target_time)
        compact: dict[str, Any] = {"time": target_text}
        for metric in _METRIC_LABELS:
            if metric in point:
                compact[metric] = point[metric]
        output.append(compact)
    return output


def _scheduled_notification_key(rule: NotificationRule) -> str | None:
    if rule.schedule is None:
        return None
    return "|".join(
        [
            str(rule.user_id),
            str(rule.telegram_chat_id),
            str(rule.telegram_message_thread_id or ""),
            str(rule.location_id),
            rule.expression,
            rule.schedule,
            notification_context_fingerprint(rule.notification_context),
        ]
    )


def _numeric_values(points: list[dict[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for point in points:
        raw = point.get(metric)
        if isinstance(raw, (int, float)):
            values.append(float(raw))
    return values


def _range_text(values: list[float], suffix: str) -> str | None:
    if not values:
        return None
    return f"{min(values):.1f}-{max(values):.1f}{suffix}"


def _build_scheduled_fallback_message(
    rule: NotificationRule,
    evaluation_detail: dict[str, Any],
) -> str:
    context = rule.notification_context
    location = context.location_name if context is not None else None
    header = context.human_request if context is not None else rule.description
    lines = [header or f"Aktualna prognoza pogody{f' dla {location}' if location else ''}."]

    points_raw = evaluation_detail.get("forecast_points")
    points = [p for p in points_raw if isinstance(p, dict)] if isinstance(points_raw, list) else []
    if points:
        first_time = str(points[0].get("time", ""))
        last_time = str(points[-1].get("time", ""))
        if first_time and last_time:
            lines.append(f"Okres: {first_time} - {last_time}.")

        temp = _range_text(_numeric_values(points, "temperature_2m_c"), "°C")
        wind = _range_text(_numeric_values(points, "wind_speed_10m_ms"), " m/s")
        gusts = _numeric_values(points, "wind_gusts_10m_ms")
        precipitation = sum(_numeric_values(points, "precipitation_mm"))

        details: list[str] = []
        if temp is not None:
            details.append(f"temperatura {temp}")
        if wind is not None:
            details.append(f"wiatr {wind}")
        if gusts:
            details.append(f"porywy do {max(gusts):.1f} m/s")
        details.append(f"opady {precipitation:.1f} mm")
        lines.append(", ".join(details) + ".")

    return "\n".join(lines)


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
