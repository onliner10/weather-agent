from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.notifications.deduplication import DedupeKey
from weather_agent.domain.rules.models import (
    NotificationEvent,
    NotificationRule,
    ShortIdCollisionError,
)
from weather_agent.domain.rules.service import _MAX_COLLISION_RETRIES
from weather_agent.domain.rules.short_id_generator import (
    generate_short_id,
    strip_hash_prefix,
)
from weather_agent.infrastructure.db.base import (
    NotificationEvent as NotificationEventORM,
)
from weather_agent.infrastructure.db.base import (
    NotificationRule as NotificationRuleORM,
)
from weather_agent.infrastructure.db.base import (
    RuleEvaluationRun as RuleEvaluationRunORM,
)
from weather_agent.infrastructure.worker.rule_evaluator import EvaluationResult
from weather_agent.observability.logging import AuditLogger


class EventNotFoundError(Exception):
    def __init__(self, short_id: str | None = None, event_id: int | None = None) -> None:
        if short_id is not None:
            msg = f"Event with short_id '{short_id}' not found"
        elif event_id is not None:
            msg = f"Event {event_id} not found"
        else:
            msg = "Event not found"
        self.short_id = short_id
        self.event_id = event_id
        super().__init__(msg)


def _orm_to_domain(orm: NotificationEventORM) -> NotificationEvent:
    return NotificationEvent(
        id=orm.id,
        short_id=orm.short_id,
        rule_id=orm.rule_id,
        evaluation_run_id=orm.evaluation_run_id,
        telegram_chat_id=orm.telegram_chat_id,
        telegram_message_thread_id=orm.telegram_message_thread_id,
        sent_at=orm.sent_at,
        suppressed=orm.suppressed,
        suppress_reason=orm.suppress_reason,
        payload_hash=orm.payload_hash,
        message_text=orm.message_text,
        created_at=orm.created_at,
    )


class NotificationEventService:
    def __init__(self, session: AsyncSession, audit_logger: AuditLogger) -> None:
        self._session = session
        self._audit = audit_logger

    async def _generate_unique_short_id(self) -> str:
        for _ in range(_MAX_COLLISION_RETRIES):
            short_id = generate_short_id("E")
            stmt = select(NotificationEventORM).where(NotificationEventORM.short_id == short_id)
            result = await self._session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing is None:
                return short_id
        raise ShortIdCollisionError("")

    async def create_event(
        self,
        rule: NotificationRule,
        evaluation: EvaluationResult,
        dedupe_key: DedupeKey | None,
        payload: dict[str, object],
    ) -> NotificationEvent:
        short_id = await self._generate_unique_short_id()
        now = datetime.now(UTC)

        evaluation_run_id: int | None = None
        if evaluation.evaluation_detail is not None:
            raw_run_id = evaluation.evaluation_detail.get("evaluation_run_id")
            if isinstance(raw_run_id, int) and raw_run_id > 0:
                evaluation_run_id = raw_run_id

        payload_hash: str | None = None
        if dedupe_key is not None:
            payload_hash = hashlib.sha256(dedupe_key.dedupe_key.encode()).hexdigest()
        elif payload:
            canonical = json.dumps(payload, sort_keys=True, default=str)
            payload_hash = hashlib.sha256(canonical.encode()).hexdigest()

        orm = NotificationEventORM(
            short_id=short_id,
            rule_id=rule.id,
            evaluation_run_id=evaluation_run_id,
            telegram_chat_id=rule.telegram_chat_id,
            telegram_message_thread_id=rule.telegram_message_thread_id,
            sent_at=None,
            suppressed=False,
            suppress_reason=None,
            payload_hash=payload_hash,
            message_text=None,
            created_at=now,
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)

        await self._audit.log_event(
            event_type="notification_sent",
            user_id=rule.user_id,
            context_key=str(rule.telegram_chat_id),
            details={
                "event_short_id": short_id,
                "rule_short_id": rule.short_id,
                "rule_id": rule.id,
                "expression": rule.expression,
                "notification_candidate": evaluation.notification_candidate,
            },
        )

        return _orm_to_domain(orm)

    async def mark_sent(
        self,
        event_id: int,
        message_text: str | None = None,
    ) -> NotificationEvent:
        orm = await self._session.get(NotificationEventORM, event_id)
        if orm is None:
            raise EventNotFoundError(event_id=event_id)
        orm.sent_at = datetime.now(UTC)
        if message_text is not None:
            orm.message_text = message_text
        await self._session.flush()
        await self._session.refresh(orm)

        await self._audit.log_event(
            event_type="notification_sent",
            details={
                "event_id": orm.id,
                "event_short_id": orm.short_id,
                "action": "mark_sent",
            },
        )

        return _orm_to_domain(orm)

    async def mark_suppressed(self, event_id: int, reason: str) -> NotificationEvent:
        orm = await self._session.get(NotificationEventORM, event_id)
        if orm is None:
            raise EventNotFoundError(event_id=event_id)
        orm.suppressed = True
        orm.suppress_reason = reason
        await self._session.flush()
        await self._session.refresh(orm)

        await self._audit.log_event(
            event_type="notification_suppressed",
            details={
                "event_id": orm.id,
                "event_short_id": orm.short_id,
                "suppress_reason": reason,
            },
        )

        return _orm_to_domain(orm)

    async def get_event(
        self,
        short_id: str | None = None,
        event_id: int | None = None,
    ) -> NotificationEvent | None:
        if event_id is not None:
            orm = await self._session.get(NotificationEventORM, event_id)
            if orm is None:
                return None
            return _orm_to_domain(orm)
        if short_id is not None:
            clean_id = strip_hash_prefix(short_id)
            stmt = select(NotificationEventORM).where(NotificationEventORM.short_id == clean_id)
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            if orm is None:
                return None
            return _orm_to_domain(orm)
        return None

    async def list_events_for_rule(self, rule_id: int, limit: int = 10) -> list[NotificationEvent]:
        stmt = (
            select(NotificationEventORM)
            .where(NotificationEventORM.rule_id == rule_id)
            .order_by(NotificationEventORM.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [_orm_to_domain(r) for r in result.scalars().all()]


_METRIC_LABELS: dict[str, str] = {
    "temperature_2m_c": "temperatura",
    "apparent_temperature_c": "temperatura odczuwalna",
    "precipitation_mm": "opady",
    "precipitation_probability_pct": "prawdopodobieństwo opadów",
    "rain_mm": "deszcz",
    "snowfall_cm": "opady śniegu",
    "cloud_cover_pct": "zachmurzenie",
    "wind_speed_10m_ms": "prędkość wiatru",
    "wind_gusts_10m_ms": "porywy wiatru",
    "wind_direction_10m_deg": "kierunek wiatru",
    "pressure_msl_hpa": "ciśnienie",
    "relative_humidity_2m_pct": "wilgotność",
    "weather_code": "kod pogody",
}

_FUNCTION_LABELS: dict[str, str] = {
    "max": "maksimum",
    "min": "minimum",
    "avg": "średnia",
    "sum": "suma",
    "median": "mediana",
    "stddev": "odchylenie standardowe",
    "pctl": "percentyl",
    "delta": "zmiana",
    "abs_delta": "bezwzględna zmiana",
    "rate_of_change": "tempo zmiany",
    "forecast_delta": "zmiana prognozy",
    "weekend": "weekend",
    "now": "teraz",
    "today": "dzisiaj",
    "tomorrow": "jutro",
    "next_hours": "kolejne godziny",
    "between": "między",
    "duration_where": "czas trwania warunku",
    "count_where": "liczba wystąpień",
    "any": "którykolwiek",
    "all": "wszystkie",
    "minutes": "minuty",
    "hours": "godziny",
    "date_range": "zakres dat",
    "previous_snapshot": "poprzednia prognoza",
    "abs": "wartość bezwzględna",
    "round": "zaokrąglenie",
    "clamp": "ograniczenie",
}

_UNIT_SUFFIXES: dict[str, str] = {
    "temperature_2m_c": "°C",
    "apparent_temperature_c": "°C",
    "precipitation_mm": "mm",
    "precipitation_probability_pct": "%",
    "rain_mm": "mm",
    "snowfall_cm": "cm",
    "cloud_cover_pct": "%",
    "wind_speed_10m_ms": "m/s",
    "wind_gusts_10m_ms": "m/s",
    "wind_direction_10m_deg": "°",
    "pressure_msl_hpa": "hPa",
    "relative_humidity_2m_pct": "%",
}


def _label_metric(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric)


def _label_function(func: str) -> str:
    return _FUNCTION_LABELS.get(func, func)


def _format_value(metric: str, value: object) -> str:
    if value is None:
        return "brak danych"
    unit = _UNIT_SUFFIXES.get(metric, "")
    if unit:
        return f"{value} {unit}"
    return str(value)


class ExplanationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def explain_notification(self, event_short_id: str) -> str:
        clean_id = strip_hash_prefix(event_short_id)

        event_stmt = select(NotificationEventORM).where(NotificationEventORM.short_id == clean_id)
        event_result = await self._session.execute(event_stmt)
        event_orm = event_result.scalar_one_or_none()
        if event_orm is None:
            raise EventNotFoundError(short_id=event_short_id)

        rule_orm: NotificationRuleORM | None = None
        if event_orm.rule_id is not None:
            rule_orm = await self._session.get(NotificationRuleORM, event_orm.rule_id)

        evaluation_detail: dict[str, object] = {}
        if event_orm.evaluation_run_id is not None:
            eval_run = await self._session.get(RuleEvaluationRunORM, event_orm.evaluation_run_id)
            if eval_run is not None:
                evaluation_detail = eval_run.evaluation_detail or {}

        rule_label = f"#{rule_orm.short_id}" if rule_orm else "(reguła usunięta)"
        expression = rule_orm.expression if rule_orm else "(wyrażenie niedostępne)"

        status = "wstrzymane" if event_orm.suppressed else "wysłane"
        if event_orm.suppress_reason:
            status += f" — powód: {event_orm.suppress_reason}"

        parts: list[str] = [
            f"Otrzymałeś powiadomienie #{event_orm.short_id} ({status}), "
            f"ponieważ reguła {rule_label} ({expression}) została spełniona."
        ]

        evaluated_metrics = evaluation_detail.get("evaluated_metrics")
        if isinstance(evaluated_metrics, list) and evaluated_metrics:
            metric_labels = []
            for m in evaluated_metrics:
                if isinstance(m, str):
                    metric_labels.append(_label_metric(m))
            if metric_labels:
                parts.append(f"Mierzone parametry: {', '.join(metric_labels)}.")

        key_metrics = evaluation_detail.get("key_metrics")
        if isinstance(key_metrics, dict) and key_metrics:
            value_parts = []
            for m, v in key_metrics.items():
                if isinstance(m, str):
                    value_parts.append(f"{_label_metric(m)}: {_format_value(m, v)}")
            if value_parts:
                parts.append("Wartość: " + ", ".join(value_parts) + ".")

        evaluated_functions = evaluation_detail.get("evaluated_functions")
        if isinstance(evaluated_functions, list) and evaluated_functions:
            func_labels = []
            for f in evaluated_functions:
                if isinstance(f, str):
                    func_labels.append(_label_function(f))
            if func_labels:
                parts.append(f"Funkcje: {', '.join(func_labels)}.")

        window_start = evaluation_detail.get("forecast_window_start")
        window_end = evaluation_detail.get("forecast_window_end")
        if window_start and window_end:
            parts.append(f"Okno prognozy: {window_start} – {window_end}.")

        snapshot_id = evaluation_detail.get("snapshot_id")
        if snapshot_id is not None:
            parts.append(f"Zrzut prognozy: {snapshot_id}.")

        return " ".join(parts)
