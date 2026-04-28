from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.rules.models import NotificationRule
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.infrastructure.db.base import ForecastPoint as ForecastPointORM
from weather_agent.infrastructure.db.base import RuleEvaluationRun as RuleEvaluationRunORM
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.settings import SchedulerSettings

logger = logging.getLogger(__name__)


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
    async def fetch_fresh_forecast(self, location_id: int) -> int | None:
        ...


class RuleEvaluationWorker:
    def __init__(
        self,
        session: AsyncSession,
        forecast_repo: ForecastRepository,
        cel_evaluator: CELEvaluator,
        rule_service: NotificationRuleService,
        settings: SchedulerSettings,
        forecast_fetcher: ForecastFetcher | None = None,
    ) -> None:
        self._session = session
        self._forecast_repo = forecast_repo
        self._cel = cel_evaluator
        self._rule_service = rule_service
        self._settings = settings
        self._forecast_fetcher = forecast_fetcher

    async def evaluate_rules(self, dry_run: bool = False) -> list[EvaluationResult]:
        rules = await self._rule_service.list_all_enabled_rules()
        results: list[EvaluationResult] = []

        for rule in rules:
            rule_dry_run = dry_run or rule.dry_run
            result = await self._evaluate_single_rule(rule, rule_dry_run)
            results.append(result)

        return results

    async def _evaluate_single_rule(
        self, rule: NotificationRule, dry_run: bool
    ) -> EvaluationResult:
        if self._forecast_fetcher is not None:
            try:
                await self._forecast_fetcher.fetch_fresh_forecast(rule.location_id)
            except Exception as exc:
                logger.warning(
                    "forecast fetch failed for location_id=%d: %s",
                    rule.location_id,
                    exc,
                )

        data = await self._build_evaluation_data(rule.location_id)

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
            error = (
                f"Expression did not return boolean, "
                f"got {result_type}: {cel_result.result}"
            )
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
            evaluation_detail["forecast_window_start"] = str(
                first_point.get("target_time", "")
            )
            evaluation_detail["forecast_window_end"] = str(
                last_point.get("target_time", "")
            )
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
                "dry-run: notification candidate for rule #%s (id=%d) expression=%r",
                rule.short_id,
                rule.id,
                rule.expression,
            )

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
        points = await self._forecast_repo.get_points_by_time_range(
            str(location_id),
            start=now - timedelta(hours=1),
            end=now + timedelta(days=7),
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
                prev_points = await self._forecast_repo.get_points_by_time_range(
                    str(location_id),
                    start=now - timedelta(hours=1),
                    end=now + timedelta(days=7),
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

    async def run_once(self) -> None:
        await self.evaluate_rules()

    async def run_loop(self) -> None:
        interval_seconds = self._settings.rule_evaluation_minutes * 60
        while True:
            try:
                await self.run_once()
            except Exception:
                logger.exception("rule evaluation cycle failed")
            await asyncio.sleep(interval_seconds)