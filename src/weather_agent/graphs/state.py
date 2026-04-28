from __future__ import annotations

from typing import Any, TypedDict

from weather_agent.domain.cel.evaluator import CELEvaluationResult
from weather_agent.domain.date_resolver import ResolvedTimeRange
from weather_agent.domain.weather import ForecastResult, LocationRef, ObservationResult


class ConversationState(TypedDict, total=False):
    authorized_user_id: int | None
    chat_id: int
    message_thread_id: int | None
    context_key: str
    user_message: str | None
    resolved_intent: str | None
    resolved_location: LocationRef | None
    resolved_time_range: ResolvedTimeRange | None
    user_focus: str | None
    forecast_result: ForecastResult | None
    observation_result: ObservationResult | None
    pending_confirmation: dict[str, Any] | None
    cel_expression: str | None
    cel_validation_result: CELEvaluationResult | None
    answer: str | None
    error: str | None