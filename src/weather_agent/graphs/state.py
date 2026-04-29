from __future__ import annotations

from typing import Any, TypedDict

from weather_agent.domain.cel.evaluator import CELEvaluationResult
from weather_agent.domain.date_resolver import ResolvedTimeRange
from weather_agent.domain.weather import ForecastResult, LocationRef, ObservationResult


class TurnRecord(TypedDict, total=False):
    message_id: int | None
    role: str
    text: str | None
    answer_summary: str | None
    resolved_location: dict[str, Any] | None
    resolved_time_range: dict[str, Any] | None
    user_focus: str | None
    timestamp: str | None


class ConversationState(TypedDict, total=False):
    authorized_user_id: int | None
    chat_id: int
    message_thread_id: int | None
    context_key: str
    user_message: str | None
    message_id: int | None
    reply_to_message_id: int | None
    reply_to_message_text: str | None
    reply_anchor: dict[str, Any] | None
    recent_context: list[dict[str, Any]] | None
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
    bot_message_id: int | None