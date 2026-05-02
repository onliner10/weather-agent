from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

from prometheus_client import Counter, Gauge, Histogram

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Bot / Conversation runtime
# ---------------------------------------------------------------------------

TELEGRAM_MESSAGES_TOTAL = Counter(
    "weather_agent_telegram_messages_total",
    "Total number of Telegram messages received",
)
AUTHORIZATION_FAILURES_TOTAL = Counter(
    "weather_agent_authorization_failures_total",
    "Total number of authorization failures",
)
CONVERSATION_TURNS_TOTAL = Counter(
    "weather_agent_conversation_turns_total",
    "Total number of conversation turns processed",
)
CONVERSATION_TURN_DURATION_SECONDS = Histogram(
    "weather_agent_conversation_turn_duration_seconds",
    "Duration of conversation turn processing in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
CONVERSATION_FAILURES_TOTAL = Counter(
    "weather_agent_conversation_failures_total",
    "Total number of conversation turn failures",
)
REPLY_SEND_TOTAL = Counter(
    "weather_agent_reply_send_total",
    "Total number of reply send attempts",
    ["outcome"],
)
REPLY_SEND_DURATION_SECONDS = Histogram(
    "weather_agent_reply_send_duration_seconds",
    "Duration of reply sending in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
REPLY_CONTEXT_HITS_TOTAL = Counter(
    "weather_agent_reply_context_hits_total",
    "Total number of reply context hits",
    ["source"],
)
TOOL_CALLS_TOTAL = Counter(
    "weather_agent_tool_calls_total",
    "Total number of tool calls",
    ["tool"],
)
TOOL_CALL_DURATION_SECONDS = Histogram(
    "weather_agent_tool_call_duration_seconds",
    "Duration of tool calls in seconds",
    ["tool"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)


@contextmanager
def observe_tool_call(tool_name: str) -> Iterator[None]:
    """Record call count and elapsed time for one LangChain tool invocation."""
    TOOL_CALLS_TOTAL.labels(tool=tool_name).inc()
    start = time.perf_counter()
    try:
        yield
    finally:
        TOOL_CALL_DURATION_SECONDS.labels(tool=tool_name).observe(time.perf_counter() - start)


LLM_REQUESTS_TOTAL = Counter(
    "weather_agent_llm_requests_total",
    "Total number of LLM requests",
    ["outcome"],
)
LLM_REQUEST_DURATION_SECONDS = Histogram(
    "weather_agent_llm_request_duration_seconds",
    "Duration of LLM requests in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
GEOCODE_REQUESTS_TOTAL = Counter(
    "weather_agent_geocode_requests_total",
    "Total number of geocode requests",
    ["outcome"],
)
GEOCODE_DURATION_SECONDS = Histogram(
    "weather_agent_geocode_duration_seconds",
    "Duration of geocode requests in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
PROVIDER_REQUESTS_TOTAL = Counter(
    "weather_agent_provider_requests_total",
    "Total number of provider requests",
    ["provider", "outcome"],
)
PROVIDER_REQUEST_DURATION_SECONDS = Histogram(
    "weather_agent_provider_request_duration_seconds",
    "Duration of provider requests in seconds",
    ["provider"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

WORKER_CYCLES_TOTAL = Counter(
    "weather_agent_worker_cycles_total",
    "Total number of worker cycles",
)
WORKER_CYCLE_DURATION_SECONDS = Histogram(
    "weather_agent_worker_cycle_duration_seconds",
    "Duration of worker cycles in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
RULES_EVALUATED_TOTAL = Counter(
    "weather_agent_rules_evaluated_total",
    "Total number of rules evaluated",
    ["outcome"],
)
RULE_EVALUATION_DURATION_SECONDS = Histogram(
    "weather_agent_rule_evaluation_duration_seconds",
    "Duration of rule evaluations in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
RULE_EVALUATION_FAILURES_TOTAL = Counter(
    "weather_agent_rule_evaluation_failures_total",
    "Total number of rule evaluation failures",
)
NOTIFICATIONS_TOTAL = Counter(
    "weather_agent_notifications_total",
    "Total number of notifications",
    ["type"],
)
NOTIFICATION_SEND_DURATION_SECONDS = Histogram(
    "weather_agent_notification_send_duration_seconds",
    "Duration of notification sends in seconds",
    ["type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
NOTIFICATION_FAILURES_TOTAL = Counter(
    "weather_agent_notification_failures_total",
    "Total number of notification failures",
    ["type"],
)
FORECAST_REFRESH_TOTAL = Counter(
    "weather_agent_forecast_refresh_total",
    "Total number of forecast refreshes",
    ["outcome"],
)
FORECAST_REFRESH_DURATION_SECONDS = Histogram(
    "weather_agent_forecast_refresh_duration_seconds",
    "Duration of forecast refreshes in seconds",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)
LAST_SUCCESSFUL_WORKER_CYCLE_TIMESTAMP_SECONDS = Gauge(
    "weather_agent_last_successful_worker_cycle_timestamp_seconds",
    "Timestamp of the last successful worker cycle",
)
LAST_SUCCESSFUL_FORECAST_REFRESH_TIMESTAMP_SECONDS = Gauge(
    "weather_agent_last_successful_forecast_refresh_timestamp_seconds",
    "Timestamp of the last successful forecast refresh",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_value(metric: Counter | Gauge) -> float:
    """Return the current value of a single-label-less metric."""
    return cast(float, metric._value.get())


def _get_labeled_value(metric: Counter, **labels: str) -> float:
    """Return the current value of a labeled counter."""
    return cast(float, metric.labels(**labels)._value.get())


def _count_histogram_observations(metric: Histogram, **labels: str) -> float:
    """Return the current observation count of a histogram."""
    histogram = metric.labels(**labels) if labels else metric
    collected = next(iter(histogram.collect()), None)
    if collected is None:
        return 0.0
    for sample in collected.samples:
        if sample.name.endswith("_count"):
            return float(sample.value)
    return 0.0
