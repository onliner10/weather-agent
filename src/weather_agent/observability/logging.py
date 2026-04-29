from __future__ import annotations

import contextlib
import logging
import re
import sys
import uuid
from collections.abc import Iterator
from typing import cast

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.contextvars import bind_contextvars, unbind_contextvars

from weather_agent.infrastructure.db.base import AuditLog

_SECRET_PATTERNS: tuple[str, ...] = (
    "token",
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "authorization",
    "auth",
    "bearer",
    "cookie",
    "session",
    "private_key",
    "access_token",
    "refresh_token",
    "webhook_secret",
    "openai_api_key",
    "anthropic_api_key",
    "langsmith_api_key",
    "database_url",
    "db_url",
)

_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d+:[A-Za-z0-9_-]+$"),  # Telegram bot token
    re.compile(r"^sk-[A-Za-z0-9]{20,}$"),  # OpenAI-style API key
    re.compile(r"^Bearer\s+.+$", re.IGNORECASE),  # Bearer token
    re.compile(r"^Basic\s+[A-Za-z0-9+/=]+$", re.IGNORECASE),  # Basic auth
)


def _redact_secrets(
    logger: logging.Logger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(pattern in key_lower for pattern in _SECRET_PATTERNS):
            event_dict[key] = "[REDACTED]"
            continue
        value = event_dict[key]
        if isinstance(value, str):
            for pat in _SECRET_VALUE_PATTERNS:
                if pat.match(value):
                    event_dict[key] = "[REDACTED]"
                    break
    return event_dict


def _rename_logger_to_logger_name(
    logger: logging.Logger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    if "logger" in event_dict and "logger_name" not in event_dict:
        event_dict["logger_name"] = event_dict.pop("logger")
    elif "logger" not in event_dict and "logger_name" not in event_dict:
        record = event_dict.get("_record")
        if record is not None:
            event_dict["logger_name"] = record.name
        elif hasattr(logger, "name"):
            event_dict["logger_name"] = logger.name
    return event_dict


def _add_service_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    if "service" not in event_dict:
        event_dict["service"] = "weather_agent"
    if "component" not in event_dict:
        event_dict["component"] = "unknown"
    return event_dict


_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    _rename_logger_to_logger_name,
    _add_service_context,
    _redact_secrets,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=_SHARED_PROCESSORS + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=_SHARED_PROCESSORS,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def generate_correlation_id() -> str:
    return str(uuid.uuid4())


class AuditLogger:
    VALID_EVENT_TYPES = frozenset(
        {
            "authorized_message",
            "unauthorized_attempt",
            "rule_created",
            "rule_updated",
            "rule_deleted",
            "notification_sent",
            "notification_suppressed",
            "weather_provider_call",
            "rule_evaluation",
            "confirmation_accepted",
            "confirmation_declined",
        }
    )

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._log = get_logger("weather_agent.audit")

    async def log_event(
        self,
        event_type: str,
        user_id: int | None = None,
        context_key: str | None = None,
        details: dict[str, object] | None = None,
        correlation_id: str | None = None,
    ) -> int:
        if event_type not in self.VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid audit event type: {event_type!r}. "
                f"Valid types: {sorted(self.VALID_EVENT_TYPES)}"
            )
        safe_details = _sanitize_details(details or {})
        row = AuditLog(
            event_type=event_type,
            user_id=user_id,
            context_key=context_key,
            details={"correlation_id": correlation_id, **safe_details}
            if correlation_id
            else safe_details,
        )
        self._session.add(row)
        await self._session.flush()
        assert row.id is not None
        self._log.info(
            "audit_event_persisted",
            event_type=event_type,
            audit_id=row.id,
            user_id=user_id,
            context_key=context_key,
            correlation_id=correlation_id,
        )
        return row.id


def _sanitize_details(details: dict[str, object]) -> dict[str, object]:
    sanitized: dict[str, object] = {}
    for key, value in details.items():
        if any(pattern in key.lower() for pattern in _SECRET_PATTERNS):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, str) and len(value) > 500:
            sanitized[key] = value[:500] + "...[TRUNCATED]"
        else:
            sanitized[key] = value
    return sanitized


def get_audit_logger(session: AsyncSession) -> AuditLogger:
    return AuditLogger(session)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    log = structlog.get_logger(name)
    if name is not None:
        log = log.bind(logger_name=name)
    return cast(structlog.stdlib.BoundLogger, log.bind())


_TELEGRAM_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "correlation_id",
        "service",
        "component",
        "chat_id",
        "message_thread_id",
        "telegram_user_id",
        "message_id",
        "reply_to_message_id",
        "context_key",
    }
)


@contextlib.contextmanager
def bound_telegram_context(
    *,
    correlation_id: str | None = None,
    chat_id: int | None = None,
    message_thread_id: int | None = None,
    telegram_user_id: int | None = None,
    message_id: int | None = None,
    reply_to_message_id: int | None = None,
    context_key: str | None = None,
) -> Iterator[str]:
    cid = correlation_id or generate_correlation_id()
    bind_contextvars(
        correlation_id=cid,
        service="bot",
        component="telegram_handler",
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        telegram_user_id=telegram_user_id,
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
        context_key=context_key,
    )
    try:
        yield cid
    finally:
        unbind_contextvars(*_TELEGRAM_CONTEXT_KEYS)


_WORKER_CONTEXT_KEYS: frozenset[str] = frozenset(
    {
        "correlation_id",
        "service",
        "component",
    }
)


@contextlib.contextmanager
def bound_worker_context(
    *,
    correlation_id: str | None = None,
) -> Iterator[str]:
    cid = correlation_id or generate_correlation_id()
    bind_contextvars(
        correlation_id=cid,
        service="worker",
        component="rule_evaluator",
    )
    try:
        yield cid
    finally:
        unbind_contextvars(*_WORKER_CONTEXT_KEYS)
