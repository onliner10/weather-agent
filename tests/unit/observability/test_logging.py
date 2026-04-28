from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from io import StringIO

import pytest
import pytest_asyncio
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.infrastructure.db.base import AuditLog, Base
from weather_agent.observability.logging import (
    AuditLogger,
    _redact_secrets,
    _rename_logger_to_logger_name,
    configure_logging,
    generate_correlation_id,
    get_audit_logger,
    get_logger,
)


def _setup_test_logging(output: StringIO) -> None:
    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            _rename_logger_to_logger_name,
            _redact_secrets,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=output),
        cache_logger_on_first_use=False,
    )


def _parse_log_output(output: StringIO) -> list[dict[str, object]]:
    output.seek(0)
    content = output.read().strip()
    if not content:
        return []
    return [json.loads(line) for line in content.split("\n") if line.strip()]


class TestStructuredLogging:
    def setup_method(self) -> None:
        self._output = StringIO()
        _setup_test_logging(self._output)

    def teardown_method(self) -> None:
        structlog.reset_defaults()
        self._output.close()

    def test_log_output_is_json(self) -> None:
        log = structlog.get_logger("test")
        log.info("hello")
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        assert entries[0]["event"] == "hello"

    def test_log_includes_timestamp(self) -> None:
        log = structlog.get_logger("test")
        log.info("test_event")
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        assert "timestamp" in entries[0]

    def test_log_includes_level(self) -> None:
        log = structlog.get_logger("test")
        log.info("test_event")
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        assert entries[0]["level"] == "info"

    def test_log_includes_event_message(self) -> None:
        log = structlog.get_logger("test")
        log.info("something_happened", user_id=42)
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        assert entries[0]["event"] == "something_happened"
        assert entries[0]["user_id"] == 42

    def test_correlation_id_in_log_context(self) -> None:
        corr_id = generate_correlation_id()
        log = structlog.get_logger("test").bind(correlation_id=corr_id)
        log.info("with_correlation")
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        assert entries[0]["correlation_id"] == corr_id

    def test_no_secrets_in_log_output(self) -> None:
        log = structlog.get_logger("test")
        log.info(
            "login_attempt",
            api_key="sk-secret-123",
            bot_token="12345:secret",
            password="hunter2",
            authorization="Bearer xyz",
            cookie="session=abc",
        )
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["api_key"] == "[REDACTED]"
        assert entry["bot_token"] == "[REDACTED]"
        assert entry["password"] == "[REDACTED]"
        assert entry["authorization"] == "[REDACTED]"
        assert entry["cookie"] == "[REDACTED]"

    def test_non_secret_fields_preserved(self) -> None:
        log = structlog.get_logger("test")
        log.info("message_received", user_id=42, chat_id=123, text="hello")
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        entry = entries[0]
        assert entry["user_id"] == 42
        assert entry["chat_id"] == 123
        assert entry["text"] == "hello"

    def test_logger_name_from_bound_log(self) -> None:
        log = get_logger("weather_agent.module_a")
        log.info("named_event")
        entries = _parse_log_output(self._output)
        assert len(entries) == 1
        assert entries[0]["logger_name"] == "weather_agent.module_a"


class TestRedactSecrets:
    def test_redacts_token_key(self) -> None:
        result = _redact_secrets(None, "info", {"token": "abc", "name": "test"})  # type: ignore[arg-type]
        assert result["token"] == "[REDACTED]"
        assert result["name"] == "test"

    def test_redacts_api_key(self) -> None:
        result = _redact_secrets(None, "info", {"api_key": "secret", "count": 5})  # type: ignore[arg-type]
        assert result["api_key"] == "[REDACTED]"
        assert result["count"] == 5

    def test_redacts_password(self) -> None:
        result = _redact_secrets(None, "info", {"password": "hunter2"})  # type: ignore[arg-type]
        assert result["password"] == "[REDACTED]"

    def test_redacts_authorization(self) -> None:
        result = _redact_secrets(None, "info", {"authorization": "Bearer xxx"})  # type: ignore[arg-type]
        assert result["authorization"] == "[REDACTED]"

    def test_does_not_redact_normal_fields(self) -> None:
        event_dict: dict[str, object] = {
            "event": "test",
            "user_id": 42,
            "context_key": "chat:123",
        }
        result = _redact_secrets(None, "info", event_dict)  # type: ignore[arg-type]
        assert result["user_id"] == 42
        assert result["context_key"] == "chat:123"


class TestCorrelationId:
    def test_generates_uuid_format(self) -> None:
        corr_id = generate_correlation_id()
        parts = corr_id.split("-")
        assert len(parts) == 5

    def test_generates_unique_ids(self) -> None:
        ids = {generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100


@pytest_asyncio.fixture()
async def async_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture()
async def async_session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        async with session.begin():
            yield session


class TestAuditLogger:
    @pytest.mark.asyncio()
    async def test_log_event_persists_to_db(self, async_session: AsyncSession) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="authorized_message",
            user_id=42,
            context_key="chat:123",
            details={"intent": "weather_query"},
        )
        assert audit_id is not None
        assert isinstance(audit_id, int)

    @pytest.mark.asyncio()
    async def test_log_event_stores_event_type(self, async_session: AsyncSession) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="rule_created",
            user_id=1,
            details={"rule_id": "abc123"},
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.event_type == "rule_created"
        assert row.user_id == 1

    @pytest.mark.asyncio()
    async def test_log_event_stores_correlation_id(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        corr_id = generate_correlation_id()
        audit_id = await logger.log_event(
            event_type="weather_provider_call",
            user_id=42,
            correlation_id=corr_id,
            details={"provider": "open-meteo"},
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.details["correlation_id"] == corr_id
        assert row.details["provider"] == "open-meteo"

    @pytest.mark.asyncio()
    async def test_log_event_stores_context_key(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="notification_sent",
            user_id=42,
            context_key="chat:123:456",
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.context_key == "chat:123:456"

    @pytest.mark.asyncio()
    async def test_log_event_stores_details_dict(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        details = {"rule_id": "r1", "expression": "temp > 30", "result": True}
        audit_id = await logger.log_event(
            event_type="rule_evaluation",
            user_id=1,
            details=details,
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.details["rule_id"] == "r1"
        assert row.details["expression"] == "temp > 30"
        assert row.details["result"] is True

    @pytest.mark.asyncio()
    async def test_log_event_rejects_invalid_type(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        with pytest.raises(ValueError, match="Invalid audit event type"):
            await logger.log_event(event_type="invalid_event_type")

    @pytest.mark.asyncio()
    async def test_log_event_with_none_optionals(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="unauthorized_attempt",
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.event_type == "unauthorized_attempt"
        assert row.user_id is None
        assert row.context_key is None

    @pytest.mark.asyncio()
    async def test_log_event_redacts_secrets_in_details(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="authorized_message",
            user_id=42,
            details={"api_key": "secret-key", "normal_field": "visible"},
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.details["api_key"] == "[REDACTED]"
        assert row.details["normal_field"] == "visible"

    @pytest.mark.asyncio()
    async def test_log_event_truncates_long_values(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        long_value = "x" * 1000
        audit_id = await logger.log_event(
            event_type="authorized_message",
            user_id=42,
            details={"prompt": long_value},
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        stored = row.details["prompt"]
        assert isinstance(stored, str)
        assert len(stored) < len(long_value)
        assert stored.endswith("...[TRUNCATED]")

    @pytest.mark.asyncio()
    async def test_all_valid_event_types(self, async_session: AsyncSession) -> None:
        logger = AuditLogger(async_session)
        for event_type in AuditLogger.VALID_EVENT_TYPES:
            audit_id = await logger.log_event(event_type=event_type)
            assert isinstance(audit_id, int)

    @pytest.mark.asyncio()
    async def test_created_at_is_set(self, async_session: AsyncSession) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="notification_suppressed",
            user_id=1,
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.created_at is not None

    @pytest.mark.asyncio()
    async def test_confirmation_accepted_event(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="confirmation_accepted",
            user_id=42,
            details={"rule_short_id": "abc"},
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.event_type == "confirmation_accepted"
        assert row.details["rule_short_id"] == "abc"

    @pytest.mark.asyncio()
    async def test_confirmation_declined_event(
        self, async_session: AsyncSession
    ) -> None:
        logger = AuditLogger(async_session)
        audit_id = await logger.log_event(
            event_type="confirmation_declined",
            user_id=42,
            details={"rule_short_id": "xyz"},
        )
        result = await async_session.execute(
            select(AuditLog).where(AuditLog.id == audit_id)
        )
        row = result.scalar_one()
        assert row.event_type == "confirmation_declined"


class TestConfigureLogging:
    def teardown_method(self) -> None:
        structlog.reset_defaults()

    def test_configure_logging_produces_json(self) -> None:
        output = StringIO()
        configure_logging("DEBUG")
        structlog.reset_defaults()
        _setup_test_logging(output)
        log = structlog.get_logger("test_config")
        log.info("configured")
        entries = _parse_log_output(output)
        assert len(entries) == 1
        assert entries[0]["event"] == "configured"
        assert entries[0]["level"] == "info"

    def test_configure_logging_includes_timestamp(self) -> None:
        output = StringIO()
        configure_logging("DEBUG")
        structlog.reset_defaults()
        _setup_test_logging(output)
        log = structlog.get_logger("test_config")
        log.info("test")
        entries = _parse_log_output(output)
        assert len(entries) == 1
        assert "timestamp" in entries[0]


class TestGetHelpers:
    def teardown_method(self) -> None:
        structlog.reset_defaults()

    def test_get_audit_logger_returns_instance(self) -> None:
        class FakeSession:
            pass

        logger = get_audit_logger(FakeSession())  # type: ignore[arg-type]
        assert isinstance(logger, AuditLogger)

    def test_get_logger_returns_proxy(self) -> None:
        configure_logging("INFO")
        log = get_logger("test_module")
        bound = log.bind()
        assert isinstance(bound, structlog.stdlib.BoundLogger)