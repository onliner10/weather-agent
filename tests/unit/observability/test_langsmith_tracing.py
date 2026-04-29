from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.rules.models import RuleCreate
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.weather import LocationRef
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
    ForecastSnapshot,
    Location,
)
from weather_agent.infrastructure.db.base import (
    ForecastPoint as ForecastPointORM,
)
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.infrastructure.worker.rule_evaluator import RuleEvaluationWorker
from weather_agent.observability.langsmith_tracing import LangSmithTracing, configure_tracing
from weather_agent.observability.tracing import (
    build_graph_config,
    build_node_metadata,
    build_run_name,
    build_telegram_turn_metadata,
    build_telegram_turn_tags,
)
from weather_agent.settings import LangSmithSettings, SchedulerSettings

# ---------------------------------------------------------------------------
# Existing env-var mutation tests
# ---------------------------------------------------------------------------


class TestLangSmithTracing:
    def setup_method(self) -> None:
        for key in (
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
            "LANGSMITH_TRACING_V2",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ):
            os.environ.pop(key, None)

    def test_env_vars_set_when_enabled(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=SecretStr("ls-test-key"),
            project="test-project",
            endpoint="https://api.smith.langchain.com",
        )
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
        assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
        assert os.environ["LANGCHAIN_PROJECT"] == "test-project"
        assert os.environ["LANGSMITH_PROJECT"] == "test-project"
        assert os.environ["LANGCHAIN_ENDPOINT"] == "https://api.smith.langchain.com"
        assert os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"

    def test_no_env_vars_when_disabled(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "old-key"
        settings = LangSmithSettings(enabled=False)
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert "LANGCHAIN_TRACING_V2" not in os.environ
        assert "LANGCHAIN_API_KEY" not in os.environ

    def test_is_enabled_returns_true_when_set(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        assert LangSmithTracing.is_enabled() is True

    def test_is_enabled_returns_false_when_not_set(self) -> None:
        assert LangSmithTracing.is_enabled() is False

    def test_app_works_without_langsmith_configured(self) -> None:
        settings = LangSmithSettings(enabled=False)
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert not LangSmithTracing.is_enabled()

    def test_enabled_without_api_key_sets_tracing_flag(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=None,
            project="test-project",
            endpoint="https://api.smith.langchain.com",
        )
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert "LANGCHAIN_API_KEY" not in os.environ

    def test_configure_tracing_with_explicit_settings(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=SecretStr("explicit-key"),
            project="explicit-project",
        )
        tracing = LangSmithTracing()
        tracing.configure_tracing(settings)
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "explicit-key"

    def test_default_project_name(self) -> None:
        settings = LangSmithSettings(enabled=False)
        assert settings.project == "weather-agent-dev"

    def test_configure_tracing_module_function(self) -> None:
        settings = LangSmithSettings(
            enabled=True,
            api_key=SecretStr("func-key"),
            project="func-project",
        )
        configure_tracing(settings)
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_API_KEY"] == "func-key"

    def test_langsmith_env_triggers_is_enabled(self) -> None:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        assert LangSmithTracing.is_enabled() is True

    def test_get_status_returns_rich_status(self) -> None:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "test-key"
        os.environ["LANGSMITH_PROJECT"] = "test-project"
        settings = LangSmithSettings(enabled=True, api_key=SecretStr("test-key"))
        status = LangSmithTracing.get_status(settings)
        assert status.configured is True
        assert status.env_tracing_enabled is True
        assert status.has_api_key is True
        assert status.project == "test-project"

    def test_disable_clears_langsmith_namespace(self) -> None:
        os.environ["LANGSMITH_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "old-key"
        settings = LangSmithSettings(enabled=False)
        tracing = LangSmithTracing(settings)
        tracing.configure_tracing()
        assert "LANGSMITH_TRACING_V2" not in os.environ
        assert "LANGSMITH_API_KEY" not in os.environ

    def test_missing_api_key_status(self) -> None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        settings = LangSmithSettings(enabled=True, api_key=None)
        status = LangSmithTracing.get_status(settings)
        assert status.env_tracing_enabled is True
        assert status.has_api_key is False

    def teardown_method(self) -> None:
        for key in (
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_API_KEY",
            "LANGCHAIN_PROJECT",
            "LANGCHAIN_ENDPOINT",
            "LANGSMITH_TRACING_V2",
            "LANGSMITH_API_KEY",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ):
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Trace emission verification helpers (worker tests)
# ---------------------------------------------------------------------------


@dataclass
class _CapturedTrace:
    name: str
    run_type: str
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class _TraceStub:
    """Hermetic stub for ``langsmith.trace`` that records every invocation."""

    def __init__(self) -> None:
        self.calls: list[_CapturedTrace] = []

    def __call__(
        self,
        name: str,
        run_type: str = "chain",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _TraceContextManager:
        self.calls.append(
            _CapturedTrace(
                name=name,
                run_type=run_type,
                tags=tags,
                metadata=metadata,
                kwargs=kwargs,
            )
        )
        return _TraceContextManager()


class _TraceContextManager:
    async def __aenter__(self) -> MagicMock:
        return MagicMock()

    async def __aexit__(self, *args: Any) -> bool:
        return False

    def __enter__(self) -> MagicMock:
        return MagicMock()

    def __exit__(self, *args: Any) -> bool:
        return False


# ---------------------------------------------------------------------------
# Worker flow trace tests
# ---------------------------------------------------------------------------


def _set_sqlite_foreign_keys(dbapi_connection: object, _connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
async def _worker_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    event.listen(engine.sync_engine, "connect", _set_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
def _cel_evaluator() -> CELEvaluator:
    return CELEvaluator()


@pytest.fixture()
def _scheduler_settings() -> SchedulerSettings:
    return SchedulerSettings(rule_evaluation_minutes=15)


@pytest.fixture()
def _rule_service(
    _worker_session: AsyncSession, _cel_evaluator: CELEvaluator
) -> NotificationRuleService:
    return NotificationRuleService(_worker_session, _cel_evaluator)


@pytest.fixture()
def _forecast_repo(_worker_session: AsyncSession) -> ForecastRepository:
    return ForecastRepository(_worker_session)


async def _worker_create_user(session: AsyncSession, user_id: int = 1) -> None:
    user = AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user")
    session.add(user)
    await session.flush()


async def _worker_create_location(
    session: AsyncSession,
    user_id: int = 1,
    loc_id: int = 1,
    name: str = "Test Location",
) -> None:
    loc = Location(
        id=loc_id,
        user_id=user_id,
        name=name,
        aliases=["test"],
        latitude=52.22,
        longitude=21.01,
        enabled=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(loc)
    await session.flush()


async def _worker_create_rule(
    rule_service: NotificationRuleService,
    user_id: int = 1,
    location_id: int = 1,
    expression: str = 'max("wind_gusts_10m_ms", weekend()) >= 12',
    dry_run: bool = False,
    enabled: bool = True,
) -> Any:
    data = RuleCreate(
        telegram_chat_id=12345,
        location_id=location_id,
        expression=expression,
        dry_run=dry_run,
        enabled=enabled,
    )
    return await rule_service.create_rule(user_id, data)


async def _worker_seed_forecast_data(
    session: AsyncSession,
    location_id: int = 1,
    num_points: int = 3,
    wind_gusts_base: float = 6.0,
) -> int:
    fetched = datetime.now(UTC)
    snapshot = ForecastSnapshot(
        provider="open-meteo",
        model="dwd-icon",
        location_id=location_id,
        fetched_at=fetched,
        raw_payload={"source": "test"},
    )
    session.add(snapshot)
    await session.flush()
    await session.refresh(snapshot)

    for i in range(num_points):
        point = ForecastPointORM(
            snapshot_id=snapshot.id,
            target_time=fetched + timedelta(hours=i + 1),
            location_id=location_id,
            temperature_2m_c=5.0 + i,
            apparent_temperature_c=3.0 + i,
            precipitation_mm=0.1 * i,
            precipitation_probability_pct=10.0 * i,
            rain_mm=0.1 * i,
            snowfall_cm=0.0,
            cloud_cover_pct=50.0 + i * 10,
            wind_speed_10m_ms=3.0 + i,
            wind_gusts_10m_ms=wind_gusts_base + i,
            wind_direction_10m_deg=180.0,
            pressure_msl_hpa=1013.0,
            relative_humidity_2m_pct=70.0,
            weather_code="1",
            raw_payload={"test": True},
        )
        session.add(point)

    await session.flush()
    return snapshot.id


class TestWorkerTraceEmission:
    async def test_worker_evaluate_rules_emits_traces(
        self,
        _worker_session: AsyncSession,
        _forecast_repo: ForecastRepository,
        _rule_service: NotificationRuleService,
        _cel_evaluator: CELEvaluator,
        _scheduler_settings: SchedulerSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _worker_create_user(_worker_session)
        await _worker_create_location(_worker_session)
        await _worker_seed_forecast_data(_worker_session, wind_gusts_base=14.0)
        await _worker_create_rule(_rule_service)

        stub = _TraceStub()
        monkeypatch.setattr(
            "weather_agent.infrastructure.worker.rule_evaluator.trace",
            stub,
        )

        worker = RuleEvaluationWorker(
            session=_worker_session,
            forecast_repo=_forecast_repo,
            cel_evaluator=_cel_evaluator,
            rule_service=_rule_service,
            settings=_scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].evaluated is True
        assert results[0].notification_candidate is True

        names = [c.name for c in stub.calls]
        assert "evaluate_rules" in names
        assert "evaluate_single_rule" in names
        assert "evaluation_result" in names

        eval_rules = next(c for c in stub.calls if c.name == "evaluate_rules")
        assert eval_rules.run_type == "tool"
        assert eval_rules.metadata is not None
        assert eval_rules.metadata.get("rule_count") == 1
        assert eval_rules.metadata.get("dry_run") is False

        eval_single = next(c for c in stub.calls if c.name == "evaluate_single_rule")
        assert eval_single.run_type == "tool"
        assert eval_single.metadata is not None
        assert eval_single.metadata.get("rule_id") is not None
        assert eval_single.metadata.get("rule_short_id") is not None
        assert eval_single.metadata.get("location_id") == 1
        assert eval_single.metadata.get("dry_run") is False

        eval_result = next(c for c in stub.calls if c.name == "evaluation_result")
        assert eval_result.run_type == "tool"
        assert eval_result.metadata is not None
        assert eval_result.metadata.get("evaluated") is True
        assert eval_result.metadata.get("result") is True
        assert eval_result.metadata.get("notification_candidate") is True
        assert eval_result.metadata.get("dry_run") is False

    async def test_worker_dry_run_reflected_in_trace_metadata(
        self,
        _worker_session: AsyncSession,
        _forecast_repo: ForecastRepository,
        _rule_service: NotificationRuleService,
        _cel_evaluator: CELEvaluator,
        _scheduler_settings: SchedulerSettings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _worker_create_user(_worker_session)
        await _worker_create_location(_worker_session)
        await _worker_seed_forecast_data(_worker_session, wind_gusts_base=14.0)
        await _worker_create_rule(_rule_service, dry_run=True)

        stub = _TraceStub()
        monkeypatch.setattr(
            "weather_agent.infrastructure.worker.rule_evaluator.trace",
            stub,
        )

        worker = RuleEvaluationWorker(
            session=_worker_session,
            forecast_repo=_forecast_repo,
            cel_evaluator=_cel_evaluator,
            rule_service=_rule_service,
            settings=_scheduler_settings,
        )
        results = await worker.evaluate_rules()

        assert len(results) == 1
        assert results[0].dry_run is True

        eval_single = next(c for c in stub.calls if c.name == "evaluate_single_rule")
        assert eval_single.metadata is not None
        assert eval_single.metadata.get("dry_run") is True

        eval_result = next(c for c in stub.calls if c.name == "evaluation_result")
        assert eval_result.metadata is not None
        assert eval_result.metadata.get("dry_run") is True


# ---------------------------------------------------------------------------
# Tracing utility tests
# ---------------------------------------------------------------------------


class TestTracingUtilities:
    def test_build_telegram_turn_metadata_excludes_secrets_and_large_payloads(self) -> None:
        from types import SimpleNamespace

        loc = LocationRef(id="1", name="Warszawa", latitude=52.22, longitude=21.01)
        tr = SimpleNamespace(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 5, 1, 23, 59, tzinfo=UTC),
            explanation="Jutro",
        )
        state: dict[str, Any] = {
            "authorized_user_id": 12345,
            "chat_id": 999,
            "message_thread_id": 1,
            "context_key": "999:1",
            "user_message": "x" * 200,
            "message_id": 42,
            "reply_to_message_id": 41,
            "resolved_intent": "weather",
            "resolved_location": loc,
            "resolved_time_range": tr,
            "forecast_result": {"raw_payload": {"large": "data"}},  # type: ignore[dict-item]
            "pending_confirmation": {"action": "activate_rule"},
            "cel_expression": "temp > 30",
        }

        metadata = build_telegram_turn_metadata(state)

        # Included
        assert metadata["context_key"] == "999:1"
        assert metadata["chat_id"] == 999
        assert metadata["message_thread_id"] == 1
        assert metadata["telegram_user_id"] == 12345
        assert metadata["inbound_message_id"] == 42
        assert metadata["reply_to_message_id"] == 41
        assert metadata["is_reply_follow_up"] is True
        assert metadata["resolved_intent"] == "weather"
        assert metadata["resolved_location_name"] == "Warszawa"
        assert metadata["resolved_location_id"] == "1"
        assert metadata["resolved_time_explanation"] == "Jutro"

        # Only preview
        assert metadata["user_message_preview"] == "x" * 80

        # Excluded
        assert "reply_context_turns" not in metadata
        assert "forecast_result" not in metadata
        assert "observation_result" not in metadata
        assert "pending_confirmation" not in metadata
        assert "cel_expression" not in metadata
        assert "cel_validation_result" not in metadata

    def test_build_telegram_turn_tags(self) -> None:
        state: dict[str, Any] = {
            "chat_id": 1,
            "context_key": "1",
            "resolved_intent": "weather",
        }
        tags = build_telegram_turn_tags(state)
        assert tags == ["telegram", "conversation", "intent:weather"]

    def test_build_telegram_turn_tags_with_reply(self) -> None:
        state: dict[str, Any] = {
            "chat_id": 1,
            "context_key": "1",
            "reply_to_message_id": 42,
        }
        tags = build_telegram_turn_tags(state)
        assert "reply-follow-up" in tags

    def test_build_run_name(self) -> None:
        state: dict[str, Any] = {
            "chat_id": 1,
            "context_key": "1:2",
            "resolved_intent": "rule",
        }
        name = build_run_name(state)
        assert name == "telegram-turn:1:2:rule"

    def test_build_run_name_defaults_to_unknown(self) -> None:
        state: dict[str, Any] = {"chat_id": 1, "context_key": "1"}
        name = build_run_name(state)
        assert name == "telegram-turn:1:unknown"

    def test_build_graph_config(self) -> None:
        state: dict[str, Any] = {
            "chat_id": 1,
            "context_key": "1",
            "resolved_intent": "weather",
            "user_message": "pogoda?",
        }
        config = build_graph_config(state)
        assert config["run_name"] == "telegram-turn:1:weather"
        assert "telegram" in config["tags"]
        assert config["metadata"]["context_key"] == "1"

    def test_build_node_metadata_inherits_conversation_context(self) -> None:
        state: dict[str, Any] = {
            "chat_id": 1,
            "context_key": "1",
            "resolved_intent": "weather",
        }
        node_meta = build_node_metadata(state, "resolve_location")
        assert node_meta["node"] == "resolve_location"
        assert node_meta["context_key"] == "1"
        assert node_meta["resolved_intent"] == "weather"

    def test_build_telegram_turn_metadata_omits_none_values(self) -> None:
        state: dict[str, Any] = {
            "chat_id": 1,
            "context_key": "1",
            "user_message": None,
        }
        metadata = build_telegram_turn_metadata(state)
        assert "user_message_preview" not in metadata
        assert "resolved_intent" not in metadata
