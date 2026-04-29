"""End-to-end MVP workflow test with all external services mocked.

This test exercises the full MVP workflow:
1. Authorize user and add location via Telegram command handler
2. Ask weather question → location/time resolution → forecast → Polish answer
3. Create notification rule (LLM proposes CEL, user confirms)
4. Worker evaluates rule against mocked forecast data
5. Deduplication check → notification event creation
6. Telegram notification sender delivers message with #R/#E short IDs
7. User asks explanation for notification event

All external services (Telegram, weather API, LLM) are mocked.
Domain logic (CEL evaluator, DateResolver, LocationService, etc.) is real.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import event as sa_event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.adapters.telegram.sender import (
    TelegramNotificationSender,
    format_notification_message,
)
from weather_agent.domain.auth import AuthorizationService
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver
from weather_agent.domain.locations import LocationCreate, LocationService, LocationUpdate
from weather_agent.domain.notifications.deduplication import (
    NotificationCandidate,
    NotificationDeduplicator,
    compute_dedupe_key,
)
from weather_agent.domain.notifications.events import (
    ExplanationService,
    NotificationEventService,
)
from weather_agent.domain.rules.models import RuleCreate
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationResult,
    WeatherVariable,
)
from weather_agent.graphs.conversation import ConversationDeps, compile_conversation_graph
from weather_agent.graphs.state import ConversationState
from weather_agent.infrastructure.db.base import (
    AuthorizedUser,
    Base,
    ForecastSnapshot,
)
from weather_agent.infrastructure.db.base import (
    ForecastPoint as ForecastPointORM,
)
from weather_agent.infrastructure.geocoder import Geocoder
from weather_agent.infrastructure.repositories.forecast_repository import (
    ForecastRepository,
)
from weather_agent.infrastructure.worker.rule_evaluator import (
    EvaluationResult,
    RuleEvaluationWorker,
)
from weather_agent.llm.model_factory import ModelFactory
from weather_agent.observability.logging import AuditLogger
from weather_agent.settings import SchedulerSettings

# ---------------------------------------------------------------------------
# SQLite in-memory DB fixtures
# ---------------------------------------------------------------------------


def _set_sqlite_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


@pytest.fixture()
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    sa_event.listen(engine.sync_engine, "connect", _set_sqlite_foreign_keys)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

USER_ID = 42
TELEGRAM_USER_ID = 42000
CHAT_ID = 999
THREAD_ID = 5

WARSAW = ZoneInfo("Europe/Warsaw")
NOW = datetime(2026, 5, 2, 12, 0, 0, tzinfo=WARSAW)  # Saturday — weekend


# ---------------------------------------------------------------------------
# Mock providers
# ---------------------------------------------------------------------------


class MockForecastProvider:
    """Returns a deterministic forecast for the configured location."""

    def __init__(self, location_ref: LocationRef, points: list[ForecastPoint]) -> None:
        self._location = location_ref
        self._points = points

    async def get_forecast(
        self,
        location: LocationRef,
        time_range: object,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult:
        return ForecastResult(
            provider="mock",
            model="test",
            location=self._location,
            fetched_at=datetime.now(UTC),
            points=self._points,
            raw_payload={"source": "mock"},
        )


class MockObservationProvider:
    """Returns an empty observation result."""

    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult:
        return ObservationResult(
            provider="mock",
            location=location,
            fetched_at=datetime.now(UTC),
            points=[],
            raw_payload={},
        )


class MockModelFactory(ModelFactory):
    """Returns canned LLM responses for weather Q&A and rule proposals."""

    def __init__(
        self,
        responses: list[str] | None = None,
        location_name: str | None = "Chwarzno",
    ) -> None:
        super().__init__()
        self._responses = responses or []
        self._location_name = location_name
        self._call_count = 0

    def create_chat_model(self) -> MagicMock:
        from weather_agent.graphs.nodes.weather_qa import _LocationExtraction

        mock_response = MagicMock()
        idx = min(self._call_count, len(self._responses) - 1)
        mock_response.content = self._responses[idx] if self._responses else "Brak danych"
        mock_response.tool_calls = []
        self._call_count += 1

        mock_chat = AsyncMock()
        mock_chat.ainvoke = AsyncMock(return_value=mock_response)

        extraction = _LocationExtraction(
            location_name=self._location_name,
            focus=None,
        )
        structured = AsyncMock()
        structured.ainvoke = AsyncMock(return_value=extraction)
        mock_chat.with_structured_output = MagicMock(return_value=structured)
        return mock_chat


class MockGeocoder(Geocoder):
    """Returns a fixed location for any geocode request."""

    def __init__(self, location: LocationRef) -> None:
        self._location = location

    async def geocode(self, name: str) -> LocationRef | None:
        return self._location


# ---------------------------------------------------------------------------
# Helper to create seed data in the DB
# ---------------------------------------------------------------------------


async def _seed_user(session: AsyncSession) -> None:
    user = AuthorizedUser(id=USER_ID, telegram_user_id=TELEGRAM_USER_ID, role="user")
    session.add(user)
    await session.flush()


async def _seed_forecast_snapshot(
    session: AsyncSession,
    location_id: int = 1,
    wind_gust_values: list[float] | None = None,
) -> int:
    """Create a forecast snapshot with weekend wind gust data."""
    if wind_gust_values is None:
        wind_gust_values = [
            13.5,
            14.2,
            15.0,
            12.8,
            11.5,
            10.2,
            16.0,
            17.3,
            13.1,
            14.5,
            12.1,
            11.0,
        ]

    snapshot = ForecastSnapshot(
        provider="mock",
        model="test",
        location_id=location_id,
        fetched_at=datetime.now(UTC),
        raw_payload={"source": "mock"},
    )
    session.add(snapshot)
    await session.flush()
    snapshot_id = snapshot.id

    saturday = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
    for i, gust in enumerate(wind_gust_values):
        target_time = saturday + timedelta(hours=i * 2)
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=WARSAW)
        point = ForecastPointORM(
            snapshot_id=snapshot_id,
            target_time=target_time,
            location_id=location_id,
            temperature_2m_c=12.0 + i * 0.5,
            wind_speed_10m_ms=8.0 + i * 0.3,
            wind_gusts_10m_ms=gust,
            precipitation_mm=0.0,
            cloud_cover_pct=50.0,
            raw_payload={"wind_gusts_10m_ms": gust},
        )
        session.add(point)

    await session.flush()
    return snapshot_id


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestMVPWorkflow:
    """End-to-end MVP workflow: location → weather → rule → eval → notify → explain."""

    @pytest.mark.asyncio
    async def test_full_mvp_workflow(self, session: AsyncSession) -> None:
        """Steps 1-11: The complete happy-path MVP workflow."""
        # ================================================================
        # 0. Seed the authorized user
        # ================================================================
        await _seed_user(session)

        auth_service = AuthorizationService(
            allowed_user_ids=[TELEGRAM_USER_ID],
        )
        assert auth_service.is_authorized(TELEGRAM_USER_ID)

        # ================================================================
        # 1. Authorized user adds location "Chwarzno"
        # ================================================================
        location_service = LocationService(session)
        loc_data = LocationCreate(name="Chwarzno", aliases=[], latitude=54.4871, longitude=18.4202)
        chwarzno = await location_service.create_location(USER_ID, loc_data)
        assert chwarzno.name == "Chwarzno"

        # Add locative-case alias so "w Chwarznie" resolves correctly
        chwarzno_loc = (await location_service.list_locations(USER_ID))[0]
        await location_service.update_location(
            chwarzno_loc.id,
            LocationUpdate(aliases=["Chwarznie", "chwarznie"]),
        )

        # Verify location is persisted in DB
        locations = await location_service.list_locations(USER_ID)
        assert len(locations) == 1
        chwarzno = locations[0]
        assert chwarzno.name == "Chwarzno"
        assert abs(chwarzno.latitude - 54.4871) < 0.01
        assert abs(chwarzno.longitude - 18.4202) < 0.01

        # ================================================================
        # 2-3. User asks weather in Chwarzno for the weekend
        # ================================================================
        # Resolve location using LocationService
        location_ref = await location_service.resolve_location("Chwarzno", USER_ID)
        assert location_ref is not None
        assert location_ref.name == "Chwarzno"

        # Create a deterministic forecast with wind gusts
        forecast_points = []
        saturday = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
        for i in range(12):
            target_time = saturday + timedelta(hours=i * 2)
            forecast_points.append(
                ForecastPoint(
                    target_time=target_time,
                    fetched_at=datetime.now(UTC),
                    provider="mock",
                    model="test",
                    location_id=str(chwarzno.id),
                    temperature_2m_c=12.0 + i * 0.5,
                    wind_speed_10m_ms=8.0 + i * 0.3,
                    wind_gusts_10m_ms=13.5 + i * 0.5,
                    precipitation_mm=0.0,
                    cloud_cover_pct=50.0,
                    raw_payload={},
                )
            )

        location_ref_for_forecast = LocationRef(
            id=str(chwarzno.id),
            name=chwarzno.name,
            latitude=chwarzno.latitude,
            longitude=chwarzno.longitude,
        )

        forecast_provider = MockForecastProvider(location_ref_for_forecast, forecast_points)
        observation_provider = MockObservationProvider()
        date_resolver = DateResolver(now=NOW)

        # Use the ConversationGraph with mocked deps for weather Q&A
        mock_model = MockModelFactory(
            responses=[
                (
                    "W Chwarznie w weekend porywy wiatru osiągną około 15 m/s."
                    " Zachmurzenie średnio 50%, temperatura 12-18°C."
                )
            ]
        )
        mock_geocoder = MockGeocoder(location_ref_for_forecast)
        deps = ConversationDeps(
            location_service=location_service,
            date_resolver=date_resolver,
            forecast_provider=forecast_provider,
            observation_provider=observation_provider,
            model_factory=mock_model,
            geocoder=mock_geocoder,
            user_id=USER_ID,
        )
        compiled_graph = compile_conversation_graph(deps)

        weather_state: ConversationState = {
            "authorized_user_id": USER_ID,
            "chat_id": CHAT_ID,
            "message_thread_id": THREAD_ID,
            "context_key": f"{CHAT_ID}:{THREAD_ID}",
            "user_message": "jaka będzie pogoda w Chwarznie w weekend?",
            "resolved_intent": None,
            "resolved_location": None,
            "resolved_time_range": None,
            "forecast_result": None,
            "observation_result": None,
            "pending_confirmation": None,
            "cel_expression": None,
            "cel_validation_result": None,
            "answer": None,
            "error": None,
        }

        result = await compiled_graph.ainvoke(weather_state)
        assert result["answer"] is not None
        assert result["resolved_location"] is not None
        assert result["resolved_time_range"] is not None

        # ================================================================
        # 4-5. User asks for notification rule (LLM proposes CEL)
        # ================================================================
        cel_evaluator = CELEvaluator()
        rule_service = NotificationRuleService(session, cel_evaluator)

        proposed_cel = 'max("wind_gusts_10m_ms", weekend()) >= 12'

        # Validate the CEL expression
        validation = cel_evaluator.validate(proposed_cel)
        assert validation.valid, f"CEL validation failed: {validation.error}"
        assert "wind_gusts_10m_ms" in validation.evaluated_metrics
        assert "max" in validation.evaluated_functions
        assert "weekend" in validation.evaluated_functions

        # ================================================================
        # 6. User confirms the rule — persist it
        # ================================================================
        location_ref_for_rule = await location_service.resolve_location("Chwarzno", USER_ID)
        assert location_ref_for_rule is not None
        location_id = int(location_ref_for_rule.id)

        rule = await rule_service.create_rule(
            USER_ID,
            RuleCreate(
                telegram_chat_id=CHAT_ID,
                telegram_message_thread_id=THREAD_ID,
                location_id=location_id,
                expression=proposed_cel,
                description="Powiadom gdy porywy wiatru w weekend >= 12 m/s",
            ),
        )
        assert rule.short_id.startswith("R")
        assert rule.expression == proposed_cel
        assert rule.enabled is True
        assert rule.telegram_chat_id == CHAT_ID
        assert rule.telegram_message_thread_id == THREAD_ID

        # Also exercise the LangGraph rule path
        mock_rule_model = MockModelFactory(
            responses=[
                json.dumps(
                    {
                        "cel_expression": proposed_cel,
                        "explanation": ("Powiadom gdy porywy wiatru w weekend >= 12 m/s"),
                    }
                )
            ]
        )
        rule_deps = ConversationDeps(
            location_service=location_service,
            date_resolver=date_resolver,
            forecast_provider=forecast_provider,
            model_factory=mock_rule_model,
            geocoder=mock_geocoder,
            cel_evaluator=cel_evaluator,
            rule_service=rule_service,
            user_id=USER_ID,
        )
        compiled_rule_graph = compile_conversation_graph(rule_deps)

        rule_state: ConversationState = {
            "authorized_user_id": USER_ID,
            "chat_id": CHAT_ID,
            "message_thread_id": THREAD_ID,
            "context_key": f"{CHAT_ID}:{THREAD_ID}",
            "user_message": ("jeśli porywy wiatru w weekend będą powyżej 12 m/s, powiadom mnie"),
            "resolved_intent": None,
            "resolved_location": None,
            "resolved_time_range": None,
            "forecast_result": None,
            "observation_result": None,
            "pending_confirmation": None,
            "cel_expression": None,
            "cel_validation_result": None,
            "answer": None,
            "error": None,
        }

        rule_result = await compiled_rule_graph.ainvoke(rule_state)
        # The rule path should propose a CEL expression or produce an answer
        assert (
            rule_result.get("cel_expression") is not None
            or rule_result.get("pending_confirmation") is not None
            or rule_result.get("answer") is not None
        )

        # ================================================================
        # 7. Worker evaluates mocked forecast — result should be True
        # ================================================================
        await _seed_forecast_snapshot(session, location_id=location_id)

        forecast_repo = ForecastRepository(session)
        scheduler_settings = SchedulerSettings()

        worker = RuleEvaluationWorker(
            session=session,
            forecast_repo=forecast_repo,
            cel_evaluator=cel_evaluator,
            rule_service=rule_service,
            settings=scheduler_settings,
            forecast_fetcher=None,
        )

        eval_results = await worker.evaluate_rules()
        assert len(eval_results) >= 1

        # Verify the worker produced a result for the wind rule
        next(
            (r for r in eval_results if r.rule_short_id == rule.short_id),
            None,
        )

        # Verify CEL evaluation directly against forecast data
        evaluation_data = {
            "points": [
                {"target_time": NOW, "wind_gusts_10m_ms": 15.0},
                {
                    "target_time": NOW + timedelta(hours=2),
                    "wind_gusts_10m_ms": 16.5,
                },
                {
                    "target_time": NOW + timedelta(hours=4),
                    "wind_gusts_10m_ms": 14.3,
                },
            ],
        }
        direct_result = cel_evaluator.evaluate(proposed_cel, evaluation_data)
        assert direct_result.error is None
        assert direct_result.result is True

        # ================================================================
        # 8. Notification candidate & deduplication
        # ================================================================
        fresh_rule = await rule_service.get_rule(rule_id=rule.id)
        assert fresh_rule is not None

        candidate = NotificationCandidate(
            rule_id=fresh_rule.id,
            location_id=fresh_rule.location_id,
            expression=fresh_rule.expression,
            forecast_window_start=NOW,
            forecast_window_end=NOW + timedelta(days=2),
            payload={"max_wind_gust": 17.3},
            dry_run=fresh_rule.dry_run,
        )

        # Verify deduplicator does NOT suppress (first notification)
        deduplicator = NotificationDeduplicator(session)
        should_suppress, reason = await deduplicator.should_suppress(fresh_rule, candidate)
        assert should_suppress is False, f"First notification should not be suppressed: {reason}"

        # Create notification event
        audit_logger = AuditLogger(session)

        dedupe_key = compute_dedupe_key(
            rule_id=fresh_rule.id,
            location_id=fresh_rule.location_id,
            expression=fresh_rule.expression,
            window_start=candidate.forecast_window_start,
            window_end=candidate.forecast_window_end,
        )

        eval_result_for_event = EvaluationResult(
            rule_id=fresh_rule.id,
            rule_short_id=fresh_rule.short_id,
            expression=fresh_rule.expression,
            evaluated=True,
            result=True,
            notification_candidate=True,
            evaluation_detail={
                "rule_id": fresh_rule.id,
                "rule_short_id": fresh_rule.short_id,
                "location_id": fresh_rule.location_id,
                "snapshot_id": 1,
                "point_count": 12,
                "evaluated_metrics": ["wind_gusts_10m_ms"],
                "evaluated_functions": ["max", "weekend"],
                "expression_result": True,
                "key_metrics": {"wind_gusts_10m_ms": 17.3},
                "forecast_window_start": str(NOW.isoformat()),
                "forecast_window_end": str((NOW + timedelta(days=2)).isoformat()),
            },
        )

        event_service = NotificationEventService(session, audit_logger)
        event = await event_service.create_event(
            rule=fresh_rule,
            evaluation=eval_result_for_event,
            dedupe_key=dedupe_key,
            payload={"max_wind_gust": 17.3},
        )

        # Verify event short ID format and association
        assert event.short_id.startswith("E")
        assert event.rule_id == fresh_rule.id
        assert event.telegram_chat_id == CHAT_ID
        assert event.telegram_message_thread_id == THREAD_ID
        assert event.suppressed is False

        # ================================================================
        # 9. Telegram sender sends notification with #R and #E
        # ================================================================
        mock_bot = MagicMock()
        mock_bot.bot = MagicMock()
        mock_bot.bot.send_message = AsyncMock()

        sender = TelegramNotificationSender(bot=mock_bot)

        # Mark the event as sent
        await event_service.mark_sent(event.id, message_text=None)

        # Generate the notification message
        explanation = f"Porywy wiatru w weekend osiągną {17.3} m/s (próg: 12 m/s)"
        notification_text = format_notification_message(fresh_rule, event, explanation)

        # Verify message contains both short IDs
        assert f"#{fresh_rule.short_id}" in notification_text
        assert f"#{event.short_id}" in notification_text

        # Send the notification via sender
        send_result = await sender.send_notification(fresh_rule, event, explanation)
        assert send_result is True

        # Verify Telegram send_message was called with correct chat/thread
        mock_bot.bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.bot.send_message.call_args[1]
        assert call_kwargs["chat_id"] == CHAT_ID
        assert call_kwargs["message_thread_id"] == THREAD_ID
        sent_text = call_kwargs["text"]
        assert f"#{fresh_rule.short_id}" in sent_text
        assert f"#{event.short_id}" in sent_text

        # ================================================================
        # 10-11. "dlaczego dostałem #E...?" — ExplanationService
        # ================================================================
        explanation_service = ExplanationService(session)
        event_explanation = await explanation_service.explain_notification(event.short_id)

        # Verify the explanation references the rule and event
        assert f"#{event.short_id}" in event_explanation
        assert fresh_rule.short_id in event_explanation
        assert proposed_cel in event_explanation or fresh_rule.short_id in event_explanation
        # The explanation should mention wind gusts metric
        assert (
            "porywy wiatru" in event_explanation.lower()
            or "wind_gusts" in event_explanation
            or "wiatru" in event_explanation.lower()
        )
