from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agent.domain.cel.evaluator import CELEvaluationResult
from weather_agent.domain.locations import Location, LocationCreate
from weather_agent.domain.rules.models import NotificationRule, RuleCreate
from weather_agent.domain.weather import LocationRef
from weather_agent.llm.tools.rules_tools import (
    RulesToolbox,
    ScheduleRuleToolResult,
)


@pytest.fixture()
def mock_rule_service() -> MagicMock:
    svc = MagicMock()
    svc.create_rule = AsyncMock()
    svc.get_rule = AsyncMock()
    svc.get_rule_for_user = AsyncMock()
    svc.update_rule = AsyncMock()
    svc.list_rules = AsyncMock()
    return svc


@pytest.fixture()
def mock_location_service() -> MagicMock:
    svc = MagicMock()
    svc.get_default_location = AsyncMock(return_value=None)
    svc.resolve_location = AsyncMock(return_value=None)
    svc.create_location = AsyncMock()
    return svc


@pytest.fixture()
def mock_cel_evaluator() -> MagicMock:
    ev = MagicMock()
    ev.validate.return_value = CELEvaluationResult(expression="True")
    return ev


@pytest.fixture()
def mock_geocoder() -> MagicMock:
    g = MagicMock()
    g.geocode = AsyncMock(return_value=None)
    return g


@pytest.fixture()
def mock_memory_service() -> MagicMock:
    m = MagicMock()
    m.store_pending_confirmation = AsyncMock()
    m.get_pending_confirmation = AsyncMock(return_value=None)
    m.clear_pending_confirmation = AsyncMock()
    return m


@pytest.fixture()
def toolbox(
    mock_rule_service: MagicMock,
    mock_location_service: MagicMock,
    mock_cel_evaluator: MagicMock,
    mock_geocoder: MagicMock,
    mock_memory_service: MagicMock,
) -> RulesToolbox:
    return RulesToolbox(
        rule_service=mock_rule_service,
        location_service=mock_location_service,
        cel_evaluator=mock_cel_evaluator,
        geocoder=mock_geocoder,
        memory_service=mock_memory_service,
        context_key="test:1",
        user_id=100,
        chat_id=200,
        message_thread_id=1,
    )


class TestToolRegistration:
    def test_to_langchain_tools_includes_schedule_notification(self, toolbox: RulesToolbox) -> None:
        tools = toolbox.to_langchain_tools()
        names = [t.name for t in tools]
        assert "schedule_notification" in names

    def test_schedule_notification_tool_has_schema(self, toolbox: RulesToolbox) -> None:
        tools = toolbox.to_langchain_tools()
        sn = next(t for t in tools if t.name == "schedule_notification")
        assert sn.args_schema is not None


class TestCELCapabilities:
    @pytest.mark.asyncio()
    async def test_capabilities_include_signatures_rules_and_examples(
        self,
        toolbox: RulesToolbox,
    ) -> None:
        result = await toolbox.get_cel_capabilities()

        assert result.signatures is not None
        assert result.rules is not None
        assert result.examples is not None
        assert result.signatures["max"] == 'max("metric_name", time_range)'
        assert 'max("wind_gusts_10m_ms", weekend()) > 12.0' in result.examples


class TestScheduleNotificationValidation:
    @pytest.mark.asyncio()
    async def test_invalid_cel_expression_returns_error(self, toolbox: RulesToolbox) -> None:
        toolbox.cel_evaluator.validate.return_value = CELEvaluationResult(
            expression="invalid",
            error="undefined: foo",
        )
        result = await toolbox.schedule_notification(
            schedule_type="once",
            schedule_expression="2026-05-01T12:00:00",
            explanation="test",
        )
        assert isinstance(result, ScheduleRuleToolResult)
        assert result.error is not None
        assert "CEL" in result.error
        assert result.pending is False

    @pytest.mark.asyncio()
    async def test_invalid_schedule_returns_error(self, toolbox: RulesToolbox) -> None:
        result = await toolbox.schedule_notification(
            schedule_type="once",
            schedule_expression="not-a-date",
            explanation="test",
        )
        assert isinstance(result, ScheduleRuleToolResult)
        assert result.error is not None
        assert "harmonogram" in result.error

    @pytest.mark.asyncio()
    async def test_invalid_cron_schedule_returns_error(self, toolbox: RulesToolbox) -> None:
        result = await toolbox.schedule_notification(
            schedule_type="cron",
            schedule_expression="not-a-cron",
            explanation="test",
        )
        assert isinstance(result, ScheduleRuleToolResult)
        assert result.error is not None
        assert "harmonogram" in result.error

    @pytest.mark.asyncio()
    async def test_unknown_schedule_type_returns_error(self, toolbox: RulesToolbox) -> None:
        result = await toolbox.schedule_notification(
            schedule_type="invalid_type",
            schedule_expression="2026-05-01T12:00:00",
            explanation="test",
        )
        assert isinstance(result, ScheduleRuleToolResult)
        assert result.error is not None


class TestScheduleNotificationStoresPending:
    @pytest.mark.asyncio()
    async def test_stores_pending_confirmation_with_schedule(
        self,
        toolbox: RulesToolbox,
        mock_memory_service: MagicMock,
    ) -> None:
        result = await toolbox.schedule_notification(
            schedule_type="once",
            schedule_expression="2026-05-01T12:00:00",
            explanation="Przypomnienie o śniegu",
        )
        assert result.pending is True
        assert result.proposal is not None
        assert "Harmonogram" in result.proposal

        mock_memory_service.store_pending_confirmation.assert_awaited_once()
        call_args = mock_memory_service.store_pending_confirmation.call_args[0]
        stored = call_args[1]
        assert stored["action"] == "schedule_notification"
        assert stored["schedule"] == "once:2026-05-01T12:00:00"
        assert stored["cel_expression"] == "True"

    @pytest.mark.asyncio()
    async def test_stores_pending_with_custom_cel(
        self,
        toolbox: RulesToolbox,
        mock_memory_service: MagicMock,
    ) -> None:
        toolbox.cel_evaluator.validate.return_value = CELEvaluationResult(
            expression="temperature_2m_c < 0",
        )
        result = await toolbox.schedule_notification(
            schedule_type="cron",
            schedule_expression="0 8 * * *",
            explanation="Poranne info o mrozie",
            cel_expression="temperature_2m_c < 0",
        )
        assert result.pending is True

        mock_memory_service.store_pending_confirmation.assert_awaited_once()
        call_args = mock_memory_service.store_pending_confirmation.call_args[0]
        stored = call_args[1]
        assert stored["schedule"] == "cron:0 8 * * *"
        assert stored["cel_expression"] == "temperature_2m_c < 0"


class TestScheduleNotificationAutoSaveLocation:
    @pytest.mark.asyncio()
    async def test_schedule_autosaves_unsaved_city(
        self,
        toolbox: RulesToolbox,
        mock_location_service: MagicMock,
        mock_geocoder: MagicMock,
        mock_memory_service: MagicMock,
    ) -> None:
        now = datetime.now(UTC)
        expected_id = 99
        mock_geocoder.geocode.return_value = LocationRef(
            id="300",
            name="Gdynia",
            latitude=54.5189,
            longitude=18.5305,
        )
        mock_location_service.create_location = AsyncMock(
            return_value=Location(
                id=expected_id,
                name="Gdynia",
                aliases=[],
                latitude=54.5189,
                longitude=18.5305,
                description=None,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        )

        result = await toolbox.schedule_notification(
            schedule_type="once",
            schedule_expression="2026-05-01T09:00:00",
            explanation="Aktualna pogoda dla Gdyni",
            location_name="Gdynia",
        )

        assert result.pending is True
        assert result.error is None

        mock_location_service.create_location.assert_awaited_once_with(
            100,
            LocationCreate(
                name="Gdynia",
                aliases=[],
                latitude=54.5189,
                longitude=18.5305,
            ),
        )

        mock_memory_service.store_pending_confirmation.assert_awaited_once()
        call_args = mock_memory_service.store_pending_confirmation.call_args[0]
        stored = call_args[1]
        assert stored["location_id"] == expected_id

    @pytest.mark.asyncio()
    async def test_propose_autosaves_unsaved_city(
        self,
        toolbox: RulesToolbox,
        mock_location_service: MagicMock,
        mock_geocoder: MagicMock,
        mock_memory_service: MagicMock,
    ) -> None:
        now = datetime.now(UTC)
        expected_id = 77
        mock_geocoder.geocode.return_value = LocationRef(
            id="400",
            name="Sopot",
            latitude=54.4418,
            longitude=18.5600,
        )
        mock_location_service.create_location = AsyncMock(
            return_value=Location(
                id=expected_id,
                name="Sopot",
                aliases=[],
                latitude=54.4418,
                longitude=18.5600,
                description=None,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        )

        result = await toolbox.propose_notification_rule(
            cel_expression="True",
            explanation="Powiadomienie o pogodzie w Sopocie",
            location_name="Sopot",
        )

        assert result.pending is True
        assert result.error is None

        mock_location_service.create_location.assert_awaited_once()

        mock_memory_service.store_pending_confirmation.assert_awaited_once()
        call_args = mock_memory_service.store_pending_confirmation.call_args[0]
        stored = call_args[1]
        assert stored["location_id"] == expected_id

    @pytest.mark.asyncio()
    async def test_unresolvable_city_still_returns_error(
        self,
        toolbox: RulesToolbox,
        mock_geocoder: MagicMock,
        mock_location_service: MagicMock,
    ) -> None:
        mock_geocoder.geocode.return_value = None
        mock_location_service.resolve_location.return_value = None

        result = await toolbox.schedule_notification(
            schedule_type="once",
            schedule_expression="2026-05-01T09:00:00",
            explanation="Test",
            location_name="NieistniejaceMiasto",
        )

        assert result.pending is False
        assert result.error is not None
        assert "Nie znaleziono lokalizacji" in result.error


class TestConfirmScheduleNotification:
    @pytest.mark.asyncio()
    async def test_confirm_creates_rule_with_schedule(
        self,
        toolbox: RulesToolbox,
        mock_rule_service: MagicMock,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.get_pending_confirmation.return_value = {
            "action": "schedule_notification",
            "cel_expression": "True",
            "explanation": "Przypomnienie",
            "validated": True,
            "location_id": 42,
            "chat_id": 200,
            "message_thread_id": 1,
            "stored_at": datetime.now(UTC).isoformat(),
            "schedule": "once:2026-05-01T12:00:00",
        }
        mock_rule_service.create_rule.return_value = NotificationRule(
            id=1,
            short_id="R1A2B3",
            user_id=100,
            telegram_chat_id=200,
            telegram_message_thread_id=1,
            location_id=42,
            expression="True",
            schedule="once:2026-05-01T12:00:00",
            description="Przypomnienie",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        result = await toolbox.confirm_pending_action()

        assert result.error is None
        assert result.answer is not None
        assert "zaplanowane" in result.answer
        assert "harmonogram" in result.answer
        assert result.short_id == "R1A2B3"

        create_call = mock_rule_service.create_rule.call_args
        assert create_call is not None
        _, rule_create = create_call[0]
        assert isinstance(rule_create, RuleCreate)
        assert rule_create.schedule == "once:2026-05-01T12:00:00"

        mock_memory_service.clear_pending_confirmation.assert_awaited_once()


class TestConfirmExistingFlows:
    @pytest.mark.asyncio()
    async def test_confirm_create_rule_still_works(
        self,
        toolbox: RulesToolbox,
        mock_rule_service: MagicMock,
        mock_memory_service: MagicMock,
        mock_location_service: MagicMock,
    ) -> None:
        mock_memory_service.get_pending_confirmation.return_value = {
            "action": "create_rule",
            "cel_expression": "temperature_2m_c > 25",
            "explanation": "Gorąco",
            "validated": True,
            "location_id": None,
            "chat_id": 200,
            "message_thread_id": 1,
            "stored_at": datetime.now(UTC).isoformat(),
        }
        mock_location_service.get_default_location.return_value = MagicMock(id=42)
        mock_rule_service.create_rule.return_value = NotificationRule(
            id=2,
            short_id="R4D5E6",
            user_id=100,
            telegram_chat_id=200,
            telegram_message_thread_id=1,
            location_id=42,
            expression="temperature_2m_c > 25",
            description="Gorąco",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        result = await toolbox.confirm_pending_action()

        assert result.error is None
        assert "Nowa reguła" in result.answer
        assert result.short_id == "R4D5E6"

        create_call = mock_rule_service.create_rule.call_args
        assert create_call is not None
        _, rule_create = create_call[0]
        assert isinstance(rule_create, RuleCreate)
        assert rule_create.schedule is None

    @pytest.mark.asyncio()
    async def test_confirm_edit_rule_still_works(
        self,
        toolbox: RulesToolbox,
        mock_rule_service: MagicMock,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.get_pending_confirmation.return_value = {
            "action": "edit_rule",
            "cel_expression": "wind_gusts_10m_kmh > 60",
            "explanation": "Silny wiatr",
            "validated": True,
            "location_id": 42,
            "chat_id": 200,
            "message_thread_id": 1,
            "stored_at": datetime.now(UTC).isoformat(),
            "edit_short_id": "R1A2B3",
        }
        existing_rule = NotificationRule(
            id=10,
            short_id="R1A2B3",
            user_id=100,
            telegram_chat_id=200,
            telegram_message_thread_id=1,
            location_id=42,
            expression="wind_gusts_10m_kmh > 60",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        mock_rule_service.get_rule.return_value = existing_rule
        mock_rule_service.update_rule.return_value = existing_rule

        result = await toolbox.confirm_pending_action()

        assert result.error is None
        assert "zaktualizowana" in result.answer

        mock_rule_service.get_rule_for_user.assert_awaited_once_with(100, short_id="R1A2B3")

    @pytest.mark.asyncio()
    async def test_confirm_edit_treats_other_users_rule_as_not_found(
        self,
        toolbox: RulesToolbox,
        mock_rule_service: MagicMock,
        mock_memory_service: MagicMock,
    ) -> None:
        mock_memory_service.get_pending_confirmation.return_value = {
            "action": "edit_rule",
            "cel_expression": "wind_gusts_10m_ms > 12",
            "explanation": "Silny wiatr",
            "validated": True,
            "location_id": 42,
            "chat_id": 200,
            "message_thread_id": 1,
            "stored_at": datetime.now(UTC).isoformat(),
            "edit_short_id": "R1A2B3",
        }
        mock_rule_service.get_rule_for_user.return_value = None

        result = await toolbox.confirm_pending_action()

        assert result.error == "Nie znaleziono reguły #R1A2B3"
        mock_rule_service.update_rule.assert_not_awaited()
