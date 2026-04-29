from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.locations import LocationService
from weather_agent.domain.rules.models import NotificationRule
from weather_agent.domain.rules.service import NotificationRuleService
from weather_agent.graphs.nodes.rule_management import (
    _build_system_prompt,
    cancel_rule_node,
    confirm_rule_node,
    is_confirmation_no,
    is_confirmation_yes,
    persist_rule_change_node,
    propose_cel_rule_node,
    require_user_confirmation_node,
)
from weather_agent.graphs.state import ConversationState
from weather_agent.llm.model_factory import ModelFactory


def _make_state(**overrides: object) -> ConversationState:
    base: ConversationState = {
        "authorized_user_id": 42,
        "chat_id": 999,
        "message_thread_id": None,
        "context_key": "999",
        "user_message": "jeśli będzie padać, powiadom mnie",
        "resolved_intent": "rule",
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
    base.update(overrides)
    return base


def _mock_model_factory(
    cel_expression: str | None,
    explanation: str = "Test explanation",
    short_id: str | None = None,
) -> MagicMock:
    from weather_agent.graphs.nodes.rule_management import RuleProposalExtraction

    res = RuleProposalExtraction(
        cel_expression=cel_expression,
        explanation=explanation,
        short_id=short_id,
    )
    mock_chat = AsyncMock()
    mock_chat.with_structured_output = MagicMock(return_value=mock_chat)
    mock_chat.ainvoke = AsyncMock(return_value=res)
    mock_factory = MagicMock(spec=ModelFactory)
    mock_factory.create_chat_model = MagicMock(return_value=mock_chat)
    return mock_factory


_VALID_CEL = 'max("wind_gusts_10m_ms", weekend()) >= 12'
_INVALID_CEL = "unknown_func(foo) > bar"




class TestBuildSystemPrompt:
    def test_prompt_contains_cel_functions(self) -> None:
        prompt = _build_system_prompt()
        assert "time_range_helpers" in prompt
        assert "aggregation" in prompt
        assert "max" in prompt
        assert "min" in prompt

    def test_prompt_contains_metrics(self) -> None:
        prompt = _build_system_prompt()
        assert "temperature_2m_c" in prompt
        assert "wind_speed_10m_ms" in prompt


class TestIsConfirmationYes:
    def test_tak(self) -> None:
        assert is_confirmation_yes("tak") is True

    def test_ok(self) -> None:
        assert is_confirmation_yes("ok") is True

    def test_potwierdzam(self) -> None:
        assert is_confirmation_yes("potwierdzam") is True

    def test_whitespace(self) -> None:
        assert is_confirmation_yes("  tak  ") is True

    def test_no(self) -> None:
        assert is_confirmation_yes("nie") is False

    def test_maybe(self) -> None:
        assert is_confirmation_yes("może") is False


class TestIsConfirmationNo:
    def test_nie(self) -> None:
        assert is_confirmation_no("nie") is True

    def test_anuluj(self) -> None:
        assert is_confirmation_no("anuluj") is True

    def test_rezygnuj(self) -> None:
        assert is_confirmation_no("rezygnuj") is True

    def test_whitespace(self) -> None:
        assert is_confirmation_no("  nie  ") is True

    def test_yes(self) -> None:
        assert is_confirmation_no("tak") is False


class TestProposeCelRuleNode:
    @pytest.mark.asyncio
    async def test_successful_proposal(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=_VALID_CEL,
            explanation="Powiadom, gdy porywy wiatru w weekend >= 12 m/s",
        )
        cel_evaluator = CELEvaluator()

        loc_ref = MagicMock()
        loc_ref.id = "5"
        state = _make_state(
            user_message="jeśli porywy wiatru w weekend będą powyżej 12 m/s, daj znać",
            resolved_location=loc_ref,
        )
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["cel_expression"] == _VALID_CEL
        assert result["error"] is None
        assert result["pending_confirmation"] is not None
        assert result["pending_confirmation"]["action"] == "create_rule"
        assert result["pending_confirmation"]["validated"] is True
        assert "porywy" in result["pending_confirmation"]["explanation"]
        assert result["pending_confirmation"]["location_id"] == 5

    @pytest.mark.asyncio
    async def test_validation_failure_invalid_cel(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=_INVALID_CEL,
            explanation="Jakiś opis",
        )
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="zrź coś dziwnego")
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["cel_expression"] is None
        assert result["error"] is not None
        assert "Przepraszam" in result["error"]

    @pytest.mark.asyncio
    async def test_llm_returns_null_expression(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=None,
            explanation="Nie da się zamienić na CEL",
        )
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="zrź coś niemożliwego")
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["cel_expression"] is None
        assert result["error"] is not None
        assert "Przepraszam" in result["error"]

    @pytest.mark.asyncio
    async def test_edit_short_id_detected(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=_VALID_CEL,
            explanation="Aktualizacja reguły",
            short_id="R7K2",
        )
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="dodaj temperaturę do #R7K2")
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["pending_confirmation"] is not None
        assert result["pending_confirmation"]["action"] == "edit_rule"
        assert result["pending_confirmation"]["edit_short_id"] == "R7K2"

    @pytest.mark.asyncio
    async def test_empty_user_message(self) -> None:
        mock_factory = _mock_model_factory(cel_expression=None)
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="")
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["error"] is not None
        assert "Brak wiadomości" in result["error"]

    @pytest.mark.asyncio
    async def test_llm_failure(self) -> None:
        mock_factory = MagicMock(spec=ModelFactory)
        mock_chat = AsyncMock()
        mock_chat.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))
        mock_factory.create_chat_model = MagicMock(return_value=mock_chat)
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="jakaś wiadomość")
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["error"] is not None
        assert "Przepraszam" in result["error"]

    @pytest.mark.asyncio
    async def test_malformed_llm_json(self) -> None:
        # Since we use structured output, malformed JSON is handled by the model
        # or raises an exception during structured output parsing.
        # This test is now less relevant for structured output but we'll mock a failure.
        mock_factory = MagicMock(spec=ModelFactory)
        mock_chat = AsyncMock()
        mock_chat.with_structured_output = MagicMock(return_value=mock_chat)
        mock_chat.ainvoke = AsyncMock(side_effect=ValueError("Malformed output"))
        mock_factory.create_chat_model = MagicMock(return_value=mock_chat)
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="dodaj regułę")
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        assert result["error"] is not None
        assert "Przepraszam" in result["error"]

    @pytest.mark.asyncio
    async def test_allowlist_in_model_context(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=_VALID_CEL,
            explanation="Test allowlist",
        )
        cel_evaluator = CELEvaluator()

        state = _make_state(user_message="powiadom o wietrze")
        await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        mock_chat = mock_factory.create_chat_model.return_value
        call_args = mock_chat.ainvoke.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert "temperature_2m_c" in system_msg.content
        assert "wind_gusts_10m_ms" in system_msg.content
        assert "aggregation" in system_msg.content

    @pytest.mark.asyncio
    async def test_pending_confirmation_includes_metadata(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=_VALID_CEL,
            explanation="Test metadata",
        )
        cel_evaluator = CELEvaluator()

        loc_ref = MagicMock()
        loc_ref.id = "10"
        state = _make_state(
            user_message="powiadom o wietrze",
            resolved_location=loc_ref,
            chat_id=999,
            message_thread_id=5,
        )
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        pending = result["pending_confirmation"]
        assert pending is not None
        assert pending["location_id"] == 10
        assert pending["chat_id"] == 999
        assert pending["message_thread_id"] == 5
        assert "stored_at" in pending

    @pytest.mark.asyncio
    async def test_pending_confirmation_without_location(self) -> None:
        mock_factory = _mock_model_factory(
            cel_expression=_VALID_CEL,
            explanation="Test no location",
        )
        cel_evaluator = CELEvaluator()

        state = _make_state(
            user_message="powiadom o wietrze",
            resolved_location=None,
        )
        result = await propose_cel_rule_node(state, mock_factory, cel_evaluator)

        pending = result["pending_confirmation"]
        assert pending is not None
        assert pending["location_id"] is None


class TestRequireUserConfirmationNode:
    @pytest.mark.asyncio
    async def test_proposes_new_rule_confirmation(self) -> None:
        state = _make_state(
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Gdy wiatr >= 12 m/s",
                "validated": True,
            },
        )
        result = await require_user_confirmation_node(state)

        assert result["answer"] is not None
        assert _VALID_CEL in result["answer"]
        assert "Gdy wiatr >= 12 m/s" in result["answer"]
        assert "tak/nie" in result["answer"].lower() or "Czy" in result["answer"]

    @pytest.mark.asyncio
    async def test_proposes_edit_rule_confirmation(self) -> None:
        state = _make_state(
            pending_confirmation={
                "action": "edit_rule",
                "edit_short_id": "R7K2",
                "cel_expression": _VALID_CEL,
                "explanation": "Aktualizacja",
                "validated": True,
            },
        )
        result = await require_user_confirmation_node(state)

        assert "R7K2" in result["answer"]
        assert "edycji" in result["answer"].lower()

    @pytest.mark.asyncio
    async def test_no_pending_confirmation(self) -> None:
        state = _make_state(pending_confirmation=None)
        result = await require_user_confirmation_node(state)

        assert "oczekującej" in result["answer"].lower() or "nie ma" in result["answer"].lower()


class TestConfirmRuleNode:
    @pytest.mark.asyncio
    async def test_confirm_create_rule(self) -> None:
        mock_rule = NotificationRule(
            id=1,
            short_id="R7K2",
            user_id=42,
            telegram_chat_id=999,
            telegram_message_thread_id=None,
            location_id=5,
            expression_language="cel",
            expression=_VALID_CEL,
            description="Opis",
            enabled=True,
            dry_run=False,
            cooldown_minutes=60,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.create_rule = AsyncMock(return_value=mock_rule)
        location_service = AsyncMock(spec=LocationService)

        loc_ref = MagicMock()
        loc_ref.id = "5"
        state = _make_state(
            user_message="tak",
            resolved_location=loc_ref,
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis reguły",
                "validated": True,
                "location_id": 5,
                "chat_id": 999,
                "message_thread_id": None,
            },
        )

        result = await confirm_rule_node(state, rule_service, location_service)

        assert result["error"] is None
        assert result["pending_confirmation"] is None
        assert "R7K2" in result["answer"]
        assert "zapisana" in result["answer"].lower()
        rule_service.create_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirm_edit_rule(self) -> None:
        mock_rule = NotificationRule(
            id=1,
            short_id="R7K2",
            user_id=42,
            telegram_chat_id=999,
            telegram_message_thread_id=None,
            location_id=5,
            expression_language="cel",
            expression=_VALID_CEL,
            description="Zaktualizowany opis",
            enabled=True,
            dry_run=False,
            cooldown_minutes=60,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.get_rule = AsyncMock(return_value=mock_rule)
        rule_service.update_rule = AsyncMock(return_value=mock_rule)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            user_message="tak",
            pending_confirmation={
                "action": "edit_rule",
                "edit_short_id": "R7K2",
                "cel_expression": _VALID_CEL,
                "explanation": "Zaktualizowany opis",
                "validated": True,
                "location_id": 5,
            },
        )

        result = await confirm_rule_node(state, rule_service, location_service)

        assert result["error"] is None
        assert "R7K2" in result["answer"]
        assert "zaktualizowana" in result["answer"].lower()
        rule_service.update_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirm_uses_location_from_pending(self) -> None:
        mock_rule = NotificationRule(
            id=1,
            short_id="RA1B2",
            user_id=42,
            telegram_chat_id=999,
            telegram_message_thread_id=None,
            location_id=7,
            expression_language="cel",
            expression=_VALID_CEL,
            description="Opis",
            enabled=True,
            dry_run=False,
            cooldown_minutes=60,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.create_rule = AsyncMock(return_value=mock_rule)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            user_message="tak",
            resolved_location=None,
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": 7,
                "chat_id": 999,
                "message_thread_id": None,
            },
        )

        result = await confirm_rule_node(state, rule_service, location_service)

        assert result["error"] is None
        rule_service.create_rule.assert_called_once()
        call_args = rule_service.create_rule.call_args
        assert call_args[0][1].location_id == 7

    @pytest.mark.asyncio
    async def test_confirm_no_pending(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(pending_confirmation=None)
        result = await confirm_rule_node(state, rule_service, location_service)

        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_confirm_unauthorized_user(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            authorized_user_id=None,
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": 5,
            },
        )

        result = await confirm_rule_node(state, rule_service, location_service)
        assert result["error"] is not None
        assert "autoryzowany" in result["error"].lower()
        assert result["pending_confirmation"] is None

    @pytest.mark.asyncio
    async def test_confirm_no_location(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": None,
            },
        )

        result = await confirm_rule_node(state, rule_service, location_service)
        assert result["error"] is not None
        assert "lokalizacj" in result["error"].lower()
        assert result["pending_confirmation"] is None

    @pytest.mark.asyncio
    async def test_confirm_edit_rule_not_found(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.get_rule = AsyncMock(return_value=None)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            pending_confirmation={
                "action": "edit_rule",
                "edit_short_id": "RXXXX",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": 5,
            },
        )

        result = await confirm_rule_node(state, rule_service, location_service)
        assert result["error"] is not None
        assert "nie znaleziono" in result["error"].lower()


class TestCancelRuleNode:
    @pytest.mark.asyncio
    async def test_cancel_create_rule(self) -> None:
        state = _make_state(
            user_message="nie",
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
            },
        )

        result = await cancel_rule_node(state)

        assert result["answer"] is not None
        assert "anulowana" in result["answer"].lower()
        assert result["pending_confirmation"] is None
        assert result["cel_expression"] is None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_cancel_edit_rule(self) -> None:
        state = _make_state(
            user_message="nie",
            pending_confirmation={
                "action": "edit_rule",
                "edit_short_id": "R7K2",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
            },
        )

        result = await cancel_rule_node(state)

        assert "R7K2" in result["answer"]
        assert "anulowana" in result["answer"].lower()
        assert result["pending_confirmation"] is None

    @pytest.mark.asyncio
    async def test_cancel_no_pending(self) -> None:
        state = _make_state(pending_confirmation=None)
        result = await cancel_rule_node(state)

        assert "brak" in result["answer"].lower() or "oczekującej" in result["answer"].lower()


class TestPersistRuleChangeNode:
    @pytest.mark.asyncio
    async def test_user_confirms_create_rule(self) -> None:
        mock_rule = NotificationRule(
            id=1,
            short_id="R7K2",
            user_id=42,
            telegram_chat_id=999,
            telegram_message_thread_id=None,
            location_id=5,
            expression_language="cel",
            expression=_VALID_CEL,
            description="Opis",
            enabled=True,
            dry_run=False,
            cooldown_minutes=60,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.create_rule = AsyncMock(return_value=mock_rule)
        location_service = AsyncMock(spec=LocationService)

        loc_ref = MagicMock()
        loc_ref.id = "5"
        state = _make_state(
            user_message="tak",
            resolved_location=loc_ref,
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis reguły",
                "validated": True,
                "location_id": 5,
            },
        )

        result = await persist_rule_change_node(state, rule_service, location_service)

        assert result["error"] is None
        assert result["pending_confirmation"] is None
        assert "R7K2" in result["answer"]
        assert "zapisana" in result["answer"].lower()
        rule_service.create_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_rejects_rule(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            user_message="nie",
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
            },
        )

        result = await persist_rule_change_node(state, rule_service, location_service)

        assert result["answer"] is not None
        assert "anulowana" in result["answer"].lower()
        assert result["pending_confirmation"] is None
        rule_service.create_rule.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_response(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            user_message="może",
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
            },
        )

        result = await persist_rule_change_node(state, rule_service, location_service)

        assert "potwierdzenie" in result["answer"].lower() or "oczekuję" in result["answer"].lower()
        rule_service.create_rule.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_pending_confirmation(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(pending_confirmation=None)
        result = await persist_rule_change_node(state, rule_service, location_service)

        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_unauthorized_user(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        loc_ref = MagicMock()
        loc_ref.id = "5"
        state = _make_state(
            authorized_user_id=None,
            user_message="tak",
            resolved_location=loc_ref,
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": 5,
            },
        )

        result = await persist_rule_change_node(state, rule_service, location_service)
        assert result["error"] is not None
        assert "autoryzowany" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_resolved_location(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        state = _make_state(
            user_message="tak",
            resolved_location=None,
            pending_confirmation={
                "action": "create_rule",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": None,
            },
        )

        result = await persist_rule_change_node(state, rule_service, location_service)
        assert result["error"] is not None
        assert "lokalizacj" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_edit_rule_not_found(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.get_rule = AsyncMock(return_value=None)
        location_service = AsyncMock(spec=LocationService)

        loc_ref = MagicMock()
        loc_ref.id = "5"
        state = _make_state(
            user_message="tak",
            resolved_location=loc_ref,
            pending_confirmation={
                "action": "edit_rule",
                "edit_short_id": "RXXXX",
                "cel_expression": _VALID_CEL,
                "explanation": "Opis",
                "validated": True,
                "location_id": 5,
            },
        )

        result = await persist_rule_change_node(state, rule_service, location_service)
        assert result["error"] is not None
        assert "nie znaleziono" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_polish_yes_variants(self) -> None:
        mock_rule = NotificationRule(
            id=1,
            short_id="RA1B2",
            user_id=42,
            telegram_chat_id=999,
            telegram_message_thread_id=None,
            location_id=5,
            expression_language="cel",
            expression=_VALID_CEL,
            description="Opis",
            enabled=True,
            dry_run=False,
            cooldown_minutes=60,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        rule_service = AsyncMock(spec=NotificationRuleService)
        rule_service.create_rule = AsyncMock(return_value=mock_rule)
        location_service = AsyncMock(spec=LocationService)

        for word in ("tak", "potwierdzam", "ok"):
            rule_service.create_rule.reset_mock()
            loc_ref = MagicMock()
            loc_ref.id = "5"
            state = _make_state(
                user_message=word,
                resolved_location=loc_ref,
                pending_confirmation={
                    "action": "create_rule",
                    "cel_expression": _VALID_CEL,
                    "explanation": "Opis",
                    "validated": True,
                    "location_id": 5,
                },
            )

            result = await persist_rule_change_node(state, rule_service, location_service)
            assert result["error"] is None, f"Expected success for '{word}'"
            rule_service.create_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_polish_no_variants(self) -> None:
        rule_service = AsyncMock(spec=NotificationRuleService)
        location_service = AsyncMock(spec=LocationService)

        for word in ("nie", "anuluj", "rezygnuj"):
            rule_service.create_rule.reset_mock()
            state = _make_state(
                user_message=word,
                pending_confirmation={
                    "action": "create_rule",
                    "cel_expression": _VALID_CEL,
                    "explanation": "Opis",
                    "validated": True,
                },
            )

            result = await persist_rule_change_node(state, rule_service, location_service)
            assert result["answer"] is not None
            rule_service.create_rule.assert_not_called()
