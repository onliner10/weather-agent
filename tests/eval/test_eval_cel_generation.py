from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from dataset import EVAL_CASES, EvalCase

from weather_agent.domain.cel.allowlist import ALL_ALLOWED_FUNCTION_NAMES, ALLOWED_METRICS
from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.graphs.nodes.rule_management import propose_cel_rule_node
from weather_agent.graphs.state import ConversationState

CEL_CASES = [c for c in EVAL_CASES if c.category in ("rule_create", "rule_edit", "rule_delete")]


def _state(message: str, **overrides: object) -> ConversationState:
    base: ConversationState = {
        "authorized_user_id": 12345,
        "chat_id": 999,
        "message_thread_id": None,
        "context_key": "999",
        "user_message": message,
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
    base.update(overrides)
    return base


def _mock_model_factory_with_cel(
    cel_expression: str | None, explanation: str = "Test"
) -> AsyncMock:
    factory = AsyncMock()
    chat = AsyncMock()
    response = AsyncMock()
    response.content = json.dumps(
        {
            "cel_expression": cel_expression,
            "explanation": explanation,
        }
    )
    chat.ainvoke = AsyncMock(return_value=response)
    factory.create_chat_model = lambda: chat
    return factory


class TestCELValidationForExpectedExpressions:
    @pytest.mark.parametrize(
        "case",
        [c for c in CEL_CASES if c.expected_cel is not None],
        ids=[c.id for c in CEL_CASES if c.expected_cel is not None],
    )
    def test_expected_cel_validates(self, case: EvalCase) -> None:
        assert case.expected_cel is not None
        evaluator = CELEvaluator()
        result = evaluator.validate(case.expected_cel)
        assert result.valid, (
            f"Case {case.id}: expected CEL {case.expected_cel!r} is invalid: {result.error}"
        )
        assert result.error is None

    @pytest.mark.parametrize(
        "case",
        [c for c in CEL_CASES if c.expected_cel is not None],
        ids=[c.id for c in CEL_CASES if c.expected_cel is not None],
    )
    def test_expected_cel_uses_only_allowed_functions(self, case: EvalCase) -> None:
        assert case.expected_cel is not None
        evaluator = CELEvaluator()
        result = evaluator.validate(case.expected_cel)
        assert result.valid
        for func_name in result.evaluated_functions:
            assert func_name in ALL_ALLOWED_FUNCTION_NAMES, (
                f"Case {case.id}: function {func_name!r} not in allowlist"
            )

    @pytest.mark.parametrize(
        "case",
        [c for c in CEL_CASES if c.expected_cel is not None],
        ids=[c.id for c in CEL_CASES if c.expected_cel is not None],
    )
    def test_expected_cel_uses_only_allowed_metrics(self, case: EvalCase) -> None:
        assert case.expected_cel is not None
        evaluator = CELEvaluator()
        result = evaluator.validate(case.expected_cel)
        assert result.valid
        for metric in result.evaluated_metrics:
            assert metric in ALLOWED_METRICS, (
                f"Case {case.id}: metric {metric!r} not in allowed metrics"
            )


class TestProposeCelRuleNodeWithMockedLLM:
    @pytest.mark.parametrize(
        "case",
        [c for c in CEL_CASES if c.expected_cel is not None],
        ids=[c.id for c in CEL_CASES if c.expected_cel is not None],
    )
    @pytest.mark.asyncio
    async def test_proposed_cel_validates_and_produces_confirmation(self, case: EvalCase) -> None:
        assert case.expected_cel is not None
        mock_factory = _mock_model_factory_with_cel(case.expected_cel)
        evaluator = CELEvaluator()
        state = _state(message=case.input_message)
        result = await propose_cel_rule_node(state, mock_factory, evaluator)

        assert result.get("error") is None or result.get("cel_expression") is not None, (
            f"Case {case.id}: unexpected error: {result.get('error')}"
        )
        if result.get("cel_expression") is not None:
            validation = evaluator.validate(result["cel_expression"])
            assert validation.valid, (
                f"Case {case.id}: produced CEL {result['cel_expression']!r} is invalid"
            )

        pending = result.get("pending_confirmation")
        assert pending is not None, (
            f"Case {case.id}: expected pending_confirmation for rule creation"
        )
        assert pending.get("validated") is True
        assert pending.get("cel_expression") is not None

    @pytest.mark.parametrize(
        "case",
        [c for c in CEL_CASES if c.category == "rule_edit"],
        ids=[c.id for c in CEL_CASES if c.category == "rule_edit"],
    )
    @pytest.mark.asyncio
    async def test_edit_rule_detects_short_id(self, case: EvalCase) -> None:
        cel_expr = 'max("wind_speed_10m_ms", weekend()) > 10.0'
        mock_factory = _mock_model_factory_with_cel(cel_expr)
        evaluator = CELEvaluator()
        state = _state(message=case.input_message)
        result = await propose_cel_rule_node(state, mock_factory, evaluator)

        pending = result.get("pending_confirmation")
        if pending is not None:
            assert pending.get("action") in ("edit_rule", "create_rule")

    @pytest.mark.asyncio
    async def test_llm_failure_returns_error(self) -> None:
        mock_factory = AsyncMock()
        mock_chat = AsyncMock()
        mock_chat.ainvoke = AsyncMock(side_effect=RuntimeError("Model unavailable"))
        mock_factory.create_chat_model = lambda: mock_chat
        evaluator = CELEvaluator()
        state = _state(message="powiadom mnie gdy będzie wietrznie")
        result = await propose_cel_rule_node(state, mock_factory, evaluator)
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_null_cel_expression_returns_error(self) -> None:
        mock_factory = _mock_model_factory_with_cel(None, "Cannot interpret")
        evaluator = CELEvaluator()
        state = _state(message="zrób coś niezrozumiałego")
        result = await propose_cel_rule_node(state, mock_factory, evaluator)
        assert result.get("error") is not None
        assert result.get("cel_expression") is None

    @pytest.mark.asyncio
    async def test_invalid_cel_expression_returned_by_llm(self) -> None:
        mock_factory = _mock_model_factory_with_cel(
            'unknown_func("temperature") > 10', "Bad expression"
        )
        evaluator = CELEvaluator()
        state = _state(message="powiadom mnie o czymś")
        result = await propose_cel_rule_node(state, mock_factory, evaluator)
        assert result.get("error") is not None
        assert result.get("pending_confirmation") is None

    @pytest.mark.asyncio
    async def test_malformed_json_response(self) -> None:
        factory = AsyncMock()
        chat = AsyncMock()
        response = AsyncMock()
        response.content = "this is not json"
        chat.ainvoke = AsyncMock(return_value=response)
        factory.create_chat_model = lambda: chat
        evaluator = CELEvaluator()
        state = _state(message="powiadom mnie o deszczu")
        result = await propose_cel_rule_node(state, factory, evaluator)
        assert result.get("error") is not None


class TestCELEvalDatasetCompleteness:
    def test_at_least_20_cases(self) -> None:
        assert len(EVAL_CASES) >= 20, f"Expected at least 20 eval cases, got {len(EVAL_CASES)}"

    def test_all_categories_covered(self) -> None:
        categories = {c.category for c in EVAL_CASES}
        required = {
            "weather_qa",
            "rule_create",
            "rule_edit",
            "rule_delete",
            "location_resolve",
            "time_resolve",
            "ambiguity",
            "provider_failure",
        }
        missing = required - categories
        assert not missing, f"Missing categories: {missing}"

    def test_all_cases_have_unique_ids(self) -> None:
        ids = [c.id for c in EVAL_CASES]
        assert len(ids) == len(set(ids)), "Duplicate eval case IDs found"

    def test_all_cases_have_input_messages(self) -> None:
        for case in EVAL_CASES:
            assert case.input_message.strip(), f"Case {case.id} has empty input_message"

    def test_weather_qa_cases_have_intents(self) -> None:
        weather_cases = [c for c in EVAL_CASES if c.category == "weather_qa"]
        for case in weather_cases:
            assert case.expected_intent is not None, (
                f"Case {case.id} (weather_qa) missing expected_intent"
            )

    def test_rule_cases_have_intents(self) -> None:
        rule_cases = [
            c for c in EVAL_CASES if c.category in ("rule_create", "rule_edit", "rule_delete")
        ]
        for case in rule_cases:
            assert case.expected_intent == "rule", (
                f"Case {case.id} ({case.category}) should have intent 'rule', "
                f"got {case.expected_intent!r}"
            )

    def test_cel_expressions_are_valid(self) -> None:
        evaluator = CELEvaluator()
        cases_with_cel = [c for c in EVAL_CASES if c.expected_cel is not None]
        for case in cases_with_cel:
            result = evaluator.validate(case.expected_cel)
            assert result.valid, (
                f"Case {case.id}: CEL {case.expected_cel!r} is invalid: {result.error}"
            )

    def test_ambiguity_cases_exist(self) -> None:
        ambiguity_cases = [c for c in EVAL_CASES if c.category == "ambiguity"]
        assert len(ambiguity_cases) >= 1, "At least one ambiguity case required"

    def test_provider_failure_cases_exist(self) -> None:
        failure_cases = [c for c in EVAL_CASES if c.category == "provider_failure"]
        assert len(failure_cases) >= 1, "At least one provider_failure case required"
