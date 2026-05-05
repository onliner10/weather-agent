from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from weather_agent.domain.rule_expression.evaluator import RuleExpressionEvaluator
from weather_agent.domain.rules.schedule import parse_schedule
from weather_agent.eval import notification_rule_proposal_targets
from weather_agent.eval.notification_rule_dataset import (
    DATASET_NAME,
    generate_notification_rule_cases,
)
from weather_agent.eval.notification_rule_evaluators import notification_rule_proposal_fidelity
from weather_agent.eval.notification_rule_proposal_targets import (
    RecordingRulesToolbox,
    build_notification_rule_async_target_from_factory,
)
from weather_agent.eval.notification_rule_schemas import (
    ExpectedRuleProposal,
    RuleToolCallRecord,
)


def _case_expected(case_id: str) -> ExpectedRuleProposal:
    return next(case.expected for case in generate_notification_rule_cases() if case.id == case_id)


def _reference(case_id: str) -> dict[str, object]:
    case = next(case for case in generate_notification_rule_cases() if case.id == case_id)
    return {
        "expected": case.expected.model_dump(mode="json"),
        "current_time": case.current_time.isoformat(),
    }


def _output(
    *,
    case_id: str,
    call: RuleToolCallRecord,
    answer: str = "Czy chcesz potwierdzić? (tak/nie)",
    extra_calls: list[RuleToolCallRecord] | None = None,
) -> dict[str, object]:
    calls = [*(extra_calls or []), call]
    return {
        "example_id": case_id,
        "answer": answer,
        "tool_calls": [tool_call.model_dump(mode="json") for tool_call in calls],
    }


def _proposal_call(
    *,
    rule_expression: str,
    location: str,
    pending: bool | None = True,
    error: str | None = None,
) -> RuleToolCallRecord:
    return RuleToolCallRecord(
        name="propose_notification_rule",
        args={
            "rule_expression": rule_expression,
            "explanation": "Powiadomienie testowe",
            "location_name": location,
            "edit_short_id": "",
        },
        result_pending=pending,
        result_error=error,
    )


def _schedule_call(
    *,
    rule_expression: str,
    location: str,
    schedule_type: str,
    schedule_expression: str,
    pending: bool | None = True,
    error: str | None = None,
) -> RuleToolCallRecord:
    return RuleToolCallRecord(
        name="schedule_notification",
        args={
            "schedule_type": schedule_type,
            "schedule_expression": schedule_expression,
            "explanation": "Powiadomienie testowe",
            "location_name": location,
            "rule_expression": rule_expression,
        },
        result_pending=pending,
        result_error=error,
    )


class TestNotificationRuleDataset:
    def test_dataset_name_is_versioned(self) -> None:
        assert DATASET_NAME == "weather-agent-notification-rule-proposal-v2"

    def test_generates_ten_unambiguous_cases(self) -> None:
        cases = generate_notification_rule_cases()

        assert len(cases) == 10
        assert len({case.id for case in cases}) == len(cases)
        assert all("pogorszenie" not in case.question.casefold() for case in cases)

    def test_expected_rule_expression_and_schedules_are_valid(self) -> None:
        evaluator = RuleExpressionEvaluator()

        for case in generate_notification_rule_cases():
            validation = evaluator.validate(case.expected.expected_rule_expression)
            assert validation.valid, case.id
            if case.expected.expected_schedule_type is not None:
                parsed = parse_schedule(
                    f"{case.expected.expected_schedule_type}:"
                    f"{case.expected.expected_schedule_expression}"
                )
                assert parsed.valid, case.id


class TestNotificationRuleEvaluator:
    def test_all_reference_cases_pass_with_expected_tool_arguments(self) -> None:
        for case in generate_notification_rule_cases():
            expected = case.expected
            call = (
                _schedule_call(
                    rule_expression=expected.expected_rule_expression,
                    location=expected.expected_location,
                    schedule_type=str(expected.expected_schedule_type),
                    schedule_expression=str(expected.expected_schedule_expression),
                )
                if expected.expected_tool == "schedule_notification"
                else _proposal_call(
                    rule_expression=expected.expected_rule_expression,
                    location=expected.expected_location,
                )
            )

            result = notification_rule_proposal_fidelity(
                _output(case_id=case.id, call=call),
                {
                    "expected": expected.model_dump(mode="json"),
                    "current_time": case.current_time.isoformat(),
                },
            )

            assert result["score"] == 1.0, (case.id, result["comment"])

    def test_exact_expected_rule_expression_passes(self) -> None:
        expected = _case_expected("rule-proposal-001")
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression=expected.expected_rule_expression, location="Warszawa"
                ),
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 1.0

    def test_semantically_equivalent_rule_expression_passes_without_exact_string_match(
        self,
    ) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression="max_metric('wind_gusts_10m_ms', weekend()) > 12",
                    location="warszawa",
                ),
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 1.0

    def test_wrong_metric_fails(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression='max_metric("wind_speed_10m_ms", weekend()) > 12',
                    location="Warszawa",
                ),
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 0.0
        assert "metric_mismatch" in str(result["comment"])

    def test_wrong_threshold_fails(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression='max_metric("wind_gusts_10m_ms", weekend()) > 13',
                    location="Warszawa",
                ),
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 0.0
        assert "rule_expression_profile_mismatch:true_case" in str(result["comment"])

    def test_wrong_aggregation_fails(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-005",
                call=_proposal_call(
                    rule_expression='max_metric("precipitation_mm", next_hours(hours(6))) > 5',
                    location="Chwarzno",
                ),
            ),
            _reference("rule-proposal-005"),
        )

        assert result["score"] == 0.0
        assert "aggregation_guard" in str(result["comment"])

    def test_wrong_time_helper_fails(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-007",
                call=_schedule_call(
                    rule_expression='max_metric("wind_speed_10m_ms", today()) > 10',
                    location="Gdynia",
                    schedule_type="once",
                    schedule_expression="2026-05-02T08:00:00+02:00",
                ),
            ),
            _reference("rule-proposal-007"),
        )

        assert result["score"] == 0.0
        assert "outside_time_guard" in str(result["comment"])

    def test_invalid_rule_expression_fails(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression="not valid rule expression", location="Warszawa"
                ),
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 0.0
        assert "invalid_rule_expression" in str(result["comment"])

    def test_non_boolean_rule_expression_fails(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-006",
                call=_schedule_call(
                    rule_expression="1",
                    location="Warszawa",
                    schedule_type="cron",
                    schedule_expression="0 7 * * *",
                ),
            ),
            _reference("rule-proposal-006"),
        )

        assert result["score"] == 0.0
        assert "rule_expression_profile_error:true_case" in str(result["comment"])

    def test_forbidden_confirmation_call_fails(self) -> None:
        expected = _case_expected("rule-proposal-001")
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression=expected.expected_rule_expression, location="Warszawa"
                ),
                extra_calls=[
                    RuleToolCallRecord(
                        name="confirm_pending_action",
                        args={},
                        result_error="not allowed",
                    )
                ],
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 0.0
        assert "forbidden_tool_call:confirm_pending_action" in str(result["comment"])

    def test_wrong_schedule_type_expression_and_location_fail(self) -> None:
        expected = _case_expected("rule-proposal-006")
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-006",
                call=_schedule_call(
                    rule_expression=expected.expected_rule_expression,
                    location="Kraków",
                    schedule_type="once",
                    schedule_expression="2026-05-02T07:00:00+02:00",
                ),
            ),
            _reference("rule-proposal-006"),
        )

        assert result["score"] == 0.0
        comment = str(result["comment"])
        assert "location_mismatch" in comment
        assert "schedule_type_mismatch" in comment

    def test_recurring_rc_request_requires_weekday_cron_schedule(self) -> None:
        expected = _case_expected("rule-proposal-009")
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-009",
                call=_schedule_call(
                    rule_expression=expected.expected_rule_expression,
                    location="Chwarzno",
                    schedule_type="cron",
                    schedule_expression="0 8-18 * * *",
                ),
            ),
            _reference("rule-proposal-009"),
        )

        assert result["score"] == 0.0
        assert "cron_mismatch:expected=0 8-18 * * 1-5" in str(result["comment"])

    def test_recurring_rc_request_rejects_static_date_range_expression(self) -> None:
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-009",
                call=_schedule_call(
                    rule_expression=(
                        'max_metric("wind_gusts_10m_ms", '
                        'date_range("2026-05-04T08:00:00+02:00", '
                        '"2026-05-08T18:00:00+02:00")) <= 6.0 && '
                        'max_metric("wind_speed_10m_ms", '
                        'date_range("2026-05-04T08:00:00+02:00", '
                        '"2026-05-08T18:00:00+02:00")) <= 4.0 && '
                        'sum_metric("precipitation_mm", '
                        'date_range("2026-05-04T08:00:00+02:00", '
                        '"2026-05-08T18:00:00+02:00")) == 0.0'
                    ),
                    location="Chwarzno",
                    schedule_type="cron",
                    schedule_expression="0 8-18 * * 1-5",
                ),
            ),
            _reference("rule-proposal-009"),
        )

        assert result["score"] == 0.0
        assert "rule_expression_profile_error" in str(result["comment"])

    def test_each_thursday_request_requires_day_of_week_cron(self) -> None:
        expected = _case_expected("rule-proposal-010")
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-010",
                call=_schedule_call(
                    rule_expression=expected.expected_rule_expression,
                    location="Gdańsk",
                    schedule_type="cron",
                    schedule_expression="0 8 * * *",
                ),
            ),
            _reference("rule-proposal-010"),
        )

        assert result["score"] == 0.0
        assert "cron_mismatch:expected=0 8 * * 4" in str(result["comment"])

    def test_missing_confirmation_surface_fails(self) -> None:
        expected = _case_expected("rule-proposal-001")
        result = notification_rule_proposal_fidelity(
            _output(
                case_id="rule-proposal-001",
                call=_proposal_call(
                    rule_expression=expected.expected_rule_expression, location="Warszawa"
                ),
                answer="Gotowe.",
            ),
            _reference("rule-proposal-001"),
        )

        assert result["score"] == 0.0
        assert "missing_confirmation_surface" in str(result["comment"])


class TestNotificationRuleTarget:
    async def test_async_target_uses_production_agent_and_records_tool_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        created_models: list[MagicMock] = []

        class FakeAgent:
            async def ainvoke(
                self,
                payload: dict[str, object],
                **kwargs: object,
            ) -> dict[str, object]:
                del kwargs
                calls["payload"] = payload
                tools = calls["tools"]
                assert isinstance(tools, list)
                propose = next(tool for tool in tools if tool.name == "propose_notification_rule")
                await propose.coroutine(
                    rule_expression='max_metric("wind_gusts_10m_ms", weekend()) > 12.0',
                    explanation="Silne porywy wiatru",
                    location_name="Warszawa",
                )
                return {"messages": [AIMessage(content="Czy chcesz potwierdzić? (tak/nie)")]}

        def fake_create_weather_agent(**kwargs: object) -> FakeAgent:
            calls["agent_kwargs"] = kwargs
            calls["tools"] = kwargs["tools"]
            return FakeAgent()

        def model_factory() -> MagicMock:
            model = MagicMock()
            created_models.append(model)
            return model

        monkeypatch.setattr(
            notification_rule_proposal_targets,
            "create_weather_agent",
            fake_create_weather_agent,
        )

        target = build_notification_rule_async_target_from_factory(model_factory)
        result = await target(
            {
                "id": "rule-proposal-001",
                "question": (
                    "Powiadom mnie, jeśli porywy wiatru w weekend w Warszawie będą powyżej 12 m/s."
                ),
                "current_time": "2026-05-01T12:00:00+02:00",
            }
        )

        assert result["example_id"] == "rule-proposal-001"
        assert result["answer"] == "Czy chcesz potwierdzić? (tak/nie)"
        assert len(result["tool_calls"]) == 1
        recorded = result["tool_calls"][0]
        assert recorded["name"] == "propose_notification_rule"
        assert recorded["result_pending"] is True
        assert recorded["result_error"] is None
        agent_kwargs = calls["agent_kwargs"]
        assert isinstance(agent_kwargs, dict)
        assert agent_kwargs["model"] is created_models[0]
        assert "2026-05-01 12:00" in str(agent_kwargs["system_prompt_suffix"])

    def test_recording_tools_expose_expected_names_and_schemas(self) -> None:
        tools = RecordingRulesToolbox().to_langchain_tools()

        by_name = {tool.name: tool for tool in tools}
        assert {
            "list_notification_rules",
            "get_rule_expression_capabilities",
            "propose_notification_rule",
            "confirm_pending_action",
            "cancel_pending_action",
            "schedule_notification",
        } == set(by_name)
        assert by_name["propose_notification_rule"].args_schema is not None
        assert by_name["schedule_notification"].args_schema is not None
