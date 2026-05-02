from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from weather_agent.eval import location_management_targets
from weather_agent.eval.location_management_dataset import (
    DATASET_NAME,
    generate_location_management_cases,
)
from weather_agent.eval.location_management_evaluators import location_management_fidelity
from weather_agent.eval.location_management_schemas import (
    ExpectedLocationAction,
    LocationToolCallRecord,
)
from weather_agent.eval.location_management_targets import (
    RecordingLocationToolbox,
    build_location_management_async_target_from_factory,
)


def _case_expected(case_id: str) -> ExpectedLocationAction:
    return next(
        case.expected for case in generate_location_management_cases() if case.id == case_id
    )


def _reference(case_id: str) -> dict[str, object]:
    expected = _case_expected(case_id)
    return {"expected": expected.model_dump(mode="json")}


def _output(
    *,
    case_id: str,
    call: LocationToolCallRecord | None,
    answer: str = "Zaktualizowałem lokalizację.",
    extra_calls: list[LocationToolCallRecord] | None = None,
) -> dict[str, object]:
    calls = [*(extra_calls or [])]
    if call is not None:
        calls.append(call)
    return {
        "example_id": case_id,
        "answer": answer,
        "tool_calls": [tool_call.model_dump(mode="json") for tool_call in calls],
    }


def _call(
    name: str,
    *,
    location: str = "",
    alias: str = "",
    aliases: list[str] | None = None,
    new_name: str = "",
    error: str | None = None,
    hard_delete: bool = False,
) -> LocationToolCallRecord:
    args: dict[str, object] = {"location_name": location}
    if alias:
        args["alias"] = alias
    if aliases is not None:
        args["aliases"] = aliases
    if new_name:
        args["new_name"] = new_name
    if hard_delete:
        args["hard_delete"] = True
    return LocationToolCallRecord(
        name=name,
        args=args,
        result_error=error,
        result_success=None if error else "ok",
    )


class TestLocationManagementDataset:
    def test_dataset_name_is_versioned(self) -> None:
        assert DATASET_NAME == "weather-agent-location-management-v1"

    def test_generates_expected_cases(self) -> None:
        cases = generate_location_management_cases()

        assert len(cases) == 11
        assert len({case.id for case in cases}) == len(cases)
        assert {case.expected.expected_tool for case in cases} == {
            "save_location",
            "edit_location",
            "remove_location",
            "get_forecast",
            "get_observations",
            "propose_notification_rule",
            "ask_for_location",
        }


class TestLocationManagementEvaluator:
    def test_expected_add_location_output_passes(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-001",
                call=_call("save_location", location="Gdańsk, Piotrkowska 11b", alias="dom"),
                answer="Zapamiętałem lokalizację Gdańsk, Piotrkowska 11b jako dom.",
            ),
            _reference("location-management-001"),
        )

        assert result["score"] == 1.0

    def test_default_forecast_accepts_empty_location_arg(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-005",
                call=_call("get_forecast", location=""),
                answer="Prognoza dla Chwarzna: jutro 12°C.",
            ),
            _reference("location-management-005"),
        )

        assert result["score"] == 1.0

    def test_default_rule_accepts_blank_location(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-006",
                call=_call("propose_notification_rule", location=""),
                answer="Czy chcesz potwierdzić? (tak/nie)",
            ),
            _reference("location-management-006"),
        )

        assert result["score"] == 1.0

    def test_default_observations_accept_empty_location_arg(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-010",
                call=_call("get_observations", location=""),
                answer="W Chwarznie teraz nie pada.",
            ),
            _reference("location-management-010"),
        )

        assert result["score"] == 1.0

    def test_rename_edit_requires_new_name(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-008",
                call=_call("edit_location", location="praca", new_name="biuro"),
                answer="Zaktualizowałem lokalizację praca na biuro.",
            ),
            _reference("location-management-008"),
        )

        assert result["score"] == 1.0

    def test_wrong_new_name_fails(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-008",
                call=_call("edit_location", location="praca", new_name="dom"),
            ),
            _reference("location-management-008"),
        )

        assert result["score"] == 0.0
        assert "new_name_mismatch" in str(result["comment"])

    def test_missing_default_without_tool_call_passes_when_answer_asks_for_location(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-007",
                call=None,
                answer="Podaj lokalizację, dla której mam sprawdzić pogodę.",
            ),
            _reference("location-management-007"),
        )

        assert result["score"] == 1.0

    def test_wrong_tool_fails(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-004",
                call=_call("edit_location", location="praca"),
            ),
            _reference("location-management-004"),
        )

        assert result["score"] == 0.0
        assert "terminal_tool_mismatch" in str(result["comment"])

    def test_wrong_location_fails(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-004",
                call=_call("remove_location", location="dom"),
            ),
            _reference("location-management-004"),
        )

        assert result["score"] == 0.0
        assert "location_mismatch" in str(result["comment"])

    def test_destructive_hard_delete_fails(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-004",
                call=_call("remove_location", location="praca", hard_delete=True),
            ),
            _reference("location-management-004"),
        )

        assert result["score"] == 0.0
        assert "destructive_remove_requested" in str(result["comment"])

    def test_non_polish_irrelevant_answer_fails(self) -> None:
        result = location_management_fidelity(
            _output(
                case_id="location-management-001",
                call=_call("save_location", location="Chwarzno", alias="dom"),
                answer="Done.",
            ),
            _reference("location-management-001"),
        )

        assert result["score"] == 0.0
        assert "missing_polish_location_response" in str(result["comment"])


class TestLocationManagementTarget:
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
                save = next(tool for tool in tools if tool.name == "save_location")
                await save.coroutine(location_name="Gdańsk, Piotrkowska 11b", alias="dom")
                return {
                    "messages": [
                        AIMessage(content="Zapamiętałem lokalizację Gdańsk, Piotrkowska 11b.")
                    ]
                }

        def fake_create_weather_agent(**kwargs: object) -> FakeAgent:
            calls["agent_kwargs"] = kwargs
            calls["tools"] = kwargs["tools"]
            return FakeAgent()

        def model_factory() -> MagicMock:
            model = MagicMock()
            created_models.append(model)
            return model

        monkeypatch.setattr(
            location_management_targets,
            "create_weather_agent",
            fake_create_weather_agent,
        )

        target = build_location_management_async_target_from_factory(model_factory)
        result = await target(
            {
                "id": "location-management-001",
                "question": "Zapamiętaj Gdańsk, Piotrkowska 11b jako dom.",
                "current_time": "2026-05-01T12:00:00+02:00",
                "seed_locations": [],
            }
        )

        assert result["example_id"] == "location-management-001"
        assert result["answer"] == "Zapamiętałem lokalizację Gdańsk, Piotrkowska 11b."
        assert len(result["tool_calls"]) == 1
        recorded = result["tool_calls"][0]
        assert recorded["name"] == "save_location"
        assert recorded["args"]["alias"] == "dom"
        agent_kwargs = calls["agent_kwargs"]
        assert isinstance(agent_kwargs, dict)
        assert agent_kwargs["model"] is created_models[0]
        assert "2026-05-01 12:00" in str(agent_kwargs["system_prompt_suffix"])

    def test_recording_tools_expose_expected_names_and_schemas(self) -> None:
        tools = RecordingLocationToolbox([]).to_langchain_tools()

        by_name = {tool.name: tool for tool in tools}
        assert {
            "get_forecast",
            "get_observations",
            "save_location",
            "edit_location",
            "remove_location",
            "list_locations",
        } == set(by_name)
        assert by_name["edit_location"].args_schema is not None
        assert by_name["remove_location"].args_schema is not None
