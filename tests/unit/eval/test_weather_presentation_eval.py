from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from weather_agent.eval import targets
from weather_agent.eval.evaluators import weather_presentation_tool_use
from weather_agent.eval.schemas import WeatherFacts
from weather_agent.eval.targets import build_weather_presentation_async_target_from_factory
from weather_agent.eval.weather_presentation_dataset import (
    DATASET_NAME,
    generate_weather_presentation_cases,
)


def _output(
    *,
    tool_name: str | None,
    attachment_count: int,
    answer: str = "Dołączam wykres.",
    tool_calls: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    effective_tool_calls = (
        tool_calls
        if tool_calls is not None
        else ([] if tool_name is None else [{"name": tool_name, "args": {}}])
    )
    return {
        "example_id": "weather-presentation-001",
        "answer": answer,
        "tool_calls": effective_tool_calls,
        "attachment_count": attachment_count,
    }


def _chart_tool_call_args(
    *,
    variables: list[str],
    field: str,
    start_date: str = "2026-05-04",
    end_date: str = "2026-05-04",
) -> dict[str, object]:
    return {
        "name": "render_forecast_chart",
        "args": {
            "location_name": "Chwarzno",
            "start_date": start_date,
            "end_date": end_date,
            "variables": variables,
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "data": {"name": "forecast"},
                "mark": "line",
                "encoding": {
                    "x": {"field": "time", "type": "temporal"},
                    "y": {"field": field, "type": "quantitative"},
                },
            },
        },
    }


class TestWeatherPresentationToolUse:
    def test_expected_chart_passes_with_render_call_and_attachment(self) -> None:
        result = weather_presentation_tool_use(
            _output(tool_name="render_forecast_chart", attachment_count=1),
            {"expect_chart": True},
        )

        assert result["score"] == 1.0
        assert result["comment"] == "ok"

    def test_expected_chart_fails_without_render_call(self) -> None:
        result = weather_presentation_tool_use(
            _output(tool_name="get_forecast", attachment_count=0),
            {"expect_chart": True},
        )

        assert result["score"] == 0.0
        assert "missing_render_forecast_chart_call" in str(result["comment"])
        assert "missing_chart_attachment" in str(result["comment"])

    def test_text_only_case_fails_on_unexpected_chart(self) -> None:
        result = weather_presentation_tool_use(
            _output(tool_name="render_forecast_chart", attachment_count=1),
            {"expect_chart": False},
        )

        assert result["score"] == 0.0
        assert "unexpected_render_forecast_chart_call" in str(result["comment"])

    def test_optional_chart_policy_allows_text_only_answer(self) -> None:
        result = weather_presentation_tool_use(
            _output(tool_name="get_forecast", attachment_count=0),
            {"expect_chart": None},
        )

        assert result["score"] == 1.0
        assert result["comment"] == "ok"

    def test_optional_chart_policy_fails_when_chosen_chart_has_no_attachment(self) -> None:
        result = weather_presentation_tool_use(
            _output(tool_name="render_forecast_chart", attachment_count=0),
            {"expect_chart": None},
        )

        assert result["score"] == 0.0
        assert "chart_call_without_attachment" in str(result["comment"])

    def test_expected_chart_fails_on_repeated_successful_render_calls(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=2,
                tool_calls=[
                    {"name": "render_forecast_chart", "args": {}, "result_error": None},
                    {"name": "render_forecast_chart", "args": {}, "result_error": None},
                ],
            ),
            {"expect_chart": True},
        )

        assert result["score"] == 0.0
        assert "repeated_successful_render_forecast_chart_calls:2" in str(result["comment"])

    def test_expected_chart_allows_retry_after_failed_render_call(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=1,
                tool_calls=[
                    {
                        "name": "render_forecast_chart",
                        "args": {},
                        "result_error": "Nieznane pole danych",
                    },
                    {"name": "render_forecast_chart", "args": {}, "result_error": None},
                ],
            ),
            {"expect_chart": True},
        )

        assert result["score"] == 1.0
        assert result["comment"] == "ok"

    def test_expected_chart_checks_requested_variable_and_spec_field(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=1,
                tool_calls=[
                    _chart_tool_call_args(
                        variables=["temperature_2m_c"],
                        field="temperature_2m_c",
                    )
                ],
            ),
            {
                "expect_chart": True,
                "expected_chart_variables": ["temperature_2m_c"],
                "expected_chart_start_date": "2026-05-04",
                "expected_chart_end_date": "2026-05-04",
            },
        )

        assert result["score"] == 1.0
        assert result["comment"] == "ok"

    def test_expected_chart_fails_when_variable_is_missing_from_tool_args(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=1,
                tool_calls=[
                    _chart_tool_call_args(
                        variables=["wind_speed_10m_ms"],
                        field="temperature_2m_c",
                    )
                ],
            ),
            {"expect_chart": True, "expected_chart_variables": ["temperature_2m_c"]},
        )

        assert result["score"] == 0.0
        assert "missing_expected_chart_variables:temperature_2m_c" in str(result["comment"])

    def test_expected_chart_allows_default_spec_when_tool_omits_spec(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=1,
                tool_calls=[
                    {
                        "name": "render_forecast_chart",
                        "args": {
                            "location_name": "Chwarzno",
                            "start_date": "2026-05-04",
                            "end_date": "2026-05-04",
                            "variables": ["wind_speed_10m_ms"],
                        },
                        "result_error": None,
                    }
                ],
            ),
            {"expect_chart": True, "expected_chart_variables": ["wind_speed_10m_ms"]},
        )

        assert result["score"] == 1.0
        assert result["comment"] == "ok"

    def test_expected_chart_fails_when_variable_is_missing_from_spec_fields(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=1,
                tool_calls=[
                    _chart_tool_call_args(
                        variables=["temperature_2m_c"],
                        field="wind_speed_10m_ms",
                    )
                ],
            ),
            {"expect_chart": True, "expected_chart_variables": ["temperature_2m_c"]},
        )

        assert result["score"] == 0.0
        assert "missing_expected_chart_fields:temperature_2m_c" in str(result["comment"])

    def test_expected_chart_fails_when_chart_date_is_wrong(self) -> None:
        result = weather_presentation_tool_use(
            _output(
                tool_name=None,
                attachment_count=1,
                tool_calls=[
                    _chart_tool_call_args(
                        variables=["temperature_2m_c"],
                        field="temperature_2m_c",
                        start_date="2026-05-05",
                    )
                ],
            ),
            {
                "expect_chart": True,
                "expected_chart_start_date": "2026-05-04",
                "expected_chart_end_date": "2026-05-04",
            },
        )

        assert result["score"] == 0.0
        assert "chart_start_date_mismatch:2026-05-04" in str(result["comment"])


class TestWeatherPresentationTarget:
    async def test_async_target_records_chart_tool_call_and_attachment(
        self,
        monkeypatch,
    ) -> None:
        calls: dict[str, object] = {}
        created_models: list[MagicMock] = []

        chart_args: dict[str, Any] = {
            "location_name": "Chwarzno",
            "start_date": "2026-05-04",
            "end_date": "2026-05-04",
            "variables": ["wind_speed_10m_ms"],
            "vega_lite_spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "title": "Wiatr w czasie",
                "data": {"name": "forecast"},
                "mark": "line",
                "encoding": {
                    "x": {"field": "time", "type": "temporal"},
                    "y": {"field": "wind_speed_10m_ms", "type": "quantitative"},
                },
            },
        }

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
                chart_tool = next(tool for tool in tools if tool.name == "render_forecast_chart")
                tool_result = await chart_tool.coroutine(**chart_args)
                assert tool_result["success"]
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "render_forecast_chart",
                                    "args": chart_args,
                                    "id": "chart-call",
                                }
                            ],
                        ),
                        ToolMessage(
                            content='{"success": "Wykres został przygotowany"}',
                            tool_call_id="chart-call",
                        ),
                        AIMessage(content="Dołączam wykres wiatru."),
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

        monkeypatch.setattr(targets, "create_weather_agent", fake_create_weather_agent)

        target = build_weather_presentation_async_target_from_factory(model_factory)
        result = await target(
            {
                "id": "weather-presentation-001",
                "question": "Rozrysuj wiatr w Chwarznie jutro.",
                "current_time": "2026-05-03T12:00:00+02:00",
                "frozen_facts": WeatherFacts(
                    location="Chwarzno",
                    period="jutro",
                    wind_speed_ms=8.0,
                ).model_dump(),
            }
        )

        assert result["example_id"] == "weather-presentation-001"
        assert result["answer"] == "Dołączam wykres wiatru."
        assert result["attachment_count"] == 1
        assert result["tool_calls"][0]["name"] == "render_forecast_chart"
        assert result["tool_calls"][0]["args"]["variables"] == ["wind_speed_10m_ms"]
        assert result["tool_calls"][0]["result_error"] is None
        agent_kwargs = calls["agent_kwargs"]
        assert isinstance(agent_kwargs, dict)
        assert agent_kwargs["model"] is created_models[0]
        assert "2026-05-03 12:00" in str(agent_kwargs["system_prompt_suffix"])


class TestWeatherPresentationDataset:
    def test_generates_chart_and_text_only_cases(self) -> None:
        cases = generate_weather_presentation_cases()

        assert DATASET_NAME == "weather-agent-weather-presentation-v2"
        assert {case["expect_chart"] for case in cases} == {False, None, True}
        expected_variables: set[str] = set()
        for case in cases:
            raw_variables = case.get("expected_chart_variables")
            if isinstance(raw_variables, list):
                expected_variables.update(
                    variable for variable in raw_variables if isinstance(variable, str)
                )
        assert {"wind_speed_10m_ms", "temperature_2m_c", "precipitation_mm"} <= expected_variables
        assert len({case["id"] for case in cases}) == len(cases)
        assert len(cases) == 5
        for case in cases:
            WeatherFacts.model_validate(case["frozen_facts"])
            assert "hourly_values" in case
            assert "current_time" in case
