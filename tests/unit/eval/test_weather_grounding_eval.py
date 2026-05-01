from __future__ import annotations

import asyncio
from datetime import datetime
from typing import cast
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage

from weather_agent.domain.weather import ForecastResolution, ForecastResult, LocationRef, TimeRange
from weather_agent.eval import targets
from weather_agent.eval.dataset_gen import (
    ATTRIBUTES,
    EVAL_CURRENT_TIME,
    FORECAST_HOURS,
    PERIODS,
    PeriodLabel,
    build_question,
    expected_target_time,
    generate_cases,
)
from weather_agent.eval.evaluators import weather_functional_correctness
from weather_agent.eval.schemas import WeatherFacts
from weather_agent.eval.targets import (
    _build_fixture_weather_tools,
    _FixtureForecastProvider,
    _normalize_hourly_values,
    build_weather_answer_async_target_from_factory,
    build_weather_answer_target,
    build_weather_answer_target_from_factory,
)


def _facts(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "location": "Chwarzno",
        "period": "teraz",
        "temperature_c": 12.0,
        "wind_speed_ms": 8.0,
        "wind_direction_deg": 270.0,
        "pressure_hpa": 1012.0,
        "humidity_pct": 82.0,
        "precipitation_mm": 3.0,
    }
    data.update(overrides)
    return data


def _functional_score(
    answer: str,
    facts: dict[str, object] | None = None,
    requested_attributes: list[str] | None = None,
    required_location: bool = True,
) -> dict[str, object]:
    return weather_functional_correctness(
        outputs={"answer": answer},
        reference_outputs={
            "expected_facts": facts or _facts(),
            "required_location": required_location,
            "requested_attributes": requested_attributes or ["temperature_c"],
        },
    )


class TestWeatherFunctionalCorrectness:
    def test_temperature_answer_with_required_facts_passes(self) -> None:
        result = _functional_score("W Chwarznie temperatura wynosi około 12°C.")

        assert result["key"] == "weather_functional_correctness"
        assert result["score"] == 1.0

    def test_temperature_answer_with_single_in_range_value_passes(self) -> None:
        result = _functional_score("W Chwarznie temperatura wynosi około 12°C.")

        assert result["score"] == 1.0

    def test_qualitative_temperature_answer_fails_when_value_requested(self) -> None:
        result = _functional_score("W Chwarznie będzie raczej chłodno.")

        assert result["score"] == 0.0
        assert "missing_attribute_value:temperature_c" in str(result["comment"])

    def test_temperature_without_celsius_unit_fails(self) -> None:
        result = _functional_score("W Chwarznie temperatura wyniesie około 12.")

        assert result["score"] == 0.0
        assert "missing_attribute_value:temperature_c" in str(result["comment"])

    def test_temperature_outside_range_fails(self) -> None:
        result = _functional_score("W Chwarznie będzie 22°C.")

        assert result["score"] == 0.0
        assert "attribute_value_mismatch:temperature_c:22" in str(result["comment"])

    def test_missing_location_fails(self) -> None:
        result = _functional_score("Jutro będzie około 12°C.")

        assert result["score"] == 0.0
        assert "missing_location:Chwarzno" in str(result["comment"])

    def test_can_disable_location_requirement(self) -> None:
        result = _functional_score(
            "Temperatura wyniesie około 12°C.",
            required_location=False,
        )

        assert result["score"] == 1.0

    def test_wind_speed_value_with_unit_passes(self) -> None:
        result = _functional_score(
            "Nad Jeziorakiem wiatr ma prędkość 8,5 m/s.",
            facts=_facts(location="Jeziorak", wind_speed_ms=8.5),
            requested_attributes=["wind_speed_ms"],
        )

        assert result["score"] == 1.0

    def test_wind_speed_without_unit_fails(self) -> None:
        result = _functional_score(
            "Nad Jeziorakiem wiatr ma prędkość 8,5.",
            facts=_facts(location="Jeziorak", wind_speed_ms=8.5),
            requested_attributes=["wind_speed_ms"],
        )

        assert result["score"] == 0.0
        assert "missing_attribute_value:wind_speed_ms" in str(result["comment"])

    def test_humidity_percent_value_passes(self) -> None:
        result = _functional_score(
            "W Gdyni wilgotność wyniesie 82%.",
            facts=_facts(location="Gdynia", humidity_pct=82.0),
            requested_attributes=["humidity_pct"],
        )

        assert result["score"] == 1.0

    def test_pressure_hpa_value_passes(self) -> None:
        result = _functional_score(
            "W Chwarznie ciśnienie wyniesie 1012 hPa.",
            facts=_facts(pressure_hpa=1012.0),
            requested_attributes=["pressure_hpa"],
        )

        assert result["score"] == 1.0

    def test_precipitation_mm_value_passes(self) -> None:
        result = _functional_score(
            "W Warszawie suma opadów wynosi 3 mm.",
            facts=_facts(location="Warszawa", precipitation_mm=3.0),
            requested_attributes=["precipitation_mm"],
        )

        assert result["score"] == 1.0

    def test_wind_direction_degree_value_passes(self) -> None:
        result = _functional_score(
            "W Chwarznie kierunek wiatru to 270°.",
            facts=_facts(wind_direction_deg=270.0),
            requested_attributes=["wind_direction_deg"],
        )

        assert result["score"] == 1.0

    def test_unknown_requested_attribute_fails(self) -> None:
        result = weather_functional_correctness(
            outputs={"answer": "W Chwarznie będzie około 12°C."},
            reference_outputs={
                "expected_facts": _facts(),
                "requested_attributes": ["unsupported_metric"],
            },
        )

        assert result["score"] == 0.0
        assert "unknown_requested_attribute:unsupported_metric" in str(result["comment"])


class TestBuildWeatherAnswerTarget:
    def test_target_uses_production_deepagent_constructor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        fake_model = MagicMock()

        class FakeAgent:
            def invoke(self, payload: dict[str, object], **kwargs: object) -> dict[str, object]:
                del kwargs
                calls["payload"] = payload
                return {"messages": [AIMessage(content="W Chwarznie jutro będzie padać.")]}

        def fake_create_weather_agent(**kwargs: object) -> FakeAgent:
            calls["agent_kwargs"] = kwargs
            return FakeAgent()

        monkeypatch.setattr(targets, "create_weather_agent", fake_create_weather_agent)
        monkeypatch.setattr(
            targets,
            "build_current_time_prompt_suffix",
            lambda _now=None: "Bieżąca data i godzina w strefie Europe/Warsaw: 2026-05-01 20:34.",
        )

        target = build_weather_answer_target(fake_model)
        result = target(
            {
                "id": "grounding-001",
                "question": "Podaj aktualną wartość temperatury w Chwarznie.",
                "current_time": "2026-05-01T20:34:00+02:00",
                "frozen_facts": WeatherFacts(
                    location="Chwarzno",
                    period="teraz",
                    temperature_c=12.0,
                ).model_dump(),
            }
        )

        assert list(result) == ["answer", "example_id"]
        assert result["example_id"] == "grounding-001"
        assert result["answer"] == "W Chwarznie jutro będzie padać."
        agent_kwargs = calls["agent_kwargs"]
        assert isinstance(agent_kwargs, dict)
        assert agent_kwargs["model"] is fake_model
        assert (
            agent_kwargs["system_prompt_suffix"]
            == "Bieżąca data i godzina w strefie Europe/Warsaw: 2026-05-01 20:34."
        )
        assert len(agent_kwargs["tools"]) >= 2
        payload = calls["payload"]
        assert isinstance(payload, dict)
        assert payload["messages"][0].content == "Podaj aktualną wartość temperatury w Chwarznie."

    def test_factory_target_creates_fresh_model_per_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        created_models: list[MagicMock] = []

        class FakeAgent:
            def invoke(self, payload: dict[str, object], **kwargs: object) -> dict[str, object]:
                del payload, kwargs
                return {"messages": [AIMessage(content="W Warszawie jest 12°C.")]}

        def fake_create_weather_agent(**kwargs: object) -> FakeAgent:
            calls["last_agent_kwargs"] = kwargs
            return FakeAgent()

        def model_factory() -> MagicMock:
            model = MagicMock()
            created_models.append(model)
            return model

        monkeypatch.setattr(targets, "create_weather_agent", fake_create_weather_agent)
        monkeypatch.setattr(
            targets,
            "build_current_time_prompt_suffix",
            lambda _now=None: "Bieżąca data i godzina w strefie Europe/Warsaw: 2026-05-01 20:34.",
        )

        target = build_weather_answer_target_from_factory(model_factory)
        payload: dict[str, object] = {
            "id": "grounding-001",
            "question": "Podaj aktualną wartość temperatury w Warszawie.",
            "current_time": "2026-05-01T20:34:00+02:00",
            "frozen_facts": WeatherFacts(
                location="Warszawa",
                period="teraz",
                temperature_c=12.0,
            ).model_dump(),
        }

        target(payload)
        target(payload)

        assert len(created_models) == 2
        assert created_models[0] is not created_models[1]
        agent_kwargs = calls["last_agent_kwargs"]
        assert isinstance(agent_kwargs, dict)
        assert agent_kwargs["model"] is created_models[-1]
        assert (
            agent_kwargs["system_prompt_suffix"]
            == "Bieżąca data i godzina w strefie Europe/Warsaw: 2026-05-01 20:34."
        )

    async def test_async_factory_target_creates_fresh_model_per_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, object] = {}
        created_models: list[MagicMock] = []

        class FakeAgent:
            async def ainvoke(
                self, payload: dict[str, object], **kwargs: object
            ) -> dict[str, object]:
                del payload, kwargs
                return {"messages": [AIMessage(content="W Warszawie jest 12°C.")]}

        def fake_create_weather_agent(**kwargs: object) -> FakeAgent:
            calls["last_agent_kwargs"] = kwargs
            return FakeAgent()

        def model_factory() -> MagicMock:
            model = MagicMock()
            created_models.append(model)
            return model

        monkeypatch.setattr(targets, "create_weather_agent", fake_create_weather_agent)
        monkeypatch.setattr(
            targets,
            "build_current_time_prompt_suffix",
            lambda _now=None: "Bieżąca data i godzina w strefie Europe/Warsaw: 2026-05-01 20:34.",
        )

        target = build_weather_answer_async_target_from_factory(model_factory)
        payload: dict[str, object] = {
            "id": "grounding-001",
            "question": "Podaj aktualną wartość temperatury w Warszawie.",
            "current_time": "2026-05-01T20:34:00+02:00",
            "frozen_facts": WeatherFacts(
                location="Warszawa",
                period="teraz",
                temperature_c=12.0,
            ).model_dump(),
        }

        await target(payload)
        await target(payload)

        assert len(created_models) == 2
        assert created_models[0] is not created_models[1]
        agent_kwargs = calls["last_agent_kwargs"]
        assert isinstance(agent_kwargs, dict)
        assert agent_kwargs["model"] is created_models[-1]
        assert (
            agent_kwargs["system_prompt_suffix"]
            == "Bieżąca data i godzina w strefie Europe/Warsaw: 2026-05-01 20:34."
        )

    def test_fixture_weather_tools_use_frozen_facts(self) -> None:
        facts = WeatherFacts(
            location="Chwarzno",
            period="teraz",
            temperature_c=12.0,
            wind_speed_ms=8.0,
        )
        tools = _build_fixture_weather_tools(facts)

        tool_names = {tool.name for tool in tools}
        assert "get_forecast" in tool_names
        assert "get_observations" in tool_names


class TestGeneratedDataset:
    def test_generates_valid_cases(self) -> None:
        cases = generate_cases()

        assert len(cases) == len(PERIODS) * len(ATTRIBUTES)
        assert len({c["id"] for c in cases}) == len(cases)
        for c in cases:
            assert c["current_time"] == EVAL_CURRENT_TIME.isoformat()
            WeatherFacts.model_validate(c["frozen_facts"])

    def test_forecast_cases_are_explicit_hourly_retrieval_tasks(self) -> None:
        forecast_cases = [c for c in generate_cases() if "target_hour" in c]

        assert len(forecast_cases) == (len(PERIODS) - 1) * len(ATTRIBUTES)
        for c in forecast_cases:
            requested_attr = c["requested_attributes"][0]
            target_hour = c["target_hour"]
            hourly_values = c["hourly_values"]

            assert target_hour in FORECAST_HOURS
            assert f"o {target_hour:02d}:00" in c["question"]
            assert set(hourly_values) == set(range(24))
            assert c["frozen_facts"][requested_attr] == hourly_values[target_hour][requested_attr]
            period = cast(PeriodLabel, c["frozen_facts"]["period"])
            assert (
                c["expected_target_time"] == expected_target_time(period, target_hour).isoformat()
            )

    def test_forecast_questions_must_include_explicit_hour(self) -> None:
        with pytest.raises(ValueError, match="explicit hour"):
            build_question("temperature_c", "jutro", "Warszawa")

    def test_hourly_fixture_makes_requested_hour_distinguishable(self) -> None:
        for c in [case for case in generate_cases() if "target_hour" in case]:
            requested_attr = c["requested_attributes"][0]
            target_hour = c["target_hour"]
            comparison_hour = target_hour - 1 if target_hour > 0 else target_hour + 1
            hourly_values = c["hourly_values"]

            assert (
                hourly_values[target_hour][requested_attr]
                != hourly_values[comparison_hour][requested_attr]
            )

    def test_current_cases_have_no_forecast_fixture(self) -> None:
        current_cases = [c for c in generate_cases() if "target_hour" not in c]

        assert len(current_cases) == len(ATTRIBUTES)
        assert all("hourly_values" not in c for c in current_cases)

    def test_strict_grounding_cases_do_not_include_ambiguous_holiday_label(self) -> None:
        periods = {str(c["frozen_facts"]["period"]) for c in generate_cases()}

        assert "w święta" not in periods
        assert "w majówkę" not in periods
        assert "w Święto Pracy" in periods


class TestFixtureForecastProvider:
    def test_returns_24_warsaw_local_hourly_points(self) -> None:
        case = next(c for c in generate_cases() if "hourly_values" in c)
        facts = WeatherFacts.model_validate(case["frozen_facts"])
        hourly_values = cast(dict[int, dict[str, float]], case["hourly_values"])
        warsaw = ZoneInfo("Europe/Warsaw")
        expected_time = datetime.fromisoformat(str(case["expected_target_time"])).astimezone(warsaw)
        provider = _FixtureForecastProvider(facts, hourly_values, expected_time)

        async def run() -> ForecastResult:
            return await provider.get_forecast(
                location=LocationRef(
                    id="eval-location",
                    name="Warszawa",
                    latitude=52.23,
                    longitude=21.01,
                ),
                time_range=TimeRange(
                    start=expected_time.replace(hour=0, minute=0),
                    end=expected_time.replace(hour=23, minute=59),
                ),
                variables=[],
                resolution=ForecastResolution.hourly,
            )

        result = asyncio.run(run())
        points = result.points
        assert len(points) == 24
        assert [p.target_time.hour for p in points] == list(range(24))
        assert all(p.target_time.tzinfo == warsaw for p in points)
        assert all(p.target_time.date() == expected_time.date() for p in points)

    def test_forecast_fixture_returns_no_points_for_wrong_date(self) -> None:
        case = next(c for c in generate_cases() if c["frozen_facts"]["period"] == "za tydzień")
        facts = WeatherFacts.model_validate(case["frozen_facts"])
        hourly_values = cast(dict[int, dict[str, float]], case["hourly_values"])
        warsaw = ZoneInfo("Europe/Warsaw")
        expected_time = datetime.fromisoformat(str(case["expected_target_time"])).astimezone(warsaw)
        provider = _FixtureForecastProvider(facts, hourly_values, expected_time)

        async def run() -> ForecastResult:
            return await provider.get_forecast(
                location=LocationRef(
                    id="eval-location",
                    name="Warszawa",
                    latitude=52.23,
                    longitude=21.01,
                ),
                time_range=TimeRange(
                    start=datetime(2026, 5, 2, tzinfo=warsaw),
                    end=datetime(2026, 5, 2, 23, 59, tzinfo=warsaw),
                ),
                variables=[],
                resolution=ForecastResolution.hourly,
            )

        result = asyncio.run(run())

        assert result.points == []
        assert result.raw_payload["date_match"] is False
        assert result.raw_payload["expected_target_time"] == expected_time.isoformat()

    def test_accepts_langsmith_json_string_hour_keys(self) -> None:
        case = next(c for c in generate_cases() if "hourly_values" in c)
        hourly_values = cast(dict[int, dict[str, float]], case["hourly_values"])
        string_keyed = {str(k): v for k, v in hourly_values.items()}
        normalized = _normalize_hourly_values(string_keyed)
        assert normalized is not None
        assert set(normalized) == set(range(24))


class TestForecastPeriodEvaluation:
    def test_forecast_temperature_answer_passes(self) -> None:
        result = _functional_score(
            "W Warszawie jutro temperatura wyniesie 15°C.",
            facts=_facts(location="Warszawa", period="jutro", temperature_c=15.0),
            requested_attributes=["temperature_c"],
        )
        assert result["score"] == 1.0

    def test_forecast_wind_speed_with_wrong_value_fails(self) -> None:
        result = _functional_score(
            "W Warszawie za 2 dni wiatr będzie miał prędkość 5 m/s.",
            facts=_facts(location="Warszawa", period="za 2 dni", wind_speed_ms=12.0),
            requested_attributes=["wind_speed_ms"],
        )
        assert result["score"] == 0.0
        assert "attribute_value_mismatch:wind_speed_ms:5" in str(result["comment"])

    def test_forecast_precipitation_passes(self) -> None:
        result = _functional_score(
            "W Warszawie 3 maja suma opadów wyniesie 5 mm.",
            facts=_facts(location="Warszawa", period="3 maja", precipitation_mm=5.0),
            requested_attributes=["precipitation_mm"],
        )
        assert result["score"] == 1.0

    def test_forecast_missing_location_fails(self) -> None:
        result = _functional_score(
            "Jutro temperatura wyniesie 15°C.",
            facts=_facts(location="Warszawa", period="jutro", temperature_c=15.0),
            requested_attributes=["temperature_c"],
        )
        assert result["score"] == 0.0
        assert "missing_location:Warszawa" in str(result["comment"])
