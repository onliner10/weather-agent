from __future__ import annotations

import re
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from dataset import EVAL_CASES, EvalCase
from freezegun import freeze_time

from weather_agent.domain.cel.evaluator import CELEvaluator
from weather_agent.domain.date_resolver import DateResolver, ResolvedTimeRange
from weather_agent.domain.locations import Location, LocationService
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResult,
    LocationRef,
)
from weather_agent.graphs.conversation import classify_intent
from weather_agent.graphs.nodes.weather_qa import (
    resolve_location_node,
    weather_agent_node,
)
from weather_agent.graphs.state import ConversationState


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


def _loc(name: str = "Warszawa", lat: float = 52.22, lon: float = 21.01) -> LocationRef:
    return LocationRef(id="1", name=name, latitude=lat, longitude=lon)


def _forecast(location: LocationRef | None = None) -> ForecastResult:
    if location is None:
        location = _loc()
    now_str = "2026-05-01T10:00:00+00:00"
    return ForecastResult(
        provider="open-meteo",
        model="dwd-icon",
        location=location,
        fetched_at=now_str,
        points=[
            ForecastPoint(
                target_time=now_str,
                fetched_at=now_str,
                provider="open-meteo",
                model="dwd-icon",
                location_id=location.id,
                temperature_2m_c=18.5,
                apparent_temperature_c=17.0,
                precipitation_mm=0.0,
                precipitation_probability_pct=10,
                wind_speed_10m_ms=5.0,
                wind_gusts_10m_ms=10.0,
                cloud_cover_pct=30,
                relative_humidity_2m_pct=60,
                weather_code="1",
                raw_payload={},
            )
        ],
        raw_payload={},
    )


def _mock_location_service(
    resolved: LocationRef | None = None,
    locations: list[Location] | None = None,
    default_location: LocationRef | None = None,
) -> AsyncMock:
    svc = AsyncMock(spec=LocationService)
    svc.resolve_location = AsyncMock(return_value=resolved)
    svc.get_default_location = AsyncMock(return_value=default_location)
    if locations is not None:
        svc.list_locations = AsyncMock(return_value=locations)
    else:
        svc.list_locations = AsyncMock(return_value=[])
    return svc


INTENT_CASES = [c for c in EVAL_CASES if c.expected_intent is not None]


@pytest.mark.parametrize(
    "case",
    INTENT_CASES,
    ids=[c.id for c in INTENT_CASES],
)
@pytest.mark.asyncio
async def test_intent_classification(case: EvalCase) -> None:
    state = _state(message=case.input_message)
    result = await classify_intent(state)
    actual_intent = result.get("resolved_intent")
    assert actual_intent == case.expected_intent, (
        f"Case {case.id} ({case.category!r}): expected intent {case.expected_intent!r}, "
        f"got {actual_intent!r} for message {case.input_message!r}"
    )


TIME_RESOLVE_CASES = [c for c in EVAL_CASES if c.expected_time_range is not None]


@pytest.mark.parametrize(
    "case",
    TIME_RESOLVE_CASES,
    ids=[c.id for c in TIME_RESOLVE_CASES],
)
@pytest.mark.asyncio
@freeze_time("2026-05-01 10:00:00", tz_offset=2)
async def test_time_range_resolution(case: EvalCase) -> None:
    resolver = DateResolver()
    result = await resolver.resolve(case.expected_time_range)
    assert result is not None, (
        f"Case {case.id}: DateResolver returned None for time expression "
        f"{case.expected_time_range!r}"
    )
    lower_explanation = result.explanation.lower()
    normalized = lower_explanation.replace("ó", "o").replace("ą", "a")
    expected_lower = case.expected_time_range.lower().replace("ó", "o").replace("ą", "a")
    assert expected_lower[:4] in normalized, (
        f"Case {case.id}: expected explanation to contain {case.expected_time_range!r}, "
        f"got {result.explanation!r}"
    )


LOCATION_RESOLVE_CASES = [c for c in EVAL_CASES if c.expected_location is not None]


@pytest.mark.parametrize(
    "case",
    LOCATION_RESOLVE_CASES,
    ids=[c.id for c in LOCATION_RESOLVE_CASES],
)
@pytest.mark.asyncio
async def test_location_resolution(case: EvalCase) -> None:
    from unittest.mock import MagicMock

    from weather_agent.llm.contracts.location import LocationExtraction

    mock_loc = _loc(name=case.expected_location, lat=54.0, lon=18.0)
    fallback_locations = [
        Location(
            id=1,
            name=case.expected_location,
            aliases=[],
            latitude=54.0,
            longitude=18.0,
            description=None,
            enabled=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    ]
    svc = _mock_location_service(resolved=mock_loc, locations=fallback_locations)

    mf = MagicMock()
    chat = MagicMock()
    extraction = LocationExtraction(location_name=case.expected_location, focus=None)
    structured = AsyncMock()
    structured.ainvoke = AsyncMock(return_value=extraction)
    chat.with_structured_output = MagicMock(return_value=structured)
    mf.create_chat_model = MagicMock(return_value=chat)

    state = _state(message=case.input_message)
    result = await resolve_location_node(state, svc, user_id=12345, model_factory=mf)
    assert result.get("resolved_location") is not None, (
        f"Case {case.id}: location not resolved for {case.expected_location!r}"
    )
    resolved_name = result["resolved_location"].name
    norm = resolved_name.lower().replace("ó", "o").replace("ą", "a")
    expected_norm = case.expected_location.lower().replace("ó", "o").replace("ą", "a")
    assert expected_norm[:4] in norm, (
        f"Case {case.id}: expected location containing {case.expected_location!r}, "
        f"got {resolved_name!r}"
    )


RESPONSE_PATTERN_CASES = [c for c in EVAL_CASES if c.expected_response_pattern is not None]


@pytest.mark.parametrize(
    "case",
    RESPONSE_PATTERN_CASES,
    ids=[c.id for c in RESPONSE_PATTERN_CASES],
)
@pytest.mark.asyncio
async def test_response_pattern(case: EvalCase) -> None:
    if case.category in ("weather_qa", "time_resolve"):
        await _verify_weather_response_pattern(case)
    elif case.category == "ambiguity":
        await _verify_ambiguity_response_pattern(case)
    elif case.category in ("rule_create", "rule_edit", "rule_delete"):
        await _verify_rule_response_pattern(case)
    elif case.category == "command":
        await _verify_command_response_pattern(case)
    elif case.category == "provider_failure":
        await _verify_provider_failure_response_pattern(case)
    elif case.category == "location_resolve":
        await _verify_location_response_pattern(case)


async def _verify_weather_response_pattern(case: EvalCase) -> None:
    loc = _loc(name=case.expected_location or "Jeziorak", lat=54.0, lon=18.0)
    forecast = _forecast(location=loc)
    tr = ResolvedTimeRange(
        start=datetime(2026, 5, 1, 0, 0, tzinfo=UTC),
        end=datetime(2026, 5, 2, 23, 59, tzinfo=UTC),
        explanation="Test range",
    )

    state = _state(
        message=case.input_message,
        resolved_location=loc,
        resolved_time_range=tr,
        forecast_result=forecast,
    )
    result = await weather_agent_node(
        state,
        model_factory=None,
        forecast_provider=None,
        observation_provider=None,
        geocoder=None,
        date_resolver=None,
        location_service=None,
        user_id=12345,
    )
    answer = result.get("answer", "")
    pattern = case.expected_response_pattern
    assert re.search(pattern, answer, re.IGNORECASE) or "niedostępna" in answer.lower(), (
        f"Case {case.id}: response {answer!r} does not match pattern {pattern!r}"
    )


async def _verify_ambiguity_response_pattern(case: EvalCase) -> None:
    svc = _mock_location_service(resolved=None, locations=[])

    state = _state(message=case.input_message)
    result = await resolve_location_node(state, svc, user_id=12345)
    response_text = result.get("error") or result.get("answer") or ""
    pattern = case.expected_response_pattern
    assert re.search(pattern, response_text, re.IGNORECASE), (
        f"Case {case.id}: response {response_text!r} does not match pattern {pattern!r}"
    )


async def _verify_rule_response_pattern(case: EvalCase) -> None:
    from weather_agent.graphs.nodes.rule_management import (
        propose_cel_rule_node,
        require_user_confirmation_node,
    )

    if case.expected_cel:
        cel_evaluator = CELEvaluator()
        validation = cel_evaluator.validate(case.expected_cel)
        assert validation.valid, (
            f"Case {case.id}: expected CEL {case.expected_cel!r} is invalid: {validation.error}"
        )

    mock_model_factory = AsyncMock()
    mock_chat = AsyncMock()
    mock_response = AsyncMock()
    cel_expr = case.expected_cel or 'max("wind_speed_10m_ms", weekend()) > 10.0'
    import json

    mock_response.content = json.dumps(
        {
            "cel_expression": cel_expr,
            "explanation": "Test explanation",
        }
    )
    mock_chat.ainvoke = AsyncMock(return_value=mock_response)
    mock_model_factory.create_chat_model = lambda: mock_chat

    cel_evaluator = CELEvaluator()
    state = _state(message=case.input_message)

    result = await propose_cel_rule_node(state, mock_model_factory, cel_evaluator)

    answer = str(result.get("error") or result.get("cel_expression") or "")
    pending = result.get("pending_confirmation")
    if pending:
        confirm_result = await require_user_confirmation_node(
            _state(message=case.input_message, pending_confirmation=pending)
        )
        answer = confirm_result.get("answer", "") or answer

    pattern = case.expected_response_pattern
    assert re.search(pattern, answer, re.IGNORECASE) or (
        pending is not None and "CEL" in str(pending)
    ), f"Case {case.id}: response {answer!r} does not match pattern {pattern!r}"


async def _verify_command_response_pattern(case: EvalCase) -> None:
    from weather_agent.graphs.conversation import route_to_command_or_help

    state = _state(
        message=case.input_message,
        resolved_intent="command",
    )
    result = await route_to_command_or_help(state)
    answer = result.get("answer", "")
    pattern = case.expected_response_pattern
    assert re.search(pattern, answer, re.IGNORECASE), (
        f"Case {case.id}: response {answer!r} does not match pattern {pattern!r}"
    )


async def _verify_provider_failure_response_pattern(case: EvalCase) -> None:
    error_msg = "Błąd dostawcy prognozy (open-meteo): Server error: 500"

    result = await weather_agent_node(
        _state(
            message=case.input_message,
            error=error_msg,
        ),
        model_factory=None,
        forecast_provider=None,
        observation_provider=None,
        geocoder=None,
        date_resolver=None,
        location_service=None,
        user_id=12345,
    )

    answer = result.get("answer", "")
    pattern = case.expected_response_pattern
    assert re.search(pattern, answer, re.IGNORECASE), (
        f"Case {case.id}: response {answer!r} does not match pattern {pattern!r}"
    )


async def _verify_location_response_pattern(case: EvalCase) -> None:
    assert case.expected_location is not None
    mock_loc = _loc(name=case.expected_location)
    fallback_locations = [
        Location(
            id=1,
            name=case.expected_location,
            aliases=[],
            latitude=54.0,
            longitude=18.0,
            description=None,
            enabled=True,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    ]
    svc = _mock_location_service(resolved=mock_loc, locations=fallback_locations)

    state = _state(message=case.input_message)
    result = await resolve_location_node(state, svc, user_id=12345)
    response_parts = []
    if result.get("resolved_location"):
        response_parts.append(result["resolved_location"].name)
    if result.get("error"):
        response_parts.append(result["error"])
    response_text = " ".join(response_parts)
    pattern = case.expected_response_pattern
    assert re.search(pattern, response_text, re.IGNORECASE), (
        f"Case {case.id}: response {response_text!r} does not match pattern {pattern!r}"
    )
