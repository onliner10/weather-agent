from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from weather_agent.eval.location_management_schemas import (
    ExpectedLocationAction,
    LocationManagementEvalCase,
    SeedLocation,
)

DATASET_NAME = "weather-agent-location-management-v1"
EVAL_TIMEZONE = ZoneInfo("Europe/Warsaw")
EVAL_CURRENT_TIME = datetime(2026, 5, 1, 12, 0, tzinfo=EVAL_TIMEZONE)


def _seed_default() -> SeedLocation:
    return SeedLocation(name="Chwarzno", aliases=["dom"], latitude=54.4871, longitude=18.4202)


def _seed_work() -> SeedLocation:
    return SeedLocation(
        name="Gdańsk Wrzeszcz", aliases=["praca"], latitude=54.382, longitude=18.604
    )


def _seed_lake() -> SeedLocation:
    return SeedLocation(name="Jeziorak", aliases=["jezioro"], latitude=53.6108, longitude=19.6603)


def _expected(
    *,
    tool: str,
    location: str | None = None,
    alias: str | None = None,
    new_name: str | None = None,
    allow_default_alias: bool = False,
    expect_default_location: bool = False,
    expect_no_tool_call: bool = False,
) -> ExpectedLocationAction:
    return ExpectedLocationAction(
        expected_tool=tool,  # type: ignore[arg-type]
        expected_location=location,
        expected_alias=alias,
        expected_new_name=new_name,
        allow_default_alias=allow_default_alias,
        expect_default_location=expect_default_location,
        expect_no_tool_call=expect_no_tool_call,
    )


def generate_location_management_cases() -> list[LocationManagementEvalCase]:
    return [
        LocationManagementEvalCase(
            id="location-management-001",
            question="Zapamiętaj Gdańsk, Piotrkowska 11b jako dom.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[],
            expected=_expected(
                tool="save_location", location="Gdańsk, Piotrkowska 11b", alias="dom"
            ),
        ),
        LocationManagementEvalCase(
            id="location-management-002",
            question="Zapisz Jeziorak na liście lokalizacji.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default()],
            expected=_expected(tool="save_location", location="Jeziorak"),
        ),
        LocationManagementEvalCase(
            id="location-management-003",
            question="Zmień alias lokalizacji dom na mieszkanie.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default(), _seed_work()],
            expected=_expected(tool="edit_location", location="dom", alias="mieszkanie"),
        ),
        LocationManagementEvalCase(
            id="location-management-004",
            question="Usuń lokalizację praca.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default(), _seed_work()],
            expected=_expected(tool="remove_location", location="praca"),
        ),
        LocationManagementEvalCase(
            id="location-management-005",
            question="Jaka będzie jutro pogoda?",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default()],
            expected=_expected(
                tool="get_forecast",
                location="Chwarzno",
                allow_default_alias=True,
                expect_default_location=True,
            ),
        ),
        LocationManagementEvalCase(
            id="location-management-006",
            question="Powiadom mnie, gdy jutro będzie padać.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default()],
            expected=_expected(
                tool="propose_notification_rule",
                location="Chwarzno",
                allow_default_alias=True,
                expect_default_location=True,
            ),
        ),
        LocationManagementEvalCase(
            id="location-management-007",
            question="Jaka będzie jutro pogoda?",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[],
            expected=_expected(tool="ask_for_location", expect_no_tool_call=True),
        ),
        LocationManagementEvalCase(
            id="location-management-008",
            question="Zmień nazwę lokalizacji praca na biuro.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default(), _seed_work()],
            expected=_expected(tool="edit_location", location="praca", new_name="biuro"),
        ),
        LocationManagementEvalCase(
            id="location-management-009",
            question="Usuń lokalizację Jeziorak.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default(), _seed_lake()],
            expected=_expected(tool="remove_location", location="Jeziorak"),
        ),
        LocationManagementEvalCase(
            id="location-management-010",
            question="Czy teraz pada?",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[_seed_default()],
            expected=_expected(
                tool="get_observations",
                location="Chwarzno",
                allow_default_alias=True,
                expect_default_location=True,
            ),
        ),
        LocationManagementEvalCase(
            id="location-management-011",
            question="Powiadom mnie, gdy jutro będzie padać.",
            current_time=EVAL_CURRENT_TIME,
            seed_locations=[],
            expected=_expected(tool="ask_for_location", expect_no_tool_call=True),
        ),
    ]
