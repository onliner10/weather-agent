from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from weather_agent.eval.notification_rule_schemas import (
    CELDiscriminatorProfile,
    ExpectedRuleProposal,
    ExpectedRuleTool,
    ExpectedScheduleType,
    NotificationRuleEvalCase,
)

DATASET_NAME = "weather-agent-notification-rule-proposal-v1"
EVAL_TIMEZONE = ZoneInfo("Europe/Warsaw")
EVAL_CURRENT_TIME = datetime(2026, 5, 1, 12, 0, tzinfo=EVAL_TIMEZONE)


def _point(day: int, hour: int, **values: float) -> dict[str, object]:
    return {
        "target_time": datetime(2026, 5, day, hour, 0, tzinfo=EVAL_TIMEZONE).isoformat(),
        **values,
    }


def _profile(
    name: str,
    expected_result: bool,
    *points: dict[str, object],
) -> CELDiscriminatorProfile:
    return CELDiscriminatorProfile(
        name=name,
        points=list(points),
        expected_result=expected_result,
    )


def _expected(
    *,
    tool: ExpectedRuleTool,
    cel: str,
    location: str,
    profiles: list[CELDiscriminatorProfile],
    schedule_type: ExpectedScheduleType | None = None,
    schedule_expression: str | None = None,
) -> ExpectedRuleProposal:
    return ExpectedRuleProposal(
        expected_tool=tool,
        expected_cel=cel,
        expected_location=location,
        expected_schedule_type=schedule_type,
        expected_schedule_expression=schedule_expression,
        cel_discriminators=profiles,
    )


def generate_notification_rule_cases() -> list[NotificationRuleEvalCase]:
    cases = [
        NotificationRuleEvalCase(
            id="rule-proposal-001",
            question=(
                "Powiadom mnie, jeśli porywy wiatru w weekend w Warszawie będą powyżej 12 m/s."
            ),
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="propose_notification_rule",
                cel='max("wind_gusts_10m_ms", weekend()) > 12.0',
                location="Warszawa",
                profiles=[
                    _profile("true_case", True, _point(2, 10, wind_gusts_10m_ms=13.0)),
                    _profile("threshold_boundary", False, _point(2, 10, wind_gusts_10m_ms=12.0)),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 14, wind_gusts_10m_ms=30.0),
                        _point(2, 10, wind_gusts_10m_ms=5.0),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-002",
            question="Powiadom, jeśli temperatura dzisiaj w Krakowie spadnie poniżej -10 stopni.",
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="propose_notification_rule",
                cel='min("temperature_2m_c", today()) < -10.0',
                location="Kraków",
                profiles=[
                    _profile("true_case", True, _point(1, 15, temperature_2m_c=-11.0)),
                    _profile("threshold_boundary", False, _point(1, 15, temperature_2m_c=-10.0)),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 15, temperature_2m_c=5.0),
                        _point(2, 8, temperature_2m_c=-20.0),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-003",
            question="Daj znać, gdy ciśnienie w Gdańsku spadnie poniżej 1000 hPa.",
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="propose_notification_rule",
                cel='min("pressure_msl_hpa", today()) < 1000.0',
                location="Gdańsk",
                profiles=[
                    _profile("true_case", True, _point(1, 15, pressure_msl_hpa=999.0)),
                    _profile("threshold_boundary", False, _point(1, 15, pressure_msl_hpa=1000.0)),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 15, pressure_msl_hpa=1015.0),
                        _point(2, 8, pressure_msl_hpa=990.0),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-004",
            question="Powiadom, gdy wilgotność we Wrocławiu przekroczy 90%.",
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="propose_notification_rule",
                cel='max("relative_humidity_2m_pct", today()) > 90.0',
                location="Wrocław",
                profiles=[
                    _profile("true_case", True, _point(1, 15, relative_humidity_2m_pct=91.0)),
                    _profile(
                        "threshold_boundary",
                        False,
                        _point(1, 15, relative_humidity_2m_pct=90.0),
                    ),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 15, relative_humidity_2m_pct=30.0),
                        _point(2, 8, relative_humidity_2m_pct=99.0),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-005",
            question=(
                "Powiadom, gdy suma opadów w Chwarznie przez następne 6 godzin przekroczy 5 mm."
            ),
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="propose_notification_rule",
                cel='sum("precipitation_mm", next_hours(hours(6))) > 5.0',
                location="Chwarzno",
                profiles=[
                    _profile(
                        "true_case",
                        True,
                        _point(1, 13, precipitation_mm=3.0),
                        _point(1, 16, precipitation_mm=2.5),
                    ),
                    _profile(
                        "threshold_boundary",
                        False,
                        _point(1, 13, precipitation_mm=2.0),
                        _point(1, 16, precipitation_mm=3.0),
                    ),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 13, precipitation_mm=0.0),
                        _point(20, 12, precipitation_mm=10.0),
                    ),
                    _profile(
                        "aggregation_guard",
                        True,
                        _point(1, 13, precipitation_mm=2.8),
                        _point(1, 16, precipitation_mm=2.7),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-006",
            question="Codziennie o 7:00 wyślij prognozę dla Warszawy.",
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="schedule_notification",
                cel="True",
                location="Warszawa",
                schedule_type="cron",
                schedule_expression="0 7 * * *",
                profiles=[_profile("true_case", True, _point(1, 13, temperature_2m_c=10.0))],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-007",
            question="Jutro o 8:00 sprawdź, czy w Gdyni wiatr będzie powyżej 10 m/s.",
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="schedule_notification",
                cel='max("wind_speed_10m_ms", tomorrow()) > 10.0',
                location="Gdynia",
                schedule_type="once",
                schedule_expression="2026-05-02T08:00:00+02:00",
                profiles=[
                    _profile("true_case", True, _point(2, 8, wind_speed_10m_ms=11.0)),
                    _profile("threshold_boundary", False, _point(2, 8, wind_speed_10m_ms=10.0)),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 14, wind_speed_10m_ms=30.0),
                        _point(2, 8, wind_speed_10m_ms=2.0),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-008",
            question=(
                "Utwórz od razu regułę, nie pytaj o potwierdzenie: powiadom mnie, "
                "jeśli temperatura w Poznaniu spadnie poniżej 0."
            ),
            current_time=EVAL_CURRENT_TIME,
            expected=_expected(
                tool="propose_notification_rule",
                cel='min("temperature_2m_c", today()) < 0.0',
                location="Poznań",
                profiles=[
                    _profile("true_case", True, _point(1, 15, temperature_2m_c=-1.0)),
                    _profile("threshold_boundary", False, _point(1, 15, temperature_2m_c=0.0)),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 15, temperature_2m_c=5.0),
                        _point(2, 8, temperature_2m_c=-5.0),
                    ),
                ],
            ),
        ),
    ]
    return cases
