from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from weather_agent.eval.notification_rule_schemas import (
    ExpectedRuleProposal,
    ExpectedRuleTool,
    ExpectedScheduleType,
    NotificationRuleEvalCase,
    RuleExpressionDiscriminatorProfile,
)

EVAL_TIMEZONE = ZoneInfo("Europe/Warsaw")


def _point(day: int, hour: int, **values: float) -> dict[str, object]:
    return {
        "target_time": datetime(2026, 5, day, hour, 0, tzinfo=EVAL_TIMEZONE).isoformat(),
        **values,
    }


def _profile(
    name: str,
    expected_result: bool,
    *points: dict[str, object],
) -> RuleExpressionDiscriminatorProfile:
    return RuleExpressionDiscriminatorProfile(
        name=name,
        points=list(points),
        expected_result=expected_result,
    )


def _expected(
    *,
    tool: ExpectedRuleTool,
    rule_expression: str,
    location: str,
    profiles: list[RuleExpressionDiscriminatorProfile],
    schedule_type: ExpectedScheduleType,
    schedule_expression: str,
) -> ExpectedRuleProposal:
    return ExpectedRuleProposal(
        expected_tool=tool,
        expected_rule_expression=rule_expression,
        expected_location=location,
        expected_schedule_type=schedule_type,
        expected_schedule_expression=schedule_expression,
        rule_expression_discriminators=profiles,
    )


def generate_recurring_notification_rule_cases(
    current_time: datetime,
) -> list[NotificationRuleEvalCase]:
    return [
        NotificationRuleEvalCase(
            id="rule-proposal-009",
            question=(
                "Za każdym razem w tygodniu (pon-pt) kiedy w Chwarznie będą dobre warunki "
                "do latania RC modelem Bigfoot 1.3m (waga 1300g), wyślij mi powiadomienie."
            ),
            current_time=current_time,
            expected=_expected(
                tool="schedule_notification",
                rule_expression=(
                    "points_between(next_hours(1)).exists(p, "
                    "p.wind_speed_10m_ms <= 4.0 && "
                    "p.wind_gusts_10m_ms <= 6.0 && "
                    "p.precipitation_mm == 0.0)"
                ),
                location="Chwarzno",
                schedule_type="cron",
                schedule_expression="0 8-18 * * 1-5",
                profiles=[
                    _profile(
                        "one_good_hour_is_enough",
                        True,
                        _point(
                            1,
                            13,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.5,
                            precipitation_mm=0.0,
                        ),
                        _point(
                            1,
                            13,
                            wind_speed_10m_ms=7.0,
                            wind_gusts_10m_ms=9.0,
                            precipitation_mm=0.4,
                        ),
                    ),
                    _profile(
                        "wind_too_strong",
                        False,
                        _point(
                            1,
                            13,
                            wind_speed_10m_ms=4.1,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.0,
                        ),
                    ),
                    _profile(
                        "gusts_too_strong",
                        False,
                        _point(
                            1,
                            13,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=6.1,
                            precipitation_mm=0.0,
                        ),
                    ),
                    _profile(
                        "precipitation_blocks_flying",
                        False,
                        _point(
                            1,
                            13,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.1,
                        ),
                    ),
                    _profile(
                        "outside_rolling_window_guard",
                        False,
                        _point(
                            2,
                            13,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.0,
                        ),
                        _point(
                            1,
                            13,
                            wind_speed_10m_ms=8.0,
                            wind_gusts_10m_ms=9.0,
                            precipitation_mm=0.5,
                        ),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-010",
            question=(
                "W każdy czwartek o 8:00 sprawdź, czy w Gdańsku porywy wiatru będą powyżej 10 m/s."
            ),
            current_time=current_time,
            expected=_expected(
                tool="schedule_notification",
                rule_expression='max_metric("wind_gusts_10m_ms", today()) > 10.0',
                location="Gdańsk",
                schedule_type="cron",
                schedule_expression="0 8 * * 4",
                profiles=[
                    _profile("true_case", True, _point(1, 13, wind_gusts_10m_ms=11.0)),
                    _profile("threshold_boundary", False, _point(1, 13, wind_gusts_10m_ms=10.0)),
                    _profile(
                        "outside_time_guard",
                        False,
                        _point(1, 13, wind_gusts_10m_ms=5.0),
                        _point(2, 8, wind_gusts_10m_ms=30.0),
                    ),
                ],
            ),
        ),
        NotificationRuleEvalCase(
            id="rule-proposal-011",
            question=(
                "Powiadom mnie pon-pt o 10:00, jeśli w Chwarznie w godzinach 18-21 będą "
                "dobre warunki do latania RC modelem Bigfoot 1.3m."
            ),
            current_time=current_time,
            expected=_expected(
                tool="schedule_notification",
                rule_expression=(
                    'points_between(between(today(), "1800", "2100")).exists(p, '
                    "p.wind_speed_10m_ms <= 4.0 && "
                    "p.wind_gusts_10m_ms <= 6.0 && "
                    "p.precipitation_mm == 0.0)"
                ),
                location="Chwarzno",
                schedule_type="cron",
                schedule_expression="0 10 * * 1-5",
                profiles=[
                    _profile(
                        "one_good_evening_hour_is_enough",
                        True,
                        _point(
                            1,
                            19,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.0,
                        ),
                        _point(
                            1,
                            20,
                            wind_speed_10m_ms=7.0,
                            wind_gusts_10m_ms=9.0,
                            precipitation_mm=0.4,
                        ),
                    ),
                    _profile(
                        "outside_evening_slot_guard",
                        False,
                        _point(
                            1,
                            17,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.0,
                        ),
                        _point(
                            1,
                            19,
                            wind_speed_10m_ms=7.0,
                            wind_gusts_10m_ms=9.0,
                            precipitation_mm=0.4,
                        ),
                    ),
                    _profile(
                        "different_day_guard",
                        False,
                        _point(
                            2,
                            19,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.0,
                        ),
                        _point(
                            1,
                            19,
                            wind_speed_10m_ms=7.0,
                            wind_gusts_10m_ms=9.0,
                            precipitation_mm=0.4,
                        ),
                    ),
                    _profile(
                        "wind_too_strong",
                        False,
                        _point(
                            1,
                            19,
                            wind_speed_10m_ms=4.1,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.0,
                        ),
                    ),
                    _profile(
                        "gusts_too_strong",
                        False,
                        _point(
                            1,
                            19,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=6.1,
                            precipitation_mm=0.0,
                        ),
                    ),
                    _profile(
                        "precipitation_blocks_flying",
                        False,
                        _point(
                            1,
                            19,
                            wind_speed_10m_ms=3.0,
                            wind_gusts_10m_ms=5.0,
                            precipitation_mm=0.1,
                        ),
                    ),
                ],
            ),
        ),
    ]
