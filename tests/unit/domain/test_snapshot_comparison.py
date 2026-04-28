from __future__ import annotations

from datetime import UTC, datetime

import pytest

from weather_agent.domain.snapshot_comparison import (
    SnapshotComparison,
    SnapshotDelta,
    compare_snapshots,
    compute_delta,
    detect_significant_changes,
)
from weather_agent.domain.weather import ForecastPoint


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC)


_NOW = _utc(datetime(2026, 4, 28, 12, 0, 0))
_PREV_FETCHED = _utc(datetime(2026, 4, 28, 10, 0, 0))
_CURR_FETCHED = _utc(datetime(2026, 4, 28, 12, 0, 0))
_T1 = _utc(datetime(2026, 4, 28, 13, 0, 0))
_T2 = _utc(datetime(2026, 4, 28, 14, 0, 0))
_T3 = _utc(datetime(2026, 4, 28, 15, 0, 0))
_LOC = "warszawa-1"


def _make_point(
    target_time: datetime,
    fetched_at: datetime,
    location_id: str = _LOC,
    **overrides: float | str | None,
) -> ForecastPoint:
    defaults: dict[str, object] = {
        "temperature_2m_c": 18.0,
        "apparent_temperature_c": 17.0,
        "precipitation_mm": 0.0,
        "precipitation_probability_pct": 10.0,
        "rain_mm": 0.0,
        "snowfall_cm": 0.0,
        "cloud_cover_pct": 25.0,
        "wind_speed_10m_ms": 5.0,
        "wind_gusts_10m_ms": 10.0,
        "wind_direction_10m_deg": 180.0,
        "pressure_msl_hpa": 1013.0,
        "relative_humidity_2m_pct": 60.0,
        "weather_code": "0",
        "raw_payload": {},
    }
    defaults.update(overrides)
    return ForecastPoint(
        target_time=target_time,
        fetched_at=fetched_at,
        provider="open-meteo",
        model="dwd-icon",
        location_id=location_id,
        **defaults,  # type: ignore[arg-type]
    )


class TestSnapshotDelta:
    def test_round_trip(self) -> None:
        d = SnapshotDelta(
            metric="temperature_2m_c",
            previous_value=15.0,
            current_value=20.0,
            delta=5.0,
            abs_delta=5.0,
            percent_change=33.33333333333333,
        )
        data = d.model_dump()
        restored = SnapshotDelta.model_validate(data)
        assert restored == d

    def test_none_values(self) -> None:
        d = SnapshotDelta(
            metric="temperature_2m_c",
            previous_value=None,
            current_value=20.0,
            delta=None,
            abs_delta=None,
            percent_change=None,
        )
        assert d.delta is None
        assert d.abs_delta is None
        assert d.percent_change is None

    def test_forbid_extra_fields(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SnapshotDelta(
                metric="temperature_2m_c",
                previous_value=None,
                current_value=None,
                delta=None,
                abs_delta=None,
                percent_change=None,
                extra_field=True,  # type: ignore[call-arg]
            )


class TestSnapshotComparison:
    def test_round_trip(self) -> None:
        delta = SnapshotDelta(
            metric="temperature_2m_c",
            previous_value=15.0,
            current_value=20.0,
            delta=5.0,
            abs_delta=5.0,
            percent_change=33.33333333333333,
        )
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[delta],
        )
        data = comp.model_dump()
        restored = SnapshotComparison.model_validate(data)
        assert restored == comp

    def test_empty_deltas(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[],
        )
        assert comp.deltas == []


class TestComputeDelta:
    def test_positive_delta(self) -> None:
        delta, abs_delta, pct = compute_delta(10.0, 15.0)
        assert delta == 5.0
        assert abs_delta == 5.0
        assert pct == pytest.approx(50.0)

    def test_negative_delta(self) -> None:
        delta, abs_delta, pct = compute_delta(20.0, 15.0)
        assert delta == -5.0
        assert abs_delta == 5.0
        assert pct == pytest.approx(-25.0)

    def test_zero_delta(self) -> None:
        delta, abs_delta, pct = compute_delta(10.0, 10.0)
        assert delta == 0.0
        assert abs_delta == 0.0
        assert pct == pytest.approx(0.0)

    def test_none_previous(self) -> None:
        delta, abs_delta, pct = compute_delta(None, 15.0)
        assert delta is None
        assert abs_delta is None
        assert pct is None

    def test_none_current(self) -> None:
        delta, abs_delta, pct = compute_delta(10.0, None)
        assert delta is None
        assert abs_delta is None
        assert pct is None

    def test_both_none(self) -> None:
        delta, abs_delta, pct = compute_delta(None, None)
        assert delta is None
        assert abs_delta is None
        assert pct is None

    def test_percent_change_zero_previous(self) -> None:
        delta, abs_delta, pct = compute_delta(0.0, 5.0)
        assert delta == 5.0
        assert abs_delta == 5.0
        assert pct is None

    def test_percent_change_from_negative(self) -> None:
        delta, abs_delta, pct = compute_delta(-10.0, -5.0)
        assert delta == 5.0
        assert abs_delta == 5.0
        assert pct == pytest.approx(-50.0)

    def test_percent_change_negative_result(self) -> None:
        delta, abs_delta, pct = compute_delta(10.0, 5.0)
        assert delta == -5.0
        assert abs_delta == 5.0
        assert pct == pytest.approx(-50.0)


class TestCompareSnapshots:
    def test_matching_single_point(self) -> None:
        current = [
            _make_point(
                _T1,
                _CURR_FETCHED,
                temperature_2m_c=20.0,
                wind_speed_10m_ms=8.0,
            )
        ]
        previous = [
            _make_point(
                _T1,
                _PREV_FETCHED,
                temperature_2m_c=15.0,
                wind_speed_10m_ms=5.0,
            )
        ]

        result = compare_snapshots(current, previous)

        assert result.location_id == _LOC
        assert result.current_fetched_at == _CURR_FETCHED
        assert result.previous_fetched_at == _PREV_FETCHED

        temp_delta = next(d for d in result.deltas if d.metric == "temperature_2m_c")
        assert temp_delta.previous_value == 15.0
        assert temp_delta.current_value == 20.0
        assert temp_delta.delta == 5.0
        assert temp_delta.abs_delta == 5.0
        assert temp_delta.percent_change == pytest.approx(33.33333333333333)

        wind_delta = next(d for d in result.deltas if d.metric == "wind_speed_10m_ms")
        assert wind_delta.delta == 3.0

    def test_multiple_matching_points(self) -> None:
        current = [
            _make_point(_T1, _CURR_FETCHED, temperature_2m_c=20.0),
            _make_point(_T2, _CURR_FETCHED, temperature_2m_c=22.0),
        ]
        previous = [
            _make_point(_T1, _PREV_FETCHED, temperature_2m_c=15.0),
            _make_point(_T2, _PREV_FETCHED, temperature_2m_c=18.0),
        ]

        result = compare_snapshots(current, previous)

        temp_deltas = [d for d in result.deltas if d.metric == "temperature_2m_c"]
        assert len(temp_deltas) == 2
        assert temp_deltas[0].delta == 5.0
        assert temp_deltas[1].delta == 4.0

    def test_time_mismatch_skipped(self) -> None:
        current = [
            _make_point(_T1, _CURR_FETCHED, temperature_2m_c=20.0),
        ]
        previous = [
            _make_point(
                _utc(datetime(2026, 4, 28, 13, 30, 0)),
                _PREV_FETCHED,
                temperature_2m_c=15.0,
            ),
        ]

        result = compare_snapshots(current, previous)
        assert result.deltas == []

    def test_metric_missing_in_current(self) -> None:
        current = [
            _make_point(_T1, _CURR_FETCHED, temperature_2m_c=None),
        ]
        previous = [
            _make_point(_T1, _PREV_FETCHED, temperature_2m_c=15.0),
        ]

        result = compare_snapshots(current, previous)

        temp_delta = next(d for d in result.deltas if d.metric == "temperature_2m_c")
        assert temp_delta.delta is None
        assert temp_delta.abs_delta is None
        assert temp_delta.percent_change is None

    def test_metric_missing_in_previous(self) -> None:
        current = [
            _make_point(_T1, _CURR_FETCHED, temperature_2m_c=20.0),
        ]
        previous = [
            _make_point(_T1, _PREV_FETCHED, temperature_2m_c=None),
        ]

        result = compare_snapshots(current, previous)

        temp_delta = next(d for d in result.deltas if d.metric == "temperature_2m_c")
        assert temp_delta.delta is None
        assert temp_delta.abs_delta is None
        assert temp_delta.percent_change is None

    def test_metric_missing_in_both(self) -> None:
        current = [
            _make_point(_T1, _CURR_FETCHED, temperature_2m_c=None),
        ]
        previous = [
            _make_point(_T1, _PREV_FETCHED, temperature_2m_c=None),
        ]

        result = compare_snapshots(current, previous)

        temp_delta = next(d for d in result.deltas if d.metric == "temperature_2m_c")
        assert temp_delta.previous_value is None
        assert temp_delta.current_value is None
        assert temp_delta.delta is None

    def test_empty_current_snapshot(self) -> None:
        current: list[ForecastPoint] = []
        previous = [_make_point(_T1, _PREV_FETCHED)]

        result = compare_snapshots(current, previous)
        assert result.deltas == []

    def test_empty_previous_snapshot(self) -> None:
        current = [_make_point(_T1, _CURR_FETCHED)]
        previous: list[ForecastPoint] = []

        result = compare_snapshots(current, previous)
        assert result.deltas == []

    def test_both_empty(self) -> None:
        result = compare_snapshots([], [])
        assert result.deltas == []

    def test_weather_code_excluded(self) -> None:
        current = [_make_point(_T1, _CURR_FETCHED)]
        previous = [_make_point(_T1, _PREV_FETCHED)]

        result = compare_snapshots(current, previous)
        metrics = {d.metric for d in result.deltas}
        assert "weather_code" not in metrics

    def test_all_numeric_fields_compared(self) -> None:
        current = [_make_point(_T1, _CURR_FETCHED)]
        previous = [_make_point(_T1, _PREV_FETCHED)]

        result = compare_snapshots(current, previous)
        metrics = {d.metric for d in result.deltas}
        expected = {
            "temperature_2m_c",
            "apparent_temperature_c",
            "precipitation_mm",
            "precipitation_probability_pct",
            "rain_mm",
            "snowfall_cm",
            "cloud_cover_pct",
            "wind_speed_10m_ms",
            "wind_gusts_10m_ms",
            "wind_direction_10m_deg",
            "pressure_msl_hpa",
            "relative_humidity_2m_pct",
        }
        assert metrics == expected

    def test_changed_target_time_resolution(self) -> None:
        current = [
            _make_point(
                _utc(datetime(2026, 4, 28, 13, 0, 0)),
                _CURR_FETCHED,
                temperature_2m_c=20.0,
            ),
            _make_point(
                _utc(datetime(2026, 4, 28, 13, 15, 0)),
                _CURR_FETCHED,
                temperature_2m_c=20.5,
            ),
        ]
        previous = [
            _make_point(
                _utc(datetime(2026, 4, 28, 13, 0, 0)),
                _PREV_FETCHED,
                temperature_2m_c=15.0,
            ),
        ]

        result = compare_snapshots(current, previous)

        temp_deltas = [d for d in result.deltas if d.metric == "temperature_2m_c"]
        assert len(temp_deltas) == 1
        assert temp_deltas[0].delta == 5.0
        assert temp_deltas[0].current_value == 20.0

    def test_percent_change_with_zero_previous(self) -> None:
        current = [
            _make_point(_T1, _CURR_FETCHED, precipitation_mm=5.0),
        ]
        previous = [
            _make_point(_T1, _PREV_FETCHED, precipitation_mm=0.0),
        ]

        result = compare_snapshots(current, previous)

        precip_delta = next(d for d in result.deltas if d.metric == "precipitation_mm")
        assert precip_delta.delta == 5.0
        assert precip_delta.abs_delta == 5.0
        assert precip_delta.percent_change is None


class TestDetectSignificantChanges:
    def test_basic_threshold(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=15.0,
                    current_value=20.0,
                    delta=5.0,
                    abs_delta=5.0,
                    percent_change=33.33,
                ),
                SnapshotDelta(
                    metric="wind_speed_10m_ms",
                    previous_value=5.0,
                    current_value=7.0,
                    delta=2.0,
                    abs_delta=2.0,
                    percent_change=40.0,
                ),
            ],
        )

        significant = detect_significant_changes(
            comp, {"temperature_2m_c": 4.0, "wind_speed_10m_ms": 4.0}
        )

        assert len(significant) == 1
        assert significant[0].metric == "temperature_2m_c"

    def test_no_significant_changes(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=15.0,
                    current_value=16.0,
                    delta=1.0,
                    abs_delta=1.0,
                    percent_change=6.67,
                ),
            ],
        )

        significant = detect_significant_changes(comp, {"temperature_2m_c": 5.0})
        assert significant == []

    def test_threshold_not_in_dict(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=15.0,
                    current_value=25.0,
                    delta=10.0,
                    abs_delta=10.0,
                    percent_change=66.67,
                ),
            ],
        )

        significant = detect_significant_changes(
            comp, {"wind_speed_10m_ms": 5.0}
        )
        assert significant == []

    def test_none_abs_delta_excluded(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=None,
                    current_value=20.0,
                    delta=None,
                    abs_delta=None,
                    percent_change=None,
                ),
            ],
        )

        significant = detect_significant_changes(comp, {"temperature_2m_c": 1.0})
        assert significant == []

    def test_exact_threshold_not_included(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=15.0,
                    current_value=20.0,
                    delta=5.0,
                    abs_delta=5.0,
                    percent_change=33.33,
                ),
            ],
        )

        significant = detect_significant_changes(comp, {"temperature_2m_c": 5.0})
        assert significant == []

    def test_multiple_significant(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=15.0,
                    current_value=25.0,
                    delta=10.0,
                    abs_delta=10.0,
                    percent_change=66.67,
                ),
                SnapshotDelta(
                    metric="wind_gusts_10m_ms",
                    previous_value=10.0,
                    current_value=25.0,
                    delta=15.0,
                    abs_delta=15.0,
                    percent_change=150.0,
                ),
            ],
        )

        significant = detect_significant_changes(
            comp, {"temperature_2m_c": 5.0, "wind_gusts_10m_ms": 10.0}
        )
        assert len(significant) == 2

    def test_empty_thresholds(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=15.0,
                    current_value=25.0,
                    delta=10.0,
                    abs_delta=10.0,
                    percent_change=66.67,
                ),
            ],
        )

        significant = detect_significant_changes(comp, {})
        assert significant == []

    def test_negative_delta_abs_exceeds(self) -> None:
        comp = SnapshotComparison(
            location_id=_LOC,
            previous_fetched_at=_PREV_FETCHED,
            current_fetched_at=_CURR_FETCHED,
            deltas=[
                SnapshotDelta(
                    metric="temperature_2m_c",
                    previous_value=20.0,
                    current_value=10.0,
                    delta=-10.0,
                    abs_delta=10.0,
                    percent_change=-50.0,
                ),
            ],
        )

        significant = detect_significant_changes(comp, {"temperature_2m_c": 5.0})
        assert len(significant) == 1
        assert significant[0].metric == "temperature_2m_c"