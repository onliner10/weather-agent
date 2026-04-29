from __future__ import annotations

from datetime import UTC, datetime

from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
    TimeRange,
    WeatherVariable,
    WeatherWarning,
)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC)


_WARSZAWA = LocationRef(id="warszawa-1", name="Warszawa", latitude=52.2297, longitude=21.0122)
_NOW = _utc(datetime(2026, 4, 28, 12, 0, 0))
_HOUR = _utc(datetime(2026, 4, 28, 13, 0, 0))


class TestLocationRef:
    def test_round_trip(self) -> None:
        data = _WARSZAWA.model_dump()
        restored = LocationRef.model_validate(data)
        assert restored == _WARSZAWA

    def test_forbid_extra_fields(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LocationRef(id="x", name="X", latitude=0.0, longitude=0.0, unexpected=True)


class TestTimeRange:
    def test_default_timezone(self) -> None:
        tr = TimeRange(start=_NOW, end=_HOUR)
        assert tr.timezone == "Europe/Warsaw"

    def test_round_trip(self) -> None:
        tr = TimeRange(start=_NOW, end=_HOUR)
        data = tr.model_dump()
        restored = TimeRange.model_validate(data)
        assert restored == tr

    def test_forbid_extra_fields(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TimeRange(start=_NOW, end=_HOUR, timezone="Europe/Warsaw", extra=True)  # type: ignore[call-arg]

    def test_invalid_timezone_rejected(self) -> None:
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TimeRange(start=_NOW, end=_HOUR, timezone="America/New_York")  # type: ignore[call-arg]


class TestForecastPoint:
    def test_none_defaults_when_omitted(self) -> None:
        fp = ForecastPoint(
            target_time=_HOUR,
            fetched_at=_NOW,
            provider="open-meteo",
            model="dwd-icon",
            location_id="warszawa-1",
            raw_payload={"hourly": {"temperature_2m": [18.5]}},
        )
        assert fp.temperature_2m_c is None
        assert fp.apparent_temperature_c is None
        assert fp.precipitation_mm is None
        assert fp.wind_speed_10m_ms is None
        assert fp.weather_code is None

    def test_explicit_values(self) -> None:
        fp = ForecastPoint(
            target_time=_HOUR,
            fetched_at=_NOW,
            provider="open-meteo",
            model="dwd-icon",
            location_id="warszawa-1",
            temperature_2m_c=18.5,
            apparent_temperature_c=17.2,
            precipitation_mm=0.0,
            wind_speed_10m_ms=5.3,
            raw_payload={"hourly": {"temperature_2m": [18.5]}},
        )
        assert fp.temperature_2m_c == 18.5
        assert fp.apparent_temperature_c == 17.2
        assert fp.precipitation_mm == 0.0

    def test_raw_payload_preserved(self) -> None:
        raw = {"source": "open-meteo", "units": {"temperature": "°C"}}
        fp = ForecastPoint(
            target_time=_HOUR,
            fetched_at=_NOW,
            provider="open-meteo",
            model="dwd-icon",
            location_id="warszawa-1",
            raw_payload=raw,
        )
        assert fp.raw_payload == raw
        data = fp.model_dump()
        assert data["raw_payload"] == raw

    def test_round_trip(self) -> None:
        fp = ForecastPoint(
            target_time=_HOUR,
            fetched_at=_NOW,
            provider="open-meteo",
            model=None,
            location_id="warszawa-1",
            temperature_2m_c=22.0,
            raw_payload={},
        )
        data = fp.model_dump()
        restored = ForecastPoint.model_validate(data)
        assert restored == fp


class TestForecastResult:
    def test_round_trip(self) -> None:
        point = ForecastPoint(
            target_time=_HOUR,
            fetched_at=_NOW,
            provider="open-meteo",
            model="dwd-icon",
            location_id="warszawa-1",
            temperature_2m_c=18.5,
            raw_payload={},
        )
        result = ForecastResult(
            provider="open-meteo",
            model="dwd-icon",
            location=_WARSZAWA,
            fetched_at=_NOW,
            points=[point],
            raw_payload={"lat": 52.2297, "lon": 21.0122},
        )
        data = result.model_dump()
        restored = ForecastResult.model_validate(data)
        assert restored == result

    def test_empty_points(self) -> None:
        result = ForecastResult(
            provider="open-meteo",
            model="dwd-icon",
            location=_WARSZAWA,
            fetched_at=_NOW,
            points=[],
            raw_payload={},
        )
        assert result.points == []

    def test_model_none(self) -> None:
        result = ForecastResult(
            provider="open-meteo",
            model=None,
            location=_WARSZAWA,
            fetched_at=_NOW,
            points=[],
            raw_payload={},
        )
        assert result.model is None


class TestObservationPoint:
    def test_none_defaults(self) -> None:
        op = ObservationPoint(
            observed_at=_NOW,
            fetched_at=_NOW,
            provider="imgw-synop",
            raw_payload={"stacja": "Warszawa"},
        )
        assert op.station_id is None
        assert op.station_name is None
        assert op.distance_km is None
        assert op.temperature_c is None
        assert op.wind_speed_ms is None

    def test_explicit_values(self) -> None:
        op = ObservationPoint(
            observed_at=_NOW,
            fetched_at=_NOW,
            provider="imgw-synop",
            station_id="12345",
            station_name="Warszawa",
            distance_km=2.5,
            temperature_c=19.3,
            wind_speed_ms=3.8,
            raw_payload={"stacja": "Warszawa"},
        )
        assert op.temperature_c == 19.3
        assert op.distance_km == 2.5

    def test_round_trip(self) -> None:
        op = ObservationPoint(
            observed_at=_NOW,
            fetched_at=_NOW,
            provider="imgw-synop",
            station_id="12345",
            station_name="Warszawa",
            temperature_c=19.3,
            raw_payload={},
        )
        data = op.model_dump()
        restored = ObservationPoint.model_validate(data)
        assert restored == op


class TestObservationResult:
    def test_round_trip(self) -> None:
        point = ObservationPoint(
            observed_at=_NOW,
            fetched_at=_NOW,
            provider="imgw-synop",
            station_id="12345",
            station_name="Warszawa",
            temperature_c=19.3,
            raw_payload={},
        )
        result = ObservationResult(
            provider="imgw-synop",
            location=_WARSZAWA,
            fetched_at=_NOW,
            points=[point],
            raw_payload={},
        )
        data = result.model_dump()
        restored = ObservationResult.model_validate(data)
        assert restored == result


class TestWeatherWarning:
    def test_round_trip(self) -> None:
        w = WeatherWarning(
            provider="imgw",
            external_id="IMGW-METEO-2026-001",
            location_id="warszawa-1",
            severity="high",
            category="meteo",
            headline="Silny wiatr",
            description="Wiatr w porywach do 90 km/h",
            valid_from=_NOW,
            valid_to=_HOUR,
            raw_payload={"level": 2, "area": "Mazowieckie"},
        )
        data = w.model_dump()
        restored = WeatherWarning.model_validate(data)
        assert restored == w

    def test_severity_none(self) -> None:
        w = WeatherWarning(
            provider="imgw",
            external_id="IMGW-001",
            location_id="warszawa-1",
            severity=None,
            category="meteo",
            headline="Alert",
            description="Opis",
            valid_from=_NOW,
            valid_to=_HOUR,
            raw_payload={},
        )
        assert w.severity is None

    def test_raw_payload_preserved(self) -> None:
        raw = {"id": "abc", "level": 1}
        w = WeatherWarning(
            provider="imgw",
            external_id="x",
            location_id="warszawa-1",
            category="meteo",
            headline="h",
            description="d",
            valid_from=_NOW,
            valid_to=_HOUR,
            raw_payload=raw,
        )
        assert w.raw_payload == raw
        data = w.model_dump()
        assert data["raw_payload"] == raw


class TestWeatherVariable:
    def test_enum_values(self) -> None:
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
            "weather_code",
        }
        actual = {v.value for v in WeatherVariable}
        assert actual == expected


class TestForecastResolution:
    def test_enum_values(self) -> None:
        assert ForecastResolution.hourly.value == "hourly"
        assert ForecastResolution.fifteen_min.value == "fifteen_min"
