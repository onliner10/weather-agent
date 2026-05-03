from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.application.conversation_models import BotAttachment
from weather_agent.domain.locations import LocationCreate, LocationService
from weather_agent.domain.providers import ForecastProvider, ObservationProvider
from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResolution,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
    TimeRange,
    WeatherVariable,
)
from weather_agent.infrastructure.db.base import AuthorizedUser, Base
from weather_agent.llm.tools.weather_tools import WeatherToolbox
from weather_agent.observability.metrics import (
    TOOL_CALL_DURATION_SECONDS,
    TOOL_CALLS_TOTAL,
    _count_histogram_observations,
    _get_labeled_value,
)


@pytest.fixture()
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


async def _create_user(session: AsyncSession, user_id: int = 1) -> None:
    session.add(AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user"))
    await session.flush()


async def _create_location(
    service: LocationService,
    *,
    name: str = "Chwarzno",
    aliases: list[str] | None = None,
) -> int:
    location = await service.create_location(
        1,
        LocationCreate(
            name=name,
            aliases=aliases or ["dom"],
            latitude=54.4871,
            longitude=18.4202,
        ),
    )
    return location.id


class FakeGeocoder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def geocode(self, query: str) -> LocationRef | None:
        self.queries.append(query)
        return LocationRef(id="geo", name=query, latitude=54.0, longitude=18.0)


class FakeForecastProvider(ForecastProvider):
    provider = "fake"

    def __init__(self, points: list[ForecastPoint] | None = None) -> None:
        self.locations: list[LocationRef] = []
        self.time_ranges: list[TimeRange] = []
        self.variables: list[list[WeatherVariable]] = []
        self._points = points

    async def get_forecast(
        self,
        location: LocationRef,
        time_range: TimeRange,
        variables: list[WeatherVariable],
        resolution: ForecastResolution,
    ) -> ForecastResult:
        del resolution
        self.locations.append(location)
        self.time_ranges.append(time_range)
        self.variables.append(variables)
        fetched_at = datetime.now(UTC)
        return ForecastResult(
            provider=self.provider,
            model="fake",
            location=location,
            fetched_at=fetched_at,
            points=self._points
            if self._points is not None
            else [
                ForecastPoint(
                    target_time=time_range.start,
                    fetched_at=fetched_at,
                    provider=self.provider,
                    model="fake",
                    location_id=location.id,
                    temperature_2m_c=12.0,
                    wind_speed_10m_ms=4.0,
                    wind_gusts_10m_ms=7.0,
                    raw_payload={},
                )
            ],
            raw_payload={},
        )


class FakeObservationProvider(ObservationProvider):
    provider = "fake"

    async def get_observations(
        self,
        location: LocationRef,
        radius_km: float,
        variables: list[WeatherVariable],
    ) -> ObservationResult:
        del radius_km, variables
        fetched_at = datetime.now(UTC)
        return ObservationResult(
            provider=self.provider,
            location=location,
            fetched_at=fetched_at,
            points=[
                ObservationPoint(
                    observed_at=fetched_at,
                    fetched_at=fetched_at,
                    provider=self.provider,
                    station_name="fake",
                    temperature_c=12.0,
                    raw_payload={},
                )
            ],
            raw_payload={},
        )


def _toolbox(
    *,
    service: LocationService,
    geocoder: FakeGeocoder,
    forecast_provider: FakeForecastProvider | None = None,
) -> WeatherToolbox:
    return WeatherToolbox(
        forecast_provider=forecast_provider or FakeForecastProvider(),
        observation_provider=FakeObservationProvider(),
        geocoder=geocoder,  # type: ignore[arg-type]
        location_service=service,
        user_id=1,
    )


class TestWeatherToolboxLocations:
    async def test_forecast_uses_default_location_when_name_is_empty(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        await _create_location(service, name="Chwarzno", aliases=["dom"])
        geocoder = FakeGeocoder()
        forecast_provider = FakeForecastProvider()
        toolbox = _toolbox(service=service, geocoder=geocoder, forecast_provider=forecast_provider)

        result = await toolbox.get_forecast("", "2026-05-02", "2026-05-02", ["temperature_2m_c"])

        assert result.get("error") is None
        assert result["location"] == "Chwarzno"
        assert forecast_provider.locations[0].name == "Chwarzno"
        assert geocoder.queries == []

    async def test_empty_location_without_default_returns_polish_error_without_geocoding(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        geocoder = FakeGeocoder()
        toolbox = _toolbox(service=service, geocoder=geocoder)

        result = await toolbox.get_observations("")

        assert result["error"] == "Nie mam zapisanej domyślnej lokalizacji. Podaj lokalizację."
        assert geocoder.queries == []

    async def test_edit_location_updates_aliases_by_existing_alias(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        loc_id = await _create_location(service, name="Chwarzno", aliases=["dom"])
        toolbox = _toolbox(service=service, geocoder=FakeGeocoder())

        result = await toolbox.edit_location("dom", aliases=["mieszkanie"])

        assert result.get("error") is None
        updated = await service.get_location(loc_id)
        assert updated is not None
        assert updated.aliases == ["mieszkanie"]

    async def test_remove_location_disables_instead_of_deleting(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        loc_id = await _create_location(service, name="Gdańsk Wrzeszcz", aliases=["praca"])
        toolbox = _toolbox(service=service, geocoder=FakeGeocoder())

        result = await toolbox.remove_location("praca")

        assert result.get("error") is None
        fetched = await service.get_location(loc_id)
        assert fetched is not None
        assert fetched.enabled is False
        assert await service.list_locations(1) == []

    async def test_tool_instrumentation_records_call_count_and_duration(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        toolbox = _toolbox(service=service, geocoder=FakeGeocoder())
        calls_before = _get_labeled_value(TOOL_CALLS_TOTAL, tool="get_observations")
        durations_before = _count_histogram_observations(
            TOOL_CALL_DURATION_SECONDS,
            tool="get_observations",
        )

        await toolbox.get_observations("")

        assert _get_labeled_value(TOOL_CALLS_TOTAL, tool="get_observations") == calls_before + 1
        assert (
            _count_histogram_observations(
                TOOL_CALL_DURATION_SECONDS,
                tool="get_observations",
            )
            == durations_before + 1
        )


class TestWeatherToolboxCharts:
    async def test_render_forecast_chart_adds_png_attachment(self, session: AsyncSession) -> None:
        await _create_user(session)
        service = LocationService(session)
        await _create_location(service, name="Chwarzno", aliases=["dom"])
        forecast_provider = FakeForecastProvider()
        attachments: list[BotAttachment] = []
        toolbox = WeatherToolbox(
            forecast_provider=forecast_provider,
            observation_provider=FakeObservationProvider(),
            geocoder=FakeGeocoder(),  # type: ignore[arg-type]
            location_service=service,
            user_id=1,
            reply_attachments=attachments,
        )

        result = await toolbox.render_forecast_chart(
            "dom",
            "2026-05-04",
            "2026-05-04",
            ["wind_speed_10m_ms", "wind_gusts_10m_ms"],
            {
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "title": "Wiatr w czasie",
                "data": {"name": "forecast"},
                "layer": [
                    {
                        "mark": "line",
                        "encoding": {
                            "x": {"field": "time", "type": "temporal"},
                            "y": {"field": "wind_speed_10m_ms", "type": "quantitative"},
                        },
                    },
                    {
                        "mark": "line",
                        "encoding": {
                            "x": {"field": "time", "type": "temporal"},
                            "y": {"field": "wind_gusts_10m_ms", "type": "quantitative"},
                        },
                    },
                ],
            },
        )

        assert result.get("error") is None
        assert forecast_provider.variables == [
            [WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms]
        ]
        assert len(attachments) == 1
        assert attachments[0].media_type == "image/png"
        assert attachments[0].data.startswith(b"\x89PNG\r\n\x1a\n")

    async def test_render_forecast_chart_tool_defaults_when_spec_missing(
        self, session: AsyncSession
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        await _create_location(service, name="Chwarzno", aliases=["dom"])
        forecast_provider = FakeForecastProvider()
        attachments: list[BotAttachment] = []
        toolbox = WeatherToolbox(
            forecast_provider=forecast_provider,
            observation_provider=FakeObservationProvider(),
            geocoder=FakeGeocoder(),  # type: ignore[arg-type]
            location_service=service,
            user_id=1,
            reply_attachments=attachments,
        )
        chart_tool = next(
            tool for tool in toolbox.to_langchain_tools() if tool.name == "render_forecast_chart"
        )

        result = await chart_tool.ainvoke(
            {
                "location_name": "dom",
                "start_date": "2026-05-04",
                "end_date": "2026-05-04",
                "variables": ["wind_speed_10m_ms", "wind_gusts_10m_ms"],
            }
        )

        assert isinstance(result, dict)
        assert result.get("error") is None
        assert forecast_provider.variables == [
            [WeatherVariable.wind_speed_10m_ms, WeatherVariable.wind_gusts_10m_ms]
        ]
        assert len(attachments) == 1
        assert attachments[0].media_type == "image/png"
        assert attachments[0].data.startswith(b"\x89PNG\r\n\x1a\n")

    async def test_render_forecast_chart_respects_hour_window(
        self, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await _create_user(session)
        service = LocationService(session)
        await _create_location(service, name="Chwarzno", aliases=["dom"])
        fetched_at = datetime(2026, 5, 4, tzinfo=UTC)
        points = [
            ForecastPoint(
                target_time=datetime(2026, 5, 4, hour, tzinfo=UTC),
                fetched_at=fetched_at,
                provider="fake",
                model="fake",
                location_id="loc",
                wind_speed_10m_ms=float(hour),
                wind_gusts_10m_ms=float(hour + 2),
                raw_payload={},
            )
            for hour in (11, 12, 17, 18)
        ]
        forecast_provider = FakeForecastProvider(points=points)
        attachments: list[BotAttachment] = []
        captured: dict[str, object] = {}

        def fake_render_forecast_chart_png(
            *,
            spec: dict[str, object],
            records: list[dict[str, object]],
            variables: list[WeatherVariable],
            time_range: TimeRange,
        ) -> bytes:
            del spec, variables
            captured["records"] = records
            captured["time_range"] = time_range
            return b"\x89PNG\r\n\x1a\nfake"

        monkeypatch.setattr(
            "weather_agent.llm.tools.weather_tools.render_forecast_chart_png",
            fake_render_forecast_chart_png,
        )
        toolbox = WeatherToolbox(
            forecast_provider=forecast_provider,
            observation_provider=FakeObservationProvider(),
            geocoder=FakeGeocoder(),  # type: ignore[arg-type]
            location_service=service,
            user_id=1,
            reply_attachments=attachments,
        )

        result = await toolbox.render_forecast_chart(
            location_name="dom",
            start_date="2026-05-04",
            end_date="2026-05-04",
            variables=["wind_speed_10m_ms", "wind_gusts_10m_ms"],
            start_time="14:00",
            end_time="19:00",
        )

        assert result.get("error") is None
        assert result["time_range"] == "2026-05-04 14:00 – 19:00"
        assert forecast_provider.time_ranges[0].start.hour == 14
        assert forecast_provider.time_ranges[0].end.hour == 19
        assert captured["records"] == [
            {
                "time": "2026-05-04T14:00:00",
                "wind_speed_10m_ms": 12.0,
                "wind_gusts_10m_ms": 14.0,
            },
            {
                "time": "2026-05-04T19:00:00",
                "wind_speed_10m_ms": 17.0,
                "wind_gusts_10m_ms": 19.0,
            },
        ]
        assert len(attachments) == 1
