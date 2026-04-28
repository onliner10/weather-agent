from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from weather_agent.adapters.imgw.synop_provider import ImgwSynopProvider
from weather_agent.adapters.imgw.warnings_provider import (
    ImgwHydroWarningsProvider,
    ImgwMeteoWarningsProvider,
)
from weather_agent.domain.weather import (
    LocationRef,
    TimeRange,
    WarningCategory,
)
from weather_agent.settings import ImgwSettings

pytestmark = pytest.mark.smoke

WARSAW = LocationRef(id="warsaw", name="Warsaw", latitude=52.2297, longitude=21.0122)

_NOW = datetime.now(tz=UTC)
_WEEK_AHEAD = TimeRange(
    start=_NOW,
    end=_NOW + timedelta(days=7),
)


@pytest.mark.asyncio
async def test_imgw_synop_observations() -> None:
    settings = ImgwSettings()
    provider = ImgwSynopProvider(settings)

    result = await provider.get_observations(
        location=WARSAW,
        radius_km=50.0,
        variables=[],
    )

    assert result.provider == "imgw_synop", f"Unexpected provider: {result.provider}"
    assert result.raw_payload is not None, "raw_payload should be present in ObservationResult"

    payload_keys = (
        list(result.raw_payload.keys())
        if isinstance(result.raw_payload, dict)
        else type(result.raw_payload)
    )
    assert len(result.points) > 0, (
        f"Expected at least 1 observation station within 50 km of Warsaw, "
        f"got {len(result.points)}. IMGW synop API may be down. "
        f"raw_payload keys: {payload_keys}"
    )

    nearest = result.points[0]
    assert nearest.distance_km is not None, "Nearest observation should have distance_km set"
    assert nearest.distance_km <= 50.0, f"Nearest station too far: {nearest.distance_km} km"

    has_temperature = any(p.temperature_c is not None for p in result.points)
    assert has_temperature, (
        f"Expected at least one station with temperature_c, but all had None. "
        f"Stations: {[(p.station_name, p.station_id) for p in result.points[:5]]}"
    )


@pytest.mark.asyncio
async def test_imgw_meteo_warnings() -> None:
    settings = ImgwSettings()
    provider = ImgwMeteoWarningsProvider(settings)

    warnings = await provider.get_warnings(
        location=WARSAW,
        time_range=_WEEK_AHEAD,
    )

    assert isinstance(warnings, list), f"Expected list, got {type(warnings)}"

    for w in warnings:
        assert w.provider == "imgw-meteo", f"Unexpected warning provider: {w.provider}"
        assert w.category == WarningCategory.meteo, f"Unexpected category: {w.category}"
        assert w.valid_from is not None, "Warning should have valid_from"
        assert w.valid_to is not None, "Warning should have valid_to"
        assert w.raw_payload is not None, "Warning should preserve raw_payload"

    if warnings:
        first = warnings[0]
        assert first.external_id, "Warning should have a non-empty external_id"
        assert first.headline, "Warning should have a non-empty headline"


@pytest.mark.asyncio
async def test_imgw_hydro_warnings() -> None:
    settings = ImgwSettings()
    provider = ImgwHydroWarningsProvider(settings)

    warnings = await provider.get_warnings(
        location=WARSAW,
        time_range=_WEEK_AHEAD,
    )

    assert isinstance(warnings, list), f"Expected list, got {type(warnings)}"

    for w in warnings:
        assert w.provider == "imgw-hydro", f"Unexpected warning provider: {w.provider}"
        assert w.category == WarningCategory.hydro, f"Unexpected category: {w.category}"
        assert w.valid_from is not None, "Warning should have valid_from"
        assert w.valid_to is not None, "Warning should have valid_to"
        assert w.raw_payload is not None, "Warning should preserve raw_payload"

    if warnings:
        first = warnings[0]
        assert first.external_id, "Warning should have a non-empty external_id"
        assert first.headline, "Warning should have a non-empty headline"