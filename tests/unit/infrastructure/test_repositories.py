from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from weather_agent.domain.weather import (
    ForecastPoint,
    ForecastResult,
    LocationRef,
    ObservationPoint,
    ObservationResult,
    WeatherWarning,
)
from weather_agent.infrastructure.db.base import AuthorizedUser as AuthorizedUserORM
from weather_agent.infrastructure.db.base import Base, GlobalSetting
from weather_agent.infrastructure.repositories.auth_repository import (
    AuthRepository,
    InviteRedeemStatus,
)
from weather_agent.infrastructure.repositories.forecast_repository import ForecastRepository
from weather_agent.infrastructure.repositories.observation_repository import ObservationRepository
from weather_agent.infrastructure.repositories.warning_repository import WarningRepository


def _make_location(location_id: str = "1") -> LocationRef:
    return LocationRef(
        id=location_id,
        name="Warsaw",
        latitude=52.2297,
        longitude=21.0122,
    )


def _make_forecast_result(
    location_id: str = "1",
    fetched_at: datetime | None = None,
    num_points: int = 3,
) -> ForecastResult:
    fetched = fetched_at or datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    points = []
    for i in range(num_points):
        target_time = fetched + timedelta(hours=i + 1)
        points.append(
            ForecastPoint(
                target_time=target_time,
                fetched_at=fetched,
                provider="open-meteo",
                model="dwd-icon",
                location_id=location_id,
                temperature_2m_c=5.0 + i,
                apparent_temperature_c=3.0 + i,
                precipitation_mm=0.1 * i,
                precipitation_probability_pct=10.0 * i,
                rain_mm=0.1 * i,
                snowfall_cm=0.0,
                cloud_cover_pct=50.0 + i * 10,
                wind_speed_10m_ms=3.0 + i,
                wind_gusts_10m_ms=6.0 + i,
                wind_direction_10m_deg=180.0,
                pressure_msl_hpa=1013.0,
                relative_humidity_2m_pct=70.0,
                weather_code="1",
                raw_payload={"hourly": {"time": [target_time.isoformat()]}},
            )
        )
    return ForecastResult(
        provider="open-meteo",
        model="dwd-icon",
        location=_make_location(location_id),
        fetched_at=fetched,
        points=points,
        raw_payload={"source": "test"},
    )


def _make_observation_result(
    location_id: str = "1",
    fetched_at: datetime | None = None,
) -> ObservationResult:
    fetched = fetched_at or datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
    points = [
        ObservationPoint(
            observed_at=fetched - timedelta(minutes=30),
            fetched_at=fetched,
            provider="imgw-synop",
            station_id="12345",
            station_name="Warszawa",
            distance_km=5.2,
            temperature_c=3.5,
            wind_speed_ms=4.0,
            wind_direction_deg=200.0,
            pressure_hpa=1012.0,
            humidity_pct=75.0,
            precipitation_mm=0.0,
            raw_payload={"station": "Warszawa", "temp": "3.5"},
        ),
    ]
    return ObservationResult(
        provider="imgw-synop",
        location=_make_location(location_id),
        fetched_at=fetched,
        points=points,
        raw_payload={"source": "test-obs"},
    )


def _make_weather_warning(
    external_id: str = "warn-1",
    location_id: str = "1",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    provider: str = "imgw-meteo",
) -> WeatherWarning:
    vf = valid_from or datetime(2026, 1, 15, 6, 0, tzinfo=UTC)
    vt = valid_to or datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
    return WeatherWarning(
        provider=provider,
        external_id=external_id,
        location_id=location_id,
        severity="high",
        category="meteo",
        headline="Strong wind warning",
        description="Wind gusts up to 90 km/h expected.",
        valid_from=vf,
        valid_to=vt,
        raw_payload={"id": external_id, "level": "high"},
    )


@pytest_asyncio.fixture()
async def async_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture()
async def async_session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        async with session.begin():
            yield session


class TestForecastRepository:
    @pytest.mark.asyncio()
    async def test_save_snapshot_returns_id(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result()
        snapshot_id = await repo.save_snapshot(result)
        assert snapshot_id is not None
        assert isinstance(snapshot_id, int)
        assert snapshot_id > 0

    @pytest.mark.asyncio()
    async def test_save_snapshot_persists_points(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result(num_points=5)
        await repo.save_snapshot(result)
        await async_session.flush()

        points = await repo.get_points_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(points) == 5

    @pytest.mark.asyncio()
    async def test_get_latest_snapshot(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        early = _make_forecast_result(fetched_at=datetime(2026, 1, 15, 6, 0, tzinfo=UTC))
        late = _make_forecast_result(fetched_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC))
        await repo.save_snapshot(early)
        await repo.save_snapshot(late)
        await async_session.flush()

        latest = await repo.get_latest_snapshot(location_id="1")
        assert latest is not None
        assert latest.fetched_at == datetime(2026, 1, 15, 12, 0)

    @pytest.mark.asyncio()
    async def test_get_latest_snapshot_none_if_empty(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        result = await repo.get_latest_snapshot(location_id="999")
        assert result is None

    @pytest.mark.asyncio()
    async def test_get_points_by_time_range_ordered_by_target_time(
        self, async_session: AsyncSession
    ) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result(num_points=3)
        await repo.save_snapshot(result)
        await async_session.flush()

        points = await repo.get_points_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(points) == 3
        for i in range(len(points) - 1):
            assert points[i].target_time <= points[i + 1].target_time

    @pytest.mark.asyncio()
    async def test_get_points_by_time_range_filters_location(
        self, async_session: AsyncSession
    ) -> None:
        repo = ForecastRepository(async_session)
        result1 = _make_forecast_result(location_id="1", num_points=2)
        result2 = _make_forecast_result(location_id="2", num_points=2)
        await repo.save_snapshot(result1)
        await repo.save_snapshot(result2)
        await async_session.flush()

        points = await repo.get_points_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert all(p.location_id == 1 for p in points)
        assert len(points) == 2

    @pytest.mark.asyncio()
    async def test_get_previous_snapshot(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        early = _make_forecast_result(fetched_at=datetime(2026, 1, 10, 6, 0, tzinfo=UTC))
        late = _make_forecast_result(fetched_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC))
        await repo.save_snapshot(early)
        await repo.save_snapshot(late)
        await async_session.flush()

        prev = await repo.get_previous_snapshot(
            location_id="1",
            before=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
        )
        assert prev is not None
        assert prev.fetched_at == datetime(2026, 1, 10, 6, 0)

    @pytest.mark.asyncio()
    async def test_get_previous_snapshot_none_if_no_earlier(
        self, async_session: AsyncSession
    ) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result(fetched_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC))
        await repo.save_snapshot(result)
        await async_session.flush()

        prev = await repo.get_previous_snapshot(
            location_id="1",
            before=datetime(2026, 1, 10, 0, 0, tzinfo=UTC),
        )
        assert prev is None

    @pytest.mark.asyncio()
    async def test_raw_payload_recoverable_from_snapshot(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result()
        await repo.save_snapshot(result)
        await async_session.flush()

        snapshot = await repo.get_latest_snapshot(location_id="1")
        assert snapshot is not None
        assert snapshot.raw_payload == {"source": "test"}

    @pytest.mark.asyncio()
    async def test_raw_payload_recoverable_from_points(self, async_session: AsyncSession) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result(num_points=2)
        await repo.save_snapshot(result)
        await async_session.flush()

        points = await repo.get_points_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(points) == 2
        for p in points:
            assert "hourly" in p.raw_payload

    @pytest.mark.asyncio()
    async def test_normalized_fields_recoverable_from_points(
        self, async_session: AsyncSession
    ) -> None:
        repo = ForecastRepository(async_session)
        result = _make_forecast_result(num_points=1)
        await repo.save_snapshot(result)
        await async_session.flush()

        points = await repo.get_points_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(points) == 1
        point = points[0]
        assert point.temperature_2m_c == 5.0
        assert point.apparent_temperature_c == 3.0
        assert point.wind_speed_10m_ms == 3.0
        assert point.weather_code == "1"

    @pytest.mark.asyncio()
    async def test_get_points_for_snapshot_scopes_by_snapshot_id(
        self, async_session: AsyncSession
    ) -> None:
        repo = ForecastRepository(async_session)
        early = _make_forecast_result(
            fetched_at=datetime(2026, 1, 15, 6, 0, tzinfo=UTC),
            num_points=2,
        )
        late = _make_forecast_result(
            fetched_at=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
            num_points=3,
        )
        await repo.save_snapshot(early)
        await repo.save_snapshot(late)
        await async_session.flush()

        early_snapshot = await repo.get_latest_snapshot(location_id="1")
        assert early_snapshot is not None
        late_snapshot_id = early_snapshot.id

        all_points = await repo.get_points_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(all_points) == 5

        late_points = await repo.get_points_for_snapshot(
            snapshot_id=late_snapshot_id,
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(late_points) == 3


class TestAuthRepository:
    @pytest.mark.asyncio()
    async def test_add_user_persists_authorized_user(self, async_session: AsyncSession) -> None:
        repo = AuthRepository(async_session)
        await repo.add_user(123)
        await async_session.flush()

        users = await repo.list_users()
        assert users[0].telegram_user_id == 123
        assert users[0].role == "user"

    @pytest.mark.asyncio()
    async def test_list_user_ids_sorted(self, async_session: AsyncSession) -> None:
        repo = AuthRepository(async_session)
        await repo.add_user(300)
        await repo.add_user(100)
        await async_session.flush()

        assert await repo.list_user_ids() == [100, 300]

    @pytest.mark.asyncio()
    async def test_create_and_redeem_invite_code(self, async_session: AsyncSession) -> None:
        repo = AuthRepository(async_session)
        now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
        await repo.create_invite_code(
            code="ABC123",
            created_by=42,
            expires_at=now + timedelta(hours=24),
        )

        result = await repo.redeem_invite_code(
            code="ABC123",
            telegram_user_id=100,
            now=now,
        )

        assert result.status is InviteRedeemStatus.REDEEMED
        user = (
            await async_session.execute(
                select(AuthorizedUserORM).where(AuthorizedUserORM.telegram_user_id == 100)
            )
        ).scalar_one()
        assert user.role == "user"
        setting = (
            await async_session.execute(
                select(GlobalSetting).where(GlobalSetting.key == "telegram_invite:ABC123")
            )
        ).scalar_one()
        assert '"used_by":100' in setting.value

    @pytest.mark.asyncio()
    async def test_redeem_invite_code_only_once(self, async_session: AsyncSession) -> None:
        repo = AuthRepository(async_session)
        now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
        await repo.create_invite_code(
            code="ABC123",
            created_by=42,
            expires_at=now + timedelta(hours=24),
        )
        first = await repo.redeem_invite_code(
            code="ABC123",
            telegram_user_id=100,
            now=now,
        )
        second = await repo.redeem_invite_code(
            code="ABC123",
            telegram_user_id=101,
            now=now,
        )

        assert first.status is InviteRedeemStatus.REDEEMED
        assert second.status is InviteRedeemStatus.USED

    @pytest.mark.asyncio()
    async def test_redeem_expired_invite_code(self, async_session: AsyncSession) -> None:
        repo = AuthRepository(async_session)
        now = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
        await repo.create_invite_code(
            code="OLD",
            created_by=42,
            expires_at=now - timedelta(seconds=1),
        )

        result = await repo.redeem_invite_code(
            code="OLD",
            telegram_user_id=100,
            now=now,
        )

        assert result.status is InviteRedeemStatus.EXPIRED


class TestObservationRepository:
    @pytest.mark.asyncio()
    async def test_save_observations_returns_ids(self, async_session: AsyncSession) -> None:
        repo = ObservationRepository(async_session)
        result = _make_observation_result()
        ids = await repo.save_observations(result)
        assert len(ids) == 1
        assert all(isinstance(i, int) for i in ids)

    @pytest.mark.asyncio()
    async def test_get_observations_by_time_range(self, async_session: AsyncSession) -> None:
        repo = ObservationRepository(async_session)
        result = _make_observation_result()
        await repo.save_observations(result)
        await async_session.flush()

        obs = await repo.get_observations_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(obs) == 1

    @pytest.mark.asyncio()
    async def test_observation_fields_recoverable(self, async_session: AsyncSession) -> None:
        repo = ObservationRepository(async_session)
        result = _make_observation_result()
        await repo.save_observations(result)
        await async_session.flush()

        obs = await repo.get_observations_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(obs) == 1
        o = obs[0]
        assert o.provider == "imgw-synop"
        assert o.station_id == "12345"
        assert o.station_name == "Warszawa"
        assert o.temperature_c == 3.5
        assert o.wind_speed_ms == 4.0
        assert o.raw_payload == {"station": "Warszawa", "temp": "3.5"}

    @pytest.mark.asyncio()
    async def test_observation_filters_by_location(self, async_session: AsyncSession) -> None:
        repo = ObservationRepository(async_session)
        r1 = _make_observation_result(location_id="1")
        r2 = _make_observation_result(location_id="2")
        await repo.save_observations(r1)
        await repo.save_observations(r2)
        await async_session.flush()

        obs = await repo.get_observations_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(obs) == 1
        assert obs[0].location_id == 1

    @pytest.mark.asyncio()
    async def test_observation_ordered_by_observed_at(self, async_session: AsyncSession) -> None:
        repo = ObservationRepository(async_session)
        fetched = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        obs1 = ObservationPoint(
            observed_at=fetched - timedelta(hours=2),
            fetched_at=fetched,
            provider="imgw-synop",
            station_id="1",
            temperature_c=2.0,
            raw_payload={},
        )
        obs2 = ObservationPoint(
            observed_at=fetched - timedelta(hours=1),
            fetched_at=fetched,
            provider="imgw-synop",
            station_id="2",
            temperature_c=3.0,
            raw_payload={},
        )
        result = ObservationResult(
            provider="imgw-synop",
            location=_make_location("1"),
            fetched_at=fetched,
            points=[obs2, obs1],
            raw_payload={},
        )
        await repo.save_observations(result)
        await async_session.flush()

        obs_list = await repo.get_observations_by_time_range(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(obs_list) == 2
        assert obs_list[0].observed_at <= obs_list[1].observed_at


class TestWarningRepository:
    @pytest.mark.asyncio()
    async def test_save_warnings_returns_ids(self, async_session: AsyncSession) -> None:
        repo = WarningRepository(async_session)
        warnings = [_make_weather_warning()]
        ids = await repo.save_warnings(warnings)
        assert len(ids) == 1
        assert isinstance(ids[0], int)

    @pytest.mark.asyncio()
    async def test_warning_dedup_same_provider_external_id_valid_from(
        self, async_session: AsyncSession
    ) -> None:
        repo = WarningRepository(async_session)
        warning = _make_weather_warning(external_id="w1")
        ids1 = await repo.save_warnings([warning])
        await async_session.flush()

        ids2 = await repo.save_warnings([warning])
        await async_session.flush()

        assert len(ids1) == 1
        assert len(ids2) == 1
        assert ids1[0] == ids2[0]

    @pytest.mark.asyncio()
    async def test_warning_allows_different_provider(self, async_session: AsyncSession) -> None:
        repo = WarningRepository(async_session)
        w1 = _make_weather_warning(external_id="w1", provider="imgw-meteo")
        w2 = _make_weather_warning(external_id="w1", provider="imgw-hydro")
        ids = await repo.save_warnings([w1, w2])
        await async_session.flush()

        assert len(ids) == 2
        assert ids[0] != ids[1]

    @pytest.mark.asyncio()
    async def test_warning_allows_different_external_id(self, async_session: AsyncSession) -> None:
        repo = WarningRepository(async_session)
        w1 = _make_weather_warning(external_id="w1")
        w2 = _make_weather_warning(external_id="w2")
        ids = await repo.save_warnings([w1, w2])
        await async_session.flush()

        assert len(ids) == 2
        assert ids[0] != ids[1]

    @pytest.mark.asyncio()
    async def test_warning_allows_different_valid_from(self, async_session: AsyncSession) -> None:
        repo = WarningRepository(async_session)
        vf1 = datetime(2026, 1, 15, 6, 0, tzinfo=UTC)
        vf2 = datetime(2026, 1, 16, 6, 0, tzinfo=UTC)
        vt1 = datetime(2026, 1, 15, 18, 0, tzinfo=UTC)
        vt2 = datetime(2026, 1, 16, 18, 0, tzinfo=UTC)
        w1 = _make_weather_warning(external_id="w1", valid_from=vf1, valid_to=vt1)
        w2 = _make_weather_warning(external_id="w1", valid_from=vf2, valid_to=vt2)
        ids = await repo.save_warnings([w1, w2])
        await async_session.flush()

        assert len(ids) == 2

    @pytest.mark.asyncio()
    async def test_get_warnings_by_time_range(self, async_session: AsyncSession) -> None:
        repo = WarningRepository(async_session)
        w = _make_weather_warning(
            valid_from=datetime(2026, 1, 15, 6, 0, tzinfo=UTC),
            valid_to=datetime(2026, 1, 15, 18, 0, tzinfo=UTC),
        )
        await repo.save_warnings([w])
        await async_session.flush()

        result = await repo.get_warnings(
            location_id="1",
            start=datetime(2026, 1, 15, 0, 0, tzinfo=UTC),
            end=datetime(2026, 1, 15, 23, 59, tzinfo=UTC),
        )
        assert len(result) == 1
        assert result[0].headline == "Strong wind warning"

    @pytest.mark.asyncio()
    async def test_warning_raw_payload_recoverable(self, async_session: AsyncSession) -> None:
        repo = WarningRepository(async_session)
        w = _make_weather_warning()
        await repo.save_warnings([w])
        await async_session.flush()

        result = await repo.get_warnings(
            location_id="1",
            start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            end=datetime(2026, 12, 31, 23, 59, tzinfo=UTC),
        )
        assert len(result) == 1
        assert result[0].raw_payload == {"id": "warn-1", "level": "high"}
