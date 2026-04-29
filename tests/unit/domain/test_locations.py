from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from weather_agent.domain.locations import (
    LocationAliasConflictError,
    LocationCreate,
    LocationNameConflictError,
    LocationService,
    LocationUpdate,
)
from weather_agent.domain.polish_utils import normalize_for_matching as _normalize_for_matching
from weather_agent.infrastructure.db.base import Base


@pytest.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture()
def service(session: AsyncSession) -> LocationService:
    return LocationService(session)


async def _create_user(session: AsyncSession, user_id: int = 1) -> None:
    from weather_agent.infrastructure.db.base import AuthorizedUser

    user = AuthorizedUser(id=user_id, telegram_user_id=user_id * 1000, role="user")
    session.add(user)
    await session.flush()


class TestNormalizeForMatching:
    def test_case_insensitive(self) -> None:
        assert _normalize_for_matching("Warszawa") == _normalize_for_matching(
            "warszawa"
        )

    def test_polish_chars(self) -> None:
        assert _normalize_for_matching("żółw") == _normalize_for_matching("zolw")
        assert _normalize_for_matching("Łódź") == _normalize_for_matching("lodz")

    def test_polish_chars_normalized(self) -> None:
        result = _normalize_for_matching("żółw")
        assert "z" in result
        assert "ż" not in result

    def test_whitespace_stripped(self) -> None:
        assert _normalize_for_matching("  Warszawa  ") == _normalize_for_matching(
            "Warszawa"
        )


class TestCreateLocation:
    async def test_create_basic(self, service: LocationService, session: AsyncSession) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Chwarzno", aliases=["chwarzno", "chwar"], latitude=54.6, longitude=18.5
        )
        location = await service.create_location(1, data)
        assert location.id > 0
        assert location.name == "Chwarzno"
        assert location.aliases == ["chwarzno", "chwar"]
        assert location.latitude == 54.6
        assert location.longitude == 18.5
        assert location.enabled is True
        assert location.description is None

    async def test_create_with_description(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home",
            aliases=["dom", "house"],
            latitude=52.2297,
            longitude=21.0122,
            description="My home in Warsaw",
        )
        location = await service.create_location(1, data)
        assert location.description == "My home in Warsaw"

    async def test_create_rejects_duplicate_name(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Chwarzno", aliases=["ch1"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data1)
        data2 = LocationCreate(
            name="chwarzno", aliases=["ch2"], latitude=54.7, longitude=18.6
        )
        with pytest.raises(LocationNameConflictError) as exc_info:
            await service.create_location(1, data2)
        assert exc_info.value.conflicting_location_id > 0

    async def test_create_rejects_alias_matching_existing_name(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Chwarzno", aliases=["ch1"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data1)
        data2 = LocationCreate(
            name="Jeziorak", aliases=["chwarzno"], latitude=53.6, longitude=19.6
        )
        with pytest.raises(LocationAliasConflictError):
            await service.create_location(1, data2)

    async def test_create_rejects_duplicate_alias(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Chwarzno", aliases=["chwarzno", "chw"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data1)
        data2 = LocationCreate(
            name="Jeziorak", aliases=["jeziorak", "chw"], latitude=53.6, longitude=19.6
        )
        with pytest.raises(LocationAliasConflictError):
            await service.create_location(1, data2)

    async def test_create_different_users_same_name_ok(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session, user_id=1)
        await _create_user(session, user_id=2)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.2297, longitude=21.0122
        )
        loc1 = await service.create_location(1, data)
        loc2 = await service.create_location(2, data)
        assert loc1.id != loc2.id

    async def test_create_rejects_polish_char_conflict_in_name(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Łódź", aliases=["lodz"], latitude=51.7, longitude=19.4
        )
        await service.create_location(1, data1)
        data2 = LocationCreate(
            name="Lodz", aliases=["city2"], latitude=52.7, longitude=20.4
        )
        with pytest.raises(LocationNameConflictError):
            await service.create_location(1, data2)


class TestListLocations:
    async def test_list_empty(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        result = await service.list_locations(1)
        assert result == []

    async def test_list_returns_locations(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        data2 = LocationCreate(
            name="Chwarzno", aliases=["chwarzno"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data1)
        await service.create_location(1, data2)
        result = await service.list_locations(1)
        assert len(result) == 2

    async def test_list_excludes_disabled_by_default(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        await service.disable_location(loc.id)
        result = await service.list_locations(1)
        assert result == []

    async def test_list_includes_disabled_when_requested(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        await service.disable_location(loc.id)
        result = await service.list_locations(1, include_disabled=True)
        assert len(result) == 1


class TestGetLocation:
    async def test_get_existing(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        created = await service.create_location(1, data)
        fetched = await service.get_location(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Home"

    async def test_get_nonexistent(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        result = await service.get_location(9999)
        assert result is None


class TestUpdateLocation:
    async def test_update_name(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        updated = await service.update_location(loc.id, LocationUpdate(name="Mój dom"))
        assert updated.name == "Mój dom"

    async def test_update_aliases(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        updated = await service.update_location(
            loc.id, LocationUpdate(aliases=["dom", "house"])
        )
        assert updated.aliases == ["dom", "house"]

    async def test_update_rejects_conflicting_alias(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Chwarzno", aliases=["chwarzno"], latitude=54.6, longitude=18.5
        )
        data2 = LocationCreate(
            name="Jeziorak", aliases=["jeziorak"], latitude=53.6, longitude=19.6
        )
        loc1 = await service.create_location(1, data1)
        await service.create_location(1, data2)
        with pytest.raises(LocationAliasConflictError):
            await service.update_location(
                loc1.id, LocationUpdate(aliases=["jeziorak"])
            )

    async def test_update_nonexistent_raises(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        with pytest.raises(ValueError):
            await service.update_location(9999, LocationUpdate(name="x"))

    async def test_update_coordinates(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        updated = await service.update_location(
            loc.id, LocationUpdate(latitude=52.23, longitude=21.02)
        )
        assert updated.latitude == 52.23
        assert updated.longitude == 21.02


class TestDeleteLocation:
    async def test_delete_existing(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        result = await service.delete_location(loc.id)
        assert result is True
        fetched = await service.get_location(loc.id)
        assert fetched is None

    async def test_delete_nonexistent(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        result = await service.delete_location(9999)
        assert result is False


class TestEnableDisable:
    async def test_disable_location(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        assert loc.enabled is True
        disabled = await service.disable_location(loc.id)
        assert disabled.enabled is False

    async def test_enable_location(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        await service.disable_location(loc.id)
        enabled = await service.enable_location(loc.id)
        assert enabled.enabled is True

    async def test_enable_nonexistent_raises(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        with pytest.raises(ValueError):
            await service.enable_location(9999)

    async def test_disable_nonexistent_raises(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        with pytest.raises(ValueError):
            await service.disable_location(9999)


class TestResolveLocation:
    async def test_resolve_by_name(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Chwarzno", aliases=["chw"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data)
        ref = await service.resolve_location("Chwarzno", user_id=1)
        assert ref is not None
        assert ref.name == "Chwarzno"
        assert ref.latitude == 54.6

    async def test_resolve_by_alias(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Chwarzno", aliases=["chw", "chwarzno"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data)
        ref = await service.resolve_location("chw", user_id=1)
        assert ref is not None
        assert ref.name == "Chwarzno"

    async def test_resolve_case_insensitive(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Chwarzno", aliases=["chw"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data)
        ref = await service.resolve_location("chwarzno", user_id=1)
        assert ref is not None
        assert ref.name == "Chwarzno"

    async def test_resolve_polish_chars(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Łódź", aliases=["lodz"], latitude=51.7, longitude=19.4
        )
        await service.create_location(1, data)
        ref = await service.resolve_location("Lodz", user_id=1)
        assert ref is not None
        assert ref.name == "Łódź"

    async def test_resolve_by_alias_polish_chars(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Żółw", aliases=["żółw", "zolw"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data)
        ref = await service.resolve_location("zolw", user_id=1)
        assert ref is not None
        assert ref.name == "Żółw"

    async def test_resolve_returns_none_for_unknown(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        ref = await service.resolve_location("nonexistent", user_id=1)
        assert ref is None

    async def test_resolve_skips_disabled(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Chwarzno", aliases=["chw"], latitude=54.6, longitude=18.5
        )
        loc = await service.create_location(1, data)
        await service.disable_location(loc.id)
        ref = await service.resolve_location("Chwarzno", user_id=1)
        assert ref is None

    async def test_resolve_returns_location_ref(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Chwarzno", aliases=["chw"], latitude=54.6, longitude=18.5
        )
        loc = await service.create_location(1, data)
        ref = await service.resolve_location("Chwarzno", user_id=1)
        assert ref is not None
        assert ref.id == str(loc.id)
        assert ref.latitude == 54.6
        assert ref.longitude == 18.5

    async def test_resolve_does_not_cross_users(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session, user_id=1)
        await _create_user(session, user_id=2)
        data = LocationCreate(
            name="Chwarzno", aliases=["chw"], latitude=54.6, longitude=18.5
        )
        await service.create_location(1, data)
        ref = await service.resolve_location("Chwarzno", user_id=2)
        assert ref is None


class TestGetDefaultLocation:
    async def test_no_locations_returns_none(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        result = await service.get_default_location(1)
        assert result is None

    async def test_one_location_returned(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        ref = await service.get_default_location(1)
        assert ref is not None
        assert ref.id == str(loc.id)
        assert ref.name == "Home"
        assert ref.latitude == 52.22
        assert ref.longitude == 21.01

    async def test_disabled_location_skipped(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc = await service.create_location(1, data)
        await service.disable_location(loc.id)
        result = await service.get_default_location(1)
        assert result is None

    async def test_disabled_location_ignored(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Work", aliases=["praca"], latitude=54.6, longitude=18.5
        )
        data2 = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        loc1 = await service.create_location(1, data1)
        await service.disable_location(loc1.id)
        await service.create_location(1, data2)
        ref = await service.get_default_location(1)
        assert ref is not None
        assert ref.name == "Home"

    async def test_returns_earliest_created(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        data1 = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        data2 = LocationCreate(
            name="Work", aliases=["praca"], latitude=54.6, longitude=18.5
        )
        loc1 = await service.create_location(1, data1)
        await service.create_location(1, data2)
        ref = await service.get_default_location(1)
        assert ref is not None
        assert ref.id == str(loc1.id)
        assert ref.name == "Home"

    async def test_home_alias_wins_over_earliest_created(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session)
        work = LocationCreate(
            name="Work", aliases=["praca"], latitude=54.6, longitude=18.5
        )
        home = LocationCreate(
            name="Rogalińska 11, Gdańsk",
            aliases=["dom"],
            latitude=54.35,
            longitude=18.65,
        )
        await service.create_location(1, work)
        home_loc = await service.create_location(1, home)
        ref = await service.get_default_location(1)
        assert ref is not None
        assert ref.id == str(home_loc.id)
        assert ref.name == "Rogalińska 11, Gdańsk"

    async def test_does_not_cross_users(
        self, service: LocationService, session: AsyncSession
    ) -> None:
        await _create_user(session, user_id=1)
        await _create_user(session, user_id=2)
        data = LocationCreate(
            name="Home", aliases=["dom"], latitude=52.22, longitude=21.01
        )
        await service.create_location(1, data)
        result = await service.get_default_location(2)
        assert result is None
