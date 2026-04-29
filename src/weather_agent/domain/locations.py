from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.polish_utils import normalize_for_matching
from weather_agent.domain.weather import LocationRef
from weather_agent.infrastructure.db.base import Location as LocationORM


class LocationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    aliases: list[str]
    latitude: float
    longitude: float
    description: str | None = None


class LocationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    aliases: list[str] | None = None
    latitude: float | None = None
    longitude: float | None = None
    description: str | None = None
    enabled: bool | None = None


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    aliases: list[str]
    latitude: float
    longitude: float
    description: str | None
    enabled: bool
    created_at: datetime
    updated_at: datetime


class LocationAliasConflictError(Exception):
    def __init__(self, alias: str, conflicting_location_id: int) -> None:
        self.alias = alias
        self.conflicting_location_id = conflicting_location_id
        super().__init__(
            f"Alias '{alias}' already used by location {conflicting_location_id}"
        )


class LocationNameConflictError(Exception):
    def __init__(self, name: str, conflicting_location_id: int) -> None:
        self.name = name
        self.conflicting_location_id = conflicting_location_id
        super().__init__(
            f"Name '{name}' already used by location {conflicting_location_id}"
        )





def _orm_to_domain(orm: LocationORM) -> Location:
    aliases_raw = orm.aliases
    if isinstance(aliases_raw, dict):
        aliases = list(aliases_raw.keys())
    elif isinstance(aliases_raw, list):
        aliases = aliases_raw
    else:
        aliases = []
    return Location(
        id=orm.id,
        name=orm.name,
        aliases=aliases,
        latitude=orm.latitude,
        longitude=orm.longitude,
        description=orm.description,
        enabled=orm.enabled,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def _is_home_location(orm: LocationORM) -> bool:
    home_markers = {"dom", "home"}
    if normalize_for_matching(orm.name) in home_markers:
        return True
    aliases_raw = orm.aliases
    if isinstance(aliases_raw, dict):
        aliases = list(aliases_raw.keys())
    elif isinstance(aliases_raw, list):
        aliases = aliases_raw
    else:
        aliases = []
    return any(normalize_for_matching(alias) in home_markers for alias in aliases)


class LocationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _check_name_conflict(
        self, user_id: int, name: str, exclude_id: int | None = None
    ) -> None:
        stmt = select(LocationORM).where(
            LocationORM.user_id == user_id, LocationORM.enabled.is_(True)
        )
        result = await self._session.execute(stmt)
        existing = result.scalars().all()
        name_norm = normalize_for_matching(name)
        for loc in existing:
            if exclude_id is not None and loc.id == exclude_id:
                continue
            if normalize_for_matching(loc.name) == name_norm:
                raise LocationNameConflictError(name, loc.id)

    async def _check_alias_conflicts(
        self,
        user_id: int,
        aliases: list[str],
        exclude_id: int | None = None,
    ) -> None:
        stmt = select(LocationORM).where(
            LocationORM.user_id == user_id, LocationORM.enabled.is_(True)
        )
        result = await self._session.execute(stmt)
        existing = result.scalars().all()
        new_norms = {normalize_for_matching(a) for a in aliases}
        for loc in existing:
            if exclude_id is not None and loc.id == exclude_id:
                continue
            loc_name_norm = normalize_for_matching(loc.name)
            if loc_name_norm in new_norms:
                matching_alias = next(
                    a for a in aliases if normalize_for_matching(a) == loc_name_norm
                )
                raise LocationAliasConflictError(matching_alias, loc.id)
            loc_aliases_raw = loc.aliases
            if isinstance(loc_aliases_raw, dict):
                loc_aliases = list(loc_aliases_raw.keys())
            elif isinstance(loc_aliases_raw, list):
                loc_aliases = loc_aliases_raw
            else:
                loc_aliases = []
            for existing_alias in loc_aliases:
                existing_norm = normalize_for_matching(existing_alias)
                if existing_norm in new_norms:
                    matching_alias = next(
                        a
                        for a in aliases
                        if normalize_for_matching(a) == existing_norm
                    )
                    raise LocationAliasConflictError(matching_alias, loc.id)

    async def create_location(self, user_id: int, data: LocationCreate) -> Location:
        await self._check_name_conflict(user_id, data.name)
        await self._check_alias_conflicts(user_id, data.aliases)
        orm = LocationORM(
            user_id=user_id,
            name=data.name,
            aliases=data.aliases,
            latitude=data.latitude,
            longitude=data.longitude,
            description=data.description,
            enabled=True,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def list_locations(
        self, user_id: int, include_disabled: bool = False
    ) -> list[Location]:
        stmt = select(LocationORM).where(LocationORM.user_id == user_id)
        if not include_disabled:
            stmt = stmt.where(LocationORM.enabled.is_(True))
        result = await self._session.execute(stmt)
        return [_orm_to_domain(loc) for loc in result.scalars().all()]

    async def get_location(self, location_id: int) -> Location | None:
        orm = await self._session.get(LocationORM, location_id)
        if orm is None:
            return None
        return _orm_to_domain(orm)

    async def update_location(
        self, location_id: int, data: LocationUpdate
    ) -> Location:
        orm = await self._session.get(LocationORM, location_id)
        if orm is None:
            raise ValueError(f"Location {location_id} not found")
        if data.name is not None:
            await self._check_name_conflict(orm.user_id, data.name, exclude_id=location_id)
            orm.name = data.name
        if data.aliases is not None:
            await self._check_alias_conflicts(
                orm.user_id, data.aliases, exclude_id=location_id
            )
            orm.aliases = data.aliases
        if data.latitude is not None:
            orm.latitude = data.latitude
        if data.longitude is not None:
            orm.longitude = data.longitude
        if data.description is not None:
            orm.description = data.description
        if data.enabled is not None:
            orm.enabled = data.enabled
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def delete_location(self, location_id: int) -> bool:
        orm = await self._session.get(LocationORM, location_id)
        if orm is None:
            return False
        await self._session.execute(
            delete(LocationORM).where(LocationORM.id == location_id)
        )
        await self._session.flush()
        return True

    async def enable_location(self, location_id: int) -> Location:
        orm = await self._session.get(LocationORM, location_id)
        if orm is None:
            raise ValueError(f"Location {location_id} not found")
        orm.enabled = True
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def disable_location(self, location_id: int) -> Location:
        orm = await self._session.get(LocationORM, location_id)
        if orm is None:
            raise ValueError(f"Location {location_id} not found")
        orm.enabled = False
        orm.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(orm)
        return _orm_to_domain(orm)

    async def get_default_location(self, user_id: int) -> LocationRef | None:
        """
        Return the saved default/home location for a user.

        The current schema has no explicit default flag. As an MVP bridge, prefer
        locations marked by name/alias as ``dom``/``home``, then fall back to the
        first enabled saved location in stable creation order.
        """
        stmt = (
            select(LocationORM)
            .where(LocationORM.user_id == user_id, LocationORM.enabled.is_(True))
            .order_by(LocationORM.id)
        )
        result = await self._session.execute(stmt)
        locations = result.scalars().all()
        orm = next((loc for loc in locations if _is_home_location(loc)), None)
        if orm is None and locations:
            orm = locations[0]
        if orm is None:
            return None
        return LocationRef(
            id=str(orm.id),
            name=orm.name,
            latitude=orm.latitude,
            longitude=orm.longitude,
        )

    async def resolve_location(
        self, query: str, user_id: int
    ) -> LocationRef | None:
        stmt = select(LocationORM).where(
            LocationORM.user_id == user_id, LocationORM.enabled.is_(True)
        )
        result = await self._session.execute(stmt)
        locations = result.scalars().all()
        query_norm = normalize_for_matching(query)
        matched = None
        for loc in locations:
            if normalize_for_matching(loc.name) == query_norm:
                matched = loc
                break
        if matched is None:
            for loc in locations:
                loc_aliases_raw = loc.aliases
                if isinstance(loc_aliases_raw, dict):
                    loc_aliases = list(loc_aliases_raw.keys())
                elif isinstance(loc_aliases_raw, list):
                    loc_aliases = loc_aliases_raw
                else:
                    loc_aliases = []
                for alias in loc_aliases:
                    if normalize_for_matching(alias) == query_norm:
                        matched = loc
                        break
                if matched is not None:
                    break
        if matched is None:
            return None
        return LocationRef(
            id=str(matched.id),
            name=matched.name,
            latitude=matched.latitude,
            longitude=matched.longitude,
        )
