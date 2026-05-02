from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from weather_agent.domain.auth import AuthorizedUserRecord
from weather_agent.infrastructure.db.base import AuthorizedUser, GlobalSetting

_INVITE_KEY_PREFIX = "telegram_invite:"


class StoredInviteCode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_by: int
    expires_at: datetime
    used_by: int | None = None
    used_at: datetime | None = None


class InviteRedeemStatus(StrEnum):
    REDEEMED = "redeemed"
    ALREADY_AUTHORIZED = "already_authorized"
    NOT_FOUND = "not_found"
    EXPIRED = "expired"
    USED = "used"
    INVALID = "invalid"


@dataclass(frozen=True)
class InviteRedeemResult:
    status: InviteRedeemStatus


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add_user(self, telegram_user_id: int, role: str = "user") -> None:
        result = await self._session.execute(
            select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        authorized_user = result.scalar_one_or_none()
        if authorized_user is None:
            self._session.add(AuthorizedUser(telegram_user_id=telegram_user_id, role=role))
            await self._session.flush()
            return

        if authorized_user.role != "admin" and role == "admin":
            authorized_user.role = role
            await self._session.flush()

    async def remove_user(self, telegram_user_id: int) -> None:
        await self._session.execute(
            delete(AuthorizedUser).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        await self._session.flush()

    async def list_user_ids(self) -> list[int]:
        result = await self._session.execute(select(AuthorizedUser.telegram_user_id))
        return sorted(int(user_id) for user_id in result.scalars().all())

    async def list_users(self) -> list[AuthorizedUserRecord]:
        result = await self._session.execute(select(AuthorizedUser))
        return [
            AuthorizedUserRecord(
                telegram_user_id=int(user.telegram_user_id),
                role=user.role,
            )
            for user in result.scalars().all()
        ]

    async def get_or_create_authorized_user_id(
        self,
        telegram_user_id: int,
    ) -> int:
        result = await self._session.execute(
            select(AuthorizedUser).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        authorized_user = result.scalar_one_or_none()
        if authorized_user is None:
            authorized_user = AuthorizedUser(telegram_user_id=telegram_user_id)
            self._session.add(authorized_user)
            await self._session.flush()
        return int(authorized_user.id)

    async def create_invite_code(
        self,
        *,
        code: str,
        created_by: int,
        expires_at: datetime,
    ) -> None:
        invite = StoredInviteCode(created_by=created_by, expires_at=expires_at)
        self._session.add(
            GlobalSetting(
                key=_invite_key(code),
                value=invite.model_dump_json(),
            )
        )
        await self._session.flush()

    async def redeem_invite_code(
        self,
        *,
        code: str,
        telegram_user_id: int,
        now: datetime,
    ) -> InviteRedeemResult:
        if await self._is_authorized(telegram_user_id):
            return InviteRedeemResult(status=InviteRedeemStatus.ALREADY_AUTHORIZED)

        setting = await self._get_invite_setting(code)
        if setting is None:
            return InviteRedeemResult(status=InviteRedeemStatus.NOT_FOUND)

        invite = _parse_invite(setting.value)
        if invite is None:
            return InviteRedeemResult(status=InviteRedeemStatus.INVALID)
        if invite.used_by is not None:
            return InviteRedeemResult(status=InviteRedeemStatus.USED)
        if invite.expires_at <= now:
            return InviteRedeemResult(status=InviteRedeemStatus.EXPIRED)

        await self.add_user(telegram_user_id, role="user")
        used_invite = invite.model_copy(update={"used_by": telegram_user_id, "used_at": now})
        setting.value = used_invite.model_dump_json()
        await self._session.flush()
        return InviteRedeemResult(status=InviteRedeemStatus.REDEEMED)

    async def _is_authorized(self, telegram_user_id: int) -> bool:
        result = await self._session.execute(
            select(AuthorizedUser.id).where(AuthorizedUser.telegram_user_id == telegram_user_id)
        )
        return result.scalar_one_or_none() is not None

    async def _get_invite_setting(self, code: str) -> GlobalSetting | None:
        result = await self._session.execute(
            select(GlobalSetting).where(GlobalSetting.key == _invite_key(code)).with_for_update()
        )
        return result.scalar_one_or_none()


def _invite_key(code: str) -> str:
    return f"{_INVITE_KEY_PREFIX}{code}"


def _parse_invite(value: str) -> StoredInviteCode | None:
    try:
        raw: object = json.loads(value)
        invite = StoredInviteCode.model_validate(raw)
    except (json.JSONDecodeError, TypeError, ValidationError):
        return None

    if invite.expires_at.tzinfo is None:
        invite = invite.model_copy(update={"expires_at": invite.expires_at.replace(tzinfo=UTC)})
    if invite.used_at is not None and invite.used_at.tzinfo is None:
        invite = invite.model_copy(update={"used_at": invite.used_at.replace(tzinfo=UTC)})
    return invite
