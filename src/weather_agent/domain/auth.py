from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class UnauthorizedError(Exception):
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        super().__init__(f"Unauthorized access attempt by user_id={user_id}")


class AuthorizedUserRepo(Protocol):
    async def add_user(self, telegram_user_id: int) -> None: ...
    async def remove_user(self, telegram_user_id: int) -> None: ...
    async def list_user_ids(self) -> list[int]: ...


class AuthorizationService:
    def __init__(
        self,
        allowed_user_ids: list[int],
        repo: AuthorizedUserRepo | None = None,
    ) -> None:
        self._allowed: set[int] = set(allowed_user_ids)
        self._repo = repo

    def is_authorized(self, user_id: int) -> bool:
        return user_id in self._allowed

    def check_authorized(self, user_id: int) -> None:
        if not self.is_authorized(user_id):
            logger.info("Unauthorized access attempt by user_id=%s", user_id)
            raise UnauthorizedError(user_id)

    async def add_authorized_user(self, user_id: int) -> None:
        self._allowed.add(user_id)
        if self._repo is not None:
            await self._repo.add_user(user_id)

    async def remove_authorized_user(self, user_id: int) -> None:
        self._allowed.discard(user_id)
        if self._repo is not None:
            await self._repo.remove_user(user_id)

    def list_authorized_users(self) -> list[int]:
        return sorted(self._allowed)